# -*- coding: utf-8 -*-
# Adaptation of LoRAGen 911a738e: omit the task-zoo trainer and its imports.

import math
from random import random
from functools import partial
from collections import namedtuple

from tqdm.auto import tqdm

import torch
from torch import nn
import torch.nn.functional as F

from einops import reduce


# -----------------------------------------------------------------------------
# Constants & small helpers
# -----------------------------------------------------------------------------

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


def exists(x):
    return x is not None


def default(val, d):
    return val if exists(val) else (d() if callable(d) else d)


def identity(t, *args, **kwargs):
    return t


def cycle(dl):
    """Infinite dataloader."""
    while True:
        for data in dl:
            yield data


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


# -----------------------------------------------------------------------------
# Normalization helpers
# -----------------------------------------------------------------------------

def normalize_to_neg_one_to_one(x):
    return x * 2 - 1


def unnormalize_to_zero_to_one(x):
    return (x + 1) * 0.5


# -----------------------------------------------------------------------------
# Diffusion scheduling
# -----------------------------------------------------------------------------

def extract(a, t, x_shape):
    """Helper to gather a[t] and reshape to broadcast over x."""
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
    # A simple linear schedule, lightly scaled.
    scale = 1.0
    beta_start = scale * 1e-5
    beta_end = scale * 2e-2
    betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)
    return betas


def cosine_beta_schedule(timesteps, s=0.008):
    # Cosine schedule commonly used for improved stability.
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


# -----------------------------------------------------------------------------
# 1D Gaussian Diffusion (LoRA parameter generator)
# -----------------------------------------------------------------------------

class GaussianDiffusion1D(nn.Module):
    """
    A 1D diffusion wrapper that calls a denoising network with optional conditioning.
    Inputs are shaped as (B, C, N), where N is sequence length.
    """

    def __init__(
        self,
        denoising_model,
        *,
        seq_length,
        timesteps=1000,
        sampling_timesteps=None,
        loss_type='l1',
        objective='pred_noise',
        beta_schedule='cosine',
        ddim_sampling_eta=0.0,
        auto_normalize=False
    ):
        super().__init__()
        self.model = denoising_model
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition

        self.seq_length = seq_length
        self.objective = objective
        assert objective in {
            'pred_noise', 'pred_x0', 'pred_v'
        }, "objective must be one of {'pred_noise','pred_x0','pred_v'}"

        # betas
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'Unknown beta schedule: {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling config
        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # register diffusion buffers as float32
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))
        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # posterior terms q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # loss re-weighting by SNR
        snr = alphas_cumprod / (1 - alphas_cumprod)
        if objective == 'pred_noise':
            loss_weight = torch.ones_like(snr)
        elif objective == 'pred_x0':
            loss_weight = snr
        elif objective == 'pred_v':
            loss_weight = snr / (snr + 1)
        register_buffer('loss_weight', loss_weight)

        # optional auto (un)normalize
        self.normalize = normalize_to_neg_one_to_one if auto_normalize else identity
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else identity

    # --- prediction conversions ---

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    # --- core model calls ---

    def model_predictions(self, x, t, cond_emb, aux_emb, x_self_cond=None,
                          clip_x_start=False, rederive_pred_noise=False):
        """
        Call underlying denoising model with both conditioning embeddings.
        """
        model_output = self.model(x, t, cond_emb, aux_emb, x_self_cond)

        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)
            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = maybe_clip(model_output)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = maybe_clip(self.predict_start_from_v(x, t, v))
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, cond_emb, aux_emb, x_self_cond=None, clip_denoised=True):
        preds = self.model_predictions(x, t, cond_emb, aux_emb, x_self_cond)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, cond_emb, aux_emb, x_self_cond=None, clip_denoised=True):
        """
        One reverse diffusion step at (integer) time t.
        """
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device=device, dtype=torch.long)

        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x, t=batched_times, cond_emb=cond_emb, aux_emb=aux_emb,
            x_self_cond=x_self_cond, clip_denoised=clip_denoised
        )
        noise = torch.randn_like(x) if t > 0 else 0.
        pred = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred, x_start

    @torch.no_grad()
    def p_sample_loop(self, shape, cond_emb, aux_emb):
        """
        Full ancestral sampling loop.
        """
        _, device = shape[0], self.betas.device
        img = torch.randn(shape, device=device)
        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)),
                      desc='sampling loop time step', total=self.num_timesteps):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, t, cond_emb, aux_emb, self_cond, clip_denoised=True)

        img = self.unnormalize(img)
        return img

    @torch.no_grad()
    def ddim_sample(self, shape, cond_emb, aux_emb, clip_denoised=True):
        """
        DDIM sampling (when sampling_timesteps < train timesteps).
        """
        batch, device = shape[0], self.betas.device
        total_timesteps = self.num_timesteps
        sampling_timesteps = self.sampling_timesteps
        eta = self.ddim_sampling_eta

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        img = torch.randn(shape, device=device)
        x_start = None

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            self_cond = x_start if self.self_condition else None
            pred_noise, x_start, *_ = self.model_predictions(
                img, time_cond, cond_emb, aux_emb, self_cond, clip_x_start=clip_denoised
            )

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise

        img = self.unnormalize(img)
        return img

    @torch.no_grad()
    def sample(self, cond_emb, aux_emb, batch_size=16):
        """
        Public sampling API. Chooses DDPM/DDIM based on configuration.
        """
        seq_length, channels = self.seq_length, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn((batch_size, channels, seq_length), cond_emb, aux_emb)

    @torch.no_grad()
    def interpolate(self, x1, x2, cond_emb, aux_emb, t=None, lam=0.5):
        """
        Interpolate between two inputs in noisy space and denoise back.
        """
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)
        assert x1.shape == x2.shape

        t_batched = torch.full((b,), t, device=device)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))
        img = (1 - lam) * xt1 + lam * xt2

        x_start = None
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            self_cond = x_start if self.self_condition else None
            img, x_start = self.p_sample(img, i, cond_emb, aux_emb, self_cond)

        return img

    # --- training loss ---

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'Invalid loss type: {self.loss_type}')

    def p_losses(self, x_start, t, cond_emb, aux_emb, noise=None):
        """
        Standard diffusion training objective with optional self-conditioning.
        """
        noise = default(noise, lambda: torch.randn_like(x_start))
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t, cond_emb, aux_emb).pred_x_start
                x_self_cond.detach_()

        model_out = self.model(x, t, cond_emb, aux_emb, x_self_cond)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            target = self.predict_v(x_start, t, noise)
        else:
            raise ValueError(f'Unknown objective: {self.objective}')

        loss = self.loss_fn(model_out, target, reduction='none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, input_signal, cond_emb, aux_emb, *args, **kwargs):
        """
        Training forward: sample a diffusion step t and compute loss.
        """
        b, c, n = input_signal.shape
        device = input_signal.device
        assert n == self.seq_length, f'Seq length {n} must equal {self.seq_length}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        input_signal = self.normalize(input_signal)
        return self.p_losses(input_signal, t, cond_emb, aux_emb, *args, **kwargs)


