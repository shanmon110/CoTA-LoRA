"""Omni linear projections for papers 12, 14–16, 18 and 19.

The upstream revisions and integration decisions are recorded in paper_middle_sources.md.
"""
import math
import weakref
from contextvars import ContextVar
from functools import wraps

import torch
from torch import nn
from torch.nn import functional as F

from west_adapters.toplora import TopLoRABranch


class PaperProjection(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        if rank < 1 or rank > min(base.weight.shape):
            raise ValueError(f'Invalid rank {rank} for {tuple(base.weight.shape)}')
        self.base_layer = base.requires_grad_(False)
        self.rank = rank
        self.scale = alpha / rank
        self.tensor_args = {'dtype': torch.float32, 'device': base.weight.device}

    def forward(self, x):
        value = self.base_layer(x)
        with torch.autocast(x.device.type, enabled=False):
            return (value.float() + self.delta(x.float()) * self.scale).to(value.dtype)


class TopLoRA(PaperProjection):
    def __init__(self, base, rank, alpha, settings):
        super().__init__(base, rank, alpha)
        if settings.get('variant', 'token_diagonal_rms_exp') != 'token_diagonal_rms_exp':
            raise ValueError('Expected author token-diagonal RMSNorm/exp TopLoRA')
        self.branch = TopLoRABranch(base, rank, alpha, {'mode': 'toplora'})

    def forward(self, x):
        return self.branch(self.base_layer, x, self.base_layer(x))


class GraLoRA(PaperProjection):
    def __init__(self, base, rank, alpha, settings):
        super().__init__(base, rank, alpha)
        self.blocks = settings.get('blocks', 2)
        if self.blocks < 1 or any(n % self.blocks for n in (base.in_features, base.out_features, rank)):
            raise ValueError('GraLoRA input/output dimensions and rank must be divisible by blocks')
        ni, no = base.in_features // self.blocks, base.out_features // self.blocks
        matrices = [nn.init.kaiming_uniform_(torch.empty(rank, ni, **self.tensor_args), a=math.sqrt(5))
                    for _ in range(self.blocks)]
        self.down = nn.Parameter(torch.stack(matrices).transpose(1, 2).contiguous())
        self.up = nn.Parameter(torch.zeros(self.blocks, rank, no, **self.tensor_args))

    def delta(self, x):
        k = self.blocks
        projected = torch.einsum('...ni,nir->...nr', x.unflatten(-1, (k, -1)), self.down)
        # The rank is split by destination block; each destination consumes one slice from every input.
        routed = projected.unflatten(-1, (k, self.rank // k)).transpose(-3, -2).flatten(-2)
        return torch.einsum('...jr,jro->...jo', routed, self.up).flatten(-2)


class MagnitudeLoRAM(PaperProjection):
    def __init__(self, base, rank, alpha, settings):
        super().__init__(base, rank, alpha)
        dim = min(base.weight.shape)
        if dim < 2 or rank < 2:
            raise ValueError('Magnitude-LoRAM log gain requires rank and dimensions > 1')

        def dst(size):
            i = torch.arange(1, size + 1, **self.tensor_args)[:, None]
            j = torch.arange(1, rank + 1, **self.tensor_args)
            return (i * j * torch.pi / (size + 1)).sin() * math.sqrt(2 / (size + 1))

        left, right = dst(base.out_features), dst(base.in_features).T
        with torch.no_grad(), torch.autocast(base.weight.device.type, enabled=False):
            gain = (math.log(rank, dim) * base.weight.float().var() / (left @ right).var()).pow(.25)
            self.down = nn.Parameter((right * gain).contiguous())
            self.up = nn.Parameter((left * gain).contiguous())
            # A clean reload calls this constructor on exactly the same pristine base.
            base.weight.copy_(base.weight.float() - self.scale * (self.up @ self.down))

    def delta(self, x):
        return F.linear(F.linear(x, self.down), self.up)


BUILDERS = {'toplora': TopLoRA, 'gralora': GraLoRA, 'magnitude-loram': MagnitudeLoRAM}


class AuroRA(PaperProjection):
    """Author commonsense-reasoning ANL: fixed tanh branch plus learnable cubic splines."""
    def __init__(self, base, rank, alpha, settings):
        super().__init__(base, rank, alpha)
        self.order = settings.get('spline_order', 3)
        grids = settings.get('grid_size', 5)
        if self.order < 1 or grids < 1:
            raise ValueError('AuroRA requires positive spline_order and grid_size')
        self.down = nn.Parameter(torch.empty(rank, base.in_features, **self.tensor_args))
        self.up = nn.Parameter(torch.zeros(base.out_features, rank, **self.tensor_args))
        self.fixed_weight = nn.Parameter(torch.empty(rank, rank, **self.tensor_args))
        self.spline_weight = nn.Parameter(torch.empty(rank, grids+self.order, **self.tensor_args))
        nn.init.xavier_normal_(self.down)
        nn.init.xavier_normal_(self.fixed_weight)
        nn.init.uniform_(self.spline_weight, -.1, .1)
        grid = torch.arange(-self.order, grids+self.order+1, **self.tensor_args) * (2/grids) - 1
        self.register_buffer('grid', grid)

    def delta(self, x):
        z = F.linear(x, self.down)
        t = z.unsqueeze(-1)
        g = self.grid
        basis = ((t >= g[:-1]) & (t < g[1:])).float()
        for k in range(1, self.order + 1):
            left = (t - g[:-k-1]) / (g[k:-1] - g[:-k-1])
            right = (g[k+1:] - t) / (g[k+1:] - g[1:-k])
            basis = left * basis[..., :-1] + right * basis[..., 1:]
        nonlinear = F.linear(z.tanh(), self.fixed_weight).tanh() + F.linear(basis.sum(-2), self.spline_weight)
        return F.linear(nonlinear, self.up)


class FlyLoRA(PaperProjection):
    def __init__(self, base, rank, alpha, settings):
        super().__init__(base, rank, alpha)
        self.top_k = settings.get('top_k', max(rank // 4, 1))
        self.bias_step = settings.get('bias_step', .001)
        if not 0 < self.top_k < rank or rank % self.top_k or self.bias_step < 0:
            raise ValueError('FlyLoRA needs a proper top-k divisor of rank and a nonnegative bias step')
        if settings.get('routing', 'signed_author_peft') != 'signed_author_peft':
            raise ValueError('This FlyLoRA variant follows Eq. 10 and the authors’ PEFT training implementation')
        p = base.in_features // (rank // self.top_k)
        if p < 1:
            raise ValueError('Sparse FlyLoRA projection has no selected input')
        down = torch.zeros(rank, base.in_features, **self.tensor_args)
        for row in down:
            indices = torch.randperm(base.in_features, device=down.device)[:p]
            row[indices] = torch.randn(p, **self.tensor_args) / rank
        self.register_buffer('down', down)
        self.register_buffer('route_bias', torch.zeros(rank, **self.tensor_args))
        self.up = nn.Parameter(torch.zeros(base.out_features, rank, **self.tensor_args))
        self.layout = None

    def delta(self, x):
        projected = F.linear(x, self.down)
        selected = (projected.detach() + self.route_bias).topk(self.top_k, dim=-1).indices
        mask = torch.zeros_like(projected).scatter_(-1, selected, 1)
        if self.training:
            valid = self.layout.get()[-1] if self.layout is not None else None
            if valid is not None:
                if valid.shape != x.shape[:-1]:
                    raise ValueError('FlyLoRA load-balance mask and projected tokens differ')
                counts = (mask * valid.unsqueeze(-1)).reshape(-1, self.rank).sum(0)
            else:
                counts = mask.reshape(-1, self.rank).sum(0)
            self.route_bias.add_((counts.mean() - counts).sign(), alpha=self.bias_step)
        return F.linear(projected * mask, self.up)


class UniLoRA(PaperProjection):
    def __init__(self, base, rank, alpha, settings):
        super().__init__(base, rank, alpha)
        if alpha != rank:
            raise ValueError('The author Uni-LoRA branch has unit scale')
        self.register_buffer('index_a', torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.long))
        self.register_buffer('index_b', torch.empty(base.out_features, rank, device=base.weight.device, dtype=torch.long))

    def delta(self, x):
        anchor = self.anchor()
        bank = anchor.vector * anchor.inv_sqrt_count
        return F.linear(F.linear(x, bank[self.index_a]), bank[self.index_b])


def bind_unilora(modules, settings):
    owner = modules[0]
    total = sum(m.index_a.numel() + m.index_b.numel() for m in modules)
    dim = settings.get('vector_length', 23040)
    bound = settings.get('init_bound', .02)
    if not 1 <= dim <= total or not bound > 0:
        raise ValueError('Invalid Uni-LoRA vector length or initialization bound')
    if any(m.base_layer.weight.device != owner.base_layer.weight.device for m in modules):
        raise ValueError('Shared Uni-LoRA bank expects a single device')
    owner.vector = nn.Parameter(torch.empty(dim, **owner.tensor_args).uniform_(-bound, bound))
    repeats, remainder = divmod(total, dim)
    device = owner.vector.device
    indexes = torch.cat((torch.arange(dim, device=device).repeat_interleave(repeats),
                         torch.randperm(dim, device=device)[:remainder]))
    indexes = indexes[torch.randperm(total, device=device)]
    owner.register_buffer('inv_sqrt_count', indexes.bincount(minlength=dim).float().rsqrt())
    i = 0
    for m in modules:
        m.anchor = weakref.ref(owner)
        for output in (m.index_a, m.index_b):
            output.copy_(indexes[i:i+output.numel()].view_as(output))
            i += output.numel()


def after_install(model, spec):
    method, settings = spec['method'], spec['settings']
    if method == 'lowra':
        from .paper_middle_lowra_runtime import bind_lowra
        bind_lowra(model, spec)
        return
    if method not in ('unilora', 'flylora', 'alora'):
        return
    modules = [model.get_submodule(p) for p in spec['targets']]
    if method == 'unilora':
        bind_unilora(modules, settings)
        return
    if method == 'alora':
        install_alora_layout(model, modules, settings)
        return
    frames = ContextVar('flylora_live_tokens', default=())
    for m in modules:
        m.layout = frames

    def start(_, args, kwargs):
        frames.set((*frames.get(), kwargs.get('attention_mask')))

    def finish(_, args, output):
        frames.set(frames.get()[:-1])

    model.register_forward_pre_hook(start, prepend=True, with_kwargs=True)
    model.register_forward_hook(finish, always_call=True)


def prepare_trainer(trainer, method, resume=None):
    if method == 'lowra' and trainer.args.device.type == 'cuda':
        from .paper_middle_lowra_runtime import require_cuda_parity
        device = trainer.args.device.index
        marker = require_cuda_parity(torch.cuda.current_device() if device is None else device)
        trainer.model._paper_adapter_spec['lowra_cuda_parity'] = marker
    if method in ('flylora', 'alora') and trainer.args.gradient_checkpointing:
        raise ValueError(f'{method} requires checkpointing off for consistent routing context during backward')


BUILDERS.update(aurora=AuroRA, flylora=FlyLoRA, unilora=UniLoRA)


def activation_mask(input_ids, attention_mask, invocation):
    """The final complete invocation activates its first token and the remainder (PEFT v0.18)."""
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
        raise ValueError('aLoRA needs original unpacked token IDs before Swift embeds audio')
    valid = torch.ones_like(input_ids, dtype=torch.bool) if attention_mask is None else attention_mask.bool()
    if valid.shape != input_ids.shape:
        raise ValueError('aLoRA needs a 2D padding mask matching the input IDs')
    if input_ids.shape[1] < len(invocation):
        return torch.zeros_like(valid)
    windows = input_ids.unfold(1, len(invocation), 1)
    matches = windows.eq(input_ids.new_tensor(invocation)).all(-1) & valid.unfold(1, len(invocation), 1).all(-1)
    positions = torch.arange(input_ids.shape[1], device=input_ids.device)
    last = torch.where(matches, positions[:windows.shape[1]], -1).amax(-1, keepdim=True)
    return valid & (last >= 0) & (positions >= last)


class ActivatedLoRA(PaperProjection):
    def __init__(self, base, rank, alpha, settings):
        super().__init__(base, rank, alpha)
        self.down = nn.Parameter(torch.empty(rank, base.in_features, **self.tensor_args))
        self.up = nn.Parameter(torch.zeros(base.out_features, rank, **self.tensor_args))
        nn.init.kaiming_uniform_(self.down, a=math.sqrt(5))
        self.layout = None

    def delta(self, x):
        if self.layout is None or not self.layout.get():
            raise ValueError('aLoRA requires an active prompt context')
        mask = self.layout.get()[-1]
        if mask.shape != x.shape[:-1]:
            if x.shape[:-1] != (mask.shape[0], 1):
                raise ValueError('aLoRA requires batched prefill or one cached decode token')
            mask = mask.any(-1, keepdim=True)
        # Computing the branch only on the suffix is equivalent and saves activations for long audio prompts.
        out = x.new_zeros(*x.shape[:-1], self.up.shape[0])
        out[mask] = F.linear(F.linear(x[mask], self.down), self.up)
        return out


def install_alora_layout(model, modules, settings):
    invocation = settings.get('invocation_tokens')
    if not isinstance(invocation, list) or not invocation or any(type(i) is not int or i < 0 for i in invocation):
        raise ValueError('aLoRA requires nonempty integer invocation_tokens')
    frames = ContextVar('alora_active_positions', default=())
    for m in modules:
        m.layout = frames

    def layout(args, kwargs):
        ids = kwargs.get('input_ids', kwargs.get('inputs', args[0] if args else None))
        mask = activation_mask(ids, kwargs.get('attention_mask'), invocation)
        if settings.get('require_marker', True) and not mask.any(-1).all():
            raise ValueError('aLoRA invocation missing from a formatted training/evaluation prompt')
        return mask

    def start(_, args, kwargs):
        old = frames.get()
        frames.set((*old, None))
        frames.set((*old, layout(args, kwargs)))

    def finish(_, args, output):
        frames.set(frames.get()[:-1])

    model.register_forward_pre_hook(start, prepend=True, with_kwargs=True)
    model.register_forward_hook(finish, always_call=True)
    original = model.generate

    @wraps(original)
    def generate(*args, **kwargs):
        if kwargs.get('num_beams', 1) != 1 or kwargs.get('num_return_sequences', 1) != 1:
            raise ValueError('aLoRA comparison supports greedy generation without beam expansion')
        if kwargs.get('past_key_values') is not None:
            raise ValueError('This Omni scoring entry expects the full prompt at generation start')
        token = frames.set((*frames.get(), layout(args, kwargs)))
        try:
            return original(*args, **kwargs)
        finally:
            frames.reset(token)

    model.generate = generate


BUILDERS['alora'] = ActivatedLoRA


class LowRA(PaperProjection):
    def __init__(self, base, rank, alpha, settings):
        super().__init__(base, rank, alpha)

    def delta(self, x):
        return F.linear(F.linear(x, self.down), self.up)


BUILDERS['lowra'] = LowRA
