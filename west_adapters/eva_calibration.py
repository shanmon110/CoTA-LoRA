"""Explained Variance Adaptation for Qwen decoder text-attention o_proj inputs.

Uses centered incremental PCA and global explained-variance-ratio ranking
(Paischer et al., NeurIPS 2025, §3.2/3.5, Appendix F; ml-jku/EVA and PEFT).
Unlike their convergence-controlled loop, we consume a fixed, finite training
subset once. This makes calibration cost explicit; it does not certify that
the PCs have converged. No whitening or singular-value rescaling is applied.
"""

import math
from collections import Counter

import torch

from .base import move_inputs, targets


class _IncrementalPCA:
    """Bounded-memory centered SVD; track the *full* scatter trace separately."""

    def __init__(self, cap):
        self.cap = cap
        self.count = 0
        self.mean = self.trace = self.s = self.vh = None

    def update(self, rows):
        if not rows.isfinite().all().item():
            raise ValueError("EVA encountered nonfinite activations")
        # FP64 moments also avoid catastrophic cancellation at large offsets.
        x = rows.double()
        n = x.size(0)
        batch_mean = x.mean(0)
        x = x - batch_mean
        scatter = x.square().sum()
        if self.count:
            total = self.count + n
            shift = batch_mean - self.mean
            correction = shift * math.sqrt(n * self.count / total)
            # Merged centered scatter = old scatter + local scatter +
            # correction.T @ correction. Only the old PCA basis is truncated;
            # its discarded tail still counts in the exact ratio denominator.
            self.trace += scatter + correction.square().sum()
            self.mean += shift * (n / total)
            block = torch.cat([self.s[:, None] * self.vh, x.float(), correction[None].float()])
        else:
            self.trace, self.mean = scatter, batch_mean
            block = x.float()
        self.count += n
        _, s, vh = torch.linalg.svd(block, full_matrices=False)
        self.s = s[:self.cap].clone()
        self.vh = vh[:self.cap].clone(memory_format=torch.contiguous_format)

    def ratios(self):
        if self.count <= self.cap:
            raise ValueError(f"EVA needs at least {self.cap + 1} selected training tokens "
                             f"for centered rank {self.cap}; observed {self.count}. "
                             "Increase calibration batches or change the EVA token mask.")
        energy = self.trace.item()
        if energy == 0:
            return [0.] * self.cap
        # A missing eigen-direction cannot be identified from these data.
        tol = self.s[0] * max(self.vh.shape) * torch.finfo(self.s.dtype).eps
        return (self.s.double().square().masked_fill(self.s <= tol, 0) / energy).tolist()


@torch.no_grad()
def calibrate_eva(thinker, loader, *, last_n, rank, rho, device, use_label_mask=True,
                  forward_model=None):
    """Return {full o_proj name: FP32 CPU [allocated rank, input width]}.

    Call on the unadapted base, passing a finite iterable of TRAIN batches.
    Both policies exclude padding. By default, like PEFT's SFT preparation, use
    input positions whose *same-position* labels are not -100 (no causal loss
    shift). This includes assistant answer/EOS but not prompt/audio positions.
    Set use_label_mask=False to use all non-padding positions, as in the paper's
    short-answer experiments (Appendix F.5).
    """
    layers = thinker.model.layers
    if type(last_n) is not int or not 1 <= last_n <= len(layers):
        raise ValueError("EVA last_n must be in [1, number of text layers]")
    if type(rank) is not int or rank < 1 or not math.isfinite(rho) or rho < 1:
        raise ValueError("EVA requires positive integer rank and finite rho >= 1")
    if any(name.endswith((".ta_adapter", ".baseline_adapter")) for name, _ in thinker.named_modules()):
        raise ValueError("EVA must calibrate the unadapted Thinker before adapter installation")

    projections, states = {}, {}
    for name, proj in targets(thinker, last_n, "o").items():
        limit = min(proj.in_features, proj.out_features)
        if rank > limit:
            raise ValueError(f"EVA rank {rank} exceeds the projection dimensions at {name}")
        cap = limit if rho >= limit / rank else round(rank * rho)
        projections[name], states[name] = proj, _IncrementalPCA(cap)

    indices = shape = None

    def pre_hook(state):
        def collect(module, args):
            x = args[0]
            if x.ndim != 3 or x.shape[:2] != shape:
                raise ValueError("EVA expected o_proj inputs aligned with unpacked [batch, sequence] tokens")
            # Bound the PCA workspace even for a long transcript. The hook does
            # not retain the forward activation, nor form a width-by-width Gram.
            for ix in indices.split(128):
                state.update(x[ix[:, 0], ix[:, 1]])
        return collect

    dtype = next(thinker.parameters()).dtype
    # TouchASU inserts speech embeddings outside its decoder. Collect from
    # the decoder projections while running its enclosing multimodal model.
    runner = thinker if forward_model is None else forward_model
    flags = [(m, m.training) for m in runner.modules()]
    # Native 4.52.3 updates this cache even when use_cache=False.
    has_rope = hasattr(thinker, "rope_deltas")
    old_rope = getattr(thinker, "rope_deltas", None)
    handles = []
    try:
        runner.eval()
        for name, proj in projections.items():
            handles.append(proj.register_forward_pre_hook(pre_hook(states[name])))
        for batch in loader:
            inputs = move_inputs(batch, device, dtype)
            labels = inputs.pop("labels", None)
            mask = inputs["attention_mask"].bool()
            shape = inputs["input_ids"].shape
            if mask.ndim != 2 or mask.shape != shape:
                raise ValueError("EVA requires a 2D attention_mask matching input_ids")
            if use_label_mask:
                if labels is None or labels.shape != shape:
                    raise ValueError("EVA label mask requires labels matching input_ids")
                mask = mask & labels.ne(-100)
            indices = mask.nonzero()
            if indices.size(0):
                runner(**inputs, use_cache=False)
    finally:
        for h in handles:
            h.remove()
        if has_rope:
            thinker.rope_deltas = old_rope
        for m, was_training in flags:
            m.training = was_training

    candidates = [(ratio, name) for name, state in states.items() for ratio in state.ratios()
                  if ratio > 0]
    budget = rank * last_n
    if len(candidates) < budget:
        raise ValueError(f"EVA found {len(candidates)} nonzero-variance directions for rank budget "
                         f"{budget}; increase calibration diversity or reduce rank.")
    # Stable order for ties: earlier layer, then earlier component.
    candidates.sort(key=lambda item: item[0], reverse=True)
    counts = Counter(name for _, name in candidates[:budget])
    return {name: state.vh[:counts[name]].to(device="cpu", copy=True) for name, state in states.items()}
