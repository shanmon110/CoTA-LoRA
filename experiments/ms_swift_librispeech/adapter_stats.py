"""Per-factor gradient and weight statistics for LoRA / TA adapters during training.

Enabled with ``"diagnostics": {"adapter_stats_every": N}`` in the experiment config. One JSON line
is appended every N optimizer updates, just before the update is applied. Trainer clips gradients
before this event, so norms are post-clipping; ratios between factors and the cosines are unaffected
because clipping multiplies every gradient by the same scalar.

``grad_cosine_to_previous`` is the cosine between a factor's gradient at this record and at the
previous record. A weight that receives a consistent learning signal shows clearly positive cosines;
values scattered around zero mean Adam is integrating noise for that factor.
"""
import json
from pathlib import Path

import torch
from transformers import TrainerCallback

FACTORS = ('ta_adapter.down', 'ta_adapter.up', 'ta_adapter.gate', 'ta_adapter.transport_up',
           'ta_adapter.gate_weight', 'lora_A', 'lora_B', 'lora_magnitude_vector')


def factor_of(name):
    for factor in FACTORS:
        if name.endswith(factor) or f'.{factor}.' in name:
            return factor.removeprefix('ta_adapter.')
    return 'other'


@torch.no_grad()
def adapter_statistics(model, previous=None):
    """Return ({factor: stats}, {factor: flat gradient}) over the trainable parameters."""
    squares, grads = {}, {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        factor = factor_of(name)
        entry = squares.setdefault(factor, dict(tensors=0, elements=0, weight=0., grad=0., missing_grad=0,
                                                max_abs_weight=0.))
        entry['tensors'] += 1
        entry['elements'] += param.numel()
        entry['weight'] += param.detach().float().square().sum().item()
        entry['max_abs_weight'] = max(entry['max_abs_weight'], param.detach().abs().max().item())
        if param.grad is None:
            entry['missing_grad'] += 1
            continue
        grad = param.grad.detach().float()
        entry['grad'] += grad.square().sum().item()
        grads.setdefault(factor, []).append(grad.flatten().cpu())
    flat = {factor: torch.cat(parts) for factor, parts in grads.items()}
    stats = {}
    for factor, entry in squares.items():
        row = dict(tensors=entry['tensors'], elements=entry['elements'], weight_norm=entry['weight'] ** .5,
                   grad_norm=entry['grad'] ** .5, max_abs_weight=entry['max_abs_weight'],
                   tensors_without_grad=entry['missing_grad'])
        before = (previous or {}).get(factor)
        if before is not None and factor in flat and before.shape == flat[factor].shape:
            denom = before.norm() * flat[factor].norm()
            row['grad_cosine_to_previous'] = (before @ flat[factor] / denom).item() if denom > 0 else None
        stats[factor] = row
    return stats, flat


class AdapterStats(TrainerCallback):
    def __init__(self, path, every):
        if type(every) is not int or every < 1:
            raise ValueError('adapter_stats_every must be a positive integer')
        self.path = Path(path)
        self.every = every
        self.previous = None

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        update = state.global_step + 1  # the update about to be applied
        if model is None or update % self.every:
            return
        stats, self.previous = adapter_statistics(model, self.previous)
        with self.path.open('a') as stream:
            print(json.dumps(dict(update=update, factors=stats), allow_nan=False), file=stream)
