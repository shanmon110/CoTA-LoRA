"""Compatibility API for standalone projection branches."""
import math

from torch import nn
from west_adapters.base import FP32_ADD_THEN_CAST, add_fp32_residual
from west_adapters.registry import checkpoint_method


def build_branch(proj: nn.Linear, mode: str, rank: int, alpha: float, blocks: int = 2) -> nn.Module:
    if not isinstance(proj, nn.Linear):
        raise TypeError("projection must be an nn.Linear")
    if type(rank) is not int or rank <= 0 or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("expected positive rank and finite, positive alpha")
    config = dict(mode=mode, target_set="qv" if mode == "lora" else "o",
                  rank=rank, alpha=alpha, bora_blocks=blocks)
    method = checkpoint_method(config)
    if method.attention:
        raise ValueError(f"unknown projection mode: {mode!r}")
    return method.build(proj, rank, config)
