"""Blockwise LoRA with learned per-block gains."""
import math

import torch
from torch import nn

from .base import AdapterMethod, ProjectionBranch


class BoRABranch(ProjectionBranch):
    def __init__(self, proj, rank, alpha, config):
        b = config["bora_blocks"]
        if type(b) is not int or b <= 0 or proj.in_features % b or proj.out_features % b:
            raise ValueError("BoRA blocks must be a positive integer dividing both projection widths")
        super().__init__(proj, rank, alpha, config)
        self.blocks = b
        self.sigma = nn.Parameter(self.down.new_empty(b, b, rank))
        bound = 1 / math.sqrt(proj.in_features)
        nn.init.uniform_(self.sigma, -bound, bound)

    def delta(self, proj, x, base_output):
        b = self.blocks
        x = x.unflatten(-1, (b, -1))
        a = self.down.float().unflatten(-1, (b, -1))
        up = self.up.float().unflatten(0, (b, -1))
        # Contract A_j x_j once per input block, then mix it for every B_i.
        # No dense [out,in] block matrix or flattened [tokens,b,b,r] is built.
        coeff = torch.einsum("...jn,rjn->...jr", x, a)
        sigma = self.sigma.float()
        mav = sigma.abs().mean()
        # Eq. 5 is undefined only for all-zero σ; use identity gains then.
        gain = (sigma / torch.where(mav == 0, 1.0, mav)).exp()
        # For finite sigma the exact normalized magnitude is bounded by numel.
        # Allow a factor of two for rounding a subnormal mean. The default
        # 2x2x8 grid therefore needs no per-token device synchronization.
        if (2 * self.sigma.numel() >= math.log(torch.finfo(gain.dtype).max)
                and not torch.isfinite(gain).all()):
            raise FloatingPointError(
                f"Nonfinite BoRA gain (blocks={b}, rank={self.sigma.shape[-1]}).")
        mixed = torch.einsum("...jr,ijr->...ir", coeff, gain)
        return torch.einsum("...ir,imr->...im", mixed, up).flatten(-2) * self.scale

class BoRA(AdapterMethod):
    name = mode = "bora"
    options = {"bora_blocks": 2}
    branch = BoRABranch

    def validate_options(self, config):
        if type(config.get("bora_blocks")) is not int or config["bora_blocks"] < 1:
            raise ValueError("bora_blocks must be a positive integer")

    def shapes(self, proj, rank, config):
        b = config["bora_blocks"]
        if proj.baseline_adapter.blocks != b:
            raise ValueError("bora_blocks mismatch")
        return {**super().shapes(proj, rank, config), "sigma": (b, b, rank)}
