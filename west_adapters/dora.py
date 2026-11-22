"""Magnitude and direction decomposition with detached, row-blocked norms."""
import torch
from torch import nn

from .base import AdapterMethod, ProjectionBranch


class DoRABranch(ProjectionBranch):
    def __init__(self, proj, rank, alpha, config):
        super().__init__(proj, rank, alpha, config)
        self.magnitude = nn.Parameter(proj.weight.detach().float().norm(dim=1))

    @torch.no_grad()
    def _direction_norm(self, weight):
        # Eq. 11, §4.3: recompute the direction norm but detach it from
        # backprop. Block the output rows to bound peak temporary memory.
        down = self.down.float()
        return torch.cat([
            w.float().addmm(b.float(), down, alpha=self.scale).norm(dim=1)
            for w, b in zip(weight.split(256), self.up.split(256))
        ])

    def delta(self, proj, x, base_output):
        update = super().delta(proj, x, base_output)
        norm = self._direction_norm(proj.weight)
        ratio = self.magnitude.float() / torch.where(norm == 0, 1.0, norm)
        base = base_output.float()
        unbiased = base if proj.bias is None else base - proj.bias.float()
        return (ratio - 1) * unbiased + ratio * update


class DoRA(AdapterMethod):
    name = mode = "dora"
    branch = DoRABranch

    def shapes(self, proj, rank, config):
        return {**super().shapes(proj, rank, config), "magnitude": (proj.out_features,)}
