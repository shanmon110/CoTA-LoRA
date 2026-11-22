"""Token-dependent singular values for TopLoRA."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .base import AdapterMethod, ProjectionBranch


class TopLoRABranch(ProjectionBranch):
    def __init__(self, proj, rank, alpha, config):
        super().__init__(proj, rank, alpha, config)
        self.projector = nn.Parameter(torch.empty_like(self.down))
        nn.init.kaiming_uniform_(self.projector, a=math.sqrt(5))
        self.rms_weight = nn.Parameter(self.down.new_ones(rank))

    def coefficients(self, x):
        z = F.linear(x, self.projector.float())
        return super().coefficients(x) * F.rms_norm(z, (z.shape[-1],), self.rms_weight.float()).exp()


class TopLoRA(AdapterMethod):
    name = mode = "toplora"
    branch = TopLoRABranch

    def shapes(self, proj, rank, config):
        return {**super().shapes(proj, rank, config), "projector": (rank, proj.in_features), "rms_weight": (rank,)}
