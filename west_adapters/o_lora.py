"""LoRA at the attention output, with the historical ta_adapter tensor layout."""
import torch
from torch.nn import functional as F

from .base import AdapterMethod, LowRankWeights


class OutputAtoms(LowRankWeights):
    def __init__(self, proj, rank, alpha, residual_precision):
        super().__init__(proj, rank, alpha)
        self.residual_precision = residual_precision
        self.gate = None

    def coefficients(self, ordinary, contrast):
        return F.linear(ordinary.float(), self.down.float())

    def forward(self, ordinary, contrast=None):
        with torch.autocast(device_type=ordinary.device.type, enabled=False):
            return self.scale * F.linear(self.coefficients(ordinary, contrast), self.up.float())

    def attention_delta(self, ordinary, logits, values, mask, cache_position, layout):
        """Receive native eager attention state and return the output-projection residual."""
        return self(ordinary, None)


class OutputLoRA(AdapterMethod):
    name = "o-lora"
    mode = "lora"
    attention = True
    atoms = OutputAtoms

    def atom_shapes(self, module, config):
        rank = config["rank"]
        if module.down.ndim != 2 or module.up.ndim != 2:
            raise ValueError("Expected matrix-valued attention atoms")
        return {"down": (rank, module.down.shape[1]), "up": (module.up.shape[0], rank)}
