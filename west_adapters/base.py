"""Shared contracts for Qwen decoder projections and attention adapters."""
import math

import torch
from torch import nn
from torch.nn import functional as F


FP32_ADD_THEN_CAST = "fp32_add_then_cast"


def add_fp32_residual(base, delta):
    return (base.float() + delta).to(base.dtype)


def move_inputs(batch, device, dtype):
    return {k: v.to(device=device, dtype=dtype if v.is_floating_point() else v.dtype)
            for k, v in batch.items()}


def targets(model, last_n, target_set):
    layers = model.model.layers
    if type(last_n) is not int or not 1 <= last_n <= len(layers):
        raise ValueError(f"last_n must be between 1 and {len(layers)}")
    if target_set not in ("o", "qv"):
        raise ValueError("target_set must be o or qv")
    result = {}
    for i in range(len(layers) - last_n, len(layers)):
        for label in target_set:
            attr = label + "_proj"
            name = f"model.layers.{i}.self_attn.{attr}"
            proj = getattr(layers[i].self_attn, attr)
            if not isinstance(proj, nn.Linear):
                raise ValueError(f"expected a native linear projection: {name}")
            result[name] = proj
    return result


class LowRankWeights(nn.Module):
    def __init__(self, proj, rank, alpha):
        super().__init__()
        opts = {"dtype": torch.float32, "device": proj.weight.device}
        self.down = nn.Parameter(torch.empty(rank, proj.in_features, **opts))
        self.up = nn.Parameter(torch.zeros(proj.out_features, rank, **opts))
        nn.init.kaiming_uniform_(self.down, a=math.sqrt(5))
        self.scale = float(alpha) / rank


class ProjectionBranch(LowRankWeights):
    def __init__(self, proj, rank, alpha, config):
        super().__init__(proj, rank, alpha)
        self.mode = config["mode"]

    def coefficients(self, x):
        # ZeRO-2 stores BF16 model parameters alongside FP32 optimizer masters.
        # Read the model weights in FP32 for the low-rank and residual arithmetic.
        return F.linear(x, self.down.float())

    def delta(self, proj, x, base_output):
        return F.linear(self.coefficients(x), self.up.float()) * self.scale

    def forward(self, proj, x, base_output):
        with torch.autocast(device_type=x.device.type, enabled=False):
            return add_fp32_residual(base_output, self.delta(proj, x.float(), base_output))


class AdapterMethod:
    """A method owns its train-time options, rank policy and adapter branch."""
    target_set = "o"
    attention = False
    needs_audio_context = False
    uses_directions = False
    branch = ProjectionBranch
    options = {}

    def resolve_rank(self, model, options):
        return options["rank"]

    def checkpoint_options(self, options):
        return {key: options[key] for key in self.options}

    def validate_options(self, options):
        pass

    def validate_ranks(self, ranks, config):
        if any(r != config["rank"] for r in ranks.values()):
            raise ValueError("layer_ranks must match configured rank")

    def build(self, proj, rank, config):
        return self.branch(proj, rank, config["alpha"], config)

    def shapes(self, proj, rank, config):
        return {"down": (rank, proj.in_features), "up": (proj.out_features, rank)}
