"""Swift projection adapters for BoRA and LoKRA.

BoRA reuses the project's Eq. 5 contraction in west_adapters.bora.
LoKRA follows AMD-AGI/LoKRA finetune_krank.py at 9e118de65047, including
its random minor size, stabilization, normalization, and per-layer mean.
"""
import math
import random

import torch
from torch import nn

from west_adapters.base import ProjectionBranch
from west_adapters.bora import BoRABranch
from .paper_late_loft import LoFTLinear, make_optimizer
from .paper_late_ralora import RaLoRALinear
from .paper_late_igu import IGULinear
from .paper_late_mixer import MixerLinear


class BranchLinear(nn.Module):
    def __init__(self, base, rank, alpha, settings):
        super().__init__()
        self.base_layer = base.requires_grad_(False)
        self.branch = self.branch_type(base, rank, alpha, {"mode": self.method, **settings})

    def forward(self, x):
        return self.branch(self.base_layer, x, self.base_layer(x))


class BoRALinear(BranchLinear):
    method = "bora"
    branch_type = BoRABranch

    def __init__(self, base, rank, alpha, settings):
        super().__init__(base, rank, alpha, {"bora_blocks": 2, **settings})


class LoKRALinear(BranchLinear):
    method = "lokra"
    branch_type = ProjectionBranch

    def __init__(self, base, rank, alpha, settings):
        if not 3 <= rank <= min(base.in_features, base.out_features):
            raise ValueError("LoKRA requires rank between 3 and the smaller projection width")
        super().__init__(base, rank, alpha, settings)


class StableLinear(BranchLinear):
    method = "stable-lora"
    branch_type = ProjectionBranch


BUILDERS = {"bora": BoRALinear, "lokra": LoKRALinear, "stable-lora": StableLinear,
            "loft": LoFTLinear, "ralora": RaLoRALinear, "igu-lora": IGULinear,
            "lora-mixer": MixerLinear}


def initialize_training(trainer, method, config):
    if method == "lora-mixer":
        from .paper_late_mixer import prepare_experts
        return prepare_experts(trainer, config)
    if method == "igu-lora":
        from .paper_late_igu import prune
        return prune(trainer, config["method_options"][method])
    if method == "ralora":
        from .paper_late_ralora import calibrate
        return calibrate(trainer, config["method_options"][method])
    return {}


def install_optimizer(method, model, optimizer, settings):
    if method == "loft":
        return make_optimizer(model, optimizer, settings)
    if method != "stable-lora":
        raise ValueError(f"No Stable-LoRA optimizer for {method}")
    shrink = float(settings.get("shrink_ratio", 0.001))
    if not 0 < shrink < 1:
        raise ValueError("shrink_ratio must be strictly between zero and one")
    pairs = [(name, layer.branch) for name, layer in model.named_modules()
             if isinstance(layer, StableLinear)]
    if not pairs:
        raise ValueError("No Stable-LoRA projections found")
    # The authors keep a monotone per-pair stop flag. Keeping it in a group
    # makes ordinary torch optimizer save/load also restore interrupted runs.
    optimizer.param_groups[0]["stable_lora_names"] = [name for name, _ in pairs]
    optimizer.param_groups[0]["stable_lora_stopped"] = [False] * len(pairs)

    @torch.no_grad()
    def shrink_before_step(opt, args, kwargs):
        group = opt.param_groups[0]
        if group["stable_lora_names"] != [name for name, _ in pairs]:
            raise ValueError("Stable-LoRA checkpoint target names differ from this model")
        flags = group["stable_lora_stopped"]
        active = [(i, branch) for i, (_, branch) in enumerate(pairs) if not flags[i]]
        if not active:
            return
        # One synchronization per device instead of one per projection.
        devices = {}
        for i, branch in active:
            a, b = branch.down, branch.up
            devices.setdefault(a.device, []).append((i, a, a.norm() < b.norm() * (a.numel() / b.numel())))
        for entries in devices.values():
            stopped = torch.stack([flag for _, _, flag in entries]).tolist()
            for (idx, a, _), stop in zip(entries, stopped):
                if stop:
                    flags[idx] = True
                else:
                    a.mul_(1 - shrink)

    optimizer.register_step_pre_hook(shrink_before_step)
    return optimizer


def _negative_minor_logdet(matrix, k):
    indices = torch.randperm(matrix.shape[0], device="cpu")[:k].to(matrix.device)
    rows = matrix.index_select(0, indices).float()
    gram = rows @ rows.T
    identity = torch.eye(k, device=rows.device, dtype=rows.dtype)
    return -torch.linalg.slogdet(gram + 1e-6 * identity).logabsdet


def training_penalty(model, method, settings):
    if method == "lora-mixer":
        from .paper_late_mixer import mixer_loss
        return mixer_loss(model, settings)
    if method == "igu-lora":
        from .paper_late_igu import orthogonality
        return getattr(model, "_igu_pruning_accumulation", 1) * orthogonality(
            m for m in model.modules() if isinstance(m, IGULinear))
    if method != "lokra":
        raise ValueError(f"No LoKRA penalty for {method}")
    layers = [m for m in model.modules() if isinstance(m, LoKRALinear)]
    if not layers:
        raise ValueError("No LoKRA projections found")
    if not model.training:
        return layers[0].branch.down.new_zeros(())
    coefficient = float(settings.get("lambda_coef", 0.001))
    if not math.isfinite(coefficient) or coefficient < 0:
        raise ValueError("LoKRA lambda_coef must be nonnegative and finite")
    penalties = []
    for layer in layers:
        a, b = layer.branch.down.T, layer.branch.up
        k = random.randint(3, a.shape[1])
        with torch.autocast(device_type=a.device.type, enabled=False):
            # The authors sample k in [3,r] and rescale before summing A and B.
            norm = k * (math.log(k) - 1) / 2
            penalties.append((_negative_minor_logdet(a, k) + _negative_minor_logdet(b, k)) / norm)
    return torch.stack(penalties).mean() * coefficient
