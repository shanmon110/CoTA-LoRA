"""LoFT integration around the authors' unmodified MIT-licensed optimizer.

optimizer.py, optim_helper.py and LICENSE originate from
https://github.com/tnurbek/loft/tree/148998d98f901cd7744db7baa5b2b4868fa16bf0 .
"""
import torch
from torch import nn

from .optim_helper import clip_grads_with_norm_, rescale_gradients
from .optimizer import LoFTAdamW


class LoFTLinear(nn.Module):
    def __init__(self, base, rank, alpha, settings):
        super().__init__()
        if alpha != rank:
            raise ValueError("LoFT requires alpha equal to rank (unit scaling)")
        self.base_layer = base.requires_grad_(False)
        kw = {"bias": False, "device": base.weight.device, "dtype": torch.float32}
        self.lora_A = nn.Linear(base.in_features, rank, **kw)
        self.lora_B = nn.Linear(rank, base.out_features, **kw)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        base = self.base_layer(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            return (base.float() + self.lora_B(self.lora_A(x.float()))).to(base.dtype)


class RestartableLoFT(LoFTAdamW):
    def state_dict(self):
        state = super().state_dict()
        state["loft_update_A"] = self.update_A
        state["loft_old_params"] = self.old_params.copy()
        return state

    def load_state_dict(self, state_dict):
        self.update_A = state_dict["loft_update_A"]
        self.old_params = {}
        for name, saved in state_dict["loft_old_params"].items():
            current = self.lora_name_to_params.get(name)
            if current is None or saved.shape != current.shape:
                raise ValueError(f"LoFT checkpoint target mismatch: {name}")
            self.old_params[name] = saved.to(current)
        super().load_state_dict(state_dict)


@torch.no_grad()
def projected_clip(optimizer, max_norm):
    # Paper's Building Block 6 uses ||projected effective gradient||_F.
    # The released hf_callbacks.py imports clip_grad_norm from the wrong module.
    # Taking the trace of two rank-by-rank Gram matrices avoids the dense update.
    squared_norms = []
    for param, name in optimizer.lora_params_to_name.items():
        if param.grad is None:
            continue
        is_a = "lora_A" in name
        if optimizer.update_A != is_a:
            continue
        other = optimizer.lora_name_to_params[optimizer._get_other_param_name(name)].detach()
        rescaled, _ = rescale_gradients(param.grad, other, is_a)
        if is_a:
            gram, cov = other.T @ other, rescaled @ rescaled.T
        else:
            gram, cov = other @ other.T, rescaled.T @ rescaled
        squared_norms.append((gram * cov).sum())
    if squared_norms:
        norm = torch.stack(squared_norms).sum().clamp_min(0).sqrt()
        clip_grads_with_norm_(optimizer.lora_params_to_name.keys(), max_norm, norm)


def make_optimizer(model, existing, settings):
    eps = float(settings.get("eps", 1e-4))
    clip = float(settings.get("max_grad_norm", 1.))
    if not eps > 0 or not clip >= 0:
        raise ValueError("LoFT requires eps > 0 and max_grad_norm >= 0")
    groups = [{**group, "eps": eps} for group in existing.param_groups]
    # Upstream subtracts weight_decay directly; this comparison uses no decay.
    if any(group.get("weight_decay", 0) for group in groups):
        raise ValueError("LoFT recipe uses weight_decay=0")
    optimizer = RestartableLoFT(groups, model=model, eps=eps, weight_decay=0.)
    if not optimizer.lora_params_to_name:
        raise ValueError("LoFT optimizer found no trainable A/B factors")
    if clip:
        optimizer.register_step_pre_hook(lambda opt, args, kwargs: projected_clip(opt, clip))
    return optimizer
