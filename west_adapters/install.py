"""Install registered adapters and validate the exact trainable state layout."""
import math

import torch
from torch import nn

from .base import FP32_ADD_THEN_CAST, targets
from .registry import checkpoint_method


def validate_options(config):
    for key in ("rank", "last_n"):
        if type(config.get(key)) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    alpha = config.get("alpha")
    if type(alpha) not in (int, float) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be a finite, positive number")
    if "residual_precision" in config and config["residual_precision"] != FP32_ADD_THEN_CAST:
        raise ValueError(f"unknown residual_precision: {config['residual_precision']!r}")
    method = checkpoint_method(config)
    method.validate_options(config)
    if method.attention:
        if "layer_ranks" in config:
            raise ValueError("legacy O adapters have a uniform rank, not layer_ranks")
        return
    ranks = config.get("layer_ranks")
    count = config["last_n"] * len(method.target_set)
    if (not isinstance(ranks, dict) or len(ranks) != count or
            any(type(r) is not int or r < 0 for r in ranks.values())):
        raise ValueError("invalid layer_ranks")
    method.validate_ranks(ranks, config)


def _with_adapter(proj, args, output):
    return proj.baseline_adapter(proj, args[0], output)


def install_from_config(model, config, *, directions=None, attention_installer=None):
    """Install on a pristine decoder; only fresh EVA training supplies directions."""
    validate_options(config)
    if any(n.endswith((".baseline_adapter", ".ta_adapter")) for n, _ in model.named_modules()):
        raise ValueError("adapter already installed")
    method = checkpoint_method(config)
    if directions is not None and not method.uses_directions:
        raise ValueError("EVA directions only apply to EVA")
    if method.attention:
        if attention_installer is None:
            raise ValueError(f"{method.name} requires a host attention installer")
        return attention_installer(model, **{k: config[k] for k in ("rank", "alpha", "mode", "last_n")},
                                   residual_precision=config.get("residual_precision"))
    projs = targets(model, config["last_n"], method.target_set)
    ranks = config["layer_ranks"]
    if ranks.keys() != projs.keys() or (directions is not None and directions.keys() != projs.keys()):
        raise ValueError("layer_ranks/directions do not name exactly the intended target projections")
    branches = {}
    for name, proj in projs.items():
        rank = ranks[name]
        if directions is not None and directions[name].shape != (rank, proj.in_features):
            raise ValueError(f"wrong EVA directions for {name}")
        if not rank:
            continue
        branch = method.build(proj, rank, config)
        if directions is not None:
            with torch.no_grad():
                branch.down.copy_(directions[name])
        branches[name] = branch
    model.requires_grad_(False)
    for name, branch in branches.items():
        projs[name].add_module("baseline_adapter", branch)
        projs[name].register_forward_hook(_with_adapter)
    return list(branches.values())


def baseline_trainables(model, config):
    method = checkpoint_method(config)
    projs = targets(model, config["last_n"], method.target_set)
    ranks = config["layer_ranks"]
    if ranks.keys() != projs.keys():
        raise ValueError("configured layer_ranks differ from target projections")
    expected = set()
    for name, proj in projs.items():
        rank = ranks[name]
        branch = getattr(proj, "baseline_adapter", None)
        if rank == 0 and branch is None:
            continue
        if branch is None or branch.mode != method.mode or rank == 0:
            raise ValueError(f"installed adapter differs from config in {name}")
        if not math.isclose(branch.scale, config["alpha"] / config["rank"], rel_tol=1e-12):
            raise ValueError(f"scale mismatch at {name}")
        for key, shape in method.shapes(proj, rank, config).items():
            param = getattr(branch, key, None)
            if not isinstance(param, nn.Parameter) or param.shape != shape:
                raise ValueError(f"invalid {key} at {name}")
            expected.add(f"{name}.baseline_adapter.{key}")
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    if not expected or params.keys() != expected:
        raise ValueError("trainable set differs from the configured projection adapters")
    return params
