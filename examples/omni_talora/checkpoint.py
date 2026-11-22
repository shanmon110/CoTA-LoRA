"""Strict adapter-only checkpoint I/O. No pickle and no base model state."""

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path


CONFIG = "adapter_config.json"
WEIGHTS = "adapter.safetensors"
_FORMAT = "west-omni-talora-v1"


def _validate(config):
    required = {"model_name_or_path", "revision", "task", "rank", "alpha", "last_n", "mode"}
    if not isinstance(config, dict) or not required <= config.keys():
        raise ValueError(f"adapter config must contain {sorted(required)}")
    if not isinstance(config["model_name_or_path"], str) or not config["model_name_or_path"].strip():
        raise ValueError("model_name_or_path must be a nonempty string")
    revision = config["revision"]
    if revision is not None and (not isinstance(revision, str) or
                                 not re.fullmatch("[a-fA-F0-9]{40}", revision)):
        raise ValueError("revision must be a resolved 40-hex commit or null")
    if config["task"] not in ("asr", "ser"):
        raise ValueError("unsupported adapter mode or task")
    from .adapters import validate_options
    validate_options(config)


def _metadata(config):
    raw = json.dumps(config, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    return {"format": _FORMAT, "config_sha256": hashlib.sha256(raw).hexdigest()}


def read_config(directory):
    path = Path(directory)
    config = json.loads((path / CONFIG).read_text(encoding="utf-8"))
    _validate(config)
    from safetensors import safe_open

    with safe_open(str(path / WEIGHTS), framework="pt", device="cpu") as file:
        if file.metadata() != _metadata(config):
            raise ValueError("adapter config does not match the safetensors archive")
    return config


def _trainable(thinker, config):
    from west_adapters.install import baseline_trainables
    from west_adapters.registry import checkpoint_method
    method = checkpoint_method(config)
    if not method.attention:
        return baseline_trainables(thinker, config)
    params = {n: p for n, p in thinker.named_parameters() if p.requires_grad}
    if not params:
        raise ValueError("no trainable adapter parameters; install the adapter first")
    modules = [(n, m) for n, m in thinker.named_modules() if n.split(".")[-1] == "ta_adapter"]
    if len(modules) != config["last_n"]:
        raise ValueError("installed adapter layer count does not match config")
    rank = config["rank"]
    expected = set()
    for name, module in modules:
        if getattr(module, "residual_precision", None) != config.get("residual_precision"):
            raise ValueError(f"installed residual_precision differs from config in {name}")
        shapes = method.atom_shapes(module, config)
        actual = {k: tuple(p.shape) for k, p in module.named_parameters()}
        if shapes != actual:
            raise ValueError(f"attention adapter parameters differ from config at {name}")
        expected.update(f"{name}.{key}" for key in shapes)
        if module.scale != config["alpha"] / rank:
            raise ValueError(f"adapter scale differs from configured alpha/rank in {name}")
    if expected != params.keys():
        raise ValueError("trainable set differs from installed adapter config: "
                         f"missing={sorted(expected-params.keys())}, extra={sorted(params.keys()-expected)}")
    return params


def save_adapter(thinker, directory, config: dict):
    """Write only installed trainable adapter tensors and a bound configuration."""
    _validate(config)
    metadata = _metadata(config)
    import torch
    from safetensors.torch import save_file

    state = {n: p.detach().cpu().contiguous() for n, p in _trainable(thinker, config).items()}
    for name, tensor in state.items():
        if not tensor.is_floating_point() or not torch.isfinite(tensor).all():
            raise ValueError(f"nonfloating or nonfinite adapter parameter: {name}")
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    if any(os.path.lexists(path / f) for f in (CONFIG, WEIGHTS)):
        raise FileExistsError(f"checkpoint already exists (or is incomplete): {path}")
    # Hard links publish completed files atomically, and unlike os.replace they
    # cannot clobber a concurrent winner. The config is the final commit marker.
    with tempfile.TemporaryDirectory(dir=path) as stage:
        stage = Path(stage)
        save_file(state, str(stage / WEIGHTS), metadata=metadata)
        (stage / CONFIG).write_text(json.dumps(config, indent=2, allow_nan=False) + "\n",
                                    encoding="utf-8")
        os.link(stage / WEIGHTS, path / WEIGHTS)
        try:
            os.link(stage / CONFIG, path / CONFIG)
        except OSError:
            (path / WEIGHTS).unlink()  # only the link this call successfully created
            raise


def load_adapter(thinker, directory):
    """Load into an already installed adapter only after validating every tensor."""
    import torch
    from safetensors import safe_open

    config = read_config(directory)
    params = _trainable(thinker, config)
    with safe_open(str(Path(directory) / WEIGHTS), framework="pt", device="cpu") as file:
        if file.metadata() != _metadata(config):
            raise ValueError("adapter config changed while opening weights")
        keys = set(file.keys())
        if keys != params.keys():
            raise ValueError(f"adapter tensor names differ: missing={sorted(params.keys()-keys)}, "
                             f"extra={sorted(keys-params.keys())}")
        state = {}
        for name, param in params.items():
            value = file.get_tensor(name)
            if value.shape != param.shape or value.dtype != param.dtype:
                raise ValueError(f"adapter tensor shape or dtype mismatch for {name}")
            if not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError(f"nonfloating or nonfinite checkpoint tensor: {name}")
            state[name] = value
        with torch.no_grad():
            for name, value in state.items():
                params[name].copy_(value)
    return config
