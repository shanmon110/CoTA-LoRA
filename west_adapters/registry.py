"""Method resolution for tagged WEST LoRA configs and adapter checkpoints."""
import math
from collections.abc import Mapping

from .base import FP32_ADD_THEN_CAST, AdapterMethod, targets
from .bora import BoRA
from .dora import DoRA
from .eva import EVA
from .o_lora import OutputLoRA
from .qv_lora import QueryValueLoRA
from .ta_lora import TransportLoRA
from .toplora import TopLoRA


_METHODS = {}
_CHECKPOINT_METHODS = {}
_COMMON_FIELDS = {"method", "r", "lora_alpha", "last_n", "target_modules", "lora_dropout",
                  "task_type", "inference_mode"}


def register_method(method: AdapterMethod):
    """Register one config name and one unambiguous checkpoint (mode, target_set)."""
    if (not isinstance(method, AdapterMethod) or
            not all(isinstance(getattr(method, key, None), str) and getattr(method, key).strip()
                    for key in ("name", "mode")) or method.target_set not in ("o", "qv")):
        raise ValueError("Adapters must declare a name, checkpoint mode and o or qv targets")
    if method.attention and method.target_set != "o":
        raise ValueError("Attention adapters require output projection targets")
    reserved = _COMMON_FIELDS | {"mode", "target_set", "rank", "alpha", "layer_ranks", "residual_precision"}
    if not isinstance(method.options, Mapping) or reserved.intersection(method.options):
        raise ValueError("Method options must be a mapping with distinct method-specific keys")
    key = (method.mode, method.target_set)
    if method.name in _METHODS or key in _CHECKPOINT_METHODS:
        raise ValueError(f"Duplicate WEST adapter method: {method.name} / {key}")
    _METHODS[method.name] = _CHECKPOINT_METHODS[key] = method


for _method in (OutputLoRA(), TransportLoRA(), QueryValueLoRA(), TopLoRA(), BoRA(), DoRA(), EVA()):
    register_method(_method)


def get_method(name):
    if not isinstance(name, str) or name not in _METHODS:
        raise ValueError(f"Unknown WEST adapter method {name!r}; choose from {', '.join(_METHODS)}")
    return _METHODS[name]


def checkpoint_method(config):
    key = config["mode"], config.get("target_set", "o")
    try:
        return _CHECKPOINT_METHODS[key]
    except (KeyError, TypeError):
        raise ValueError(f"unsupported mode/target_set: {key}") from None


def resolve_lora_config(spec):
    """Translate a tagged PEFT-style dict to the options used by WEST trainers."""
    if not isinstance(spec, Mapping):
        raise ValueError("lora_config must be a mapping")
    method = get_method(spec.get("method"))
    allowed = _COMMON_FIELDS | method.options.keys()
    unknown = spec.keys() - allowed
    if unknown:
        raise ValueError(f"Unsupported {method.name} options: {sorted(unknown)}")
    targets_required = [s + "_proj" for s in method.target_set]
    modules = spec.get("target_modules", targets_required)
    if (not isinstance(modules, (list, tuple)) or not all(isinstance(v, str) for v in modules)
            or sorted(modules) != sorted(targets_required)):
        raise ValueError(f"{method.name} requires target_modules {targets_required}")
    dropout = spec.get("lora_dropout", 0.)
    if (spec.get("task_type", "CAUSAL_LM") != "CAUSAL_LM" or
            spec.get("inference_mode", False) is not False or type(dropout) not in (int, float) or dropout != 0.):
        raise ValueError("Tagged adapters require CAUSAL_LM, inference_mode=false and lora_dropout=0")
    options = dict(mode=method.mode, target_set=method.target_set, rank=spec.get("r", 8),
                   alpha=spec.get("lora_alpha", 16.), last_n=spec.get("last_n", 4))
    if (type(options["rank"]) is not int or options["rank"] < 1 or
            type(options["last_n"]) is not int or options["last_n"] < 1 or
            type(options["alpha"]) not in (int, float) or not math.isfinite(options["alpha"]) or
            options["alpha"] <= 0):
        raise ValueError("r and last_n must be positive integers and lora_alpha must be finite and positive")
    options.update({name: spec.get(name, default) for name, default in method.options.items()})
    method.validate_options(options)
    return options


def resolve_layout(model, options):
    """Resolve the train-time rank against the host's actual projection widths."""
    method = checkpoint_method(options)
    method.validate_options(options)
    config = dict(rank=method.resolve_rank(model, options), alpha=options["alpha"],
                  last_n=options["last_n"], mode=method.mode, residual_precision=FP32_ADD_THEN_CAST,
                  **method.checkpoint_options(options))
    if not method.attention:
        config["target_set"] = method.target_set
    return method, config, targets(model, config["last_n"], method.target_set)


def finalize_ranks(config, projections, directions=None):
    method = checkpoint_method(config)
    if directions is not None:
        if not method.uses_directions or directions.keys() != projections.keys():
            raise ValueError("Calibration directions must match the method and target projections")
        ranks = {name: rows.shape[0] for name, rows in directions.items()}
    else:
        ranks = dict.fromkeys(projections, config["rank"])
    if not method.attention:
        config["layer_ranks"] = ranks
    return ranks
