"""ms-swift tuner for Thinker TA-LoRA on O with optional native PEFT Q/K/V adapters."""
import hashlib
import json
import re
import torch
from pathlib import Path
from safetensors.torch import load_file, save_file
from peft.tuners.lora.layer import Linear as LoraLinear
if __package__:
    from .adapter_audit import MLP_PROJECTIONS, ordered_projections
    from .swift_asr_ta_runtime import (TRANSPORT_GATES, attach, enable_transport_up, set_gate_multiplier,
                                       set_transport_gate, set_transport_queries, set_transport_weighting)
else:
    from adapter_audit import MLP_PROJECTIONS, ordered_projections
    from swift_asr_ta_runtime import (TRANSPORT_GATES, attach, enable_transport_up, set_gate_multiplier,
                                      set_transport_gate, set_transport_queries, set_transport_weighting)

from swift.tuner_plugin import Tuner, tuners_map


def _target_layout(model, names):
    layers = model.thinker.model.layers
    last_n = len({target.rsplit('.', 2)[0] for target in names})
    if not 1 <= last_n <= len(layers):
        raise ValueError('Expected a nonempty suffix of Thinker decoder layers')
    projections = ordered_projections({target.rsplit('.', 1)[-1] for target in names}, require_o=True)
    targets = [f'thinker.model.layers.{i}.{"mlp" if p in MLP_PROJECTIONS else "self_attn"}.{p}'
               for i in range(len(layers) - last_n, len(layers)) for p in projections]
    if set(targets) != set(names):
        raise ValueError('Select the same projections in every targeted Thinker decoder layer')
    return targets, last_n, projections


def _install(model, cfg, last_n, projections):
    # Formats 2 and 3 both follow native PEFT rounding. Format 3 also stores Q/K/V.
    residual_version = cfg['residual_format_version'] if cfg['format_version'] >= 4 else min(cfg['format_version'], 2)
    attach(model, last_n=last_n, rank=cfg['rank'], alpha=cfg['alpha'], format_version=residual_version)
    model._ta_lora_config = cfg
    set_gate_multiplier(model, cfg.get('gate_multiplier', 1.))
    set_transport_queries(model, cfg.get('transport_queries', 'text'))
    if cfg.get('independent_transport_up', False):
        enable_transport_up(model)
    set_transport_weighting(model, cfg.get('transport_weighting', 'uniform'))
    if cfg.get('transport_gate') is not None:
        set_transport_gate(model, cfg['transport_gate'])
    for layer in model.thinker.model.layers[-last_n:]:
        for name in projections:
            if name == 'o_proj':
                continue
            attn = layer.mlp if name in MLP_PROJECTIONS else layer.self_attn
            linear = LoraLinear(getattr(attn, name), 'default', r=cfg['rank'], lora_alpha=cfg['alpha'])
            linear.lora_A.float()
            linear.lora_B.float()
            setattr(attn, name, linear)


def _params(model, targets):
    params = dict(model.named_parameters())
    selected = {}
    for target in targets:
        if target.endswith('.o_proj'):
            prefix = target.removesuffix('o_proj') + 'ta_adapter.'
            factors = ('down', 'up', 'gate')
            if model._ta_lora_config.get('independent_transport_up', False):
                factors += ('transport_up',)
            if model._ta_lora_config.get('transport_gate') in ('question', 'question_audio'):
                factors += ('gate_weight',)
        else:
            prefix = target + '.'
            factors = ('lora_A.default.weight', 'lora_B.default.weight')
        for factor in factors:
            name = prefix + factor
            selected[name] = params[name]
    return selected


class TALoRATuner(Tuner):
    tuner_type = 'ta_lora'

    @classmethod
    def prepare_model(cls, args, model):
        for flag in ('packing', 'padding_free', 'vit_gradient_checkpointing', 'use_dora',
                     'use_rslora', 'modules_to_save', 'deepspeed', 'fsdp', 'quant_method'):
            if getattr(args, flag, False):
                raise ValueError(f'TA-LoRA recipe does not support {flag}')
        if getattr(args, 'sequence_parallel_size', 1) != 1 or getattr(args, 'lora_dropout', 0.) != 0.:
            raise ValueError('TA-LoRA requires sequence_parallel_size=1 and lora_dropout=0')
        if getattr(args, 'gradient_checkpointing', False):
            opts = getattr(args, 'gradient_checkpointing_kwargs', None) or {}
            if model.config.model_type != 'qwen3_asr' or opts.get('use_reentrant') is not False:
                raise ValueError('TA checkpointing requires Qwen3-ASR and non-reentrant replay')
        spec = getattr(args, 'target_modules', None)
        matched = {n for n, _ in model.named_modules() if isinstance(spec, str) and re.fullmatch(spec, n)}
        targets, last_n, projections = _target_layout(model, matched)
        cfg = dict(format_version=2 if len(projections) == 1 else 3, tuner_type=cls.tuner_type,
                   rank=args.lora_rank, alpha=args.lora_alpha, targets=targets)
        cfg['model_type'] = model.config.model_type
        _install(model, cfg, last_n, projections)
        # Store the initialized factors; paired fresh runs replace them with the shared draw.
        initial = hashlib.sha256()
        for name, p in _params(model, targets).items():
            if name.endswith('.gate'):
                p.requires_grad_(cls.tuner_type != 'ta_gate_off')
            else:
                initial.update(p.detach().cpu().contiguous().numpy().tobytes())
        cfg['ab_init_sha256'] = initial.hexdigest()
        model._ta_lora_config = cfg
        return model

    @staticmethod
    def save_pretrained(model, save_directory, state_dict=None, safe_serialization=True, **kwargs):
        # Never materialize a CPU copy of the full base. If Swift supplied a gathered
        # state dict, use it rather than potentially sharded local parameter data.
        params = _params(model, model._ta_lora_config['targets'])
        source = params if state_dict is None else state_dict
        tensors = {}
        for name, p in params.items():
            if (name not in source and name.endswith('.gate') and model._ta_lora_config['tuner_type'] == 'ta_gate_off'):
                value = p  # immutable zero; may be omitted from a trainable-only state dict
            else:
                value = source[name]
            tensors[name] = value.detach().cpu().contiguous()
        dest = Path(save_directory)
        dest.mkdir(parents=True, exist_ok=True)
        save_file(tensors, str(dest / 'adapter_model.safetensors'), metadata={'format': 'pt'})
        (dest / 'adapter_config.json').write_text(json.dumps(model._ta_lora_config, indent=2) + '\n')

    @classmethod
    def from_pretrained(cls, model, model_id, is_trainable=False, **kwargs):
        src = Path(model_id)
        cfg = json.loads((src / 'adapter_config.json').read_text())
        targets = cfg.get('targets')
        version = cfg.get('format_version')
        if (cfg.get('model_type', 'qwen2_5_omni') != model.config.model_type or version not in (1, 2, 3, 4, 5, 6, 7, 8) or cfg.get('tuner_type') != cls.tuner_type
                or (version >= 4 and ('gate_multiplier' not in cfg or cfg.get('residual_format_version') not in (1, 2)))
                or (version < 4 and cfg.get('gate_multiplier', 1.) != 1.)
                or (version >= 5 and cfg.get('transport_queries') not in ('text', 'all_valid'))
                or (version < 5 and cfg.get('transport_queries', 'text') != 'text')
                or (version < 7 and cfg.get('independent_transport_up', False) is not (version == 6))
                or (version < 7 and cfg.get('transport_weighting', 'uniform') != 'uniform')
                or (version >= 7 and (type(cfg.get('independent_transport_up')) is not bool
                                      or cfg.get('transport_weighting') not in ('uniform', 'attention_overlap')))
                or (version == 8 and cfg.get('transport_gate') not in TRANSPORT_GATES)
                or (version < 8 and 'transport_gate' in cfg)
                or not isinstance(targets, list) or not targets):
            raise ValueError('Invalid TA-LoRA checkpoint format')
        expected, last_n, projections = _target_layout(model, targets)
        if (targets != expected or (version < 3 and len(projections) != 1)
                or (version == 1 and last_n != 4)):
            raise ValueError('TA-LoRA checkpoint format or Thinker target layers do not match')
        tensors = load_file(str(src / 'adapter_model.safetensors'))
        fixed_gate = cls.tuner_type == 'ta_gate_off'
        if fixed_gate and any(v.count_nonzero() for k, v in tensors.items() if k.endswith('.gate')):
            raise ValueError('ta_gate_off checkpoint contains a nonzero gate')
        _install(model, cfg, last_n, projections)
        params = _params(model, targets)
        if tensors.keys() != params.keys() or any(tensors[k].shape != p.shape for k, p in params.items()):
            raise ValueError('TA-LoRA checkpoint keys or tensor shapes do not match')
        with torch.no_grad():
            for name, param in params.items():
                param.copy_(tensors[name])
                param.requires_grad_(is_trainable and not (fixed_gate and name.endswith('.gate')))
        model._ta_lora_config = cfg
        return model


class TAGateOffTuner(TALoRATuner):
    tuner_type = 'ta_gate_off'


tuners_map[TALoRATuner.tuner_type] = TALoRATuner
tuners_map[TAGateOffTuner.tuner_type] = TAGateOffTuner
