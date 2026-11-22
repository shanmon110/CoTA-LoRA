"""Swift external tuner for paper-specific linear modules on the Omni Thinker."""
import hashlib
import importlib
import json
import math
from pathlib import Path
import re

from safetensors.torch import load_file, save_file
import torch
from torch import nn
from swift.tuner_plugin import Tuner, tuners_map

from experiments.ms_swift_librispeech.adapter_audit import MLP_PROJECTIONS, ordered_projections


# Only the selected module is imported. An optional dependency for another paper
# must not prevent native PEFT or any other experiment from starting.
FAMILIES = {
    'lowra':'middle', 'gralora': 'middle', 'toplora': 'middle', 'magnitude-loram': 'middle',
    'aurora': 'middle', 'flylora': 'middle', 'bora': 'late', 'lokra': 'late',
    'stable-lora': 'late', 'loft': 'late', 'ralora':'late', 'igu-lora':'late', 'lora-mixer':'late', 'unilora':'middle', 'alora':'middle',
}


def family(method):
    return importlib.import_module('experiments.ms_swift_aqa.paper_' + FAMILIES[method])


def select_targets(model, targets):
    if not isinstance(targets, list) or not targets or len(set(targets)) != len(targets):
        raise ValueError('Expected distinct Thinker projection paths')
    layers = model.thinker.model.layers
    projections = ordered_projections({path.rsplit('.', 1)[1] for path in targets})
    indices = sorted({int(path.split('.')[3]) for path in targets})
    expected = {
        f'thinker.model.layers.{i}.{"mlp" if p in MLP_PROJECTIONS else "self_attn"}.{p}'
        for i in indices for p in projections}
    if indices != list(range(len(layers)-len(indices), len(layers))) or expected != set(targets):
        raise ValueError('Targets must form a common set of decoder projections in consecutive final layers')
    return indices, projections


def install(model, spec):
    method = spec['method']
    if (spec['format_version'] != 1 or model.config.model_type != spec['model_type']
            or spec['model_type'] != 'qwen2_5_omni' or not isinstance(spec['settings'], dict)
            or type(spec['rank']) != int or spec['rank'] < 1 or not math.isfinite(spec['alpha'])):
        raise ValueError('Invalid paper adapter specification or incompatible base model')
    select_targets(model, spec['targets'])
    build = family(method).BUILDERS[method]
    model.requires_grad_(False)
    for path in spec['targets']:
        parent_name, key = path.rsplit('.', 1)
        parent = model.get_submodule(parent_name)
        linear = getattr(parent, key)
        if not isinstance(linear, nn.Linear):
            raise TypeError(f'{path} is not a clean linear projection')
        new = build(linear, spec['rank'], spec['alpha'], spec['settings'])
        if new.base_layer is not linear or any(p.requires_grad for p in linear.parameters()):
            raise ValueError(f'{method} changed the trainable scope of {path}')
        setattr(parent, key, new.to(device=linear.weight.device))
    model._paper_adapter_spec = spec
    finalizer = getattr(family(method), 'after_install', None)
    if finalizer:
        finalizer(model, spec)
    return model


def adapter_state(model):
    state = {}
    for path in model._paper_adapter_spec['targets']:
        module = model.get_submodule(path)
        # Keep persistent buffers such as permutations along with trainable tensors.
        for name, tensor in module.state_dict(keep_vars=True).items():
            if not name.startswith('base_layer.'):
                state[f'{path}.{name}'] = tensor
    return state


class PaperTuner(Tuner):
    @classmethod
    def prepare_model(cls, args, model):
        targets = sorted(name for name, module in model.named_modules()
                         if isinstance(module, nn.Linear) and re.fullmatch(args.target_modules, name))
        spec = dict(format_version=1, method=cls.method, model_type=model.config.model_type,
                    rank=args.lora_rank, alpha=args.lora_alpha, targets=targets,
                    settings=args.model_kwargs.get('paper_settings', {}))
        return install(model, spec)

    @staticmethod
    def save_pretrained(model, save_directory, state_dict=None, **kwargs):
        dest = Path(save_directory)
        dest.mkdir(parents=True, exist_ok=True)
        state = adapter_state(model)
        trainables = dict(model.named_parameters())
        source = {}
        for name, value in state.items():
            if state_dict is not None and name in state_dict:
                value = state_dict[name]
            elif state_dict is not None and name in trainables and trainables[name].requires_grad:
                raise ValueError(f'Missing gathered trainable tensor {name}')
            source[name] = value.detach().cpu().contiguous().clone()
        save_file(source, str(dest/'adapter_model.safetensors'), metadata={'format': 'pt'})
        (dest/'adapter_config.json').write_text(json.dumps(model._paper_adapter_spec, indent=2)+'\n')

    @classmethod
    def from_pretrained(cls, model, model_id, is_trainable=False, **kwargs):
        cp = Path(model_id)
        spec = json.loads((cp/'adapter_config.json').read_text())
        if spec['method'] != cls.method:
            raise ValueError('Adapter and requested method disagree')
        install(model, spec)
        expected = adapter_state(model)
        values = load_file(str(cp/'adapter_model.safetensors'))
        if set(values) != set(expected):
            raise ValueError('Incomplete or incompatible paper adapter tensors')
        for key, value in values.items():
            if expected[key].shape != value.shape or expected[key].dtype != value.dtype:
                raise ValueError(f'Invalid adapter tensor {key}')
        with torch.no_grad():
            for key, value in values.items():
                expected[key].copy_(value)
        if not is_trainable:
            model.requires_grad_(False)
        return model


def options(method, config, base_options):
    return dict(base_options, tuner_type='paper_' + method,
                external_plugins=[str(Path(__file__).resolve())],
                model_kwargs={'paper_settings':config['method_options'][method]})


def audit(model, method, config):
    spec = model._paper_adapter_spec
    state = adapter_state(model)
    targets, projections = select_targets(model, spec['targets'])
    if (method != spec['method'] or spec['settings'] != config['method_options'][method]
            or len(targets) != config['adapter']['last_n']
            or projections != ordered_projections(config['adapter']['target_modules'])):
        raise ValueError('Paper module layout differs from the training configuration')
    trained = {k:p for k,p in model.named_parameters() if p.requires_grad}
    if not trained or set(trained)-set(state) or any(p.dtype != torch.float32 for p in trained.values()):
        raise ValueError('Unexpected trainable tensors in paper adapter')
    checksum = hashlib.sha256()
    for k, p in sorted(trained.items()):
        checksum.update(k.encode()+b'\0')
        checksum.update(p.detach().cpu().contiguous().numpy().tobytes())
    return dict(param_sha256=checksum.hexdigest(), trainable_parameters=sum(p.numel() for p in trained.values()),
                target_module_count=len(spec['targets']), target_layers=targets, target_projections=projections)


def prepare_trainer(trainer, method, resume=None):
    if method in ('stable-lora', 'loft'):
        from swift.optimizers import OptimizerCallback

        class PaperOptimizer(OptimizerCallback):
            def create_optimizer(self, model=None):
                if self.trainer.optimizer is not None:
                    return self.trainer.optimizer
                target = self.trainer.model if model is None else model
                original = super().create_optimizer(model=model)
                settings = target._paper_adapter_spec['settings']
                optimizer = family(method).install_optimizer(method, target, original, settings)
                self.trainer.optimizer = optimizer
                return optimizer

        trainer.optimizer_callback = PaperOptimizer(trainer.args, trainer)
    callback = getattr(family(method), 'prepare_trainer', None)
    if callback:
        callback(trainer, method, resume=resume)


def initialize_training(trainer, method, config):
    initialize = getattr(family(method), 'initialize_training', None)
    extra = initialize(trainer, method, config) if initialize else {}
    return {**audit(trainer.model, method, config), **extra}


for name in FAMILIES:
    tuners_map['paper_' + name] = type('PaperTuner_' + name, (PaperTuner,), {'method': name})
