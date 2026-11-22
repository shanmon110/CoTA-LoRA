"""Paper methods built on upstream LoRA, using Swift's native trainer and precision."""
import hashlib
import json
import math
import os
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import TrainerCallback

from experiments.ms_swift_librispeech.adapter_audit import audit_adapter, initialize_pair


RESIDUAL = 'paper_residual.safetensors'


def options(method, config, base_options):
    settings = config['method_options'][method]
    if base_options['tuner_type'] != 'lora' or base_options['tuner_backend'] != 'peft':
        raise ValueError('Expected native PEFT LoRA as the starting adapter')
    if method == 'lora-plus':
        if set(settings) != {'lorap_lr_ratio'} or not math.isfinite(settings['lorap_lr_ratio']) or settings['lorap_lr_ratio'] <= 1:
            raise ValueError('Specify a finite LoRA+ B:A learning rate ratio above one')
        return {**base_options, **settings}
    if method == 'lora-ga':
        if (set(settings) != {'direction','scale','stable_gamma','calibration_batches','calibration_batch_size'}
                or settings['direction'] != 'ArB2r' or settings['scale'] != 'stable'
                or not math.isfinite(settings['stable_gamma']) or settings['stable_gamma'] <= 0
                or any(type(settings[k]) is not int or settings[k]<1 for k in ('calibration_batches','calibration_batch_size'))):
            raise ValueError('Unsupported LoRA-GA initialization or calibration settings')
        return dict(base_options)
    if method == 'randlora':
        register_randlora(settings)
        return {**base_options, 'tuner_type': 'randlora'}
    if method == 'pissa' and settings == {'init_weights': 'pissa_niter_16'}:
        return {**base_options, **settings}
    raise ValueError(f'Unsupported method or settings: {method} {settings}')


def audit(model, method, config):
    if method == 'randlora':
        return audit_randlora(model, config)
    ad = config['adapter']
    info = audit_adapter(model, 'lora', ad['last_n'], ad['target_modules'])
    info['param_sha256'] = info['ab_sha256']
    info['target_module_count'] = len(info['target_layers']) * len(info['target_projections'])
    cfg = model.peft_config['default']
    if method == 'lora-plus':
        if cfg.lorap_lr_ratio != config['method_options'][method]['lorap_lr_ratio']:
            raise ValueError('Configured LoRA+ ratio differs from model')
        info['lorap_lr_ratio'] = cfg.lorap_lr_ratio
    return info


def initialize(model, method, config):
    if method == 'lora-plus':
        a = config['adapter']
        initialize_pair(model, 'lora', config['training']['seed'], a['last_n'], a['target_modules'])
    return audit(model, method, config)


def initialize_training(trainer, method, config):
    if method != 'lora-ga':
        return initialize(trainer.model, method, config)
    from .paper_early_ga import calibrate
    audit(trainer.model, method, config)
    calibration = calibrate(trainer, config['method_options'][method], config['training']['seed'],
                            _residual_layers(trainer.model))
    return {**audit(trainer.model, method, config), **calibration}


def check_loraplus_optimizer(model, optimizer, base_lr, ratio):
    """Verify each actual optimizer parameter, including strict coverage and LR."""
    rates = {'lora_A': base_lr, 'lora_B': base_lr * ratio}
    expected = {id(p): (name, rates[name.split('.')[-3]]) for name, p in model.named_parameters() if p.requires_grad}
    seen = set()
    for group in optimizer.param_groups:
        for param in group['params']:
            name, lr = expected[id(param)]
            if id(param) in seen or not math.isclose(group['lr'], lr, rel_tol=1e-8, abs_tol=0):
                raise ValueError(f'Incorrect LoRA+ learning rate for {name}: expected {lr}, found {group["lr"]}')
            seen.add(id(param))
    if seen != expected.keys():
        raise ValueError('Missing trainable LoRA+ parameters in optimizer')
    return rates


def _residual_layers(model):
    return {name: module for name, module in model.get_base_model().named_modules()
            if hasattr(module, 'lora_A') and 'default' in module.lora_A}


def save_residual(model, path):
    """Retain the exact BF16 W_res which was used for training after native initialization."""
    weights = {name: module.get_base_layer().weight.detach().cpu().contiguous()
               for name, module in _residual_layers(model).items()}
    if not weights:
        raise ValueError('No LoRA targets from which to persist residuals')
    tmp = path.with_suffix('.tmp')
    save_file(weights, tmp)
    tmp.replace(path)
    digest = path.with_suffix('.sha256')
    digest.write_text(_file_hash(path) + '\n')


def _file_hash(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


@torch.no_grad()
def _install_saved_residual(model, path):
    if _file_hash(path) != path.with_suffix('.sha256').read_text().strip():
        raise ValueError('Frozen residual SHA-256 mismatch')
    layers = _residual_layers(model)
    with safe_open(path, framework='pt', device='cpu') as weights:
        if set(weights.keys()) != layers.keys():
            raise ValueError('Residual sidecar does not match adapter target names')
        for name, layer in layers.items():
            weight = layer.get_base_layer().weight
            value = weights.get_tensor(name)
            if value.shape != weight.shape or value.dtype != weight.dtype:
                raise ValueError(f'Incorrect residual shape or dtype for {name}')
            weight.copy_(value)


def restore_residual_adapter(base, checkpoint):
    """Rebuild exactly the trained forward path, including rounding of the frozen residual."""
    checkpoint = Path(checkpoint)
    cfg = LoraConfig.from_pretrained(checkpoint)
    cfg.init_lora_weights = True  # W_res is already persisted. No random decomposition on reload.
    model = PeftModel.from_pretrained(base, checkpoint, config=cfg, is_trainable=False)
    _install_saved_residual(model, checkpoint / RESIDUAL)
    return model


def load_best(config, method, checkpoint, opts):
    from swift.arguments import InferArguments
    from swift.pipelines.utils import prepare_model_template

    inference_keys = ('model', 'model_type', 'use_hf', 'attn_impl', 'torch_dtype', 'tuner_backend', 'load_args')
    args = InferArguments(**{k: opts[k] for k in inference_keys}, tuner_type='lora',
                          adapters=[] if method in ('pissa', 'lora-ga') else [str(Path(checkpoint).resolve())],
                          max_length=config['training']['max_length'], load_data_args=False)
    if method == 'randlora':
        args.tuner_type = 'randlora'
    model, template = prepare_model_template(args)
    if method == 'randlora':
        expected = json.loads((Path(checkpoint)/'randlora-basis.json').read_text())['sha256']
        if rand_basis_hash(model) != expected:
            raise ValueError('Reconstructed frozen RandLoRA basis differs from training')
    if method in ('pissa', 'lora-ga'):
        model = restore_residual_adapter(model, checkpoint)
    return model.eval(), template


def prepare_trainer(trainer, method, resume=None):
    if method == 'lora-plus':
        class VerifyRates(TrainerCallback):
            def on_train_begin(self, args, state, control, model=None, optimizer=None, **kwargs):
                ratio = model.peft_config['default'].lorap_lr_ratio
                # At resume a scheduler can overwrite the current lr. Validate scheduler bases.
                groups = [dict(g, lr=g.get('initial_lr', g['lr'])) for g in optimizer.param_groups]
                from types import SimpleNamespace
                rates = check_loraplus_optimizer(model, SimpleNamespace(param_groups=groups), args.learning_rate, ratio)
                from experiments.ms_swift_librispeech.train import save_json
                save_json(Path(args.output_dir) / 'loraplus-optimizer.json', rates)
        trainer.add_callback(VerifyRates())
    elif method == 'randlora':
        from experiments.ms_swift_librispeech.train import save_json
        basis = rand_basis_hash(trainer.model)
        if resume and json.loads((Path(resume)/'randlora-basis.json').read_text())['sha256'] != basis:
            raise ValueError('Reconstructed frozen RandLoRA basis differs from the resumed checkpoint')
        class BasisCheckpoint(TrainerCallback):
            def on_save(self, args, state, control, **kwargs):
                save_json(Path(args.output_dir) / f'checkpoint-{state.global_step}' / 'randlora-basis.json',
                          {'sha256': basis})
        trainer.add_callback(BasisCheckpoint())
    elif method in ('pissa', 'lora-ga'):
        # Link the frozen residual once per run. Each checkpoint remains independently copyable.
        path = Path(trainer.args.output_dir) / RESIDUAL
        if resume:
            source = Path(resume) / RESIDUAL
            _install_saved_residual(trainer.model, source)
            if path.exists() and not path.samefile(source) and _file_hash(path) != _file_hash(source):
                raise ValueError('Run residual and resumed checkpoint disagree')
        if not path.exists():
            save_residual(trainer.model, path)

        class ResidualCheckpoint(TrainerCallback):
            def on_save(self, args, state, control, **kwargs):
                target = Path(args.output_dir) / f'checkpoint-{state.global_step}' / RESIDUAL
                if target.exists() and not target.samefile(path):
                    raise FileExistsError(target)
                if not target.exists():
                    os.link(path, target)
                digest = target.with_suffix('.sha256')
                if not digest.exists():
                    os.link(path.with_suffix('.sha256'), digest)
        trainer.add_callback(ResidualCheckpoint())


def register_randlora(settings):
    """Use author-endorsed HF implementation in an isolated PEFT environment."""
    from peft import RandLoraConfig, get_peft_model
    from swift.tuner_plugin import tuners_map
    from swift.tuner_plugin.base import PeftTuner

    if set(settings) != {'r', 'randlora_alpha', 'projection_prng_key', 'save_projection'} or settings['save_projection'] is not False:
        raise ValueError('Use seeded shared RandLoRA bases and explicit basis rank, key and alpha')

    class SwiftRandLoRA(PeftTuner):
        @staticmethod
        def prepare_model(args, model):
            if not args.target_regex:
                raise ValueError('RandLoRA requires a decoder projection regex')
            return get_peft_model(model, RandLoraConfig(target_modules=args.target_regex, **settings))

    tuners_map['randlora'] = SwiftRandLoRA


def rand_basis_hash(model):
    import hashlib
    h = hashlib.sha256()
    for attr in ('randlora_A', 'randlora_B'):
        tensor = getattr(model.base_model, attr)['default'].detach().cpu().contiguous()
        h.update(str((attr, tensor.shape, tensor.dtype)).encode())
        h.update(tensor.view(torch.uint8).numpy())
    return h.hexdigest()


def audit_randlora(model, config):
    import hashlib
    from peft import RandLoraConfig
    from peft.tuners.randlora.layer import RandLoraLayer
    from experiments.ms_swift_librispeech.adapter_audit import MLP_PROJECTIONS, ordered_projections
    cfg = model.peft_config['default']
    settings = config['method_options']['randlora']
    if not isinstance(cfg, RandLoraConfig) or any(getattr(cfg, k) != v for k, v in settings.items()):
        raise ValueError('Unexpected native RandLoRA config')
    outer = model.get_base_model()
    n = config['adapter']['last_n']
    indices = list(range(len(outer.thinker.model.layers)-n, len(outer.thinker.model.layers)))
    projections = ordered_projections(config['adapter']['target_modules'])
    wanted = {f'thinker.model.layers.{i}.{"mlp" if p in MLP_PROJECTIONS else "self_attn"}.{p}'
              for i in indices for p in projections}
    adapted = {name: m for name, m in outer.named_modules() if isinstance(m, RandLoraLayer)}
    if wanted != adapted.keys():
        raise ValueError('RandLoRA target projections differ from the requested scope')
    expected = set()
    for m in adapted.values():
        d = min(m.in_features, m.out_features)
        count = math.ceil(d / cfg.r)
        if (m.randlora_lambda['default'].shape != (cfg.r, count) or m.randlora_gamma['default'].shape != (count, d)
                or m.scaling['default'] != cfg.randlora_alpha / cfg.r or list(m.active_adapters) != ['default']):
            raise ValueError('Incorrect native RandLoRA basis dimensions')
        expected.update((id(m.randlora_lambda['default']), id(m.randlora_gamma['default'])))
    trainable = {k: p for k, p in model.named_parameters() if p.requires_grad}
    if {id(p) for p in trainable.values()} != expected:
        raise ValueError('Unexpected RandLoRA trainable parameters')
    h = hashlib.sha256()
    for name, p in sorted(trainable.items()):
        h.update(name.encode())
        h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy())
    return dict(trainable_parameters=sum(p.numel() for p in trainable.values()), trainable_names=sorted(trainable),
                param_sha256=h.hexdigest(), basis_sha256=rand_basis_hash(model), target_layers=indices,
                target_projections=list(projections), target_module_count=len(adapted))
