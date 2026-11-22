"""Dispatch the additional Swift AQA tuners and identify every inference weight in a checkpoint."""
from hashlib import sha256
import json


PEFT_METHODS = {'adalora', 'vera', 'boft', 'fourierft', 'bone', 'ia3'}
STRUCTURAL_METHODS = {'adapter', 'llamapro', 'reft', 'longlora', 'lora_llm'}
EXTRA = PEFT_METHODS | STRUCTURAL_METHODS | {'full'}


def implementation(method):
    if method == 'rolora-federated':
        from . import paper_federated as impl
    elif method == 'longlora-omni':
        from . import paper_longlora as impl
    elif method == 'loram-train-small':
        from . import paper_early_loram as impl
    elif method == 'eva':
        from . import paper_eva as impl
    elif method in ('lora-plus', 'pissa', 'lora-ga', 'randlora'):
        from . import paper_early as impl
    elif method in ('lowra', 'gralora', 'toplora', 'magnitude-loram', 'aurora', 'flylora', 'bora', 'lokra',
                    'stable-lora', 'loft', 'ralora', 'unilora', 'alora', 'igu-lora', 'lora-mixer'):
        from . import paper_plugin as impl
    elif method in PEFT_METHODS:
        from . import peft_suite as impl
    elif method in STRUCTURAL_METHODS:
        from . import structural_suite as impl
    elif method == 'full':
        from . import full_suite as impl
    else:
        raise ValueError(f'Unknown AQA tuner: {method}')
    return impl


def checkpoint_files(checkpoint, method):
    if method == 'full':
        config_files = [checkpoint / 'config.json']
        index = checkpoint / 'model.safetensors.index.json'
        if index.is_file():
            config_files.append(index)
            names = sorted(set(json.loads(index.read_text())['weight_map'].values()))
            if not names or any('/' in name or not name.endswith('.safetensors') for name in names):
                raise ValueError(f'Invalid weight index in {checkpoint}')
            weights = [checkpoint / name for name in names]
        else:
            weights = [checkpoint / 'model.safetensors']
    else:
        config_files = sorted(checkpoint.rglob('adapter_config.json'))
        if not config_files:
            raise FileNotFoundError(f'Missing adapter config in {checkpoint}')
        weights = []
        for cfg in config_files:
            files = sorted(cfg.parent.glob('*.safetensors'))
            if not files:
                raise FileNotFoundError(f'Missing adapter weights alongside {cfg}')
            weights.extend(files)
            if method == 'reft':
                layers = json.loads(cfg.read_text())['layers']
                interventions = sorted(cfg.parent.glob('intkey_*.bin'))
                if (len(interventions) != len(layers) or not layers or
                        cfg.parent/'rotation_state.safetensors' not in files):
                    raise FileNotFoundError(f'Incomplete ReFT intervention or rotation state in {cfg.parent}')
                weights.extend(interventions)
        if method == 'lora_llm' and checkpoint / 'vit.safetensors' not in weights:
            raise FileNotFoundError(f'Missing jointly trained towers in {checkpoint}')
    for path in [*config_files, *weights]:
        if not path.is_file() or not path.stat().st_size:
            raise FileNotFoundError(path)
    return weights, config_files


def checkpoint_identity(checkpoint, method):
    from experiments.ms_swift_librispeech.train import digest

    def group(files):
        hashes = {str(path.relative_to(checkpoint)): digest(path) for path in files}
        if len(hashes) == 1:
            return next(iter(hashes.values())), hashes
        return sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(), hashes

    weights, configs = checkpoint_files(checkpoint, method)
    weight_digest, weight_map = group(weights)
    config_digest, config_map = group(configs)
    return dict(adapter_sha256=weight_digest, config_sha256=config_digest,
                files={**weight_map, **config_map})
