"""Native Swift structural tuners and validation of their exact trainable scopes."""
import hashlib
from pathlib import Path
import sys

import torch


METHODS = {'adapter', 'llamapro', 'reft', 'longlora', 'lora_llm'}
NATIVE_OPTIONS = {
    'adapter': {'adapter_length', 'adapter_act'},
    'llamapro': {'llamapro_num_new_blocks', 'llamapro_num_groups'},
    'reft': {'reft_rank', 'reft_layer_key', 'reft_layers', 'reft_intervention_type'},
    'longlora': set(),
    'lora_llm': set(),
}


def options(method, config, base_options):
    """Build the full native Swift model options for both SFT and clean reload."""
    if method not in METHODS or base_options['model_type'] != 'qwen2_5_omni':
        raise ValueError(f'Unsupported structural method/backbone: {method}, {base_options["model_type"]}')
    if method == 'longlora':
        raise ValueError('LongLoRA is unsupported on qwen2_5_omni: swift/pipelines/train/tuner.py '
                         'requires LongLoRAModelType.LLAMA in args.model_type before replacing attention.')
    overrides = config.get('method_options', {}).get(method, {})
    if unexpected := set(overrides) - NATIVE_OPTIONS[method]:
        raise ValueError(f'Unknown {method} options: {sorted(unexpected)}')
    if method == 'reft' and (overrides.get('reft_intervention_type', 'LoreftIntervention') != 'LoreftIntervention'
                             or overrides.get('reft_layer_key') != 'thinker.model.layers'):
        raise ValueError('This AQA ReFT arm requires LoReFT on thinker.model.layers')
    opts = dict(base_options, tuner_type=method, **overrides)
    if method in {'adapter', 'llamapro', 'reft'}:
        plugin = str(Path(__file__).with_name('structural_plugin.py').resolve())
        opts['external_plugins'] = [*opts.get('external_plugins', []), plugin]
    if method == 'reft' and config.get('reft_dependency_path'):
        deps = str(Path(config['reft_dependency_path']).resolve(strict=True))
        if deps not in sys.path:
            sys.path.append(deps)
    # lora_llm determines its own language projection regex with get_multimodal_target_regex.
    opts.pop('target_regex', None)
    return opts


def _trainable_parts(model, method, settings, config):
    layers = model.thinker.model.layers
    if method == 'adapter':
        width = settings.get('adapter_length', 128)
        result = []
        for layer in layers:
            adapter = layer.mlp.adapter_default
            if adapter.linear1.out_features != width or adapter.linear2.in_features != width:
                raise ValueError('Unexpected bottleneck width')
            result.extend((adapter.linear1, adapter.linear2))
        return result
    if method == 'llamapro':
        new = settings.get('llamapro_num_new_blocks', 4)
        groups = settings.get('llamapro_num_groups') or new
        original = len(layers) - new
        if new != groups or original < 1 or original % groups or len(layers) != model.config.get_text_config().num_hidden_layers:
            raise ValueError('Unexpected thinker block expansion')
        if [layer.self_attn.layer_idx for layer in layers] != list(range(len(layers))):
            raise ValueError('Expanded thinker has inconsistent attention cache indices')
        stride = original // groups + 1
        return [layers[i] for i in range(stride - 1, len(layers), stride)]
    if method == 'reft':
        reft = model.base_model
        requested = settings.get('reft_layers') or list(range(len(layers)))
        positions = [r.layer for r in reft.representations.values()]
        if sorted(positions) != sorted(requested):
            raise ValueError(f'ReFT layer mismatch: {positions}')
        result = []
        for key, obj in reft.interventions.items():
            site = reft.representations[key]
            if (type(obj).__name__ != 'LoreftIntervention' or
                    site.component != f'thinker.model.layers[{site.layer}].output' or
                    obj.learned_source.out_features != settings.get('reft_rank', 4)):
                raise ValueError(f'Unexpected ReFT intervention at {site.component}')
            # Torch's Householder map casts these fixed signs to integers. AdamW decay
            # on the original coordinates can silently zero a full column of the rotation.
            signs = obj.rotate_layer.parametrizations.weight.original.detach().diagonal()
            if not signs.is_meta and not torch.equal(signs.abs(), torch.ones_like(signs)):
                raise ValueError(f'LoReFT lost its orthogonal rotation signs at {site.component}')
            result.extend((obj.learned_source, obj.rotate_layer))
        return result
    if method == 'lora_llm':
        # Native Swift unfreezes both modality towers and their nested aligners. The thinker head is frozen.
        result = [model.thinker.audio_tower, model.thinker.visual]
        for layer in layers:
            for target in config['adapter']['target_modules']:
                owner = layer.self_attn if hasattr(layer.self_attn, target) else layer.mlp
                projection = getattr(owner, target)
                result.extend((projection.lora_A['default'], projection.lora_B['default']))
        return result
    raise ValueError(f'No auditable structural update scope for {method}')


def audit(model, method, config):
    """Require the exact native parameter set and hash its names, shapes, dtypes and bytes."""
    settings = config.get('method_options', {}).get(method, {})
    parts = _trainable_parts(model, method, settings, config)
    expected = {id(p) for part in parts for p in part.parameters()}
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    actual = {id(p) for p in trainable.values()}
    if not expected or expected != actual:
        extra = [name for name, p in trainable.items() if id(p) not in expected]
        missing_ids = expected - actual
        missing = [name for name, p in model.named_parameters() if id(p) in missing_ids]
        raise ValueError(f'Unexpected {method} trainable scope, extra={extra[:5]}, missing={missing[:5]}')
    h = hashlib.sha256()
    components = {key: hashlib.sha256() for key in ('audio_tower', 'visual', 'language_lora')} if method == 'lora_llm' else {}
    for name, p in sorted(trainable.items()):
        header = f'{name}|{tuple(p.shape)}|{p.dtype}\0'.encode()
        payload = p.detach().cpu().contiguous().view(torch.uint8).numpy()
        h.update(header)
        h.update(payload)
        if components:
            tower = next((key for key in ('audio_tower', 'visual') if f'.thinker.{key}.' in name), 'language_lora')
            components[tower].update(header)
            components[tower].update(payload)
    result = {'trainable_parameters': sum(p.numel() for p in trainable.values()), 'param_sha256': h.hexdigest(),
              'trainable_tensors': len(trainable)}
    if components:
        result['component_sha256'] = {key: digest.hexdigest() for key, digest in components.items()}
    return result


def initialize(model, method, config):
    if method == 'llamapro':
        settings = config.get('method_options', {}).get(method, {})
        for layer in _trainable_parts(model, method, settings, config):
            if any(p.count_nonzero().item() for part in (layer.self_attn.o_proj, layer.mlp.down_proj)
                   for p in part.parameters()):
                raise ValueError('LLaMA-Pro output projections must start at zero')
    if method != 'lora_llm':
        return audit(model, method, config)
    from experiments.ms_swift_librispeech.adapter_audit import initialize_pair

    towers = [model.thinker.audio_tower, model.thinker.visual]
    if any(not p.requires_grad for tower in towers for p in tower.parameters()):
        raise ValueError('Native lora_llm must enable every modality tower parameter')
    try:
        for tower in towers:
            tower.requires_grad_(False)
        adapter = config['adapter']
        init = initialize_pair(model, 'lora', config['training']['seed'], adapter['last_n'], adapter['target_modules'])
    finally:
        for tower in towers:
            tower.requires_grad_(True)
    return dict(audit(model, method, config), shared_ab_sha256=init['ab_init_sha256'])
