"""Native ms-swift full tuning of the Qwen2.5-Omni Thinker language model."""
import hashlib
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open


LANGUAGE_MODEL = ('thinker.model.', 'thinker.lm_head.')
FREEZE_OPTIONS = dict(freeze_llm=False, freeze_vit=True, freeze_aligner=True)
HASH_CHUNK_BYTES = 8 * 1024 * 1024


def options(method, config, base_options):
    """Return SftArguments model options, preserving the caller's verified base and device."""
    if method != 'full' or config.get('model_type') != 'qwen2_5_omni':
        raise ValueError('Full Thinker tuning requires full and qwen2_5_omni')
    if Path(base_options['model']).resolve() != Path(config['model']).resolve():
        raise ValueError('Base options and configured model path disagree')
    adapter_fields = {'target_regex', 'target_modules', 'use_dora', 'use_rslora', 'modules_to_save'}
    result = {key: value for key, value in base_options.items()
              if key not in adapter_fields and not key.startswith('lora_')}
    model_kwargs = result.get('model_kwargs') or {}
    if isinstance(model_kwargs, str):
        model_kwargs = json.loads(model_kwargs)
    result.update(tuner_type='full', load_args=False, adapters=[], external_plugins=[],
                  model_kwargs={**model_kwargs, 'enable_audio_output': 0}, **FREEZE_OPTIONS)
    return result


def _weight_shapes(directory, prefixes):
    """Read safetensors headers and confirm the index maps each tensor to its actual shard."""
    directory = Path(directory)
    index = directory / 'model.safetensors.index.json'
    weight_map = json.loads(index.read_text())['weight_map'] if index.is_file() else None
    files = sorted(set(weight_map.values())) if weight_map is not None else ['model.safetensors']
    locations, shapes = {}, {}
    for filename in files:
        if Path(filename).name != filename:
            raise ValueError(f'Expected a local shard in {index}, got {filename!r}')
        with safe_open(directory / filename, framework='pt', device='cpu') as shard:
            for name in shard.keys():
                if name in locations:
                    raise ValueError(f'Duplicate weight {name} in {directory}')
                locations[name] = filename
                if name.startswith(prefixes):
                    shapes[name] = tuple(shard.get_slice(name).get_shape())
    if weight_map is not None and locations != weight_map:
        raise ValueError(f'Tensor keys or shard assignments disagree with {index}')
    if not shapes:
        raise ValueError(f'No {prefixes} weights in {directory}')
    return shapes


def _verify_shapes(expected, actual, description):
    different = sorted(n for n in expected.keys() | actual.keys() if expected.get(n) != actual.get(n))
    if different:
        raise ValueError(f'{description} differs from the verified base at {different[:8]}')


@torch.no_grad()
def audit(model, method, config):
    """Check the exact trainable names and shapes, then SHA-256 every trainable byte in bounded chunks."""
    if method != 'full' or config.get('model_type') != 'qwen2_5_omni':
        raise ValueError('Expected a full Qwen2.5-Omni model')
    expected = _weight_shapes(config['model'], LANGUAGE_MODEL)
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    _verify_shapes(expected, {name: tuple(p.shape) for name, p in trainable.items()}, 'Full trainable scope')
    digest = hashlib.sha256()
    for name, param in sorted(trainable.items()):
        digest.update(json.dumps([name, list(param.shape), str(param.dtype)]).encode() + b'\0')
        row_bytes = param.element_size() * math.prod(param.shape[1:])
        for part in param.detach().split(max(1, HASH_CHUNK_BYTES // row_bytes)):
            digest.update(part.cpu().contiguous().view(torch.uint8).numpy())
    return dict(trainable_parameters=sum(p.numel() for p in trainable.values()),
                trainable_names=sorted(trainable), trainable_dtypes=sorted({str(p.dtype) for p in trainable.values()}),
                target_layers=sorted({int(n.split('.')[3]) for n in trainable if n.startswith('thinker.model.layers.')}),
                param_sha256=digest.hexdigest())


def load_best(config, checkpoint, base_options):
    """Load the native full HF checkpoint and construct its input template from the original pinned processor.

    ms-swift saves a standalone Thinker with its frozen audio and visual towers, HF configuration,
    processor assets and spk_dict.pt. Both indexed shards and a single model.safetensors are supported.
    """
    checkpoint = Path(checkpoint).resolve(strict=True)
    expected = _weight_shapes(config['model'], 'thinker.')
    _verify_shapes(expected, _weight_shapes(checkpoint, 'thinker.'), 'Full checkpoint')
    from swift.arguments import InferArguments
    from swift.template import update_generation_config_eos_token

    kwargs = options('full', config, base_options)
    for key in FREEZE_OPTIONS:
        kwargs.pop(key)
    kwargs.update(model=str(checkpoint), load_data_args=False, max_length=config['training']['max_length'])
    args = InferArguments(**kwargs)
    model, _ = args.get_model_processor()
    _, processor = args.get_model_processor(model=config['model'], load_model=False)
    template = args.get_template(processor)
    if template.use_model:
        template.model = model
    update_generation_config_eos_token(model.generation_config, template)
    return model.eval(), template
