import json

import pytest
import torch
from safetensors.torch import save_file
from torch import nn


@pytest.fixture
def miniature(tmp_path):
    model = nn.Module()
    model.thinker = nn.Module()
    model.thinker.model = nn.Module()
    model.thinker.model.embed_tokens = nn.Embedding(17, 4)
    model.thinker.model.layers = nn.ModuleList([nn.Linear(4, 4, bias=False)])
    model.thinker.lm_head = nn.Linear(4, 17, bias=False)
    model.thinker.audio_tower = nn.Linear(4, 4).requires_grad_(False)
    model.thinker.visual = nn.Linear(4, 4).requires_grad_(False)
    model.bfloat16()
    weights = model.state_dict()
    files = ['model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors']
    assignment = {name: files[i % 2] for i, name in enumerate(weights)}
    for file in files:
        save_file({name: w for name, w in weights.items() if assignment[name] == file}, tmp_path / file)
    (tmp_path / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': assignment}))
    return model, {'model': str(tmp_path), 'model_type': 'qwen2_5_omni'}


def test_native_full_options_remove_adapter_settings(miniature):
    from experiments.ms_swift_aqa.full_suite import options
    _, config = miniature
    base = dict(config, torch_dtype='bfloat16', attn_impl='eager', device_map='cpu', use_hf=True,
                tuner_type='lora', target_regex='q_proj', lora_rank=8, use_dora=True, external_plugins=['test.py'],
                adapters=['earlier-adapter'], model_kwargs={'sampling_rate': 16000})
    result = options('full', config, base)
    assert result['model'] == base['model'] and result['device_map'] == 'cpu'
    assert result['tuner_type'] == 'full' and result['external_plugins'] == []
    assert result['adapters'] == [] and result['model_kwargs'] == {'sampling_rate': 16000, 'enable_audio_output': 0}
    assert (result['freeze_llm'], result['freeze_vit'], result['freeze_aligner']) == (False, True, True)
    assert not {'target_regex', 'lora_rank', 'use_dora'} & result.keys()
    assert base['tuner_type'] == 'lora'


def test_audit_covers_every_bit_of_each_trainable_tensor(miniature):
    from experiments.ms_swift_aqa.full_suite import audit
    model, config = miniature
    initial = audit(model, 'full', config)
    assert initial['trainable_parameters'] == 152
    assert set(initial['trainable_names']) == {
        'thinker.model.embed_tokens.weight', 'thinker.model.layers.0.weight', 'thinker.lm_head.weight'}
    with torch.no_grad():
        model.thinker.visual.weight[0, 0].add_(1)
    assert audit(model, 'full', config)['param_sha256'] == initial['param_sha256']
    with torch.no_grad():
        bits = model.thinker.model.embed_tokens.weight.view(torch.int16)
        bits[11, 2].bitwise_xor_(1)
    assert audit(model, 'full', config)['param_sha256'] != initial['param_sha256']


@pytest.mark.parametrize('module,trainable', [('audio_tower', True), ('lm_head', False)])
def test_audit_rejects_changed_trainable_scope(miniature, module, trainable):
    from experiments.ms_swift_aqa.full_suite import audit
    model, config = miniature
    getattr(model.thinker, module).requires_grad_(trainable)
    with pytest.raises(ValueError, match=module):
        audit(model, 'full', config)


def test_reload_rejects_partial_full_weights_before_initializing_model(miniature, tmp_path):
    from experiments.ms_swift_aqa.full_suite import load_best
    model, config = miniature
    checkpoint = tmp_path / 'checkpoint-1'
    checkpoint.mkdir()
    weights = model.state_dict()
    del weights['thinker.lm_head.weight']
    save_file(weights, checkpoint / 'model.safetensors')
    with pytest.raises(ValueError, match='lm_head'):
        load_best(config, checkpoint, config)
