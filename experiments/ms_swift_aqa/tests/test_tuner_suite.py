import json
from pathlib import Path

import pytest
import torch

from experiments.ms_swift_aqa.run_tuners import matched_trajectory, resolved
from experiments.ms_swift_aqa.tuner_suite import checkpoint_identity
from experiments.ms_swift_librispeech.train import InputTrajectory


def test_bundle_identity_includes_tower_and_all_full_shards(tmp_path):
    (tmp_path/'adapter_config.json').write_text('{}')
    (tmp_path/'adapter_model.safetensors').write_bytes(b'A')
    with pytest.raises(FileNotFoundError, match='towers'):
        checkpoint_identity(tmp_path, 'lora_llm')
    vit = tmp_path/'vit.safetensors'
    vit.write_bytes(b'first')
    first = checkpoint_identity(tmp_path, 'lora_llm')
    vit.write_bytes(b'second')
    assert checkpoint_identity(tmp_path, 'lora_llm')['adapter_sha256'] != first['adapter_sha256']
    index = tmp_path/'model.safetensors.index.json'
    (tmp_path/'config.json').write_text('{}')
    index.write_text(json.dumps({'weight_map': {'first': 'a.safetensors', 'second': 'b.safetensors'}}))
    (tmp_path/'a.safetensors').write_bytes(b'weights')
    with pytest.raises(FileNotFoundError):
        checkpoint_identity(tmp_path, 'full')
    (tmp_path/'b.safetensors').write_bytes(b'weights2')
    full = checkpoint_identity(tmp_path, 'full')
    assert set(full['files']) == {'config.json', index.name, 'a.safetensors', 'b.safetensors'}


def test_first_examples_are_invariant_to_batch_size_and_padding():
    def batch(items):
        def pad(sequences):
            return torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        ids = pad([torch.arange(1, n + 1) for n in items])
        frames = pad([torch.arange(7, n+7) for n in items])
        return dict(input_ids=ids, labels=ids.masked_fill(ids <= 1, -100), attention_mask=ids.ne(0),
                    feature_attention_mask=frames.ne(0), input_features=frames[:, None, :].bfloat16())
    a = InputTrajectory(record_examples=True)
    b = InputTrajectory(record_examples=True)
    a.record(batch([3, 2]))
    b.record(batch([3]))
    b.record(batch([2]))
    assert a.examples == b.examples
    assert a.batches != b.batches
    assert len(set(a.examples)) == 2


def test_reft_identity_checks_every_intervention(tmp_path):
    component = tmp_path/'reft'
    component.mkdir()
    (component/'adapter_config.json').write_text('{"layers": [0, 1]}')
    (component/'rotation_state.safetensors').write_bytes(b'coordinates')
    (component/'intkey_first.bin').write_bytes(b'first')
    with pytest.raises(FileNotFoundError, match='Incomplete ReFT'):
        checkpoint_identity(tmp_path, 'reft')
    last = component/'intkey_second.bin'
    last.write_bytes(b'second')
    before = checkpoint_identity(tmp_path, 'reft')
    last.write_bytes(b'changed')
    assert checkpoint_identity(tmp_path, 'reft')['adapter_sha256'] != before['adapter_sha256']


def test_protocol_and_shared_initialization(tmp_path):
    root = Path(__file__).resolve().parents[1]
    plan = json.loads((root/'omni3b-tuners.json').read_text())
    base = json.loads((root/plan['base_config']).read_text())
    for method in plan['methods']:
        for overrides in [{}, *plan['on_training_cuda_oom']]:
            config = resolved(plan, base, method, overrides)
            assert config['methods'] == [method]
            assert config['generation'] == base['generation']
    anchor = {'init_sha256': 'baseline', 'input_trajectory': list(range(8))}
    assert matched_trajectory(dict(**anchor, method='rslora'), anchor, base, tmp_path/'trajectory') is None
    with pytest.raises(ValueError, match='identical A/B'):
        matched_trajectory({**anchor, 'init_sha256': 'wrong', 'method': 'rslora'}, anchor, base, tmp_path/'trajectory')


def test_only_append_new_methods_to_a_pinned_queue(tmp_path):
    from experiments.ms_swift_aqa.run_tuners import pin_plan
    old = dict(methods=['vera'], method_options={'vera': {'rank': 256}}, training_overrides={}, data='frozen')
    new = {**old, 'methods': ['vera', 'ia3'], 'method_options': {**old['method_options'], 'ia3': {}}}
    pin_plan(tmp_path, old)
    with pytest.raises(ValueError):
        pin_plan(tmp_path, new)
    with pytest.raises(ValueError):
        pin_plan(tmp_path, {**new, 'data': 'other'}, extend=True)
    with pytest.raises(ValueError):
        pin_plan(tmp_path, {**new, 'method_options': {'vera': {'rank': 8}}}, extend=True)
    with pytest.raises(ValueError):
        pin_plan(tmp_path, {**new, 'methods': ['ia3', 'vera']}, extend=True)
    pin_plan(tmp_path, new, extend=True)
    assert json.loads((tmp_path/'plan.json').read_text()) == new
    assert [json.loads(p.read_text()) for p in (tmp_path/'plans').iterdir()] == [old]
