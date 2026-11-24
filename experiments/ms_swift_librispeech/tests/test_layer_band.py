"""Adapters on a run of consecutive decoder layers other than the last ones.

    python -m pytest -q experiments/ms_swift_librispeech/tests/test_layer_band.py
"""
import copy
import json
import re
from types import SimpleNamespace

import pytest
import torch

from test_adapter_pair import tiny_omni
from test_conditional_gate import inputs
from experiments.ms_swift_librispeech.adapter_audit import audit_adapter, band, initialize_pair

PROMPT = [[1, 5, 5, 6, 3, 8, 2]]


def targets(indices):
    return rf'thinker\.model\.layers\.[{indices}]\.self_attn\.o_proj'


def build(indices, gate_multiplier=None):
    from experiments.ms_swift_librispeech.plugin import TALoRATuner
    from experiments.ms_swift_librispeech.swift_asr_ta_runtime import set_gate_multiplier

    torch.manual_seed(3)
    base = tiny_omni().eval()
    base.requires_grad_(False)
    model = copy.deepcopy(base)
    TALoRATuner.prepare_model(SimpleNamespace(target_modules=targets(indices), lora_rank=2, lora_alpha=4), model)
    if gate_multiplier is not None:
        set_gate_multiplier(model, gate_multiplier)
    return base, model.eval()


def test_band_is_the_last_layers_unless_a_first_layer_is_given():
    assert list(band(36, 4)) == [32, 33, 34, 35]
    assert list(band(36, 4, 0)) == [0, 1, 2, 3] and list(band(36, 4, 16)) == [16, 17, 18, 19]
    assert list(band(36, 4, 32)) == list(band(36, 4)) and list(band(36, 36)) == list(range(36))
    for last_n, first in ((0, None), (37, None), (4, 33), (4, -1), (4, 1.0), (True, None)):
        with pytest.raises(ValueError):
            band(36, last_n, first)


@pytest.mark.parametrize('indices, expected', [('01', [0, 1]), ('12', [1, 2]), ('34', [3, 4])])
def test_adapters_are_installed_and_trainable_only_on_the_chosen_layers(indices, expected):
    _, model = build(indices)
    layers = model.thinker.model.layers
    assert [i for i, layer in enumerate(layers) if hasattr(layer.self_attn, 'ta_adapter')] == expected
    audit = audit_adapter(model, 'ta_lora', 2, first_layer=expected[0])
    assert audit['target_layers'] == expected
    assert all(re.search(rf'layers\.({expected[0]}|{expected[1]})\.', name) for name in audit['trainable_names'])
    assert model._ta_lora_config['targets'] == [f'thinker.model.layers.{i}.self_attn.o_proj' for i in expected]
    if expected != [3, 4]:
        with pytest.raises(Exception):            # the default still means the last layers
            audit_adapter(model, 'ta_lora', 2)


def test_first_layer_at_the_end_equals_the_default():
    _, model = build('34')
    assert audit_adapter(model, 'ta_lora', 2, first_layer=3) == audit_adapter(model, 'ta_lora', 2)


def test_layers_that_are_not_consecutive_are_refused():
    from experiments.ms_swift_librispeech.plugin import TALoRATuner
    with pytest.raises(ValueError, match='consecutive'):
        TALoRATuner.prepare_model(SimpleNamespace(target_modules=targets('13'), lora_rank=2, lora_alpha=4), tiny_omni())


def test_an_early_band_changes_the_output_only_once_it_is_trained_and_survives_a_reload(tmp_path):
    from experiments.ms_swift_librispeech.plugin import TALoRATuner

    base, model = build('01', gate_multiplier=0.)
    with torch.no_grad():
        plain = base(**inputs(PROMPT)).logits
        assert torch.equal(model(**inputs(PROMPT)).logits, plain)      # up is zero at the start
        initialize_pair(model, 'ta_lora', 5, 2, first_layer=0)
        for layer in model.thinker.model.layers[:2]:
            layer.self_attn.ta_adapter.up.normal_(std=.4)
        trained = model(**inputs(PROMPT)).logits
    assert not torch.equal(trained, plain)
    TALoRATuner.save_pretrained(model, tmp_path)
    assert json.loads((tmp_path / 'adapter_config.json').read_text())['targets'][0] == 'thinker.model.layers.0.self_attn.o_proj'
    restored = copy.deepcopy(base)
    TALoRATuner.from_pretrained(restored, tmp_path)
    assert [i for i, layer in enumerate(restored.thinker.model.layers) if hasattr(layer.self_attn, 'ta_adapter')] == [0, 1]
    with torch.no_grad():
        assert torch.equal(restored.eval()(**inputs(PROMPT)).logits, trained)


def test_training_options_select_the_band_from_the_config(tmp_path):
    from experiments.ms_swift_librispeech.train import digest, model_options

    tiny = tiny_omni()
    tiny.config.to_json_file(tmp_path / 'config.json')
    manifest = tmp_path / 'snapshot.json'
    manifest.write_text(json.dumps({'revision': 'mini', 'files': {'config.json': digest(tmp_path / 'config.json')}}))
    cfg = dict(model=str(tmp_path), snapshot_manifest=str(manifest), model_revision='mini', adapter=dict(rank=2, alpha=4, last_n=2))

    def chosen(**adapter):
        regex = model_options(dict(cfg, adapter=dict(cfg['adapter'], **adapter)), 'ta-lora')['target_regex']
        return sorted(int(re.search(r'layers\.(\d+)', n)[1]) for n, _ in tiny.named_modules() if re.fullmatch(regex, n))

    assert chosen() == [3, 4] and chosen(first_layer=0) == [0, 1] and chosen(first_layer=2) == [2, 3]
    with pytest.raises(ValueError):
        chosen(first_layer=4)
