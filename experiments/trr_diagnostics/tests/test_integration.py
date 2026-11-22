"""Integration regressions that do not require a model or a GPU."""
import json
import sys
import types
from pathlib import Path

import pytest
from scipy.stats import binomtest

from experiments.trr_diagnostics import d2, stats, twin
from experiments.trr_diagnostics.tests.test_d2 import CONFIG, corpus
from experiments.trr_diagnostics.tests.test_twin import SineVocoder


@pytest.fixture
def ready(corpus, monkeypatch):
    config, tmp = corpus
    out = tmp / 'prepared'
    d2.prepare(config, out, backend=SineVocoder())
    calls = []

    def score(model, template, manifest, generation, predictions):
        calls.append(Path(manifest).stem)
        rows = d2.read_jsonl(manifest)
        predictions.parent.mkdir(parents=True, exist_ok=True)
        asr = Path(manifest).stem.startswith('asr-')
        d2.write_jsonl(predictions, [dict(id=r['id'], reference=r['messages'][1]['content'],
                                        prediction=r['messages'][1]['content'],
                                        **({'word_errors': 0, 'reference_words': 3} if asr else {})) for r in rows])
        return {'wer': 0.0} if asr else {'uar': 1.0}

    model = types.SimpleNamespace(requires_grad_=lambda _: None)
    modules = {
        'torch': types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False)),
        'experiments.ms_swift_iemocap.evaluate': types.SimpleNamespace(score=score),
        'experiments.ms_swift_librispeech.evaluate': types.SimpleNamespace(score=score),
        'experiments.ms_swift_librispeech.train': types.SimpleNamespace(load_best=lambda *args: (model, None)),
        'experiments.ms_swift_aqa.evaluate_frozen': types.SimpleNamespace(load_frozen=lambda cfg: (model, None)),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    model_config = tmp / 'model.json'
    d2.write(model_config, {'model': str(tmp), 'model_revision': 'test'})
    return out, model_config, calls


@pytest.mark.parametrize('target', ['config', 'manifest', 'predictions', 'audio'])
def test_resume_rejects_modified_inputs_and_completed_predictions(ready, target):
    out, model_config, calls = ready
    d2.run(out, model_config, 'frozen-base', None, 'base')
    calls.clear()
    if target == 'config':
        path = out / 'config.json'
        cfg = d2.load(path)
        cfg['decision']['min_use_pp'] = 99
        d2.write(path, cfg)
    elif target == 'manifest':
        path = out / 'manifests/orig.jsonl'
        path.write_text(path.read_text() + '\n')
    elif target == 'predictions':
        path = out / 'runs/base/orig/predictions.jsonl'
        rows = d2.read_jsonl(path)
        rows[0]['prediction'] = 'D'
        d2.write_jsonl(path, rows)
    else:
        path = Path(d2.read_jsonl(out / 'manifests/orig.jsonl')[0]['audios'][0])
        path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='changed|hash|digest'):
        d2.run(out, model_config, 'frozen-base', None, 'base')
    assert calls == []


def test_resume_pins_adapter_configuration_as_well_as_weights(ready):
    out, model_config, calls = ready
    checkpoint = out.parent / 'checkpoint'
    checkpoint.mkdir()
    (checkpoint / 'adapter_model.safetensors').write_bytes(b'fake weights for identity test')
    d2.write(checkpoint / 'adapter_config.json', {'r': 8, 'lora_alpha': 16})
    d2.run(out, model_config, 'lora', checkpoint, 'adapter')
    d2.write(checkpoint / 'adapter_config.json', {'r': 8, 'lora_alpha': 64})
    with pytest.raises(ValueError, match='different model|changed'):
        d2.run(out, model_config, 'lora', checkpoint, 'adapter')


@pytest.mark.parametrize('target', ['config', 'predictions', 'asr_predictions'])
def test_report_rejects_changed_completed_artifacts(ready, target):
    out, model_config, _ = ready
    d2.run(out, model_config, 'frozen-base', None, 'base')
    if target == 'config':
        path = out / 'config.json'
        cfg = d2.load(path)
        cfg['decision']['min_use_pp'] = 99
        d2.write(path, cfg)
    else:
        condition = 'orig' if target == 'predictions' else 'asr-sham'
        path = out / 'runs/base' / condition / 'predictions.jsonl'
        rows = d2.read_jsonl(path)
        if target == 'predictions':
            rows[0]['prediction'] = 'D'
        else:
            rows = rows[:-1]
        d2.write_jsonl(path, rows)
    with pytest.raises(ValueError, match='changed|hash|digest|cover'):
        d2.report(out, 'base')


def test_report_does_not_treat_an_unfinished_prediction_file_as_complete(ready):
    out, model_config, _ = ready
    d2.run(out, model_config, 'frozen-base', None, 'base')
    (out / 'runs/base/flat_both/metrics.json').unlink()
    result, _ = d2.report(out, 'base')
    assert 'flat_both' not in result['conditions']
    assert result['verdicts']['f0_energy_use'] == 'not_interpretable'


@pytest.mark.parametrize('blocker', ['qc', 'missing_asr', 'smoke'])
def test_report_withholds_prosody_interpretation_without_valid_controls(ready, blocker):
    out, model_config, _ = ready
    info = d2.load(out / 'prepare.json')
    if blocker == 'qc':
        info['qc']['passed'] = False
    if blocker == 'smoke':
        info['smoke_limit'] = info['utterances']
    d2.write(out / 'prepare.json', info)
    d2.run(out, model_config, 'frozen-base', None, 'base', skip_asr=blocker == 'missing_asr')
    result, markdown = d2.report(out, 'base')
    assert result['verdicts']['f0_energy_use'] == 'not_interpretable'
    assert result['interpretation']['eligible'] is False
    assert result['interpretation']['reasons']
    assert 'NOT INTERPRETABLE' in markdown


def test_prepare_rejects_a_session_outside_the_declared_development_split(corpus):
    config, tmp = corpus
    rows = d2.read_jsonl(tmp / 'dev.jsonl')
    rows[0]['session'] = 5
    d2.write_jsonl(tmp / 'dev.jsonl', rows)
    with pytest.raises(ValueError, match='session|Session'):
        d2.prepare(config, tmp / 'bad', backend=SineVocoder())


def test_acoustic_qc_cannot_pass_when_peak_guard_changes_global_loudness():
    base = dict(f0_sd_semitones=4.0, energy_sd_db=4.0, rms=0.1)
    edited = {'sham': dict(base), 'flat_f0': dict(base, f0_sd_semitones=0.1),
              'flat_energy': dict(base, energy_sd_db=0.1),
              'flat_both': dict(base, f0_sd_semitones=0.1, energy_sd_db=0.1, rms=0.05)}
    record = dict(stats={'orig': base, **edited}, clipped_samples={}, peak_scale={'flat_both': 0.5})
    result = d2.summarize_qc([record], list(twin.AUDIO_CONDITIONS), CONFIG['decision']['qc'])
    assert not result['passed']


def test_mcnemar_handles_a_full_session_of_discordant_predictions():
    assert stats.mcnemar_exact(515, 516) == pytest.approx(binomtest(515, 1031, 0.5).pvalue)
