import json
import sys
import types
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

from experiments.trr_diagnostics import d2, twin
from experiments.trr_diagnostics.tests.test_twin import RATE, SineVocoder

HERE = Path(d2.__file__).parent
CONFIG = json.loads((HERE / 'd2-iemocap-session4.json').read_text())
QUESTION = CONFIG['prompts']['question']
LINES = {  # id suffix -> (label, transcript line)
    'impro01_F000': ('A', 'I told you already.'),
    'impro01_M000': ('D', 'I told you already.'),          # same words, different label
    'impro01_F001': ('B', '[LAUGHTER] That is great!'),    # annotation must be removed
    'impro01_M001': ('C', 'Okay.'),
    'impro02_F000': ('A', 'Stop it.'),
    'impro02_M000': ('B', 'We won the game.'),
    'impro02_F001': ('C', 'She is gone.'),
    'impro02_M001': ('D', 'It is Tuesday.'),
    'script01_1_F000': ('A', 'Why did you do that?'),
    'script01_1_M000': ('B', 'What a day!'),
    'script01_1_F001': ('C', 'I miss him.'),
    'script01_1_M001': ('D', 'The bus leaves at nine.'),
    'script01_1_F002': ('D', '[BREATHING]'),                # no lexical content -> excluded
    'script01_1_M002': ('C', None),                         # no transcript -> excluded
}


@pytest.fixture
def corpus(tmp_path):
    rows, by_dialogue = [], {}
    for index, (suffix, (label, text)) in enumerate(LINES.items()):
        uid = 'Ses04F_' + suffix
        wav = tmp_path / 'wav' / f'{uid}.wav'
        wav.parent.mkdir(exist_ok=True)
        twin.write_wav(wav, twin.synthetic_utterance(RATE, seconds=1.0 + 0.05 * index, harmonics=1), RATE)
        rows.append(dict(id=uid, speaker='Ses04' + suffix.rsplit('_', 1)[1][0], session=4, audios=[str(wav)],
                         messages=[dict(role='user', content='<audio>' + QUESTION), dict(role='assistant', content=label)]))
        if text is not None:
            by_dialogue.setdefault(uid.rsplit('_', 1)[0], []).append(f'{uid} [000.0000-001.0000]: {text}')
    manifest = tmp_path / 'dev.jsonl'
    d2.write_jsonl(manifest, rows)
    archive = tmp_path / 'release.zip'
    with ZipFile(archive, 'w') as release:
        for dialogue, lines in by_dialogue.items():
            release.writestr(f'IEMOCAP_full_release/Session4/dialog/transcriptions/{dialogue}.txt', '\n'.join(lines) + '\n')
        release.writestr('IEMOCAP_full_release/Session5/dialog/transcriptions/Ses05F_impro01.txt',
                         'Ses04F_impro01_F000 [0-1]: wrong session must be ignored\n')
    cfg = json.loads(json.dumps(CONFIG))
    cfg['data'].update(manifest=str(manifest), archive=str(archive))
    cfg['asr']['subset'] = 5
    cfg['bootstrap']['resamples'] = 300
    config = tmp_path / 'config.json'
    config.write_text(json.dumps(cfg))
    return config, tmp_path


def test_shipped_config_matches_the_supervised_iemocap_prompt_and_known_twins():
    iemocap = json.loads((HERE.parent / 'ms_swift_iemocap' / 'config.json').read_text())
    librispeech = json.loads((HERE.parent / 'ms_swift_librispeech' / 'config.json').read_text())
    assert QUESTION == iemocap['prompt'] and CONFIG['prompts']['asr'] == librispeech['prompt']
    assert CONFIG['classes'] == sorted(set(iemocap['labels'].values()))
    assert CONFIG['data']['sessions'] == iemocap['sessions']['dev']       # development session only
    assert CONFIG['conditions']['audio'][0] == 'orig'
    assert set(CONFIG['conditions']['audio'][1:]) == set(twin.AUDIO_CONDITIONS)
    assert set(CONFIG['twin']) == set(twin.DEFAULTS)
    named = {name for pair in CONFIG['contrasts'] + CONFIG['strata_contrasts'] for name in pair}
    assert named <= set(CONFIG['conditions']['audio']) | set(CONFIG['conditions']['text'])
    assert CONFIG['decision']['equivalence_pp'] <= CONFIG['decision']['min_use_pp']


def test_clean_transcript_removes_non_lexical_annotations():
    assert d2.clean_transcript('[LAUGHTER]  Oh,  no. [BREATHING]') == ('Oh, no.', True)
    assert d2.clean_transcript('Oh, no.') == ('Oh, no.', False)
    assert d2.lexical_key("I TOLD you -- don't!") == "i told you don't"


def test_prepare_builds_aligned_conditions(corpus):
    config, tmp = corpus
    out = tmp / 'd2'
    info = d2.prepare(config, out, backend=SineVocoder())
    assert info['utterances'] == 12 and info['excluded'] == {'no_lexical_content': 1, 'missing_transcript': 1}
    assert info['class_counts'] == {'A': 3, 'B': 3, 'C': 3, 'D': 3} and info['dialogues'] == 3
    assert info['same_text_different_label'] == 2 and info['transcripts_with_annotations_removed'] == 1
    assert info['smoke_limit'] is None and info['qc']['passed'], info['qc']['checks']

    manifests = {p.stem: d2.read_jsonl(p) for p in (out / 'manifests').glob('*.jsonl')}
    assert set(manifests) == {'orig', 'sham', 'flat_f0', 'flat_energy', 'flat_both', 'text', 'question_only',
                              'asr-orig', 'asr-sham', 'asr-flat_both'}
    ids = [row['id'] for row in manifests['orig']]
    for name in ('sham', 'flat_f0', 'flat_energy', 'flat_both', 'text', 'question_only'):
        assert [row['id'] for row in manifests[name]] == ids
        assert [row['messages'][1]['content'] for row in manifests[name]] == [r['reference'] for r in manifests['orig']]
    original = {row['id']: row['audios'][0] for row in d2.read_jsonl(tmp / 'dev.jsonl')}
    for row in manifests['orig']:
        assert row['audios'] == [original[row['id']]] and row['messages'][0]['content'] == '<audio>' + QUESTION
    for row in manifests['flat_both']:
        path = Path(row['audios'][0])
        assert path.parent == out / 'audio' / 'flat_both' and row['messages'][0]['content'] == '<audio>' + QUESTION
        flat, rate = twin.read_wav(path)
        source, _ = twin.read_wav(original[row['id']])
        assert rate == RATE and flat.size == source.size                       # duration unchanged
        assert twin.rms(flat) == pytest.approx(twin.rms(source), rel=0.02)     # level unchanged
    laughing = next(row for row in manifests['text'] if row['id'].endswith('impro01_F001'))
    assert laughing['audios'] == [] and '<audio>' not in laughing['messages'][0]['content']
    assert laughing['messages'][0]['content'] == f'The speaker said: "That is great!"\n{QUESTION}'
    assert all(row['messages'][0]['content'] == QUESTION and row['audios'] == [] for row in manifests['question_only'])
    asr_ids = [row['id'] for row in manifests['asr-orig']]
    assert len(asr_ids) == 5 and asr_ids == [row['id'] for row in manifests['asr-flat_both']]
    assert all(row['messages'][0]['content'] == '<audio>' + CONFIG['prompts']['asr'] and
               row['messages'][1]['content'] == row['transcript'] for row in manifests['asr-sham'])
    assert {row['id'][7:] for row in manifests['orig'] if row['same_text_different_label']} == {'impro01_F000', 'impro01_M000'}
    assert info['manifests']['text'] == d2.digest(out / 'manifests' / 'text.jsonl')
    with pytest.raises(FileExistsError):
        d2.prepare(config, out, backend=SineVocoder())


def test_prepare_rejects_a_manifest_with_another_question(corpus):
    config, tmp = corpus
    rows = d2.read_jsonl(tmp / 'dev.jsonl')
    rows[0]['messages'][0]['content'] = '<audio>How does the speaker feel?'
    d2.write_jsonl(tmp / 'dev.jsonl', rows)
    with pytest.raises(ValueError):
        d2.prepare(config, tmp / 'bad', backend=SineVocoder())
    assert not (tmp / 'bad').exists()


def test_smoke_limit_is_class_balanced_and_flagged(corpus):
    config, tmp = corpus
    info = d2.prepare(config, tmp / 'smoke', limit=4, backend=SineVocoder())
    assert info['utterances'] == 4 and info['utterances_before_limit'] == 12 and info['smoke_limit'] == 4
    assert info['class_counts'] == {'A': 1, 'B': 1, 'C': 1, 'D': 1}


def _write_predictions(folder, name, manifest, predictions):
    (folder / name).mkdir(parents=True)
    d2.write_jsonl(folder / name / 'predictions.jsonl',
                   [dict(id=m['id'], reference=m['reference'], prediction=p) for m, p in zip(manifest, predictions, strict=True)])


def test_report_contrasts_strata_and_verdicts(corpus):
    config, tmp = corpus
    out = tmp / 'd2'
    d2.prepare(config, out, backend=SineVocoder())
    manifest = d2.read_jsonl(out / 'manifests' / 'orig.jsonl')
    gold = [m['reference'] for m in manifest]
    wrong = lambda label: 'ABCD'[('ABCD'.index(label) + 1) % 4]
    folder = out / 'runs' / 'toy'
    folder.mkdir(parents=True)
    d2.write(folder / 'identity.json', dict(arm='toy'))
    _write_predictions(folder, 'orig', manifest, gold)
    _write_predictions(folder, 'sham', manifest, gold)
    flat = [wrong(y) if i % 2 == 0 else y for i, y in enumerate(gold)]  # flattening breaks half
    _write_predictions(folder, 'flat_both', manifest, flat)
    # The transcript fails on every improvised line and on one scripted line.
    first_scripted = next(m['id'] for m in manifest if m['scripted'])
    text = [y if m['scripted'] and m['id'] != first_scripted else wrong(y) for m, y in zip(manifest, gold)]
    _write_predictions(folder, 'text', manifest, text)
    (folder / 'question_only').mkdir()
    d2.write(folder / 'question_only' / 'failure.json', dict(error='boom'))
    for name, errors in (('asr-sham', 1), ('asr-flat_both', 3)):
        (folder / name).mkdir()
        rows = d2.read_jsonl(out / 'manifests' / f'{name}.jsonl')
        d2.write_jsonl(folder / name / 'predictions.jsonl',
                       [dict(id=r['id'], word_errors=errors if i == 0 else 0, reference_words=4) for i, r in enumerate(rows)])

    result, markdown = d2.report(out, 'toy')
    assert set(result['conditions']) == {'orig', 'sham', 'flat_both', 'text'}
    assert result['conditions']['orig']['uar'] == 1.0 and result['conditions']['flat_both']['accuracy'] == 0.5
    primary = result['contrasts']['flat_both-vs-sham']
    assert primary['delta_accuracy_pp'] == pytest.approx(-50) and primary['reference_only_correct'] == 6
    assert result['contrasts']['sham-vs-orig']['delta_uar_pp'] == 0
    assert 'flat_f0-vs-sham' in result['skipped_contrasts'] and 'text-vs-question_only' in result['skipped_contrasts']
    assert result['failed_conditions'] == ['question_only']
    strata = result['strata']['orig-vs-text']
    assert strata['scripted']['delta_accuracy_pp'] == 25 and strata['improvised']['delta_accuracy_pp'] == 100
    assert strata['scripted']['clusters'] == 1 and strata['scripted']['accuracy_interval_95_pp'] is None
    assert strata['improvised']['clusters'] == 2 and strata['improvised']['accuracy_interval_95_pp'] == [100.0, 100.0]
    assert strata['same_text_different_label']['utterances'] == 2
    assert set(result['strata']['flat_both-vs-sham']) >= {'transcript_alone_correct', 'transcript_alone_wrong',
                                                          'speaker:Ses04F', 'speaker:Ses04M'}
    assert result['asr_wer'] == {'asr-sham': pytest.approx(1 / 20), 'asr-flat_both': pytest.approx(3 / 20)}
    assert result['wer_contrasts']['asr-flat_both-vs-asr-sham']['delta_wer_pp'] == pytest.approx(10)
    assert result['verdicts'] == dict(vocoder_artifacts='pass', f0_energy_use='uses',
                                      audio_beyond_transcript='yes', flattening_keeps_words='fail')
    assert '| flat_both-vs-sham |' in markdown and 'SMOKE' not in markdown and 'failed to run: question_only' in markdown
    assert json.loads((folder / 'report.json').read_text())['verdicts'] == result['verdicts']

    _write_predictions(out / 'runs' / 'short', 'orig', manifest[:-1], gold[:-1])
    d2.write(out / 'runs' / 'short' / 'identity.json', dict(arm='short'))
    with pytest.raises(ValueError):
        d2.report(out, 'short')


@pytest.mark.parametrize('delta,low,high,expected', [
    (-5.0, -8.0, -2.0, 'uses'), (-5.0, -10.0, 0.5, 'inconclusive'), (-1.0, -2.9, 0.9, 'at_most_small'),
    (-2.0, -2.9, -1.1, 'at_most_small'), (-2.5, -4.0, -1.0, 'inconclusive'), (4.0, 1.0, 7.0, 'inconclusive')])
def test_prosody_verdict_thresholds(delta, low, high, expected):
    contrasts = {'flat_both-vs-sham': dict(delta_uar_pp=delta, uar_interval_95_pp=[low, high])}
    assert d2.decide(CONFIG['decision'], contrasts, {}) == {'f0_energy_use': expected}


def test_other_verdicts():
    decision = CONFIG['decision']
    make = lambda delta, low, high: dict(delta_uar_pp=delta, uar_interval_95_pp=[low, high])
    assert d2.decide(decision, {'sham-vs-orig': make(-2.5, -4, 0)}, {})['vocoder_artifacts'] == 'fail'
    assert d2.decide(decision, {'sham-vs-orig': make(1.5, -1, 4)}, {})['vocoder_artifacts'] == 'pass'
    assert d2.decide(decision, {'orig-vs-text': make(-3, -6, -0.1)}, {})['audio_beyond_transcript'] == 'no'
    assert d2.decide(decision, {'orig-vs-text': make(1, -1, 3)}, {})['audio_beyond_transcript'] == 'inconclusive'
    wer = {'asr-flat_both-vs-asr-sham': dict(delta_wer_pp=1.9)}
    assert d2.decide(decision, {}, wer) == {'flattening_keeps_words': 'pass'}
    damaged = {'asr-sham-vs-asr-orig': dict(delta_wer_pp=16.6)}
    assert d2.decide(decision, {}, damaged) == {'resynthesis_keeps_words': 'fail'}


def test_run_scores_conditions_once_and_survives_optional_failures(corpus, monkeypatch):
    config, tmp = corpus
    out = tmp / 'd2'
    d2.prepare(config, out, backend=SineVocoder())
    calls = []

    def score_choice(model, template, manifest, generation, predictions):
        name = Path(manifest).stem
        calls.append(name)
        if name == 'text':
            raise RuntimeError('template rejects prompts without audio')
        rows = d2.read_jsonl(manifest)
        predictions.parent.mkdir(parents=True, exist_ok=True)
        d2.write_jsonl(predictions, [dict(id=r['id'], reference=r['reference'], prediction=r['reference']) for r in rows])
        return dict(uar=1.0, accuracy=1.0, utterances=len(rows))

    def score_asr(model, template, manifest, generation, predictions):
        calls.append(Path(manifest).stem)
        assert generation == CONFIG['asr']['generation']
        rows = d2.read_jsonl(manifest)
        predictions.parent.mkdir(parents=True, exist_ok=True)
        d2.write_jsonl(predictions, [dict(id=r['id'], word_errors=0, reference_words=3) for r in rows])
        return dict(wer=0.0)

    loaded = []
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    modules = {
        'torch': fake_torch,
        'experiments.ms_swift_iemocap.evaluate': types.SimpleNamespace(score=score_choice),
        'experiments.ms_swift_librispeech.evaluate': types.SimpleNamespace(score=score_asr),
        'experiments.ms_swift_librispeech.train': types.SimpleNamespace(load_best=lambda *a: pytest.fail('adapter loaded')),
        'experiments.ms_swift_aqa.evaluate_frozen': types.SimpleNamespace(load_frozen=lambda cfg: (loaded.append(cfg) or 'model', 'template')),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    model_config = tmp / 'model.json'
    model_config.write_text(json.dumps(dict(model=str(tmp), model_revision='rev')))

    failed = d2.run(out, model_config, 'frozen-base', None, 'frozen-base')
    assert failed == ['text'] and len(loaded) == 1
    assert calls == ['orig', 'sham', 'flat_f0', 'flat_energy', 'flat_both', 'text', 'question_only', 'asr-orig', 'asr-sham', 'asr-flat_both']
    folder = out / 'runs' / 'frozen-base'
    assert (folder / 'text' / 'failure.json').exists() and not (folder / 'text' / 'metrics.json').exists()
    metrics = json.loads((folder / 'flat_both' / 'metrics.json').read_text())
    assert metrics['condition'] == 'flat_both' and metrics['predictions_sha256'] == d2.digest(folder / 'flat_both' / 'predictions.jsonl')
    identity = json.loads((folder / 'identity.json').read_text())
    assert identity['method'] == 'frozen-base' and identity['model_revision'] == 'rev' and 'checkpoint' not in identity

    calls.clear()
    assert d2.run(out, model_config, 'frozen-base', None, 'frozen-base', skip_asr=True) == ['text']
    assert calls == ['text']                                   # finished conditions are never rescored
    result, _ = d2.report(out, 'frozen-base')
    assert result['failed_conditions'] == ['text'] and result['verdicts']['f0_energy_use'] == 'at_most_small'

    other = tmp / 'other.json'
    other.write_text(json.dumps(dict(model=str(tmp), model_revision='another')))
    with pytest.raises(ValueError, match='different model'):
        d2.run(out, other, 'frozen-base', None, 'frozen-base')
    with pytest.raises(ValueError):
        d2.run(out, model_config, 'frozen-base', tmp, 'with-checkpoint')
    with pytest.raises(ValueError):
        d2.run(out, model_config, 'lora', None, 'lora')

    (out / 'manifests' / 'orig.jsonl').write_text('{}\n')
    with pytest.raises(ValueError, match='manifest changed'):
        d2.run(out, model_config, 'frozen-base', None, 'tampered')
