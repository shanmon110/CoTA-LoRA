"""Second protocol: level twins, rotated option orders, a word check per twin."""
import json
from pathlib import Path

import pytest

from experiments.trr_diagnostics import d2, twin
from experiments.trr_diagnostics.tests.test_d2 import corpus  # noqa: F401  (fixture)
from experiments.trr_diagnostics.tests.test_twin import SineVocoder

HERE = Path(d2.__file__).parent
V2 = json.loads((HERE / 'd2b-iemocap-session4.json').read_text())
V1 = json.loads((HERE / 'd2-iemocap-session4.json').read_text())


def test_second_protocol_extends_the_first_without_changing_shared_settings():
    for key in ('data', 'classes', 'twin', 'generation', 'bootstrap'):
        assert V2[key] == V1[key]
    assert V2['prompts']['question'] == V1['prompts']['question']
    assert V2['prompts']['question_template'].format(*(V2['prompts']['options'][c] for c in V2['classes'])) == V2['prompts']['question']
    assert V2['rotations'] == [0, 1, 2, 3]
    assert {k: v for k, v in V2['decision'].items() if k not in ('primary', 'qc')} == \
           {k: v for k, v in V1['decision'].items() if k != 'qc'}
    named = {n for pair in V2['contrasts'] + V2['strata_contrasts'] for n in pair}
    assert named <= set(V2['conditions']['audio']) | set(V2['conditions']['text'])
    assert set(V2['decision']['primary']) <= set(V2['conditions']['audio']) & set(V2['asr']['conditions'])
    assert set(V2['asr']['conditions']) - {'orig'} == set(twin.AUDIO_CONDITIONS)


@pytest.fixture
def prepared(corpus, tmp_path):  # noqa: F811
    config, tmp = corpus
    cfg = json.loads(config.read_text())
    v2 = dict(V2, data=cfg['data'], bootstrap=cfg['bootstrap'], asr=dict(V2['asr'], subset=5))
    config.write_text(json.dumps(v2))
    out = tmp / 'd2b'
    return out, d2.prepare(config, out, backend=SineVocoder())


def test_every_utterance_is_asked_under_each_option_order(prepared):
    out, info = prepared
    assert info['utterances'] == 12 and info['rotations'] == [0, 1, 2, 3]
    assert set(info['speaker_reference']) == {'Ses04F', 'Ses04M'}
    rows = d2.read_jsonl(out / 'manifests' / 'level_both.jsonl')
    assert len(rows) == 48 and len({r['id'] for r in rows}) == 48
    one = [r for r in rows if r['utterance'].endswith('impro02_F001')]      # labelled C: sad
    assert [r['rotation'] for r in one] == [0, 1, 2, 3] and {r['label'] for r in one} == {'C'}
    assert [r['reference'] for r in one] == ['C', 'B', 'A', 'D']             # the letter that says "sad" moves
    assert [r['messages'][1]['content'] for r in one] == ['C', 'B', 'A', 'D']
    assert 'A: angry; B: happy or excited; C: sad; D: neutral' in one[0]['messages'][0]['content']
    assert 'A: sad; B: neutral; C: angry; D: happy or excited' in one[2]['messages'][0]['content']
    assert all(r['option_order'][ 'ABCD'.index(r['reference'])] == r['label'] for r in rows)
    assert len({r['audios'][0] for r in one}) == 1 and '/audio/level_both/' in one[0]['audios'][0]
    text = d2.read_jsonl(out / 'manifests' / 'text.jsonl')
    assert len(text) == 48 and 'A: sad; B: neutral' in text[2]['messages'][0]['content'] and text[2]['audios'] == []
    asr = d2.read_jsonl(out / 'manifests' / 'asr-flat_f0.jsonl')            # transcribed but not classified
    assert len(asr) == 5 and '#' not in asr[0]['id'] and (out / 'audio' / 'flat_f0').is_dir()
    assert not (out / 'manifests' / 'flat_f0.jsonl').exists()


def test_level_twins_bring_a_speakers_utterances_to_one_pitch_height_and_loudness(prepared):
    out, info = prepared
    records = d2.read_jsonl(out / 'qc.jsonl')
    for record in records:
        ref, stats = record['reference'], record['stats']
        scale = record['peak_scale']
        assert stats['level_both']['rms'] == pytest.approx(scale * ref['rms'], rel=0.01)
        assert stats['level_energy']['rms'] == pytest.approx(scale * ref['rms'], rel=0.01)
        assert stats['sham']['rms'] == pytest.approx(scale * stats['orig']['rms'], rel=0.01)
        assert stats['level_f0']['rms'] == pytest.approx(stats['sham']['rms'], rel=0.01)
        assert stats['level_f0']['f0_median_hz'] == pytest.approx(ref['f0_hz'], rel=0.03)
        assert stats['neutral']['f0_median_hz'] == pytest.approx(ref['f0_hz'], rel=0.03)
        assert stats['level_f0']['f0_sd_semitones'] == pytest.approx(stats['sham']['f0_sd_semitones'], rel=0.2)
    checks = info['qc']['checks']
    assert checks['level_energy:level_matches_other_twins'] and checks['sham:level_matches_other_twins']
    assert {'level_f0:f0_median_hz_spread:levelled', 'level_both:rms_spread:levelled',
            'neutral:median_f0_sd_ratio_to_orig:flat'} <= set(checks)
    assert 'sham:rms_spread:levelled' not in checks


def _answers(out, arm, name, choose):
    manifest = d2.read_jsonl(out / 'manifests' / f'{name}.jsonl')
    folder = out / 'runs' / arm / name
    folder.mkdir(parents=True)
    d2.write_jsonl(folder / 'predictions.jsonl', [dict(id=m['id'], reference=m['reference'], prediction=choose(m)) for m in manifest])
    info = json.loads((out / 'prepare.json').read_text())
    d2.write(folder / 'metrics.json', dict(manifest_sha256=info['manifests'][name],
                                           predictions_sha256=d2.digest(folder / 'predictions.jsonl')))


def _transcripts(out, arm, name, errors):
    manifest = d2.read_jsonl(out / 'manifests' / f'asr-{name}.jsonl')
    folder = out / 'runs' / arm / f'asr-{name}'
    folder.mkdir(parents=True)
    d2.write_jsonl(folder / 'predictions.jsonl', [dict(id=m['id'], reference=m['transcript'], word_errors=errors if i == 0 else 0,
                                                       reference_words=4) for i, m in enumerate(manifest)])
    info = json.loads((out / 'prepare.json').read_text())
    d2.write(folder / 'metrics.json', dict(manifest_sha256=info['manifests'][f'asr-{name}'],
                                           predictions_sha256=d2.digest(folder / 'predictions.jsonl')))


def test_report_scores_classes_and_exposes_a_letter_preference(prepared):
    out, info = prepared
    arm = out / 'runs' / 'toy'
    arm.mkdir(parents=True)
    d2.write(arm / 'identity.json', dict(arm='toy', prepared_sha256=d2.digest(out / 'prepare.json')))
    right = lambda m: m['reference']
    always_a = lambda m: 'A'
    # Correct on the original and on every twin except level_both, where it falls back to always answering "A".
    for name in ('orig', 'sham', 'flat_both', 'level_f0', 'level_energy', 'neutral', 'text'):
        _answers(out, 'toy', name, right)
    _answers(out, 'toy', 'level_both', always_a)
    _answers(out, 'toy', 'question_only', always_a)
    for name in V2['asr']['conditions']:
        _transcripts(out, 'toy', name, 3 if name == 'neutral' else 0)

    result, markdown = d2.report(out, 'toy')
    assert result['utterances'] == 12 and result['scored_rows'] == 48
    assert result['conditions']['orig']['uar'] == 1.0
    # Always "A" is right exactly once per utterance across the four orders: chance for every class.
    assert result['conditions']['level_both']['uar'] == pytest.approx(0.25)
    assert result['conditions']['level_both']['recalls'] == {c: pytest.approx(0.25) for c in 'ABCD'}
    assert result['option_letter_share']['level_both'] == {'A': 1.0, 'B': 0.0, 'C': 0.0, 'D': 0.0}
    assert result['option_letter_share']['orig'] == {c: pytest.approx(0.25) for c in 'ABCD'}
    assert set(result['uar_by_rotation']['orig']) == {'0', '1', '2', '3'}
    assert result['contrasts']['level_both-vs-sham']['delta_uar_pp'] == pytest.approx(-75)
    assert result['raw_verdicts']['level_both_use'] == 'uses' and result['verdicts']['level_both_use'] == 'uses'
    assert result['verdicts']['f0_energy_use'] == 'at_most_small'
    # neutral lost words in transcription, so its contrast is withheld; the other edits are still read.
    assert result['raw_verdicts']['neutral_keeps_words'] == 'fail' and result['verdicts']['neutral_use'] == 'not_interpretable'
    assert result['interpretations']['neutral_use']['reasons'] == ['neutral_keeps_words:fail']
    assert result['interpretations']['level_both_use']['eligible']
    assert '**neutral_use: NOT INTERPRETABLE.**' in markdown and 'A 100%' in markdown and '4 option order(s)' in markdown
