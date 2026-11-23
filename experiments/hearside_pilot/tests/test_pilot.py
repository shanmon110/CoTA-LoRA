"""Hear-side pilot: protocol, training manifests, arm configs and the cross-arm summary (no torch)."""
import json
import sys
import types
from collections import Counter
from pathlib import Path

import pytest

from experiments.hearside_pilot import configs, data, summarize
from experiments.trr_diagnostics import d2
from experiments.trr_diagnostics.tests.test_d2 import corpus  # noqa: F401  (fixture)
from experiments.trr_diagnostics.tests.test_twin import SineVocoder

HERE = Path(configs.__file__).parent
PILOT = json.loads((HERE / 'pilot-iemocap-session4.json').read_text())
D2B = json.loads((HERE.parent / 'trr_diagnostics' / 'd2b-iemocap-session4.json').read_text())
CLASSES, OPTIONS = PILOT['classes'], PILOT['prompts']['options']
REWORDED = PILOT['conditions']['reworded']['orig_reworded']['question_template']


def test_protocol_keeps_the_diagnostic_settings_and_adds_one_reworded_question():
    for key in ('data', 'classes', 'prompts', 'rotations', 'twin', 'generation', 'bootstrap'):
        assert PILOT[key] == D2B[key]
    assert PILOT['data']['sessions'] == [4] and PILOT['rotations'] == [0, 1, 2, 3]
    assert list(PILOT['conditions']['reworded']) == [summarize.REWORDED]
    assert PILOT['conditions']['audio'][0] == 'orig' and set(PILOT['asr']['conditions']) <= set(PILOT['conditions']['audio'])
    named = {n for pair in PILOT['contrasts'] + PILOT['strata_contrasts'] for n in pair}
    assert named <= set(d2.condition_names(PILOT))
    for shift in range(4):
        order = [CLASSES[(j + shift) % 4] for j in range(4)]
        asked = REWORDED.format(*(OPTIONS[c] for c in order))
        trained = PILOT['prompts']['question_template'].format(*(OPTIONS[c] for c in order))
        assert asked != trained and all(asked.count(OPTIONS[c]) == 1 for c in CLASSES)
        # No sentence of the training question survives in the reworded one.
        assert 'Which emotion' not in asked and 'Reply with exactly' not in asked and '; B: ' not in asked


def source_rows():
    rows = []
    for session in (1, 2, 3):
        for index in range(22):
            gender = 'FM'[index % 2]
            uid = f'Ses{session:02}{gender}_impro{index // 6:02}_{gender}{index:03}'
            rows.append(dict(id=uid, speaker=f'Ses{session:02}{gender}', session=session, audios=[f'/wav/{uid}.wav'],
                             messages=[dict(role='user', content='<audio>' + PILOT['prompts']['question']),
                                       dict(role='assistant', content=CLASSES[(index * 7 + session) % 4])]))
    return rows


@pytest.fixture
def splits(tmp_path):
    source = tmp_path / 'train.jsonl'
    data.write_jsonl(source, source_rows())
    return source, tmp_path / 'data', data.build(source, tmp_path / 'data', PILOT)


def test_training_manifests_share_utterances_and_differ_only_in_the_option_order(splits):
    source, out, summary = splits
    rows = source_rows()
    fixed, shuffled = data.read_jsonl(out / 'train-fixed.jsonl'), data.read_jsonl(out / 'train-shuffled.jsonl')
    assert fixed == [r for r in rows if r['session'] in (1, 2)]                 # untouched rows of Sessions 1-2
    assert data.read_jsonl(out / 'dev-fixed.jsonl') == [r for r in rows if r['session'] == 3]
    assert [r['id'] for r in shuffled] == [r['id'] for r in fixed] and [r['audios'] for r in shuffled] == [r['audios'] for r in fixed]
    assert set(shuffled[0]) - set(fixed[0]) == {'option_shift', 'emotion'}       # no column that ms-swift interprets
    for plain, moved in zip(fixed, shuffled):
        shift, label = moved['option_shift'], plain['messages'][1]['content']
        order = [CLASSES[(j + shift) % 4] for j in range(4)]
        assert moved['emotion'] == label and order[CLASSES.index(moved['messages'][1]['content'])] == label
        assert moved['messages'][0]['content'] == '<audio>' + PILOT['prompts']['question_template'].format(*(OPTIONS[c] for c in order))
        if shift == 0:
            assert moved['messages'] == plain['messages']
    for split in ('train', 'dev'):
        moved = data.read_jsonl(out / f'{split}-shuffled.jsonl')
        for label in CLASSES:                                                   # every order equally often within a class
            counts = Counter(r['option_shift'] for r in moved if r['emotion'] == label)
            assert set(counts) == {0, 1, 2, 3} and max(counts.values()) - min(counts.values()) <= 1
        letters = summary[f'{split}-shuffled']['answer_letters']
        assert set(letters) == set(CLASSES) and summary[f'{split}-shuffled']['classes'] == summary[f'{split}-fixed']['classes']
        assert summary[f'{split}-fixed']['sha256'] == data.digest(out / f'{split}-fixed.jsonl')
    assert summary['train-fixed']['sessions'] == [1, 2] and summary['dev-fixed']['sessions'] == [3]
    assert not set(summary['train-fixed']['speakers']) & set(summary['dev-fixed']['speakers'])
    assert json.loads((out / 'splits.json').read_text()) == summary
    assert data.build(source, out.with_name('again'), PILOT) == summary        # same seed, same split


def test_training_data_rejects_evaluation_sessions_other_prompts_and_existing_output(splits, tmp_path):
    source, out, _ = splits
    with pytest.raises(FileExistsError):
        data.build(source, out, PILOT)
    leaked = tmp_path / 'leak.jsonl'
    data.write_jsonl(leaked, source_rows() + [dict(source_rows()[0], id='Ses04F_impro01_F000', session=4, speaker='Ses04F')])
    with pytest.raises(ValueError, match='evaluation sessions'):
        data.build(leaked, tmp_path / 'leak', PILOT)
    other = tmp_path / 'other.jsonl'
    rows = source_rows()
    rows[3]['messages'][0]['content'] = '<audio>What is the emotion?'
    data.write_jsonl(other, rows)
    with pytest.raises(ValueError, match='declared question'):
        data.build(other, tmp_path / 'other', PILOT)


@pytest.fixture
def prepared(corpus, tmp_path):  # noqa: F811
    config, tmp = corpus
    cfg = json.loads(config.read_text())
    config.write_text(json.dumps(dict(PILOT, data=cfg['data'], bootstrap=cfg['bootstrap'], asr=dict(PILOT['asr'], subset=5))))
    out = tmp / 'pilot'
    return out, d2.prepare(config, out, backend=SineVocoder())


def test_evaluation_asks_the_trained_orders_and_one_reworded_question(prepared):
    out, info = prepared
    orig, reworded = (d2.read_jsonl(out / 'manifests' / f'{n}.jsonl') for n in ('orig', 'orig_reworded'))
    assert len(orig) == len(reworded) == 48 and set(info['manifests']) >= {'orig', 'sham', 'neutral', 'orig_reworded', 'text',
                                                                            'question_only', 'asr-orig'}
    for a, b in zip(orig, reworded):
        assert {k: v for k, v in a.items() if k != 'messages'} == {k: v for k, v in b.items() if k != 'messages'}
        assert a['messages'][1] == b['messages'][1] and a['messages'][0] != b['messages'][0]
        assert b['messages'][0]['content'] == '<audio>' + REWORDED.format(*(OPTIONS[c] for c in b['option_order']))
    # A shuffled training row and the evaluation row for the same order carry the same question and letter.
    for row in orig:
        plain = dict(id=row['utterance'], messages=[dict(role='user', content='<audio>' + PILOT['prompts']['question']),
                                                     dict(role='assistant', content=row['label'])])
        moved = data.rotate(plain, row['rotation'], CLASSES, PILOT['prompts']['question_template'], OPTIONS)
        assert moved['messages'] == row['messages']
    assert d2.condition_names(PILOT) == ['orig', 'sham', 'neutral', 'orig_reworded', 'text', 'question_only']


def test_only_scores_the_named_conditions_and_the_rest_can_follow(prepared, monkeypatch):
    out, _ = prepared
    calls = []

    def score(model, template, manifest, generation, predictions):
        calls.append(Path(manifest).stem)
        rows = d2.read_jsonl(manifest)
        predictions.parent.mkdir(parents=True, exist_ok=True)
        if Path(manifest).stem.startswith('asr-'):
            d2.write_jsonl(predictions, [dict(id=r['id'], reference=r['transcript'], prediction=r['transcript'],
                                            word_errors=0, reference_words=3) for r in rows])
            return dict(wer=0.0)
        d2.write_jsonl(predictions, [dict(id=r['id'], reference=r['reference'], prediction=r['reference']) for r in rows])
        return dict(uar=1.0, accuracy=1.0, utterances=len(rows))

    modules = {'torch': types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False)),
               'experiments.ms_swift_iemocap.evaluate': types.SimpleNamespace(score=score),
               'experiments.ms_swift_librispeech.evaluate': types.SimpleNamespace(score=score),
               'experiments.ms_swift_librispeech.train': types.SimpleNamespace(load_best=lambda *a: pytest.fail('adapter loaded')),
               'experiments.ms_swift_aqa.evaluate_frozen': types.SimpleNamespace(load_frozen=lambda cfg: ('model', 'template'))}
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    model_config = out.parent / 'model.json'
    model_config.write_text(json.dumps(dict(model=str(out.parent), model_revision='rev')))
    wanted = ['orig', 'orig_reworded', 'text', 'question_only', 'asr-orig']
    assert d2.run(out, model_config, 'frozen-base', None, 'frozen-base', only=wanted) == []
    assert calls == wanted
    result, _ = d2.report(out, 'frozen-base')                   # a partial arm still reports what it has
    assert set(result['conditions']) == {'orig', 'orig_reworded', 'text', 'question_only'}
    assert result['contrasts']['orig_reworded-vs-orig']['delta_uar_pp'] == 0 and 'neutral-vs-sham' in result['skipped_contrasts']
    assert result['verdicts']['neutral_use'] == 'not_interpretable'
    # What `d2 run` leaves behind is exactly what the cross-arm summary reads.
    table = summarize.summarize(out, ['frozen-base'], resamples=50)
    assert table['results']['frozen-base']['orig']['uar']['other_orders'] == 1.0
    assert table['results']['frozen-base']['asr_wer'] == 0.0 and table['verdict']['outcome'] == 'undecided'
    with pytest.raises(ValueError, match='Unknown conditions'):
        d2.run(out, model_config, 'frozen-base', None, 'frozen-base', only=['orig', 'flat_both'])
    calls.clear()
    assert d2.run(out, model_config, 'frozen-base', None, 'frozen-base') == []
    assert calls == ['sham', 'neutral', 'asr-sham', 'asr-neutral']             # earlier conditions are kept


BASE_CONFIG = dict(methods=['lora', 'ta-lora'], model='/models/omni', model_revision='rev', prompt=PILOT['prompts']['question'],
                   adapter=dict(rank=8, alpha=16, last_n=4),
                   training=dict(seed=42, per_device_train_batch_size=2, gradient_accumulation_steps=4, num_train_epochs=1,
                                 eval_steps=100, save_steps=100, learning_rate=2e-4),
                   patience=1, task='ser', generation=dict(batch_size=2))


def test_arm_configs_differ_only_in_where_the_update_is_added(tmp_path):
    base = tmp_path / 'base.json'
    base.write_text(json.dumps(BASE_CONFIG))
    arms = configs.build(base, tmp_path / 'data', tmp_path / 'configs')
    assert list(arms) == ['A-all', 'B-audio', 'C-post-audio', 'D-all-shuffled'] == summarize.ARMS[1:]
    loaded = {name: json.loads(Path(arm['config']).read_text()) for name, arm in arms.items()}
    assert loaded['A-all'] == loaded['D-all-shuffled']                          # D differs from A in its data only
    assert [loaded[n]['adapter']['update_positions'] for n in arms] == ['all', 'audio', 'post_audio', 'all']
    for cfg in loaded.values():
        rest = {k: v for k, v in cfg['adapter'].items() if k != 'update_positions'}
        assert rest == dict(rank=8, alpha=16, last_n=36, gate_multiplier=0.0)   # a plain low-rank update on every layer
        assert cfg['training']['eval_steps'] == cfg['training']['save_steps'] == 88 and cfg['patience'] == 3
        assert {k: v for k, v in cfg['training'].items() if k not in ('eval_steps', 'save_steps')} == \
               {k: v for k, v in BASE_CONFIG['training'].items() if k not in ('eval_steps', 'save_steps')}
        assert 'methods' not in cfg and cfg['model'] == '/models/omni' and cfg['task'] == 'ser'
    assert [Path(a['train']).name for a in arms.values()] == ['train-fixed.jsonl'] * 3 + ['train-shuffled.jsonl']
    assert [Path(a['dev']).name for a in arms.values()] == ['dev-fixed.jsonl'] * 3 + ['dev-shuffled.jsonl']
    assert json.loads((tmp_path / 'configs' / 'arms.json').read_text()) == arms
    with pytest.raises(FileExistsError):
        configs.build(base, tmp_path / 'data', tmp_path / 'configs')
    base.write_text(json.dumps(dict(BASE_CONFIG, objective='closed_set_4way')))
    with pytest.raises(ValueError):
        configs.build(base, tmp_path / 'data', tmp_path / 'other')


def answers(out, arm, name, choose):
    manifest = d2.read_jsonl(out / 'manifests' / f'{name}.jsonl')
    folder = out / 'runs' / arm / name
    folder.mkdir(parents=True)
    d2.write_jsonl(folder / 'predictions.jsonl', [dict(id=m['id'], reference=m['reference'], prediction=choose(m)) for m in manifest])
    commit(out, folder, name)


def commit(out, folder, name):
    """The marker `d2 run` writes once a condition is complete."""
    info = json.loads((out / 'prepare.json').read_text())
    d2.write(folder / 'metrics.json', dict(condition=name, manifest_sha256=info['manifests'][name],
                                           predictions_sha256=d2.digest(folder / 'predictions.jsonl')))


def transcripts(out, arm, extra_words):
    manifest = d2.read_jsonl(out / 'manifests' / 'asr-orig.jsonl')
    folder = out / 'runs' / arm / 'asr-orig'
    folder.mkdir(parents=True)
    d2.write_jsonl(folder / 'predictions.jsonl', [dict(id=m['id'], reference=m['transcript'],
                                                       prediction='The transcript is: ' + m['transcript'] + ' uh' * extra_words)
                                                  for m in manifest])
    commit(out, folder, 'asr-orig')


right = lambda m: m['reference']                                 # reads the options: correct under every order
locked = lambda m: m['label']                                    # the letter that was right in the training order
half = lambda m: m['reference'] if m['utterance'].endswith(('F000', 'M000')) else 'A'


def fill(out, shuffled_reworded=locked, shuffled_extra_words=1, arms=summarize.ARMS):
    behaviour = {'frozen-base': (half, half, 0), 'A-all': (locked, locked, 2), 'B-audio': (right, right, 0),
                 'C-post-audio': (locked, locked, 2), 'D-all-shuffled': (right, shuffled_reworded, shuffled_extra_words)}
    for arm in arms:
        on_orig, on_reworded, extra = behaviour[arm]
        answers(out, arm, 'orig', on_orig)
        answers(out, arm, 'orig_reworded', on_reworded)
        answers(out, arm, 'text', half)
        transcripts(out, arm, extra)


def test_summary_separates_the_training_order_from_the_others_and_applies_the_declared_rule(prepared):
    out, _ = prepared
    fill(out)
    result = summarize.summarize(out, resamples=300)
    assert result['utterances'] == 12 and result['scored_rows'] == 48 and result['dialogues'] == 3
    plain, hear = result['results']['A-all'], result['results']['B-audio']
    assert plain['orig']['uar'] == dict(training_order=1.0, other_orders=0.0, all_orders=pytest.approx(0.25))
    assert hear['orig']['uar'] == dict(training_order=1.0, other_orders=1.0, all_orders=1.0)
    assert hear['orig']['letter_share'] == {c: pytest.approx(0.25) for c in CLASSES}
    assert plain['orig']['recalls']['other_orders'] == {c: 0.0 for c in CLASSES}
    assert 'question_only' not in hear and 'sham' not in hear                  # unscored conditions are simply absent
    assert hear['asr_wer'] == 0.0 and plain['asr_wer'] > 0                     # the lead-in is not counted as errors
    c = result['contrasts']
    assert c['B-audio - A-all | orig | training_order']['delta_uar_pp'] == 0
    assert c['B-audio - A-all | orig | other_orders']['delta_uar_pp'] == pytest.approx(100)
    assert c['B-audio - D-all-shuffled | orig_reworded | all_orders']['delta_uar_pp'] == pytest.approx(75)
    assert c['B-audio - D-all-shuffled | orig | all_orders']['answers_changed'] == 0
    assert c['B-audio - frozen-base | text | all_orders']['delta_uar_pp'] == 0
    assert result['wer_contrasts']['B-audio - D-all-shuffled']['interval_95_pp'][1] < 0
    assert result['verdict'] == dict(outcome='continue', checks=dict(
        hear_side_learns_the_task=True, hear_side_not_below_base_in_other_orders=True,
        hear_side_beats_plain_in_other_orders=True, hear_side_beats_shuffling_somewhere=True))
    text = summarize.render(result)
    assert '| A-all | 100.0 | 0.0 | 25.0 |' in text and 'Outcome under the declared rule: **continue**' in text
    assert '| B-audio - A-all | orig | other_orders | +100.00 |' in text and '| n/a |' in text


def test_summary_refuses_predictions_that_changed_after_they_were_scored(prepared):
    out, _ = prepared
    fill(out, arms=['frozen-base', 'B-audio'])
    path = out / 'runs' / 'B-audio' / 'orig' / 'predictions.jsonl'
    rows = d2.read_jsonl(path)
    rows[0]['prediction'] = 'D' if rows[0]['prediction'] != 'D' else 'A'
    d2.write_jsonl(path, rows)
    with pytest.raises(ValueError, match='hash changed'):
        summarize.summarize(out, ['frozen-base', 'B-audio'], resamples=50)


def test_shuffling_that_matches_the_hear_side_arm_everywhere_stops_the_line(prepared):
    out, _ = prepared
    fill(out, shuffled_reworded=right, shuffled_extra_words=0)
    verdict = summarize.summarize(out, resamples=300)['verdict']
    assert verdict['outcome'] == 'stop' and verdict['checks']['hear_side_beats_shuffling_somewhere'] is False
    assert all(v for k, v in verdict['checks'].items() if k != 'hear_side_beats_shuffling_somewhere')


def test_a_lower_word_error_rate_alone_also_counts_and_a_missing_arm_is_undecided(prepared, tmp_path):
    out, _ = prepared
    fill(out, shuffled_reworded=right, shuffled_extra_words=1, arms=['frozen-base', 'A-all', 'B-audio', 'D-all-shuffled'])
    result = summarize.summarize(out, ['frozen-base', 'A-all', 'B-audio', 'D-all-shuffled'], resamples=300)
    assert result['verdict']['outcome'] == 'continue'                           # C is a mechanism control, not a criterion
    assert not any('C-post-audio' in key for key in result['contrasts'])
    partial = summarize.summarize(out, ['frozen-base', 'A-all', 'B-audio'], resamples=300)['verdict']
    assert partial['outcome'] == 'undecided' and partial['checks']['hear_side_beats_shuffling_somewhere'] is None
    assert summarize.decide({}, {})['outcome'] == 'undecided'


@pytest.mark.parametrize('same_format,other_base,other_plain,expected', [
    (-3.0, 0.0, 10.0, 'continue'), (-3.1, 5.0, 20.0, 'stop'), (0.0, -0.1, 20.0, 'stop'), (0.0, 5.0, 9.9, 'stop')])
def test_thresholds_of_the_declared_rule(same_format, other_base, other_plain, expected):
    make = lambda delta: dict(delta_uar_pp=delta, interval_95_pp=[delta - 1, delta + 1])
    contrasts = {'B-audio - A-all | orig | training_order': make(same_format),
                 'B-audio - frozen-base | orig | other_orders': make(other_base),
                 'B-audio - A-all | orig | other_orders': make(other_plain),
                 'B-audio - D-all-shuffled | orig_reworded | all_orders': make(4.0)}
    assert summarize.decide(contrasts, {})['outcome'] == expected
    contrasts['B-audio - D-all-shuffled | orig_reworded | all_orders'] = dict(delta_uar_pp=4.0, interval_95_pp=[-0.5, 8.0])
    assert summarize.decide(contrasts, {})['outcome'] == 'stop'                 # an interval that includes 0 is not a win
