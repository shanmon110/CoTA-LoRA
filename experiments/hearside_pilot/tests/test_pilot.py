"""Hear-side pilot: protocol, training manifests, arm configs and the cross-arm summary (no torch)."""
import json
import re
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


# ----------------------------------------------------------------------------- follow-up: depth and seeds
from experiments.hearside_pilot import followup  # noqa: E402
from experiments.trr_diagnostics import stats  # noqa: E402

RESULTS = HERE / 'results' / '20261123-160928'


def test_followup_arms_change_one_setting_of_a_pilot_arm(tmp_path):
    base = tmp_path / 'base.json'
    base.write_text(json.dumps(BASE_CONFIG))
    pilot = configs.build(base, tmp_path / 'data', tmp_path / 'pilot')
    arms = configs.build(base, tmp_path / 'data', tmp_path / 'followup', plan='followup')
    assert list(arms) == ['A-last4', 'A-all-s43', 'B-audio-s43', 'D-all-shuffled-s43', 'A-all-s44', 'B-audio-s44', 'D-all-shuffled-s44']
    assert set(arms) == {m for n in followup.CORE for m in followup.FAMILIES[n][1:]} | {'A-last4'}
    read = lambda arm: json.loads(Path(arm['config']).read_text())
    for name, arm in arms.items():
        like, cfg = read(pilot[arm['like']]), read(arm)
        assert (arm['train'], arm['dev'], arm['update_positions']) == tuple(pilot[arm['like']][k] for k in ('train', 'dev', 'update_positions'))
        if name == 'A-last4':
            assert cfg['adapter'].pop('last_n') == 4 and like['adapter'].pop('last_n') == 36 and cfg == like
            assert (arm['last_n'], arm['seed']) == (4, 42)
        else:
            seed = int(name[-2:])
            assert (cfg['training'].pop('seed'), cfg['training'].pop('data_seed', seed)) == (seed, seed)
            like['training'].pop('seed'), like['training'].pop('data_seed', None)
            assert cfg == like and (arm['last_n'], arm['seed']) == (36, seed)


def test_the_builder_still_writes_the_configs_the_pilot_ran_with(tmp_path):
    recorded = {name: json.loads((RESULTS / 'configs' / f'{name}.json').read_text()) for name in configs.ARMS}
    (tmp_path / 'base.json').write_text(json.dumps(recorded['A-all']))
    arms = configs.build(tmp_path / 'base.json', tmp_path / 'data', tmp_path / 'again')
    assert {name: json.loads(Path(arm['config']).read_text()) for name, arm in arms.items()} == recorded
    follow = configs.build(tmp_path / 'base.json', tmp_path / 'data', tmp_path / 'next', plan='followup')
    s43 = json.loads(Path(follow['B-audio-s43']['config']).read_text())
    assert (s43['training']['seed'], s43['training']['data_seed']) == (43, 43)
    assert s43['adapter'] == recorded['B-audio']['adapter']


@pytest.fixture
def two_runs(corpus):  # noqa: F811
    config, tmp = corpus
    cfg = json.loads(config.read_text())
    config.write_text(json.dumps(dict(PILOT, data=cfg['data'], bootstrap=cfg['bootstrap'], asr=dict(PILOT['asr'], subset=5))))
    for name in ('pilot', 'next'):
        d2.prepare(config, tmp / name, backend=SineVocoder())
    return tmp / 'pilot', tmp / 'next', config


wrong_on = lambda *suffixes: (lambda m: m['reference'] if not m['utterance'].endswith(suffixes) else 'ABCD'[('ABCD'.index(m['reference']) + 1) % 4])


def place(folder, arm, on_orig, on_reworded=None, extra_words=0):
    answers(folder, arm, 'orig', on_orig)
    answers(folder, arm, 'orig_reworded', on_reworded or on_orig)
    answers(folder, arm, 'text', half)
    transcripts(folder, arm, extra_words)


def test_followup_reads_both_runs_and_judges_the_declared_claims(two_runs):
    pilot, later, _ = two_runs
    place(pilot, 'frozen-base', half)
    place(pilot, 'C-post-audio', locked)
    # A: one, two and no utterances wrong; B: always right; D: as A. The last-four-layer arm answers the trained letter.
    place(pilot, 'A-all', wrong_on('impro01_F000'), extra_words=1)
    place(pilot, 'B-audio', right)
    place(pilot, 'D-all-shuffled', wrong_on('impro01_F000'))
    place(later, 'A-all-s43', wrong_on('impro01_F000', 'impro02_M000'), extra_words=1)
    place(later, 'A-all-s44', right)
    for seed in (43, 44):
        place(later, f'B-audio-s{seed}', right)
        place(later, f'D-all-shuffled-s{seed}', wrong_on('impro01_F000', 'impro02_M000') if seed == 43 else right)
    place(later, 'A-last4', locked)
    result = followup.summarize([pilot, later], resamples=400)
    assert result['seeds'] == [42, 43, 44] and result['utterances'] == 12 and result['scored_rows'] == 48
    assert list(result['models']) == ['frozen-base', 'C-post-audio', 'A-last4', *(m for n in followup.CORE for m in followup.FAMILIES[n])]
    assert 'A-last4' not in result['families'] and 'A-last4 - A-all' not in result['contrasts']   # one seed of that depth

    # The vectorised score agrees with the scorer used everywhere else.
    manifest = d2.read_jsonl(pilot / 'manifests' / 'orig.jsonl')
    rows = [(m['label'], m['option_order'][CLASSES.index(wrong_on('impro01_F000')(m))]) for m in manifest]
    assert result['models']['A-all']['uar_all_orders'] == pytest.approx(stats.classification(rows, CLASSES)['uar'])
    assert result['models']['A-all']['uar_all_orders'] == pytest.approx(1 - 1 / 12)      # one of three angry utterances
    assert result['models']['A-last4']['order_gap'] == pytest.approx(1.0)
    assert result['models']['B-audio']['order_gap'] == 0 and result['models']['B-audio']['wer'] == 0
    assert result['models']['A-all']['wer'] > 0 and result['models']['A-all']['uar_text'] == result['models']['B-audio']['uar_text']

    family = result['families']['A-all']
    assert family['per_seed']['uar_all_orders'] == pytest.approx([1 - 1 / 12, 1 - 2 / 12, 1.0])
    assert family['mean']['uar_all_orders'] == pytest.approx(1 - 1 / 12) and family['sd']['uar_all_orders'] == pytest.approx(1 / 12)
    gain = result['contrasts']['B-audio - A-all']['uar_all_orders']
    assert gain['per_seed_pp'] == pytest.approx([100 / 12, 200 / 12, 0]) and gain['mean_pp'] == pytest.approx(100 / 12)
    assert 0 <= gain['interval_95_pp'][0] <= gain['mean_pp'] <= gain['interval_95_pp'][1] <= 100   # B is never worse than A
    same = result['contrasts']['D-all-shuffled - A-all']['uar_all_orders']
    assert same['per_seed_pp'] == [0, 0, 0] and same['interval_95_pp'] == [0, 0]
    depth = result['depth']
    assert depth['order_gap']['delta_pp'] == pytest.approx(100) and depth['order_gap']['interval_95_pp'][0] > 0
    assert depth['last4_minus_base_other_orders']['delta_pp'] < 0
    assert result['claims'] == dict(audio_positions_match_all_positions='inconclusive',     # +8.3 pp with a wide interval
                                    audio_positions_less_bound_to_option_order='inconclusive',
                                    shuffled_options_improve_all_positions='inconclusive',
                                    last_layers_bind_to_option_order='supported')
    assert followup.summarize([pilot, later], resamples=400) == result                    # same seed, same resamples
    text = followup.render(result)
    assert '| A-last4 | 100.0 | 0.0 | 25.0 | +100.0 ' in text and '- last_layers_bind_to_option_order: **supported**' in text
    assert '| B-audio - A-all | 42, 43, 44 | uar_all_orders | +8.33 / +16.67 / +0.00 | +8.33 |' in text


def test_followup_reports_missing_runs_and_refuses_mismatched_inputs(two_runs):
    pilot, later, config = two_runs
    place(pilot, 'frozen-base', half)
    for arm in ('A-all', 'B-audio', 'D-all-shuffled'):
        place(pilot, arm, right)
    alone = followup.summarize([pilot, later], resamples=100)                              # nothing trained yet
    assert alone['seeds'] == [42] and alone['families'] == {} and alone['depth'] is None
    assert set(alone['claims'].values()) == {None} and 'not available' in followup.render(alone)
    for arm in ('A-all', 'B-audio', 'D-all-shuffled'):
        place(later, f'{arm}-s43', right)
    two = followup.summarize([pilot, later], resamples=100)                                # the third seed failed
    assert two['seeds'] == [42, 43] and two['claims']['audio_positions_match_all_positions'] == 'supported'
    assert two['claims']['last_layers_bind_to_option_order'] is None
    place(later, 'A-last4', right)
    assert followup.summarize([pilot, later], resamples=100)['claims']['last_layers_bind_to_option_order'] == 'not supported'
    place(later, 'B-audio', right)
    with pytest.raises(ValueError, match='more than one folder'):
        followup.summarize([pilot, later], resamples=100)
    other = json.loads(config.read_text())
    other['conditions']['reworded']['orig_reworded']['question_template'] += ' Thanks.'
    config.write_text(json.dumps(other))
    d2.prepare(config, pilot.parent / 'different', backend=SineVocoder())
    with pytest.raises(ValueError, match='not scored on the same manifest'):
        followup.summarize([pilot, pilot.parent / 'different'], resamples=100)


@pytest.mark.parametrize('interval,expected', [([-2.9, 2.9], 'supported'), ([-3.1, 1.0], 'inconclusive'), ([3.2, 5.0], 'contradicted'),
                                               ([-6.0, -3.5], 'contradicted')])
def test_equivalence_reading_uses_the_declared_margin(interval, expected):
    entry = lambda low, high: dict(interval_95_pp=[low, high])
    contrasts = {'B-audio - A-all': dict(uar_all_orders=entry(*interval), order_gap=entry(-4.0, -0.5)),
                 'D-all-shuffled - A-all': dict(uar_all_orders=entry(-1.0, 2.0))}
    claims = followup.judge(contrasts, dict(order_gap=dict(delta_pp=9.9, interval_95_pp=[5.0, 14.0])))
    assert claims == dict(audio_positions_match_all_positions=expected, audio_positions_less_bound_to_option_order='supported',
                          shuffled_options_improve_all_positions='inconclusive', last_layers_bind_to_option_order='not supported')
    assert followup.judge(contrasts, dict(order_gap=dict(delta_pp=10.0, interval_95_pp=[0.1, 20.0])))['last_layers_bind_to_option_order'] == 'supported'
    assert followup.judge(contrasts, dict(order_gap=dict(delta_pp=15.0, interval_95_pp=[-0.1, 30.0])))['last_layers_bind_to_option_order'] == 'not supported'


def test_followup_inputs_are_compared_with_the_pilot(corpus, tmp_path):  # noqa: F811
    config, tmp = corpus
    cfg = json.loads(config.read_text())
    config.write_text(json.dumps(dict(PILOT, data=cfg['data'], bootstrap=cfg['bootstrap'], asr=dict(PILOT['asr'], subset=5))))
    source = tmp_path / 'train.jsonl'
    data.write_jsonl(source, source_rows())
    for run in ('first', 'second'):
        data.build(source, tmp_path / run / 'data', PILOT)
        d2.prepare(config, tmp_path / run / 'eval', backend=SineVocoder())
    assert followup.same_inputs(tmp_path / 'first', tmp_path / 'second') == []      # both steps are deterministic
    data.build(source, tmp_path / 'third' / 'data', PILOT, seed=7)                  # other option shifts
    other = json.loads(config.read_text())
    other['asr']['subset'] = 4
    config.write_text(json.dumps(other))
    d2.prepare(config, tmp_path / 'third' / 'eval', backend=SineVocoder())
    assert followup.same_inputs(tmp_path / 'first', tmp_path / 'third') == ['data/train-shuffled.jsonl', 'data/dev-shuffled.jsonl',
                                                                           'eval/manifests/asr-orig.jsonl']


def test_seeds_of_the_last_four_layer_arm_are_compared_with_the_same_seeds_of_the_full_depth_arm(two_runs, tmp_path):
    pilot, later, config = two_runs
    d2.prepare(config, pilot.parent / 'third', backend=SineVocoder())
    third = pilot.parent / 'third'
    place(pilot, 'frozen-base', half)
    for arm in ('A-all', 'B-audio', 'D-all-shuffled'):
        place(pilot, arm, right)
    place(later, 'A-last4', locked)
    place(later, 'A-all-s43', right)                    # seed 43 of A only: the core families share seed 42 alone
    place(third, 'A-last4-s43', locked)
    place(third, 'A-last4-s44', locked)                 # no A-all-s44 to pair with
    result = followup.summarize([pilot, later, third], resamples=200)
    assert result['seeds'] == [42] and set(result['families']) == {'A-last4'}
    assert result['families']['A-last4']['seeds'] == [42, 43, 44] and result['families']['A-last4']['mean']['order_gap'] == pytest.approx(1.0)
    gap = result['contrasts']['A-last4 - A-all']['order_gap']
    assert gap['seeds'] == [42, 43] and gap['per_seed_pp'] == pytest.approx([100, 100]) and gap['interval_95_pp'][0] > 0
    assert result['claims']['last_layers_bind_to_option_order_across_seeds'] == 'supported'
    assert result['claims']['audio_positions_match_all_positions'] is None
    assert '| A-last4 - A-all | 42, 43 | order_gap | +100.00 / +100.00 | +100.00 |' in followup.render(result)
    base = tmp_path / 'base.json'
    base.write_text(json.dumps(BASE_CONFIG))
    arms = configs.build(base, tmp_path / 'data', tmp_path / 'cfg', plan='last4-seeds')
    assert {n: (a['last_n'], a['seed'], a['update_positions'], a['option_order']) for n, a in arms.items()} == {
        'A-last4-s43': (4, 43, 'all', 'fixed'), 'A-last4-s44': (4, 44, 'all', 'fixed')}


# ----------------------------------------------------------------------------- other answer formats
from experiments.hearside_pilot import formats, headroom  # noqa: E402


@pytest.mark.parametrize('text,expected', [('Yes.', 'yes'), ('no, the speaker is calm', 'no'), ('The speaker sounds angry', None), ('', None)])
def test_yes_no_answers_are_read_from_the_first_word(text, expected):
    assert formats.parse_yesno(text) == expected


@pytest.mark.parametrize('text,expected', [('Angry.', 'A'), ('happy or excited', 'B'), ('The speaker sounds sad', 'C'), ('Neutral', 'D'),
                                           ('unhappy', 'C'), ('frustrated', 'A'), ('happy and sad', None), ('confused', None), ('', None)])
def test_emotion_words_map_to_one_class_or_are_invalid(text, expected):
    assert formats.parse_emotion(text) == expected


def format_answers(folder, arm, yes=lambda m: m['group'], word=lambda m: formats.EMOTIONS[m['group']], free=None):
    for name, choose in (('yesno', yes), ('word', word), ('open', free or word)):
        out = folder / 'runs' / arm / name
        out.mkdir(parents=True)
        d2.write_jsonl(out / 'predictions.jsonl', [dict(id=m['id'], reference=m['messages'][1]['content'], prediction=choose(m))
                                                   for m in d2.read_jsonl(folder / f'{name}.jsonl')])
        d2.write(out / 'metrics.json', dict(predictions_sha256=d2.digest(out / 'predictions.jsonl')))


def test_format_manifests_and_the_declared_reading(prepared, tmp_path):
    out, _ = prepared
    folder = tmp_path / 'formats'
    info = formats.build(out / 'manifests' / 'orig.jsonl', folder)
    assert info['utterances'] == 12 and {n: m['rows'] for n, m in info['manifests'].items()} == dict(yesno=24, word=24, open=12)
    yesno, word, free = (d2.read_jsonl(folder / f'{n}.jsonl') for n in formats.FORMATS)
    for a, b in zip(yesno[::2], yesno[1::2]):
        assert (a['group'], b['group']) == ('yes', 'no') and a['asked'] == a['label'] != b['asked'] and a['audios'] == b['audios']
        assert a['messages'][0]['content'] == f"<audio>Does the speaker sound {formats.EMOTIONS[a['label']]}? Answer yes or no."
    assert {r['asked'] for r in yesno[1::2]} == set(CLASSES)                      # wrong emotions are spread over the classes
    assert word[0]['messages'][0]['content'].index('angry') < word[0]['messages'][0]['content'].index('sad')
    assert word[1]['messages'][0]['content'].index('sad') < word[1]['messages'][0]['content'].index('angry')
    assert all('angry' not in r['messages'][0]['content'] and r['messages'][1]['content'] == formats.EMOTIONS[r['label']] for r in free)
    for rows in (yesno, word, free):                                              # nothing ms-swift would misread, no option letters
        assert all(not re.search(r'\b[ABCD][:.]', r['messages'][0]['content']) for r in rows)

    always_yes = lambda m: 'Yes'
    format_answers(folder, 'frozen-base', yes=always_yes, word=lambda m: 'neutral', free=lambda m: 'I am not sure')
    for seed, arm in enumerate(formats.FAMILIES['D-all-shuffled']):               # perfect on yes/no, never on the open question
        format_answers(folder, arm, free=lambda m: 'The speaker sounds neutral.' if seed else 'hmm')
    format_answers(folder, 'A-all', yes=always_yes, word=lambda m: 'B')           # a letter is not an emotion word
    result = formats.summarize(folder, 17.4, resamples=200)
    table = result['table']
    assert table['frozen-base']['yesno']['mean'] == 0.5 and table['frozen-base']['word']['mean'] == 0.25
    assert table['frozen-base']['open'] == dict(per_model={'frozen-base': 0.0}, mean=0.0, parseable=0.0)
    assert table['D-all-shuffled']['yesno']['mean'] == 1.0 and len(table['D-all-shuffled']['open']['per_model']) == 3
    assert table['D-all-shuffled']['open']['mean'] == pytest.approx((0 + .25 + .25) / 3) and table['A-all']['word']['parseable'] == 0
    gain = result['contrasts']['D-all-shuffled - frozen-base']
    assert gain['yesno']['delta_pp'] == pytest.approx(50) and gain['yesno']['interval_95_pp'] == [50, 50]
    assert result['contrasts']['D-all-shuffled - A-all']['word']['delta_pp'] == pytest.approx(100)
    assert result['readings'] == dict(yesno='transfers', word='transfers', open='transfers')            # +16.7 pp is over half of +17.4
    assert formats.summarize(folder, 40.0, resamples=50)['readings']['open'] == 'gap'                   # but under half of +40
    assert formats.summarize(folder, None, resamples=50)['readings'] == dict(yesno=None, word=None, open=None)
    text = formats.render(result)
    assert '| frozen-base (1) | 50.0 [100%] | 25.0 [100%] | 0.0 [0%] |' in text and '- open: **' in text
    rows = d2.read_jsonl(folder / 'runs' / 'A-all' / 'open' / 'predictions.jsonl')
    d2.write_jsonl(folder / 'runs' / 'A-all' / 'open' / 'predictions.jsonl', rows[::-1])
    with pytest.raises(ValueError, match='changed after'):
        formats.summarize(folder, 17.4, resamples=50)


@pytest.mark.parametrize('base,gain,interval,expected', [(0.5, 9.0, [1, 17], 'transfers'), (0.5, 8.0, [1, 15], 'gap'), (0.5, -6.0, [-9, -1], 'harmed'),
                                                         (0.9, 1.0, [-1, 3], 'base at ceiling'), (0.5, -1.0, [-4, 2], 'gap')])
def test_transfer_reading_thresholds(base, gain, interval, expected):
    table = {'frozen-base': {n: dict(mean=base) for n in formats.FORMATS}}
    contrasts = {'D-all-shuffled - frozen-base': {n: dict(delta_pp=gain, interval_95_pp=interval) for n in formats.FORMATS}}
    assert set(formats.judge(table, contrasts, 17.4).values()) == {expected}


def test_mcq_gain_is_read_from_the_followup_summary(tmp_path):
    path = tmp_path / 'followup.json'
    d2.write(path, dict(models={'frozen-base': dict(uar_all_orders=0.561), 'D-all-shuffled': dict(uar_all_orders=0.760)},
                        families={'D-all-shuffled': dict(mean=dict(uar_all_orders=0.735))}))
    assert formats.mcq_gain(path) == pytest.approx(17.4)
    d2.write(path, dict(models={'frozen-base': dict(uar_all_orders=0.561), 'D-all-shuffled': dict(uar_all_orders=0.760)}, families={}))
    assert formats.mcq_gain(path) == pytest.approx(19.9)


# ----------------------------------------------------------------------------- headroom for input-conditional adaptation
def headroom_rows():
    rows = []
    for session in (1, 2, 3):
        for index in range(24):
            g, kind = 'FM'[index % 2], ('impro', 'script')[index // 2 % 2]
            uid = f'Ses{session:02}{g}_{kind}{index // 8:02}_{g}{index:03}'
            rows.append(dict(id=uid, speaker=f'Ses{session:02}{g}', session=session, audios=[f'/wav/{index % 9}/{uid}.wav'],
                             messages=[dict(role='user', content='<audio>' + PILOT['prompts']['question']),
                                       dict(role='assistant', content=CLASSES[(index * 7 + session) % 4])]))
    return rows


@pytest.fixture
def specialists(prepared, tmp_path):
    out, _ = prepared
    source = tmp_path / 'train.jsonl'
    data.write_jsonl(source, headroom_rows())
    data.build(source, tmp_path / 'data', PILOT)
    (tmp_path / 'base.json').write_text(json.dumps(dict(BASE_CONFIG, training=dict(BASE_CONFIG['training'], eval_steps=88, save_steps=88))))
    length = lambda path: 1.0 + (int(Path(path).parent.name) if path.startswith('/wav/') else sum(map(ord, Path(path).stem)) % 9)
    info = headroom.build(tmp_path / 'data', out / 'manifests' / 'orig.jsonl', tmp_path / 'base.json', tmp_path / 'headroom', duration=length)
    return out, tmp_path / 'headroom', info


def test_specialist_manifests_partition_every_split_by_one_property(specialists, tmp_path):
    out, folder, info = specialists
    assert len(info['specialists']) == 7 + 5 and len(info['duration_cuts_seconds']) == 2
    full = {s: d2.read_jsonl(p) for s, p in (('train', tmp_path / 'data' / 'train-shuffled.jsonl'), ('dev', tmp_path / 'data' / 'dev-shuffled.jsonl'),
                                              ('test', out / 'manifests' / 'orig.jsonl'))}
    for grouping in (*headroom.GROUPINGS, 'random2', 'random3'):
        keys = [k for k, v in info['specialists'].items() if v['grouping'] == grouping]
        for split, rows in full.items():
            parts = [d2.read_jsonl(folder / k / f'{split}.jsonl') for k in keys]
            assert sorted(r['id'] for part in parts for r in part) == sorted(r['id'] for r in rows)     # a partition, nothing lost or doubled
    for row in d2.read_jsonl(folder / 'gender-F' / 'train.jsonl'):
        assert row['id'].split('_')[-1][0] == 'F'
    assert all('_script' in r['id'] for r in d2.read_jsonl(folder / 'scripted-script' / 'dev.jsonl'))
    test = d2.read_jsonl(folder / 'duration-short' / 'test.jsonl')
    assert len(test) % 4 == 0 and len({r['utterance'] for r in test}) == len(test) // 4                 # all four option orders of an utterance
    sizes = info['train_group_sizes']
    assert [info['specialists'][f'random2-part{i}']['rows']['train'] for i in range(2)] == sizes['gender']
    assert sum(sizes['duration']) == 48 and max(sizes['duration']) - min(sizes['duration']) <= 12
    cfg = json.loads((folder / 'duration-long' / 'config.json').read_text())
    steps = info['specialists']['duration-long']['steps']
    assert steps == -(-info['specialists']['duration-long']['rows']['train'] // 8) and cfg['training']['eval_steps'] == cfg['training']['save_steps'] == -(-steps // 3)
    assert {k: v for k, v in cfg.items() if k != 'training'} == {k: v for k, v in json.loads((tmp_path / 'base.json').read_text()).items() if k != 'training'}
    with pytest.raises(FileExistsError):
        headroom.build(tmp_path / 'data', out / 'manifests' / 'orig.jsonl', tmp_path / 'base.json', folder)


def specialist_answers(folder, key, choose):
    out = folder / key / 'eval'
    out.mkdir()
    d2.write_jsonl(out / 'predictions.jsonl', [dict(id=m['id'], reference=m['reference'], prediction=choose(m))
                                               for m in d2.read_jsonl(folder / key / 'test.jsonl')])
    d2.write(out / 'metrics.json', dict(predictions_sha256=d2.digest(out / 'predictions.jsonl')))


def test_headroom_needs_a_gain_over_the_shared_adapter_the_random_split_and_the_seed_spread(specialists):
    out, folder, info = specialists
    place(out, 'frozen-base', half)
    place(out, 'D-all-shuffled', half)                                       # the shared adapter is right on a third of the utterances
    before = headroom.summarize(folder, [out], seed_sd_pp=4.0, resamples=200)
    assert before['outcome'] == 'undecided' and set(before['groupings'].values()) == {None} and before['shared_seeds'] == ['D-all-shuffled']
    assert sum(g['utterances'] for g in before['groups']['gender'].values()) == 12
    assert all(len(g['shared_uar_per_seed']) == 1 for by in before['groups'].values() for g in by.values())
    for key, spec in info['specialists'].items():
        specialist_answers(folder, key, right if spec['grouping'] == 'gender' else half)
    result = headroom.summarize(folder, [out], seed_sd_pp=4.0, resamples=200)
    gender, duration = result['groupings']['gender'], result['groupings']['duration']
    assert gender['uar'] == 100 and gender['gain_over_shared_pp'] == pytest.approx(100 - gender['shared_uar']) and gender['headroom'] is True
    assert gender['gain_over_random_pp'] == pytest.approx(gender['gain_over_shared_pp']) and gender['random_minus_shared_pp'] == 0
    assert duration['gain_over_shared_pp'] == 0 and duration['answers_changed'] == 0 and duration['headroom'] is False
    assert result['outcome'] == 'headroom found'
    assert headroom.summarize(folder, [out], seed_sd_pp=60.0, resamples=200)['outcome'] == 'closed'       # inside the seed spread
    assert headroom.summarize(folder, [out], resamples=200)['outcome'] == 'undecided'                     # seed spread unknown
    text = headroom.render(result)
    assert '| gender | 100.0 |' in text and 'Outcome under the declared rule: **headroom found**' in text
    entry = dict(gain_over_shared_pp=9.0, gain_over_shared_interval_95_pp=[1, 17], gain_over_random_pp=3.0, gain_over_random_interval_95_pp=[-0.1, 6])
    assert headroom.judge(entry, 4.0) is False                                                            # not better than a random split
    assert headroom.judge(dict(entry, gain_over_random_interval_95_pp=[0.1, 6]), 4.0) is True
    assert headroom.judge(dict(entry, gain_over_random_interval_95_pp=[0.1, 6]), 4.6) is False
