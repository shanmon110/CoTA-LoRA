import json
from pathlib import Path

import numpy as np
import pytest

from experiments.hearside_pilot import crosscorpus, formats
from experiments.trr_diagnostics import d2

PROTOCOL = Path(__file__).parents[1] / 'pilot-iemocap-session4.json'
EMOTION = dict(A='angry', B='happy', C='sad', D='neutral')


def corpus(tmp_path, actors=12):
    audio = tmp_path / 'AudioWAV'
    audio.mkdir()
    for actor in range(1001, 1001 + actors):
        for sentence in ('DFA', 'IEO', 'IOM', 'ITH', 'TIE'):
            for code in ('ANG', 'HAP', 'SAD', 'NEU', 'DIS', 'FEA'):
                (audio / f'{actor}_{sentence}_{code}_XX.wav').touch()
    (audio / 'notes.txt').touch()
    return audio


def answers(folder, arm, rule, seed=0):
    """Write predictions for every task. `rule(row, task)` returns the probability of a right answer."""
    rng = np.random.default_rng(seed)
    info = d2.load(folder / 'crosscorpus.json')
    for task in crosscorpus.TASKS:
        rows = d2.read_jsonl(crosscorpus.manifest_path(folder, task))
        out = []
        for row in rows:
            right = rng.random() < rule(row, task)
            if task == 'mcq':
                text = row['reference'] if right else 'ABCD'[('ABCD'.index(row['reference']) + 1) % 4]
            elif task == 'yesno':
                text = row['group'] if right else {'yes': 'no', 'no': 'yes'}[row['group']]
            else:
                text = EMOTION[row['group']] if right else EMOTION['ABCD'[('ABCD'.index(row['group']) + 1) % 4]]
            out.append(dict(id=row['id'], reference=row['messages'][1]['content'], prediction=text))
        path = folder / 'runs' / arm / task
        path.mkdir(parents=True)
        d2.write_jsonl(path / 'predictions.jsonl', out)
        d2.write(path / 'metrics.json', dict(predictions_sha256=d2.digest(path / 'predictions.jsonl'),
                                             generation=info['mcq_generation'] if task == 'mcq' else info['text_generation'],
                                             manifest_sha256=info['manifests'][task]['sha256']))


def test_selection_is_three_per_actor_and_class_of_the_four_classes_and_repeatable(tmp_path):
    audio = corpus(tmp_path)
    first, second = crosscorpus.select(audio), crosscorpus.select(audio)
    assert first == second and len(first) == 12 * 4 * 3
    assert {label for _, label, _ in first} == set('ABCD')
    assert all('_DIS_' not in p.name and '_FEA_' not in p.name for _, _, p in first)
    assert crosscorpus.select(audio, seed=7) != first


def test_manifests_ask_the_trained_question_under_four_orders(tmp_path):
    info = crosscorpus.build(corpus(tmp_path), PROTOCOL, tmp_path / 'out')
    rows = d2.read_jsonl(tmp_path / 'out' / 'mcq.jsonl')
    assert info['utterances'] == 144 and info['actors'] == 12 and len(rows) == 576
    first = [r for r in rows if r['rotation'] == 0]
    assert all(r['messages'][0]['content'] == '<audio>' + d2.load(PROTOCOL)['prompts']['question'] for r in first)
    assert all(r['reference'] == r['label'] for r in first)
    for row in rows:
        assert row['option_order']['ABCD'.index(row['reference'])] == row['label'] and row['dialogue'] == row['speaker']
    one = [r for r in rows if r['utterance'] == rows[0]['utterance']]
    assert sorted(r['reference'] for r in one) == list('ABCD')
    assert {n: info['manifests'][n]['rows'] for n in crosscorpus.TASKS} == dict(mcq=576, yesno=288, word=288, open=144)


def test_an_empty_folder_is_refused(tmp_path):
    (tmp_path / 'AudioWAV').mkdir()
    with pytest.raises(ValueError, match='No CREMA-D clips'):
        crosscorpus.build(tmp_path / 'AudioWAV', PROTOCOL, tmp_path / 'out')


def scenario(tmp_path, last4):
    folder = tmp_path / 'out'
    crosscorpus.build(corpus(tmp_path, actors=40), PROTOCOL, folder)
    answers(folder, 'frozen-base', lambda row, task: 0.5)
    for i, arm in enumerate(formats.FAMILIES['D-all-shuffled'] + formats.FAMILIES['A-all']):
        answers(folder, arm, lambda row, task: 0.8, seed=i + 1)
    for i, arm in enumerate(formats.FAMILIES['A-last4']):
        answers(folder, arm, last4, seed=i + 10)
    return crosscorpus.summarize(folder, resamples=400)


def test_readings_when_the_pattern_of_the_first_corpus_repeats(tmp_path):
    result = scenario(tmp_path, lambda row, task: (0.75 if row['rotation'] == 0 else 0.4) if task == 'mcq' else 0.5)
    assert result['readings']['full_depth_gain_on_unseen_corpus'] == 'transfers'
    assert result['readings']['last_layers_bind_to_option_order_on_unseen_corpus'] == 'replicated'
    assert result['readings']['last_layers_gain_is_format_specific_on_unseen_corpus'] == 'replicated'
    assert abs(result['table']['A-last4']['order_gap']['mean'] - 0.35) < 0.06
    assert abs(result['contrasts']['D-all-shuffled - frozen-base']['mcq_all_orders']['delta_pp'] - 30) < 6
    assert 'frozen-base' in crosscorpus.render(result)


def test_readings_when_the_last_layers_behave_like_full_depth(tmp_path):
    result = scenario(tmp_path, lambda row, task: 0.8)
    assert result['readings']['last_layers_bind_to_option_order_on_unseen_corpus'] == 'not replicated'
    assert result['readings']['last_layers_gain_is_format_specific_on_unseen_corpus'] == 'not replicated'


def test_a_small_gain_is_read_as_a_gap_and_a_loss_as_harm(tmp_path):
    folder = tmp_path / 'out'
    crosscorpus.build(corpus(tmp_path, actors=40), PROTOCOL, folder)
    answers(folder, 'frozen-base', lambda row, task: 0.5)
    for i, arm in enumerate(formats.FAMILIES['D-all-shuffled']):
        answers(folder, arm, lambda row, task: 0.53, seed=i + 1)
    assert crosscorpus.summarize(folder, resamples=400)['readings']['full_depth_gain_on_unseen_corpus'] == 'gap'
    table = {'frozen-base': {'mcq_all_orders': dict(mean=0.5)}}
    harmed = {'D-all-shuffled - frozen-base': {'mcq_all_orders': dict(delta_pp=-9.0, interval_95_pp=[-12.0, -6.0])}}
    assert crosscorpus.judge(table, harmed)['full_depth_gain_on_unseen_corpus'] == 'harmed'
    table['frozen-base']['mcq_all_orders']['mean'] = 0.9
    assert crosscorpus.judge(table, harmed)['full_depth_gain_on_unseen_corpus'] == 'base at ceiling'


def test_changed_predictions_are_refused(tmp_path):
    folder = tmp_path / 'out'
    crosscorpus.build(corpus(tmp_path), PROTOCOL, folder)
    answers(folder, 'frozen-base', lambda row, task: 0.5)
    path = folder / 'runs' / 'frozen-base' / 'mcq' / 'predictions.jsonl'
    path.write_text(path.read_text().replace('"A"', '"B"', 1))
    with pytest.raises(ValueError, match='predictions changed'):
        crosscorpus.summarize(folder, resamples=50)


def test_models_scored_with_different_batch_sizes_are_not_compared(tmp_path):
    folder = tmp_path / 'out'
    crosscorpus.build(corpus(tmp_path), PROTOCOL, folder)
    for arm, size in (('frozen-base', 2), ('A-all', 4)):
        answers(folder, arm, lambda row, task: 0.5)
        path = folder / 'runs' / arm / 'mcq' / 'metrics.json'
        d2.write(path, dict(d2.load(path), generation=dict(batch_size=size)))
    with pytest.raises(ValueError, match='different generation settings'):
        crosscorpus.summarize(folder, resamples=50)


def test_models_scored_with_different_free_text_settings_are_not_compared(tmp_path):
    """The free-text formats are compared too, so their decoding settings have to agree."""
    folder = tmp_path / 'out'
    crosscorpus.build(corpus(tmp_path), PROTOCOL, folder)
    for arm in ('frozen-base', 'A-all'):
        answers(folder, arm, lambda row, task: 0.5)
    path = folder / 'runs' / 'A-all' / 'word' / 'metrics.json'
    d2.write(path, dict(d2.load(path), generation=dict(formats.GENERATION, batch_size=6)))
    with pytest.raises(ValueError, match='different generation settings'):
        crosscorpus.summarize(folder, resamples=50)


def test_the_batch_size_is_fixed_at_build_time_and_changes_nothing_else(tmp_path):
    audio = corpus(tmp_path)
    usual = crosscorpus.build(audio, PROTOCOL, tmp_path / 'a')
    larger = crosscorpus.build(audio, PROTOCOL, tmp_path / 'b', batch_size=6)
    assert usual['mcq_generation'] == dict(batch_size=2) and usual['text_generation'] == formats.GENERATION
    assert larger['mcq_generation'] == dict(batch_size=6)
    assert larger['text_generation'] == dict(formats.GENERATION, batch_size=6) and formats.GENERATION['batch_size'] == 4
    assert larger['manifests'] == usual['manifests']
