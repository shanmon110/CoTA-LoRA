import numpy as np

from experiments.hearside_pilot import conflict, crosscorpus
from experiments.hearside_pilot.tests.test_crosscorpus import PROTOCOL, corpus
from experiments.trr_diagnostics import d2


def write(folder, arm, rule):
    rows = d2.read_jsonl(folder / 'mcq.jsonl')
    path = folder / 'runs' / arm / 'mcq'
    path.mkdir(parents=True)
    d2.write_jsonl(path / 'predictions.jsonl', [dict(id=r['id'], reference=r['reference'], prediction=rule(r)) for r in rows])


def test_an_option_reader_keeps_the_emotion_and_a_letter_map_keeps_the_letter(tmp_path):
    folder = tmp_path / 'out'
    crosscorpus.build(corpus(tmp_path, actors=30), PROTOCOL, folder)
    rng = np.random.default_rng(0)
    heard = {r['utterance']: (r['label'] if rng.random() < 0.7 else 'ABCD'[rng.integers(4)]) for r in d2.read_jsonl(folder / 'mcq.jsonl')}
    for arm in crosscorpus.FAMILIES['A-all']:
        write(folder, arm, lambda r: 'ABCD'[r['option_order'].index(heard[r['utterance']])])      # finds the heard emotion among the options
    for arm in crosscorpus.FAMILIES['A-last4']:
        write(folder, arm, lambda r: heard[r['utterance']])                                        # says the letter that emotion had in training
    result = conflict.summarize(folder, resamples=300)
    reader, mapper = result['table']['A-all'], result['table']['A-last4']
    assert reader['emotion_agreement']['value'] == 1 and reader['letter_agreement']['value'] == 0
    assert mapper['letter_agreement']['value'] == 1 and mapper['emotion_agreement']['value'] == 0
    assert mapper['training_letter_share_of_errors']['value'] > 0.7 and abs(reader['training_letter_share_of_errors']['value'] - 1 / 3) < 0.08   # an option reader's errors land there by chance
    assert abs(mapper['training_letter_rate']['value'] - 0.775) < 0.05
    assert result['reading'] == 'the last-four-layer arm follows a letter map' and 'A-last4' in conflict.render({'toy': result})


def test_no_reading_without_a_difference(tmp_path):
    folder = tmp_path / 'out'
    crosscorpus.build(corpus(tmp_path, actors=30), PROTOCOL, folder)
    rng = np.random.default_rng(1)
    for arm in crosscorpus.FAMILIES['A-all'] + crosscorpus.FAMILIES['A-last4']:
        write(folder, arm, lambda r: 'ABCD'[rng.integers(4)])
    result = conflict.summarize(folder, resamples=300)
    assert result['reading'] == 'not shown'
    assert abs(result['table']['A-last4']['training_letter_share_of_errors']['value'] - 1 / 3) < 0.05
    assert abs(result['table']['A-all']['letter_agreement']['value'] - 0.25) < 0.04
