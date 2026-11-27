import json
from pathlib import Path
from string import ascii_uppercase

import numpy as np

from experiments.hearside_pilot import aqa
from experiments.ms_swift_aqa.prompt import format_question
from experiments.trr_diagnostics import d2

BASE = Path(__file__).parents[2] / 'ms_swift_aqa' / 'omni3b-pilot.json'


def questions(path, n=240):
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        k = (2, 3, 4, 4, 4, 5)[i % 6]
        choices = [f'thing {i} {j}' for j in range(k)]
        gold = int(rng.integers(k))
        rows.append(dict(id=f'q{i}', question=f'What is heard in clip {i}?', choices=choices, audios=[f'/a/{i // 2}.wav'], domain='sound',
                         messages=[dict(role='user', content=format_question(f'What is heard in clip {i}?', choices)),
                                   dict(role='assistant', content=ascii_uppercase[gold])]))
    d2.write_jsonl(path, rows)


def answers(folder, arm, rule, seed):
    """`rule(row, task)`: probability of the right answer."""
    rng = np.random.default_rng(seed)
    for task in aqa.TASKS:
        out = []
        for row in d2.read_jsonl(folder / f'{task}.jsonl'):
            right = rng.random() < rule(row, task)
            wrong = next(c for c in row['choices'] if c != row['gold_text'])
            said = row['gold_text'] if right else wrong
            out.append(dict(id=row['id'], prediction=ascii_uppercase[row['choices'].index(said)] if task == 'mcq' else f'{said}.'))
        path = folder / 'runs' / arm / task
        path.mkdir(parents=True)
        d2.write_jsonl(path / 'predictions.jsonl', out)
        d2.write(path / 'metrics.json', dict(predictions_sha256=d2.digest(path / 'predictions.jsonl')))


def test_configs_differ_only_in_the_layers_and_the_seed(tmp_path):
    arms = aqa.arm_configs(BASE, tmp_path / 'configs')
    assert len(arms) == 12 and arms['AQ-mid4-s43'] == dict(config=str(tmp_path / 'configs' / 'AQ-mid4-s43.json'), seed=43, last_n=4, first_layer=16)
    base = json.loads(BASE.read_text())
    last, first = (json.loads((tmp_path / 'configs' / f'{name}.json').read_text()) for name in ('AQ-last4', 'AQ-first4-s44'))
    assert last['adapter'] == dict(rank=8, alpha=16, last_n=4, gate_multiplier=0.0, update_positions='all')
    assert first['adapter'] == dict(last['adapter'], first_layer=0) and first['training'] == dict(base['training'], seed=44, data_seed=44)
    assert last['training'] == base['training'] and 'methods' not in last and last['task'] == 'aqa'


def test_shifted_questions_keep_the_right_option_text(tmp_path):
    questions(tmp_path / 'test.jsonl', n=12)
    info = aqa.build(tmp_path / 'test.jsonl', tmp_path / 'out')
    rows = d2.read_jsonl(tmp_path / 'out' / 'mcq.jsonl')
    assert info['questions'] == 12 and info['audios'] == 6 and info['options'] == {'2': 2, '3': 2, '4': 6, '5': 2}
    assert len(rows) == 2 * 2 + 2 * 3 + 8 * 4
    for row in rows:
        letter = row['messages'][1]['content']
        assert row['choices'][ascii_uppercase.index(letter)] == row['gold_text'] and sorted(row['choices']) == sorted(set(row['choices']))
        assert row['messages'][0]['content'] == format_question(row['question'], row['choices'])
    four = [r for r in rows if r['source'] == 'q2']
    assert [r['shift'] for r in four] == [0, 1, 2, 3] and len({r['messages'][1]['content'] for r in four}) == 4
    text = d2.read_jsonl(tmp_path / 'out' / 'text.jsonl')[2]
    assert 'A.' not in text['messages'][0]['content'] and text['messages'][0]['content'].endswith('Reply with the text of the correct option.')


def test_answers_are_matched_to_options():
    row = dict(choices=['A dog barking', 'Rain', 'A dog'], gold_text='Rain')
    assert aqa.by_text(row, 'rain.') == (True, True) and aqa.by_text(row, 'It is rain') == (True, True)
    assert aqa.by_text(row, 'A dog barking') == (False, True)          # an exact match wins over an option it contains
    assert aqa.by_text(row, 'a dog barking in the rain') == (False, False) and aqa.by_text(row, 'thunder') == (False, False)
    assert aqa.by_letter(row, 'B') == (True, True) and aqa.by_letter(row, 'D') == (False, False) and aqa.by_letter(row, 'A') == (False, True)


def test_a_truncated_answer_names_the_option_it_starts():
    row = dict(choices=['No, it is not present.', 'Yes, it is present.'], gold_text='No, it is not present.')
    assert aqa.by_text(row, 'No') == (True, True)                    # the start of exactly one option
    assert aqa.by_text(row, 'Yes') == (False, True)
    # A prefix that starts two options at once names neither.
    both = dict(choices=['No, it is not present.', 'No, it is present.'], gold_text='No, it is not present.')
    assert aqa.by_text(both, 'No') == (False, False)
    # A prefix that contradicts nothing still names the wrong option when the wrong one is meant.
    assert aqa.by_text(row, 'No, it is not') == (True, True)


def test_a_prefix_has_to_end_at_a_word_boundary_and_be_long_enough():
    # "air" is the start of "airbased on the given audio ...", but only in the middle of a word.
    row = dict(choices=['desert', 'airbased on the given audio what is the most likely setting'], gold_text='desert')
    assert aqa.by_text(row, 'Air') == (False, False)
    # A single character is not read as naming an option.
    short = dict(choices=['A dog barking', 'Rain'], gold_text='Rain')
    assert aqa.by_text(short, 'R') == (False, False)


def scenario(tmp_path, last4):
    tmp_path.mkdir()
    questions(tmp_path / 'test.jsonl')
    folder = tmp_path / 'out'
    aqa.build(tmp_path / 'test.jsonl', folder)
    answers(folder, 'frozen-base', lambda row, task: 0.5, 0)
    for family, rule in (('AQ-all', lambda row, task: 0.8), ('AQ-first4', lambda row, task: 0.8), ('AQ-last4', last4)):
        for i, arm in enumerate(aqa.FAMILIES[family]):
            answers(folder, arm, rule, sum(map(ord, arm)) + i)
    return aqa.summarize(folder, resamples=300)


def test_readings(tmp_path):
    bound = scenario(tmp_path / 'a', lambda row, task: 0.8 if task == 'mcq' and row['shift'] == 0 else 0.5)
    assert bound['readings'] == dict(bands={'AQ-first4': 'transfers', 'AQ-last4': 'prompt-bound', 'AQ-all': 'transfers'}, claim_on_second_task='holds')
    assert abs(bound['table']['AQ-last4']['order_gap']['mean'] - 0.3) < 0.08 and 'AQ-last4' in aqa.render(bound)
    assert bound['table']['AQ-all']['same_option_under_all_orders'] > bound['table']['frozen-base']['same_option_under_all_orders']
    same = scenario(tmp_path / 'b', lambda row, task: 0.8)
    assert same['readings']['claim_on_second_task'] == 'does not extend'
    nothing = scenario(tmp_path / 'c', lambda row, task: 0.5)
    assert nothing['readings']['bands']['AQ-last4'] == 'no gain' and nothing['readings']['claim_on_second_task'] == 'holds'
