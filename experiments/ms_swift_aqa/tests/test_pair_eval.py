import json

import pytest

from experiments.ms_swift_aqa import pair_eval
from experiments.ms_swift_aqa.prompt import format_question


def write_jsonl(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    return path


def member(pair, index, kind, choices, gold, source_ids):
    identifier = f'{pair}-{index}'
    question = 'What happened first?'
    return {
        'id': identifier,
        'pair_id': pair,
        'pair_kind': kind,
        'pair_member': index,
        'source_ids': source_ids,
        'question': question,
        'choices': choices,
        'domain': 'timing',
        'audios': [f'{identifier}.wav'],
        'messages': [
            {'role': 'user', 'content': format_question(question, choices)},
            {'role': 'assistant', 'content': gold},
        ],
    }


def rows_for(pair, gold='AB', sources=('recording',)):
    kind = 'change' if gold[0] != gold[1] else 'invariant'
    return [
        member(pair, index, kind, ['First', 'Second'], letter, list(sources))
        for index, letter in enumerate(gold)
    ]


def predictions_for(rows, letters):
    return [
        {'id': row['id'], 'prediction': letter, 'is_correct': letter == row['messages'][1]['content']}
        for row, letter in zip(rows, letters, strict=True)
    ]


def test_measures_both_correct_separately_from_prediction_flips(tmp_path):
    rows = rows_for('a') + rows_for('b', 'AA') + rows_for('c', 'AB')
    predictions = predictions_for(rows, 'ABBAAA')
    metrics = pair_eval.measure_pairs(
        write_jsonl(tmp_path / 'manifest.jsonl', rows),
        write_jsonl(tmp_path / 'predictions.jsonl', predictions[::-1]),
    )

    assert metrics['overall']['pairs'] == 3
    assert metrics['overall']['both_correct'] == 1
    assert metrics['overall']['individual_accuracy'] == pytest.approx(4 / 6)
    assert metrics['by_kind']['change']['both_correct_rate'] == .5
    assert metrics['by_kind']['change']['flip_rate'] == .5
    assert metrics['by_kind']['invariant']['flip_rate'] == 1.
    assert metrics['by_kind']['invariant']['both_correct_rate'] == 0.
    assert metrics['by_domain']['timing'] == metrics['overall']


@pytest.mark.parametrize('defect', ['incomplete', 'duplicate', 'extra', 'wrong_label', 'invalid_letter'])
def test_rejects_invalid_prediction_coverage_or_score(defect, tmp_path):
    rows = rows_for('a')
    predictions = predictions_for(rows, 'AB')
    if defect == 'incomplete':
        predictions.pop()
    if defect == 'duplicate':
        predictions.append(predictions[0].copy())
    if defect == 'extra':
        predictions[-1]['id'] = 'unseen'
    if defect == 'wrong_label':
        predictions[-1]['is_correct'] = False
    if defect == 'invalid_letter':
        predictions[-1]['prediction'] = 'C'

    with pytest.raises(ValueError):
        pair_eval.measure_pairs(
            write_jsonl(tmp_path / 'manifest.jsonl', rows),
            write_jsonl(tmp_path / 'predictions.jsonl', predictions),
        )


@pytest.mark.parametrize('defect', ['kind', 'choices', 'domain', 'audio', 'member', 'pair_size'])
def test_rejects_malformed_pair(defect, tmp_path):
    rows = rows_for('a')
    if defect == 'kind':
        rows[1]['pair_kind'] = 'invariant'
        rows[0]['pair_kind'] = 'invariant'
    if defect == 'choices':
        rows[1]['choices'].reverse()
        rows[1]['messages'][0]['content'] = format_question(rows[1]['question'], rows[1]['choices'])
    if defect == 'domain':
        rows[1]['domain'] = 'speech'
    if defect == 'audio':
        rows[1]['audios'] = rows[0]['audios'].copy()
    if defect == 'member':
        rows[1]['pair_member'] = 0
    if defect == 'pair_size':
        rows[1]['pair_id'] = 'b'

    with pytest.raises(ValueError):
        pair_eval.measure_pairs(
            write_jsonl(tmp_path / 'manifest.jsonl', rows),
            write_jsonl(tmp_path / 'predictions.jsonl', predictions_for(rows, 'AB')),
        )


def test_paired_bootstrap_groups_pairs_with_shared_sources(tmp_path):
    rows = (
        rows_for('a', sources=('same',))
        + rows_for('b', sources=('same', 'adjacent'))
        + rows_for('c', sources=('adjacent',))
        + rows_for('d', sources=('other',))
    )
    paths = (
        write_jsonl(tmp_path / 'manifest.jsonl', rows),
        write_jsonl(tmp_path / 'left.jsonl', predictions_for(rows, 'ABABABAA')),
        write_jsonl(tmp_path / 'right.jsonl', predictions_for(rows, 'AAAAAAAB')),
    )
    metrics = pair_eval.compare_pairs(*paths, draws=2000, seed=42)
    assert metrics == pair_eval.compare_pairs(*paths, draws=2000, seed=42)
    assert metrics['overall']['clusters'] == 2
    assert metrics['overall']['delta_pp'] == 50.
    assert metrics['overall']['interval_95_pp'] == [-100., 100.]
    assert metrics['overall']['candidate_only_correct'] == 3
    assert metrics['overall']['reference_only_correct'] == 1
    assert metrics['by_kind']['change'] == metrics['overall']
