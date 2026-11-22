"""Score matched audio questions from native AQA prediction files."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from string import ascii_uppercase

import numpy as np

from .evaluate import correct_letters, read_rows
from .summarize_pilot import read_jsonl


def _sources(row):
    sources = row.get('source_ids', [])
    primary = row.get('source_id')
    if not isinstance(sources, list) or any(not isinstance(s, str) or not s for s in sources):
        raise ValueError(f"{row['id']}: source_ids must contain nonempty strings")
    if primary is not None and (not isinstance(primary, str) or not primary):
        raise ValueError(f"{row['id']}: source_id must be a nonempty string")
    return set(sources) | ({primary} if primary else set())


def _cluster_pairs(pairs):
    """Join pairs with shared recordings, including through intermediate pairs."""
    parent = {pair_id: pair_id for pair_id in pairs}
    owner = {}

    def representative(pair_id):
        while pair_id != parent[pair_id]:
            pair_id = parent[pair_id]
        return pair_id

    for pair_id, members in sorted(pairs.items()):
        sources = _sources(members[0]) | _sources(members[1])
        for source in sorted(sources):
            left = representative(pair_id)
            right = representative(owner.setdefault(source, pair_id))
            parent[max(left, right)] = min(left, right)
    return {pair_id: representative(pair_id) for pair_id in pairs}


def _pair_groups(manifest):
    groups = defaultdict(list)
    for item in read_rows(manifest):
        if not isinstance(item['id'], str):
            raise ValueError(f"{item['id']!r}: expected a string question ID")
        pair_id = item.get('pair_id')
        if not isinstance(pair_id, str) or not pair_id:
            raise ValueError(f"{item['id']}: expected a nonempty pair_id")
        if item.get('pair_kind') not in ('change', 'invariant'):
            raise ValueError(f"{item['id']}: expected change or invariant as pair_kind")
        domain = item.get('domain')
        if domain is not None and (not isinstance(domain, str) or not domain):
            raise ValueError(f"{item['id']}: invalid domain")
        groups[pair_id].append(item)

    for pair_id, items in groups.items():
        if len(items) != 2:
            raise ValueError(f'{pair_id}: expected two items in a pair')
        if any('pair_member' in item for item in items):
            members = [item.get('pair_member') for item in items]
            if any(type(member) is not int for member in members) or set(members) != {0, 1}:
                raise ValueError(f'{pair_id}: pair_member must be 0 and 1')
            items.sort(key=lambda item: item['pair_member'])
        first, second = items
        if any(first.get(k) != second.get(k) for k in ('question', 'choices', 'domain', 'pair_kind')):
            raise ValueError(f'{pair_id}: question, ordered options, domain and kind must match')
        if first['audios'] == second['audios']:
            raise ValueError(f'{pair_id}: expected different recordings')
        gold_changes = correct_letters(first) != correct_letters(second)
        if gold_changes != (first['pair_kind'] == 'change'):
            raise ValueError(f'{pair_id}: gold answers disagree with pair_kind')
    return groups


def _prediction_index(predictions, pairs):
    indexed = {}
    for row in read_jsonl(Path(predictions)):
        identifier = row.get('id')
        if not isinstance(identifier, str) or not identifier or identifier in indexed:
            raise ValueError(f'Invalid or duplicate prediction ID {identifier!r}')
        indexed[identifier] = row
    expected = {item['id'] for items in pairs.values() for item in items}
    if indexed.keys() != expected:
        missing = sorted(expected - indexed.keys())
        extra = sorted(indexed.keys() - expected)
        raise ValueError(f'Prediction IDs differ from paired manifest, missing {missing[:3]}, extra {extra[:3]}')
    return indexed


def _score_member(item, prediction):
    identifier = item['id']
    letter = prediction.get('prediction')
    legal = ascii_uppercase[:len(item['choices'])]
    if not isinstance(letter, str) or len(letter) != 1 or letter not in legal:
        raise ValueError(f'{identifier}: invalid predicted option {letter!r}')
    gold = correct_letters(item)
    answer_text = item['choices'][legal.index(letter)]
    correct = letter in gold
    reference = item['messages'][1]['content']
    if (prediction.get('is_correct') is not correct or prediction.get('reference', reference) != reference
            or prediction.get('correct_letters', gold) != gold
            or prediction.get('answer_text', answer_text) != answer_text):
        raise ValueError(f'{identifier}: prediction score or metadata differs from manifest')
    return answer_text, correct


def _outcomes(pairs, clusters, predictions):
    indexed = _prediction_index(predictions, pairs)
    outcomes = {}
    for pair_id, items in sorted(pairs.items()):
        answers, correct = zip(*(_score_member(item, indexed[item['id']]) for item in items))
        outcomes[pair_id] = dict(
            pair_kind=items[0]['pair_kind'], domain=items[0].get('domain'),
            cluster=clusters[pair_id], individual_correct=sum(correct), both_correct=all(correct),
            flipped=answers[0] != answers[1],
        )
    return outcomes


def _scores(rows):
    n = len(rows)
    both_correct = sum(row['both_correct'] for row in rows)
    individual_correct = sum(row['individual_correct'] for row in rows)
    flips = sum(row['flipped'] for row in rows)
    return dict(pairs=n, both_correct=both_correct, both_correct_rate=both_correct / n,
                individual_correct=individual_correct, individual_accuracy=individual_correct / (2 * n),
                flips=flips, flip_rate=flips / n)


def _group(outcomes, field):
    groups = defaultdict(list)
    for pair_id, row in outcomes.items():
        if row[field] is not None:
            groups[row[field]].append(pair_id)
    return dict(sorted(groups.items()))


def measure_pairs(manifest, predictions):
    """Summarize individual accuracy, pair accuracy and flips on validated recordings."""
    pairs = _pair_groups(manifest)
    outcomes = _outcomes(pairs, _cluster_pairs(pairs), predictions)
    by_kind = {}
    by_kind_and_domain = {}
    for kind, ids in _group(outcomes, 'pair_kind').items():
        members = {pair_id: outcomes[pair_id] for pair_id in ids}
        by_kind[kind] = _scores(list(members.values()))
        by_kind_and_domain[kind] = {
            domain: _scores([members[pair_id] for pair_id in domain_ids])
            for domain, domain_ids in _group(members, 'domain').items()
        }
    return dict(
        overall=_scores(list(outcomes.values())),
        by_kind=by_kind,
        by_domain={k: _scores([outcomes[p] for p in ids]) for k, ids in _group(outcomes, 'domain').items()},
        by_kind_and_domain=by_kind_and_domain,
        outcomes=outcomes,
    )


def _paired_interval(candidate, reference, ids, draws, seed):
    delta = np.asarray([int(candidate[k]['both_correct']) - int(reference[k]['both_correct']) for k in ids])
    _, index = np.unique([candidate[k]['cluster'] for k in ids], return_inverse=True)
    size = np.bincount(index)
    net = np.bincount(index, weights=delta)
    rng = np.random.default_rng(seed)
    samples = np.empty(draws)
    for start in range(0, draws, 256):
        sample = samples[start:start + 256]
        selected = rng.integers(0, len(size), size=(len(sample), len(size)))
        sample[:] = 100 * net[selected].sum(axis=1) / size[selected].sum(axis=1)
    return dict(pairs=len(ids), clusters=len(size), resamples=draws, seed=seed, delta_pp=float(100 * delta.mean()),
                interval_95_pp=np.quantile(samples, [0.025, 0.975]).tolist(),
                candidate_only_correct=int((delta == 1).sum()), reference_only_correct=int((delta == -1).sum()))


def compare_pairs(manifest, candidate_predictions, reference_predictions, *, draws=10000, seed=42):
    """Resample recording clusters for the paired difference in both-correct rates."""
    if type(draws) is not int or draws < 1:
        raise ValueError('Expected a positive number of bootstrap resamples')
    pairs = _pair_groups(manifest)
    clusters = _cluster_pairs(pairs)
    candidate = _outcomes(pairs, clusters, candidate_predictions)
    reference = _outcomes(pairs, clusters, reference_predictions)
    return dict(
        overall=_paired_interval(candidate, reference, list(candidate), draws, seed),
        by_kind={k: _paired_interval(candidate, reference, ids, draws, seed)
                 for k, ids in _group(candidate, 'pair_kind').items()},
        by_domain={k: _paired_interval(candidate, reference, ids, draws, seed)
                   for k, ids in _group(candidate, 'domain').items()},
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    args = parser.parse_args()
    result = measure_pairs(args.manifest, args.predictions)
    if args.reference is not None:
        result['versus_reference'] = compare_pairs(args.manifest, args.predictions, args.reference)
    print(json.dumps(result, indent=2, ensure_ascii=False))
