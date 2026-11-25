"""When the option order changes, does a model keep the emotion or keep the letter?

Uses only predictions that already exist. For every utterance the trained question was asked under four
option orders. A model that reads the options picks the same emotion four times, hence four different
letters. A model that maps what it hears straight to the letter that emotion had during training picks
the same letter four times, hence four different emotions.

Written down on 2026-11-25 before this decomposition was computed (the aggregate scores were known):
the last-four-layer arm "follows a letter map" if, among its wrong answers under the three orders it was
not trained on, the share that is the letter the true emotion had in the training order is at least 50%
with a 95% interval above the 33.3% expected from errors spread evenly, and exceeds the full-depth arm's
share by at least 15 points with an interval above 0. Post hoc on data already seen in aggregate.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.hearside_pilot import crosscorpus
from experiments.trr_diagnostics import d2

CLASSES = 'ABCD'
CRITERIA = dict(min_share=0.5, chance=1 / 3, min_excess_pp=15.0)
ARMS = ('frozen-base', 'A-first4', 'A-mid4', 'A-last4', 'A-all', 'B-audio', 'D-all-shuffled')
MEASURES = ('letter_agreement', 'emotion_agreement', 'training_letter_share_of_errors', 'training_letter_rate')


def per_utterance(manifest, predictions):
    """Per utterance: [same-letter pairs, same-emotion pairs, errors off the training order, of which the training letter,
    training-letter answers off the training order], and its cluster."""
    rows = {}
    for row, given in zip(manifest, predictions, strict=True):
        if row['id'] != given['id']:
            raise ValueError('Predictions do not follow the manifest')
        letter = given['prediction'].strip()
        emotion = row['option_order'][CLASSES.index(letter)] if letter in CLASSES and letter else None
        rows.setdefault(row['utterance'], dict(cluster=row['dialogue'], label=row['label'], answers={}))['answers'][row['rotation']] = (letter, emotion)
    names, stats, clusters = [], [], []
    for name, item in rows.items():
        if sorted(item['answers']) != [0, 1, 2, 3]:
            raise ValueError(f'{name}: expected four option orders')
        letters, emotions = zip(*(item['answers'][k] for k in range(4)))
        pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
        other = [(letters[k], emotions[k]) for k in (1, 2, 3)]
        wrong = [letter for letter, emotion in other if emotion != item['label']]
        stats.append([sum(letters[i] == letters[j] for i, j in pairs), sum(emotions[i] == emotions[j] for i, j in pairs),
                      len(wrong), sum(letter == item['label'] for letter in wrong), sum(letter == item['label'] for letter, _ in other)])
        names.append(name)
        clusters.append(item['cluster'])
    return np.array(stats, dtype=float), clusters


def measures(totals):
    """From summed per-utterance counts (..., 6): the four measures. `totals[..., 5]` is the number of utterances."""
    return np.stack([totals[..., 0] / (6 * totals[..., 5]), totals[..., 1] / (6 * totals[..., 5]),
                     totals[..., 3] / np.maximum(totals[..., 2], 1), totals[..., 4] / (3 * totals[..., 5])], axis=-1)


def summarize(folder, *, resamples=10000, seed=42):
    folder = Path(folder)
    manifest = d2.read_jsonl(folder / 'mcq.jsonl')
    groups = {a: [m for m in crosscorpus.FAMILIES.get(a, [a]) if (folder / 'runs' / m / 'mcq' / 'predictions.jsonl').exists()] for a in ARMS}
    groups = {a: members for a, members in groups.items() if members}
    counts, clusters = {}, None
    for members in groups.values():
        for model in members:
            stats, clusters = per_utterance(manifest, d2.read_jsonl(folder / 'runs' / model / 'mcq' / 'predictions.jsonl'))
            counts[model] = np.concatenate([stats, np.ones((len(stats), 1))], axis=1)
    names = sorted(set(clusters))
    member = np.zeros((len(names), len(clusters)))
    member[[names.index(c) for c in clusters], np.arange(len(clusters))] = 1
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(names), np.full(len(names), 1 / len(names)), size=resamples)
    picks = rng.integers(0, 3, size=(resamples, 3))
    value, boot = {}, {}
    for arm, members in groups.items():
        by_cluster = np.stack([member @ counts[m] for m in members])                       # models, clusters, 6
        value[arm] = measures(by_cluster.sum(axis=1)).mean(axis=0)
        draws = measures(np.einsum('rk,mkc->mrc', weights, by_cluster))                    # models, resamples, 4
        chosen = picks[:, :len(members)] % len(members)
        boot[arm] = draws[chosen.T, np.arange(resamples)].mean(axis=0)
    def interval(x):
        return [float(v) for v in np.percentile(x, [2.5, 97.5])]
    table = {arm: {n: dict(value=float(value[arm][i]), interval_95=interval(boot[arm][:, i])) for i, n in enumerate(MEASURES)} for arm in groups}
    out = dict(folder=str(folder), utterances=len(clusters), models={a: m for a, m in groups.items()}, criteria=CRITERIA, table=table, reading=None)
    if 'A-last4' in groups and 'A-all' in groups:
        i = MEASURES.index('training_letter_share_of_errors')
        excess = boot['A-last4'][:, i] - boot['A-all'][:, i]
        share = table['A-last4']['training_letter_share_of_errors']
        out['last4_minus_all_share_pp'] = dict(delta=100 * float(value['A-last4'][i] - value['A-all'][i]), interval_95=[100 * v for v in interval(excess)])
        follows = (share['value'] >= CRITERIA['min_share'] and share['interval_95'][0] > CRITERIA['chance']
                   and out['last4_minus_all_share_pp']['delta'] >= CRITERIA['min_excess_pp'] and out['last4_minus_all_share_pp']['interval_95'][0] > 0)
        out['reading'] = 'the last-four-layer arm follows a letter map' if follows else 'not shown'
    return out


def render(results):
    lines = ['# Keeping the emotion or keeping the letter when the option order changes', '',
             'Agreement: share of the six pairs of option orders in which a model gave the same letter, or chose the same emotion. '
             'Training letter: the letter the true emotion had in the training order, counted under the three other orders; '
             'among errors, 33.3% is expected if errors were spread evenly. Percent, with 95% intervals over clusters and seeds.', '']
    for name, result in results.items():
        lines += [f"## {name} ({result['utterances']} utterances)", '',
                  '| model (seeds) | same letter | same emotion | training letter among errors | training letter, all answers |', '|---|---|---|---|---|']
        for arm, row in result['table'].items():
            lines.append(f"| {arm} ({len(result['models'][arm])}) | " + ' | '.join(
                f"{100 * row[n]['value']:.1f} [{100 * row[n]['interval_95'][0]:.1f}, {100 * row[n]['interval_95'][1]:.1f}]" for n in MEASURES) + ' |')
        if result['reading']:
            d = result['last4_minus_all_share_pp']
            lines += ['', f"A-last4 minus A-all, training letter among errors: {d['delta']:+.1f} pp [{d['interval_95'][0]:+.1f}, {d['interval_95'][1]:+.1f}]. "
                          f"Reading: **{result['reading']}**"]
        lines.append('')
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--folders', nargs='+', required=True, help='name=folder pairs of crosscorpus folders')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    results = {name: summarize(folder) for name, folder in (item.split('=', 1) for item in args.folders)}
    args.output.mkdir(parents=True, exist_ok=True)
    d2.write(args.output / 'conflict.json', results)
    (args.output / 'conflict.md').write_text(render(results))
    print(render(results))
