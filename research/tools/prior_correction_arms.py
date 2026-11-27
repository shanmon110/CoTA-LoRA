"""Does the label-free class-prior correction change the trained arms?

For every arm whose letter logits are tracked, the frozen base and the three seeds of A-last4,
A-all and so on, the same correction is applied: one additive bias per emotion, fitted so that the
predicted emotion distribution is uniform, fitted on one half of the clusters and scored on the
other. Both macro recall (UAR) and micro accuracy are reported, because a prior correction that
only moves the answer distribution can improve one and not the other.
"""
"""Independent recomputation of the AQA numbers. Run from the repository root:

    python3 -I research/tools/<this file>.py

The run folders live outside the tracked tree (they are the server run folders copied in);
the tools find them by name under the repository root.
"""
import os
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[2]
RESULTS = REPO / 'experiments' / 'hearside_pilot' / 'results'
import collections
import json
import os
import pathlib
import sys

import numpy as np

RES = pathlib.Path(str(REPO / 'experiments' / 'hearside_pilot' / 'results'))
L = pathlib.Path(next(e.path for e in os.scandir(RES) if e.name.startswith('letters')))
LOGS = L / 'logits'
CANON = {'A': 'angry', 'B': 'happy', 'C': 'sad', 'D': 'neutral'}
EMOS = ['angry', 'happy', 'sad', 'neutral']


def load(folder):
    rows = [json.loads(line) for line in (folder / 'rows.jsonl').open()]
    return np.load(folder / 'logits.npy'), rows


def macro(rows, chosen):
    """Mean per-class recall over the four utterances of an emotion.

    The rows repeat each utterance once per option rotation, so the rows are first collapsed to
    one row per utterance (by utterance id), with the class being the emotion of the canonical
    gold letter.
    """
    by_utt = {}
    for r, e in zip(rows, chosen):
        by_utt.setdefault(r['id'].rsplit('#', 1)[0], []).append(EMOS[e] == CANON[r['label']])
    per_class = collections.defaultdict(list)
    for r in rows:
        per_class[CANON[r['label']]].append(np.mean(by_utt[r['id'].rsplit('#', 1)[0]]))
    return 100.0 * float(np.mean([np.mean(v) for v in per_class.values()]))


def micro(rows, chosen):
    return 100.0 * float(np.mean([EMOS[e] == CANON[r['label']] for r, e in zip(rows, chosen)]))


def emo_scores(lg, rows):
    out = np.zeros((len(rows), 4))
    for i, r in enumerate(rows):
        for slot, canonical in enumerate(r['option_order']):
            out[i, EMOS.index(CANON[canonical])] = lg[i][slot]
    return out


def fit_bias(scores, target, iters=400):
    bias = np.zeros(scores.shape[1])
    for _ in range(iters):
        share = np.bincount((scores + bias).argmax(axis=1), minlength=scores.shape[1]) / len(scores)
        bias -= 2.0 * (share - target)
    return bias


target = np.full(4, 0.25)
for split in ('session4', 'cremad'):
    print('=' * 100)
    print(f'== {split}: label-free uniform-prior correction, fitted on one half of the clusters, scored on the other')
    for arm in sorted(p.name for p in (LOGS).iterdir()):
        folder = LOGS / arm / split
        if not folder.exists():
            continue
        lg, rows = load(folder)
        scores = emo_scores(lg, rows)
        slot = np.array([int(np.argmax(l)) for l in lg])
        base = [EMOS.index(CANON[r['option_order'][s]]) for r, s in zip(rows, slot)]
        key = [r['cluster'] for r in rows]
        uniq = sorted(set(key))
        rng = np.random.default_rng(0)
        perm = rng.permutation(len(uniq))
        halves = [set(uniq[i] for i in perm[: len(uniq) // 2]), set(uniq[i] for i in perm[len(uniq) // 2:])]
        before_m, before_a, after_m, after_a = [], [], [], []
        for fit_idx, score_idx in ((0, 1), (1, 0)):
            fit = np.array([i for i, k in enumerate(key) if k in halves[fit_idx]])
            sco = np.array([i for i, k in enumerate(key) if k in halves[score_idx]])
            bias = fit_bias(scores[fit], target)
            chosen = [int(i) for i in (scores[sco] + bias).argmax(axis=1)]
            before_m.append(macro([rows[i] for i in sco], [base[i] for i in sco]))
            before_a.append(micro([rows[i] for i in sco], [base[i] for i in sco]))
            after_m.append(macro([rows[i] for i in sco], chosen))
            after_a.append(micro([rows[i] for i in sco], chosen))
        print(f'   {arm:18s} UAR {np.mean(before_m):6.2f} -> {np.mean(after_m):6.2f} ({np.mean(after_m) - np.mean(before_m):+5.2f})   '
              f'acc {np.mean(before_a):6.2f} -> {np.mean(after_a):6.2f} ({np.mean(after_a) - np.mean(before_a):+5.2f})')
