"""Label-free class-prior correction on the frozen base, with a held-out split-half protocol.

The correction: one additive bias per emotion, fitted so that the predicted emotion distribution
matches a target prior. Three target priors, none of which uses a label on the test side:

  uniform   the four classes are equally likely (true for both corpora)
  self      the model's own answer distribution on the *training* split
  inferred the class prior inferred from the model's own answers, by the standard EM for a
            mixture: P(class) is estimated from the model's soft predictions, so no external
            prior is needed at all

The bias is fitted on one half of the utterances (by dialogue or actor cluster) and scored on the
other half, so the reported gain is not the gain of a correction fitted on the data it is scored
on. A cluster bootstrap gives the interval.
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
CANON2EMO = {v: k for k, v in CANON.items()}
EMOS = ['angry', 'happy', 'sad', 'neutral']


def load(folder):
    rows = [json.loads(line) for line in (folder / 'rows.jsonl').open()]
    return np.load(folder / 'logits.npy'), rows


def uar(rows, chosen_emotion):
    by = collections.defaultdict(list)
    for r, e in zip(rows, chosen_emotion):
        by[CANON[r['label']]].append(e == CANON[r['label']])
    return 100.0 * float(np.mean([np.mean(v) for v in by.values()]))


def emotion_scores(lg, rows):
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


def fit_prior_em(scores, iters=300):
    """The class prior implied by the model's own soft predictions, by EM on a mixture."""
    z = scores - scores.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    prior = p.mean(axis=0)
    for _ in range(iters):
        w = p * prior / np.maximum((p * prior).sum(axis=1, keepdims=True), 1e-12)
        prior = w.mean(axis=0)
    return prior


for split in ('session4', 'cremad'):
    lg, rows = load(LOGS / 'frozen-base' / split)
    scores = emotion_scores(lg, rows)
    slot = np.array([int(np.argmax(l)) for l in lg])
    base = [CANON[r['option_order'][s]] for r, s in zip(rows, slot)]
    print('=' * 92)
    print(f'== {split}: frozen base UAR {uar(rows, base):.2f}   rows {len(rows)}')

    tlg, trows = load(LOGS / 'frozen-base' / 'train')
    tslot = [int(np.argmax(l)) for l in tlg]
    tdist = collections.Counter(CANON[r['option_order'][s]] for r, s in zip(trows, tslot))
    self_prior = np.array([tdist[e] / len(trows) for e in EMOS])
    em_prior = fit_prior_em(emotion_scores(tlg, trows))
    print(f'   the model\'s own training-split answer prior: {np.round(self_prior, 3).tolist()}')
    print(f'   the prior implied by the model\'s soft predictions: {np.round(em_prior, 3).tolist()}')

    key = [r['cluster'] for r in rows]
    uniq = sorted(set(key))
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(uniq))
    halves = [set(uniq[i] for i in perm[: len(uniq) // 2]), set(uniq[i] for i in perm[len(uniq) // 2:])]

    for name, target in (('uniform', np.full(4, 0.25)), ('self', self_prior), ('EM (no external prior)', em_prior)):
        gains = []
        for fit_idx, score_idx in ((0, 1), (1, 0)):
            fit = np.array([i for i, k in enumerate(key) if k in halves[fit_idx]])
            sco = np.array([i for i, k in enumerate(key) if k in halves[score_idx]])
            bias = fit_bias(scores[fit], target)
            chosen = [EMOS[i] for i in (scores[sco] + bias).argmax(axis=1)]
            before = uar([rows[i] for i in sco], [base[i] for i in sco])
            after = uar([rows[i] for i in sco], chosen)
            gains.append(after - before)
        # a cluster bootstrap over the whole split, with the bias refitted inside each draw
        clusters = sorted(set(key))
        bres = []
        for _ in range(600):
            pick = rng.choice(len(clusters), size=len(clusters), replace=True)
            idx = []
            for c in pick:
                idx.extend([i for i, k in enumerate(key) if k == clusters[c]])
            if not idx:
                continue
            bias = fit_bias(scores[idx], target)
            chosen = [EMOS[i] for i in (scores[idx] + bias).argmax(axis=1)]
            bres.append(uar([rows[i] for i in idx], chosen) - uar([rows[i] for i in idx], [base[i] for i in idx]))
        lo, hi = np.percentile(bres, [2.5, 97.5])
        print(f'   {name:24s}: split-half gain {np.mean(gains):+.2f} pp '
              f'[{min(gains):+.2f}, {max(gains):+.2f}]   bootstrap {np.mean(bres):+.2f} [{lo:+.2f}, {hi:+.2f}]')
