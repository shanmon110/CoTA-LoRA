"""Is each arm's gain a decision-rule correction, or a change of the representation?

Peng & Nuchged (arXiv 2608.06409) split a model's shortfall into a decision-rule gap (the
option logits could be read better by an affine rule) and a readout-coverage gap (the hidden
state holds more than the answer logits expose).  This script asks the same question of each
trained arm, using only the tracked letter logits:

  A. the arm's own answers                       (what the arm emits)
  B. an affine rule on the arm's OWN option logits, fitted on one half of the clusters and
     scored on the other                             (is the arm's logit readout miscalibrated?)
  C. the frozen base's answer logits with the arm's affine rule transplanted (not possible for
     the arms without a base logit pair, so not run)

B > A means the arm's gain is partly a calibration artefact of its own decision rule.
B close to A means the arm changed the representation, not just the readout.
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

import numpy as np

RES = RESULTS
L = pathlib.Path(next(e.path for e in os.scandir(RES) if e.name.startswith('letters')))
LOGS = L / 'logits'
CANON = {'A': 'angry', 'B': 'happy', 'C': 'sad', 'D': 'neutral'}
EMOS = ['angry', 'happy', 'sad', 'neutral']


def load(folder):
    rows = [json.loads(line) for line in (folder / 'rows.jsonl').open()]
    return np.load(folder / 'logits.npy'), rows


def emo_scores(lg, rows):
    """(rows, 4): the logit of the display slot that carries each emotion."""
    out = np.zeros((len(rows), 4))
    for i, r in enumerate(rows):
        for slot, canonical in enumerate(r['option_order']):
            out[i, EMOS.index(CANON[canonical])] = lg[i][slot]
    return out


def fit_affine(scores, y, iters=3000, lr=0.5):
    """A multinomial logistic rule on the four option logits, fitted by gradient descent."""
    W = np.zeros((scores.shape[1], 4))
    b = np.zeros(4)
    for _ in range(iters):
        z = scores @ W + b
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        onehot = np.zeros_like(p)
        onehot[np.arange(len(y)), y] = 1
        g = p - onehot
        W -= lr * (scores.T @ g) / len(y)
        b -= lr * g.mean(axis=0)
    return W, b


def uar(rows, chosen):
    by = collections.defaultdict(list)
    for r, e in zip(rows, chosen):
        by[CANON[r['label']]].append(e == CANON[r['label']])
    return 100.0 * float(np.mean([np.mean(v) for v in by.values()]))


for split in ('session4', 'cremad'):
    print('=' * 100)
    print(f'== {split}')
    for arm in sorted(p.name for p in LOGS.iterdir()):
        lg, rows = load(LOGS / arm / split)
        scores = emo_scores(lg, rows)
        slot = np.array([int(np.argmax(l)) for l in lg])
        base = [EMOS.index(CANON[r['option_order'][s]]) for r, s in zip(rows, slot)]
        y = np.array([EMOS.index(CANON[r['label']]) for r in rows])
        key = [r['cluster'] for r in rows]
        uniq = sorted(set(key))
        rng = np.random.default_rng(0)
        perm = rng.permutation(len(uniq))
        halves = [set(uniq[i] for i in perm[: len(uniq) // 2]), set(uniq[i] for i in perm[len(uniq) // 2:])]
        after = []
        for f, s in ((0, 1), (1, 0)):
            fit = np.array([i for i, k in enumerate(key) if k in halves[f]])
            sco = np.array([i for i, k in enumerate(key) if k in halves[s]])
            W, b = fit_affine(scores[fit], y[fit])
            chosen = [(scores[sco] @ W + b).argmax(axis=1)]
            after.append(uar([rows[i] for i in sco], [EMOS[c] for c in chosen[0]]))
        # the arm's own answers, on the same rows
        own = []
        for s in range(2):
            sco = np.array([i for i, k in enumerate(key) if k in halves[1 - s]])
            own.append(uar([rows[i] for i in sco], [EMOS[base[i]] for i in sco]))
        print(f'   {arm:16s} own answers {np.mean(own):6.2f}   affine rule on its own logits {np.mean(after):6.2f}'
              f'   delta {np.mean(after) - np.mean(own):+5.2f} pp')
