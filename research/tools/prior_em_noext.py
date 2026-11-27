"""Does the frozen base's class prior survive WITHOUT any external class prior?

The handoff claims an emotion-prior subtraction that needs no labels and only a handful of
unlabelled clips.  This script asks the strongest form of that question:

  1. Can the class prior be read off the AUDIO alone, with no labels and no external
     information?  The frozen base's own audio-site linear probe is exactly such a reader:
     it was trained on IEMOCAP Sessions 1-2 (with labels) and applied to Session 4 and
     CREMA-D, so it estimates P(emotion | audio) without any test-side label.
     Its predicted distribution is compared with the gold distribution.
  2. If that distribution is usable as a target prior, does correcting the answer logits
     towards it improve accuracy?  Split-half: fit on one half of the clusters, score on
     the other.

The probe artefacts are per-item correct flags, so only the probe's own predicted class
distribution is available from them; the probe's accuracy (not its per-item predictions)
is what the tracked artefacts keep.  What is tracked is `correct.npz`, so this script
uses the PROBE ACCURACY to bound the question, and uses the base's own soft prediction
distribution as the label-free prior estimate.
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


def uar(rows, chosen):
    by = collections.defaultdict(list)
    for r, e in zip(rows, chosen):
        by[CANON[r['label']]].append(e == CANON[r['label']])
    return 100.0 * float(np.mean([np.mean(v) for v in by.values()]))


for split in ('session4', 'cremad'):
    lg, rows = load(LOGS / 'frozen-base' / split)
    scores = emotion_scores(lg, rows)
    slot = np.array([int(np.argmax(l)) for l in lg])
    base = [CANON[r['option_order'][s]] for r, s in zip(rows, slot)]
    print('=' * 96)
    print(f'== {split}: frozen base, UAR as answered {uar(rows, base):.2f}   rows {len(rows)}')
    gold = collections.Counter(CANON[r['label']] for r in rows)
    print('   gold distribution:      ' + str({e: round(100 * gold[e] / len(rows), 1) for e in EMOS}))
    ans = collections.Counter(base)
    print('   answer distribution:    ' + str({e: round(100 * ans[e] / len(rows), 1) for e in EMOS}))
    # the model's own soft prediction distribution on the same rows (label free, no external prior)
    z = scores - scores.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    soft = p.mean(axis=0)
    print('   soft-prediction mean:   ' + str({e: round(100 * soft[i], 1) for i, e in enumerate(EMOS)}))

    key = [r['cluster'] for r in rows]
    uniq = sorted(set(key))
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(uniq))
    halves = [set(uniq[i] for i in perm[: len(uniq) // 2]), set(uniq[i] for i in perm[len(uniq) // 2:])]

    for name, target in (('uniform (needs an external class prior)', np.full(4, 0.25)),
                         ("the model's own soft distribution (no external prior)", soft)):
        gains = []
        for fit_idx, score_idx in ((0, 1), (1, 0)):
            fit = np.array([i for i, k in enumerate(key) if k in halves[fit_idx]])
            sco = np.array([i for i, k in enumerate(key) if k in halves[score_idx]])
            bias = fit_bias(scores[fit], target)
            chosen = [EMOS[i] for i in (scores[sco] + bias).argmax(axis=1)]
            gains.append(uar([rows[i] for i in sco], chosen) - uar([rows[i] for i in sco], [base[i] for i in sco]))
        print(f'   {name:52s} split-half gain {np.mean(gains):+.2f} pp  [{min(gains):+.2f}, {max(gains):+.2f}]')
