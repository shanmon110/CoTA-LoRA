"""The label-free class-prior correction applied to every arm, with three target priors.

This is the measurement behind the plan's P1 item: Peng & Nuchged (arXiv 2608.06409) correct
the option logits with an offset derived from the marginal option preference of an unlabelled
batch, towards a uniform target prior.  The same family of correction is applied here to every
arm whose answer-letter logits are tracked, so that the correction can be read as a BASELINE,
not as a contribution.

Three target priors, none of which uses a label on the test side:

  uniform    the four classes are equally likely (true for both corpora) -- this is the prior
             Peng & Nuchged assume, and it is external information
  train-prior
             the class balance of the training split (IEMOCAP Sessions 1-2), which is known at
             deployment time for a fixed-label task
  self-soft  the model's own soft prediction distribution on the scored rows, which needs no
             external information at all

The bias is fitted on one half of the clusters (dialogue for Session 4, actor for CREMA-D) and
scored on the other, so the reported gain is not the gain of a correction fitted on the data it
is scored on.  Several split trials are averaged, because the split itself moves the answer.

Read this together with `prior_em_noext.py` (the same correction on the frozen base only) and
`decision_rule_gap.py` (whether the arm's gain is a decision-rule artefact).
"""
import collections
import json
import os
import pathlib

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
RESULTS = REPO / 'experiments' / 'hearside_pilot' / 'results'
L = pathlib.Path(next(e.path for e in os.scandir(RESULTS) if e.name.startswith('letters')))
LOGS = L / 'logits'
CANON = {'A': 'angry', 'B': 'happy', 'C': 'sad', 'D': 'neutral'}
EMOS = ['angry', 'happy', 'sad', 'neutral']
TRIALS = 5


def load(folder):
    rows = [json.loads(line) for line in (folder / 'rows.jsonl').open()]
    return np.load(folder / 'logits.npy'), rows


def emotion_scores(lg, rows):
    """(rows, 4): the logit of the display slot that carries each emotion."""
    out = np.zeros((len(rows), 4))
    for i, r in enumerate(rows):
        for slot, canonical in enumerate(r['option_order']):
            out[i, EMOS.index(CANON[canonical])] = lg[i][slot]
    return out


def fit_bias(scores, target, iters=400):
    """One additive bias per class, so the predicted class distribution matches the target."""
    bias = np.zeros(scores.shape[1])
    for _ in range(iters):
        share = np.bincount((scores + bias).argmax(axis=1), minlength=scores.shape[1]) / len(scores)
        bias -= 2.0 * (share - target)
    return bias


def uar(rows, chosen_emotion):
    by = collections.defaultdict(list)
    for r, e in zip(rows, chosen_emotion):
        by[CANON[r['label']]].append(e == CANON[r['label']])
    return 100.0 * float(np.mean([np.mean(v) for v in by.values()]))


tlg, trows = load(LOGS / 'frozen-base' / 'train')
tg = collections.Counter(CANON[r['label']] for r in trows)
train_gold = np.array([tg[e] / len(trows) for e in EMOS])
print('the training split class balance: ' + str({e: round(v, 3) for e, v in zip(EMOS, train_gold)}))

for split in ('session4', 'cremad'):
    print('=' * 104)
    print(f'== {split}: the class-prior correction applied to every arm '
          f'(mean over {TRIALS} split trials, each fitted on one half and scored on the other)')
    for arm in sorted(p.name for p in LOGS.iterdir()):
        lg, rows = load(LOGS / arm / split)
        scores = emotion_scores(lg, rows)
        slot = np.array([int(np.argmax(l)) for l in lg])
        base = [EMOS.index(CANON[r['option_order'][s]]) for r, s in zip(rows, slot)]
        key = [r['cluster'] for r in rows]
        uniq = sorted(set(key))
        out = {}
        for name, target in (('train-prior', train_gold), ('uniform', np.full(4, 0.25)), ('self-soft', None)):
            gains = []
            for trial in range(TRIALS):
                rng = np.random.default_rng(300 + trial)
                perm = rng.permutation(len(uniq))
                halves = [set(uniq[i] for i in perm[: len(uniq) // 2]),
                          set(uniq[i] for i in perm[len(uniq) // 2:])]
                if target is None:
                    z = scores - scores.max(axis=1, keepdims=True)
                    p = np.exp(z)
                    p /= p.sum(axis=1, keepdims=True)
                    target = p.mean(axis=0)
                for f, s in ((0, 1), (1, 0)):
                    fit = np.array([i for i, k in enumerate(key) if k in halves[f]])
                    sco = np.array([i for i, k in enumerate(key) if k in halves[s]])
                    bias = fit_bias(scores[fit], target)
                    chosen = (scores[sco] + bias).argmax(axis=1)
                    gains.append(uar([rows[i] for i in sco], [EMOS[c] for c in chosen])
                                 - uar([rows[i] for i in sco], [EMOS[base[i]] for i in sco]))
            out[name] = round(float(np.mean(gains)), 2)
        print(f'   {arm:16s} {out}')
