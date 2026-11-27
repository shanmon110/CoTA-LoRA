"""Does the label-free correction remove the adapter's gain, or only its order sensitivity?

The correction in `emotion_prior_all_arms.py` lifts the frozen base by 8.4 pp on Session 4 and
13.0 pp on CREMA-D, with no labels and no training.  The adapted arms keep a gap over the
corrected base of +9 to +13 pp on the arms that train many layers, but `A-last4` lands within
0.3 pp of the corrected base.

This script asks a sharper question.  The correction removes the model's *marginal* preference
over classes.  It says nothing about what happens when the option order changes.  So:

  1. the training-order score, where the adapter's gain was measured;
  2. the other-order score, where the adapter collapses;
  3. the order gap (training order minus other orders), before and after the correction.

If the correction removes the adapter's gain in the training order but leaves the collapse in
the other orders, then the adapter's behaviour is *not* a prior problem that a calibration can
fix -- there is a separate order-coupling mechanism.

Scores are computed per ROW (one row is one utterance under one option order), because the
training order is a property of the row, not of the utterance.  The prior is fitted on one
half of the clusters (dialogue for Session 4, actor for CREMA-D) and applied to the other
half, and the two halves are swapped.

Run from the repository root:

    python3 -I research/tools/prior_correction_by_order.py
"""
import json
import os
import pathlib

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
RESULTS = REPO / 'experiments' / 'hearside_pilot' / 'results'
L = pathlib.Path(next(e.path for e in os.scandir(RESULTS) if e.name.startswith('letters')))
LOGS = L / 'logits'
L4 = 'ABCD'


def logsm(z):
    m = z.max(1, keepdims=True)
    return z - m - np.log(np.exp(z - m).sum(1, keepdims=True))


def main():
    for split in ('session4', 'cremad'):
        print('=' * 106)
        print(f'== {split}: training order, other orders, and the order gap, before and after '
              f'the label-free correction')
        z0, rows = load('frozen-base', split)
        lab = np.array([L4.index(r['label']) for r in rows])
        order = np.array([[L4.index(c) for c in r['option_order']] for r in rows])
        rot = np.array([r['rotation'] for r in rows])
        clus = np.array([r['cluster'] for r in rows])
        cn = sorted(set(clus))
        ci = np.array([cn.index(c) for c in clus])
        half = np.array([c % 2 for c in ci])
        is_train = rot == 0

        def uar(pred, mask):
            """Class-averaged recall over the rows in `mask`."""
            return 100 * float(np.mean([
                (pred[mask & (lab == c)] == c).mean() for c in range(4)]))

        for model in sorted(p.name for p in LOGS.iterdir()):
            z = np.load(LOGS / model / split / 'logits.npy').astype(np.float64)
            lp = logsm(z)
            # (rows, emotions): the log-probability of the emotion each slot carries
            emo = np.zeros((len(rows), 4))
            for i in range(len(rows)):
                emo[i, order[i]] = lp[i]
            raw = emo.argmax(1)
            # the prior: one mean log-probability per emotion, per option order, from the
            # held-out half's rows
            cor = np.zeros(len(rows), int)
            for h in (0, 1):
                fit = half != h
                app = half == h
                # the prior: one mean log-probability per emotion, per option order, from the
                # held-out half's rows only
                prior = np.zeros((4, 4))
                for r in range(4):
                    m = fit & (rot == r)
                    if m.any():
                        prior[r] = emo[m].mean(0)
                # subtract the prior of each row's own option order
                corr = emo.copy()
                for r in range(4):
                    m = rot == r
                    if m.any():
                        corr[m] = emo[m] - prior[r]
                cor[app] = corr[app].argmax(1)
            raw_tr, raw_ot = uar(raw, is_train), uar(raw, ~is_train)
            cor_tr, cor_ot = uar(cor, is_train), uar(cor, ~is_train)
            print(f'   {model:16s} raw  training {raw_tr:5.1f}  other {raw_ot:5.1f}  '
                  f'gap {raw_tr - raw_ot:+6.1f}   |   corrected  training {cor_tr:5.1f}  '
                  f'other {cor_ot:5.1f}  gap {cor_tr - cor_ot:+6.1f}')


def load(model, split):
    z = np.load(LOGS / model / split / 'logits.npy').astype(np.float64)
    rows = [json.loads(line) for line in (LOGS / model / split / 'rows.jsonl').open()]
    return z, rows


if __name__ == '__main__':
    main()
