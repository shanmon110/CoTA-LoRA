"""The label-free class-marginal correction, with intervals, applied to every arm.

This is the measurement that decides what the layer-band adapters actually learned.  The
correction is:

  1. score the four option letters and re-index them by the EMOTION each slot carries, so the
     score axis is comparable across option orders;
  2. for each option order separately, estimate the model's own marginal over classes from
     UNLABELLED utterances -- the mean log-probability of each emotion -- and subtract it;
  3. take the argmax.

No label is used, and no parameter is fitted on the training order.  The only external
knowledge is the option list, which is given in the prompt.

The prior is fitted on one half of the clusters (dialogue for Session 4, actor for CREMA-D)
and the OTHER half is scored, then the two halves are swapped; the reported score is the
average over both directions and over five random splits.  So the reported gain is never the
gain of a correction fitted on the rows it is scored on.

Reported for every arm: the score before and after the correction, the gain with a cluster
bootstrap interval, and the gap over the CORRECTED frozen base.  The gap is the number that
matters -- it is what the adapter is worth once the frozen base has had its class marginal
removed too.

Run from the repository root:

    python3 -I research/tools/label_free_calibration.py
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
BOOT = 2000
TRIALS = 5


def logsm(z):
    m = z.max(1, keepdims=True)
    return z - m - np.log(np.exp(z - m).sum(1, keepdims=True))


def load(model, split):
    z = np.load(LOGS / model / split / 'logits.npy').astype(np.float64)
    rows = [json.loads(line) for line in (LOGS / model / split / 'rows.jsonl').open()]
    return z, rows


def main():
    for split in ('session4', 'cremad'):
        print('=' * 112)
        print(f'== {split}: the label-free class-marginal correction, every arm '
              f'({TRIALS} cluster splits, both directions)')
        z0, rows = load('frozen-base', split)
        lab = np.array([L4.index(r['label']) for r in rows])
        order = np.array([[L4.index(c) for c in r['option_order']] for r in rows])
        rot = np.array([r['rotation'] for r in rows])
        clus = np.array([r['cluster'] for r in rows])
        cn = sorted(set(clus))
        ci = np.array([cn.index(c) for c in clus])
        nclust = len(cn)

        def uar(pred, rows_idx, mask, weights):
            """Class-averaged recall over `rows_idx` (the scored rows), with per-row weights.

            `mask` selects which of those rows to include; `weights` is indexed by the full
            row array and is sliced the same way."""
            lab_s = lab[rows_idx]
            w_s = weights[rows_idx]
            out = []
            for c in range(4):
                m = (lab_s == c) & mask
                tot = w_s[m].sum()
                out.append(float((w_s[m] * (pred[m] == c)).sum() / tot) if tot else 0.0)
            return 100 * float(np.mean(out))

        rng = np.random.default_rng(7)
        splits = []
        for _ in range(TRIALS):
            perm = rng.permutation(nclust)
            splits.append([set(perm[: nclust // 2]), set(perm[nclust // 2:])])

        results = {}
        for model in sorted(p.name for p in LOGS.iterdir()):
            z = np.load(LOGS / model / split / 'logits.npy').astype(np.float64)
            lp = logsm(z)
            emo = np.zeros((len(rows), 4))
            for i in range(len(rows)):
                emo[i, order[i]] = lp[i]
            raw = emo.argmax(1)
            # for each split and direction: the scored rows, their raw and corrected predictions
            pieces = []
            for h in splits:
                for f, s in ((0, 1), (1, 0)):
                    fit = np.isin(ci, list(h[f]))
                    app = np.isin(ci, list(h[s]))
                    cor = emo.copy()
                    for r in range(4):
                        m = fit & (rot == r)
                        if m.any():
                            cor[rot == r] = emo[rot == r] - emo[m].mean(0)
                    idx = np.where(app)[0]
                    pieces.append((idx, raw[idx].copy(), cor[idx].argmax(1)))
            ones = np.ones(len(rows))

            def mean_score(corrected, weights):
                """The score over the scored rows of every piece, weighted by the cluster weights."""
                vals = []
                for rows_idx, praw, pcor in pieces:
                    mask = np.ones(len(rows_idx), bool)
                    pred = pcor if corrected else praw
                    vals.append(uar(pred, rows_idx, mask, weights))
                return float(np.mean(vals))

            base_score = mean_score(0, ones)
            corr_score = mean_score(1, ones)
            gains = []
            for _ in range(BOOT):
                wc = np.bincount(rng.integers(0, nclust, nclust), minlength=nclust).astype(float)
                w = wc[ci]
                gains.append(mean_score(1, w) - mean_score(0, w))
            lo, hi = np.percentile(gains, [2.5, 97.5])
            results[model] = (base_score, corr_score, lo, hi)
            print(f'   {model:16s} raw {base_score:5.1f}   corrected {corr_score:5.1f}   '
                  f'gain {corr_score - base_score:+6.2f} [{lo:+.2f}, {hi:+.2f}]')
        bcorr = results['frozen-base'][1]
        braw = results['frozen-base'][0]
        print(f'   -- the gap over the frozen base, before and after calibration '
              f'(base {braw:.1f} -> {bcorr:.1f}):')
        for model, (b, c, lo, hi) in results.items():
            if model == 'frozen-base':
                continue
            print(f'      {model:16s} uncorrected gap {b - braw:+6.2f}   corrected gap '
                  f'{c - bcorr:+6.2f}   calibration gain {c - b:+6.2f}')


if __name__ == '__main__':
    main()
