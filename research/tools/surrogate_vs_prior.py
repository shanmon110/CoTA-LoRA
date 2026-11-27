"""Does the three-parameter letter-bias surrogate add anything on top of the label-free
class-prior correction?

The repository's surrogate adds one constant per letter to the frozen base's option logits,
fitted on the training order, and recovers 79% of the last-four-layers arm's training-order
gain.  The label-free correction subtracts the model's own per-order class prior, fitted on
one half of the clusters, and needs no labels.

If the surrogate and the correction are capturing the same thing, then after the correction
the surrogate adds nothing, and the "last four layers are mostly a letter bias" reading
reduces to "the last four layers learned a class prior".

The test fits both corrections on one half of the clusters and scores the other half:

  base                 the frozen base as it answers
  base + surrogate     the frozen base plus the three fitted letter constants
  base - prior         the frozen base minus its own per-order class prior
  base - prior + surr  both at once

Run from the repository root:

    python3 -I research/tools/surrogate_vs_prior.py
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


def load(model, split):
    z = np.load(LOGS / model / split / 'logits.npy').astype(np.float64)
    rows = [json.loads(line) for line in (LOGS / model / split / 'rows.jsonl').open()]
    return z, rows


def fit_surrogate(zt, rt):
    """The three-parameter letter bias, fitted on the training split as the repository does."""
    y = np.array([L4.index(r['label']) for r in rt])
    w = (len(y) / (4 * np.bincount(y, minlength=4)))[y]
    b = np.zeros(4)
    for _ in range(60):
        p = np.exp(logsm(zt + b))
        g = ((p - np.eye(4)[y]) * w[:, None]).mean(0)
        H = (np.einsum('n,ni,nj->ij', w, p, -p) + np.diag((w[:, None] * p).sum(0))) / len(y) \
            + 1e-6 * np.eye(4)
        b = b - np.linalg.solve(H, g)
        b -= b.mean()
    return b


def main():
    zt, rt = load('frozen-base', 'train')
    b = fit_surrogate(zt, rt)
    print('the surrogate constants (A B C D): ' + ' '.join(f'{v:+.2f}' for v in b))
    for split in ('session4', 'cremad'):
        print('=' * 100)
        print(f'== {split}')
        z, rows = load('frozen-base', split)
        lab = np.array([L4.index(r['label']) for r in rows])
        order = np.array([[L4.index(c) for c in r['option_order']] for r in rows])
        rot = np.array([r['rotation'] for r in rows])
        clus = np.array([r['cluster'] for r in rows])
        cn = sorted(set(clus))
        ci = np.array([cn.index(c) for c in clus])
        half = np.array([c % 2 for c in ci])
        is_train = rot == 0
        lp = logsm(z)
        emo = np.zeros((len(rows), 4))
        for i in range(len(rows)):
            emo[i, order[i]] = lp[i]
        # the surrogate adds a constant to a LETTER, so it must be applied in SLOT space,
        # before the re-indexing into emotion space
        emo_sur = np.zeros((len(rows), 4))
        lp_sur = logsm(z + b)
        for i in range(len(rows)):
            emo_sur[i, order[i]] = lp_sur[i]

        def uar(pred, mask):
            return 100 * float(np.mean([
                (pred[mask & (lab == c)] == c).mean() for c in range(4)]))

        preds = {k: np.zeros(len(rows), int) for k in
                 ('base', 'surrogate', 'prior', 'prior+surr')}
        for h in (0, 1):
            fit = half != h
            app = half == h
            preds['base'][app] = emo[app].argmax(1)
            preds['surrogate'][app] = emo_sur[app].argmax(1)
            prior = np.zeros((4, 4))
            for r in range(4):
                m = fit & (rot == r)
                if m.any():
                    prior[r] = emo[m].mean(0)
            corr = emo.copy()
            for r in range(4):
                m = rot == r
                corr[m] = emo[m] - prior[r]
            preds['prior'][app] = corr[app].argmax(1)
            ps = np.zeros((len(rows), 4))
            for i in range(len(rows)):
                ps[i, order[i]] = lp_sur[i]
            ps_corr = ps.copy()
            for r in range(4):
                m = rot == r
                ps_corr[m] = ps[m] - prior[r]
            preds['prior+surr'][app] = ps_corr[app].argmax(1)
        allk = np.ones(len(rows), bool)
        for name in ('base', 'surrogate', 'prior', 'prior+surr'):
            p = preds[name]
            print(f'   {name:12s} all {uar(p, allk):5.1f}   training {uar(p, is_train):5.1f}   '
                  f'other {uar(p, ~is_train):5.1f}')
        # and the last-four-layers arm, for reference
        for arm in ('A-last4', 'A-all'):
            z2, _ = load(arm, split)
            lp2 = logsm(z2)
            e2 = np.zeros((len(rows), 4))
            for i in range(len(rows)):
                e2[i, order[i]] = lp2[i]
            cor2 = np.zeros(len(rows), int)
            for h in (0, 1):
                fit = half != h
                app = half == h
                prior = np.zeros((4, 4))
                for r in range(4):
                    m = fit & (rot == r)
                    if m.any():
                        prior[r] = e2[m].mean(0)
                c2 = e2.copy()
                for r in range(4):
                    m = rot == r
                    c2[m] = e2[m] - prior[r]
                cor2[app] = c2[app].argmax(1)
            print(f'   {arm:12s} raw all {uar(e2.argmax(1), allk):5.1f}   '
                  f'corrected all {uar(cor2, allk):5.1f}   '
                  f'corrected training {uar(cor2, is_train):5.1f}   '
                  f'corrected other {uar(cor2, ~is_train):5.1f}')


if __name__ == '__main__':
    main()
