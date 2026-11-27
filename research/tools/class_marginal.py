"""Each arm's class marginal, against the truth and against the frozen base's.

The label-free correction works because the model's marginal over classes is wrong, and the
correction fixes it.  This script shows what the adapters did to that marginal.

For every arm it reports:

  predicted share   the share of answers that name each emotion
  soft mean         the model's mean predicted probability per emotion (the marginal itself)
  true              the share of the gold labels
  KL                the divergence between the model's soft marginal and the true marginal
  distance to base  the total-variation distance between this arm's marginal and the frozen
                    base's

The last line is the point.  If an adapter's marginal is close to the frozen base's, then the
adapter did not learn a new prior -- it kept the base's, and the reason it scores better in the
training order is that the base's prior is wrong in a way the training order happens to
reward.  If an adapter's marginal has moved a long way from the base's, it has learned
something new.

Run from the repository root:

    python3 -I research/tools/class_marginal.py
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
EMOS = ['angry', 'happy', 'sad', 'neutral']


def logsm(z):
    m = z.max(1, keepdims=True)
    return z - m - np.log(np.exp(z - m).sum(1, keepdims=True))


def marginal(folder):
    """(predicted share, soft mean) over the four emotions."""
    z = np.load(folder / 'logits.npy').astype(np.float64)
    rows = [json.loads(line) for line in (folder / 'rows.jsonl').open()]
    order = np.array([[L4.index(c) for c in r['option_order']] for r in rows])
    emo = np.zeros((len(rows), 4))
    lp = logsm(z)
    for i in range(len(rows)):
        emo[i, order[i]] = lp[i]
    p = np.exp(emo)
    p /= p.sum(axis=1, keepdims=True)
    pred = np.bincount(emo.argmax(axis=1), minlength=4) / len(rows)
    lab = np.array([L4.index(r['label']) for r in rows])
    true = np.bincount(lab, minlength=4) / len(rows)
    return pred, p.mean(axis=0), true


def tv(a, b):
    return 0.5 * float(np.abs(a - b).sum())


def per_order(folder):
    """The model's mean predicted probability per emotion, for each option order."""
    z = np.load(folder / 'logits.npy').astype(np.float64)
    rows = [json.loads(line) for line in (folder / 'rows.jsonl').open()]
    order = np.array([[L4.index(c) for c in r['option_order']] for r in rows])
    rot = np.array([r['rotation'] for r in rows])
    emo = np.zeros((len(rows), 4))
    lp = logsm(z)
    for i in range(len(rows)):
        emo[i, order[i]] = lp[i]
    p = np.exp(emo)
    p /= p.sum(axis=1, keepdims=True)
    out = []
    for r in range(4):
        m = rot == r
        out.append(p[m].mean(axis=0))
    return out


def _true_per_order(split):
    """The true class marginal within each option order (the label distribution is the same
    in every order, so this is the overall label distribution)."""
    folder = LOGS / 'frozen-base' / split
    rows = [json.loads(line) for line in (folder / 'rows.jsonl').open()]
    lab = np.array([L4.index(r['label']) for r in rows])
    share = np.bincount(lab, minlength=4) / len(rows)
    return [share] * 4


TRUE = None


def main():
    global TRUE
    for split in ('session4', 'cremad'):
        TRUE = _true_per_order(split)
        print('=' * 108)
        print(f'== {split}: each arm\'s class marginal')
        # the frozen base's own marginal is the reference
        base = marginal(LOGS / 'frozen-base' / split)[1]
        for arm in sorted(p.name for p in LOGS.iterdir()):
            folder = LOGS / arm / split
            pred, soft, true = marginal(folder)
            kl = float(np.sum(soft * np.log(np.maximum(soft, 1e-12) / true)))
            print(f'   {arm:16s} answers ' + ' '.join(f'{100 * v:5.1f}' for v in pred)
                  + '   soft marginal ' + ' '.join(f'{100 * v:5.1f}' for v in soft)
                  + '   true ' + ' '.join(f'{100 * v:5.1f}' for v in true)
                  + f'   KL {kl:.4f}   TV from the base {tv(soft, base):.3f}')
        print(f'   (the order of the four columns is {", ".join(EMOS)})')
        print('   -- the PER-ORDER marginal, which is what the correction actually subtracts')
        base_per = per_order(LOGS / 'frozen-base' / split)
        for arm in sorted(p.name for p in LOGS.iterdir()):
            per = per_order(LOGS / arm / split)
            # how far each order's marginal is from the base's, and from the true marginal
            from_base = [tv(per[r], base_per[r]) for r in range(4)]
            from_true = [tv(per[r], TRUE[r]) for r in range(4)]
            print(f'   {arm:16s} TV from the base per order '
                  + ' '.join(f'{v:.3f}' for v in from_base)
                  + '   TV from the true marginal per order '
                  + ' '.join(f'{v:.3f}' for v in from_true))


if __name__ == '__main__':
    main()
