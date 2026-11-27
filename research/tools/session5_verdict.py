"""The Session 5 verdict, read off the frozen pre-declared criteria.

`experiments/hearside_pilot/session5.py analyze` applies the class-marginal correction to
every arm and reads the pre-declared criteria off the last-four-layers family.  This tool adds
the three things the pre-declaration (`research/notes-20261128-session5-confirmation.md`, §2)
asks for on top of that table, and nothing else:

  1. the reading for each SEED of the last-four-layers arm separately, not just the mean,
     because the pre-declaration says the mean alone is not allowed to carry the verdict;
  2. each arm's class marginal on Session 5, against the true marginal and against the frozen
     base's, so the mechanism claim can be checked directly and not only through the score;
  3. the per-order marginal, which is what the correction actually subtracts, and its
     total-variation distance to the true marginal and to the base's.

The criteria are imported from the analysis module, so they cannot drift from the ones that
were written down before the run.

Run from the repository root:

    python3 -I research/tools/session5_verdict.py <logits folder>
"""
import json
import sys
from pathlib import Path

import numpy as np

from experiments.hearside_pilot import letters
from experiments.hearside_pilot import session5 as S

CLASSES = 'ABCD'
EMOS = ['angry', 'happy', 'sad', 'neutral']


def logsm(z):
    m = z.max(axis=1, keepdims=True)
    return z - m - np.log(np.exp(z - m).sum(axis=1, keepdims=True))


def emotion_scores(logits, rows):
    """(rows, 4): the log-probability of the emotion each slot carries."""
    lp = logsm(logits.astype(np.float64))
    order = np.array([[CLASSES.index(c) for c in r['option_order']] for r in rows])
    out = np.zeros((len(rows), 4))
    for i in range(len(rows)):
        out[i, order[i]] = lp[i]
    return out


def marginal(scores, label, rot=None):
    """The soft mean probability per emotion, and the per-order soft means."""
    p = np.exp(scores)
    p = p / p.sum(axis=1, keepdims=True)
    per_order = [p[rot == r].mean(axis=0) for r in range(4)] if rot is not None else None
    return p.mean(axis=0), per_order


def tv(a, b):
    return 0.5 * float(np.abs(np.asarray(a) - np.asarray(b)).sum())


def main(root, split='session5'):
    root = Path(root)
    base_logits, base_rows = letters.load(root / 'frozen-base' / split)
    label = np.array([CLASSES.index(r['label']) for r in base_rows])
    rot = np.array([r['rotation'] for r in base_rows])
    cluster = np.array([r['cluster'] for r in base_rows])
    true_share = np.bincount(label, minlength=4) / len(label)

    arms = {}
    for folder in sorted(p for p in root.iterdir() if (p / split / 'done.json').exists()):
        logits, rows = letters.load(folder / split)
        arms[folder.name] = emotion_scores(logits, rows)

    entry = {name: S.analyse_arm(base_rows, label, rot, cluster, sc) for name, sc in arms.items()}
    base = entry['frozen-base']

    print('=' * 108)
    print(f'== {split}: the pre-declared criteria, read per seed')
    print(f"   criteria: parity |gap| < {S.CRITERIA['parity_pp']} pp and |order gap| < "
          f"{S.CRITERIA['order_gap_pp']} pp -> {S.CRITERIA['parity_reading']}; "
          f"gap < {S.CRITERIA['refuted_gap_pp']} pp -> {S.CRITERIA['refuted_reading']}")
    print(f"   frozen base: raw {base['raw']:.1f}, corrected {base['corrected']:.1f}, "
          f"order gap raw {base['order_gap_raw']:+.1f}, corrected {base['order_gap_corrected']:+.1f}")
    print()
    seeds = [m for m in ('A-last4', 'A-last4-s43', 'A-last4-s44') if m in entry]
    readings = []
    for name in seeds:
        reading = S.judge(entry, name, 'frozen-base')
        readings.append(reading['reading'])
        print(f"   {name:14s} raw {entry[name]['raw']:5.1f}  corrected {entry[name]['corrected']:5.1f}  "
              f"gap {reading['uncorrected_gap_pp']:+6.2f} -> {reading['corrected_gap_pp']:+6.2f} pp   "
              f"order gap {reading['uncorrected_order_gap_pp']:+6.2f} -> "
              f"{reading['corrected_order_gap_pp']:+6.2f} pp   -> {reading['reading']}")
    family = S.judge(entry, 'A-last4 (mean)' if 'A-last4 (mean)' in entry else seeds[0], 'frozen-base')
    print(f"   the family reading: {family['reading']}")
    if len(set(readings)) > 1:
        print(f"   the seeds disagree: {sorted(set(readings))}.  The family reading above is the "
              f"one the criteria are read off; the disagreement is reported as it stands.")

    print()
    print('=' * 108)
    print(f'== {split}: each arm\'s class marginal')
    base_soft, base_per = marginal(arms['frozen-base'], label, rot)
    for name in sorted(arms):
        soft, per = marginal(arms[name], label, rot)
        kl = float(np.sum(soft * np.log(np.maximum(soft, 1e-12) / true_share)))
        print(f"   {name:16s} soft marginal " + ' '.join(f'{100 * v:5.1f}' for v in soft)
              + '   true ' + ' '.join(f'{100 * v:5.1f}' for v in true_share)
              + f'   KL {kl:.4f}   TV from the base {tv(soft, base_soft):.3f}')
    print(f'   (the four columns are {", ".join(EMOS)})')
    print('   -- the PER-ORDER marginal, which is what the correction subtracts')
    for name in sorted(arms):
        soft, per = marginal(arms[name], label, rot)
        from_base = ' '.join(f'{tv(per[r], base_per[r]):.3f}' for r in range(4))
        from_true = ' '.join(f'{tv(per[r], true_share):.3f}' for r in range(4))
        print(f'   {name:16s} TV from the base per order {from_base}   '
              f'TV from the true marginal per order {from_true}')


if __name__ == '__main__':
    argv = sys.argv[1:]
    split = 'session5'
    if argv and argv[0].startswith('--split='):
        split = argv[0].split('=', 1)[1]
        argv = argv[1:]
    main(argv[0] if argv else
         Path('experiments/hearside_pilot/results/letters-20261125-170918/logits'), split)
