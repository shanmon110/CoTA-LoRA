"""Is the option-order sensitivity a SLOT preference or a CLASS preference?

This is the decisive test for the "letter bias" reading of the last-four-layers adapter.

The corpus is a fixed four-class emotion task, so the option order is a rotation of the class
list: under rotation r, class (j + r) mod 4 sits in slot j.  Two readings of the same
behaviour are possible:

  slot-anchored    the model prefers the same display SLOT (A/B/C/D) under every order.  Then
                   the bias is on the letter, and a letter-bias surrogate can reproduce it.
  class-anchored   the model prefers the same CLASS under every order.  Then the bias is on
                   the emotion the option carries, and a letter-bias surrogate CANNOT reproduce
                   it -- the surrogate only adds a constant per letter, so it can only produce a
                   slot-anchored preference.

The surrogate's fitted constants are the same three free parameters the repository's
letter-bias experiment uses.  The test is therefore: does the surrogate reproduce the arm's
per-order slot shares?  If the arm is class-anchored, the surrogate's slot shares will be
uniform while the arm's will be sharply peaked, and the surrogate will fail to reproduce the
collapse.

Run from the repository root:

    python3 -I research/tools/slot_vs_class_anchor.py
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


def load(folder):
    rows = [json.loads(line) for line in (folder / 'rows.jsonl').open()]
    return np.load(folder / 'logits.npy'), rows


def fit_surrogate(base_lg, base_rows, target_lg, target_rows):
    """The repository's three-parameter letter-bias surrogate: the frozen base's four option
    logits each get a constant (summing to zero), fitted by class-weighted likelihood on the
    training order."""
    def emotion_slot_scores(lg, rows):
        """(rows, 4): the logit of the slot carrying each emotion."""
        out = np.zeros((len(rows), 4))
        for i, r in enumerate(rows):
            for slot, canonical in enumerate(r['option_order']):
                out[i, EMOS.index(CANON[canonical])] = lg[i][slot]
        return out

    X = emotion_slot_scores(base_lg, base_rows)
    y = np.array([EMOS.index(CANON[r['label']]) for r in base_rows])
    weights = np.ones(len(y))
    # class-weighted likelihood: each class carries the same total weight
    for c in range(4):
        m = y == c
        if m.any():
            weights[m] = len(y) / (4 * m.sum())
    bias = np.zeros(4)
    for _ in range(2000):
        z = X + bias
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        onehot = np.zeros_like(p)
        onehot[np.arange(len(y)), y] = 1
        g = (p - onehot) * weights[:, None]
        bias -= 0.5 * g.mean(axis=0)
        bias -= bias.mean()
    return bias


def slot_shares(lg, rows):
    """The share of answers per display slot, per option order and overall."""
    pick = lg.argmax(axis=1)
    out = {}
    for rot in sorted({r['rotation'] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r['rotation'] == rot]
        s = np.bincount(pick[idx], minlength=4)
        out[rot] = s / s.sum()
    s = np.bincount(pick, minlength=4)
    out['all'] = s / s.sum()
    return out


def class_shares(lg, rows):
    pick = lg.argmax(axis=1)
    out = {}
    for rot in sorted({r['rotation'] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r['rotation'] == rot]
        c = collections.Counter(CANON[rows[i]['option_order'][pick[i]]] for i in idx)
        out[rot] = np.array([c[e] / len(idx) for e in EMOS])
    c = collections.Counter(CANON[rows[i]['option_order'][pick[i]]] for i in range(len(rows)))
    out['all'] = np.array([c[e] / len(rows) for e in EMOS])
    return out


blg, brows = load(LOGS / 'frozen-base' / 'train')
surrogate_bias = fit_surrogate(blg, brows, None, None)
print('the surrogate constants (A B C D slots), fitted on the training split: '
      + ' '.join(f'{v:+.2f}' for v in surrogate_bias))

for split in ('session4', 'cremad'):
    print('=' * 108)
    print(f'== {split}: the arm\'s slot shares against the surrogate\'s slot shares')
    base_lg, base_rows = load(LOGS / 'frozen-base' / split)
    for arm in sorted(p.name for p in LOGS.iterdir()):
        lg, rows = load(LOGS / arm / split)
        arm_slots = slot_shares(lg, rows)
        arm_classes = class_shares(lg, rows)
        if arm == 'frozen-base':
            sur_slots = arm_slots
            sur_classes = arm_classes
        else:
            sur_lg = base_lg + surrogate_bias
            sur_slots = slot_shares(sur_lg, base_rows)
            sur_classes = class_shares(sur_lg, base_rows)
        rots = sorted(k for k in arm_slots if k != 'all')
        print(f'   {arm:16s}')
        for rot in rots:
            print(f'      r{rot}  slot  arm ' + ' '.join(f'{100 * v:5.1f}' for v in arm_slots[rot])
                  + '   surrogate ' + ' '.join(f'{100 * v:5.1f}' for v in sur_slots[rot])
                  + '   | class  arm ' + ' '.join(f'{100 * v:5.1f}' for v in arm_classes[rot])
                  + '   surrogate ' + ' '.join(f'{100 * v:5.1f}' for v in sur_classes[rot]))
        # how peaked is the arm's slot preference, and the surrogate's?
        peak_arm = max(100 * arm_slots[r].max() for r in rots)
        peak_sur = max(100 * sur_slots[r].max() for r in rots)
        print(f'      peak slot share under any order: arm {peak_arm:.1f}%   surrogate {peak_sur:.1f}%')
