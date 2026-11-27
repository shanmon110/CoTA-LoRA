"""What the arms answer: the emotion (content) or the slot (position)?

For every utterance the model answers under four option orders.  Two readings of the same
answer are available:

  content    the emotion the chosen option stands for
  position   the display slot (A/B/C/D) the model picks

This script reports, for every arm and corpus:

  content accuracy        the fraction of answers that name the gold emotion
  same emotion            the fraction of utterances answered with the same emotion under all
                          four option orders
  same slot               the fraction answered with the same slot under all four orders
  slot share per order     how the position preference moves with the option order

The last line is the decisive one for the letter-bias reading: if the preference were a fixed
position bias, the preferred slot would be the same under every order.  If it is a bias by
CLASS, the preferred slot rotates with the class.

Together with `letter_bias_decompose.py` (removing the slot preference by a fitted per-slot
bias) and `prior_correction_all_arms.py` (removing the class preference) this separates the
two components of an adapter's behaviour.
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


for split in ('session4', 'cremad'):
    print('=' * 104)
    print(f'== {split}')
    for arm in sorted(p.name for p in LOGS.iterdir()):
        lg, rows = load(LOGS / arm / split)
        pick = lg.argmax(axis=1)
        content = [CANON[rows[i]['option_order'][p]] == CANON[rows[i]['label']] for i, p in enumerate(pick)]
        by_emotion, by_slot = collections.defaultdict(dict), collections.defaultdict(dict)
        for i, r in enumerate(rows):
            key = r['id'].rsplit('#', 1)[0]
            by_emotion[key][r['rotation']] = CANON[r['option_order'][pick[i]]]
            by_slot[key][r['rotation']] = pick[i]
        same_emotion = float(np.mean([len(set(v.values())) == 1 for v in by_emotion.values()]))
        same_slot = float(np.mean([len(set(v.values())) == 1 for v in by_slot.values()]))
        print(f'   {arm:16s} content accuracy {100 * float(np.mean(content)):5.1f}   '
              f'same emotion under all orders {100 * same_emotion:5.1f}%   same slot {100 * same_slot:5.1f}%')
        shares = []
        for rot in sorted({r['rotation'] for r in rows}):
            idx = [i for i, r in enumerate(rows) if r['rotation'] == rot]
            slots = np.bincount(pick[idx], minlength=4)
            shares.append('r%d ' % rot + ' '.join(f'{100 * s / slots.sum():4.1f}' for s in slots))
        print(f'                     slot share A B C D per order: ' + ' | '.join(shares))
