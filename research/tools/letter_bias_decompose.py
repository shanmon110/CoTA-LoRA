"""Decompose each arm's answers into a position component and a content component.

For every utterance the model answers one of the four display slots under each of the four
option orders.  Two readings of the same answer are possible:

  position   the slot (A/B/C/D) the model picks, independent of what the option says
  content    the emotion the picked option stands for

The frozen base is strongly position-biased (it prefers slot A).  The question this script
answers: if the model's slot preference were removed, would the arm's content accuracy
survive?  The removal is done with a per-slot additive bias fitted on one half of the
dialogues and applied to the other, so nothing is fitted on the data it is scored on.

This is a measurement of what a *decision-rule* correction could buy, not a proposal.
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
LETTERS = 'ABCD'


def load(folder):
    rows = [json.loads(line) for line in (folder / 'rows.jsonl').open()]
    return np.load(folder / 'logits.npy'), rows


def uar_by(rows, chosen_emotion):
    by = collections.defaultdict(list)
    for r, e in zip(rows, chosen_emotion):
        by[CANON[r['label']]].append(e == CANON[r['label']])
    return 100.0 * float(np.mean([np.mean(v) for v in by.values()]))


def slot_bias(lg, iters=300):
    """One additive bias per display slot, fitted so the chosen slot is uniform."""
    bias = np.zeros(4)
    for _ in range(iters):
        share = np.bincount((lg + bias).argmax(axis=1), minlength=4) / len(lg)
        bias -= 2.0 * (share - 0.25)
    return bias


for split in ('session4', 'cremad'):
    print('=' * 104)
    print(f'== {split}: the answer with the position preference removed (slot bias fitted on one half '
          f'of the dialogues, scored on the other)')
    for arm in sorted(p.name for p in LOGS.iterdir()):
        lg, rows = load(LOGS / arm / split)
        key = [r['cluster'] for r in rows]
        uniq = sorted(set(key))
        rng = np.random.default_rng(0)
        perm = rng.permutation(len(uniq))
        halves = [set(uniq[i] for i in perm[: len(uniq) // 2]), set(uniq[i] for i in perm[len(uniq) // 2:])]
        own, debiased = [], []
        for f, s in ((0, 1), (1, 0)):
            fit = np.array([i for i, k in enumerate(key) if k in halves[f]])
            sco = np.array([i for i, k in enumerate(key) if k in halves[s]])
            bias = slot_bias(lg[fit])
            slot = (lg[sco] + bias).argmax(axis=1)
            chosen = [CANON[rows[i]['option_order'][sl]] for i, sl in zip(sco, slot)]
            debiased.append(uar_by([rows[i] for i in sco], chosen))
            own.append(uar_by([rows[i] for i in sco],
                               [CANON[rows[i]['option_order'][int(np.argmax(lg[i]))]] for i in sco]))
        slots = np.bincount(lg.argmax(axis=1), minlength=4)
        print(f'   {arm:16s} slot share A B C D ' + ' '.join(f'{100 * s / slots.sum():4.1f}' for s in slots)
              + f'   UAR {np.mean(own):6.2f} -> de-biased {np.mean(debiased):6.2f} ({np.mean(debiased) - np.mean(own):+5.2f} pp)')
