"""Independent check of the letters.json numbers against the raw logits."""
"""Independent recomputation of the AQA numbers. Run from the repository root:

    python3 -I research/tools/<this file>.py

The run folders live outside the tracked tree (they are the server run folders copied in);
the tools find them by name under the repository root.
"""
import os
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[2]
RESULTS = REPO / 'experiments' / 'hearside_pilot' / 'results'
import json, os, pathlib
import numpy as np
from collections import defaultdict
RES = pathlib.Path(str(REPO / 'experiments' / 'hearside_pilot' / 'results'))
L = pathlib.Path(next(e.path for e in os.scandir(RES) if e.name.startswith('letters')))
LOGS = L/'logits'; CLASSES='ABCD'
def load(f):
    rows=[json.loads(l) for l in (f/'rows.jsonl').open()]
    return np.load(f/'logits.npy'), rows
def recall(rows, right, keep):
    hit=defaultdict(list)
    for r,ok in zip(rows,right):
        if keep(r): hit[r['label']].append(ok)
    return 100.0*float(np.mean([np.mean(v) for v in hit.values()]))
rec = json.loads((L/'summary'/'letters.json').read_text())
for split in ('session4','cremad'):
    print('==', split)
    R = rec['corpora'][split]['recall']
    for arm in ('frozen-base','A-all','A-first4','A-mid4','A-last4','A-last4-s43','A-last4-s44','D-all-shuffled'):
        folder = LOGS/arm/split
        if not folder.exists(): continue
        lg, rows = load(folder)
        right=[]
        for r, l in zip(rows, lg):
            slot = CLASSES[int(np.argmax(l))]
            right.append(r['option_order'][CLASSES.index(slot)]==r['label'])
        tr = recall(rows, right, lambda r: r['rotation']==0)
        ot = recall(rows, right, lambda r: r['rotation']!=0)
        rr = R[arm]
        print(f'   {arm:16s} train {tr:6.2f} vs recorded {rr["training_order"]:6.2f}   other {ot:6.2f} vs {rr["other_orders"]:6.2f}')
    # the surrogate
    for arm in ('surrogate',):
        folder = LOGS/arm/split
        if not folder.exists():
            print('   (no surrogate logits folder; surrogate computed from frozen-base + fitted bias)')
            continue
    # share of gain recovered, recomputed
    base = R['frozen-base']; sur = R['surrogate']; l4 = R['A-last4 (mean)']
    b_gain = sur['training_order']-base['training_order']
    l4_gain = l4['training_order']-base['training_order']
    b_loss = base['other_orders']-sur['other_orders']
    l4_loss = base['other_orders']-l4['other_orders']
    print(f'   share of training-order gain recovered: {b_gain/l4_gain:.4f}  (recorded {rec["corpora"][split]["share_of_gain_recovered"]["value"]:.4f})')
    print(f'   share of other-order loss reproduced:   {l4_loss/b_loss:.4f}')
