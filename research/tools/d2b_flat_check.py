"""The D2b per-emotion flattening deltas, from the run's own per-condition records.

The handoff quotes: angry -8.3 / happy +5.4 / sad +8.0 / neutral -9.8 for the frozen base.
This script recomputes them from the run folder, using the run's own per-condition per-emotion
recalls, so the numbers can be checked against the run rather than against my own re-implementation
of the option-slot mapping.
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
import glob
import json
import os

RUN = str(REPO / 'd2b-iemocap-s4-20261123-061428' / 'runs')
CANON = {'A': 'angry', 'B': 'happy', 'C': 'sad', 'D': 'neutral'}
for model in sorted(os.listdir(RUN)):
    folder = os.path.join(RUN, model)
    files = sorted(glob.glob(os.path.join(folder, '**', '*.jsonl'), recursive=True))
    if not files:
        files = sorted(glob.glob(os.path.join(folder, '**', '*.json'), recursive=True))
    print('==', model, len(files), 'files')
    by_cond = collections.defaultdict(list)
    for f in files:
        if f.endswith('.jsonl'):
            for line in open(f):
                r = json.loads(line)
                if 'condition' in r:
                    by_cond[r['condition']].append(r)
    if not by_cond:
        print('   (no per-row condition field)', files[:3])
        continue
    for cond in ('orig', 'sham', 'flat_f0', 'flat_energy', 'flat_both', 'neutral', 'level_f0', 'level_energy', 'level_both'):
        rows = by_cond.get(cond)
        if not rows:
            continue
        print('  ', cond, 'rows', len(rows), 'keys', sorted(rows[0].keys())[:14])
        break
