"""The probe ceiling per corpus, and the gap to the model's answers.

The tracked probe artefacts hold, for every layer and both sites (audio, answer), a linear probe
fitted on the training split, selected on the dev split, and evaluated on session4 and cremad. The
layer used in the reported numbers was chosen on dev, which is an in-corpus split, so the
per-corpus ceiling can be larger than the reported probe.

For every model and site, this recomputes:
  best      the best layer for that corpus, and its recall
  reported  the recall at the layer chosen on dev
  answer    the model's own answer accuracy
so the gap between what is decodable and what the model outputs can be read per corpus.
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
import sys

import numpy as np

RES = str(REPO / 'experiments' / 'hearside_pilot' / 'results')
P = os.path.join(RES, next(e.name for e in os.scandir(RES) if e.name.startswith('probes')))
# the per-layer probe predictions live in the server run folder, not in the tracked copy
RUNP = str(REPO / 'hearside-probes-20261125-093959')
CROSS = str(REPO / 'hearside-cross-20261124-213539')
ANSWER_FOLDERS = {'session4': os.path.join(RUNP, 'session4'), 'cremad': os.path.join(CROSS, 'cremad')}
MODELS = ('frozen-base', 'A-all', 'A-last4', 'D-all-shuffled')


def macro(values, labels):
    by = collections.defaultdict(list)
    for v, l in zip(values, labels):
        by[l].append(v)
    return 100.0 * float(np.mean([np.mean(v) for v in by.values()]))


def answers(folder, model, ids):
    by = collections.defaultdict(list)
    for r in (json.loads(l) for l in open(os.path.join(folder, 'runs', model, 'mcq', 'predictions.jsonl'))):
        by[r['id'].rsplit('#', 1)[0]].append(r['prediction'].strip() == r['reference'])
    return np.array([np.mean(by[i]) for i in ids])


for split in ('session4', 'cremad'):
    print('=' * 104)
    print(f'== {split}')
    for model in MODELS:
        info = json.load(open(os.path.join(P, 'probes', model, 'probes.json')))
        correct = np.load(os.path.join(RUNP, 'probes', model, 'correct.npz'))
        rows = [json.loads(l) for l in open(os.path.join(RUNP, 'rows', model, f'{split}.json'))]
        ids = [r['id'] for r in rows]
        labels = np.array([r['label'] for r in rows])
        answer = macro(answers(ANSWER_FOLDERS[split], model, ids), labels)
        parts = []
        for site in ('audio', 'answer'):
            vals = [(int(k.split('/')[1]), v) for k, v in info['table'].items() if k.startswith(site + '/')]
            best_layer, best_entry = max(vals, key=lambda x: x[1][split])
            chosen = info['chosen_layer'][site]
            chosen_recall = macro(correct[f'{site}/{chosen}/{split}'].astype(float), labels)
            parts.append(f'{site}: best L{best_layer} {100 * best_entry[split]:.1f} | '
                         f'reported L{chosen} {chosen_recall:.1f}')
        print(f'   {model:16s} answer {answer:5.1f}   ' + '   '.join(parts))
