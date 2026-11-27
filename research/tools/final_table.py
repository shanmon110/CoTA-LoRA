"""Independent recomputation of the AQA numbers. Run from the repository root:

    python3 -I research/tools/<this file>.py

The run folders live outside the tracked tree (they are the server run folders copied in);
the tools find them by name under the repository root.
"""
import os
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[2]
RESULTS = REPO / 'experiments' / 'hearside_pilot' / 'results'
import json, numpy as np
d=json.load(open('out/recompute2.json'))
FAM=['AQ-first4','AQ-mid4','AQ-last4','AQ-all']
def seeds(f): return [f, f+'-s43', f+'-s44']
print('=== AQA MCQ: per-family, per-domain delta vs frozen base (seed-averaged)')
bd={k:v['mcq'] for k,v in d['by_domain']['frozen-base'].items()}
print(f'{"":12s} ' + ' '.join(f'{dom:>18s}' for dom in ('music','sound','speech')))
for f in FAM:
    row=[]
    for dom in ('music','sound','speech'):
        deltas=[d['by_domain'][a][dom]['mcq']-bd[dom] for a in seeds(f)]
        row.append(f'{np.mean(deltas):+7.2f} [{min(deltas):+5.2f},{max(deltas):+5.2f}]')
    print(f'{f:12s} ' + ' '.join(f'{r:>18s}' for r in row))
print()
print('=== AQA text (strict): per-family, per-domain delta vs frozen base (seed-averaged)')
bd={k:v['text_strict'] for k,v in d['by_domain']['frozen-base'].items()}
for f in FAM:
    row=[]
    for dom in ('music','sound','speech'):
        deltas=[d['by_domain'][a][dom]['text_strict']-bd[dom] for a in seeds(f)]
        row.append(f'{np.mean(deltas):+7.2f}')
    print(f'{f:12s} ' + ' '.join(f'{r:>18s}' for r in row))
print()
print('=== AQ-mid4 text speech, per seed (the +1.20 claim)')
for a in seeds('AQ-mid4'):
    print('   ',a, f'{d["by_domain"][a]["speech"]["text_strict"]-bd["speech"] if False else d["by_domain"][a]["speech"]["text_strict"]-d["by_domain"]["frozen-base"]["speech"]["text_strict"]:+.2f}')
print('=== AQ-all text sound, per seed (the -6.9 claim)')
for a in seeds('AQ-all'):
    print('   ',a, f'{d["by_domain"][a]["sound"]["text_strict"]-d["by_domain"]["frozen-base"]["sound"]["text_strict"]:+.2f}')
print()
print('=== split-half CV: mean held-out gain of the fitted rule vs the frozen base, per arm')
for f in FAM:
    for a in seeds(f):
        folds=d['split_half_cv'][a]
        g=[x['rule_accuracy']-x['base_accuracy'] for x in folds]
        never=sum(1 for x in folds if x['kept']==[])
        print(f'   {a:14s} mean gain {np.mean(g):+.2f} pp   folds choosing "never use the adapter": {never}/{len(folds)}   keys used: {[x["key"] for x in folds]}')
print()
print('=== oracle ceilings')
for f in FAM:
    for a in seeds(f):
        o=d['oracle'][a]
        print(f'   {a:14s} domain-gate ceiling {o["domain_gate_ceiling_pp"]:+.2f} pp   source-gate ceiling {o["source_gate_ceiling_pp"]:+.2f} pp   sources better/worse {o["sources_better"]}/{o["sources_worse"]}')
