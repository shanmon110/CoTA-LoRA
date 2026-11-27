"""Independent recomputation of the AQA numbers. Run from the repository root:

    python3 -I research/tools/<this file>.py

The run folders live outside the tracked tree (they are the server run folders copied in);
the tools find them by name under the repository root.
"""
import os
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[2]
RESULTS = REPO / 'experiments' / 'hearside_pilot' / 'results'
import json, os, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from scipy_free import spearman
R=str(RESULTS / next(e.name for e in os.scandir(RESULTS) if e.name.startswith('aqa-')))
mcq={json.loads(l)['id']: json.loads(l) for l in open(R+'/mmau/mcq.jsonl')}
arms=[a for a in sorted(os.listdir(R+'/mmau/runs')) if a!='frozen-base']
dev=[]; mm=[]; names=[]
for a in arms:
    # dev accuracy from the training result
    best=None
    p=os.path.join(R,'train',a,'result.json')
    best=json.load(open(p)).get('best_accuracy') if os.path.exists(p) else None
    rows=[json.loads(l) for l in open(f'{R}/mmau/runs/{a}/mcq/predictions.jsonl')]
    acc=100*np.mean([bool(g['is_correct']) for g in rows])
    dev.append(best); mm.append(acc); names.append(a)
    print(f'{a:14s} dev {best:.4f}   MMAU(all) {acc:.2f}')
print()
print(f'Pearson  r = {np.corrcoef(dev, mm)[0,1]:.4f}   (n=12)')
print(f'Spearman r = {spearman(dev, mm):.4f}')
# also: MMAU original order only
mm0=[]
for a in arms:
    rows=[json.loads(l) for l in open(f'{R}/mmau/runs/{a}/mcq/predictions.jsonl')]
    keep=[bool(g['is_correct']) for g in rows if mcq[g['id']]['shift']==0]
    mm0.append(100*np.mean(keep))
print(f'Pearson (MMAU original order only) r = {np.corrcoef(dev, mm0)[0,1]:.4f}   Spearman {spearman(dev, mm0):.4f}')
# drop AQ-last4
idx=[i for i,n in enumerate(names) if 'last4' not in n]
print(f'without the AQ-last4 arms (n={len(idx)}): Pearson {np.corrcoef([dev[i] for i in idx],[mm[i] for i in idx])[0,1]:.4f}  Spearman {spearman([dev[i] for i in idx],[mm[i] for i in idx]):.4f}')
# by family: mean of 3 seeds
import collections
fam=collections.defaultdict(lambda: ([],[]))
for n,dv,mv in zip(names,dev,mm):
    f=n.rsplit('-s',1)[0] if '-s' in n else n
    fam[f][0].append(dv); fam[f][1].append(mv)
fd=[np.mean(v[0]) for v in fam.values()]; fm=[np.mean(v[1]) for v in fam.values()]
print('family means (4 families):')
for (f,(ds,ms)) in fam.items(): print(f'   {f:12s} dev {np.mean(ds):.4f}  MMAU {np.mean(ms):.2f}')
print(f'   Pearson r = {np.corrcoef(fd, fm)[0,1]:.4f}  Spearman {spearman(fd, fm):.4f}')
