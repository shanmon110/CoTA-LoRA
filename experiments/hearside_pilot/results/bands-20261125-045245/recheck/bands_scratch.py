import json, sys, re, collections, os, random
import numpy as np
B, P = sys.argv[1], sys.argv[2]
def load(p): return [json.loads(l) for l in open(p)]
for corpus in ('session4', 'cremad'):
    mcq = {r['id']: r for r in load(f'{B}/{corpus}/mcq.jsonl')}
    runs = sorted(r for r in os.listdir(f'{B}/{corpus}/runs') if not r.startswith('recheck'))
    fam = collections.defaultdict(list)
    for r in runs: fam[re.sub(r'-s4[34]$', '', r)].append(r)
    ok = {r: {p['id']: p['prediction'].strip() == p['reference'] for p in load(f'{B}/{corpus}/runs/{r}/mcq/predictions.jsonl')} for r in runs}
    def uar(run, ids):
        by = collections.defaultdict(list)
        for i in ids: by[mcq[i]['label']].append(ok[run][i])
        return 100 * sum(sum(v) / len(v) for v in by.values()) / len(by)
    tr = [i for i, r in mcq.items() if r['rotation'] == 0]; ot = [i for i, r in mcq.items() if r['rotation'] != 0]
    print('==', corpus, len(runs), 'models')
    for k in ('frozen-base', 'A-first4', 'A-mid4', 'A-last4', 'A-all'):
        per = [(uar(m, tr), uar(m, ot)) for m in fam[k]]
        print(f'  {k:12s} train {np.mean([a for a,_ in per]):.1f} other {np.mean([b for _,b in per]):.1f} gap {np.mean([a-b for a,b in per]):+.1f}  per-seed other', [round(b,1) for _,b in per], 'gap', [round(a-b,1) for a,b in per])
# probes
S = json.load(open(f'{P}/summary/probes.json'))
for split in ('session4', 'cremad'):
    for m in ('frozen-base', 'A-all'):
        info = json.load(open(f'{P}/probes/{m}/probes.json')); c = np.load(f'{P}/probes/{m}/correct.npz')
        rows = load(f'{P}/rows/{m}/{split}.json'); lab = np.array([r['label'] for r in rows])
        out = []
        for site in ('audio', 'answer'):
            L = info['chosen_layer'][site]; right = c[f'{site}/{L}/{split}']
            out.append(f"{site}@{L} {100*np.mean([right[lab==k].mean() for k in 'ABCD']):.1f} (dev {100*info['table'][f'{site}/{L}']['dev']:.1f})")
        best_any = {site: max(range(info['layers']), key=lambda l: info['table'][f'{site}/{l}'][split]) for site in ('audio','answer')}
        print(split, m, out, 'train-order answer', f"{100*np.mean([np.mean([r['answer']==r['label'] for r in rows if r['label']==k]) for k in 'ABCD']):.1f}",
              '| oracle-best layer', {s: (l, round(100*info['table'][f'{s}/{l}'][split],1)) for s,l in best_any.items()})
