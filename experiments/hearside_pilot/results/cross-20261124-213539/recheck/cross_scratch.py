import json, sys, re, collections, random
B = sys.argv[1]
def load(p): return [json.loads(l) for l in open(p)]
mcq = {r['id']: r for r in load(f'{B}/cremad/mcq.jsonl')}
import os
runs = sorted(os.listdir(f'{B}/cremad/runs'))
fam = collections.defaultdict(list)
for r in runs: fam[re.sub(r'-s4[34]$', '', r)].append(r)
ok = {r: {p['id']: p['prediction'].strip() == p['reference'] for p in load(f'{B}/cremad/runs/{r}/mcq/predictions.jsonl')} for r in runs}
assert all(set(v) == set(mcq) for v in ok.values())
def uar(run, ids):
    by = collections.defaultdict(list)
    for i in ids: by[mcq[i]['label']].append(ok[run][i])
    return 100 * sum(sum(v) / len(v) for v in by.values()) / len(by)
tr = [i for i, r in mcq.items() if r['rotation'] == 0]; ot = [i for i, r in mcq.items() if r['rotation'] != 0]
print(len(mcq), len(tr), len(ot), len({r['speaker'] for r in mcq.values()}))
for k, members in fam.items():
    per = [(uar(m, tr), uar(m, ot), uar(m, list(mcq))) for m in members]
    mean = [sum(x) / len(x) for x in zip(*per)]
    print(f'{k:16s} train {mean[0]:.1f} other {mean[1]:.1f} all {mean[2]:.1f} gap {mean[0]-mean[1]:+.1f} | per seed gap', [round(a - b, 1) for a, b, _ in per], 'other', [round(b, 1) for _, b, _ in per])
# bootstrap over actors and seeds for the two key numbers
by_actor = collections.defaultdict(lambda: ([], []))
for i, r in mcq.items(): by_actor[r['speaker']][0 if r['rotation'] == 0 else 1].append(i)
actors = sorted(by_actor); rng = random.Random(7)
def fm(k, members, ids): return sum(uar(m, ids) for m in members) / len(members)
gapd, gain = [], []
for _ in range(1500):
    a = [rng.choice(actors) for _ in actors]
    t = [i for x in a for i in by_actor[x][0]]; o = [i for x in a for i in by_actor[x][1]]
    idx = [rng.randrange(3) for _ in range(3)]
    L = [fam['A-last4'][j] for j in idx]; A = [fam['A-all'][j] for j in idx]
    gapd.append((fm(0, L, t) - fm(0, L, o)) - (fm(0, A, t) - fm(0, A, o)))
    D = [rng.choice(fam['D-all-shuffled']) for _ in range(3)]
    gain.append(fm(0, D, t + o) - uar('frozen-base', t + o))
for name, v in (('A-last4 gap - A-all gap', gapd), ('D - base all orders', gain)):
    v.sort(); print(name, f'[{v[37]:+.1f}, {v[1462]:+.1f}]')
# what does the base answer?
c = collections.Counter(p['prediction'].strip().lower() for p in load(f'{B}/cremad/runs/frozen-base/open/predictions.jsonl'))
print(c.most_common(12))
lab = collections.defaultdict(collections.Counter)
for p in load(f'{B}/cremad/runs/D-all-shuffled/mcq/predictions.jsonl'):
    r = mcq[p['id']]; pred = r['option_order']['ABCD'.index(p['prediction'].strip())] if p['prediction'].strip() in 'ABCD' and p['prediction'].strip() else '?'
    lab[r['label']][pred] += 1
for k in 'ABCD': print('D seed42 true', k, dict(lab[k]))
