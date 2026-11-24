import json, sys, re, collections, glob, os, random
root = sys.argv[1]
def load(p): return [json.loads(l) for l in open(p)]
meta = {r['id']: r for r in load(f'{root}/formats/open.jsonl')}
BASE = {'angry': ['angry','anger','mad','furious','irritat','annoy','frustrat','rage'],
        'happy': ['happy','happi','joy','excite','cheer','delight','amus','pleased','glad','elat'],
        'sad': ['sad','sorrow','depress','unhappy','grief','melanchol'],
        'neutral': ['neutral','calm','flat','matter-of-fact','indifferent','no emotion']}
def parser(extra):
    W = {k: v + extra.get(k, []) for k, v in BASE.items()}
    def f(text):
        t = text.lower(); hit = [e for e, ws in W.items() if any(w in t for w in ws)]
        return hit[0] if len(hit) == 1 else None
    return f
variants = {'declared-like': parser({}), 'relaxed->neutral': parser({'neutral': ['relaxed']})}
runs = sorted(os.path.basename(p) for p in glob.glob(f'{root}/formats/runs/*'))
dialogues = sorted({m['dialogue'] for m in meta.values()})
for name, f in variants.items():
    corr = {}
    for run in runs:
        corr[run] = {p['id']: (p['reference'].replace('happy or excited','happy'), f(p['prediction']) == p['reference'].replace('happy or excited','happy')) for p in load(f'{root}/formats/runs/{run}/open/predictions.jsonl')}
    def uar(run, ids):
        by = collections.defaultdict(list)
        for i in ids: by[corr[run][i][0]].append(corr[run][i][1])
        return 100 * sum(sum(v)/len(v) for v in by.values()) / len(by)
    fam = collections.defaultdict(list)
    for r in runs: fam[re.sub(r'-s4[34]$', '', r)].append(r)
    ids_by_d = collections.defaultdict(list)
    for i, m in meta.items(): ids_by_d[m['dialogue']].append(i)
    allids = list(meta)
    def famuar(k, ids, members=None): 
        members = members or fam[k]; return sum(uar(r, ids) for r in members) / len(members)
    print('==', name, {k: round(famuar(k, allids), 1) for k in fam})
    rng = random.Random(42)
    for a, b in (('B-audio','A-all'), ('D-all-shuffled','frozen-base'), ('A-all','frozen-base'), ('B-audio','frozen-base'), ('A-last4','frozen-base')):
        point = famuar(a, allids) - famuar(b, allids); bs = []
        for _ in range(2000):
            ds = [rng.choice(dialogues) for _ in dialogues]; ids = [i for d in ds for i in ids_by_d[d]]
            ma = [rng.choice(fam[a]) for _ in fam[a]]; mb = [rng.choice(fam[b]) for _ in fam[b]]
            bs.append(famuar(a, ids, ma) - famuar(b, ids, mb))
        bs.sort(); print(f'   {a} - {b}: {point:+.1f} [{bs[50]:+.1f}, {bs[1949]:+.1f}]')
