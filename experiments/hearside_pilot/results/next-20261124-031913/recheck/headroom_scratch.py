import json, sys, collections
B, P = sys.argv[1], sys.argv[2]
def load(p): return [json.loads(l) for l in open(p)]
def uar(rows):
    by = collections.defaultdict(list)
    for lab, ok in rows: by[lab].append(ok)
    return 100 * sum(sum(v)/len(v) for v in by.values()) / len(by)
def rows(folder):
    lab = {r['id']: r['label'] for r in load(f'{folder}/test.jsonl')}
    return {p['id']: (lab[p['id']], p['prediction'].strip() == p['reference']) for p in load(f'{folder}/eval/predictions.jsonl')}
groups = {'gender': ['gender-F','gender-M'], 'duration': ['duration-short','duration-medium','duration-long'], 'scripted': ['scripted-impro','scripted-script'], 'random2': ['random2-part0','random2-part1'], 'random3': ['random3-part0','random3-part1','random3-part2']}
for g, parts in groups.items():
    allr = {}
    for p in parts:
        r = rows(f'{B}/headroom/{p}'); assert not set(r) & set(allr); allr.update(r)
        n = sum(1 for _ in open(f'{B}/headroom/{p}/train.jsonl'))
        print(f'   {p:18s} train rows {n:5d}  test rows {len(r):5d}  UAR {uar(r.values()):.1f}')
    assert len(allr) == 4104
    print(f'{g:9s} pooled specialists UAR {uar(allr.values()):.2f}')
# shared D seed 42
man = {json.loads(l)['id']: json.loads(l)['label'] for l in open(f'{B}/headroom/gender-F/test.jsonl')}
man.update({json.loads(l)['id']: json.loads(l)['label'] for l in open(f'{B}/headroom/gender-M/test.jsonl')})
d = [(man[p['id']], p['prediction'].strip() == p['reference']) for p in load(f'{P}/eval/runs/D-all-shuffled/orig/predictions.jsonl')]
print('shared D seed 42', round(uar(d), 2), len(d))
