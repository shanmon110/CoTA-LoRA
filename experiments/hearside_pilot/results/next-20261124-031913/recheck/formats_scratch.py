import json, sys, re, collections, random
root = sys.argv[1]
def load(p): return [json.loads(l) for l in open(p)]
meta = {f: {r['id']: r for r in load(f'{root}/formats/{f}.jsonl')} for f in ('yesno', 'word', 'open')}
print({f: len(m) for f, m in meta.items()})
print('example yesno:', json.dumps(next(iter(meta['yesno'].values())))[:600])
WORDS = {'angry': ['angry','anger','mad','furious','irritat','annoy','frustrat','rage','upset'],
         'happy': ['happy','happi','joy','excite','cheer','delight','amus','pleased','glad','elat'],
         'sad': ['sad','sorrow','depress','unhappy','grief','melanchol','down','upset'],
         'neutral': ['neutral','calm','flat','matter-of-fact','indifferent','no emotion']}
def emo(text):
    t = text.lower()
    hit = [e for e, ws in WORDS.items() if any(w in t for w in ws if w != 'upset')]
    return hit[0] if len(hit) == 1 else None
def yn(text):
    t = re.sub(r'[^a-z ]', ' ', text.lower()).split()
    if not t: return None
    return t[0] if t[0] in ('yes', 'no') else None
import glob, os
runs = sorted(os.path.basename(p) for p in glob.glob(f'{root}/formats/runs/*'))
out = {}
for run in runs:
    row = {}
    for f in ('yesno', 'word', 'open'):
        preds = load(f'{root}/formats/runs/{run}/{f}/predictions.jsonl')
        assert len(preds) == len(meta[f]) and {p['id'] for p in preds} == set(meta[f])
        by = collections.defaultdict(list); parsed = 0
        for p in preds:
            ref = p['reference'].strip().lower().replace('happy or excited','happy')
            got = yn(p['prediction']) if f == 'yesno' else emo(p['prediction'])
            parsed += got is not None
            by[ref].append(got == ref)
        row[f] = (100 * sum(sum(v) / len(v) for v in by.values()) / len(by), 100 * parsed / len(preds), {k: len(v) for k, v in by.items()})
    out[run] = row
    print(f'{run:22s}', '  '.join(f'{f} {row[f][0]:5.1f} [{row[f][1]:3.0f}%]' for f in row))
print(out['frozen-base']['yesno'][2], out['frozen-base']['open'][2])
fam = collections.defaultdict(list)
for run, row in out.items():
    fam[re.sub(r'-s4[34]$', '', run)].append(row)
for k, rows in fam.items():
    print(f'{k:16s} n={len(rows)}', '  '.join(f"{f} {sum(r[f][0] for r in rows)/len(rows):5.1f}" for f in ('yesno','word','open')))
