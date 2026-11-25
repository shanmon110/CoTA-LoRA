import json, sys
import numpy as np
B = sys.argv[1]; L = 'ABCD'
def logsm(z): m = z.max(1, keepdims=True); return z - m - np.log(np.exp(z - m).sum(1, keepdims=True))
for split in ('session4', 'cremad'):
    z = np.load(f'{B}/logits/frozen-base/{split}/logits.npy').astype(np.float64)
    rows = [json.loads(l) for l in open(f'{B}/logits/frozen-base/{split}/rows.jsonl')]
    lab = np.array([L.index(r['label']) for r in rows]); order = np.array([[L.index(c) for c in r['option_order']] for r in rows]); rot = np.array([r['rotation'] for r in rows])
    utt = [r['id'].rsplit('#', 1)[0] for r in rows]; names = sorted(set(utt)); ui = np.array([names.index(u) for u in utt])
    emo = np.zeros((len(names), 4, 4)); lp = logsm(z)
    for i in range(len(rows)): emo[ui[i], rot[i], order[i]] = lp[i]
    ulab = np.zeros(len(names), int); ulab[ui] = lab
    clus = np.empty(len(names), object); clus[ui] = [r['cluster'] for r in rows]
    def uar(pred, keep): return 100 * np.mean([(pred[keep & (ulab == c)] == c).mean() for c in range(4)])
    rng = np.random.default_rng(1)
    print('==', split, 'base', round(uar(emo.argmax(2), np.ones(len(names), bool)), 1))
    for n in (8, 16, 32, 64, 128, 256):
        vals = []
        for _ in range(200):
            fit_clusters = rng.permutation(sorted(set(clus)))
            chosen = []                                         # n unlabelled utterances from whole clusters; the rest is scored
            used = set()
            for c in fit_clusters:
                members = np.where(clus == c)[0]; used.add(c); chosen += list(members)
                if len(chosen) >= n: break
            chosen = np.array(chosen[:n]); app = ~np.isin(clus, list(used))
            vals.append(uar((emo - emo[chosen].mean(0, keepdims=True)).argmax(2), app))
        print(f'   emotion prior from {n:3d} unlabelled utterances: {np.mean(vals):.1f}  (5th-95th percentile over draws {np.percentile(vals, 5):.1f}-{np.percentile(vals, 95):.1f})')
    # with the true class frequencies known (not label-free): add log class prior back
    prior = np.log(np.bincount(ulab) / len(ulab))
    half = np.array([sorted(set(clus)).index(c) % 2 for c in clus]); pred = np.zeros((len(names), 4), int)
    for h in (0, 1): pred[half == h] = (emo[half == h] - emo[half != h].mean(0, keepdims=True) + prior).argmax(2)
    print('   same, with the true class frequencies supplied:', round(uar(pred, np.ones(len(names), bool)), 1))
    # the constants themselves, training order
    print('   mean log-probability per emotion, training order:', np.round(emo[:, 0].mean(0), 2), ' rotation 1:', np.round(emo[:, 1].mean(0), 2))
