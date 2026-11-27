"""Independent recomputation of the surrogate, then label-free calibration of the frozen base (exploratory)."""
import json, sys
import numpy as np
B = sys.argv[1]
L = 'ABCD'
def load(model, split):
    z = np.load(f'{B}/logits/{model}/{split}/logits.npy').astype(np.float64)
    rows = [json.loads(l) for l in open(f'{B}/logits/{model}/{split}/rows.jsonl')]
    return z, rows
def logsm(z): return z - np.log(np.exp(z - z.max(1, keepdims=True)).sum(1, keepdims=True)) - z.max(1, keepdims=True)
def uar(right, lab, keep=None):
    keep = np.ones(len(lab), bool) if keep is None else keep
    return 100 * np.mean([right[keep & (lab == c)].mean() for c in range(4)])

# 1. the surrogate, refitted with a different optimiser (Newton steps on three free numbers)
zt, rt = load('frozen-base', 'train'); yt = np.array([L.index(r['label']) for r in rt])
w = (len(yt) / (4 * np.bincount(yt, minlength=4)))[yt]
b = np.zeros(4)
for _ in range(60):
    p = np.exp(logsm(zt + b)); g = ((p - np.eye(4)[yt]) * w[:, None]).mean(0)
    H = (np.einsum('n,ni,nj->ij', w, p, -p) + np.diag((w[:, None] * p).sum(0))) / len(yt) + 1e-6 * np.eye(4)
    b = b - np.linalg.solve(H, g); b -= b.mean()
print('letter constants, refitted:', dict(zip(L, np.round(b, 2))))
out = {}
for split in ('session4', 'cremad'):
    z, rows = load('frozen-base', split)
    lab = np.array([L.index(r['label']) for r in rows]); order = np.array([[L.index(c) for c in r['option_order']] for r in rows])
    rot = np.array([r['rotation'] for r in rows]); tr = rot == 0
    right = lambda letters: order[np.arange(len(rows)), letters] == lab
    last = [load(m, split)[0].argmax(1) for m in ('A-last4', 'A-last4-s43', 'A-last4-s44')]
    sur = (z + b).argmax(1)
    print(f'== {split}: base {uar(right(z.argmax(1)), lab, tr):.1f}/{uar(right(z.argmax(1)), lab, ~tr):.1f}  surrogate {uar(right(sur), lab, tr):.1f}/{uar(right(sur), lab, ~tr):.1f}'
          f'  last4 {np.mean([uar(right(l), lab, tr) for l in last]):.1f}/{np.mean([uar(right(l), lab, ~tr) for l in last]):.1f}'
          f'  same letter as last4: base {100*np.mean([np.mean(z.argmax(1) == l) for l in last]):.1f} surrogate {100*np.mean([np.mean(sur == l) for l in last]):.1f}')

    # 2. label-free calibration of the frozen base. Scores per utterance and emotion, per option order.
    utt = np.array([r['id'].rsplit('#', 1)[0] for r in rows]); names = sorted(set(utt)); ui = np.array([names.index(u) for u in utt])
    lp = logsm(z)
    emo = np.zeros((len(names), 4, 4))                      # utterance, rotation, emotion -> log-probability of that emotion's letter
    for i in range(len(rows)):
        emo[ui[i], rot[i], order[i]] = lp[i]
    ulab = np.zeros(len(names), int); ulab[ui] = lab
    clus = np.empty(len(names), object); clus[ui] = [r['cluster'] for r in rows]
    cn = sorted(set(clus)); ci = np.array([cn.index(c) for c in clus])
    rng = np.random.default_rng(0)
    def score(pred_by_rot):                                  # (utterances, rotations) predicted emotion -> UAR over all rows, with CI
        r = (pred_by_rot == ulab[:, None]).astype(float)
        def u(wt): return 100 * np.mean([(wt[:, None] * r)[ulab == c].sum() / (wt[ulab == c].sum() * r.shape[1]) for c in range(4)])
        boots = []
        for _ in range(1000):
            wc = np.bincount(rng.integers(0, len(cn), len(cn)), minlength=len(cn)).astype(float); boots.append(u(wc[ci]))
        return u(np.ones(len(names))), np.percentile(boots, [2.5, 97.5])
    def show(name, pred, note=''):
        v, (lo, hi) = score(pred); tr_v = 100 * np.mean([(pred[ulab == c, 0] == c).mean() for c in range(4)]); ot_v = 100 * np.mean([(pred[ulab == c, 1:] == c).mean() for c in range(4)]) if pred.shape[1] > 1 else float('nan')
        print(f'   {name:58s} all orders {v:5.1f} [{lo:.1f}, {hi:.1f}]   training order {tr_v:5.1f}  other orders {ot_v:5.1f} {note}')
        return v
    show('frozen base, as it answers', emo.argmax(2))
    # (a) average the four option orders (costs four passes per utterance)
    avg = emo.mean(1).argmax(1); show('(a) average of the four option orders', np.repeat(avg[:, None], 1, 1), '(one answer per utterance)')
    # (b) letter prior estimated without labels: mean log-probability of each letter over all orders of a held-out half of the clusters
    def crossfit(fn):
        pred = np.zeros((len(names), 4), int); half = np.array([cn.index(c) % 2 for c in clus])
        for h in (0, 1): pred[half == h] = fn(half != h, half == h)
        return pred
    def letter_prior(fit, app):
        rr = np.isin(ui, np.where(fit)[0]); prior = lp[rr].mean(0)          # each emotion sits at each letter equally often
        out = np.zeros((app.sum(), 4), int); idx = np.where(app)[0]
        for k in range(4):
            zz = np.zeros((len(names), 4)); sel = rot == k; zz[ui[sel]] = lp[sel] - prior
            oo = np.zeros((len(names), 4), int); oo[ui[sel]] = order[sel]
            out[:, k] = oo[idx, zz[idx].argmax(1)]
        return out
    show('(b) minus a letter prior from unlabelled data', crossfit(letter_prior))
    # (c) emotion prior estimated without labels, per option order: mean log-probability of each emotion over a held-out half
    def emotion_prior(fit, app):
        return (emo[app] - emo[fit].mean(0, keepdims=True)).argmax(2)
    show('(c) minus an emotion prior per option order, unlabelled', crossfit(emotion_prior), '(assumes classes equally frequent)')
    # (d) both: average the four orders, then subtract the unlabelled emotion prior
    def both(fit, app):
        m = emo.mean(1); return np.repeat((m[app] - m[fit].mean(0)).argmax(1)[:, None], 4, 1)
    show('(d) average of four orders, minus the emotion prior', crossfit(both)[:, :1], '(one answer per utterance)')
    for m in ('A-first4', 'A-all', 'D-all-shuffled'):
        zz = load(m, split)[0]; pr = np.zeros((len(names), 4), int); pr[ui, rot] = order[np.arange(len(rows)), zz.argmax(1)]
        show(f'{m} (seed 42), as it answers', pr)
    print('   class shares:', np.round(np.bincount(ulab) / len(ulab), 2))
