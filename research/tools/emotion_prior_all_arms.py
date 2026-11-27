"""The label-free emotion-prior subtraction, applied to EVERY arm, using the run's own code.

This is `recheck/letters_scratch.py` from the letter-logit run, with the calibration section
extended from the frozen base to every arm.  The method is unchanged:

  1. score every option letter, then re-index the four scores by the EMOTION the slot carries,
     so the score axis is comparable across option orders;
  2. estimate the class prior from UNLABELLED utterances -- the mean log-probability of each
     emotion, per option order -- on one half of the clusters;
  3. subtract that prior and take the argmax;
  4. score on the held-out half, and swap.

No label enters step 2.  The correction is a per-(option order, emotion) additive offset, so
it is the same family as the class-prior correction of Peng & Nuchged (arXiv 2608.06409).

Two questions:

  Q1  Does the correction lift the FROZEN BASE by the amount the run reported (55.6 -> 64.0
      on Session 4, 48.0 -> 61.0 on CREMA-D)?
  Q2  What is the ADAPTED arm's score AFTER the correction, against the corrected frozen
      base?  If an arm's advantage disappears once the frozen base is calibrated the same
      way, the arm was not learning anything the base did not already have.

Run from the repository root:

    python3 -I research/tools/emotion_prior_all_arms.py
"""
import json
import os
import pathlib

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
RESULTS = REPO / 'experiments' / 'hearside_pilot' / 'results'
L = pathlib.Path(next(e.path for e in os.scandir(RESULTS) if e.name.startswith('letters')))
LOGS = L / 'logits'
L4 = 'ABCD'


def logsm(z):
    m = z.max(1, keepdims=True)
    return z - m - np.log(np.exp(z - m).sum(1, keepdims=True))


def load(model, split):
    z = np.load(LOGS / model / split / 'logits.npy').astype(np.float64)
    rows = [json.loads(line) for line in (LOGS / model / split / 'rows.jsonl').open()]
    return z, rows


def main():
    for split in ('session4', 'cremad'):
        print('=' * 100)
        print(f'== {split}: the label-free emotion-prior subtraction, every arm')
        # the frozen base defines the cluster split, so every arm is scored on the same rows
        z0, rows = load('frozen-base', split)
        lab = np.array([L4.index(r['label']) for r in rows])
        order = np.array([[L4.index(c) for c in r['option_order']] for r in rows])
        rot = np.array([r['rotation'] for r in rows])
        utt = np.array([r['id'].rsplit('#', 1)[0] for r in rows])
        names = sorted(set(utt))
        ui = np.array([names.index(u) for u in utt])
        clus = np.empty(len(names), object)
        clus[ui] = [r['cluster'] for r in rows]
        cn = sorted(set(clus))
        ci = np.array([cn.index(c) for c in clus])
        ulab = np.zeros(len(names), int)
        ulab[ui] = lab
        half = np.array([cn.index(c) % 2 for c in clus])

        def score(pred, rng):
            """(utterances, rotations) predicted emotion -> UAR over all rows, with a cluster
            bootstrap interval."""
            r = (pred == ulab[:, None]).astype(float)

            def u(wt):
                return 100 * np.mean([(wt[:, None] * r)[ulab == c].sum()
                                       / (wt[ulab == c].sum() * r.shape[1]) for c in range(4)])
            boots = []
            for _ in range(1000):
                wc = np.bincount(rng.integers(0, len(cn), len(cn)), minlength=len(cn)).astype(float)
                boots.append(u(wc[ci]))
            return u(np.ones(len(names))), np.percentile(boots, [2.5, 97.5])

        def emo_of(z):
            """(utterances, rotations, emotions) log-probability of each emotion."""
            emo = np.zeros((len(names), 4, 4))
            lp = logsm(z)
            for i in range(len(rows)):
                emo[ui[i], rot[i], order[i]] = lp[i]
            return emo

        rng = np.random.default_rng(0)
        rows_out = []
        for model in sorted(p.name for p in LOGS.iterdir()):
            z, _ = load(model, split)
            emo = emo_of(z)
            # as it answers
            v_ans, ci_ans = score(emo.argmax(2), rng)
            # (c) minus an emotion prior per option order, estimated on the held-out half
            pred = np.zeros((len(names), 4), int)
            for h in (0, 1):
                fit = half != h
                app = half == h
                pred[app] = (emo[app] - emo[fit].mean(0, keepdims=True)).argmax(2)
            v_cor, ci_cor = score(pred, rng)
            rows_out.append((model, v_ans, v_cor))
            print(f'   {model:16s} as it answers {v_ans:5.1f}   corrected {v_cor:5.1f} '
                  f'[{ci_cor[0]:.1f}, {ci_cor[1]:.1f}]   gain {v_cor - v_ans:+6.2f}')
        base = [r for r in rows_out if r[0] == 'frozen-base'][0]
        print(f'   -- against the CORRECTED frozen base ({base[2]:.1f}):')
        for model, v_ans, v_cor in rows_out:
            if model == 'frozen-base':
                continue
            print(f'      {model:16s} uncorrected gap {v_ans - base[1]:+6.2f}   '
                  f'corrected gap {v_cor - base[2]:+6.2f}')


if __name__ == '__main__':
    main()
