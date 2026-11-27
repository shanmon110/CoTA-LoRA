"""The label-free emotion-prior subtraction, applied to EVERY arm.

The method (from the repository's own letter-bias run, `recheck/calib_n.py`):

  1. take the frozen (or adapted) model's log-softmax over the four option letters;
  2. re-index them so that axis 1 is the EMOTION a slot carries, not the slot itself;
  3. estimate the class prior from a set of UNLABELLED utterances: the mean log-probability
     of each emotion, per option order;
  4. subtract that prior from the scores and take the argmax.

Step 4 is a per-(order, emotion) additive offset, so on a fixed-label task it is a
per-(order, slot) additive bias -- the same family as the class-prior / logit-adjustment
correction of Peng & Nuchged (arXiv 2608.06409).  The prior is estimated from the model's own
marginal, so it needs no labels.

This script applies that method to every arm, so the correction can be read as a BASELINE on
the frozen base and as a measurement on the adapted arms.  The prior is fitted on one half of
the clusters (dialogue for Session 4, actor for CREMA-D) and scored on the other half, so a
reported gain is not the gain of a correction fitted on the rows it is scored on.

The two questions this answers:

  Q1  Does the correction lift the FROZEN BASE by the amount the repository reported
      (55.6 -> 64.0 on Session 4, 48.0 -> 61.0 on CREMA-D)?
  Q2  Does it help the ADAPTED arms, and does the corrected frozen base catch up with them?
      If the corrected base reaches the adapted arms, then a large part of what the adapter
      "learned" is recoverable with no training at all.

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
LETT = 'ABCD'
EMOS = ['angry', 'happy', 'sad', 'neutral']
TRIALS = 5


def logsm(z):
    m = z.max(1, keepdims=True)
    return z - m - np.log(np.exp(z - m).sum(1, keepdims=True))


def load_emotions(folder):
    """(utterances, 4 orders, 4 emotions) log-softmax scores, indexed by EMOTION.

    The four utterances of one clip differ only in the option order, so the emotion axis is
    comparable across orders and the per-order mean is the model's class prior under that
    order.
    """
    z = np.load(folder / 'logits.npy').astype(np.float64)
    rows = [json.loads(line) for line in (folder / 'rows.jsonl').open()]
    lab = np.array([LETT.index(r['label']) for r in rows])
    order = np.array([[LETT.index(c) for c in r['option_order']] for r in rows])
    rot = np.array([r['rotation'] for r in rows])
    utt = [r['id'].rsplit('#', 1)[0] for r in rows]
    names = sorted(set(utt))
    ui = np.array([names.index(u) for u in utt])
    emo = np.zeros((len(names), 4, 4))
    lp = logsm(z)
    for i in range(len(rows)):
        emo[ui[i], rot[i], order[i]] = lp[i]
    ulab = np.zeros(len(names), int)
    ulab[ui] = lab
    clus = np.empty(len(names), object)
    clus[ui] = [r['cluster'] for r in rows]
    return emo, ulab, clus


def uar(ulab, pred, keep):
    return 100.0 * float(np.mean([(pred[keep & (ulab == c)] == c).mean() for c in range(4)]))


def main():
    for split in ('session4', 'cremad'):
        print('=' * 100)
        print(f'== {split}')
        for arm in sorted(p.name for p in LOGS.iterdir()):
            emo, ulab, clus = load_emotions(LOGS / arm / split)
            base = uar(ulab, emo.argmax(axis=2), np.ones(len(ulab), bool))
            uniq = sorted(set(clus))
            # the whole corpus as the unlabelled source, scored on everything (the run's own
            # headline number, for comparison)
            whole = uar(ulab, (emo - emo.mean(axis=0, keepdims=True)).argmax(axis=2),
                        np.ones(len(ulab), bool))
            # honest: fit the prior on one half of the clusters, score the other
            gains = []
            for trial in range(TRIALS):
                rng = np.random.default_rng(300 + trial)
                perm = rng.permutation(len(uniq))
                halves = [set(uniq[i] for i in perm[: len(uniq) // 2]),
                          set(uniq[i] for i in perm[len(uniq) // 2:])]
                for f, s in ((0, 1), (1, 0)):
                    fit = np.array([i for i, k in enumerate(clus) if k in halves[f]])
                    sco = np.array([i for i, k in enumerate(clus) if k in halves[s]])
                    prior = emo[fit].mean(axis=0, keepdims=True)
                    corrected = (emo[sco] - prior).argmax(axis=2)
                    gains.append(uar(ulab[sco], corrected, np.ones(len(sco), bool))
                                 - uar(ulab[sco], emo[sco].argmax(axis=2), np.ones(len(sco), bool)))
            print(f'   {arm:16s} base {base:5.1f}   whole-corpus prior {whole:5.1f} '
                  f'(+{whole - base:4.1f})   split-half honest gain {np.mean(gains):+5.2f} '
                  f'[{np.percentile(gains, 5):+.2f}, {np.percentile(gains, 95):+.2f}]')


if __name__ == '__main__':
    main()
