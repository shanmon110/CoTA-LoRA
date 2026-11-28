"""P5: the surrogate test, generalised to AQA.

On IEMOCAP the surrogate is the repository's own method: one constant per option letter, added
to the frozen base's option logits and fitted on the training split, which reconstructs 79% of
the last-four-layers arm's training-order gain.  That shows that a 131,104-parameter adapter
behaved, in the training order, like a three-degree-of-freedom output bias.

P5 turns that into a general protocol: can the behaviour of ANY adapter be reconstructed by a
few degrees of freedom?  AQA has no fixed class-to-letter correspondence -- every question has
its own options -- so the surrogate takes a different form there: one constant per SLOT, applied
to the arm's OWN option logits, with the rule fitted on half the questions and scored on the
other half.

The question the test answers is the one the surrogate was built for: is the arm's gain a
decision-rule artefact, or a change of the representation?  If an affine rule on the arm's own
logits reproduces most of the gain, the gain is a readout offset.  If it does not, the arm's
logits moved in a way a fixed offset cannot capture.

The measurements, for every arm:

  arm as it answers        the accuracy the arm actually gets
  base + arm's rule        the FROZEN BASE's logits with the arm's fitted rule transplanted,
                            which asks whether the rule alone is what the arm learned
  arm + arm's rule         an affine rule on the arm's OWN logits, fitted on one half of the
                            questions and scored on the other

The rule is fitted on one half of the questions and scored on the other, both directions, so no
number is a fit on the rows it is scored on.  The interval is a bootstrap over the questions.

Run from the repository root:

    python3 -I research/tools/aqa_surrogate.py --root <logits folder> [--output <folder>]
"""
import argparse
import json
import pathlib
from pathlib import Path

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
LETTERS = 'ABCDEFGH'
BOOT = 2000
TRIALS = 5
SEED = 42


def load(root, arm):
    folder = pathlib.Path(root) / arm / 'mcq'
    z = np.load(folder / 'logits.npy').astype(np.float64)
    rows = [json.loads(line) for line in open(folder / 'rows.jsonl')]
    return z, rows


def fit_rule(logits, gold, n_options, *, steps=400, rate=0.5):
    """One constant per slot, added to the logits, fitted on the gold letters.

    The gold is the letter of the correct option in the DISPLAY order, so the rule is a per-slot
    offset, exactly the AQA counterpart of the IEMOCAP per-letter offset.

    The rule is fitted per option count, because a row with two options has no slot C.  The rows
    here all have the same number of options, so the rule has that many degrees of freedom.
    """
    width = int(n_options.max())
    counts = np.bincount(gold, minlength=width)
    weight = (len(gold) / np.maximum(width * counts, 1))[gold]
    bias = np.zeros(width)
    onehot = np.zeros_like(logits)
    onehot[np.arange(len(gold)), gold] = 1
    for _ in range(steps):
        z = logits + bias
        p = np.exp(z - z.max(axis=1, keepdims=True))
        p /= p.sum(axis=1, keepdims=True)
        bias -= rate * ((p - onehot) * weight[:, None]).mean(axis=0)
        bias -= bias.mean()
    return bias


def main(root, output=None):
    root = pathlib.Path(root)
    base_z, base_rows = load(root, 'frozen-base')
    gold = np.array([LETTERS.index(r['gold']) for r in base_rows])
    n_options = np.array([r['n'] for r in base_rows])
    keep = n_options == 4
    # the questions are the independent unit: the four option orders of one question are not
    question = np.array([r['question'] for r in base_rows])
    names = sorted(set(question))
    index = np.array([names.index(q) for q in question])
    nclust = len(names)

    rng = np.random.default_rng(SEED)
    splits = []
    for _ in range(TRIALS):
        perm = rng.permutation(nclust)
        splits.append((set(perm[:nclust // 2]), set(perm[nclust // 2:])))

    def accuracy(pred, rows_mask, weights, labels):
        """The mean per-class recall over the rows in the mask."""
        out = []
        for c in range(4):
            m = rows_mask & (labels == c)
            total = weights[m].sum()
            out.append(float((weights[m] * (pred[m] == c)).sum() / total) if total else 0.0)
        return 100 * float(np.mean(out))

    print('=' * 104)
    print('== P5: can a few degrees of freedom on the arm\'s own logits reconstruct its behaviour?')
    print(f'   rows {len(base_rows)}, questions {nclust}, four-option rows {int(keep.sum())}')
    print()
    print(f'   {"arm":16s} {"as it answers":>14s} {"arm + its rule":>15s} {"base + its rule":>16s} '
          f'{"share of the gain":>19s}')

    base_score = None
    for arm in sorted(p.name for p in root.iterdir() if (p / 'mcq' / 'done.json').exists()):
        z, rows = load(root, arm)
        if [r['id'] for r in rows] != [r['id'] for r in base_rows]:
            raise ValueError(f'{arm}: rows differ from the frozen base')
        raw = z.argmax(axis=1)
        ones = np.ones(len(gold))
        arm_score = accuracy(raw, keep, ones, gold)

        # the rule is fitted on one half of the questions and applied to the other
        pieces = []
        for left, right in splits:
            for fit_idx, app_idx in ((left, right), (right, left)):
                fit = np.isin(index, list(fit_idx)) & keep
                app = np.isin(index, list(app_idx)) & keep
                bias = fit_rule(z[fit], gold[fit], n_options[fit])
                idx = np.where(app)[0]
                pieces.append((idx, raw[idx].copy(), (z[idx] + bias).argmax(axis=1),
                               (base_z[idx] + bias).argmax(axis=1)))
        ones_p = np.ones(len(gold))

        def mean_score(kind, weights):
            vals = []
            for idx, praw, parc, pbase in pieces:
                pred = praw if kind == 'raw' else (parc if kind == 'arm' else pbase)
                # `idx` indexes the global rows, so every per-row array is sliced with it
                vals.append(accuracy(pred, keep[idx], weights[idx], gold[idx]))
            return float(np.mean(vals))

        with_rule = mean_score('arm', ones_p)
        base_with_rule = mean_score('base', ones_p)
        if arm == 'frozen-base':
            base_score = arm_score
            print(f'   {arm:16s} {arm_score:13.2f}% {with_rule:14.2f}% {base_with_rule:15.2f}% '
                  f'{"--":>19s}')
            continue

        # the bootstrap over the questions
        draws = []
        for _ in range(BOOT):
            picked = rng.integers(0, nclust, nclust)
            w = np.bincount(picked, minlength=nclust).astype(float)[index]
            draws.append(mean_score('arm', w) - mean_score('raw', w))
        lo, hi = np.percentile(draws, [2.5, 97.5])
        gain = arm_score - base_score
        recovered = with_rule - base_score
        share = 100 * recovered / gain if gain else float('nan')
        print(f'   {arm:16s} {arm_score:13.2f}% {with_rule:14.2f}% {base_with_rule:15.2f}% '
              f'{share:17.1f}%   (the rule on its own {recovered:+.2f} pp of {gain:+.2f} pp; '
              f'the rule on the arm adds {with_rule - arm_score:+.2f} pp [{lo:+.2f}, {hi:+.2f}])')

    print()
    print('The pre-declared reading: if an affine rule on the arm\'s own option logits, fitted on')
    print('one half of the questions and scored on the other, reproduces most of the arm\'s gain')
    print('over the frozen base, then the gain is a decision-rule artefact.  If it does not, the')
    print('arm\'s logits moved in a way a fixed offset cannot capture, and the change is in the')
    print('representation.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    main(args.root, args.output)
