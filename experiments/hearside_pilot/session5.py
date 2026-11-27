"""The pre-declared confirmation of the class-marginal correction on IEMOCAP Session 5.

The manifest is built by `experiments/hearside_pilot/session5_manifest.py`.

Session 5 has taken no part in choosing the correction: the correction's method was chosen
after seeing IEMOCAP Session 4 and CREMA-D, so on those two corpora it is a finding, not a
conclusion.  This module applies it to Session 5 and reads off the criteria that were written
down before any Session 5 logit existed.

The criteria (research/notes-20261127-diagnosis-and-plan.md, P1'):

  CONFIRMED   the corrected last-four-layers arm returns to parity with the corrected frozen
              base (|gap| < 2 pp) and its order gap shrinks below 5 pp
  REFUTED     the corrected last-four-layers arm is still clearly below the corrected frozen
              base (gap < -5 pp)
  INCONCLUSIVE  anything in between

The measurement, for every arm:

  raw                  the score as the arm answers, over all rows
  corrected            after the label-free class-marginal correction, with the prior fitted on
                       one half of the dialogue clusters and the other half scored, both
                       directions, averaged over five random splits
  order gap            the training-order score minus the other-order score, before and after

The correction is the one from the letter-logit run's own recheck script: re-index the four
option logits by the emotion their slot carries, estimate the model's marginal over classes
per option order from unlabelled utterances, subtract it, take the argmax.  No label is used
and nothing is fitted on the training order.

Run from the repository root:

    python3 -I -m experiments.hearside_pilot.session5 analyze --root <logits folder> --output <folder>
"""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.hearside_pilot import letters
from experiments.trr_diagnostics import d2

CLASSES = 'ABCD'
EMOS = ['angry', 'happy', 'sad', 'neutral']
# Written down before any Session 5 logit was extracted.
CRITERIA = dict(parity_pp=2.0, order_gap_pp=5.0, refuted_gap_pp=-5.0,
                parity_reading='CONFIRMED', refuted_reading='REFUTED')
TRIALS = 5
BOOT = 2000


def emotion_scores(logits, rows):
    """(rows, 4): the log-probability of the emotion each slot carries."""
    z = logits.astype(np.float64)
    z = z - z.max(axis=1, keepdims=True)
    lp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    order = np.array([[CLASSES.index(c) for c in r['option_order']] for r in rows])
    out = np.zeros((len(rows), 4))
    for i in range(len(rows)):
        out[i, order[i]] = lp[i]
    return out


def uar(pred, label, mask, weights=None):
    """Mean per-class recall over the rows in `mask`."""
    if weights is None:
        weights = np.ones(len(label))
    out = []
    for c in range(4):
        m = mask & (label == c)
        total = weights[m].sum()
        out.append(float((weights[m] * (pred[m] == c)).sum() / total) if total else 0.0)
    return 100 * float(np.mean(out))


def analyse_arm(rows, label, rot, cluster, scores):
    """The raw and corrected scores of one arm, plus the order gap before and after."""
    names = sorted(set(cluster))
    index = np.array([names.index(c) for c in cluster])
    nclust = len(names)
    rng = np.random.default_rng(7)
    splits = []
    for _ in range(TRIALS):
        perm = rng.permutation(nclust)
        splits.append((set(perm[: nclust // 2]), set(perm[nclust // 2:])))

    raw = scores.argmax(axis=1)
    is_train = rot == 0
    # for each split and direction: the scored rows and their corrected prediction
    pieces = []
    for left, right in splits:
        for fit_idx, app_idx in ((left, right), (right, left)):
            fit = np.isin(index, list(fit_idx))
            app = np.isin(index, list(app_idx))
            corrected = scores.copy()
            for r in range(4):
                m = fit & (rot == r)
                if m.any():
                    corrected[rot == r] = scores[rot == r] - scores[m].mean(axis=0)
            idx = np.where(app)[0]
            pieces.append((idx, raw[idx].copy(), corrected[idx].argmax(axis=1)))

    ones = np.ones(len(label))

    def score(corrected, weights, mask=None):
        vals = []
        for idx, praw, pcor in pieces:
            keep = np.ones(len(idx), bool) if mask is None else mask[idx]
            pred = pcor if corrected else praw
            vals.append(uar(pred, label[idx], keep, weights[idx]))
        return float(np.mean(vals))

    def order_gap(corrected, weights):
        vals = []
        for idx, praw, pcor in pieces:
            pred = pcor if corrected else praw
            tr = rot[idx] == 0
            a = uar(pred[tr], label[idx][tr], np.ones(tr.sum(), bool), weights[idx][tr])
            b = uar(pred[~tr], label[idx][~tr], np.ones((~tr).sum(), bool), weights[idx][~tr])
            vals.append(a - b)
        return float(np.mean(vals))

    gains = []
    for _ in range(BOOT):
        wc = np.bincount(rng.integers(0, nclust, nclust), minlength=nclust).astype(float)
        w = wc[index]
        gains.append(score(True, w) - score(False, w))
    lo, hi = np.percentile(gains, [2.5, 97.5])
    return dict(
        raw=score(False, ones), corrected=score(True, ones),
        gain=score(True, ones) - score(False, ones), gain_interval=[float(lo), float(hi)],
        order_gap_raw=order_gap(False, ones), order_gap_corrected=order_gap(True, ones))


def judge(entry, last4, base):
    """The pre-declared reading, from the corrected gap and the corrected order gap."""
    gap = entry[last4]['corrected'] - entry[base]['corrected']
    order_gap = entry[last4]['order_gap_corrected']
    if abs(gap) < CRITERIA['parity_pp'] and abs(order_gap) < CRITERIA['order_gap_pp']:
        reading = CRITERIA['parity_reading']
    elif gap < CRITERIA['refuted_gap_pp']:
        reading = CRITERIA['refuted_reading']
    else:
        reading = 'INCONCLUSIVE'
    return dict(reading=reading, corrected_gap_pp=gap, corrected_order_gap_pp=order_gap,
                uncorrected_gap_pp=entry[last4]['raw'] - entry[base]['raw'],
                uncorrected_order_gap_pp=entry[last4]['order_gap_raw'],
                criteria=CRITERIA)


def analyze(root, *, output=None, split='session5'):
    root = Path(root)
    base_logits, base_rows = letters.load(root / 'frozen-base' / split)
    label = np.array([CLASSES.index(r['label']) for r in base_rows])
    rot = np.array([r['rotation'] for r in base_rows])
    cluster = np.array([r['cluster'] for r in base_rows])
    entry = {}
    for folder in sorted(p for p in root.iterdir() if (p / split / 'done.json').exists()):
        logits, rows = letters.load(folder / split)
        if [r['id'] for r in rows] != [r['id'] for r in base_rows]:
            raise ValueError(f'{folder.name}/{split}: rows differ from the frozen base')
        entry[folder.name] = analyse_arm(rows, label, rot, cluster,
                                         emotion_scores(logits, rows))
    if 'frozen-base' not in entry:
        raise ValueError('the frozen base has no Session 5 logits')
    last4 = [m for m in ('A-last4', 'A-last4-s43', 'A-last4-s44') if m in entry]
    if not last4:
        raise ValueError('no last-four-layers arm has Session 5 logits')
    # the last-four-layers arm as a family, averaged over its seeds
    family = dict(
        raw=float(np.mean([entry[m]['raw'] for m in last4])),
        corrected=float(np.mean([entry[m]['corrected'] for m in last4])),
        gain=float(np.mean([entry[m]['gain'] for m in last4])),
        gain_interval=[float(np.min([entry[m]['gain_interval'][0] for m in last4])),
                       float(np.max([entry[m]['gain_interval'][1] for m in last4]))],
        order_gap_raw=float(np.mean([entry[m]['order_gap_raw'] for m in last4])),
        order_gap_corrected=float(np.mean([entry[m]['order_gap_corrected'] for m in last4])))
    entry['A-last4 (mean)'] = family
    base = entry['frozen-base']
    result = dict(criteria=CRITERIA,
                  corpora={split: dict(rows=len(base_rows),
                                      utterances=len(set(cluster)),
                                      clusters=len(set(cluster)),
                                      arms=entry)},
                  reading=judge(entry, 'A-last4 (mean)', 'frozen-base'))
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        d2.write(output / f'{split}.json', result)
        (output / f'{split}.md').write_text(render(result))
    return result


def render(result):
    split = next(iter(result['corpora']))
    entry = result['corpora'][split]
    lines = [f'# The class-marginal correction on IEMOCAP {split.replace("session", "Session ")}', '',
             'Session 5 took no part in choosing the correction. The criteria below were written '
             'down before any Session 5 logit existed.', '',
             f"Rows: {entry['rows']}, utterances: {entry['utterances']}, dialogue clusters: {entry['clusters']}.", '',
             '| arm | raw | corrected | gain [95% CI] | order gap raw | order gap corrected |',
             '|---|---:|---:|---|---:|---:|']
    for name, e in entry['arms'].items():
        lines.append(f"| {name} | {e['raw']:.1f} | {e['corrected']:.1f} | "
                     f"{e['gain']:+.1f} [{e['gain_interval'][0]:+.1f}, {e['gain_interval'][1]:+.1f}] | "
                     f"{e['order_gap_raw']:+.1f} | {e['order_gap_corrected']:+.1f} |")
    r = result['reading']
    lines += ['', '## The pre-declared reading', '',
              f"Corrected gap of the last-four-layers arm over the frozen base: {r['corrected_gap_pp']:+.2f} pp "
              f"(uncorrected {r['uncorrected_gap_pp']:+.2f} pp).",
              f"Corrected order gap: {r['corrected_order_gap_pp']:+.2f} pp "
              f"(uncorrected {r['uncorrected_order_gap_pp']:+.2f} pp).", '',
              f"Criteria: parity within {CRITERIA['parity_pp']} pp and order gap below "
              f"{CRITERIA['order_gap_pp']} pp reads **{CRITERIA['parity_reading']}**; a gap below "
              f"{CRITERIA['refuted_gap_pp']} pp reads **{CRITERIA['refuted_reading']}**.", '',
              f"Reading: **{r['reading']}**", '']
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('analyze')
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--split', default='session5')
    args = parser.parse_args()
    result = analyze(args.root, output=args.output, split=args.split)
    print(render(result))
