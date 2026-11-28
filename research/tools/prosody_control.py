"""P3: is the prosody line alive?  The same sentence, the same actor, a different emotion.

The D2b line read "sensitive but not calibrated" from a comparison that flattened the F0 or the
energy of the audio.  The run's own gate rejects that comparison: flattening raised the word
error rate from 8.95% to 12.90%, so the words changed too and the comparison cannot separate
"the model uses prosody" from "the model reads a different sentence".  The D2b reading is
therefore downgraded to "not interpretable" and the prosody line is close.

CREMA-D provides the control that was missing, with no new data and no new model.  Its files
are named `<sentence>_<actor>_<emotion>_<intensity>.wav`, so for one sentence and one actor
there are several files that say **the same words** and differ only in how they are said:

    1001_IEO_ANG_HI.wav   1001_IEO_HAP_LO.wav   1001_IEO_SAD_MD.wav   ...

So the comparison is word-preserving by construction: the tokens are identical, only the
prosody differs.  If the model's answer changes with the emotion of the same sentence, then the
model is using prosody, and the D2b line can be rewritten around this control instead of around
the flattening one.

The measurement, for every arm:

  per-emotion recall        the class-averaged recall within each emotion
  the spread                the largest per-emotion difference, with a bootstrap interval over
                            the actors (the independent unit: one actor says one sentence in
                            several emotions)

The pre-declared reading (research/notes-20261127-diagnosis-and-plan.md, P3):

  if the answers differ significantly between the emotions of the same sentence -- the interval
  excludes zero -- and the movement is in the direction of the emotion, then prosody sensitivity
  holds and the D2b conclusion can be rewritten with this control;
  if the spread is at or below 1.2 pp, the effect size the flattening comparison reported, and
  the interval crosses zero, then **the line is closed**: the result is reported as a negative
  one, and the flattening comparison's numbers may not be used to support any prosody claim.

Run from the repository root:

    python3 -I research/tools/prosody_control.py [cremad run folder]
"""
import collections
import json
import pathlib
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
CROSS = REPO / 'hearside-cross-20261124-213539'
EMOTIONS = ('ANG', 'DIS', 'FEA', 'HAP', 'NEU', 'SAD')
CLASS_OF = {'ANG': 'A', 'DIS': 'C', 'FEA': 'A', 'HAP': 'B', 'NEU': 'D', 'SAD': 'C'}
BOOT = 10000
SEED = 42
EFFECT = 1.2      # the effect size the flattening comparison reported, in pp


def read_rows(folder):
    return [json.loads(line) for line in open(folder / 'mcq.jsonl')]


def read_predictions(folder):
    out = {}
    for arm in sorted(p.name for p in (folder / 'runs').iterdir()):
        file = folder / 'runs' / arm / 'mcq' / 'predictions.jsonl'
        if file.exists():
            out[arm] = [json.loads(line) for line in open(file)]
    return out


def parse_utterance(uid):
    """`1001_IEO_ANG_HI` -> (actor, sentence, emotion, intensity)."""
    parts = uid.split('_')
    return parts[0], parts[1], parts[2], parts[3]


def uar(correct, labels, keep, weights=None):
    """The class-averaged recall over the rows in `keep`, with optional per-row weights.

    The weights are what the cluster bootstrap resamples: an actor that is drawn twice counts
    twice.  Filtering the rows out instead would change the population on every draw.
    """
    if weights is None:
        weights = np.ones(len(correct))
    out = []
    for c in ('A', 'B', 'C', 'D'):
        m = keep & (labels == c)
        total = weights[m].sum()
        out.append(float((weights[m] * correct[m]).sum() / total) if total else 0.0)
    return 100 * float(np.mean(out))


def main(folder):
    folder = pathlib.Path(folder)
    rows = read_rows(folder)
    arms = read_predictions(folder)
    if 'frozen-base' not in arms:
        raise SystemExit('no frozen base predictions')
    meta = {}
    for row in rows:
        actor, sentence, emotion, intensity = parse_utterance(row['utterance'])
        meta[row['id']] = (actor, sentence, emotion, intensity)
    labels = np.array([r['label'] for r in rows])
    index = {r['id']: k for k, r in enumerate(rows)}
    actors = sorted({m[0] for m in meta.values()})
    actor_index = {a: k for k, a in enumerate(actors)}
    emotion_of = np.array([m[2] for m in meta.values()])
    actor_of = np.array([actor_index[m[0]] for m in meta.values()])

    print('=' * 104)
    print('== P3: the same sentence and the same actor, a different emotion')
    groups = collections.defaultdict(set)
    for k, (actor, sentence, emotion, intensity) in enumerate(meta.values()):
        groups[(actor, sentence)].add(emotion)
    print(f'   rows {len(rows)}, actors {len(actors)}, '
          f'(actor, sentence) groups {len(groups)}, '
          f'emotions per group {min(len(v) for v in groups.values())}'
          f'-{max(len(v) for v in groups.values())}')
    print('   this control is word-preserving by construction: CREMA-D files of one sentence and '
          'one actor differ only in emotion and intensity')

    for arm in sorted(arms):
        preds = arms[arm]
        by_id = {p['id']: p for p in preds}
        correct = np.zeros(len(rows), bool)
        seen = np.zeros(len(rows), bool)
        for r in rows:
            p = by_id.get(r['id'])
            if p is None:
                continue
            seen[index[r['id']]] = True
            correct[index[r['id']]] = p['prediction'] == p['reference']
        if not seen.all():
            print(f'   {arm}: {int((~seen).sum())} rows without a prediction, skipped')
            continue
        print()
        print(f'   {arm}')
        # Within one emotion there is exactly one gold class, so the class-averaged recall
        # degenerates to the plain accuracy.  The accuracy per emotion is the measure.
        recalls = {}
        for emo in EMOTIONS:
            m = emotion_of == emo
            if not m.any():
                continue
            recalls[emo] = 100 * float(correct[m].mean())
        for emo in EMOTIONS:
            if emo in recalls:
                n = int((emotion_of == emo).sum())
                print(f'      {emo}: accuracy {recalls[emo]:5.1f}   ({n} rows)')
        spread = max(recalls.values()) - min(recalls.values()) if recalls else 0.0
        # The bootstrap is over the actors, which are the independent unit: one actor says one
        # sentence in several emotions, so the actor is resampled as a whole.
        rng = np.random.default_rng(SEED)
        draws = []
        for _ in range(BOOT):
            picked = rng.integers(0, len(actors), len(actors))
            w = np.bincount(picked, minlength=len(actors)).astype(float)[actor_of]
            per = {}
            for emo in EMOTIONS:
                m = emotion_of == emo
                if not m.any():
                    continue
                per[emo] = 100 * float((w[m] * correct[m]).sum() / max(w[m].sum(), 1))
            if len(per) > 1:
                draws.append(max(per.values()) - min(per.values()))
        lo, hi = np.percentile(draws, [2.5, 97.5])
        print(f'      spread across the emotions {spread:5.1f} pp  '
              f'[{lo:+.1f}, {hi:+.1f}]  (the actors are resampled together)')
        print(f'      the pre-declared threshold is {EFFECT:.1f} pp with an interval that '
              f'excludes zero: '
              + ('HELD -- the prosody line stays open'
                 if spread > EFFECT and lo > 0 else
                 'NOT HELD -- the line is closed and the flattening numbers may not be used'))

    print()
    print('=' * 104)
    print('== the direction of the movement: does the answer follow the emotion?')
    print("   for each emotion, how often the arm's answer names that emotion")
    print('   the four classes are A angry, B happy, C sad, D neutral; CREMA-D has six emotions,')
    print('   so DIS (disgust) and FEA (fear) are scored under angry and SAD under sad')
    for arm in sorted(arms):
        if arm != 'frozen-base' and not arm.startswith('A-last4'):
            continue
        preds = arms[arm]
        by_id = {p['id']: p for p in preds}
        line = []
        for emo in EMOTIONS:
            m = emotion_of == emo
            idx = np.where(m)[0]
            if not len(idx):
                continue
            gold = np.array([CLASS_OF[e] for e in emotion_of[idx]])
            hit = np.mean([by_id[rows[k]['id']]['prediction'] == g for k, g in zip(idx, gold)])
            line.append(f'{emo} {100 * hit:4.1f}%')
        print(f'   {arm:16s} ' + '   '.join(line))


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else CROSS / 'cremad')
