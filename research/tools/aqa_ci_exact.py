"""Recompute the AQA contrasts with the repository's own bootstrap machinery, independently.

The repository's `summarize` uses `formats.Scores`, which resamples *dialogue* clusters (for AQA
the `dialogue` field is the audio file) and averages the three seeds' bootstrap draws with an
independent seed draw.  This script re-derives the same quantities from the raw predictions, with
the same judge functions imported from the repository, and additionally a variant in which the
seeds are resampled *jointly* with the clusters (the variant the handoff asks for).
"""
"""Independent recomputation of the AQA numbers. Run from the repository root:

    python3 -I research/tools/<this file>.py

The run folders live outside the tracked tree (they are the server run folders copied in);
the tools find them by name under the repository root.
"""
import os
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[2]
RESULTS = REPO / 'experiments' / 'hearside_pilot' / 'results'
import json
import os
import re
import sys

import numpy as np


RUN = str(REPO / 'hearside-aqa-20261125-224035' / 'mmau')
FAMILIES = {a: [a, a + '-s43', a + '-s44'] for a in ('AQ-first4', 'AQ-mid4', 'AQ-last4', 'AQ-all')}


def normal(text):
    return ' '.join(re.sub(r'[^\w\s]', ' ', text.casefold()).split())


def BY_LETTER(row, answer):
    """The repository's `aqa.by_letter`: the letter must name the gold option."""
    answer = answer.strip()
    if len(answer) != 1 or answer not in 'ABCDEFGH'[:len(row['choices'])]:
        return False, False
    return row['choices']['ABCDEFGH'.index(answer)] == row['gold_text'], True


def BY_TEXT(row, answer):
    """The repository's `aqa.by_text`: equal to the gold option, or the only option it contains,
    or the unique option it is a word-boundary prefix of."""
    said, options = normal(answer), {normal(c) for c in row['choices']}
    named = {said} & options or {o for o in options if o and re.search(rf'(?<!\w){re.escape(o)}(?!\w)', said)}
    if len(named) == 1:
        return named == {normal(row['gold_text'])}, True
    starts = {o for o in options if o and len(said) >= 2 and re.match(rf'(?<!\w){re.escape(said)}(?!\w)', o)}
    if len(starts) == 1:
        return starts == {normal(row['gold_text'])}, True
    return False, False


class Scores:
    """The repository's `formats.Scores`: mean per-group recall, with a cluster bootstrap over
    the dialogue (for AQA, the audio file).  Copied here so that this recomputation does not
    depend on the repository's import chain (which reaches scipy)."""

    def __init__(self, manifest, name, resamples, rng, judge=None):
        self.manifest, self.name = manifest, name
        self.judge = judge or (lambda row, text: (False, False))
        groups = sorted({r['group'] for r in manifest})
        names = sorted({r['dialogue'] for r in manifest})
        self.group = np.array([groups.index(r['group']) for r in manifest])
        self.dialogue = np.array([names.index(r['dialogue']) for r in manifest])
        self.shape = (len(names), len(groups))
        draws = rng.integers(0, len(names), size=(resamples, len(names)))
        self.weights = np.stack([np.bincount(row, minlength=len(names)) for row in draws]).astype(float)
        self.total = self.count(np.ones(len(manifest)))
        self.valid = (self.weights @ self.total > 0).all(axis=1)

    def count(self, values):
        out = np.zeros(self.shape)
        np.add.at(out, (self.dialogue, self.group), values)
        return out

    def measure(self, rows):
        if [r['id'] for r in rows] != [m['id'] for m in self.manifest]:
            raise ValueError(f'{self.name}: predictions do not cover the manifest in order')
        judged = [self.judge(m, r['prediction']) for m, r in zip(self.manifest, rows)]
        hits = self.count(np.array([j[0] for j in judged], float))
        with np.errstate(divide='ignore', invalid='ignore'):
            boot = ((self.weights @ hits) / (self.weights @ self.total)).mean(axis=1)
        return dict(score=float((hits.sum(0) / self.total.sum(0)).mean()), parseable=float(np.mean([j[1] for j in judged])),
                    recalls=[float(v) for v in hits.sum(0) / self.total.sum(0)]), np.where(self.valid, boot, np.nan)


def interval(boot):
    low, high = np.nanpercentile(boot, [2.5, 97.5])
    return [100 * float(low), 100 * float(high)]


def read(p):
    return [json.loads(l) for l in open(p)]


mcq = read(os.path.join(RUN, 'mcq.jsonl'))
arms = sorted(os.listdir(os.path.join(RUN, 'runs')))
pred = {(a, t): read(os.path.join(RUN, 'runs', a, t, 'predictions.jsonl'))
        for a in arms for t in ('mcq', 'text')}

RES = 10000
rng = np.random.default_rng(42)
value, boot = {a: {} for a in arms}, {a: {} for a in arms}
for name, keep in (('original_order', [r['shift'] == 0 for r in mcq]),
                   ('shifted_orders', [r['shift'] > 0 for r in mcq])):
    sub = [r for r, k in zip(mcq, keep) if k]
    sc = Scores(sub, name, RES, np.random.default_rng(42), judge=BY_LETTER)
    for a in arms:
        v, b = sc.measure([r for r, k in zip(pred[a, 'mcq'], keep) if k])
        value[a][name], boot[a][name] = v['score'], b
text_manifest = read(os.path.join(RUN, 'text.jsonl'))
sc = Scores(text_manifest, 'text', RES, np.random.default_rng(42), judge=BY_TEXT)
for a in arms:
    v, b = sc.measure(pred[a, 'text'])
    value[a]['text'], boot[a]['text'] = v['score'], b

print('=== point estimates (seed-averaged) and the repository bootstrap')
draws = {}


def mean_boot(members, name, joint=False):
    """joint=True: the same seed draw index for every bootstrap resample (seeds resampled together
    with the clusters), which is the conservative reading."""
    if joint:
        idx = np.arange(RES) % len(members)
        stacked = np.stack([boot[m][name] for m in members])
        return stacked[idx, np.arange(RES)].mean(axis=0)
    if len(members) not in draws:
        draws[len(members)] = np.random.default_rng(43).integers(0, len(members), size=(RES, len(members)))
    stacked = np.stack([boot[m][name] for m in members])
    return stacked[draws[len(members)].T, np.arange(RES)].mean(axis=0)


groups = {'frozen-base': ['frozen-base']}
groups.update({f: [m for m in ms if m in arms] for f, ms in FAMILIES.items() if any(m in arms for m in ms)})
for g, ms in groups.items():
    if g == 'frozen-base':
        continue
    for name in ('original_order', 'shifted_orders', 'text'):
        delta = 100 * (float(np.mean([value[m][name] for m in ms])) - value['frozen-base'][name])
        rep = interval(mean_boot(ms, name) - boot['frozen-base'][name])
        jnt = interval(mean_boot(ms, name, joint=True) - boot['frozen-base'][name])
        print(f'   {g:10s} {name:15s} delta {delta:+6.2f}   repo-style CI [{rep[0]:+.2f}, {rep[1]:+.2f}]'
              f'   seed-joint CI [{jnt[0]:+.2f}, {jnt[1]:+.2f}]')
