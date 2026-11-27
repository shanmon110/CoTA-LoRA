"""Is the AQA negative transfer an artefact of the answer-letter prior?

Measurements, all from the tracked per-item predictions:

  skew      the distribution of answer letters per domain against the gold letter distribution.
            The gold letter is read from the prompt that was actually sent, so it is exact for
            every shift.
  remap ceiling
            the accuracy of the best fixed letter permutation of an arm's own answers, per
            (domain, number of options) cell, chosen with the gold. This is the ceiling of any
            correction of the form "re-map the model's letter answers", including a label-free
            one: if the ceiling is small, the deficit cannot be a letter-prior artefact.
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
import collections
import itertools
import json
import os
import re
import sys

RUN = sys.argv[1] if len(sys.argv) > 1 else str(REPO / 'hearside-aqa-20261125-224035' / 'mmau')
mcq = {json.loads(l)['id']: json.loads(l) for l in open(os.path.join(RUN, 'mcq.jsonl'))}
arms = sorted(os.listdir(os.path.join(RUN, 'runs')))
DOMAINS = ('music', 'sound', 'speech')
OPTION_RE = re.compile(r'^([A-H])\.\s*(.*)$')


def shown_options(row):
    """The option texts as they were shown, in display order, read from the prompt."""
    out = {}
    for line in row['messages'][0]['content'].split('\n'):
        m = OPTION_RE.match(line.strip())
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def gold_letter(row):
    shown = shown_options(row)
    gold = row['gold_text'].strip()
    for letter, text in shown.items():
        if text == gold:
            return letter
    return None


def n_options(row):
    return len(row['choices'])


rows_by_arm = {}
for arm in arms:
    rows_by_arm[arm] = [json.loads(l) for l in
                         open(os.path.join(RUN, 'runs', arm, 'mcq', 'predictions.jsonl'))]

print('=== sanity: the gold letter read from the prompt')
rows = rows_by_arm['frozen-base']
bad = sum(1 for g in rows if gold_letter(mcq[g['id']]) is None)
print(f'   frozen-base: rows whose gold letter could not be read: {bad} of {len(rows)}')
r = mcq[rows[0]['id']]
print('   example shown:', shown_options(r), '| gold', repr(r['gold_text']), '-> letter', gold_letter(r))

print()
print('=== the answer-letter distribution per domain')
for arm in arms:
    rows = rows_by_arm[arm]
    print(f'   {arm}')
    for dom in DOMAINS:
        subset = [(mcq[g['id']], g) for g in rows if mcq[g['id']]['domain'] == dom]
        ans = collections.Counter(g['prediction'].strip()[:1] for _, g in subset)
        gold = collections.Counter(gold_letter(m) for m, _ in subset)
        n = len(subset)
        print(f'        {dom:6s} n={n:4d}  ans ' + ' '.join(f'{k} {100 * ans[k] / n:5.1f}' for k in 'ABCD')
              + '  gold ' + ' '.join(f'{k} {100 * gold[k] / n:5.1f}' for k in 'ABCD'))

print()
print('=== the ceiling of a fixed letter remap, per domain (chosen with the gold, an upper bound)')
for arm in arms:
    rows = rows_by_arm[arm]
    out = []
    for dom in DOMAINS:
        subset = [(mcq[g['id']], g) for g in rows if mcq[g['id']]['domain'] == dom]
        cells = collections.defaultdict(list)
        for m, g in subset:
            cells[n_options(m)].append((gold_letter(m), g['prediction'].strip()[:1]))
        base_correct = sum(1 for items in cells.values() for gl, a in items if a == gl)
        best_correct = 0
        for k, items in cells.items():
            best = max((sum(1 for gl, a in items if dict(zip('ABCD'[:k], p)).get(a) == gl), p)
                       for p in itertools.permutations('ABCD'[:k]))
            best_correct += best[0]
        n = len(subset)
        out.append(f'{dom}: {100 * base_correct / n:6.2f} -> {100 * best_correct / n:6.2f} '
                   f'({100 * (best_correct - base_correct) / n:+5.2f} pp)')
    print(f'   {arm:14s} ' + ' | '.join(out))
