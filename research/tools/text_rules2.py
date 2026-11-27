"""Text scoring rules compared, on the same predictions.

Rules
  strict        the prediction equals the gold option text (exact string)
  norm          equal after case folding, whitespace collapsing and trailing punctuation
  contains      the gold option text occurs inside the prediction
  prefix        the prediction is a prefix of the gold option text
  repo_by_text  the repository's rule (BY_TEXT): names exactly one option, or is a
                truncated start of exactly one, at a word boundary
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

sys.path.insert(0, str(REPO))

R = sys.argv[1] if len(sys.argv) > 1 else str(REPO / 'hearside-aqa-20261125-224035' / 'mmau')
man = {json.loads(l)['id']: json.loads(l) for l in open(R + '/text.jsonl')}
arms = sorted(os.listdir(R + '/runs'))


def norm(s):
    return ' '.join(s.strip().casefold().rstrip('.!?').split())


def normal(text):
    return ' '.join(re.sub(r'[^\w\s]', ' ', text.casefold()).split())


def BY_TEXT(row, answer):
    """The repository's `aqa.by_text`: names exactly one option, or is a truncated start of
    exactly one, at a word boundary."""
    said, options = normal(answer), {normal(c) for c in row['choices']}
    named = {said} & options or {o for o in options if o and re.search(rf'(?<!\w){re.escape(o)}(?!\w)', said)}
    if len(named) == 1:
        return named == {normal(row['gold_text'])}, True
    starts = {o for o in options if o and len(said) >= 2 and re.match(rf'(?<!\w){re.escape(said)}(?!\w)', o)}
    if len(starts) == 1:
        return starts == {normal(row['gold_text'])}, True
    return False, False


def BY_LETTER(row, answer):
    answer = answer.strip()
    if len(answer) != 1 or answer not in 'ABCDEFGH'[:len(row['choices'])]:
        return False, False
    return row['choices']['ABCDEFGH'.index(answer)] == row['gold_text'], True


print(f'{"arm":14s} {"strict":>7s} {"norm":>7s} {"contains":>9s} {"prefix":>7s} {"by_text":>8s} {"parse%":>8s}')
for arm in arms:
    rows = [json.loads(l) for l in open(f'{R}/runs/{arm}/text/predictions.jsonl')]
    repo = [BY_TEXT(man[g['id']], g['prediction']) for g in rows]
    strict = [g['prediction'].strip() == g['reference'].strip() for g in rows]
    nrm = [norm(g['prediction']) == norm(g['reference']) for g in rows]
    cont = [g['reference'].strip().casefold() in g['prediction'].strip().casefold() for g in rows]
    pref = [bool(g['reference'].strip()) and bool(g['prediction'].strip())
            and g['reference'].strip().casefold().startswith(g['prediction'].strip().casefold()) for g in rows]
    n = len(rows)
    print(f'{arm:14s} {100 * sum(strict) / n:7.1f} {100 * sum(nrm) / n:7.1f} {100 * sum(cont) / n:9.1f} '
          f'{100 * sum(pref) / n:7.1f} {100 * sum(c[0] for c in repo) / n:8.1f} {100 * sum(c[1] for c in repo) / n:8.1f}')
