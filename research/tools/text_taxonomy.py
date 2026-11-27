"""Text taxonomy for the AQA text format: what kind of answer each arm gives.

Categories (mutually exclusive, first match wins):
  exact          the prediction string equals the gold option text
  contains      the gold option text is a substring of the prediction
  prefix        the prediction is a prefix of the gold option text
  names_other   the prediction names some option other than the gold one
  names_none    the prediction names no option at all
  none_prefix   names_none, and the prediction is a prefix of the gold text
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
import sys

R = sys.argv[1] if len(sys.argv) > 1 else str(REPO / 'hearside-aqa-20261125-224035' / 'mmau')
man = {json.loads(l)['id']: json.loads(l) for l in open(R + '/text.jsonl')}
arms = sorted(os.listdir(R + '/runs'))
print(f'{"arm":14s} {"exact":>7s} {"contains":>9s} {"prefix":>7s} {"names_other":>12s} {"names_none":>11s} {"none_prefix":>12s}')
for arm in arms:
    rows = [json.loads(l) for l in open(f'{R}/runs/{arm}/text/predictions.jsonl')]
    c_exact = c_cont = c_pref = c_other = c_none = c_nonepref = 0
    for g in rows:
        said = g['prediction'].strip()
        gold = g['reference'].strip()
        sl = said.casefold()
        gl = gold.casefold()
        choices = [c.casefold() for c in man[g['id']]['choices']]
        if said == gold:
            c_exact += 1
            continue
        if gl and gl in sl:
            c_cont += 1
            continue
        if gl and sl and gl.startswith(sl):
            c_pref += 1
            continue
        named = [c for c in choices if c and (c in sl or (sl and c.startswith(sl)))]
        if named:
            c_other += 1
        else:
            c_none += 1
            if sl and gl and gl.startswith(sl):
                c_nonepref += 1
    n = len(rows)
    print(f'{arm:14s} {100 * c_exact / n:7.1f} {100 * c_cont / n:9.1f} {100 * c_pref / n:7.1f} '
          f'{100 * c_other / n:12.1f} {100 * c_none / n:11.1f} {100 * c_nonepref / n:12.1f}')
