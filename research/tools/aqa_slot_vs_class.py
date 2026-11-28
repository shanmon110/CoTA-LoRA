"""P0b: is the AQA negative transfer a position shortcut or a content degeneration?

The layer-band question needs the same test on the second task that was run on IEMOCAP: is the
adapter's order sensitivity anchored to the SLOT (the display letter) or to the CLASS (what the
option says)?  `research/tools/slot_vs_class_anchor.py` runs that test on IEMOCAP Session 4,
where the four classes have a fixed letter-to-emotion correspondence in the training order.  AQA
has no such correspondence -- every question has its own options -- so on AQA the two readings
become:

  slot    the display letter the model picks
  class   the option TEXT the model picks, compared across the four cyclic orders of the SAME
          question

An arm that has learned a position shortcut answers the same letter whatever the options say;
an arm that has learned something about content answers the same option TEXT across the four
orders, because the content does not change when the options are rotated.

The measurements, all from the tracked per-item predictions on MMAU Test-mini:

  1. same text       the fraction of questions answered with the same option TEXT under all four
                     orders.  A collapse of this is a position shortcut.
  2. slot peak       the share of answers landing on the model's own most frequent letter,
                     per order.  A collapse of this is a position shortcut.
  3. accuracy by shift
                     the accuracy under each cyclic option order.  The training order is shift 0
                     (`aqa.py build` writes the original order first).  An arm that memorised
                     the training order is better at shift 0 and worse at the others; this is the
                     AQA version of the IEMOCAP order gap.
  4. accuracy by the gold's display position
                     the accuracy when the gold answer happens to sit in the first, second, ...
                     display slot.  A position shortcut shows up here as a systematic tilt.
  5. where the negative flips land
                     for every question the frozen base got right and the arm got wrong, whether
                     the arm's new answer is the arm's own preferred letter, and whether it is the
                     arm's preferred option TEXT.  A position shortcut concentrates the flips on
                     a few slots; a content degeneration scatters them.

The pre-declared reading (research/notes-20261127-diagnosis-and-plan.md, P0b):

  if the negative flips concentrate on the arm's preferred letter, or the arms collapse towards
  a fixed letter while their same-text rate falls, then part of the AQA negative transfer is a
  position shortcut, and chapter 5 has to be split in two;
  if the arms keep their same-text rate and their flips scatter, then the negative transfer is not
  a position shortcut but a degeneration of the representation, which is the more general claim
  and the one chapter 5 currently makes.

Run from the repository root:

    python3 -I research/tools/aqa_slot_vs_class.py [run folder]
"""
import collections
import json
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
RUN = pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                   else REPO / 'hearside-aqa-20261125-224035' / 'mmau')
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


def load():
    mcq = {}
    for line in open(RUN / 'mcq.jsonl'):
        row = json.loads(line)
        mcq[row['id']] = row
    arms = {}
    for arm in sorted(p.name for p in (RUN / 'runs').iterdir()):
        folder = RUN / 'runs' / arm / 'mcq' / 'predictions.jsonl'
        if folder.exists():
            arms[arm] = [json.loads(line) for line in open(folder)]
    return mcq, arms


def by_question(rows):
    """Question id -> {shift: prediction row}."""
    out = collections.defaultdict(dict)
    for row in rows:
        shift = int(row['id'].rsplit('#s', 1)[1])
        out[row['id'].rsplit('#s', 1)[0]][shift] = row
    return out


def main():
    mcq, arms = load()
    base = by_question(arms['frozen-base'])
    print(f'questions scored by the frozen base: {len(base)}')

    print()
    print('=' * 100)
    print('== 1 and 2: is the arm anchored to the letter or to the content?')
    print(f'   {"arm":16s} {"same text":>10s} {"slot peak":>10s} {"slot peak by shift":>34s}')
    print(f'   {"":16s} {"":10s} {"":10s} {"s0":>10s} {"s1":>10s} {"s2":>10s} {"s3":>10s}')
    anchor = {}
    for arm in arms:
        byq = by_question(arms[arm])
        qids = [q for q in byq if len(byq[q]) == 4]
        same_text = 0
        peaks = []
        per_shift = {s: [] for s in range(4)}
        for q in qids:
            shifts = byq[q]
            texts = [shifts[s]['answer_text'].strip() for s in sorted(shifts)]
            same_text += len(set(texts)) == 1
            for s in sorted(shifts):
                counts = collections.Counter(
                    shifts[t]['prediction'].strip()[:1] for t in sorted(shifts))
                per_shift[s].append(max(counts.values()) / 4)
        n = len(qids)
        anchor[arm] = (100 * same_text / n, per_shift)
        print(f'   {arm:16s} {100 * same_text / n:9.1f}% '
              + ' '.join(f'{100 * sum(per_shift[s]) / n:9.1f}%' for s in range(4)))

    print()
    print('=' * 100)
    print('== 3: accuracy by the cyclic option order (the training order is shift 0)')
    print(f'   {"arm":16s} {"overall":>8s} {"s0":>8s} {"s1":>8s} {"s2":>8s} {"s3":>8s} {"order gap":>10s}')
    order_gap = {}
    for arm in arms:
        rows = arms[arm]
        by_shift = collections.defaultdict(list)
        for row in rows:
            by_shift[int(row['id'].rsplit('#s', 1)[1])].append(row)
        accs = {}
        for s, items in by_shift.items():
            accs[s] = 100 * sum(r['is_correct'] for r in items) / len(items)
        overall = 100 * sum(r['is_correct'] for r in rows) / len(rows)
        others = [accs[s] for s in accs if s != 0]
        gap = accs.get(0, 0) - (sum(others) / len(others) if others else 0)
        order_gap[arm] = gap
        print(f'   {arm:16s} {overall:7.2f}% '
              + ' '.join(f'{accs.get(s, float("nan")):7.2f}%' for s in range(4))
              + f' {gap:+9.2f}')

    print()
    print('=' * 100)
    print('== 4: accuracy by the gold answer\'s display position (the slot the gold sits in)')
    print(f'   {"arm":16s} ' + ' '.join(f'{c:>8s}' for c in 'ABCD'))
    for arm in arms:
        rows = arms[arm]
        cells = collections.defaultdict(list)
        for row in rows:
            m = mcq[row['id']]
            gl = gold_letter(m)
            if gl:
                cells[gl].append(row['is_correct'])
        print(f'   {arm:16s} '
              + ' '.join(f'{100 * sum(cells[c]) / len(cells[c]):7.2f}%' if cells[c] else '      --'
                         for c in 'ABCD'))

    print()
    print('=' * 100)
    print('== 5: where the negative flips land')
    print('   a flip is a question the frozen base got right and the arm got wrong')
    for arm in arms:
        if arm == 'frozen-base':
            continue
        rows = arms[arm]
        byq = by_question(rows)
        pref_letter = collections.Counter(r['prediction'].strip()[:1] for r in rows).most_common(1)[0][0]
        pref_text = collections.Counter(r['answer_text'].strip() for r in rows).most_common(1)[0][0]
        flips = 0
        on_pref_letter = 0
        on_pref_text = 0
        flip_letters = collections.Counter()
        for q in byq:
            if q not in base or len(byq[q]) != 4 or len(base[q]) != 4:
                continue
            for s in sorted(byq[q]):
                b, a = base[q][s], byq[q][s]
                if not b['is_correct'] or a['is_correct']:
                    continue
                flips += 1
                letter = a['prediction'].strip()[:1]
                flip_letters[letter] += 1
                on_pref_letter += letter == pref_letter
                on_pref_text += a['answer_text'].strip() == pref_text
        if not flips:
            continue
        share = 100 * on_pref_letter / flips
        base_share = 100 * sum(1 for r in arms['frozen-base']
                               if r['prediction'].strip()[:1] == pref_letter) / len(arms['frozen-base'])
        print(f'   {arm:16s} flips {flips:5d}  on its preferred letter {pref_letter} '
              f'{share:5.1f}% (the base uses it {base_share:4.1f}% of the time)  '
              f'on its preferred text {100 * on_pref_text / flips:5.1f}%  '
              f'letters ' + ' '.join(f'{c} {100 * n / flips:4.1f}' for c, n in sorted(flip_letters.items())))

    print()
    print('=' * 100)
    print('== summary')
    base_text, _ = anchor['frozen-base']
    for arm in arms:
        if arm == 'frozen-base':
            continue
        text_rate, _ = anchor[arm]
        print(f'   {arm:16s} same text {text_rate:5.1f}% (base {base_text:5.1f}%, '
              f'{text_rate - base_text:+5.1f})   order gap {order_gap[arm]:+6.2f} pp '
              f'(base {order_gap["frozen-base"]:+6.2f})')


if __name__ == '__main__':
    main()
