"""P0, step 1: does the AQA negative transfer fall on the questions that need audio?

The AQA arms are all below the frozen base on MMAU.  Three explanations are open, and this
tool separates the first two:

  audio pathway   the arms lost the ability to use the audio, so the loss falls on the
                  questions that need the audio;
  language drift  the arms shifted the language distribution, so the loss falls on the
                  questions that the frozen base could already answer from the text alone.

The proxy for "needs audio" is the frozen base's own TEXT-format answer.  The text format is
the same question with no audio at all, so:

  answerable without audio   the frozen base's text-format prediction equals the gold text.
                             The base gets these right with no audio, so the audio was not
                             needed to answer them;
  needs the audio            the frozen base's text-format prediction is wrong.  The base can
                             only get these right by listening, so the audio is doing the work.

This is a proxy, not a ground truth: a question the base also gets wrong without audio might
still need audio (the base might just be bad at it).  The proxy is only used to STRATIFY, and
both strata are reported, so the reading does not depend on the proxy being right.

What is measured, per arm, against the frozen base:

  net flips      the arm's answers that changed, minus the ones that became wrong.  This is
                 the same quantity as the recomputation's "net flips".
  by stratum     the same, split by the stratum of the question.

The pre-declared reading (notes-20261127-diagnosis-and-plan.md, P0, step 1):

  if the deficit concentrates on the "needs the audio" stratum, the loss is on the audio
  pathway, and the layer-band question was tested on the second task;
  if it concentrates on the "answerable without audio" stratum, the loss is a language
  distribution shift, and the second task's question format is what moved.

Run from the repository root:

    python3 -I research/tools/aqa_audio_strata.py [run folder]
"""
import collections
import json
import os
import pathlib
import sys

RUN = sys.argv[1] if len(sys.argv) > 1 else '/root/autodl-fs/CoTA-LoRA/hearside-aqa-20261125-224035/mmau'
DOMAINS = ('music', 'sound', 'speech')


def read_jsonl(path):
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load(run):
    """The MCQ manifest, the tracked MCQ answers and the tracked text answers, per arm."""
    mcq_manifest = {row['id']: row for row in read_jsonl(os.path.join(run, 'mcq.jsonl'))}
    text_manifest = {row['id']: row for row in read_jsonl(os.path.join(run, 'text.jsonl'))}

    # the text format has no option order, so it is keyed by the question
    text_answers = {}
    for arm in sorted(os.listdir(os.path.join(run, 'runs'))):
        path = os.path.join(run, 'runs', arm, 'text', 'predictions.jsonl')
        if not os.path.exists(path):
            continue
        text_answers[arm] = {row['id']: row for row in read_jsonl(path)}

    mcq_answers = {}
    for arm in sorted(os.listdir(os.path.join(run, 'runs'))):
        path = os.path.join(run, 'runs', arm, 'mcq', 'predictions.jsonl')
        if not os.path.exists(path):
            continue
        mcq_answers[arm] = {row['id']: row for row in read_jsonl(path)}
    return mcq_manifest, text_manifest, mcq_answers, text_answers


def stratum_of(question, text_manifest, base_text):
    """'audio' if the frozen base's text-format answer is wrong, 'text' if it is right."""
    row = text_manifest.get(question)
    if row is None:
        return None
    gold = row['gold_text']
    pred = base_text.get(question, {}).get('prediction')
    if pred is None:
        return None
    return 'text' if pred.strip() == gold.strip() else 'audio'


def main():
    mcq_manifest, text_manifest, mcq_answers, text_answers = load(RUN)
    arms = sorted(mcq_answers)
    if 'frozen-base' not in arms:
        raise SystemExit('the frozen base is not among the tracked arms')

    base_mcq = mcq_answers['frozen-base']
    base_text = text_answers.get('frozen-base', {})

    # the stratum of each QUESTION, from the frozen base's text-format answer
    stratum = {}
    for question in text_manifest:
        s = stratum_of(question, text_manifest, base_text)
        if s is not None:
            stratum[question] = s

    questions = sorted({row['source'] for row in mcq_manifest.values()})
    covered = [q for q in questions if q in stratum]
    n_audio = sum(1 for q in covered if stratum[q] == 'audio')
    n_text = len(covered) - n_audio
    print('=' * 100)
    print('== P0 step 1: where does the AQA negative transfer fall, audio or language?')
    print(f'   the proxy is the frozen base\'s own text-format answer on the {len(covered)} '
          f'questions that have one')
    print(f'   answerable without audio (the base got the text right): {n_text}')
    print(f'   needs the audio (the base got the text wrong):          {n_audio}')
    print()

    # the base's own accuracy in each stratum, as a reference point
    def base_acc(ids):
        hit = tot = 0
        for i in ids:
            row = mcq_answers['frozen-base'].get(i)
            if row is None:
                continue
            hit += bool(row['is_correct'])
            tot += 1
        return (100 * hit / tot) if tot else float('nan'), tot

    all_ids = list(mcq_manifest)
    ids_text = [i for i in all_ids if stratum.get(mcq_manifest[i]['source']) == 'text']
    ids_audio = [i for i in all_ids if stratum.get(mcq_manifest[i]['source']) == 'audio']
    a_text, n_t = base_acc(ids_text)
    a_audio, n_a = base_acc(ids_audio)
    print(f'   the frozen base: {a_text:.1f}% on the answerable-without-audio rows ({n_t} rows), '
          f'{a_audio:.1f}% on the needs-audio rows ({n_a} rows)')
    print()
    # The two strata are NOT equally hard for the base -- it scores 91.6% on one and 33.7% on
    # the other -- so a raw net-flip count is confounded by the headroom.  The comparable
    # measure is the LOST rate: of the rows the base got RIGHT, the fraction the arm got
    # wrong.  It is bounded and does not depend on how many rows the base got right.
    print(f'   {"arm":16s} {"all":>22s} {"answerable w/o audio":>26s} {"needs the audio":>26s}')
    print(f'   {"":16s} {"acc   lost  fixed":>22s} {"acc   lost  fixed":>26s} '
          f'{"acc   lost  fixed":>26s}')

    for arm in arms:
        cells = []
        for ids in (all_ids, ids_text, ids_audio):
            net = 0
            hit = tot = 0
            lost_den = lost_num = fixed_den = fixed_num = 0
            for i in ids:
                row = mcq_answers[arm].get(i)
                base = base_mcq.get(i)
                if row is None or base is None:
                    continue
                tot += 1
                hit += bool(row['is_correct'])
                if row['prediction'] != base['prediction']:
                    net += (1 if row['is_correct'] else 0) - (1 if base['is_correct'] else 0)
                if base['is_correct']:
                    lost_den += 1
                    lost_num += not row['is_correct']
                else:
                    fixed_den += 1
                    fixed_num += bool(row['is_correct'])
            acc = 100 * hit / tot if tot else float('nan')
            lost = 100 * lost_num / lost_den if lost_den else float('nan')
            fixed = 100 * fixed_num / fixed_den if fixed_den else float('nan')
            cells.append((net, acc, lost, fixed, tot))
        name = arm if arm != 'frozen-base' else '  (the base)'
        allc, tc, ac = cells
        print(f'   {name:16s} {allc[1]:6.1f}% {allc[2]:5.1f}% {allc[3]:6.1f}% '
              f'{tc[1]:6.1f}% {tc[2]:5.1f}% {tc[3]:6.1f}% '
              f'{ac[1]:6.1f}% {ac[2]:5.1f}% {ac[3]:6.1f}%')

    # The bootstrap over the QUESTIONS (the independent unit) for the difference in loss rate
    # between the two strata.  The question is the unit because the four option orders of one
    # question are not independent.
    print()
    print('   the loss rate in each stratum, and the difference, with a bootstrap over questions')
    import numpy as np
    rng = np.random.default_rng(42)
    qnames = sorted({mcq_manifest[i]['source'] for i in all_ids})
    qindex = {q: k for k, q in enumerate(qnames)}

    def lost_by_question(arm, want):
        """For each question, the base-correct rows and the arm-wrong-among-them."""
        base_ok = np.zeros(len(qnames))
        lost = np.zeros(len(qnames))
        for i in all_ids:
            q = mcq_manifest[i]['source']
            if stratum.get(q) != want:
                continue
            row = mcq_answers[arm].get(i)
            b = base_mcq.get(i)
            if row is None or b is None:
                continue
            k = qindex[q]
            if b['is_correct']:
                base_ok[k] += 1
                lost[k] += not row['is_correct']
        return base_ok, lost

    print(f'   {"arm":16s} {"lost w/o audio":>16s} {"lost needs audio":>18s} {"difference":>26s}')
    for arm in arms:
        if arm == 'frozen-base':
            continue
        bo_t, lo_t = lost_by_question(arm, 'text')
        bo_a, lo_a = lost_by_question(arm, 'audio')

        def rates(bo, lo, weights):
            den = (bo * weights).sum()
            return 100 * (lo * weights).sum() / den if den else float('nan')

        w = np.ones(len(qnames))
        rt = rates(bo_t, lo_t, w)
        ra = rates(bo_a, lo_a, w)
        draws = []
        for _ in range(2000):
            picked = rng.integers(0, len(qnames), len(qnames))
            wt = np.bincount(picked, minlength=len(qnames)).astype(float)
            a1 = rates(bo_t, lo_t, wt)
            a2 = rates(bo_a, lo_a, wt)
            draws.append(a2 - a1)
        lo_ci, hi_ci = np.percentile(draws, [2.5, 97.5])
        print(f'   {arm:16s} {rt:14.1f}% {ra:16.1f}% {ra - rt:+12.1f} pp '
              f'[{lo_ci:+.1f}, {hi_ci:+.1f}]')

    # the per-domain breakdown, so the strata are not confounded with the domain mix
    print()
    print('   the same loss rate, split by domain (the base correctness per domain is shown)')
    # Not every corpus labels its rows by domain.  The MMAU one does; the held-out AVQA one
    # does not, and there the breakdown is simply absent rather than wrong.
    available = [d for d in DOMAINS if any('domain' in mcq_manifest.get(i, {}) for i in all_ids)]
    if not available:
        print('   (this corpus does not label its rows by domain)')
        return
    for domain in available:
        ids = [i for i in all_ids if mcq_manifest[i].get('domain') == domain]
        base_ok = sum(1 for i in ids if base_mcq.get(i, {}).get('is_correct'))
        line = [f'   {domain:9s} base {100 * base_ok / max(len(ids), 1):5.1f}% of {len(ids):5d} rows']
        for arm in arms:
            if arm == 'frozen-base':
                continue
            ok = sum(1 for i in ids if base_mcq.get(i, {}).get('is_correct'))
            lost = sum(1 for i in ids if base_mcq.get(i, {}).get('is_correct')
                       and not mcq_answers[arm].get(i, {}).get('is_correct'))
            line.append(f'{arm} {100 * lost / max(ok, 1):.1f}%')
        print('   ' + '  '.join(line))

    print()
    print('"lost" is the percentage of the base CORRECT rows the arm got wrong; "fixed" is')
    print('the percentage of the base WRONG rows the arm got right.  The net flip count is')
    print('also in the table, but it is confounded: the two strata are not equally hard for')
    print('the base, so the stratum with more headroom can lose more rows for the same rate.')
    print()
    print('The pre-declared reading: if the LOSS RATE falls on the "needs the audio" rows, the')
    print('loss is on the audio pathway; if it falls on the "answerable without audio" rows,')
    print('the loss is a language distribution shift.  The proxy is imperfect -- the base can')
    print('be wrong without audio for a question that still needs audio -- so both strata are')
    print('reported and the reading does not rest on the proxy being right.')


if __name__ == '__main__':
    main()
