"""Compare the pilot arms on Session 4 and apply the criteria declared in the README.

Reads the per-condition predictions that `experiments.trr_diagnostics.d2 run` wrote for each arm.
Everything is paired on the same utterances and option orders; intervals are cluster bootstrap
over dialogues. One training seed per arm: the intervals describe evaluation noise only, not
what a second training run would give.
"""
import argparse
import json
from pathlib import Path

from experiments.trr_diagnostics import d2, stats

BASE, PLAIN, HEAR, ANSWER, SHUFFLED = 'frozen-base', 'A-all', 'B-audio', 'C-post-audio', 'D-all-shuffled'
ARMS = [BASE, PLAIN, HEAR, ANSWER, SHUFFLED]
PAIRS = [(HEAR, PLAIN), (ANSWER, PLAIN), (SHUFFLED, PLAIN), (HEAR, SHUFFLED), (HEAR, ANSWER),
         (PLAIN, BASE), (HEAR, BASE), (ANSWER, BASE), (SHUFFLED, BASE)]
# Written down before any arm was trained; see README.md.
CRITERIA = dict(max_drop_in_training_format_pp=3.0, min_gain_in_other_orders_pp=10.0)
REWORDED = 'orig_reworded'


def load_letters(prepared, info, arm, condition):
    """Answer letters of one arm for one condition, or None if that condition was not scored.
    The stored hashes of the manifest and of the predictions are checked on the way."""
    rows = d2.completed_predictions(prepared, prepared / 'runs' / arm, condition, info)
    return None if rows is None else [r['prediction'] for r in rows]


def load_asr(prepared, info, arm, condition='asr-orig'):
    """id -> (word errors, reference words), scored after removing a "The transcript is:" lead-in."""
    rows = d2.completed_predictions(prepared, prepared / 'runs' / arm, condition, info)
    if rows is None:
        return None
    return {r['id']: stats.transcript_errors(r['reference'], r['prediction'])[:2] if 'prediction' in r
            else (r['word_errors'], r['reference_words']) for r in rows}


def summarize(prepared, arms=ARMS, *, resamples=10000, seed=42):
    prepared = Path(prepared)
    cfg, info = d2.validate_prepared(prepared)
    classes = cfg['classes']
    manifest = d2.read_jsonl(prepared / 'manifests' / 'orig.jsonl')
    labels = [m['label'] for m in manifest]
    clusters = [m['dialogue'] for m in manifest]
    subsets = {'training_order': [m['rotation'] == 0 for m in manifest],
               'other_orders': [m['rotation'] != 0 for m in manifest],
               'all_orders': [True] * len(manifest)}
    conditions = ['orig', *cfg['conditions'].get('reworded', {}), *cfg['conditions']['text']]
    letters = {arm: {c: load_letters(prepared, info, arm, c) for c in conditions} for arm in arms}
    # A letter is scored as the class it stood for under that row's option order.
    preds = {arm: {c: None if v is None else [m['option_order'][classes.index(x)] for x, m in zip(v, manifest)]
                   for c, v in by.items()} for arm, by in letters.items()}
    asr = {arm: load_asr(prepared, info, arm) for arm in arms}

    def score(values, keep):
        return stats.classification([(y, p) for y, p, k in zip(labels, values, keep) if k], classes)

    table = {}
    for arm in arms:
        row = {}
        for condition in conditions:
            if preds[arm][condition] is None:
                continue
            scored = {name: score(preds[arm][condition], keep) for name, keep in subsets.items()}
            row[condition] = dict(uar={name: s['uar'] for name, s in scored.items()},
                                  recalls={name: s['recalls'] for name, s in scored.items()},
                                  letter_share={c: letters[arm][condition].count(c) / len(labels) for c in classes})
        if asr[arm] is not None:
            row['asr_wer'] = sum(e for e, _ in asr[arm].values()) / sum(n for _, n in asr[arm].values())
        table[arm] = row

    def contrast(candidate, reference, condition, subset):
        a, b = preds.get(candidate, {}).get(condition), preds.get(reference, {}).get(condition)
        if a is None or b is None:
            return None
        idx = [i for i, keep in enumerate(subsets[subset]) if keep]
        out = stats.paired_contrast([labels[i] for i in idx], [a[i] for i in idx], [b[i] for i in idx],
                                    [clusters[i] for i in idx], classes, resamples=resamples, seed=seed)
        return dict(delta_uar_pp=out['delta_uar_pp'], interval_95_pp=out['uar_interval_95_pp'],
                    answers_changed=out['prediction_changed_rate'])

    contrasts = {}
    for candidate, reference in PAIRS:
        for condition in conditions:
            for subset in subsets:
                value = contrast(candidate, reference, condition, subset)
                if value is not None:
                    contrasts[f'{candidate} - {reference} | {condition} | {subset}'] = value
    dialogue = {m['utterance']: m['dialogue'] for m in manifest}
    wer = {}
    for candidate, reference in PAIRS:
        if asr.get(candidate) is not None and asr.get(reference) is not None:
            out = stats.wer_contrast(asr[candidate], asr[reference], dialogue, resamples=resamples, seed=seed)
            wer[f'{candidate} - {reference}'] = dict(delta_wer_pp=out['delta_wer_pp'], interval_95_pp=out['interval_95_pp'])

    return dict(prepared=str(prepared), arms=list(arms), criteria=CRITERIA, utterances=len({m['utterance'] for m in manifest}),
                scored_rows=len(manifest), dialogues=len(set(clusters)), results=table, contrasts=contrasts,
                wer_contrasts=wer, verdict=decide(contrasts, wer))


def decide(contrasts, wer):
    """The continue/stop rule written down before training. A missing arm leaves its check undecided."""
    def delta(key):
        return None if key not in contrasts else contrasts[key]['delta_uar_pp']

    checks = {}
    d = delta(f'{HEAR} - {PLAIN} | orig | training_order')
    checks['hear_side_learns_the_task'] = None if d is None else d >= -CRITERIA['max_drop_in_training_format_pp']
    d = delta(f'{HEAR} - {BASE} | orig | other_orders')
    checks['hear_side_not_below_base_in_other_orders'] = None if d is None else d >= 0
    d = delta(f'{HEAR} - {PLAIN} | orig | other_orders')
    checks['hear_side_beats_plain_in_other_orders'] = None if d is None else d >= CRITERIA['min_gain_in_other_orders_pp']
    reworded = contrasts.get(f'{HEAR} - {SHUFFLED} | {REWORDED} | all_orders')
    words = wer.get(f'{HEAR} - {SHUFFLED}')
    if reworded is None and words is None:
        checks['hear_side_beats_shuffling_somewhere'] = None
    else:
        better_reworded = reworded is not None and reworded['interval_95_pp'][0] > 0     # higher UAR on unseen wording
        better_words = words is not None and words['interval_95_pp'][1] < 0              # lower WER on transcription
        checks['hear_side_beats_shuffling_somewhere'] = better_reworded or better_words
    values = list(checks.values())
    outcome = 'stop' if False in values else 'undecided' if None in values else 'continue'
    return dict(checks=checks, outcome=outcome)


def render(result):
    def pct(value):
        return 'n/a' if value is None else f'{100 * value:.1f}'

    lines = ['# Hear-side pilot: Session 4', '',
             f"{result['utterances']} utterances, {result['scored_rows']} rows over the option orders, "
             f"{result['dialogues']} dialogues. UAR in %, scored on classes.",
             'One training seed per arm: the intervals cover evaluation noise only.', '',
             '| arm | recordings, training order | recordings, other orders | reworded question, all orders | '
             'transcript only | question only | transcription WER % |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for arm, row in result['results'].items():
        def cell(condition, subset):
            return pct(row.get(condition, {}).get('uar', {}).get(subset))
        lines.append(f"| {arm} | {cell('orig', 'training_order')} | {cell('orig', 'other_orders')} | "
                     f"{cell(REWORDED, 'all_orders')} | {cell('text', 'all_orders')} | "
                     f"{cell('question_only', 'all_orders')} | {pct(row.get('asr_wer'))} |")
    lines += ['', f'{HEAR} adds nothing to an input without audio, so its transcript-only and question-only '
              'scores equal the frozen base by construction.', '',
              'Share of each answer letter on the recordings (the gold letters are balanced over the four orders):', '',
              '| arm | A | B | C | D |', '|---|---:|---:|---:|---:|']
    for arm, row in result['results'].items():
        if 'orig' in row:
            lines.append(f'| {arm} | ' + ' | '.join(pct(v) for v in row['orig']['letter_share'].values()) + ' |')
    lines += ['', '| contrast | input | option orders | dUAR pp | 95% CI | answers changed % |', '|---|---|---|---:|---|---:|']
    for key, c in result['contrasts'].items():
        pair, condition, subset = key.split(' | ')
        if condition not in ('orig', REWORDED) and subset != 'all_orders':
            continue                                   # inputs without audio: one row each; summary.json has the rest
        lines.append(f"| {pair} | {condition} | {subset} | {c['delta_uar_pp']:+.2f} | "
                     f"[{c['interval_95_pp'][0]:+.2f}, {c['interval_95_pp'][1]:+.2f}] | {pct(c['answers_changed'])} |")
    if result['wer_contrasts']:
        lines += ['', '| transcription contrast | dWER pp | 95% CI |', '|---|---:|---|']
        lines += [f"| {key} | {w['delta_wer_pp']:+.2f} | [{w['interval_95_pp'][0]:+.2f}, {w['interval_95_pp'][1]:+.2f}] |"
                  for key, w in result['wer_contrasts'].items()]
    lines += ['', 'Checks declared before training:', '']
    lines += [f"- {name}: **{'not available' if value is None else 'yes' if value else 'no'}**"
              for name, value in result['verdict']['checks'].items()]
    lines += ['', f"Outcome under the declared rule: **{result['verdict']['outcome']}**"]
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--arms', nargs='+', default=ARMS)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(args.prepared, args.arms)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    text = render(summary)
    (args.output / 'summary.md').write_text(text)
    print(text)
