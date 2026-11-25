"""A second task for the layer-band question: adapters trained on AVQA, scored on MMAU Test-mini.

Four arms, three seeds each, on the same base and with the same adapter as the emotion runs (O projection,
rank 8): layers 0-3, 16-19, 32-35 and all 36. Unlike the emotion task, every question has its own options
and the right answer sits at a different letter from question to question.

    configs    one training config per arm
    build      the test questions under shifted option orders, and in a format without letters
    run        score one model
    summarize  compare; readings declared in research/second-task-and-grpo-20261125.md

Written before any of these arms was trained.
"""
import argparse
import copy
import json
import re
from pathlib import Path
from string import ascii_uppercase

import numpy as np

from experiments.hearside_pilot import configs as pilot
from experiments.hearside_pilot import formats
from experiments.ms_swift_aqa.prompt import format_question
from experiments.trr_diagnostics import d2

BANDS = {'AQ-first4': dict(last_n=4, first_layer=0), 'AQ-mid4': dict(last_n=4, first_layer=16),
         'AQ-last4': dict(last_n=4), 'AQ-all': dict(last_n=36)}
SEEDS = {'': 42, '-s43': 43, '-s44': 44}
FAMILIES = {arm: [arm + suffix for suffix in SEEDS] for arm in BANDS}
PAIRS = (('AQ-last4', 'AQ-all'), ('AQ-first4', 'AQ-all'), ('AQ-mid4', 'AQ-all'), ('AQ-last4', 'AQ-first4'), ('AQ-last4', 'AQ-mid4'))
MAX_SHIFTS = 4                                     # the original order and up to three cyclic shifts
TASKS = ('mcq', 'text')
TEXT_GENERATION = dict(do_sample=False, num_beams=1, batch_size=4, max_new_tokens=48)
MEASURES = ('original_order', 'shifted_orders', 'order_gap', 'text')
SHALLOW = ('AQ-first4', 'AQ-mid4', 'AQ-last4')


def arm_configs(base_config, output):
    base, output = d2.load(base_config), Path(output)
    if base.get('task') != 'aqa' or base.get('objective', 'sft') != 'sft':
        raise ValueError('Expected the supervised audio question answering recipe')
    output.mkdir(parents=True, exist_ok=False)
    arms = {}
    for arm, where in BANDS.items():
        for suffix, seed in SEEDS.items():
            cfg = copy.deepcopy(base)
            cfg.pop('methods', None)
            cfg['adapter'] = dict(pilot.ADAPTER, update_positions='all', **where)      # as in the emotion runs; no transport term
            cfg['training'] = dict(cfg['training'], seed=seed, data_seed=seed)        # the recipe of the existing AVQA runs otherwise
            path = output / f'{arm}{suffix}.json'
            d2.write(path, cfg)
            arms[arm + suffix] = dict(config=str(path), seed=seed, **where)
    d2.write(output / 'arms.json', arms)
    return arms


def normal(text):
    return ' '.join(re.sub(r'[^\w\s]', ' ', text.casefold()).split())


def build(test, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    mcq, text = [], []
    for r in d2.read_jsonl(test):
        choices = list(r['choices'])
        gold = choices[ascii_uppercase.index(r['messages'][1]['content'])]
        keep = {k: r[k] for k in ('domain', 'category', 'difficulty') if r.get(k)}
        common = dict(question=r['question'], audios=r['audios'], source=r['id'], gold_text=gold, group='all', dialogue=r['audios'][0], **keep)
        for shift in range(min(len(choices), MAX_SHIFTS)):
            moved = choices[shift:] + choices[:shift]
            letter = ascii_uppercase[moved.index(gold)]
            mcq.append(dict(id=f"{r['id']}#s{shift}", shift=shift, choices=moved, **common,
                            messages=[dict(role='user', content=format_question(r['question'], moved)), dict(role='assistant', content=letter)]))
        prompt = '\n'.join(['<audio>', r['question'], *(f'- {c}' for c in choices), 'Reply with the text of the correct option.'])
        text.append(dict(id=r['id'], choices=choices, **common, messages=[dict(role='user', content=prompt), dict(role='assistant', content=gold)]))
    d2.write_jsonl(output / 'mcq.jsonl', mcq)
    d2.write_jsonl(output / 'text.jsonl', text)
    info = dict(source=str(Path(test).resolve()), source_sha256=d2.digest(test), questions=len(text), audios=len({r['dialogue'] for r in text}),
                options=dict(sorted({str(k): sum(len(r['choices']) == k for r in text) for k in {len(r['choices']) for r in text}}.items())),
                manifests={t: dict(sha256=d2.digest(output / f'{t}.jsonl'), rows=n) for t, n in (('mcq', len(mcq)), ('text', len(text)))})
    d2.write(output / 'aqa.json', info)
    return info


def run(folder, model_config, method, checkpoint, arm):
    from experiments.ms_swift_aqa.evaluate import score as score_letters
    from experiments.ms_swift_librispeech.evaluate import score as score_text

    folder = Path(folder)
    info = d2.load(folder / 'aqa.json')
    out = folder / 'runs' / arm
    pending = [t for t in TASKS if not (out / t / 'metrics.json').exists()]
    if not pending:
        return
    cfg = d2.load(model_config)
    if method == 'frozen-base':
        from experiments.ms_swift_aqa.evaluate_frozen import load_frozen
        model, template = load_frozen(cfg)
    else:
        from experiments.ms_swift_librispeech.train import load_best
        model, template = load_best(cfg, method, Path(checkpoint))
        model.requires_grad_(False)
    for task in pending:
        manifest = folder / f'{task}.jsonl'
        if d2.digest(manifest) != info['manifests'][task]['sha256']:
            raise ValueError(f'{task}: manifest changed after it was built')
        generation = cfg['generation'] if task == 'mcq' else TEXT_GENERATION
        (score_letters if task == 'mcq' else score_text)(model, template, manifest, generation, out / task / 'predictions.jsonl')
        d2.write(out / task / 'metrics.json', dict(task=task, arm=arm, checkpoint=str(checkpoint) if checkpoint else None, generation=generation,
                                                   manifest_sha256=info['manifests'][task]['sha256'],
                                                   predictions_sha256=d2.digest(out / task / 'predictions.jsonl')))
        print(f'{task}: done', flush=True)


def by_letter(row, answer):
    answer = answer.strip()
    if len(answer) != 1 or answer not in ascii_uppercase[:len(row['choices'])]:
        return False, False
    return row['choices'][ascii_uppercase.index(answer)] == row['gold_text'], True


def by_text(row, answer):
    """The option the answer names: equal to it, or the only option it contains."""
    said, options = normal(answer), {normal(c) for c in row['choices']}
    named = {said} & options or {o for o in options if o and re.search(rf'(?<!\w){re.escape(o)}(?!\w)', said)}
    return (len(named) == 1 and named == {normal(row['gold_text'])}), len(named) == 1


def summarize(folder, *, resamples=10000, seed=42):
    folder = Path(folder)
    info = d2.load(folder / 'aqa.json')
    arms = sorted(p.name for p in (folder / 'runs').iterdir() if all((p / t / 'metrics.json').exists() for t in TASKS))
    predictions = {}
    for arm in arms:
        for task in TASKS:
            path = folder / 'runs' / arm / task
            if d2.digest(path / 'predictions.jsonl') != d2.load(path / 'metrics.json')['predictions_sha256']:
                raise ValueError(f'{arm}/{task}: predictions changed after they were written')
            predictions[arm, task] = d2.read_jsonl(path / 'predictions.jsonl')
    mcq, text = d2.read_jsonl(folder / 'mcq.jsonl'), d2.read_jsonl(folder / 'text.jsonl')
    value, boot, extra = ({a: {} for a in arms} for _ in range(3))
    for name, keep in (('original_order', [r['shift'] == 0 for r in mcq]), ('shifted_orders', [r['shift'] > 0 for r in mcq])):
        scores = formats.Scores([r for r, k in zip(mcq, keep) if k], name, resamples, np.random.default_rng(seed), judge=by_letter)
        for arm in arms:
            result, draws = scores.measure([r for r, k in zip(predictions[arm, 'mcq'], keep) if k])
            value[arm][name], boot[arm][name] = result['score'], draws
    scores = formats.Scores(text, 'text', resamples, np.random.default_rng(seed), judge=by_text)
    for arm in arms:
        result, draws = scores.measure(predictions[arm, 'text'])
        value[arm]['text'], boot[arm]['text'], extra[arm]['text_parseable'] = result['score'], draws, result['parseable']
        value[arm]['order_gap'] = value[arm]['original_order'] - value[arm]['shifted_orders']
        boot[arm]['order_gap'] = boot[arm]['original_order'] - boot[arm]['shifted_orders']
        chosen = {}
        letters = np.zeros(MAX_SHIFTS)
        for row, given in zip(mcq, predictions[arm, 'mcq']):
            answer = given['prediction'].strip()
            chosen.setdefault(row['source'], []).append(row['choices'][ascii_uppercase.index(answer)] if by_letter(row, answer)[1] else None)
            if len(row['choices']) == 4 and by_letter(row, answer)[1]:
                letters[ascii_uppercase.index(answer)] += 1
        extra[arm]['same_option_under_all_orders'] = float(np.mean([len(set(v)) == 1 for v in chosen.values() if len(v) > 1]))
        extra[arm]['letter_shares_four_options'] = (letters / max(letters.sum(), 1)).tolist()
    rng = np.random.default_rng(seed + 1)
    picks = {}

    def mean_boot(members, name):
        if len(members) not in picks:
            picks[len(members)] = rng.integers(0, len(members), size=(resamples, len(members)))
        stacked = np.stack([boot[m][name] for m in members])
        return stacked[picks[len(members)].T, np.arange(resamples)].mean(axis=0)

    groups = {a: [a] for a in ('frozen-base',) if a in arms}
    groups.update({f: [m for m in members if m in arms] for f, members in FAMILIES.items() if any(m in arms for m in members)})
    table = {g: dict({n: dict(per_model={m: value[m][n] for m in members}, mean=float(np.mean([value[m][n] for m in members]))) for n in MEASURES},
                     **{k: (np.mean([extra[m][k] for m in members], axis=0).tolist() if k == 'letter_shares_four_options'
                            else float(np.mean([extra[m][k] for m in members]))) for k in extra[members[0]]})
             for g, members in groups.items()}
    contrasts = {}
    if 'frozen-base' in groups:
        for g, members in groups.items():
            if g != 'frozen-base':
                contrasts[f'{g} - frozen-base'] = {n: dict(delta_pp=100 * (table[g][n]['mean'] - table['frozen-base'][n]['mean']),
                                                           interval_95_pp=formats.interval(mean_boot(members, n) - boot['frozen-base'][n])) for n in MEASURES}
    for a, b in PAIRS:
        common = [i for i in range(3) if FAMILIES[a][i] in arms and FAMILIES[b][i] in arms]
        if common:
            first, second = [FAMILIES[a][i] for i in common], [FAMILIES[b][i] for i in common]
            contrasts[f'{a} - {b}'] = {n: dict(delta_pp=100 * float(np.mean([value[x][n] - value[y][n] for x, y in zip(first, second)])),
                                               interval_95_pp=formats.interval(mean_boot(first, n) - mean_boot(second, n))) for n in MEASURES}
    return dict(folder=str(folder), questions=info['questions'], audios=info['audios'], arms=arms, table=table, contrasts=contrasts,
                readings=judge(contrasts))


def judge(contrasts):
    """Readings declared in research/second-task-and-grpo-20261125.md, part 1a."""
    def low(arm, measure):
        return contrasts[f'{arm} - frozen-base'][measure]['interval_95_pp'][0]

    bands = {}
    for arm in (*SHALLOW, 'AQ-all'):
        if f'{arm} - frozen-base' not in contrasts:
            continue
        if low(arm, 'original_order') <= 0:
            bands[arm] = 'no gain'
        elif low(arm, 'shifted_orders') > 0 and low(arm, 'text') > 0:
            bands[arm] = 'transfers'
        elif low(arm, 'shifted_orders') <= 0 and low(arm, 'text') <= 0:
            bands[arm] = 'prompt-bound'
        else:
            bands[arm] = 'partial'
    out = dict(bands=bands, claim_on_second_task=None)
    if 'AQ-last4' in bands and ('AQ-first4' in bands or 'AQ-all' in bands):
        if bands['AQ-last4'] == 'transfers':
            out['claim_on_second_task'] = 'does not extend'
        elif bands['AQ-last4'] in ('prompt-bound', 'no gain') and 'transfers' in (bands.get('AQ-first4'), bands.get('AQ-all')):
            out['claim_on_second_task'] = 'holds'
        else:
            out['claim_on_second_task'] = 'unclear'
    return out


def render(result):
    names = dict(original_order='original order', shifted_orders='shifted orders', order_gap='order gap', text='option text, no letters')
    lines = ['# Adapters trained on AVQA, scored on MMAU Test-mini', '',
             f"{result['questions']} questions on {result['audios']} audio clips. Accuracy in %; the order gap is in pp. "
             'In brackets: share of free-text answers naming exactly one option.', '',
             '| model (seeds) | ' + ' | '.join(names[n] for n in MEASURES) + ' | same option under all orders | letters chosen A B C D % |',
             '|---|' + '---:|' * len(MEASURES) + '---:|---|']
    for g, row in result['table'].items():
        cells = [f"{100 * row[n]['mean']:+.1f}" if n == 'order_gap' else f"{100 * row[n]['mean']:.1f}" for n in MEASURES]
        cells[-1] += f" [{100 * row['text_parseable']:.0f}%]"
        lines.append(f"| {g} ({len(row['text']['per_model'])}) | " + ' | '.join(cells) + f" | {100 * row['same_option_under_all_orders']:.1f} | "
                     + ' '.join(f'{100 * s:.0f}' for s in row['letter_shares_four_options']) + ' |')
    lines += ['', '| difference pp [95% CI over audio clips and seeds] | ' + ' | '.join(names[n] for n in MEASURES) + ' |', '|---|' + '---|' * len(MEASURES)]
    for key, by in result['contrasts'].items():
        lines.append(f'| {key} | ' + ' | '.join(
            f"{by[n]['delta_pp']:+.1f} [{by[n]['interval_95_pp'][0]:+.1f}, {by[n]['interval_95_pp'][1]:+.1f}]" for n in MEASURES) + ' |')
    lines += ['', 'Readings declared before training:', '']
    lines += [f'- {arm}: **{reading}**' for arm, reading in result['readings']['bands'].items()]
    lines.append(f"- the claim on this second task: **{result['readings']['claim_on_second_task'] or 'not available'}**")
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('configs')
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p = commands.add_parser('build')
    p.add_argument('--test', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p = commands.add_parser('run')
    p.add_argument('--folder', type=Path, required=True)
    p.add_argument('--model-config', type=Path, required=True)
    p.add_argument('--method', choices=['frozen-base', 'ta-lora'], required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--arm', required=True)
    p = commands.add_parser('summarize')
    p.add_argument('--folder', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'configs':
        print(json.dumps(sorted(arm_configs(args.base, args.output))))
    elif args.command == 'build':
        print(json.dumps(build(args.test, args.output)))
    elif args.command == 'run':
        run(args.folder, args.model_config, args.method, args.checkpoint, args.arm)
    else:
        summary = summarize(args.folder)
        args.output.mkdir(parents=True, exist_ok=True)
        d2.write(args.output / 'aqa.json', summary)
        (args.output / 'aqa.md').write_text(render(summary))
        print(render(summary))
