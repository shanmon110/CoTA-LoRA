"""Do the adapters trained on IEMOCAP hold up on a corpus they never saw?

CREMA-D: 91 actors each speak the same 12 sentences in several emotions, so the words say nothing about
the label. Nothing is trained here. Every adapter of the earlier runs, and the frozen base, answers

mcq     the trained question under the four option orders
yesno, word, open   the three formats of formats.py

Subcommands: `build` selects the clips and writes the manifests, `run` scores one model, `summarize`
compares the models. The selection, the scores and the readings are fixed in this file and in
research/cross-corpus-cremad-20261124.md before any model was scored on CREMA-D.
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np

from experiments.hearside_pilot import formats
from experiments.trr_diagnostics import d2

CLASSES = ('A', 'B', 'C', 'D')
CODES = dict(ANG='A', HAP='B', SAD='C', NEU='D')            # disgust and fear have no counterpart among the trained classes
FILE = re.compile(r'^(\d{4})_([A-Z]{3})_([A-Z]{3})_([A-Z]{2})\.wav$')
PER_ACTOR_AND_CLASS = 3
ROTATIONS = (0, 1, 2, 3)
TASKS = ('mcq',) + formats.FORMATS
FAMILIES = formats.FAMILIES
SINGLE = formats.SINGLE
REFERENCE = formats.REFERENCE
# Written down before any model was scored on CREMA-D.
CRITERIA = dict(in_corpus_gain_pp=17.4, min_transfer_share=0.5, base_ceiling=0.85, min_order_gap_difference_pp=10.0)
MEASURES = ('mcq_training_order', 'mcq_other_orders', 'mcq_all_orders', 'order_gap') + formats.FORMATS


def select(audio, *, seed=42):
    """Three clips per actor and class, drawn with a fixed seed; an actor with fewer in a class gives what there is."""
    found = {}
    for path in sorted(Path(audio).glob('*.wav')):
        match = FILE.match(path.name)
        if match and match[3] in CODES:
            found.setdefault((match[1], CODES[match[3]]), []).append(path)
    rng = np.random.default_rng(seed)
    chosen = []
    for key in sorted(found):
        paths = found[key]
        for index in sorted(rng.permutation(len(paths))[:PER_ACTOR_AND_CLASS]):
            chosen.append((key[0], key[1], paths[index]))
    return chosen


def build(audio, protocol, output, *, seed=42):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    protocol = d2.load(protocol)
    prompts = protocol['prompts']
    clips = select(audio, seed=seed)
    if not clips:
        raise ValueError(f'No CREMA-D clips of the four classes in {audio} (expected names like 1001_DFA_ANG_XX.wav)')
    rows = []
    for actor, label, path in clips:
        sentence = FILE.match(path.name)[2]
        for shift in ROTATIONS:
            order = [CLASSES[(j + shift) % 4] for j in range(4)]
            letter = CLASSES[order.index(label)]
            question = prompts['question_template'].format(*(prompts['options'][c] for c in order))
            rows.append(dict(id=f'{path.stem}#r{shift}', utterance=path.stem, speaker=actor, dialogue=actor, session=0, scripted=True,
                             sentence=sentence, rotation=shift, option_order=order, label=label, group=label, reference=letter,
                             audios=[str(path.resolve())],
                             messages=[dict(role='user', content='<audio>' + question), dict(role='assistant', content=letter)]))
    d2.write_jsonl(output / 'mcq.jsonl', rows)
    info = formats.build(output / 'mcq.jsonl', output / 'formats')
    counts = {c: sum(label == c for _, label, _ in clips) for c in CLASSES}
    info = dict(corpus='CREMA-D', audio=str(Path(audio).resolve()), seed=seed, per_actor_and_class=PER_ACTOR_AND_CLASS,
                utterances=len(clips), actors=len({a for a, _, _ in clips}), clips_per_class=counts, question_template=prompts['question_template'],
                mcq_generation=protocol['generation'],          # as on Session 4, whatever the model config says
                manifests=dict({'mcq': dict(sha256=d2.digest(output / 'mcq.jsonl'), rows=len(rows))},
                               **{n: v for n, v in info['manifests'].items()}))
    d2.write(output / 'crosscorpus.json', info)
    return info


def manifest_path(folder, task):
    return Path(folder) / ('mcq.jsonl' if task == 'mcq' else f'formats/{task}.jsonl')


def run(folder, model_config, method, checkpoint, arm):
    """All four tasks for one model; finished tasks are kept."""
    from experiments.ms_swift_iemocap.evaluate import score as score_choice
    from experiments.ms_swift_librispeech.evaluate import score as score_text

    folder = Path(folder)
    info = d2.load(folder / 'crosscorpus.json')
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
        manifest = manifest_path(folder, task)
        if d2.digest(manifest) != info['manifests'][task]['sha256']:
            raise ValueError(f'{task}: manifest changed after it was built')
        if task == 'mcq':
            score_choice(model, template, manifest, info['mcq_generation'], out / task / 'predictions.jsonl')
        else:
            score_text(model, template, manifest, formats.GENERATION, out / task / 'predictions.jsonl')
        d2.write(out / task / 'metrics.json', dict(task=task, arm=arm, checkpoint=str(checkpoint) if checkpoint else None,
                                                   manifest_sha256=info['manifests'][task]['sha256'],
                                                   predictions_sha256=d2.digest(out / task / 'predictions.jsonl')))
        print(f'{task}: done', flush=True)


def letter(row, text):
    answer = text.strip()
    return answer == row['reference'], answer in CLASSES


def interval(boot):
    low, high = np.nanpercentile(boot, [2.5, 97.5])
    return [100 * float(low), 100 * float(high)]


def summarize(folder, *, resamples=10000, seed=42):
    folder = Path(folder)
    info = d2.load(folder / 'crosscorpus.json')
    arms = sorted(p.name for p in (folder / 'runs').iterdir() if all((p / t / 'metrics.json').exists() for t in TASKS))
    predictions = {}
    for arm in arms:
        for task in TASKS:
            path = folder / 'runs' / arm / task
            if d2.digest(path / 'predictions.jsonl') != d2.load(path / 'metrics.json')['predictions_sha256']:
                raise ValueError(f'{arm}/{task}: predictions changed after they were written')
            predictions[arm, task] = d2.read_jsonl(path / 'predictions.jsonl')
    value, boot, parseable = ({a: {} for a in arms} for _ in range(3))
    mcq = d2.read_jsonl(folder / 'mcq.jsonl')
    parts = dict(mcq_training_order=[r['rotation'] == 0 for r in mcq], mcq_other_orders=[r['rotation'] != 0 for r in mcq],
                 mcq_all_orders=[True] * len(mcq))
    for name, keep in parts.items():
        # The same seed for every part: all actors appear in each, so the three share their resamples.
        scores = formats.Scores([r for r, k in zip(mcq, keep) if k], name, resamples, np.random.default_rng(seed), judge=letter)
        for arm in arms:
            result, draws = scores.measure([r for r, k in zip(predictions[arm, 'mcq'], keep) if k])
            value[arm][name], boot[arm][name] = result['score'], draws
    for arm in arms:
        value[arm]['order_gap'] = value[arm]['mcq_training_order'] - value[arm]['mcq_other_orders']
        boot[arm]['order_gap'] = boot[arm]['mcq_training_order'] - boot[arm]['mcq_other_orders']
    for name in formats.FORMATS:
        scores = formats.Scores(d2.read_jsonl(manifest_path(folder, name)), name, resamples, np.random.default_rng(seed))
        for arm in arms:
            result, draws = scores.measure(predictions[arm, name])
            value[arm][name], boot[arm][name], parseable[arm][name] = result['score'], draws, result['parseable']
    rng = np.random.default_rng(seed + 1)
    picks = {}

    def mean_boot(members, name):
        if len(members) not in picks:
            picks[len(members)] = rng.integers(0, len(members), size=(resamples, len(members)))
        stacked = np.stack([boot[m][name] for m in members])
        return stacked[picks[len(members)].T, np.arange(resamples)].mean(axis=0)

    groups = {a: [a] for a in SINGLE if a in arms}
    groups.update({f: [m for m in members if m in arms] for f, members in FAMILIES.items() if any(m in arms for m in members)})
    table = {g: {n: dict(per_model={m: value[m][n] for m in members}, mean=float(np.mean([value[m][n] for m in members])))
                 for n in MEASURES} for g, members in groups.items()}
    for g, members in groups.items():
        for n in formats.FORMATS:
            table[g][n]['parseable'] = float(np.mean([parseable[m][n] for m in members]))
    contrasts = {}
    if 'frozen-base' in groups:
        for g, members in groups.items():
            if g != 'frozen-base':
                contrasts[f'{g} - frozen-base'] = {
                    n: dict(delta_pp=100 * (table[g][n]['mean'] - table['frozen-base'][n]['mean']),
                            interval_95_pp=interval(mean_boot(members, n) - boot['frozen-base'][n])) for n in MEASURES}
    for a, b in (('A-last4', 'A-all'), ('D-all-shuffled', 'A-all'), ('B-audio', 'A-all')):
        common = [i for i in range(3) if FAMILIES[a][i] in arms and FAMILIES[b][i] in arms]
        if common:
            first, second = [FAMILIES[a][i] for i in common], [FAMILIES[b][i] for i in common]
            contrasts[f'{a} - {b}'] = {
                n: dict(delta_pp=100 * float(np.mean([value[x][n] - value[y][n] for x, y in zip(first, second)])),
                        interval_95_pp=interval(mean_boot(first, n) - mean_boot(second, n))) for n in MEASURES}
    return dict(folder=str(folder), corpus=info['corpus'], utterances=info['utterances'], actors=info['actors'], arms=arms,
                criteria=CRITERIA, table=table, contrasts=contrasts, readings=judge(table, contrasts))


def judge(table, contrasts):
    """The three readings declared in research/cross-corpus-cremad-20261124.md."""
    out = dict(full_depth_gain_on_unseen_corpus=None, last_layers_bind_to_option_order_on_unseen_corpus=None,
               last_layers_gain_is_format_specific_on_unseen_corpus=None)
    key = f'{REFERENCE} - frozen-base'
    if key in contrasts:
        gain, (low, high) = (contrasts[key]['mcq_all_orders'][k] for k in ('delta_pp', 'interval_95_pp'))
        threshold = CRITERIA['min_transfer_share'] * CRITERIA['in_corpus_gain_pp']
        if table['frozen-base']['mcq_all_orders']['mean'] >= CRITERIA['base_ceiling']:
            reading = 'base at ceiling'
        elif high < 0:
            reading = 'harmed'
        elif gain >= threshold and low > 0:
            reading = 'transfers'
        else:
            reading = 'gap'
        out['full_depth_gain_on_unseen_corpus'] = reading
    if 'A-last4 - A-all' in contrasts:
        gap = contrasts['A-last4 - A-all']['order_gap']
        out['last_layers_bind_to_option_order_on_unseen_corpus'] = (
            'replicated' if gap['delta_pp'] >= CRITERIA['min_order_gap_difference_pp'] and gap['interval_95_pp'][0] > 0 else 'not replicated')
    if 'A-last4 - frozen-base' in contrasts and 'A-all - frozen-base' in contrasts:
        last, full = contrasts['A-last4 - frozen-base'], contrasts['A-all - frozen-base']
        usable = [n for n in formats.FORMATS if full[n]['interval_95_pp'][0] > 0]     # formats where full depth gains at all
        if not usable:
            reading = 'not testable'
        elif all(last[n]['interval_95_pp'][1] < CRITERIA['min_transfer_share'] * full[n]['delta_pp'] for n in usable):
            reading = 'replicated'
        else:
            reading = 'not replicated'
        out['last_layers_gain_is_format_specific_on_unseen_corpus'] = reading
        out['formats_where_full_depth_gains'] = usable
    return out


def render(result):
    names = dict(mcq_training_order='training order', mcq_other_orders='other orders', mcq_all_orders='all orders', order_gap='order gap')
    lines = [f"# Adapters trained on IEMOCAP, scored on {result['corpus']}", '',
             f"{result['utterances']} clips of {result['actors']} actors; nothing was trained on this corpus. Mean per-class recall in %; "
             'the order gap is in pp. In brackets: share of parseable free-text answers.', '',
             '| model (mean over its seeds) | ' + ' | '.join(names.get(n, n) for n in MEASURES) + ' |', '|---|' + '---:|' * len(MEASURES)]
    for g, row in result['table'].items():
        cells = []
        for n in MEASURES:
            cell = f"{100 * row[n]['mean']:+.1f}" if n == 'order_gap' else f"{100 * row[n]['mean']:.1f}"
            cells.append(cell + (f" [{100 * row[n]['parseable']:.0f}%]" if 'parseable' in row[n] else ''))
        lines.append(f"| {g} ({len(row['mcq_all_orders']['per_model'])}) | " + ' | '.join(cells) + ' |')
    lines += ['', '| difference pp [95% CI over actors and seeds] | ' + ' | '.join(names.get(n, n) for n in MEASURES) + ' |',
              '|---|' + '---|' * len(MEASURES)]
    for key, by in result['contrasts'].items():
        lines.append(f'| {key} | ' + ' | '.join(
            f"{by[n]['delta_pp']:+.1f} [{by[n]['interval_95_pp'][0]:+.1f}, {by[n]['interval_95_pp'][1]:+.1f}]" for n in MEASURES) + ' |')
    lines += ['', 'Readings declared before scoring:', '']
    lines += [f"- {k}: **{'not available' if v is None else v}**" for k, v in result['readings'].items()]
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('build')
    p.add_argument('--audio', type=Path, required=True, help='the AudioWAV folder of CREMA-D')
    p.add_argument('--protocol', type=Path, required=True, help='pilot-iemocap-session4.json, for the trained question')
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
    if args.command == 'build':
        print(json.dumps({k: v for k, v in build(args.audio, args.protocol, args.output).items() if k != 'question_template'}, indent=2))
    elif args.command == 'run':
        run(args.folder, args.model_config, args.method, args.checkpoint, args.arm)
    else:
        summary = summarize(args.folder)
        args.output.mkdir(parents=True, exist_ok=True)
        d2.write(args.output / 'crosscorpus.json', summary)
        (args.output / 'crosscorpus.md').write_text(render(summary))
        print(render(summary))
