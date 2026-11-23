"""Headroom test for input-conditional adaptation (declared in research/headroom-test-input-conditional-20261124.md).

If utterances are split into groups by a property of the audio and every group gets its own adapter, is that
better than one shared adapter?  Three groupings, each on its own, plus random splits of the same sizes that
remove the effect of training each adapter on less data.

Subcommands: `build` writes the per-specialist manifests and configs, `score` evaluates one specialist on its
Session 4 utterances, `summarize` assembles the specialists of each grouping and compares them with the shared adapter.
"""
import argparse
import json
import math
import wave
from pathlib import Path
from random import Random

import numpy as np

from experiments.hearside_pilot import followup
from experiments.trr_diagnostics import d2

SHARED = 'D-all-shuffled'                         # the shared adapter; its recipe is the specialists' recipe
GROUPINGS = {'gender': ('F', 'M'), 'duration': ('short', 'medium', 'long'), 'scripted': ('impro', 'script')}
RANDOM_OF = {'gender': 'random2', 'scripted': 'random2', 'duration': 'random3'}
# Written down before any specialist was trained.
CRITERIA = dict(min_gain_pp=2.0, seed_sd_multiple=2.0)
BATCH = 8                                         # utterances per optimizer step in the recipe (2 x 4 accumulation)


def seconds(path):
    with wave.open(str(path)) as audio:
        return audio.getnframes() / audio.getframerate()


def utterance(row):
    return row.get('utterance', row['id'])


def properties(row, cuts, duration=seconds):
    uid = utterance(row)
    length = duration(row['audios'][0])
    return dict(gender=uid.split('_')[-1][0], scripted='script' if '_script' in uid else 'impro',
                duration='short' if length <= cuts[0] else 'medium' if length <= cuts[1] else 'long')


def random_groups(ids, sizes, seed):
    """A random partition of the utterances into parts with the given relative sizes."""
    ids = sorted(ids)
    Random(seed).shuffle(ids)
    total, out, start = sum(sizes), {}, 0
    for part, size in enumerate(sizes):
        end = len(ids) if part == len(sizes) - 1 else start + round(len(ids) * size / total)
        out.update({uid: f'part{part}' for uid in ids[start:end]})
        start = end
    return out


def build(data, test_manifest, base_config, output, *, seed=42, duration=seconds):
    data, output = Path(data), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    splits = {'train': d2.read_jsonl(data / 'train-shuffled.jsonl'), 'dev': d2.read_jsonl(data / 'dev-shuffled.jsonl'),
              'test': d2.read_jsonl(test_manifest)}
    lengths = sorted(duration(r['audios'][0]) for r in splits['train'])
    cuts = [lengths[len(lengths) // 3], lengths[2 * len(lengths) // 3]]            # terciles of the training set
    assign = {s: {} for s in splits}                                               # split -> utterance -> grouping -> group
    for split, rows in splits.items():
        for row in rows:
            assign[split].setdefault(utterance(row), properties(row, cuts, duration))
    sizes = {g: [sum(p[g] == name for p in assign['train'].values()) for name in names] for g, names in GROUPINGS.items()}
    for name, like in (('random2', 'gender'), ('random3', 'duration')):
        for split in splits:
            parts = random_groups(assign[split], sizes[like], seed)
            for uid, part in parts.items():
                assign[split][uid][name] = part
    base = json.loads(Path(base_config).read_text())
    specialists = {}
    groupings = {**GROUPINGS, 'random2': ('part0', 'part1'), 'random3': ('part0', 'part1', 'part2')}
    for grouping, names in groupings.items():
        for name in names:
            key = f'{grouping}-{name}'
            folder = output / key
            folder.mkdir()
            counts = {}
            for split, rows in splits.items():
                part = [r for r in rows if assign[split][utterance(r)][grouping] == name]
                d2.write_jsonl(folder / f'{split}.jsonl', part)
                counts[split] = len(part)
            if not all(counts.values()):
                raise ValueError(f'{key}: a split has no utterances: {counts}')
            steps = math.ceil(counts['train'] / BATCH)
            cfg = json.loads(json.dumps(base))
            cfg['training'].update(eval_steps=math.ceil(steps / 3), save_steps=math.ceil(steps / 3))   # three checks, as the shared arm
            d2.write(folder / 'config.json', cfg)
            specialists[key] = dict(grouping=grouping, group=name, rows=counts, steps=steps,
                                    test_sha256=d2.digest(folder / 'test.jsonl'))
    info = dict(seed=seed, duration_cuts_seconds=cuts, test_manifest_sha256=d2.digest(test_manifest),
                base_config_sha256=d2.digest(base_config), train_group_sizes=sizes, specialists=specialists,
                test_groups={uid: p for uid, p in sorted(assign['test'].items())})
    d2.write(output / 'headroom.json', info)
    return info


def score(folder, key, model_config, checkpoint):
    """Session 4 answers of one specialist on the utterances of its group."""
    from experiments.ms_swift_iemocap.evaluate import score as score_choice
    from experiments.ms_swift_librispeech.train import load_best

    folder = Path(folder)
    info = d2.load(folder / 'headroom.json')
    manifest, out = folder / key / 'test.jsonl', folder / key / 'eval'
    if (out / 'metrics.json').exists():
        return
    if d2.digest(manifest) != info['specialists'][key]['test_sha256']:
        raise ValueError(f'{key}: test manifest changed after it was built')
    cfg = d2.load(model_config)
    model, template = load_best(cfg, 'ta-lora', Path(checkpoint))
    model.requires_grad_(False)
    try:
        metrics = score_choice(model, template, manifest, cfg['generation'], out / 'predictions.jsonl')
    except ValueError as error:                       # a group may lack a class; the pooled score does not need this one
        if 'reference classes' not in str(error):
            raise
        (out / 'predictions.tmp').replace(out / 'predictions.jsonl')
        metrics = {}
    d2.write(out / 'metrics.json', dict(metrics, specialist=key, checkpoint=str(checkpoint),
                                        predictions_sha256=d2.digest(out / 'predictions.jsonl')))


def summarize(folder, eval_folders, *, seed_sd_pp=None, resamples=10000, seed=42):
    """`eval_folders`: prepared evaluation folders that hold the shared adapter's Session 4 predictions."""
    folder = Path(folder)
    info = d2.load(folder / 'headroom.json')
    cfg, first, arms = followup.load(eval_folders)
    manifest = d2.read_jsonl(first / 'manifests' / 'orig.jsonl')
    if d2.digest(first / 'manifests' / 'orig.jsonl') != info['test_manifest_sha256']:
        raise ValueError('The specialists were built from another Session 4 manifest')
    scorer = followup.Scorer(manifest, cfg['classes'], resamples, seed)
    position = {m['id']: i for i, m in enumerate(manifest)}

    def shared(arm):
        where = arms[arm]
        rows = d2.completed_predictions(where[0], where[1], 'orig', where[2])
        return scorer.classes_of([r['prediction'] for r in rows])

    members = [m for m in followup.FAMILIES[SHARED] if m in arms]
    reference = shared(SHARED)
    value, boot = scorer.uar(reference, 'all_orders')

    # Step zero: how the shared adapter does within each group.
    groups = {}
    for grouping, names in GROUPINGS.items():
        of = np.array([info['test_groups'][m['utterance']][grouping] for m in manifest])
        groups[grouping] = {}
        for name in names:
            keep = of == name
            per_seed = []
            for arm in members:
                predicted = shared(arm)
                recalls = [float(np.mean(predicted[keep & (scorer.label == c)] == c)) for c in range(len(cfg['classes']))
                           if (keep & (scorer.label == c)).any()]
                per_seed.append(float(np.mean(recalls)))
            groups[grouping][name] = dict(utterances=int(keep.sum() // 4), shared_uar_per_seed=per_seed,
                                          classes_present=int(sum((keep & (scorer.label == c)).any() for c in range(4))))

    def assembled(grouping, names):
        letters = [None] * len(manifest)
        for name in names:
            path = folder / f'{grouping}-{name}' / 'eval'
            if not (path / 'metrics.json').exists():
                return None
            if d2.digest(path / 'predictions.jsonl') != d2.load(path / 'metrics.json')['predictions_sha256']:
                raise ValueError(f'{grouping}-{name}: predictions changed after they were written')
            for row in d2.read_jsonl(path / 'predictions.jsonl'):
                letters[position[row['id']]] = row['prediction']
        if None in letters:
            raise ValueError(f'{grouping}: the specialists do not cover every Session 4 row')
        return scorer.classes_of(letters)

    systems = {}
    for grouping in (*GROUPINGS, 'random2', 'random3'):
        names = sorted({p[grouping] for p in info['test_groups'].values()})
        predicted = assembled(grouping, names)
        if predicted is not None:
            v, b = scorer.uar(predicted, 'all_orders')
            systems[grouping] = dict(value=v, boot=b, predicted=predicted)

    result = {}
    for grouping in GROUPINGS:
        control = RANDOM_OF[grouping]
        if grouping not in systems:
            result[grouping] = None
            continue
        s = systems[grouping]
        entry = dict(uar=100 * s['value'], shared_uar=100 * value, gain_over_shared_pp=100 * (s['value'] - value),
                     gain_over_shared_interval_95_pp=followup.interval(s['boot'] - boot),
                     answers_changed=float(np.mean(s['predicted'] != reference)))
        if control in systems:
            c = systems[control]
            entry.update(random_uar=100 * c['value'], gain_over_random_pp=100 * (s['value'] - c['value']),
                         gain_over_random_interval_95_pp=followup.interval(s['boot'] - c['boot']),
                         random_minus_shared_pp=100 * (c['value'] - value))
        entry['headroom'] = judge(entry, seed_sd_pp)
        result[grouping] = entry
    decided = [e['headroom'] for e in result.values() if e is not None]
    outcome = ('undecided' if len(decided) < len(GROUPINGS) or None in decided else
               'headroom found' if True in decided else 'closed')
    return dict(folder=str(folder), criteria=CRITERIA, seed_sd_pp=seed_sd_pp, shared_seeds=members,
                duration_cuts_seconds=info['duration_cuts_seconds'], groups=groups, groupings=result, outcome=outcome)


def judge(entry, seed_sd_pp):
    """All three declared conditions; None when the random control or the seed spread is missing."""
    if 'gain_over_random_pp' not in entry or seed_sd_pp is None:
        return None
    gain = entry['gain_over_shared_pp']
    return bool(gain >= CRITERIA['min_gain_pp'] and entry['gain_over_shared_interval_95_pp'][0] > 0
                and entry['gain_over_random_interval_95_pp'][0] > 0
                and gain > CRITERIA['seed_sd_multiple'] * seed_sd_pp)


def render(result):
    ci = lambda pair: f'[{pair[0]:+.2f}, {pair[1]:+.2f}]'
    lines = ['# Headroom for input-conditional adaptation', '',
             f"Shared adapter: {SHARED}. Session 4, UAR in % over the four option orders.", '',
             '## Step zero: the shared adapter within each group', '',
             '| grouping | group | utterances | classes present | shared UAR per seed |', '|---|---|---:|---:|---|']
    for grouping, by in result['groups'].items():
        for name, g in by.items():
            lines.append(f"| {grouping} | {name} | {g['utterances']} | {g['classes_present']} | "
                         + ' / '.join(f'{100 * v:.1f}' for v in g['shared_uar_per_seed']) + ' |')
    lines += ['', '## Group specialists against the shared adapter (seed 42)', '',
              '| grouping | specialists | shared | random specialists | specialists − shared pp [95% CI] | specialists − random pp [95% CI] | headroom |',
              '|---|---:|---:|---:|---|---|---|']
    for grouping, e in result['groupings'].items():
        if e is None:
            lines.append(f'| {grouping} | not available | | | | | |')
            continue
        random = f"{e['random_uar']:.1f}" if 'random_uar' in e else 'n/a'
        versus = (f"{e['gain_over_random_pp']:+.2f} {ci(e['gain_over_random_interval_95_pp'])}" if 'random_uar' in e else 'n/a')
        verdict = 'not available' if e['headroom'] is None else 'yes' if e['headroom'] else 'no'
        lines.append(f"| {grouping} | {e['uar']:.1f} | {e['shared_uar']:.1f} | {random} | {e['gain_over_shared_pp']:+.2f} "
                     f"{ci(e['gain_over_shared_interval_95_pp'])} | {versus} | {verdict} |")
    sd = result['seed_sd_pp']
    lines += ['', f"Seed standard deviation of the shared adapter: {'not available' if sd is None else f'{sd:.2f} pp'}; "
              f"a gain must exceed {CRITERIA['seed_sd_multiple']:g} times it and {CRITERIA['min_gain_pp']:g} pp.",
              '', f"Outcome under the declared rule: **{result['outcome']}**"]
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('build')
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--test-manifest', type=Path, required=True)
    p.add_argument('--base-config', type=Path, required=True, help=f'the config of {SHARED}')
    p.add_argument('--output', type=Path, required=True)
    p = commands.add_parser('score')
    p.add_argument('--folder', type=Path, required=True)
    p.add_argument('--specialist', required=True)
    p.add_argument('--model-config', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p = commands.add_parser('summarize')
    p.add_argument('--folder', type=Path, required=True)
    p.add_argument('--eval', type=Path, nargs='+', required=True)
    p.add_argument('--followup', type=Path, help='followup.json with the seed spread of the shared adapter')
    p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'build':
        built = build(args.data, args.test_manifest, args.base_config, args.output)
        print(json.dumps({k: v['rows'] for k, v in built['specialists'].items()}, indent=1))
    elif args.command == 'score':
        score(args.folder, args.specialist, args.model_config, args.checkpoint)
    else:
        sd = None
        if args.followup:
            family = d2.load(args.followup)['families'].get(SHARED)
            sd = 100 * family['sd']['uar_all_orders'] if family else None
        summary = summarize(args.folder, args.eval, seed_sd_pp=sd)
        args.output.mkdir(parents=True, exist_ok=True)
        d2.write(args.output / 'headroom.json', summary)
        (args.output / 'headroom.md').write_text(render(summary))
        print(render(summary))
