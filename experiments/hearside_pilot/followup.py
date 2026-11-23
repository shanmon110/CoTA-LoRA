"""Depth and seeds: the follow-up to the hear-side pilot.

Two questions, fixed before the follow-up arms were trained (README.md, "后续：层数与种子"):

1. Depth. Does an adapter on the last four layers bind to the option order in this pipeline, as the earlier
   pair-01 adapter did in another one?  `A-last4` against `A-all`, one seed each.
2. Seeds. Do A (all positions), B (audio positions) and D (all positions, shuffled options) differ by more than a
   second and third training run would?  Seeds 42 (the pilot), 43 and 44.

Reads the Session 4 predictions of the pilot folder and of the follow-up folder. Both must have been scored on
identical manifests. Intervals: dialogues are resampled for every comparison; where three seeds are averaged the
seeds are resampled too (jointly for the arms being compared, since a seed fixes the initialisation and data order).
Three seeds make that a rough interval; the per-seed values are printed next to it.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from experiments.trr_diagnostics import d2, stats

SEEDS = (42, 43, 44)
FAMILIES = {arm: [arm, f'{arm}-s43', f'{arm}-s44'] for arm in ('A-all', 'B-audio', 'D-all-shuffled')}
OTHERS = ['frozen-base', 'C-post-audio', 'A-last4']
PAIRS = [('B-audio', 'A-all'), ('D-all-shuffled', 'A-all'), ('B-audio', 'D-all-shuffled')]
SCORED = ('orig', 'orig_reworded', 'text', 'question_only', 'asr-orig')
METRICS = ('uar_all_orders', 'uar_training_order', 'uar_other_orders', 'order_gap', 'uar_reworded')
# Written down before any follow-up arm was trained.
CRITERIA = dict(equivalence_pp=3.0, min_depth_gap_pp=10.0, min_seeds=2)


class Scorer:
    """UAR on Session 4 with a shared set of dialogue resamples, so that every comparison is paired."""

    def __init__(self, manifest, classes, resamples, seed):
        self.manifest, self.classes = manifest, classes
        self.label = np.array([classes.index(m['label']) for m in manifest])
        rotation = np.array([m['rotation'] for m in manifest])
        names = sorted({m['dialogue'] for m in manifest})
        self.dialogues = len(names)
        self.dialogue = np.array([names.index(m['dialogue']) for m in manifest])
        self.subsets = {'training_order': rotation == 0, 'other_orders': rotation != 0, 'all_orders': rotation >= 0}
        self.rng = np.random.default_rng(seed)
        draws = self.rng.integers(0, len(names), size=(resamples, len(names)))
        self.weights = np.stack([np.bincount(row, minlength=len(names)) for row in draws]).astype(float)
        self.totals = {name: self.count(np.ones(len(manifest), bool), keep) for name, keep in self.subsets.items()}
        # A resample that lost a whole class has no UAR; such draws are dropped for every model alike.
        self.valid = np.all([(self.weights @ total > 0).all(axis=1) for total in self.totals.values()], axis=0)

    def count(self, hit, keep):
        out = np.zeros((self.dialogues, len(self.classes)))
        np.add.at(out, (self.dialogue[keep], self.label[keep]), hit[keep])
        return out

    def classes_of(self, letters):
        """A letter is scored as the class it stood for under that row's option order."""
        return np.array([self.classes.index(m['option_order'][self.classes.index(x)]) for x, m in zip(letters, self.manifest)])

    def uar(self, predicted, subset):
        """(value on all dialogues, value in every resample)."""
        correct, total = self.count(predicted == self.label, self.subsets[subset]), self.totals[subset]
        with np.errstate(divide='ignore', invalid='ignore'):
            boot = ((self.weights @ correct) / (self.weights @ total)).mean(axis=1)
        return float((correct.sum(0) / total.sum(0)).mean()), np.where(self.valid, boot, np.nan)


def interval(boot):
    low, high = np.nanpercentile(boot, [2.5, 97.5])
    return [100 * float(low), 100 * float(high)]


def load(folders):
    """Arm -> its run folder, over prepared folders that share their manifests."""
    folders = [Path(f) for f in folders]
    checked = [d2.validate_prepared(f) for f in folders]
    cfg, info = checked[0]
    for folder, (_, other) in zip(folders[1:], checked[1:]):
        for name in SCORED:
            if other['manifests'].get(name) != info['manifests'].get(name):
                raise ValueError(f'{folder}: {name} was not scored on the same manifest as {folders[0]}')
    arms = {}
    for folder, (_, own) in zip(folders, checked):
        runs = folder / 'runs'
        for run in sorted(p for p in runs.iterdir() if p.is_dir()) if runs.is_dir() else []:
            if d2.completed_predictions(folder, run, 'orig', own) is None:
                continue
            if run.name in arms:
                raise ValueError(f'{run.name} was scored in more than one folder')
            arms[run.name] = (folder, run, own)
    return cfg, folders[0], arms


def measure(scorer, folder, run, info):
    """Point values and resampled values of one model."""
    def letters(condition):
        rows = d2.completed_predictions(folder, run, condition, info)
        return None if rows is None else [r['prediction'] for r in rows]

    orig = letters('orig')
    predicted = scorer.classes_of(orig)
    value, boot = {}, {}
    for subset in scorer.subsets:
        value[f'uar_{subset}'], boot[f'uar_{subset}'] = scorer.uar(predicted, subset)
    value['order_gap'] = value['uar_training_order'] - value['uar_other_orders']
    boot['order_gap'] = boot['uar_training_order'] - boot['uar_other_orders']
    for key, condition in (('uar_reworded', 'orig_reworded'), ('uar_text', 'text'), ('uar_question_only', 'question_only')):
        other = letters(condition)
        if other is not None:
            value[key], boot[key] = scorer.uar(scorer.classes_of(other), 'all_orders')
    rows = d2.completed_predictions(folder, run, 'asr-orig', info)
    if rows is not None:
        errors = [stats.transcript_errors(r['reference'], r['prediction'])[:2] for r in rows]
        value['wer'] = sum(e for e, _ in errors) / sum(n for _, n in errors)
    value['letter_share'] = {c: orig.count(c) / len(orig) for c in scorer.classes}
    return value, boot


def summarize(folders, *, resamples=10000, seed=42):
    cfg, first, arms = load(folders)
    scorer = Scorer(d2.read_jsonl(first / 'manifests' / 'orig.jsonl'), cfg['classes'], resamples, seed)
    value, boot = {}, {}
    for arm, where in arms.items():
        value[arm], boot[arm] = measure(scorer, *where)

    models = {}
    for arm in [*OTHERS, *(m for members in FAMILIES.values() for m in members)]:
        if arm in value:
            models[arm] = {k: (v if k == 'letter_share' else float(v)) for k, v in value[arm].items()}
            models[arm]['order_gap_interval_95_pp'] = interval(boot[arm]['order_gap'])
            models[arm]['uar_all_orders_interval_95_pp'] = interval(boot[arm]['uar_all_orders'])

    # Seeds that every family has, so that the family means are over the same runs.
    seeds = [s for i, s in enumerate(SEEDS) if all(members[i] in value for members in FAMILIES.values())]
    index = [SEEDS.index(s) for s in seeds]
    families, contrasts = {}, {}
    if len(seeds) >= CRITERIA['min_seeds']:
        draws = scorer.rng.integers(0, len(seeds), size=(resamples, len(seeds)))      # seeds, shared by all families

        def family_boot(name, metric):
            stacked = np.stack([boot[FAMILIES[name][i]][metric] for i in index])       # [seeds, resamples]
            return stacked[draws.T, np.arange(resamples)].mean(axis=0)

        for name, members in FAMILIES.items():
            per_seed = {m: [value[members[i]][m] for i in index] for m in (*METRICS, 'wer')
                        if all(m in value[members[i]] for i in index)}
            families[name] = dict(seeds=seeds, per_seed=per_seed, mean={m: float(np.mean(v)) for m, v in per_seed.items()},
                                  sd={m: float(np.std(v, ddof=1)) for m, v in per_seed.items()})
        for a, b in PAIRS:
            out = {}
            for metric in METRICS:
                per_seed = [value[FAMILIES[a][i]][metric] - value[FAMILIES[b][i]][metric] for i in index]
                out[metric] = dict(per_seed_pp=[100 * v for v in per_seed], mean_pp=100 * float(np.mean(per_seed)),
                                   interval_95_pp=interval(family_boot(a, metric) - family_boot(b, metric)))
            contrasts[f'{a} - {b}'] = out

    depth = None
    if 'A-last4' in value and 'A-all' in value:
        depth = {}
        for metric in METRICS:
            depth[metric] = dict(last4=100 * value['A-last4'][metric], all36=100 * value['A-all'][metric],
                                 delta_pp=100 * (value['A-last4'][metric] - value['A-all'][metric]),
                                 interval_95_pp=interval(boot['A-last4'][metric] - boot['A-all'][metric]))
        if 'frozen-base' in value:
            depth['last4_minus_base_other_orders'] = dict(
                delta_pp=100 * (value['A-last4']['uar_other_orders'] - value['frozen-base']['uar_other_orders']),
                interval_95_pp=interval(boot['A-last4']['uar_other_orders'] - boot['frozen-base']['uar_other_orders']))

    return dict(folders=[str(f) for f in folders], resamples=resamples, valid_resamples=int(scorer.valid.sum()),
                utterances=len({m['utterance'] for m in scorer.manifest}), scored_rows=len(scorer.manifest),
                criteria=CRITERIA, seeds=seeds, models=models, families=families, contrasts=contrasts, depth=depth,
                claims=judge(contrasts, depth))


def judge(contrasts, depth):
    """The readings fixed before the follow-up ran. `None` means the runs needed for a claim are missing."""
    margin = CRITERIA['equivalence_pp']

    def sign(pair, metric, flip=False):
        if pair not in contrasts:
            return None
        low, high = contrasts[pair][metric]['interval_95_pp']
        low, high = (-high, -low) if flip else (low, high)
        return 'supported' if low > 0 else 'contradicted' if high < 0 else 'inconclusive'

    claims = {}
    if 'B-audio - A-all' in contrasts:
        low, high = contrasts['B-audio - A-all']['uar_all_orders']['interval_95_pp']
        claims['audio_positions_match_all_positions'] = ('supported' if -margin < low and high < margin else
                                                        'contradicted' if high < -margin or low > margin else 'inconclusive')
    else:
        claims['audio_positions_match_all_positions'] = None
    claims['audio_positions_less_bound_to_option_order'] = sign('B-audio - A-all', 'order_gap', flip=True)
    claims['shuffled_options_improve_all_positions'] = sign('D-all-shuffled - A-all', 'uar_all_orders')
    if depth is None:
        claims['last_layers_bind_to_option_order'] = None
    else:
        gap = depth['order_gap']
        claims['last_layers_bind_to_option_order'] = ('supported' if gap['delta_pp'] >= CRITERIA['min_depth_gap_pp']
                                                      and gap['interval_95_pp'][0] > 0 else 'not supported')
    return claims


def render(result):
    def pct(v, signed=False):
        return 'n/a' if v is None else f'{100 * v:+.1f}' if signed else f'{100 * v:.1f}'

    def ci(pair):
        return f'[{pair[0]:+.2f}, {pair[1]:+.2f}]'

    lines = ['# Hear-side follow-up: depth and seeds', '',
             f"Session 4, {result['utterances']} utterances, {result['scored_rows']} rows over four option orders. UAR in %.",
             'The order gap is UAR in the training order minus UAR in the other three orders.', '',
             '| model | training order | other orders | all orders | order gap pp [95% CI] | reworded | transcript only | WER % |',
             '|---|---:|---:|---:|---|---:|---:|---:|']
    for arm, m in result['models'].items():
        lines.append(f"| {arm} | {pct(m['uar_training_order'])} | {pct(m['uar_other_orders'])} | {pct(m['uar_all_orders'])} | "
                     f"{pct(m['order_gap'], True)} {ci(m['order_gap_interval_95_pp'])} | {pct(m.get('uar_reworded'))} | "
                     f"{pct(m.get('uar_text'))} | {pct(m.get('wer'))} |")
    if result['families']:
        seeds = result['seeds']
        lines += ['', f"## Seeds {', '.join(map(str, seeds))}", '',
                  '| arm | measure | ' + ' | '.join(f'seed {s}' for s in seeds) + ' | mean | SD |',
                  '|---|---|' + '---:|' * (len(seeds) + 2)]
        for name, family in result['families'].items():
            for metric in ('uar_all_orders', 'uar_other_orders', 'order_gap', 'uar_reworded', 'wer'):
                if metric in family['per_seed']:
                    lines.append(f"| {name} | {metric} | " + ' | '.join(pct(v, metric == 'order_gap') for v in family['per_seed'][metric])
                                 + f" | {pct(family['mean'][metric], metric == 'order_gap')} | {100 * family['sd'][metric]:.2f} |")
        lines += ['', '| difference | measure | ' + ' | '.join(f'seed {s}' for s in seeds) + ' | mean pp | 95% CI (seeds and dialogues) |',
                  '|---|---|' + '---:|' * (len(seeds) + 1) + '---|']
        for pair, by in result['contrasts'].items():
            for metric, c in by.items():
                lines.append(f'| {pair} | {metric} | ' + ' | '.join(f'{v:+.2f}' for v in c['per_seed_pp'])
                             + f" | {c['mean_pp']:+.2f} | {ci(c['interval_95_pp'])} |")
    if result['depth']:
        lines += ['', '## Last four layers against all 36 (one seed each)', '',
                  '| measure | A-last4 | A-all | difference pp | 95% CI (dialogues) |', '|---|---:|---:|---:|---|']
        for metric in METRICS:
            d = result['depth'][metric]
            lines.append(f"| {metric} | {d['last4']:.1f} | {d['all36']:.1f} | {d['delta_pp']:+.2f} | {ci(d['interval_95_pp'])} |")
        if 'last4_minus_base_other_orders' in result['depth']:
            d = result['depth']['last4_minus_base_other_orders']
            lines.append(f"| A-last4 minus frozen base, other orders | | | {d['delta_pp']:+.2f} | {ci(d['interval_95_pp'])} |")
    lines += ['', 'Claims declared before the follow-up:', '']
    lines += [f"- {name}: **{'not available' if verdict is None else verdict}**" for name, verdict in result['claims'].items()]
    return '\n'.join(lines) + '\n'


def same_inputs(pilot_run, run):
    """Differences between the pilot's and the follow-up's training manifests and Session 4 inputs (none expected)."""
    pilot_run, run = Path(pilot_run), Path(run)
    old, new = (json.loads((r / 'data' / 'splits.json').read_text()) for r in (pilot_run, run))
    differences = [f'data/{name}.jsonl' for name in ('train-fixed', 'train-shuffled', 'dev-fixed', 'dev-shuffled')
                   if old[name]['sha256'] != new[name]['sha256']]
    old, new = (json.loads((r / 'eval' / 'prepare.json').read_text())['manifests'] for r in (pilot_run, run))
    return differences + [f'eval/manifests/{name}.jsonl' for name in SCORED if old.get(name) != new.get(name)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    check = commands.add_parser('check', help='the follow-up trains and scores on the same files as the pilot')
    check.add_argument('--pilot-run', type=Path, required=True)
    check.add_argument('--run', type=Path, required=True)
    table = commands.add_parser('summarize')
    table.add_argument('--pilot', type=Path, required=True, help='eval folder of the pilot run (frozen base and seed-42 arms)')
    table.add_argument('--prepared', type=Path, required=True, help='eval folder of the follow-up run')
    table.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'check':
        different = same_inputs(args.pilot_run, args.run)
        print('identical to the pilot: training manifests and Session 4 inputs' if not different else
              'DIFFERENT from the pilot: ' + ', '.join(different))
        raise SystemExit(1 if different else 0)
    summary = summarize([args.pilot, args.prepared])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'followup.json').write_text(json.dumps(summary, indent=2) + '\n')
    text = render(summary)
    (args.output / 'followup.md').write_text(text)
    print(text)
