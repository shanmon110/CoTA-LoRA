"""Is the option-order shortcut a property of the last layers, or of training only a few layers?

Two new arms train four layers at the start (0-3) and in the middle (16-19) of the 36, three seeds each;
everything else is as in A-last4 (layers 32-35) and A-all (all 36). All are scored on IEMOCAP Session 4
and on CREMA-D with the letter question under four option orders and the three other formats.

Subcommands
    assemble   gather the existing Session 4 predictions of the earlier runs into one folder of the
               layout crosscorpus.py scores, so that old and new arms pass through the same code
    same       compare two arms' predictions in a folder (a re-scored arm against its earlier answers)
    summarize  crosscorpus.summarize for both corpora plus the readings declared in
               research/layer-bands-20261125.md
"""
import argparse
import json
import shutil
from pathlib import Path

from experiments.hearside_pilot import crosscorpus, formats
from experiments.trr_diagnostics import d2

SHALLOW = ('A-first4', 'A-mid4', 'A-last4')
# Written down before any band arm was trained.
CRITERIA = dict(min_order_gap_excess_pp=5.0, min_formats_with_gain=2)


def assemble(evals, formats_folder, protocol, output):
    """Session 4 as a crosscorpus folder. `evals` are the eval folders of the earlier runs, the pilot's first."""
    evals, formats_folder, output = [Path(e) for e in evals], Path(formats_folder), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    source = evals[0] / 'manifests' / 'orig.jsonl'
    source_sha = d2.digest(source)
    rows = [dict(r, group=r['label']) for r in d2.read_jsonl(source)]
    d2.write_jsonl(output / 'mcq.jsonl', rows)
    made = d2.load(formats_folder / 'formats.json')
    (output / 'formats').mkdir()
    manifests = dict(mcq=dict(sha256=d2.digest(output / 'mcq.jsonl'), rows=len(rows), source_sha256=source_sha))
    for name in formats.FORMATS:
        if d2.digest(formats_folder / f'{name}.jsonl') != made['manifests'][name]['sha256']:
            raise ValueError(f'{name}: the format manifest changed after it was built')
        shutil.copyfile(formats_folder / f'{name}.jsonl', output / 'formats' / f'{name}.jsonl')
        manifests[name] = dict(sha256=made['manifests'][name]['sha256'], rows=made['manifests'][name]['rows'])
    ids = {task: [r['id'] for r in d2.read_jsonl(crosscorpus.manifest_path(output, task))] for task in crosscorpus.TASKS}
    arms = {}
    generation = dict(mcq=d2.load(protocol)['generation'], **{n: dict(formats.GENERATION) for n in formats.FORMATS})   # as those runs used
    for folder in evals:
        for run in sorted(p for p in (folder / 'runs').iterdir() if (p / 'orig' / 'metrics.json').exists()):
            if run.name in arms:
                raise ValueError(f'{run.name} was evaluated in two of the folders')
            metrics = d2.load(run / 'orig' / 'metrics.json')
            if metrics['manifest_sha256'] != source_sha or d2.digest(run / 'orig' / 'predictions.jsonl') != metrics['predictions_sha256']:
                raise ValueError(f'{run.name}: scored on another manifest, or its predictions changed')
            sources = dict(mcq=run / 'orig' / 'predictions.jsonl',
                           **{n: formats_folder / 'runs' / run.name / n / 'predictions.jsonl' for n in formats.FORMATS})
            if not all(p.exists() for p in sources.values()):
                continue                                   # an arm without all four tasks is left out, and listed as such
            for task, path in sources.items():
                if [r['id'] for r in d2.read_jsonl(path)] != ids[task]:
                    raise ValueError(f'{run.name}/{task}: predictions do not follow the manifest')
                target = output / 'runs' / run.name / task
                target.mkdir(parents=True)
                shutil.copyfile(path, target / 'predictions.jsonl')
                d2.write(target / 'metrics.json', dict(task=task, arm=run.name, copied_from=str(path), manifest_sha256=manifests[task]['sha256'],
                                                       generation=generation[task],
                                                       predictions_sha256=d2.digest(target / 'predictions.jsonl')))
            arms[run.name] = str(folder)
    info = dict(corpus='IEMOCAP Session 4', utterances=len({r['utterance'] for r in rows}), actors=len({r['dialogue'] for r in rows}),
                clusters='dialogues', trained_here=True, mcq_generation=generation['mcq'],
                text_generation=dict(formats.GENERATION), manifests=manifests, arms=arms)
    d2.write(output / 'crosscorpus.json', info)
    return info


def same(folder, first, second):
    """Rows on which two arms answered differently, per task."""
    out = {}
    for task in crosscorpus.TASKS:
        a, b = (d2.read_jsonl(Path(folder) / 'runs' / arm / task / 'predictions.jsonl') for arm in (first, second))
        if [r['id'] for r in a] != [r['id'] for r in b]:
            raise ValueError(f'{task}: the two arms answered different questions')
        out[task] = dict(rows=len(a), different=sum(x['prediction'] != y['prediction'] for x, y in zip(a, b)))
    return out


def judge(result):
    """The declared readings for one corpus, from a crosscorpus.summarize result."""
    contrasts = result['contrasts']

    def low(key, measure):
        return contrasts[key][measure]['interval_95_pp'][0]

    bands = {}
    for arm in SHALLOW:
        key = f'{arm} - frozen-base'
        if key not in contrasts:
            continue
        gains = [n for n in formats.FORMATS if low(key, n) > 0]
        if low(key, 'mcq_training_order') <= 0:
            reading = 'no gain'
        elif low(key, 'mcq_other_orders') > 0 and len(gains) >= CRITERIA['min_formats_with_gain']:
            reading = 'transfers'
        elif low(key, 'mcq_other_orders') <= 0 and not gains:
            reading = 'format-bound'
        else:
            reading = 'partial'
        bands[arm] = dict(reading=reading, formats_with_gain=gains,
                          **{m: contrasts[key][m] for m in ('mcq_training_order', 'mcq_other_orders')})
    out = dict(bands=bands, order_binding=None)
    needed = ['A-last4 - A-first4', 'A-last4 - A-mid4', 'A-first4 - A-all', 'A-mid4 - A-all']
    if all(k in contrasts for k in needed):
        def exceeds(key):
            gap = contrasts[key]['order_gap']
            return gap['delta_pp'] >= CRITERIA['min_order_gap_excess_pp'] and gap['interval_95_pp'][0] > 0

        late = [exceeds('A-last4 - A-first4'), exceeds('A-last4 - A-mid4')]
        others = [exceeds('A-first4 - A-all'), exceeds('A-mid4 - A-all')]
        if all(late) and not any(others):
            out['order_binding'] = 'specific to the last layers'
        elif all(others):
            out['order_binding'] = 'in every shallow band'
        elif not any(late) and not any(others):
            out['order_binding'] = 'not distinguishable between bands'
        else:
            out['order_binding'] = 'mixed'
        out['order_gap_excess'] = {k: contrasts[k]['order_gap'] for k in needed}
    return out


def summarize(folders, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    result, text = {}, []
    for name, folder in folders.items():
        summary = crosscorpus.summarize(folder)
        summary['band_readings'] = judge(summary)
        result[name] = summary
        text += [crosscorpus.render(summary).split('\nReadings declared before scoring:')[0].rstrip(), '', 'Readings of the layer-band declaration:', '']
        text += [f"- {arm}: **{b['reading']}** (formats with a gain over the base: {', '.join(b['formats_with_gain']) or 'none'})"
                 for arm, b in summary['band_readings']['bands'].items()]
        text += [f"- order binding: **{summary['band_readings']['order_binding'] or 'not available'}**", '']
    d2.write(output / 'bands.json', dict(criteria=CRITERIA, corpora=result))
    (output / 'bands.md').write_text('\n'.join(text) + '\n')
    return '\n'.join(text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('assemble')
    p.add_argument('--evals', type=Path, nargs='+', required=True, help='eval folders of the earlier runs, the pilot first')
    p.add_argument('--formats', type=Path, required=True, help='formats folder of the third run')
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p = commands.add_parser('same')
    p.add_argument('--folder', type=Path, required=True)
    p.add_argument('--arms', nargs=2, required=True)
    p.add_argument('--max-share', type=float, default=0.01, help='exit 1 when more than this share of rows differs in any task')
    p = commands.add_parser('summarize')
    p.add_argument('--session4', type=Path, required=True)
    p.add_argument('--cremad', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'assemble':
        info = assemble(args.evals, args.formats, args.protocol, args.output)
        print(json.dumps(dict(utterances=info['utterances'], dialogues=info['actors'], arms=sorted(info['arms'])), indent=1))
    elif args.command == 'same':
        found = same(args.folder, *args.arms)
        print(json.dumps(found))
        raise SystemExit(int(any(v['different'] > args.max_share * v['rows'] for v in found.values())))
    else:
        print(summarize({'IEMOCAP Session 4': args.session4, 'CREMA-D': args.cremad}, args.output))
