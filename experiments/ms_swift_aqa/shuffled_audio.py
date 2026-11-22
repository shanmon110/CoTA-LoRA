"""Score frozen adapters on AVQA dev after permuting audio across unrelated source videos."""
import argparse
import json
from pathlib import Path
from random import Random
import subprocess
import sys

from .run import METHODS


def swapped(rows, hashes, seed=42):
    n = len(rows)
    order = list(range(n))
    Random(seed).shuffle(order)
    # Keep the full audio multiset, and reject pairings of the same source or waveform.
    for shift in range(1, n):
        perm = dict(zip(order, order[shift:] + order[:shift], strict=True))
        if all(rows[i]['source_id'] != rows[j]['source_id'] and hashes[rows[i]['audios'][0]] != hashes[rows[j]['audios'][0]]
               for i, j in perm.items()):
            return [dict(row, audios=rows[perm[i]]['audios']) for i, row in enumerate(rows)]
    raise ValueError('Could not construct a distinct-source audio permutation')


def run(run_dir, data):
    if not (run_dir / 'verified-comparison.json').exists():
        raise ValueError('Run this diagnostic after frozen final scoring')
    config = run_dir / 'config.json'
    seal = json.loads((run_dir / 'selection.json').read_text())
    dest = run_dir / 'diagnostics' / 'swapped-dev-audio'
    dest.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in (data / 'dev.jsonl').read_text().splitlines()]
    augmented = swapped(rows, json.loads((data / 'audio-hashes.json').read_text()))
    manifest = dest / 'questions.jsonl'
    manifest.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in augmented))
    results = {}
    for method in sorted(METHODS):
        branch = dest / method
        with (dest / (method + '.log')).open('w') as log:
            subprocess.run([sys.executable, '-m', 'experiments.ms_swift_aqa.evaluate', '--config', config,
                            '--method', method, '--checkpoint', seal[method]['checkpoint'], '--manifest', manifest,
                            '--output', branch], stdout=log, stderr=subprocess.STDOUT, check=True)
        metrics = json.loads((branch / 'metrics.json').read_text())
        assert metrics['utterances'] == len(rows) and metrics['adapter_sha256'] == seal[method]['adapter_sha256']
        normal = json.loads((run_dir / method / 'reload-dev' / 'metrics.json').read_text())['accuracy']
        results[method] = dict(normal_dev_accuracy=normal, swapped_dev_accuracy=metrics['accuracy'],
                               decrease_pp=100 * (normal - metrics['accuracy']))
    (dest / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('run', 'data'):
        parser.add_argument('--' + flag, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.run, args.data), indent=2))
