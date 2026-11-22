"""Develop three paired AVQA SFT adapters, then score their frozen checkpoints on held-out questions."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

from experiments.ms_swift_librispeech.compare import load, write
from experiments.ms_swift_librispeech.prepare import digest
from .evaluate import correct_letters, read_rows, summarize


METHODS = {'lora', 'dora', 'ta-lora'}


def pin(path, value):
    if path.exists():
        if load(path) != value:
            raise ValueError(f'Frozen identity changed: {path}')
    else:
        write(path, value)


def source_groups(rows):
    return {r['source_id'] for r in rows if r.get('source_id')}


def audio_paths(rows):
    paths = {Path(r['audios'][0]).resolve(strict=True) for r in rows}
    if any(not p.is_file() or p.stat().st_size == 0 for p in paths):
        raise ValueError('Empty or missing audio')
    return paths


def check_disjoint(a, b):
    overlap = source_groups(a) & source_groups(b)
    if overlap or audio_paths(a) & audio_paths(b):
        raise ValueError(f'Splits share source recordings: {sorted(overlap)[:5]}')


def verify_score(folder, rows, manifest, seal):
    score = load(folder/'metrics.json')
    records = [json.loads(line) for line in (folder/'predictions.jsonl').read_text().splitlines()]
    if score['adapter_sha256'] != seal['adapter_sha256'] or score['manifest_sha256'] != digest(manifest):
        raise ValueError(f'Scoring inputs differ from frozen adapter/manifest: {folder}')
    for row, rec in zip(rows, records, strict=True):
        choices = row['choices']
        reference = row['messages'][1]['content']
        correct = correct_letters(row)
        index = ord(rec['prediction']) - ord('A')
        if (row['id'] != rec['id'] or rec['reference'] != reference or correct != rec['correct_letters']
                or not 0 <= index < len(choices) or choices[index] != rec['answer_text']
                or rec['is_correct'] != (rec['prediction'] in correct)
                or any(row.get(k) != rec.get(k) for k in ('domain', 'category', 'difficulty', 'question_relation', 'question_type') if row.get(k))):
            raise ValueError(f"Inconsistent prediction for {row['id']}")
    if any(score[k] != v for k, v in summarize(records).items()):
        raise ValueError(f'Stored accuracy differs from predictions: {folder}')
    return score


def run(cfgfile, train, dev, test, out, stage):
    cfg = load(cfgfile)
    if len(cfg['methods']) != 3 or set(cfg['methods']) != METHODS or cfg.get('task') != 'aqa':
        raise ValueError('Expected an AQA configuration with LoRA, DoRA and TA-LoRA')
    out.mkdir(parents=True, exist_ok=True)
    snapshot = out/'config.json'
    if snapshot.exists() and snapshot.read_bytes() != cfgfile.read_bytes():
        raise ValueError('Config changed; choose a fresh output directory')
    if not snapshot.exists():
        snapshot.write_bytes(cfgfile.read_bytes())
    train_rows, dev_rows = read_rows(train), read_rows(dev)
    if not all(r.get('source_id') for r in train_rows + dev_rows):
        raise ValueError('Need an audio source ID per AVQA example for split validation')
    check_disjoint(train_rows, dev_rows)
    if {r['id'] for r in train_rows} & {r['id'] for r in dev_rows}:
        raise ValueError('Train and dev question IDs overlap')
    pin(out/'training-inputs.json', dict(config_sha256=digest(snapshot), train_sha256=digest(train), dev_sha256=digest(dev),
                                         train_questions=len(train_rows), dev_questions=len(dev_rows)))

    def status(phase, method=None, **kwargs):
        write(out/'status.json', dict(stage=phase, method=method, utc=datetime.now(timezone.utc).isoformat(), **kwargs))

    def command(method, phase, module, args):
        argv = [sys.executable, '-m', module, '--config', str(snapshot), '--method', method, *map(str, args)]
        status(phase, method)
        with (out/f'{method}-{phase}.log').open('a') as log:
            print(json.dumps(argv), file=log, flush=True)
            subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=True)

    def evaluate(method, phase, source, rows, seal):
        dest = out/method/phase
        cp = Path(seal['checkpoint'])
        if digest(cp/'adapter_model.safetensors') != seal['adapter_sha256'] or digest(cp/'adapter_config.json') != seal['config_sha256']:
            raise ValueError(f'Changed checkpoint: {cp}')
        if not (dest/'metrics.json').is_file():
            command(method, phase, 'experiments.ms_swift_aqa.evaluate', ['--checkpoint', cp, '--manifest', source, '--output', dest])
        return verify_score(dest, rows, source, seal)

    results, seals = {}, {}
    try:
        for method in cfg['methods']:
            arm = out/method
            if not (arm/'result.json').is_file():
                if stage == 'final':
                    raise ValueError(f'Development incomplete: {method}')
                args = ['--train', train, '--dev', dev, '--output', arm]
                checkpoints = sorted(arm.glob('checkpoint-[0-9]*'), key=lambda p: int(p.name.rsplit('-', 1)[1]))
                if checkpoints:
                    args.extend(['--resume', checkpoints[-1]])
                command(method, 'train', 'experiments.ms_swift_librispeech.train', args)
            result = load(arm/'result.json')
            by_step = {int(f.parent.name[5:]): load(f)['accuracy'] for f in (arm/'dev').glob('step-*/metrics.json')}
            best_step = max(sorted(by_step), key=by_step.get)
            cp = arm/f'checkpoint-{best_step}'
            if (result['best_step'] != best_step or result['best_accuracy'] != by_step[best_step]
                    or Path(result['best_checkpoint']).resolve(strict=True) != cp.resolve(strict=True)):
                raise ValueError(f'Incorrect development selection for {method}')
            results[method] = result
            seals[method] = dict(checkpoint=str(cp), step=best_step, dev_accuracy=by_step[best_step],
                                 adapter_sha256=digest(cp/'adapter_model.safetensors'), config_sha256=digest(cp/'adapter_config.json'))
        anchor = results['lora']
        if not anchor['input_trajectory'] or any(r['init_sha256'] != anchor['init_sha256'] or
                r['input_trajectory'] != anchor['input_trajectory'] for r in results.values()):
            raise ValueError('Methods differed at shared initialization or initial minibatches')
        pin(out/'selection.json', seals)
        for method, seal in seals.items():
            metric = evaluate(method, 'reload-dev', dev, dev_rows, seal)
            original = out/method/'dev'/f"step-{seal['step']}"/'hypotheses.jsonl'
            if metric['accuracy'] != seal['dev_accuracy'] or digest(original) != digest(out/method/'reload-dev/predictions.jsonl'):
                raise ValueError(f'Clean base failed to reproduce selected predictions: {method}')
        if stage == 'develop':
            status('development-complete')
            return
        test_rows = read_rows(test)
        check_disjoint(train_rows + dev_rows, test_rows)
        pin(out/'final-inputs.json', dict(test_sha256=digest(test), questions=len(test_rows), selection_sha256=digest(out/'selection.json')))
        metrics = {m: evaluate(m, 'reload-test', test, test_rows, seal) for m, seal in seals.items()}
        delta = {base: 100*(metrics['ta-lora']['accuracy'] - metrics[base]['accuracy']) for base in ('lora', 'dora')}
        write(out/'comparison.json', dict(train=results, test=metrics, ta_minus_baseline_accuracy_pp=delta))
        status('complete')
    except BaseException as exc:
        state = load(out/'status.json') if (out/'status.json').is_file() else {}
        status('failed', state.get('method'), error=str(exc), previous_stage=state.get('stage'))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('config', 'train', 'dev', 'output'):
        parser.add_argument('--' + key, required=True, type=Path)
    parser.add_argument('--test', type=Path)
    parser.add_argument('--stage', choices=('develop', 'final', 'all'), default='all')
    args = parser.parse_args()
    if args.stage != 'develop' and args.test is None:
        parser.error('--test is required for final scoring')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args.config.resolve(), args.train.resolve(), args.dev.resolve(), args.test.resolve() if args.test else None, out, args.stage)
