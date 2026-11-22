"""Run independent Swift SFT arms, seal dev selection for all arms, then evaluate test-clean."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

from .prepare import digest

PACKAGE = 'experiments.ms_swift_librispeech'


def write(path, data):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n')
    tmp.replace(path)


def load(path):
    return json.loads(path.read_text())


def run(config, data, out):
    cfg = load(config)
    if set(cfg['methods']) != {'lora', 'ta-lora'} or len(cfg['methods']) != 2:
        raise ValueError('Comparison requires exactly one run of each configured method')
    data = data.resolve(strict=True)
    splits = load(data/'splits.json')
    manifests = {'train': data/'train.jsonl', 'dev': data/'dev-clean.jsonl', 'test': data/'test-clean.jsonl'}
    keys = {'train': 'train-clean-100', 'dev': 'dev-clean', 'test': 'test-clean'}
    hashes = {k: digest(v) for k, v in manifests.items()}
    if any(hashes[k] != splits[keys[k]]['sha256'] for k in keys):
        raise ValueError('Manifest digest mismatch')
    out.mkdir(parents=True, exist_ok=True)
    snapshot = out/'config.json'
    if snapshot.exists() and snapshot.read_bytes() != config.read_bytes():
        raise ValueError('Existing run uses a different configuration; use a new directory')
    if not snapshot.exists():
        snapshot.write_bytes(config.read_bytes())
    identity = {'methods':cfg['methods'], 'config_sha256':digest(snapshot),'manifests':hashes}
    if (out/'inputs.json').exists() and load(out/'inputs.json') != identity:
        raise ValueError('Inputs differ from initial run')
    write(out/'inputs.json', identity)
    state = {}
    def status(stage, method=None):
        state.update(stage=stage, method=method, time=datetime.now(timezone.utc).isoformat())
        write(out/'status.json',state)
    def command(name, stage, args):
        status(stage, name)
        with (out/f'{name}-{stage}.log').open('a') as log:
            print(json.dumps(list(map(str,args))), file=log, flush=True)
            subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, check=True)
    train_results, selection, metrics = {}, {}, {}
    try:
        for name in cfg['methods']:
            dest = out/name
            if not (dest/'result.json').is_file():
                args = [sys.executable, '-m', PACKAGE+'.train', '--config',snapshot,'--method',name,
                        '--train',manifests['train'],'--dev',manifests['dev'],'--output',dest]
                checkpoints = sorted(dest.glob('checkpoint-*'),key=lambda p:int(p.name.split('-')[-1]))
                if checkpoints:
                    cp = checkpoints[-1]
                    if not all((cp/f).exists() for f in ('trainer_state.json','optimizer.pt','scheduler.pt')):
                        raise ValueError(f'Incomplete resume checkpoint: {cp}')
                    args += ['--resume', cp]
                command(name,'train',args)
            res = load(dest/'result.json')
            cp = Path(res['best_checkpoint']).resolve(strict=True)
            if not cp.is_relative_to(dest.resolve()):
                raise ValueError('Unexpected checkpoint location')
            train_results[name] = res
            selection[name] = {'checkpoint':str(cp), 'adapter_sha256': digest(cp/'adapter_model.safetensors'),
                               'best_dev_wer': res['best_wer']}
        if train_results['lora']['init_sha256'] != train_results['ta-lora']['init_sha256']:
            raise ValueError('LoRA and TA-LoRA did not start from identical low-rank factors')
        if (not train_results['lora']['input_trajectory'] or
                train_results['lora']['input_trajectory'] != train_results['ta-lora']['input_trajectory']):
            raise ValueError('Initial native training microbatches differ')
        if (out/'dev-selection.json').exists() and load(out/'dev-selection.json') != selection:
            raise ValueError('A previously selected checkpoint changed')
        write(out/'dev-selection.json', selection)
        # Reproduce the first complete development batch from each selected checkpoint
        # on a clean base, before either arm opens test-clean for scoring.
        replay = out/'dev-replay.jsonl'
        n = cfg['generation']['batch_size']
        replay.write_text('\n'.join(manifests['dev'].read_text().splitlines()[:n])+'\n')
        for name in cfg['methods']:
            replay_out = out/(name+'-reloaded-dev')
            if not (replay_out/'metrics.json').exists():
                command(name,'reload',[sys.executable,'-m',PACKAGE+'.evaluate','--method',name,'--config',snapshot,
                                        '--checkpoint',selection[name]['checkpoint'],'--manifest',replay,'--output',replay_out])
            best = train_results[name]['best_step']
            original = (out/name/'dev'/f'step-{best}'/'hypotheses.jsonl').read_text().splitlines()[:n]
            if (replay_out/'predictions.jsonl').read_text().splitlines() != original:
                raise ValueError(f'{name}: selected checkpoint did not reproduce the development predictions')
        for name in cfg['methods']:
            dest = out/(name+'-test-clean')
            if not (dest/'metrics.json').is_file():
                command(name,'test',[sys.executable,'-m',PACKAGE+'.evaluate','--method',name,'--config',snapshot,
                                     '--checkpoint',selection[name]['checkpoint'],'--manifest',manifests['test'],
                                     '--output',dest])
            metric = load(dest/'metrics.json')
            if (metric['checkpoint'] != selection[name]['checkpoint'] or metric['manifest_sha256'] != hashes['test']
                    or metric['generation'] != cfg['generation'] or metric['utterances'] != splits['test-clean']['utterances']):
                raise ValueError('Final scoring identity differs from selection')
            metrics[name] = metric
        a,b = metrics['lora'],metrics['ta-lora']
        if a['reference_words'] != b['reference_words']:
            raise ValueError('Inconsistent WER denominators')
        delta = 100*(b['word_errors']-a['word_errors'])/a['reference_words']
        winner = 'tie' if delta == 0 else ('ta-lora' if delta < 0 else 'lora')
        write(out/'comparison.json',dict(train=train_results,test=metrics,ta_minus_lora_wer_points=delta,
                                         lower_test_wer=winner))
        lines=['# LibriSpeech ASR', '', '| Method | Best dev WER | Test WER | Parameters | Optimizer steps | Training hours | Peak allocated GiB |',
               '|---|---:|---:|---:|---:|---:|---:|']
        for name,res in train_results.items():
            lines.append(f"| {name} | {res['best_wer']:.2%} | {metrics[name]['wer']:.2%} | "
                         f"{res['trainable_parameters']:,} | {res['global_step']} | "
                         f"{res['training_seconds']/3600:.3f} | {res['peak_allocated_bytes']/2**30:.2f} |")
        lines += ['',f'TA-LoRA minus LoRA on test-clean: {delta:+.3f} WER percentage points. Single seed.']
        (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
        status('complete')
    except BaseException as exc:
        state.update(failed_stage=state.get('stage'),error=str(exc))
        status('failed',state.get('method'))
        raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('config','data','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    with (output/'.lock').open('a') as locked:
        fcntl.flock(locked,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(args.config.resolve(),args.data,output)
