"""Train an AVQA-selected adapter once and replay it on two untouched target datasets."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.ms_swift_aqa.run import pin, verify_score
from experiments.ms_swift_aqa.run_tuners import seal_selection
from experiments.ms_swift_aqa.evaluate import read_rows
from experiments.ms_swift_aqa.summarize_pilot import load_official, score_arm
from experiments.ms_swift_iemocap.evaluate import metrics as ser_metrics
from experiments.ms_swift_librispeech.compare import load, write
from experiments.ms_swift_librispeech.train import digest

HERE = Path(__file__).resolve().parent


def write_ser(config, method, checkpoint, manifest, folder):
    import time
    import torch
    from experiments.ms_swift_iemocap.evaluate import score
    from experiments.ms_swift_librispeech.train import load_best
    from experiments.ms_swift_aqa.tuner_suite import checkpoint_identity

    cfg = load(config)
    model, template = load_best(cfg, method, checkpoint)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    pred = folder/'predictions.jsonl'
    result = score(model, template, manifest, cfg['generation'], pred)
    torch.cuda.synchronize()
    write(folder/'metrics.json', dict(result, method=method, manifest_sha256=digest(manifest),
          eval_seconds=time.perf_counter()-start, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
          peak_reserved_bytes=torch.cuda.max_memory_reserved(), predictions_sha256=digest(pred),
          **checkpoint_identity(checkpoint, method)))


def check_score_config(score, seal, historical_selection=None):
    """Bind inference semantics as well as the weight file to source-dev selection."""
    got = score.get('config_sha256') or score.get('identity', {}).get('adapter_config_sha256')
    if got is None and historical_selection is not None:
        # Earlier archived AQA scores only recorded weights. Their original selection
        # manifest binds both digests to the exact selected directory and step.
        fields = ('checkpoint', 'step', 'adapter_sha256')
        if all(historical_selection.get(k) == seal[k] for k in fields):
            got = historical_selection.get('config_sha256')
    if got != seal['config_sha256']:
        raise ValueError('Scored adapter config differs from the source-development selection')


def resume_arguments(train_dir, method):
    """Keep interrupted artifacts and return the latest complete preemption point."""
    from experiments.ms_swift_aqa.tuner_suite import checkpoint_files
    if not train_dir.exists() or not any(train_dir.iterdir()):
        return []
    if (train_dir/'result.json').exists():
        raise ValueError('Refusing to restart a completed training run')
    checkpoints = sorted(train_dir.glob('checkpoint-[0-9]*'), key=lambda p:int(p.name.split('-')[-1]), reverse=True)
    required = ['trainer_state.json','optimizer.pt','scheduler.pt','rng_state.pth','efficiency.json',
                'input-trajectory.json','sample-trajectory.json']
    extra = {'pissa':['paper_residual.safetensors','paper_residual.sha256'],
             'lora-ga':['paper_residual.safetensors','paper_residual.sha256'],
             'randlora':['randlora-basis.json'],
             'loram-train-small':['loram-masks.json','recovered/adapter_config.json','recovered/adapter_model.safetensors']}
    last = None
    for candidate in checkpoints:
        try:
            if any(not (candidate/f).is_file() or (candidate/f).stat().st_size == 0 for f in required+extra.get(method,[])):
                continue
            checkpoint_files(candidate,method)
            state = load(candidate/'trainer_state.json')
            if method == 'rolora-federated':
                local = load(train_dir/'protocol.json')['config']['method_options'][method]
                if state['global_step'] % (local['clients']*local['local_steps']) and not (candidate/'federated-round.pt').is_file():
                    continue
            if state['global_step'] == int(candidate.name.split('-')[-1]):
                last = candidate
                break
        except (KeyError, ValueError, FileNotFoundError):
            continue
    if last is not None and last == checkpoints[0]:
        return ['--resume', last]
    # An interrupted first save cannot resume. Retain the old run intact; a gap
    # after a recoverable checkpoint needs only its partial newer outputs moved.
    archived = train_dir.parent/'_interrupted'
    archived.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    dest = archived/f'{train_dir.name}-{stamp}'
    if last is None:
        train_dir.rename(dest)
        return []
    dest.mkdir()
    cutoff = int(last.name.split('-')[-1])
    for item in checkpoints[:checkpoints.index(last)]:
        item.rename(dest/item.name)
    for dev in (train_dir/'dev').glob('step-*'):
        if dev.name[5:].isdigit() and int(dev.name[5:]) > cutoff:
            (dest/'dev').mkdir(exist_ok=True)
            dev.rename(dest/'dev'/dev.name)
    return ['--resume', last]


def check_ser(folder, manifest, sealed):
    pred = folder/'predictions.jsonl'
    reported = load(folder/'metrics.json')
    if reported['manifest_sha256'] != digest(manifest) or reported['predictions_sha256'] != digest(pred):
        raise ValueError('IEMOCAP manifest or predictions changed')
    actual_hash = reported.get('adapter_sha256') or reported['identity']['weights_sha256']
    check_score_config(reported, sealed)
    if actual_hash != sealed['adapter_sha256']:
        raise ValueError('IEMOCAP adapter differs from the source-development selection')
    pairs = []
    with manifest.open() as raw, pred.open() as output:
        for item, result in zip(map(json.loads, raw), map(json.loads, output), strict=True):
            if (item['session'] != 5 or item['id'] != result['id']
                    or result['reference'] != item['messages'][1]['content']):
                raise ValueError(f'IEMOCAP alignment error at {item["id"]}')
            pairs.append((result['reference'], result['prediction']))
    recalc = ser_metrics(pairs)
    if len(pairs) != 1241 or any(reported[k] != v for k, v in recalc.items()):
        raise ValueError('Incomplete or inconsistent IEMOCAP scores')
    return recalc


def process(protocol, out, method, archived=False):
    training_method = 'lora' if method == 'lora-recompute' else method
    data = Path(protocol['avqa_mmau'])
    ser = Path(protocol['ser'])
    archive = Path(protocol['archive'])
    settings_path = HERE/'configs'/f'{method}.json'
    env = os.environ.copy()
    if method == 'randlora':
        env['PYTHONPATH'] = protocol['randlora_peft'] + os.pathsep + env.get('PYTHONPATH', '')
    logdir = out/'logs'
    logdir.mkdir(exist_ok=True)
    def launch(stage, entry, *args):
        from .provenance import create_snapshot, environment, source_tree
        files, rand_version = source_tree(protocol, out)
        snapshot = create_snapshot(out/'sources', files, environment(rand_version))
        cmd = [sys.executable, '-m', entry, *map(str, args)]
        status = dict(stage=stage, method=method, time=datetime.now(timezone.utc).isoformat(), command=cmd,
                      source_snapshot=str(snapshot))
        write(out/f'status-{method}.json', status)
        with (logdir/f'{stage}-{method}.log').open('a') as log:
            print(json.dumps(status), file=log, flush=True)
            subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    if archived:
        ref = load(archive/'results'/f'{method}.json')
        train_dir = Path(ref['artifact'])
        cfg = load(train_dir/'protocol.json')['config']
    else:
        cfg = load(Path(protocol['base_config']))
        overlay = load(settings_path)
        if (overlay['method'] != method or set(overlay)-{'method', 'settings', 'rank', 'alpha', 'target_modules', 'gradient_checkpointing'}):
            raise ValueError(f'Unrecognized paper settings in {settings_path}')
        cfg.update(methods=[training_method], method_options={training_method:overlay['settings']})
        cfg['adapter'].update({k:overlay[k] for k in ('rank','alpha','target_modules') if k in overlay})
        cfg['training']['gradient_checkpointing'] = overlay.get('gradient_checkpointing', True)
        if method == 'loft':
            cfg['training']['max_grad_norm'] = 0.0  # LoFT clips its active factor inside its optimizer.
        train_dir = out/'train'/method
    config_file = out/'configs'/f'{method}.json'
    config_file.parent.mkdir(exist_ok=True)
    pin(config_file, cfg)
    if cfg['model_revision'] != protocol['model_revision'] or cfg['generation'] != {'batch_size':2}:
        raise ValueError('Unexpected backbone or decoding settings')
    known_target = method == 'known-target-qo'
    generated = method == 'loragen' or known_target
    inference_method = 'lora' if generated else training_method
    if known_target:
        train_dir = out/'theory-01/omni-control'
        meta = load(train_dir/'construction.json')
        if (meta['adapter_rank'] != cfg['adapter']['rank'] or meta['adapter_alpha'] != cfg['adapter']['alpha']
                or set(meta['target_modules']) != set(cfg['adapter']['target_modules'])):
            raise ValueError('Constructed weights differ from the independent control configuration')
        from experiments.ms_swift_aqa.tuner_suite import checkpoint_identity
        checkpoint = train_dir/'checkpoint-constructed'
        identity = checkpoint_identity(checkpoint, inference_method)
        if (meta['checkpoint_identity'] != identity or meta['model_revision'] != protocol['model_revision']
                or any(meta[f'{k}_sha256'] != protocol['hashes'][k] for k in ('train','dev'))):
            raise ValueError('Constructed adapter or its source provenance has changed')
        sealed = dict(method=method, strategy='source-selected dense teacher; independent linear construction',
                      teacher_selection=meta['teacher_selection'], construction_sha256=digest(train_dir/'construction.json'),
                      step=None, checkpoint=str(checkpoint), **identity)
    elif generated:
        library = out/'train/lora-mixer/source-experts/library.json'
        mixer_protocol = load(out/'train/lora-mixer/protocol.json')
        if mixer_protocol['train_sha256'] != protocol['hashes']['train']:
            raise ValueError('Source expert library came from different training data')
        pin(out/'configs/loragen-source.json', dict(library=str(library), sha256=digest(library)))
        if not (train_dir/'generation.json').is_file():
            launch('generator', 'experiments.ms_swift_aqa.paper_late_loragen', '--config', config_file,
                   '--library', library, '--output', train_dir)
        from experiments.ms_swift_aqa.paper_late_loragen import validate_generation
        validate_generation(config_file, library, train_dir)
        from experiments.ms_swift_aqa.tuner_suite import checkpoint_identity
        checkpoint = train_dir/'checkpoint-generated'
        sealed = dict(method=method, strategy='fixed source-task description and single generated adapter',
                      step=None, checkpoint=str(checkpoint), **checkpoint_identity(checkpoint, inference_method))
    elif not archived and not (train_dir/'result.json').exists():
        resume = resume_arguments(train_dir, training_method)
        launch('train', 'experiments.ms_swift_librispeech.train', '--config', config_file, '--method', training_method,
               '--train', data/'train.jsonl', '--dev', data/'dev.jsonl', '--output', train_dir, *resume)
    if not generated:
        recorded = load(train_dir/'protocol.json')
        if recorded['train_sha256'] != protocol['hashes']['train'] or recorded['dev_sha256'] != protocol['hashes']['dev']:
            raise ValueError('Training or selection used a different manifest')
        sealed = seal_selection(train_dir, cfg, training_method)
    pin(out/'configs'/f'{method}-selected.json', sealed)
    checkpoint = Path(sealed['checkpoint'])
    for split, manifest in [('dev',data/'dev.jsonl'), ('test',data/'test.jsonl')]:
        dest = train_dir/('reload-'+split)
        if not (dest/'metrics.json').is_file():
            launch('reload-'+split, 'experiments.ms_swift_aqa.evaluate', '--config',config_file,
                   '--method',inference_method,'--checkpoint',checkpoint,'--manifest',manifest,'--output',dest)
        score = verify_score(dest, read_rows(manifest), manifest, sealed)
        legacy = None
        if archived and 'config_sha256' not in score and (train_dir.parent/'selection.json').is_file():
            legacy = load(train_dir.parent/'selection.json').get(method)
        check_score_config(score, sealed, legacy)
        if not generated and split == 'dev' and digest(dest/'predictions.jsonl') != digest(train_dir/'dev'/f'step-{sealed["step"]}'/'hypotheses.jsonl'):
            raise ValueError('Best adapter reload disagrees with training-time development predictions')
        if generated and split == 'dev':
            if known_target:
                cost = dict(global_step=0, trainable_parameters=meta['trainable_parameters'],
                            training_seconds=meta['construction_seconds'],
                            peak_allocated_bytes=meta.get('peak_allocated_bytes', 0),
                            construction_device=meta['device'], teacher_selection=meta['teacher_selection'])
            else:
                meta = load(train_dir/'generation.json')
                cost = dict(global_step=meta['optimizer_updates'], trainable_parameters=meta['trainable_parameters'],
                            training_seconds=meta['training_seconds'], peak_allocated_bytes=meta['peak_allocated_bytes'])
            pin(train_dir/'result.json', dict(best_accuracy=load(dest/'metrics.json')['accuracy'], best_step=None,
                protocol='Fixed generated adapter; development accuracy is diagnostic only', **cost))
    dest = out/'ser'/method
    if not (dest/'metrics.json').is_file():
        launch('ser', __spec__.name, 'ser', '--config', config_file, '--method', inference_method, '--checkpoint', checkpoint,
               '--manifest', ser, '--folder', dest)
    ser_result = check_ser(dest, ser, sealed)
    source = load_official(data, Path(protocol['official_mmau']))
    mmau, _, _ = score_arm(train_dir, method, *source, official_dir=out/'mmau-official'/method)
    summary = dict(method=method, archived_training=archived, ser=ser_result, mmau=mmau,
                   checkpoint=sealed, train_dir=str(train_dir), ser_dir=str(dest))
    (out/'results').mkdir(exist_ok=True)
    write(out/'results'/f'{method}.json', summary)
    write(out/f'status-{method}.json', dict(stage='verified', method=method))
    print(method, ser_result['uar'], mmau['official_text_accuracy'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    score = sub.add_parser('ser')
    for name in ('config','checkpoint','manifest','folder'):
        score.add_argument('--'+name, type=Path, required=True)
    score.add_argument('--method',required=True)
    runner = sub.add_parser('run')
    runner.add_argument('--protocol', type=Path, default=HERE/'protocol.json')
    runner.add_argument('--out', type=Path, required=True)
    runner.add_argument('--archived', action='store_true')
    runner.add_argument('methods', nargs='+')
    args = parser.parse_args()
    if args.action == 'ser':
        return write_ser(args.config,args.method,args.checkpoint,args.manifest,args.folder)
    protocol = load(args.protocol)
    data = Path(protocol['avqa_mmau'])
    for name, sha in protocol['hashes'].items():
        path = Path(protocol['ser']) if name == 'ser' else data/f'{name}.jsonl'
        if digest(path) != sha:
            raise ValueError(f'Pinned {name} manifest changed')
    args.out.mkdir(parents=True, exist_ok=True)
    pin(args.out/'protocol.json', protocol)
    errors = []
    for method in args.methods:
        with Path(protocol['lock']).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                process(protocol,args.out,method,args.archived)
            except Exception as exc:
                import traceback
                errors.append(method)
                write(args.out/f'status-{method}.json', dict(stage='failed',method=method,error=str(exc),
                                                            traceback=traceback.format_exc()))
                traceback.print_exc()
            from .report import publish
            publish(args.out)
    if errors:
        raise SystemExit('Failed methods: '+', '.join(errors))


if __name__ == '__main__':
    main()
