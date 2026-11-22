"""Run one Omni AQA SFT tuner at a time; verify dev selection, reloading and official MMAU scores."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.ms_swift_librispeech.compare import load, write
from experiments.ms_swift_librispeech.train import digest, select
from .evaluate import read_rows
from .run import check_disjoint, pin, verify_score
from .summarize_pilot import load_official, paired_delta, read_jsonl, score_arm
from .tuner_suite import checkpoint_identity


def stamp():
    return datetime.now(timezone.utc).isoformat()


def resolved(plan, base, method, override):
    cfg = deepcopy(base)
    cfg['methods'] = [method]
    if method != 'rslora':
        cfg['method_options'] = {method: plan['method_options'].get(method, {})}
    if method == 'reft':
        default_deps = Path(plan['data']).parents[1]/'structural-deps'/'pyreft-0.1.0'
        cfg['reft_dependency_path'] = os.getenv('AQA_REFT_DEPS', str(default_deps))
    cfg['training'].update(plan['training_overrides'].get(method, {}))
    cfg['training'].update(override)
    if method == 'reft':
        # PyVene's intervention hooks exist only during the original forward.
        cfg['training']['gradient_checkpointing'] = False
    for name in ('learning_rate', 'lr_scheduler_type', 'seed', 'data_seed', 'num_train_epochs', 'eval_steps', 'save_steps'):
        if cfg['training'][name] != base['training'][name]:
            raise ValueError(f'Protocol changed for {name}')
    if cfg['training']['per_device_train_batch_size'] * cfg['training']['gradient_accumulation_steps'] != 16:
        raise ValueError('Preserve 16 training examples per optimizer update')
    return cfg


def seal_selection(arm, cfg, method):
    result = load(arm/'result.json')
    history = [{'step': int(p.parent.name[5:]), 'eval_accuracy': load(p)['accuracy']}
               for p in (arm/'dev').glob('step-*/metrics.json')]
    history.sort(key=lambda x: x['step'])
    last = result['global_step']
    expected = list(range(cfg['training']['eval_steps'], last + 1, cfg['training']['eval_steps']))
    if not expected or expected[-1] != last:
        expected.append(last)
    if [h['step'] for h in history] != expected:
        raise ValueError(f'Missing development evaluations in {arm}')
    step, accuracy, stopped = select(history, cfg['patience'], 'accuracy')
    cp = arm/f'checkpoint-{step}'
    if (result['best_step'] != step or result['best_accuracy'] != accuracy or
            Path(result['best_checkpoint']).resolve(strict=True) != cp or stopped != result['early_stopped']):
        raise ValueError(f'Development selection disagrees with Swift in {arm}')
    return dict(method=method, checkpoint=str(cp), step=step, dev_accuracy=accuracy,
                **checkpoint_identity(cp, method))


class ArmFailure(Exception):
    def __init__(self, phase, name, log, message, oom):
        super().__init__(message)
        self.detail = dict(phase=phase, attempt=name, log=str(log), error=message, cuda_oom=oom)


def command(out, label, method, phase, module, cfgfile, args):
    argv = [sys.executable, '-m', module, '--config', str(cfgfile), '--method', method, *map(str, args)]
    log = out/f'{label}-{phase}.log'
    write(out/'status.json', dict(utc=stamp(), pid=os.getpid(), method=method, phase=phase, log=str(log)))
    with log.open('ab') as stream:
        offset = stream.tell()
        stream.write((json.dumps(argv) + '\n').encode())
        stream.flush()
        process = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT)
    if process.returncode:
        with log.open('rb') as stream:
            stream.seek(max(offset, log.stat().st_size - 35000))
            lines = stream.read().decode(errors='replace').splitlines()
        error = next((s for s in reversed(lines) if s.strip() and not s.startswith('[')), f'exit {process.returncode}')
        oom = any('CUDA out of memory' in s or 'OutOfMemoryError' in s for s in lines)
        raise ArmFailure(phase, label, log, error[-1200:], oom)


def matched_trajectory(result, anchor, cfg, sample_reference):
    if cfg['training']['per_device_train_batch_size'] == 4:
        if len(anchor['input_trajectory']) != 8 or result['input_trajectory'] != anchor['input_trajectory']:
            raise ValueError('First eight training minibatches differ from the archived LoRA arm')
    if result['method'] == 'rslora' and result['init_sha256'] != anchor['init_sha256']:
        raise ValueError('rsLoRA did not start with identical A/B weights to LoRA')
    if result['method'] == 'lora_llm' and result.get('shared_ab_sha256') != anchor['init_sha256']:
        raise ValueError('The language LoRA branch of lora_llm lacks the archived shared initialization')
    samples = result.get('sample_trajectory')
    if samples is not None:
        if len(samples) != 32:
            raise ValueError('Missing first 32 unpadded training examples')
        if not sample_reference.exists() and cfg['training']['per_device_train_batch_size'] != 4:
            raise ValueError('Need a batch-4 sample reference for changed microbatch size')
        pin(sample_reference, samples)


def paired_outcomes(hits, reference, clusters):
    pair = paired_delta(hits, reference, clusters)
    pair['only_method_correct'] = pair.pop('only_ta_correct')
    pair['only_reference_correct'] = pair.pop('only_other_correct')
    return pair


def results_page(out, methods, rows, failures):
    completed = sorted(rows.items(), key=lambda v: -v[1]['exact_accuracy'])
    text = ['# Omni‑3B 微调方法串行实验', '',
            f'已验证 {len(rows)} / {len(methods)} 组。最近更新（UTC）{stamp()}。', '',
            'AVQA 训练 2,048 题，开发 256 题。MMAU Test-mini 1,000 题。开发集严格准确率选模，每 32 步验证，连续 3 次未提升早停。', '',
            '| 方法 | 开发 | 测试严格 | MMAU 官方文本 | 最佳步 / 训练步 | 可训练参数 | 训练峰值 allocated（GiB）| 训练秒 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
    for m, r in completed:
        text.append(f"| {m} | {r['dev_accuracy']:.2%} | {r['exact_accuracy']:.2%} | {r['official_text_accuracy']:.2%} | "
                    f"{r['best_step']} / {r['updates']} | {r['trainable']:,} | {r['training_peak_allocated_gib']:.2f} | {r['train_seconds']:.1f} |")
    if failures:
        text += ['', '## 未通过的组', '']
        for m, f in failures.items():
            text.append(f"- {m}，{f['phase']}，{f['error']}。[日志]({f['log']})")
    pending = [m for m in methods if m not in rows and m not in failures]
    if pending:
        text += ['', '待完成：' + '、'.join(pending) + '。']
    text += ['', 'LoRA、DoRA 和改进 TA-LoRA 来自上一轮同一快照和清单，已逐题复核。每个新方法选模后从磁盘重载，以训练时开发预测逐字节对照，然后评分测试集。训练时间含保存，扣除开发评分。单种子结果；各方法默认训练范围、可训练参数数和重算开销见各组配置与日志。', '']
    tmp = out/'REPORT.md.tmp'
    tmp.write_text('\n'.join(text))
    tmp.replace(out/'REPORT.md')


def pin_plan(out, plan, extend=False):
    """Allow an explicitly appended arm while preserving the original protocol and plan."""
    path = out/'plan.json'
    if not extend or not path.is_file() or load(path) == plan:
        return pin(path, plan)
    previous = load(path)
    methods = previous['methods']
    if plan['methods'][:len(methods)] != methods or len(plan['methods']) <= len(methods):
        raise ValueError('Only new methods may be appended to a frozen queue')
    additions = set(plan['methods'][len(methods):])
    original = {**plan, 'methods': methods}
    for field in ('method_options', 'training_overrides'):
        original[field] = {k: v for k, v in plan[field].items() if k not in additions}
    if original != previous:
        raise ValueError('An existing method or shared protocol changed while extending the queue')
    archive = out/'plans'
    archive.mkdir(exist_ok=True)
    pin(archive/f'{digest(path)}.json', previous)
    write(path, plan)


def run(plan_path, out, only, retry, extend=False):
    plan = load(plan_path)
    base = load(plan_path.parent/plan['base_config'])
    ref, data = Path(plan['reference_run']), Path(plan['data'])
    if load(ref/'config.json') != base:
        raise ValueError('Reference run configuration differs from the base protocol')
    methods = plan['reference_methods'] + plan['methods']
    if set(only) - set(plan['methods']) or len(methods) != len(set(methods)):
        raise ValueError('Unexpected or duplicate tuners')
    for directory in ('configs', 'results', 'failures', 'reference-checks'):
        (out/directory).mkdir(exist_ok=True)
    pin_plan(out, plan, extend)
    pin(out/'base.json', base)
    paths = {s: data/f'{s}.jsonl' for s in ('train', 'dev', 'test')}
    hashes = {s: digest(p) for s, p in paths.items()}
    audit = load(data/'audit.json')['splits']
    if any(audit[s]['sha256'] != v for s, v in hashes.items()):
        raise ValueError('Frozen data changed')
    ref_dev = load(ref/'training-inputs.json')
    if (hashes['train'] != ref_dev['train_sha256'] or hashes['dev'] != ref_dev['dev_sha256'] or
            hashes['test'] != load(ref/'final-inputs.json')['test_sha256']):
        raise ValueError('New and archived splits differ')
    pin(out/'manifests.json', hashes)
    dev = read_rows(paths['dev'])
    train = read_rows(paths['train'])
    test, source, scorer, official = load_official(data, Path(plan['mmau']))
    check_disjoint(train, dev)
    check_disjoint(train + dev, test)
    audio = load(data/'audio-hashes.json')
    clusters = [audio[r['audios'][0]] for r in test]
    anchor = load(ref/'lora/result.json')
    baseline = [r['is_correct'] for r in read_jsonl(ref/'lora/reload-test/predictions.jsonl')]
    ta = [r['is_correct'] for r in read_jsonl(ref/'ta-lora/reload-test/predictions.jsonl')]
    rows = {p.stem: load(p) for p in (out/'results').glob('*.json')}
    failures = {p.stem: load(p) for p in (out/'failures').glob('*.json')}
    archived_seals = load(ref/'selection.json')

    def publish(method, arm, selected, verified, official_dir=None):
        row, hits, result = score_arm(arm, method, test, source, scorer, official, official_dir)
        row.update(checkpoint=selected['checkpoint'], method=method, artifact=str(arm),
                   peak_reserved_gib=result['peak_reserved_bytes']/2**30, dev_seconds=result['dev_seconds'],
                   test_seconds=verified['eval_seconds'], test_peak_allocated_gib=verified['peak_allocated_bytes']/2**30,
                   test_peak_reserved_gib=verified['peak_reserved_bytes']/2**30,
                   vs_lora=paired_outcomes(hits, baseline, clusters), vs_ta=paired_outcomes(hits, ta, clusters))
        pin(out/'results'/f'{method}.json', row)
        rows[method] = row
        failures.pop(method, None)
        (out/'failures'/f'{method}.json').unlink(missing_ok=True)
        results_page(out, methods, rows, failures)

    for method in plan['reference_methods']:
        if method in rows:
            continue
        arm = ref/method
        selected = seal_selection(arm, base, method)
        if any(selected[k] != v for k, v in archived_seals[method].items()):
            raise ValueError(f'Archived {method} checkpoint changed')
        verify_score(arm/'reload-dev', dev, paths['dev'], selected)
        if digest(arm/'reload-dev/predictions.jsonl') != digest(arm/'dev'/f"step-{selected['step']}"/'hypotheses.jsonl'):
            raise ValueError(f'Invalid archived {method} development reload')
        verified = verify_score(arm/'reload-test', test, paths['test'], selected)
        publish(method, arm, selected, verified, out/'reference-checks'/method)

    for method in plan['methods']:
        if (only and method not in only) or method in rows or (method in failures and not retry):
            continue
        attempt = 0
        while True:
            label = method if not attempt else f'{method}-{attempt+1}'
            arm = out/label
            cfgfile = out/'configs'/f'{label}.json'
            override = {} if attempt == 0 else plan['on_training_cuda_oom'][attempt-1]
            cfg = resolved(plan, base, method, override)
            pin(cfgfile, cfg)
            try:
                if not (arm/'result.json').is_file():
                    if arm.exists() and any(arm.iterdir()):
                        # Preserve unsuccessful/aborted training artifacts when retrying a repaired implementation.
                        prior = out/'attempts'
                        prior.mkdir(exist_ok=True)
                        arm.rename(prior/f"{label}-{stamp().replace(':','-')}")
                    command(out, label, method, 'train', 'experiments.ms_swift_librispeech.train', cfgfile,
                            ['--train', paths['train'], '--dev', paths['dev'], '--output', arm])
                report = load(arm/'result.json')
                matched_trajectory(report, anchor, cfg, out/'sample-trajectory.json')
                selected = seal_selection(arm, cfg, method)
                pin(arm/'selection.json', selected)
                verified = {}
                for split, dataset in [('dev', dev), ('test', test)]:
                    phase = f'reload-{split}'
                    folder = arm/phase
                    if not (folder/'metrics.json').is_file():
                        command(out, label, method, phase, 'experiments.ms_swift_aqa.evaluate', cfgfile,
                                ['--checkpoint', selected['checkpoint'], '--manifest', paths[split], '--output', folder])
                    verified = verify_score(folder, dataset, paths[split], selected)
                    if verified['config_sha256'] != selected['config_sha256']:
                        raise ValueError(f'Changed selected model config for {method}')
                    if split == 'dev':
                        original = arm/'dev'/f"step-{selected['step']}"/'hypotheses.jsonl'
                        if (digest(original) != digest(folder/'predictions.jsonl') or
                                verified['accuracy'] != selected['dev_accuracy']):
                            raise ValueError('Selected checkpoint failed to reproduce in-training dev predictions')
                publish(method, arm, selected, verified)
                break
            except ArmFailure as exc:
                detail = exc.detail
                if detail['cuda_oom'] and detail['phase'] == 'train' and attempt < len(plan['on_training_cuda_oom']):
                    write(out/f'{label}-oom.json', detail)
                    attempt += 1
                    continue
            except (OSError, ValueError, KeyError, RuntimeError) as exc:
                detail = dict(phase='verification', attempt=label, log=str(arm), error=str(exc))
            failures[method] = detail
            write(out/'failures'/f'{method}.json', detail)
            results_page(out, methods, rows, failures)
            break
    results_page(out, methods, rows, failures)
    write(out/'status.json', dict(utc=stamp(), phase='finished' if len(rows)+len(failures)==len(methods) else 'partial',
                                  verified=list(rows), failed=list(failures),
                                  pending=[m for m in methods if m not in rows and m not in failures]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=Path(__file__).with_name('omni3b-tuners.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--only', nargs='+', default=[])
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--extend-plan', action='store_true', help='Append methods to an otherwise identical frozen plan')
    a = parser.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    with (a.output/'gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        run(a.plan, a.output.resolve(), a.only, a.retry_failed, a.extend_plan)
