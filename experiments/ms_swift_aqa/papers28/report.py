"""Publish a snapshot of independently verified scores and remaining paper-level work."""
import argparse
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import math
from pathlib import Path
import json
from tempfile import NamedTemporaryFile


def load(path):
    return json.loads(path.read_text())


def replace(path, content):
    with NamedTemporaryFile('w', encoding='utf-8', dir=path.parent, delete=False) as tmp:
        tmp.write(content)
    Path(tmp.name).replace(path)


@lru_cache(maxsize=4096)
def _digest(path, file_identity):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def digest(path):
    stat = path.stat()
    return _digest(path, (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))


def evidence(out, method):
    result = out/'results'/f'{method}.json'
    status = out/f'status-{method}.json'
    state = dict(method=method, status='missing', evidence=str(result), verification=str(status))
    if not status.is_file():
        return state, None
    try:
        marker = status.stat()
        receipt = load(status)
        if receipt['method'] != method:
            raise ValueError('Status belongs to a different experiment')
        stage = receipt['stage']
        if stage != 'verified':
            state.update(status='failed' if stage == 'failed' else 'pending', phase=stage)
            if receipt.get('error'):
                state['reason'] = str(receipt['error'])
            return state, None
        if result.stat().st_mtime_ns > marker.st_mtime_ns:
            raise ValueError('Result was modified after its verification')
        record = load(result)
        sealed = load(out/'configs'/f'{method}-selected.json')
        if record['method'] != method or record['checkpoint'] != sealed:
            raise ValueError('Result and sealed source selection disagree')
        checkpoint = Path(sealed['checkpoint'])
        files = sealed['files']
        if not files or any(digest(checkpoint/key) != sha for key, sha in files.items()):
            raise ValueError('Checkpoint differs from the verified artifact')
        ser, mmau = record['ser'], record['mmau']
        scores = (ser['uar'],ser['accuracy'],mmau['official_text_accuracy'])
        if (ser['utterances'] != 1241 or sum(x['total'] for x in mmau['by_domain'].values()) != 1000
                or not all(isinstance(v,(int,float)) and 0 <= v <= 1 for v in scores)):
            raise ValueError('Invalid two-target result')
        for directory in (Path(record['ser_dir']), Path(record['train_dir'])/'reload-test'):
            met = directory/'metrics.json'
            pred = directory/'predictions.jsonl'
            raw = load(met)
            if max(met.stat().st_mtime_ns,pred.stat().st_mtime_ns) > marker.st_mtime_ns:
                raise ValueError('Target evidence changed after verification')
            if (raw.get('adapter_sha256') or raw.get('identity',{}).get('weights_sha256')) != sealed['adapter_sha256']:
                raise ValueError('Target metrics were produced by a different adapter')
            if raw.get('predictions_sha256') and digest(pred) != raw['predictions_sha256']:
                raise ValueError('Target predictions differ from their receipt')
        # Ensure damaged/missing raw resource receipts cannot leave a stale public row.
        resource_row(method, record)
        latest = status.stat()
        if (latest.st_ino, latest.st_mtime_ns) != (marker.st_ino, marker.st_mtime_ns):
            raise ValueError('Verification status changed while reading the result')
        state['status'] = 'verified'
        state['training_source'] = 'archived' if record['archived_training'] else 'current_run'
        return state, record
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        state.update(status='invalid', reason=str(exc))
        return state, None


def theory_evidence(out):
    from .expressive import FILES, FNN, REVISION, TFN
    root = out/'theory-01'
    path = root/'result.json'
    state = dict(method='expressive-power', paper=1, source='authors_cpu_examples', evidence=str(path), status='missing')
    if not path.is_file():
        return state
    try:
        receipt = load(path)
        if receipt['author_revision'] != REVISION or receipt['source_hashes'] != FILES:
            raise ValueError('Theory source revision or files differ')
        if any(digest(root/'official'/name) != sha for name, sha in FILES.items()):
            raise ValueError('Theory source files differ from the pinned authors’ code')
        for kind, options in [('fnn',FNN),('tfn',TFN)]:
            run = receipt['runs'][kind]
            if run['command'][2:] != options.split():
                raise ValueError('Theory command differs from the registered experiment')
            log = (root/f'{kind}.log').read_text().splitlines()
            for case, prefix in [('normal','Test loss:'),('shifted','Test loss on shifted test set:')]:
                actual, = [float(line.rsplit(' ',1)[1]) for line in log if line.startswith(prefix)]
                if not math.isfinite(actual) or actual < 0 or run['reported_MSE'][case] != actual:
                    raise ValueError(f'{kind} {case} differs from the recorded MSE')
        state.update(status='verified', author_revision=REVISION)
    except (OSError, KeyError, TypeError, ValueError, AttributeError) as exc:
        state.update(status='invalid', reason=str(exc))
    return state


def frozen_evidence(out):
    path = out/'paired-target-scores.json'
    state = dict(method='frozen-base', source='archived_frozen_predictions', evidence=str(path), status='missing')
    if not path.is_file():
        return state, None
    try:
        from .paired import FROZEN_PREDICTIONS
        paired = load(path)
        scores = paired['scores']['frozen-base']
        if paired['model_revision'] != load(out/'protocol.json')['model_revision']:
            raise ValueError('Frozen reference model revision differs')
        for task, source in [('ser',Path(paired['frozen_ser_run'])/'ser/frozen-base'),
                             ('mmau',Path(paired['frozen_mmau_run']))]:
            expected = FROZEN_PREDICTIONS[task]
            if scores[task+'_predictions_sha256'] != expected or digest(source/'predictions.jsonl') != expected:
                raise ValueError('Frozen prediction stream changed')
            raw = load(source/'metrics.json')
            if task == 'ser' and any(raw[k] != v for k,v in scores['ser'].items()):
                raise ValueError('Frozen SER score differs from its receipt')
            if task == 'mmau' and raw['official_text_accuracy'] != scores['mmau']['official_accuracy']:
                raise ValueError('Frozen official MMAU score differs from its receipt')
        if scores['ser']['utterances'] != 1241 or scores['mmau']['n'] != 1000:
            raise ValueError('Incomplete frozen reference')
        state['status'] = 'verified'
        return state, scores
    except (OSError, KeyError, TypeError, ValueError, AttributeError) as exc:
        state.update(status='invalid', reason=str(exc))
        return state, None


def resource_row(name, record):
    train = Path(record['train_dir'])
    metrics = load(train/'result.json')
    prep = train/'initialization.json'
    init = load(prep) if prep.exists() else {}
    generation = train/'generation.json'
    if generation.exists():
        metrics.update(load(generation))
    mmau = load(train/'reload-test/metrics.json')
    ser = load(Path(record['ser_dir'])/'metrics.json')

    def memory(stage):
        def gib(key):
            return f"{stage[key] / 2**30:.2f}" if stage.get(key) is not None else '—'
        return gib('peak_allocated_bytes') + ' / ' + gib('peak_reserved_bytes')

    def seconds(value):
        return '—' if value is None else f'{value:.1f}'

    extra = []
    for key,label in [('calibration','校准'),('stage','前置训练')]:
        if key+'_seconds' in init:
            stage = {k+'_bytes':init.get(key+'_'+k+'_bytes',init.get(key+'_'+k)) for k in ('peak_allocated','peak_reserved')}
            extra.append(f"{label} {seconds(init[key+'_seconds'])} s，显存 {memory(stage)}")
    if generation.exists():
        extra = ['专家与生成器均计入此行训练时间']
    if record['method'] == 'known-target-qo':
        extra = ['耗时为逐层构造，另使用已训练的源任务全参教师']
    cfg = (load(train/'protocol.json')['config'] if (train/'protocol.json').exists() else {})
    recompute = '开' if cfg.get('training',{}).get('gradient_checkpointing') else '关' if cfg else '—'
    return (f"| {name} | {record['mmau']['trainable']:,} | {recompute} | {record['mmau']['updates']} | "
            f"{seconds(metrics.get('training_seconds'))} | {', '.join(extra) or '—'} | {memory(metrics)} | "
            f"{seconds(ser.get('eval_seconds'))} / {memory(ser)} | {seconds(mmau.get('eval_seconds'))} / {memory(mmau)} |")

HERE=Path(__file__).resolve().parent


def publish(out):
    catalog=load(HERE/'catalog.json')
    if [r['id'] for r in catalog]!=list(range(1,29)) or len({r['method'] for r in catalog})!=28:
        raise ValueError('Expected one row per supplied paper')
    records={}
    resources=[]
    fixed=load(out/'protocol.json')
    lines=['# Qwen2.5-Omni-3B · IEMOCAP 与 MMAU','',
           '生成时间 '+datetime.now(timezone.utc).isoformat(timespec='seconds')+'。','',
           '2048 条 AVQA 训练，256 条 AVQA 开发选模。每个选定 checkpoint 经过干净基座重载，在同一份 IEMOCAP Session 5（1241 条，四类）与 MMAU Test-mini（1000 题）上评估。MMAU 使用固定官方文本评分器；IEMOCAP 同时报告 UAR 与准确率。','',
           '七个语言侧投影覆盖 36 层，常规配置为 rank 8、alpha 16、BF16、有效 batch 16、AdamW、基础学习率 0.0002、种子 42，最多两轮，每 32 次更新验证，连续三次未改善停止。特定方法另有秩、优化器、投影、量化、附加数据阶段或可训练模块配置，见方法配置及来源说明。','',
           '| # | 论文 / 配置 | 本轮状态 | IEMOCAP UAR | IEMOCAP Acc | MMAU 官方 Acc | 最佳步 |',
           '|---:|---|---|---:|---:|---:|---:|']
    _, frozen = frozen_evidence(out)
    if frozen:
        lines.append(f"| 对照 | 冻结基座 | 原始预测复算 | {100*frozen['ser']['uar']:.2f}% | "
                     f"{100*frozen['ser']['accuracy']:.2f}% | {100*frozen['mmau']['official_accuracy']:.2f}% | — |")
    verified = {}
    def row(index,name,method):
        state, j = evidence(out,method)
        if j is not None:
            verified[method] = j
            scores=j['ser']['uar'],j['ser']['accuracy'],j['mmau']['official_text_accuracy']
            records[method]=dict(zip(('uar','ser_accuracy','mmau_official'),scores),best=j['mmau']['best_step'])
            desc='历史训练，本轮复核与补测' if j['archived_training'] else '已复核'
            if method == 'known-target-qo':
                desc='逐层构造诊断，依赖已训练教师'
            step=j['mmau']['best_step']
            lines.append(f'| {index} | {name} | {desc} | '+ ' | '.join(f'{100*x:.2f}%' for x in scores)+f' | {step if step is not None else "固定生成"} |')
            resources.append(resource_row(name, j))
        else:
            stage = state.get('phase', state['status'])
            lines.append(f'| {index} | {name} | {stage} | — | — | — | — |')
    row('对照','普通 LoRA','lora')
    row('对照','普通 LoRA（开启梯度重算）','lora-recompute')
    for paper in catalog:
        if paper['id'] == 1:
            theory = theory_evidence(out)
            desc = paper['scope'] if theory['status'] == 'verified' else theory['status']
            lines.append(f"| 1 | {paper['label']} | {desc} | — | — | — | — |")
            row('1 补充','同源已知教师，逐层 Q/O 构造','known-target-qo')
        else:
            row(paper['id'],paper['label'],paper['method'])
    lines+=['','## 耗时和显存','',
            '训练时间、验证耗时和显存由各臂原始日志记录。常规方法的前置阶段另列，独立进程的模型加载及离线量化耗时未计入常规训练秒数。显存为 allocated / reserved 峰值，单位 GiB。','',
            '| 方法 | 可训练参数 | 梯度重算 | 更新数 | 训练秒 | 前置阶段 | 训练显存 | IEMOCAP 秒 / 显存 | MMAU 秒 / 显存 |',
            '|---|---:|---|---:|---:|---|---:|---:|---:|', *resources, '',
            '[固定协议](protocol.json) · [通过验收的分数](scores.json) · [完整配置与选模记录](configs) · [训练和重载日志](logs) · [MMAU 官方评分](mmau-official) · [IEMOCAP 逐条预测](ser)','','',
            '同一输入协议下，单因素比较仍须检查覆盖层、参数量、学习率分组、额外专家训练、校准和量化。当前数字反映单个训练种子。']
    if frozen is not None and current_replay(out, verified):
        lines.extend(['','[按对话或音频聚类的配对区间](paired-target-scores.json)'])
    if theory['status'] == 'verified':
        lines.extend(['','[第 1 篇作者数值验证](theory-01/result.json)'])
    from .special_costs import stages
    replace(out/'SPECIAL_PROTOCOLS.md', stages(out))
    lines.extend(['','[固定 3B 与现有语料的教师、专家和联邦成本分项](SPECIAL_PROTOCOLS.md)'])
    replace(out/'REPORT.md','\n'.join(lines)+'\n')
    summary = dict(papers_with_both=sum(r['method'] in records for r in catalog),paper_count=28,
                   empirical_paper_count=27,model_revision=fixed['model_revision'],scores=records)
    replace(out/'scores.json',json.dumps(summary,indent=2,ensure_ascii=False)+'\n')


def current_replay(out, verified):
    """A paired report can outlive a later failed re-verification."""
    try:
        scores = load(out/'paired-target-scores.json')['scores']
        return set(scores) == set(verified) | {'frozen-base'} and all(
            scores[m]['selected_weights_sha256'] == rec['checkpoint']['adapter_sha256']
            and scores[m]['ser']['utterances'] == rec['ser']['utterances']
            and scores[m]['ser']['confusion_matrix'] == rec['ser']['confusion_matrix']
            # Python and NumPy can round the mean of the same four recalls differently.
            and math.isclose(scores[m]['ser']['uar'],rec['ser']['uar'],rel_tol=0,
                             abs_tol=2*math.ulp(rec['ser']['uar']))
            and scores[m]['ser']['accuracy'] == rec['ser']['accuracy']
            and scores[m]['mmau']['official_accuracy'] == rec['mmau']['official_text_accuracy']
            for m, rec in verified.items())
    except (OSError, ValueError, KeyError, TypeError):
        return False


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    publish(parser.parse_args().out)
