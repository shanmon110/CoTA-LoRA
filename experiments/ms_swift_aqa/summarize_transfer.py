"""Verify the registered TA transport study on AVQA and two external evaluation sets."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

from .summarize_pilot import paired_delta, read_json, read_jsonl

DEV_ARMS = ('frozen-base', 'old-ta', '42-lora', '42-ta-uniform', '42-ta-overlap',
            '42-ta-overlap-off', '42-dora', '42-lora-r12', '43-lora','43-lora-r12', '43-ta-uniform', '43-ta-overlap',
            '43-ta-overlap-off')
MMAR_ARMS = tuple(arm for arm in DEV_ARMS if not arm.startswith('43-'))
TRAINED = tuple(m for m in DEV_ARMS[2:] if not m.endswith('-off'))
SCORER_SHA = 'a3c57b829e40e67e3f7f0bbe7d54a112ea06ed7a781534241ecc5d3751486198'
META_SHA = '1c9a343e7bebf1037482b935c54eacd43cc63df7b2c668724b5560caf1f84fad'


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    with path.open('rb') as fp:
        return hashlib.file_digest(fp, 'sha256').hexdigest()


def verify_training(out, protocol, train_path, avqa_dev):
    training = {}
    capacity = read_json(out/'capacity-protocol.json')['trainable']
    for arm in TRAINED:
        path = out.parent/'o-only' if arm == '42-ta-uniform' else out/arm
        record = read_json(path/'result.json')
        identity = read_json(path/'initialization.json')
        saved = read_json(path/'protocol.json')
        cfg = saved['config']
        seed = int(arm[:2])
        require((saved['train_sha256'], saved['dev_sha256']) == (digest(train_path), digest(avqa_dev)),
                f'{arm}: AVQA dataset changed')
        require(cfg['model_revision'] == protocol['backbone'] and cfg['training']['seed'] == seed
                and cfg['training']['data_seed'] == seed and cfg['adapter']['target_modules'] == ['o_proj'],
                f'{arm}: backbone, seed or projection mismatch')
        rank = 12 if arm.endswith('-r12') else 8
        require(cfg['adapter']['rank'] == rank and cfg['adapter']['alpha'] == 2*rank, f'{arm}: wrong rank or scaling')
        require(record['init_sha256'] == identity['ab_sha256'] and record['trainable_parameters'] == identity['trainable_parameters']
                and identity['target_layers'] == list(range(36)), f'{arm}: initial weights or dimensions mismatch')
        if arm.endswith('-r12') or '-ta-' in arm:
            budget = capacity['lora_rank12' if arm.endswith('-r12') else 'ta_rank8']
            require(record['trainable_parameters'] == budget, f'{arm}: registered parameter count differs')
        evaluations = []
        for step in (path/'dev').glob('step-*'):
            scores = read_json(step/'metrics.json')
            require(scores['utterances'] == 256, f'{arm}: unexpected AVQA coverage')
            evaluations.append((int(step.name.split('-')[-1]), scores['correct']/256))
        best_step, best_score = min(evaluations, key=lambda pair: (-pair[1], pair[0]))
        best = Path(record['best_checkpoint'])
        require(best_step == record['best_step'] and best_score == record['best_accuracy'] and best.name == f'checkpoint-{best_step}',
                f'{arm}: checkpoint was not the earliest best on AVQA')
        require(record['input_trajectory'] and max(s for s, _ in evaluations) == record['global_step'],
                f'{arm}: incomplete input or validation record')
        training[arm] = dict(seed=seed, checkpoint=str(best.resolve()), best_step=best_step, avqa_accuracy=best_score,
                             steps=record['global_step'], validation_curve=sorted(evaluations),
                             trainable=record['trainable_parameters'], train_seconds=record['training_seconds'],
                             training_peak_gib=record['peak_allocated_bytes']/2**30,
                             init_ab=identity['ab_sha256'], prefix_batch_hashes=record['input_trajectory'],
                             weights_sha256=digest(best/'adapter_model.safetensors'),
                             config_sha256=digest(best/'adapter_config.json'), train_options=cfg['training'])
    for seed in (42, 43):
        ref = training[f'{seed}-lora']
        for arm, record in training.items():
            if record['seed'] == seed:
                keys = ('prefix_batch_hashes','train_options') if arm.endswith('-r12') else ('init_ab','prefix_batch_hashes','train_options')
                for key in keys:
                    require(record[key] == ref[key], f'{arm}: {key} differs from same-seed LoRA')
    return training


def evaluate_arm(folder, manifest, rows, mmar=None):
    met = read_json(folder/'metrics.json')
    pred_path = folder/'predictions.jsonl'
    preds = read_jsonl(pred_path)
    n = len(rows)
    require(n == len(preds) == met['utterances'] and met['manifest_sha256'] == digest(manifest)
            and met['predictions_sha256'] == digest(pred_path) and met['method'] == folder.name,
            f'{folder}: changed evaluation inputs or outputs')
    answers, hits, strict = [], [], []
    by_domain = {}
    for i, (row, pred) in enumerate(zip(rows, preds, strict=True)):
        letter = pred['prediction']
        require(row['id'] == pred['id'] and len(letter) == 1 and 'A' <= letter < chr(65+len(row['choices'])),
                f'{folder}: wrong or invalid prediction at {i}')
        reference = row['messages'][1]['content']
        text = row['choices'][ord(letter)-65]
        exact = text == row['choices'][ord(reference)-65]
        require(pred['reference'] == reference and pred['answer_text'] == text and pred['is_correct'] == exact,
                f'{folder}: stored answer changed at {i}')
        strict.append(exact)
        answers.append(text)
        hit = exact
        if mmar:
            gold = mmar['meta'][i]
            require(gold['id'] == row['id'] and gold['choices'] == row['choices'] and gold['answer'] == row['choices'][ord(reference)-65],
                    f'{folder}: original MMAR record mismatch at {i}')
            hit = mmar['judge'](gold['answer'], text, gold['choices'])
        hits.append(hit)
        domain = row['domain']
        by_domain.setdefault(domain, []).append(hit)
    require(sum(strict) == met['correct'], f'{folder}: strict scores disagree')
    result = dict(correct=sum(hits), n=n, accuracy=sum(hits)/n, domain={k:dict(n=len(v), correct=sum(v)) for k,v in by_domain.items()},
                  strict_correct=sum(strict), predictions_sha256=digest(pred_path), manifest_sha256=digest(manifest),
                  eval_seconds=met['eval_seconds'], peak_gib=met['peak_allocated_bytes']/2**30, identity=met['identity'])
    if mmar:
        submission = folder/'official-input.jsonl'
        saved = read_json(folder/'score.json')
        for source, pred, entry in zip(mmar['meta'], answers, read_jsonl(submission), strict=True):
            require(entry == {**source, 'answer_prediction': pred}, f'{folder}: official submission changed')
        proc = subprocess.run([sys.executable, str(mmar['script']), '--input', str(submission)],
                              capture_output=True, text=True, check=True)
        expected = f"Total Accuracy: {result['accuracy']*100:.2f}% over {n} samples"
        require(expected in proc.stdout.splitlines() and proc.stdout == (folder/'official.log').read_text()
                and saved['correct'] == result['correct'] and saved['total'] == n
                and saved['official_input_sha256'] == digest(submission), f'{folder}: official CLI discrepancy')
        result['official_input_sha256'] = digest(submission)
    return result, hits


def pair_results(hits, clusters, seeds):
    output = {}
    for seed in seeds:
        primary = f'{seed}-ta-overlap'
        controls = [f'{seed}-lora', f'{seed}-lora-r12', f'{primary}-off', f'{seed}-ta-uniform', 'old-ta', 'frozen-base']
        if f'{seed}-dora' in hits:
            controls.append(f'{seed}-dora')
        output[primary] = {m: paired_delta(hits[primary], hits[m], clusters) for m in controls}
    return output


def markdown(results, training, paired):
    lines = ['# Qwen2.5-Omni-3B 的 TA-LoRA 配对实验', '',
             '训练数据固定为原 AVQA 的 2,048 道题，仅使用其 256 题验证集选择每个方法最早达到最高准确率的检查点。'
             '新训练统一在 36 层语言解码器的 O 投影上使用缩放 2。主对照使用 rank 8、alpha 16，容量对照 LoRA 使用 rank 12、alpha 24。'
             '同种子、同 rank 核验基础低秩矩阵初值，同种子跨方法核验记录到的前几个输入批次及训练配置。'
             '不同方法的实际可训练参数量和早停步数见下表。[注册协议](protocol.json) · [容量对照协议](capacity-protocol.json)', '',
             '| 方法 | AVQA 最佳 | 最佳步 / 实际步 | 可训练参数 | 训练秒 | 训练峰值 GiB |',
             '|---|---:|---:|---:|---:|---:|']
    for name, r in training.items():
        lines.append(f"| {name} | {r['avqa_accuracy']:.2%} | {r['best_step']} / {r['steps']} | {r['trainable']:,} | {r['train_seconds']:.0f} | {r['training_peak_gib']:.2f} |")
    lines.extend(['', '## 外部开发集与 MMAR', '',
                  '开发集涵盖环境声问答、音乐问答及情绪分类，比较三类任务的准确率。MMAR 以冻结的官方脚本计算答案准确率。'
                  '全部组使用音频、原题及选项，限制生成一个合法选项字母。旧 TA 是原来七投影的检查点。'
                  '带 off 的行从相同检查点加载后在推理时将门控和独立分支置零。', '',
                  '| 方法 | 开发集正确 / 总数 | 环境声 | 音乐 | 语音 | MMAR 正确 / 1,000 | MMAR 准确率 |',
                  '|---|---:|---:|---:|---:|---:|---:|'])
    for arm in DEV_ARMS:
        d=results['dev'][arm]
        cells=[f"{d['domain'][k]['correct']}/{d['domain'][k]['n']}" for k in ('sound','music','speech')]
        m=results['mmar'].get(arm)
        end = f"{m['correct']} | {m['accuracy']:.1%}" if m else '— | —'
        lines.append(f"| {arm} | {d['correct']}/{d['n']} | {' | '.join(cells)} | {end} |")
    lines.extend(['', '## 逐题配对', '',
                  '以新 overlap 方法减去参照方法。按源录音或原视频分组重采样 10,000 次，给出固定检查点的 95% 百分位区间。'
                  '区间反映题目抽样变异。两枚种子的训练变异另行比较。', '',
                  '| 数据 | 种子 | 参照 | 差值（百分点） | 95% 区间 | 仅新方法对 | 仅参照对 |',
                  '|---|---:|---|---:|---:|---:|---:|'])
    for split, comparisons in paired.items():
        for method, controls in comparisons.items():
            for ctrl, v in controls.items():
                lo, hi = v['interval_95_pp']
                lines.append(f"| {split} | {method[:2]} | {ctrl} | {v['delta_pp']:+.2f} | {lo:+.2f} 至 {hi:+.2f} | {v['only_ta_correct']} | {v['only_other_correct']} |")
    m=results['mmar']['42-ta-overlap']
    primary = paired['mmar']['42-ta-overlap']
    signs = [paired['dev'][f'{s}-ta-overlap'][f'{s}-lora']['delta_pp'] for s in (42,43)]
    lines.extend(['', f"外部开发集上，overlap 相对同种子 rank-8 LoRA 的差距依次为 {signs[0]:+.2f}、{signs[1]:+.2f} 个百分点。"
                  f"种子 42 的 MMAR 差距为 {primary['42-lora']['delta_pp']:+.2f} 个百分点，"
                  f"相对 rank-12 容量对照为 {primary['42-lora-r12']['delta_pp']:+.2f} 个百分点。"
                  f"与相同权重关闭传输分支相比，开放分支改变了 {primary['42-ta-overlap-off']['delta_pp']:+.2f} 个百分点。"])
    lines.extend(['', f"预注册的种子 42 overlap 检查点在 MMAR 上答对 {m['correct']} 题，准确率为 {m['accuracy']:.1%}。"
                  '基准在上一轮已经公开评测，此结果适合回答固定协议下的方法差异。'
                  '公开的 StepAudio 3 Realtime 报告主系统 86.5%，其中专训 Teacher 3 报告 87.3%。'
                  f"本次检查点与 87.3% 的差距为 {m['accuracy']*100-87.3:+.1f} 个百分点。"
                  '其架构与推理协议均与本实验不同。[公开参照](https://arxiv.org/html/2609.14005v3#S6.T8)', '',
                  '开发集的语音部分来自 IEMOCAP Session 4。语音区间按录音样本分组，相同说话者和对话仍可能相关。'
                  'MMAR 仅运行预注册的种子 42。每组预测、官方评分输入与命令行原始输出保存在各组目录。'
                  '[验证记录与哈希](verified-transfer.json)'])
    return '\n'.join(lines)+'\n'


def summarize(out, dev, previous):
    protocol = read_json(out/'protocol.json')
    train = Path(protocol['train'])
    avqa_dev = train.with_name('dev.jsonl')
    training = verify_training(out, protocol, train, avqa_dev)
    original = read_json(previous/'protocol.json')
    require(original['model_revision'] == protocol['backbone'], 'Historical base revision changed')
    meta_path = Path(original['dataset'])/'MMAR-meta.jsonl'
    script = previous/'official/evaluation.py'
    require(digest(meta_path) == META_SHA and digest(script) == SCORER_SHA
            and digest(previous/'test.jsonl') == original['manifest_sha256'], 'Unrecognized MMAR release')
    spec = importlib.util.spec_from_file_location('mmar_score', script)
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    official = dict(meta=read_jsonl(meta_path), script=script, judge=scorer.string_match)
    dev_audit = read_json(dev.parent/'dev-audit.json')['splits']['dev']
    require(digest(dev) == dev_audit['sha256'], 'External development manifest changed')
    results, paired = {}, {}
    for split, arms, manifest in [('dev', DEV_ARMS, dev), ('mmar', MMAR_ARMS, previous/'test.jsonl')]:
        rows = read_jsonl(manifest)
        require(len(rows) == len({r['id'] for r in rows}), f'{split}: repeated question IDs')
        if split == 'dev':
            require(len(rows) == dev_audit['n'] and Counter(r['domain'] for r in rows) == dev_audit['domains'],
                    'Development coverage discrepancy')
        else:
            require(len(rows) == 1000, 'Incomplete MMAR')
        clusters = [r['source_id'] for r in rows]
        split_results, all_hits = {}, {}
        for arm in arms:
            folder = out/split/arm
            historical = arm in ('frozen-base', 'old-ta') and split == 'mmar'
            if historical:
                folder = previous/('ta-lora' if arm == 'old-ta' else arm)
            r, hits = evaluate_arm(folder, manifest, rows, official if split == 'mmar' else None)
            identity = r['identity']
            require(identity['model_revision'] == protocol['backbone'], f'{arm}: base changed')
            norm = arm.removesuffix('-off')
            if norm in training:
                record = training[norm]
                require(identity['weights_sha256'] == record['weights_sha256']
                        and identity['adapter_config_sha256'] == record['config_sha256']
                        and identity['checkpoint'] == record['checkpoint'], f'{split}/{arm}: checkpoint mismatch')
                if arm.endswith('-off'):
                    ab = identity.get('transport_ablation', {})
                    require(ab.get('gate_tensors') == ab.get('transport_up_tensors') == 36,
                            f'{split}/{arm}: missing transport ablation')
                else:
                    require('transport_ablation' not in identity, f'{split}/{arm}: unrequested ablation')
            split_results[arm] = r
            all_hits[arm] = hits
        results[split] = split_results
        paired[split] = pair_results(all_hits, clusters, (42,43) if split == 'dev' else (42,))
    check_old = results['dev']['old-ta']['identity']['weights_sha256']
    require(check_old == results['mmar']['old-ta']['identity']['weights_sha256'], 'Legacy checkpoint changed')
    state=dict(scores=results, contrasts=paired, training=training,
               hashes=dict(protocol=digest(out/'protocol.json'), capacity_protocol=digest(out/'capacity-protocol.json'),
                           source_protocol=digest(previous/'protocol.json'),
                           scorer=SCORER_SHA, metadata=META_SHA, code=digest(Path(__file__)),
                           avqa_train=digest(train), avqa_dev=digest(avqa_dev), external_dev=digest(dev)))
    (out/'verified-transfer.json').write_text(json.dumps(state, indent=2, ensure_ascii=False)+'\n')
    (out/'REPORT.md').write_text(markdown(results, training, paired))
    print(f"Verified {len(training)} trained checkpoints, {len(results['dev'])} development arms and {len(results['mmar'])} MMAR arms.")


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--out', type=Path, required=True)
    cli.add_argument('--dev-manifest', type=Path)
    cli.add_argument('--mmar-run', type=Path)
    args = cli.parse_args()
    folder = args.out.resolve()
    summarize(folder, args.dev_manifest or folder.parents[1]/'ta-diagnosis-01/multidomain/dev.jsonl',
              args.mmar_run or folder.parents[1]/'omni3b-avqa-01')
