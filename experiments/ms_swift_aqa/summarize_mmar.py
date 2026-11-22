"""Replay the authors' MMAR scoring command and compare the same test items across checkpoints."""
import argparse
from collections import Counter
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

from experiments.ms_swift_librispeech.compare import load, write
from experiments.ms_swift_librispeech.train import digest
from .evaluate_mmar import DOMAINS, METADATA_SHA256, SCORER_SHA256, jsonl
from .run_tuners import paired_outcomes


NAMES = {'frozen-base': '冻结基座', 'lora': 'LoRA', 'dora': 'DoRA', 'ta-lora': '改进 TA-LoRA',
         'rslora': 'rsLoRA', 'bone': 'BoNE', 'vera': 'VeRA', 'adapter': '瓶颈 Adapter', 'adalora': 'AdaLoRA',
         'fourierft': 'FourierFT', 'boft': 'BOFT', 'llamapro': 'LLaMA-Pro', 'reft': 'ReFT', 'ia3': 'IA³',
         'lora_llm': 'LoRA-LLM', 'full': '全量微调（Thinker 语言侧）'}


def report(out, protocol, ranked, scores, comparisons, accepted_counts):
    best = scores[ranked[0]]['correct']
    leaders = '、'.join(NAMES.get(m, m) for m in ranked if scores[m]['correct'] == best)
    lines = ['# Qwen2.5-Omni-3B 在 MMAR 上的答案准确率', '',
             f"本轮共有 {len(ranked)} 组完整的 1,000 题预测。总体最高为{leaders}，答对 {best} 题。"
             f"冻结基座为 {scores['frozen-base']['accuracy']:.1%}，改进 TA-LoRA 为 {scores['ta-lora']['accuracy']:.1%}。"
             ' [完整 CSV](scores.csv) · [统计与配对区间](verified-scores.json)', '',
             '| 方法 | 正确 / 1,000 | 总分 | 环境声 | 音乐 | 语音 | 声音＋音乐 | 声音＋语音 | 音乐＋语音 | 三者混合 |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for m in ranked:
        score = scores[m]
        groups = ' | '.join(f"{score['modality'][d]['accuracy']:.1%}" for d in DOMAINS)
        lines.append(f"| {NAMES.get(m,m)} | {score['correct']} | {score['accuracy']:.1%} | {groups} |")
    sizes = '、'.join(str(protocol['domains'][d]) for d in DOMAINS)
    for m in ranked:
        result = scores[m]
        choices = result['predicted_letter_counts']
        if len(choices) == 1:
            letter = next(iter(choices))
            lines.extend(['', f"{NAMES.get(m,m)} 在全部 {result['total']:,} 题上都选择了 {letter}，其中标注选项为 {letter} 的有 "
                              f"{result['strict_correct']} 题，官方脚本计为正确的有 {result['correct']} 题。"
                              f"[逐题输出]({m}/predictions.jsonl)"])
    lines.extend(['', f'七个互斥子集按表中顺序分别有 {sizes} 题。总分等于全部正确题数除以 1,000。'
                  '每组的官方日志也给出四类任务及其子类的分数。[官方评分脚本](official/evaluation.py)', '',
                  '## 改进 TA-LoRA 的逐题比较', '',
                  '| 参照 | 差值（百分点） | 95% 配对区间（百分点） | 仅 TA 正确 | 仅参照正确 |',
                  '|---|---:|---:|---:|---:|'])
    for m in ('lora','dora','frozen-base'):
        pair = comparisons['vs_frozen']['ta-lora'] if m=='frozen-base' else comparisons['ta_vs'][m]
        lo, hi = pair['interval_95_pp']
        lines.append(f"| {NAMES[m]} | {pair['delta_pp']:+.2f} | {lo:+.2f} 至 {hi:+.2f} | "
                     f"{pair['only_method_correct']} | {pair['only_reference_correct']} |")
    emotion = 'Emotion and Intention'
    emotion_methods = ('frozen-base', 'lora', 'ta-lora', 'dora')
    emotion_scores = [scores[m]['sub-category'][emotion] for m in emotion_methods]
    emotion_text = '、'.join(f"{NAMES[m]} {s['correct']}/{s['n']}" for m, s in zip(emotion_methods, emotion_scores))
    revision = protocol['official_revision']
    lines.extend(['', f"按原始视频 ID 将 1,000 道题归入 {protocol['source_videos']} 个组。区间来自 10,000 次配对重采样，固定模型权重和随机种子 42。"
                  '一次训练种子提供当前 checkpoint 的比较。[逐题预测与官方输入](ta-lora)', '',
                  f'在与副语言信息有关的官方“情绪与意图”子类中，{emotion_text}。'
                  '这组题考查情绪和意图推理，分母较小。[子类评分日志](ta-lora/official.log)', '',
                  '## 数据、模型与评分', '',
                  '沿用 AVQA 开发集选定的 15 份权重，外加未经微调的同一份基座。模型只接收音频、含原始顺序选项的题干。'
                  '所有组都使用 BF16、评估 batch 2，在每题合法字母中贪心解码一个 token。选中的选项原文写入 '
                  '`answer_prediction`，再运行作者发布的 `evaluation.py`。[协议和哈希](protocol.json) · [模型与数据集](https://github.com/ddlBoJack/MMAR/tree/'+revision+')', '',
                  f'作者的词集合匹配规则在 {sum(n for k,n in accepted_counts.items() if k > 1)} 道题上接受两个或更多原选项。'
                  'CSV 中的 `strict_correct` 另记与标注选项原文完全相同的题数。', '',
                  '元数据中 4 道题包含一个空白选项，保留该选项的原位置。作者另有 MMAR-Rubrics 协议，需要模型生成推理文本，'
                  '并由评审模型按每题的评分细则打分。本次表格采用 MMAR Accuracy。[官方评测说明](https://github.com/ddlBoJack/MMAR/tree/'+revision+'#test-your-model-)', ''])
    if (out/'processor-audit.json').is_file():
        audit = load(out/'processor-audit.json')
        record = audit['longest_prompt']
        lines += [f"实际处理了全部 {audit['count']} 题，最长提示含 {record['prompt_tokens']:,} 个 token，其中音频占 "
                  f"{record['audio_tokens']:,} 个。对应的完整录音为 {record['duration']:g} 秒。[逐题输入长度](encoded-lengths.json)", '']
    (out/'REPORT.md').write_text('\n'.join(lines))


def score_arm(out, method, protocol, gold, evaluator, match):
    folder = out/method
    metrics = load(folder/'metrics.json')
    predictions = folder/'predictions.jsonl'
    if (metrics['method'] != method or metrics['predictions_sha256'] != digest(predictions)
            or metrics['manifest_sha256'] != protocol['manifest_sha256'] or metrics['utterances'] != len(gold)):
        raise ValueError(f'Incomplete or changed predictions for {method}')
    submission = folder/'official-input.jsonl'
    temp = submission.with_suffix('.tmp')
    counts, correct, letter_counts = Counter(), Counter(), Counter()
    hits = []
    strict = 0
    with temp.open('w') as stream:
        for p, row in zip(jsonl(predictions), gold, strict=True):
            i = ord(p['prediction']) - ord('A')
            if (p['id'] != row['id'] or p['domain'] != row['modality']
                    or p['category'] != row['category'] or not 0 <= i < len(row['choices'])):
                raise ValueError(f"Incorrect question, modality or option at {row['id']}")
            answer = row['choices'][i]
            exact = answer == row['answer']
            if (p['answer_text'] != answer or p['is_correct'] != exact or
                    p['reference'] != chr(ord('A') + row['choices'].index(row['answer']))):
                raise ValueError(f"Selected option differs from logged prediction at {row['id']}")
            strict += int(exact)
            letter_counts[p['prediction']] += 1
            hit = match(row['answer'], answer, row['choices'])
            hits.append(hit)
            for group in ('modality', 'category', 'sub-category'):
                key = group, row[group]
                counts[key] += 1
                correct[key] += int(hit)
            print(json.dumps({**row, 'answer_prediction': answer}, ensure_ascii=False), file=stream)
    if strict != metrics['correct'] or len(hits) != 1000:
        raise ValueError(f'Item count or strict scores differ for {method}')
    temp.replace(submission)
    official = subprocess.run([sys.executable, str(evaluator), '--input', str(submission)],
                              capture_output=True, text=True, check=True)
    output = set(official.stdout.splitlines())
    n, total = sum(hits), len(hits)
    if f'Total Accuracy: {100*n/total:.2f}% over {total} samples' not in output:
        raise ValueError(f'Official total disagrees with independent scoring for {method}')
    grouped = {key: {} for key in ('modality', 'category', 'sub-category')}
    for (group, name), denom in counts.items():
        numerator = correct[group, name]
        if f'{name} : {100*numerator/denom:.2f}% over {denom} samples' not in output:
            raise ValueError(f'Official {group} score differs for {method}, {name}')
        grouped[group][name] = dict(correct=numerator, n=denom, accuracy=numerator/denom)
    (folder/'official.log').write_text(official.stdout)
    result = dict(method=method, correct=n, total=total, accuracy=n/total, strict_correct=strict,
                  predicted_letter_counts=dict(sorted(letter_counts.items())),
                  **grouped, predictions_sha256=metrics['predictions_sha256'],
                  official_input_sha256=digest(submission), eval_seconds=metrics['eval_seconds'],
                  peak_allocated_gib=metrics['peak_allocated_bytes']/2**30,
                  peak_reserved_gib=metrics['peak_reserved_bytes']/2**30)
    write(folder/'score.json', result)
    return result, hits


def summarize(out, only=None):
    protocol = load(out/'protocol.json')
    source = Path(protocol['dataset'])/'MMAR-meta.jsonl'
    evaluator = out/'official/evaluation.py'
    if (digest(source) != METADATA_SHA256 or digest(evaluator) != SCORER_SHA256
            or digest(out/'test.jsonl') != protocol['manifest_sha256']):
        raise ValueError('Source metadata, scorer or manifest changed')
    gold = jsonl(source)
    if len(gold) != len({r['id'] for r in gold}) or len(gold) != protocol['items']:
        raise ValueError('Unexpected number of unique MMAR items')
    spec = importlib.util.spec_from_file_location('official_mmar', evaluator)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    if only is not None:
        if only not in protocol['methods']:
            raise ValueError(f'Unknown method {only}')
        result, _ = score_arm(out, only, protocol, gold, evaluator, official.string_match)
        print(f"{only} {result['correct']}/{result['total']}", flush=True)
        return
    scores, hits = {}, {}
    for method in protocol['methods']:
        scores[method], hits[method] = score_arm(out, method, protocol, gold, evaluator, official.string_match)
    ranked = sorted(scores, key=lambda m: -scores[m]['correct'])
    clusters = [r['id'].rsplit('_', 2)[0] for r in gold]
    accepted_counts = Counter(sum(official.string_match(r['answer'], choice, r['choices']) for choice in r['choices']) for r in gold)
    comparisons = {'vs_frozen': {m: paired_outcomes(hits[m], hits['frozen-base'], clusters) for m in ranked if m != 'frozen-base'},
                   'ta_vs': {m: paired_outcomes(hits['ta-lora'], hits[m], clusters) for m in ('lora', 'dora')}}
    write(out/'verified-scores.json', dict(ranking=ranked, scores=scores, pairs=comparisons,
                                         source_clusters=len(set(clusters)), accepted_option_counts=dict(accepted_counts),
                                         official_revision=protocol['official_revision']))
    with (out/'scores.csv').open('w', newline='') as fp:
        fields = ['method', 'official_correct', 'total', 'official_pct', *DOMAINS, 'strict_correct',
                  'eval_seconds', 'peak_allocated_gib', 'peak_reserved_gib']
        writer = csv.DictWriter(fp, fields)
        writer.writeheader()
        for method in ranked:
            r = scores[method]
            writer.writerow(dict(method=method, official_correct=r['correct'], total=r['total'], official_pct=f"{100*r['accuracy']:.2f}",
                                 **{d: f"{100*r['modality'][d]['accuracy']:.2f}" for d in DOMAINS},
                                 **{k: r[k] for k in fields[-4:]}))
    report(out, protocol, ranked, scores, comparisons, accepted_counts)
    print('All official totals, modality, category and subcategory scores verified.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--method')
    args = parser.parse_args()
    summarize(args.out.resolve(), args.method)
