# 可公开实验与回归证据

此目录是 2026-11-22 整理的只读证据子集，供 Claude 与研究者核对现有结论。包含 694 份实验文件：642 份原件逐字节副本、52 份标明转换规则的聚合派生文件，共 6,602,920 bytes；不包含本说明、清单及两个管理脚本的大小。没有重新训练或重新评测模型。模型名 `ta-lora` / `question_audio` 在历史文件中对应现称 **CoTA-LoRA** 的实验臂。

**当前没有 CoTA 稳定优于匹配 LoRA 的证据。** 28 篇方法表属于另一套协议，不应将其中较好方法的成绩算作 CoTA 的贡献。`complete` 表示当时的工程验收结束，并不等于完整复现了所有论文条件或证明了方法有效性。

## 建议阅读顺序

| 内容 | 本仓库中的证据入口 |
|---|---|
| CoTA 主结果、完整聚合复核 | [原始报告](raw/ta-conditional-01/REPORT.md)、[完整 analysis.json](raw/ta-conditional-01/analysis.json)、[分块 scores](derived/ta-conditional-01/analysis/scores.json)、[训练与选模轨迹](derived/ta-conditional-01/analysis/training.json) |
| CoTA 迁移、配对区间 | [迁移报告](raw/ta-conditional-transfer-01/REPORT.md)、[IEMOCAP 汇总](raw/ta-conditional-transfer-01/ser-verified.json)、[MMAU 汇总](raw/ta-conditional-transfer-01/mmau_scores.json)、[checkpoint 与协议哈希](raw/ta-conditional-transfer-01/protocol.json) |
| 28 篇方法、终态、特殊成本 | [28 篇报告](raw/papers28-transfer-01/REPORT.md)、[scores](raw/papers28-transfer-01/scores.json)、[results/](raw/papers28-transfer-01/results/)、[disposition](raw/papers28-transfer-01/disposition.json)、[特殊协议](raw/papers28-transfer-01/SPECIAL_PROTOCOLS.md) |
| 28 篇的独立配对复算 | [paired-target-scores](raw/papers28-transfer-01/paired-target-scores.json)；包含分数、混淆矩阵、来源哈希与聚类区间，无逐题预测 |
| 门控探针与实际优化器 LR | [原短版聚合探针](derived/ta-conditional-01/gate-probe.json)、[高门 LR 聚合探针](derived/ta-conditional-01/gate-probe-fastgate.json)、[优化器审计原件](raw/ta-conditional-01/fastgate-optimizer-audit.json) |
| 普通 LoRA 原始训练曲线 | [logging.jsonl](raw/ordinary-lora-pilot/logging.jsonl)、[开发点摘要](derived/ordinary-lora-pilot/training-curve-summary.json)、[选模与资源结果](raw/ordinary-lora-pilot/result.json)、[协议](raw/ordinary-lora-pilot/protocol.json) |
| 两个混合主臂的原始训练曲线 | [匹配 LoRA](raw/ta-conditional-01/train/lora-mixed/logging.jsonl)、[CoTA](raw/ta-conditional-01/train/question_audio-mixed/logging.jsonl)；其他论文臂见 `raw/papers28-transfer-01/train/*/logging.jsonl` |
| 历史 CPU 总回归与最终复核 | [186 passed / 4 skipped](raw/papers28-transfer-01/logs/final-replay-regression.log)、[最终结果复核](raw/papers28-transfer-01/logs/final-replay-audit.log)、[CoTA CPU 回归](raw/ta-conditional-01/logs/cpu-final.log) |
| 辅助环境与数据交集审计 | [环境版本](raw/papers28-transfer-01/sources/83075b150b88cbe603e29d7aee7aeab976fc280b9482606f906d69eb80014b5a/environment.json)、[规范化音频交集](raw/papers28-transfer-01/split-audio-audit.json)、[SER 聚合审计](derived/ta-conditional-transfer-01/ser-audit-summary.json)、[processor 聚合审计](derived/ta-conditional-transfer-01/ser-processor-audit-summary.json) |

原始报告保持原件字节，因此其部分链接仍指向本机路径或有意排除的文件。请用本页导航查找已公开的对应材料；原报告的链接并不表示相关数据或 checkpoint 已公开。

## 最新 CoTA 与匹配 LoRA

两者均使用混合训练集和变化配对选模；均选择第 160 步，训练到第 168 步。`off` 是同一个 CoTA checkpoint 在推理时完整关闭传输，并非另一个训练基线。

| 指标 | 匹配 LoRA | CoTA | 同一 CoTA 关闭传输 | 冻结基座 |
|---|---:|---:|---:|---:|
| 长变化配对，两条均正确 /96 | 70 | 73 | 65 | 5 |
| 混合变化配对，两条均正确 /96 | 68 | 68 | 62 | — |
| 真实代理开发正确 /261 | 202 | 201 | 202 | 203 |
| MMAR 官方准确率 | 61.0% | 60.5% | 60.6% | 61.9% |
| IEMOCAP UAR | 63.74% | 62.96% | 62.61% | 59.94% |
| IEMOCAP 准确率 | 62.13% | 60.52% | 60.35% | 57.21% |
| MMAU 官方准确率 | 72.5% | 72.1% | 72.5% | 75.7% |

表中配对/MMAR/代理开发数值来自 `analysis.json`，迁移数值来自 `ser-verified.json` 与 `mmau_scores.json`。长变化配对的 CoTA 开关差为 +8.33 pp，按来源配对的 95% 区间 [+2.08,+14.58]；相对独立训练的匹配 LoRA 为 +3.12 pp，区间 [-3.12,+9.38]。这是传输分支在受控配对上的局部观察，尚未建立真实任务的稳定收益。

门控探针来自历史**短版** checkpoint，不能直接代表混合版。24 条输入上，变化配对的门乘数平均绝对变化约 0.000187；门 LR 提高至 0.0032 后约 0.002915，同时保持配对也从约 0.000103 增至 0.001737。应同时考察选择性与任务收益。

## 两套实验协议必须分开解释

| 项目 | CoTA 主实验 | 28 篇方法实验 |
|---|---|---|
| 训练数据 | AVQA 2048 + 320 个配对的 640 条，共 2688 条 | AVQA 2048 条 |
| 开发选模 | 变化配对的“两条都正确”率；平分取最早 | AVQA 256 条 accuracy；平分取最早 |
| 常规覆盖 | 36 层 O 投影 | 36 层 Q/K/V/O 与三种 MLP 投影；特殊方法另列 |
| 对照 LoRA | rank 12 / alpha 24，1,769,472 参数 | rank 8 / alpha 16，14,966,784 参数 |
| CoTA | rank 8 / alpha 16，1,772,064 参数 | 不属于该表协议 |
| 有效 batch | 4 × 累积 4 = 16 | 4 × 累积 4 = 16 |
| 训练量上限 | 1 轮 / 168 更新 | 2 轮 / 256 更新 |
| 常规优化 | AdamW、恒定 2e-4、无 warmup、weight decay 0 | 相同基础设置，方法特有 LR 分组或阶段另列 |
| 验证与早停 | 每 32 更新及末步，patience 3 | 每 32 更新及末步，patience 3 |
| 梯度检查点 | false | 常规新臂开启，历史与特殊臂有差异 |
| 种子 | 42 | 42 |

所有结果均为现有固定种子 checkpoint。聚类 bootstrap 区间反映测试样本变异，不等于多训练种子不确定性，亦未作多重比较校正。IEMOCAP 为 Session 5 的 1241 条四分类样本；MMAU Test-mini 为 1000 条。普通 LoRA、CoTA 均未在 IEMOCAP 情感训练集上训练。

28 篇中的 LongLoRA 是 Omni 原生上下文内的 shifted attention 移植；LoRAM-TrainSmall 使用无额外对齐预训练消融；RoLoRA 是单卡模拟四客户端，未实测网络通信；LoRAGen 使用四个源专家的小库；第 1 篇包含作者理论示例与同源教师构造诊断。PiSSA、LoRA-GA、LowRA 的最佳 checkpoint 出现全选 A，LoRA+、EVA 也显著退化；不能将这些成绩解释为原论文方法本身较弱。详情及成本见 `SPECIAL_PROTOCOLS.md`、各方法协议与训练曲线。

## 普通 LoRA 训练轨迹

| 更新步 | 32 | 64 | 96 | 128 | 160 | 192 |
|---|---:|---:|---:|---:|---:|---:|
| 最近 32 步平均 loss | 0.2037 | 0.1843 | 0.1703 | 0.1466 | 0.0837 | 0.0728 |
| AVQA dev 正确数 /256 | 205 | 205 | 206 | 203 | 206 | 205 |

最早最佳点为第 96 步，192 步因连续三次未改善而早停。已有日志没有开发交叉熵，也没有所有训练步对应的目标测试分数，因此不能据此定位目标退化起点或断言严重过拟合。`logging.jsonl` 中的 `token_acc` 还包含结束标记，不能等同于答案准确率。

## 已观察测试集与结论边界

**MMAR、MMAU、IEMOCAP 已在多轮研究决策中被观察，当前结果属于探索性研究。** 本目录保留了 MMAR 分数触发后续阶段的 [phase-two-trigger](raw/ta-conditional-01/phase-two-trigger.json) 和 [phase-three-trigger](raw/ta-conditional-01/phase-three-trigger.json)。今后的实验需预先确定开发、选择和最终检验安排，不能把这些目标集称为此前未观察的最终测试。

音频审计的“无交集”仅针对已经检查的规范化 PCM 完全一致情形，不涵盖近似重复、重编码相似或预训练污染。权重、预测、原始数据没有随本包发布，因此这里只能审阅聚合证据及哈希；重跑评分或模型实验仍需另行取得相应许可下的数据与模型，并准备本机路径。

## 路径映射、原件与排除规则

`raw/` 下的文件保持源文件**逐字节一致**，不为 GitHub 改写其中的绝对路径。历史本机路径仅是审计定位信息，不应直接作为新环境运行参数。

| 原本机前缀 | 本目录中已纳入的对应位置 |
|---|---|
| `/root/autodl-tmp/MMAR/runs/ta-conditional-01/` | `raw/ta-conditional-01/`，或明确标注的 `derived/ta-conditional-01/` |
| `/root/autodl-tmp/MMAR/runs/ta-conditional-transfer-01/` | `raw/ta-conditional-transfer-01/`，或 `derived/ta-conditional-transfer-01/` |
| `/root/autodl-tmp/MMAR/runs/papers28-transfer-01/` | `raw/papers28-transfer-01/`，或 `derived/papers28-transfer-01/` |
| `/root/autodl-tmp/r1-aqa/runs/omni3b-pilot-01/lora/` | `raw/ordinary-lora-pilot/`，或 `derived/ordinary-lora-pilot/` |
| `/autodl-fs/data/west-ta-baselines/` | 仓库工程源码位置；以仓库顶层导航为准 |
| 模型缓存、数据、未选择的 checkpoint/预测路径 | 不随此证据子集发布 |

[manifest.json](manifest.json) 为每份已纳入证据记录原始路径、原始文件大小、原始 SHA-256、导出 SHA-256、导出方式及所删字段。`derived/` 不是原件：门控探针移除 `raw`、`paired`、`probe.selected`；初始化审计移除首样本 `supervision`；SER 审计移除逐文件哈希、逐样本/对话标识、提示与标注；processor 审计移除最长样本条目。聚合情感类别定义、数量、混淆矩阵及文件级内容哈希保留。

未包含权重、优化器状态、音频、数据集内容、原始样本清单、完整预测、题干/选项文本、`pairs.json`、联邦客户样本分区或样本 metadata。`args.json` 和大段训练 stdout 也不纳入；配置、protocol 和纯标量训练日志已保留。原始逐样本探针未发布，不能将派生文件的 SHA 当作原始文件 SHA。完整聚合 `analysis.json` 大小 1,098,199 bytes，是唯一超过 1 MB 的证据文件；其余均小于 1 MB，均为 UTF-8 文本。

## 校验与重建

从仓库根目录检查已发布文件的 SHA-256、允许类型及已知敏感字段：

```bash
python research/evidence/verify_evidence.py
```

若仍在拥有原运行目录的机器，可加 `--sources` 逐项检查来源 SHA-256，确认原件副本字节一致：

```bash
python research/evidence/verify_evidence.py --sources
```

`export_evidence.py` 记录此次白名单导出规则，只读取上述实验目录，写入本目录。它需要原环境，不是训练启动器。原 CPU 回归输出包括失败历史与最终 186 passed / 4 skipped；四项跳过为 RandLoRA 独立 PEFT 环境一项、CUDA LowRA 一项、缺少 pyreft 两项，其中前两项另有单独日志。本次导出只核验证据，不宣称这些历史测试在新环境中已经重新通过。
