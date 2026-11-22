# Qwen2.5-Omni-3B 的 ms-swift 微调方法对比

未经微调的基座在完整 MMAU Test-mini 上答对 **756/1,000（75.6%）**。微调组中得分最高的是 **LLaMA-Pro，75.4%**。VeRA、IA³ 的严格准确率为 75.2%。按 904 个音频簇进行 10,000 次配对重采样，以百分点计，对 VeRA 的差值为 +0.20（95% 区间 -0.99 至 +1.39），对 IA³ 的差值为 +0.20（95% 区间 -1.21 至 +1.68）。本轮使用单一训练种子 42。[逐题与 checkpoint 复核](/root/autodl-tmp/r1-aqa/runs/omni3b-tuners-01/verified-comparison.json) · [准确率与参数量图](/root/autodl-tmp/r1-aqa/runs/omni3b-tuners-01/accuracy-and-parameters.png)

共核对 16 个配置，15 个产生了完整测试成绩。LoRA、DoRA 和改进 TA-LoRA 复核自上一轮，本轮新增方法在同一张 GPU 上串行执行。[三方法原报告](REPORT-omni3b.md) · [运行目录](/root/autodl-tmp/r1-aqa/runs/omni3b-tuners-01)

| 方法 | AVQA 开发 | MMAU 严格 | MMAU 官方文本 | 最佳步 / 总更新 | 可训练参数 | 训练秒 | 训练峰值 allocated / reserved（GiB） |
|---|---:|---:|---:|---:|---:|---:|---:|
| LLaMA-Pro | 74.22% | 75.4% | 75.5% | 32 / 128 | 77,076,992 | 147.9 | 11.97 / 12.46 |
| VeRA | 74.22% | 75.2% | 75.3% | 224 / 256 | 1,096,704 | 416.7 | 20.61 / 21.30 |
| IA³ | 82.81% | 75.2% | 75.4% | 224 / 256 | 709,632 | 367.2 | 20.09 / 20.88 |
| 改进 TA-LoRA | 80.47% | 73.2% | 73.4% | 32 / 128 | 15,556,896 | 217.2 | 22.72 / 23.67 |
| AdaLoRA | 82.81% | 73.2% | 73.4% | 256 / 256 | 22,453,200 | 529.3 | 19.76 / 20.57 |
| LoRA | 80.47% | 72.4% | 72.6% | 96 / 192 | 14,966,784 | 288.4 | 19.30 / 20.13 |
| DoRA | 80.86% | 72.0% | 72.2% | 64 / 160 | 15,998,976 | 343.0 | 24.42 / 25.68 |
| BOFT | 83.20% | 72.0% | 72.2% | 64 / 160 | 7,741,440 | 594.8 | 11.54 / 12.34 |
| 瓶颈 Adapter | 79.30% | 71.9% | 72.2% | 64 / 160 | 14,232,960 | 218.3 | 17.18 / 17.93 |
| rsLoRA | 81.64% | 70.8% | 70.9% | 64 / 160 | 14,966,784 | 238.4 | 19.24 / 20.13 |
| ReFT | 75.39% | 66.5% | 66.6% | 64 / 160 | 1,179,936 | 280.2 | 18.13 / 18.75 |
| BoNE | 75.00% | 65.5% | 65.8% | 128 / 224 | 8,257,536 | 329.8 | 17.09 / 17.87 |
| LoRA-LLM | 67.58% | 42.5% | 42.8% | 256 / 256 | 1,321,327,616 | 783.5 | 19.14 / 19.70 |
| Full（Thinker 语言侧） | 28.12% | 39.5% | 39.8% | 32 / 128 | 3,397,103,616 | 337.8 | 30.38 / 30.71 |
| FourierFT | 30.86% | 31.4% | 31.6% | 256 / 256 | 504,000 | 802.8 | 11.46 / 12.22 |

## 数据、选模与评分

训练使用 AVQA 的 2,048 道题，开发使用按来源视频隔离的 256 道题。测试使用完整 MMAU Test-mini 的 1,000 道题。模型仅接收录音、题干和原始选项。所有方法都在该题合法选项字母中贪心解码一个 token。严格准确率按选中选项与官方正确选项的原文是否一致计分，官方文本列另用固定版 MMAU 文本评分器复算。[原始评分程序](https://github.com/Sakshi113/MMAU/blob/110127f54c0dfba3faa5ec9feee4a7e4148679c5/evaluation.py) · [数据清单与哈希](/root/autodl-tmp/r1-aqa/runs/omni3b-tuners-01/manifests.json)

统一使用 BF16、AdamW、恒定学习率 0.0002、零权重衰减、有效 batch 16，最多两轮。每 32 次更新计算完整开发准确率，连续三次未提升早停，同分取较早 checkpoint。每组在独立进程重载开发选定权重，逐字节复现开发预测后再评测试。训练秒包含保存 checkpoint，扣除开发评分。各方法的验证和测试时间保存在单组结果文件中。[完整配置](omni3b-tuners.json) · [各方法计时和显存](/root/autodl-tmp/r1-aqa/runs/omni3b-tuners-01/results)

另用同一份未经微调的基座直接评分，AVQA 开发为 182/256（71.09%），MMAU 严格为 756/1,000（75.6%），官方文本为 75.7%。冻结基座与所有微调组共用相同的提示、选项顺序和解码器。[冻结基座预测](/root/autodl-tmp/r1-aqa/runs/omni3b-tuners-01/frozen-base)

基座和 processor 使用本地验证过的官方快照，提交为 `f75b40e3da2003cdd6e1829b1f420ca70797c34e`。单卡 NVIDIA GeForce RTX 5090，显存 32,607 MiB。运行版本为 ms-swift 4.7.0.dev0、Transformers 4.52.3 和 PEFT 0.15.2。[快照核验](../ms_swift_librispeech/evidence/verified_snapshot.json) · [原运行环境与源码哈希](/root/autodl-tmp/r1-aqa/runs/omni3b-tuners-01/compatibility/provenance.json) · [补测 IA³ 的版本记录](/root/autodl-tmp/r1-aqa/runs/omni3b-tuners-01/compatibility/provenance-ia3.json)

## 训练范围与可比性

- LoRA、rsLoRA、DoRA 和改进 TA-LoRA 使用相同语言侧 A/B 初始化。Q/K/V/O 和三个前馈投影覆盖全部 36 个语言解码层。VeRA、BoNE、BOFT、AdaLoRA 和 FourierFT 使用同样的投影范围，秩、共享参数及初始化按方法配置。AdaLoRA 表中统计优化器管理的初始参数数，其选定 checkpoint 的逐投影有效秩有单独记录。[秩分配](/root/autodl-tmp/r1-aqa/runs/omni3b-tuners-01/adalora/selected-ranks.json)

- 瓶颈 Adapter 在 36 层的 MLP 输出处注入宽度 96 的模块，LLaMA-Pro 添加一个解码块。ReFT 对 36 层隐藏状态施加秩 8 的干预，保持零权重衰减并关闭语言层重算。LoRA-LLM 同时更新语言 LoRA 和音频塔，可训练参数计数还包含视觉塔的 668,684,288 个参数。纯音频训练中视觉塔权重保持逐字节一致。Full 按 Swift 多模态默认冻结设置训练完整 Thinker 语言模型及输出头，并使用 fused AdamW。[各组生效配置](/root/autodl-tmp/r1-aqa/runs/omni3b-tuners-01/configs)

- IA³ 与其他投影方法覆盖相同的 252 个语言投影。原生 Swift 示例依赖的 MLP 路径在 Omni 注册信息中为空，因此实验显式指定语言侧路径。沿用原生 PEFT 的全一初始化、注意力输出缩放和 MLP 输入缩放。另一个名为 `dummy` 的注册项直接返回传入模型。[本机 Swift 注册表核验](/root/autodl-tmp/r1-aqa/runs/omni3b-tuners-01/compatibility/tuner-coverage.json)

- FourierFT 使用该版本 PEFT 的默认非零标准正态频谱初始化，每个投影 2,000 个频率，缩放 300。BOFT 使用两个蝶形因子。两者的内存优化实现均通过稠密参考公式的输出与梯度检验。ReFT 使用独立安装的 PyReFT 0.1.0 和 PyVene 0.1.8，并接入 Omni 音频输入。时间和显存反映这些具体实现。AQA 和共用训练流程的 48 项定向 CPU 测试全部通过。[实现说明](README.md) · [定向测试日志](/root/autodl-tmp/r1-aqa/runs/omni3b-tuners-01/final-cpu-tests.log)

同一轮的学习率和数据固定，各方法的可训练参数量与训练范围不同。Test-mini 的上一轮三方法结果在扩展本轮方法时已经公开。这些数字描述当前单种子配置在该测试集上的表现。

实际物理 batch 与梯度累积组合为 4 × 4。开启语言层梯度重算的组为 BOFT、LoRA-LLM、Full（Thinker 语言侧）、FourierFT。

相对未微调基座，MMAU 严格准确率的变化以百分点计为 LLaMA-Pro -0.20，VeRA -0.40，IA³ -0.40，改进 TA-LoRA -2.40。[与基座的逐题配对区间](/root/autodl-tmp/r1-aqa/runs/omni3b-tuners-01/verified-comparison.json)

## 改进 TA-LoRA 与两组基线

改进 TA-LoRA 比 LoRA 多答对 8 道，比 DoRA 多答对 12 道。按音频簇配对，前者差值为 +0.80 个百分点（95% 区间 -1.11 至 +2.77），后者为 +1.20（-0.40 至 +2.83）。两个区间都包含零。该区间仅反映固定 checkpoint 在测试音频上的抽样波动。[逐题配对结果](/root/autodl-tmp/r1-aqa/runs/omni3b-tuners-01/results)

语言侧全量微调在统一学习率 0.0002 下，最佳 checkpoint 对全部 1,000 道 MMAU 题都选择了 A。395 道题将 A 列为正确选项，解释了这组的严格分数。[全量微调逐题预测](/root/autodl-tmp/r1-aqa/runs/omni3b-tuners-01/full/reload-test/predictions.jsonl)

## 没有测试分数的配置

- LongLoRA。原生 Swift 在替换注意力前要求 Llama 架构，Qwen2.5-Omni 未通过该检查。[运行日志](/root/autodl-tmp/r1-aqa/runs/omni3b-tuners-01/longlora-train.log)
