# CoTA-LoRA 研究与工程交接

> GitHub 阅读版：证据链接已映射到仓库内副本；历史路径、数值和命令保持原记录。部分逐题资料仅在服务器。原始本机版见 [归档](archive/CoTA-LoRA_CLAUDE_HANDOFF.local.md)，发布材料范围见 [证据说明](../research/evidence/README.md)。

交接对象为接手此项目的 Claude 和研究者。目标投稿 ACL。记录核对时间为 2026-11-21 UTC，依据当前源码、配置、checkpoint、逐题预测和原始日志。本文用于延续研究，具体投稿年份和 ARR 周期尚未确定。

## 1. 接手时先掌握的结论

1. 用户正式确定自研方法名为 **CoTA-LoRA**，英文全称 **Conditional Transport-Aware Low-Rank Adaptation**，中文为“条件传输感知低秩适配”。名称对应最新的问题与音频条件门控 TA。现有代码和旧 checkpoint 仍使用 `ta-lora`、`ta_lora`、`question_audio` 标识，加载时继续读取其原配置。
2. **CoTA 目前尚无稳定优于匹配 LoRA 的证据。** 最新长短混合 CoTA 的 IEMOCAP UAR 为 62.96%，匹配 LoRA 为 63.74%；MMAU 为 72.1% 对 72.5%；MMAR 为 60.5% 对 61.0%。当前可以继续研究机制，投稿论证仍需补强。
3. CoTA 的一个正向发现是，在长音频受控配对上，开启传输比同一 checkpoint 关闭传输提高 8.33 个百分点。相对于等预算、独立训练的 LoRA，增益区间跨零。真实任务收益尚未建立。
4. **28 篇论文方法表与 CoTA 使用两套不同协议。** 表中较强的 LoRAM-TrainSmall 来源于已有论文的无额外对齐预训练消融。它的成绩归属于该基线。CoTA 应与自己的同数据、同覆盖层、同选模协议 LoRA 对照比较。
5. 最新 CoTA 实现限定为 Omni 的 36 层 **O 投影**，在预测首个答案 token 的因果查询位置注入传输。当前动态版本的梯度检查点支持、Q/K/V 扩展和多 token ASR 生成仍待实现与验证。
6. 27 个可执行论文移植均完成双目标评估，第 1 篇另有理论示例及真实模型的已知教师诊断。工程完成状态为 `complete`。这项验收覆盖产物、重载、预测和评分一致性，原论文的全部算法条件和科研有效性还需逐项复核。
7. PiSSA、LoRA-GA、LowRA 在最佳 checkpoint 的两个目标集上全部选择 A。LoRA+、EVA 也显著退化。其初始状态或优化设置存在待解决问题，当前低分不足以支持这些论文方法本身较弱的结论。
8. MMAR、MMAU、IEMOCAP 已在多轮研究决策中被观察。后续论文应披露探索过程，为新一轮实验预先确定开发规则及最终检验安排。

## 2. 用户目标、资源和既有工作约束

- 研究目标是形成可投 ACL 的方法贡献及证据。用户希望理解音频相关任务中普通 LoRA 的局限，并证明 CoTA 的具体贡献。
- 当前确认的资源范围为固定 Qwen2.5-Omni-3B、单卡、现有语料。额外教师、专家库和联邦协议单列训练与存储成本。是否扩展范围由后续研究设计决定。
- 模型、数据和运行产物放在 `/root/autodl-tmp`。复用现有文件，新实验使用新输出目录，保留旧 checkpoint、日志和阴性结果。
- WEST 工程与 editable ms-swift 当前均无 `.git`。使用已保存的内容哈希和源码归档追踪版本。
- 本交接文档只新增文档产物。交接时 GPU 空闲，当前 28 论文队列已结束。接手后先检查实时进程再安排 GPU 工作。
- 用户倾向于完成具体工作、用实验日志说明问题。普通可逆实现选择可直接推进，长训练前应给出具体研究假设、数据范围和预算。

## 3. 环境和入口

| 项目 | 当前位置或版本 |
|---|---|
| 工程根目录 | `/autodl-fs/data/west-ta-baselines` |
| Python | `/root/autodl-tmp/iemocap/swift-ser/venv-5090/bin/python`，Python 3.12.3 |
| GPU | NVIDIA GeForce RTX 5090，`nvidia-smi` 总显存 32607 MiB |
| PyTorch / CUDA | `2.12.1+cu130` / 编译 CUDA 13.0 |
| ms-swift | `4.7.0.dev0`，源码 `/root/autodl-tmp/ms-swift-librispeech/src/ms-swift` |
| Transformers / PEFT | `4.52.3` / `0.15.2` |
| RandLoRA 独立 PEFT | `0.17.1`，`/root/autodl-tmp/r1-aqa/paper-envs/peft-0.17.1` |
| Accelerate / TRL | `1.7.0` / `0.18.2` |
| 本地基座及 processor | `/root/autodl-tmp/west-preflight/model` |
| 官方快照 revision | `f75b40e3da2003cdd6e1829b1f420ca70797c34e` |
| 基座逐文件哈希 | `experiments/ms_swift_librispeech/evidence/verified_snapshot.json` |
| 单卡互斥锁 | `/root/autodl-tmp/r1-aqa/runs/omni3b-tuners-01/gpu.lock` |

环境版本来自运行源码快照的 `environment.json`。该环境复用部分系统包，精确来源也在该文件中。运行前应再次读取实际环境；系统 `python` 与上述实验解释器可能不同。

最重要的阅读路径如下，均相对于工程根目录。

| 文件 | 用途 |
|---|---|
| `experiments/ms_swift_librispeech/swift_asr_ta_runtime.py` | CoTA 数学操作、模态分区、答案查询、门控、cache 与适用范围 |
| `experiments/ms_swift_librispeech/plugin.py` | Swift tuner 注册、版本化保存与恢复 |
| `experiments/ms_swift_librispeech/adapter_audit.py` | 覆盖层、参数集合、初始化与普通 LoRA 对照 |
| `experiments/ms_swift_librispeech/train.py` | 共用 SFT、数据入口、优化、验证、早停、checkpoint 恢复 |
| `experiments/ms_swift_librispeech/tests/test_conditional_gate.py` | 条件门控回归 |
| `experiments/ms_swift_librispeech/tests/test_adapter_pair.py` | 非零权重、BF16、普通 LoRA 与 TA 的一致性边界 |
| `experiments/ms_swift_aqa/evaluate.py` | AQA 受限选项解码 |
| `experiments/ms_swift_aqa/pair_eval.py` | 配对两条同时正确的指标 |
| `experiments/ms_swift_aqa/summarize_pilot.py` | MMAU 原始选项文本映射和官方评分 |
| `experiments/ms_swift_iemocap/evaluate.py` | 四分类 UAR、准确率、逐条结果 |
| `experiments/ms_swift_aqa/paper_plugin.py` | 外部论文方法统一插件 |
| `experiments/ms_swift_aqa/papers28/run.py` | 论文方法训练、独立重载及双集验收 |
| `experiments/ms_swift_aqa/papers28/paired.py` | 按音频或对话聚类的独立配对复算 |
| `experiments/ms_swift_aqa/papers28/report.py` | 完整性检查及实时报告 |
| `docs/superpowers/plans/2026-11-21-conditional-ta.md` | 旧版条件门控实验计划，结合当前代码阅读 |

早期 TouchASU、Qwen3-ASR、7B 和 LibriSpeech 实验仍在盘上。当前 CoTA 的主证据来自下面的 3B 运行目录。

## 4. CoTA 的当前真实实现

每层用列向量记号表示。令 `x_t` 为普通多头注意力输出、尚未经过 O 投影的向量。对每个 head，分别在因果可见、有效的音频 key 和文本 key 内归一化 attention logits，计算两组 value 的加权平均，拼接各 head 后取差。

\[
d_t = \operatorname{concat}_h\left(\mathbb E[V_h\mid\text{audio}]-\mathbb E[V_h\mid\text{text}]\right).
\]

`A` 是共享 rank-r 下投影，`B` 是普通 LoRA 上投影，`C` 是独立的传输上投影，`g` 是 rank-r 静态系数，`R` 是 r×r 条件门矩阵。定义

\[
z=A d_t,\qquad q=A e_{\rm prompt},\qquad
m=2\sigma\left(R\frac{\operatorname{norm}(q)+\operatorname{norm}(z)}{\sqrt2}\right).
\]

`e_prompt` 实际取最后一个音频 token 之后、答案之前的冻结 token embedding 均值，包含题干、选项及模板文字。门控的音频侧输入实际为投影后的音频—文本聚合差 `z`。论文方法说明需准确写出这两个定义。

当前 `gate_multiplier=1` 时，低秩残差为

\[
\Delta o_t=\frac{\alpha}{r}\left[B A x_t+
\mathbf1_{t=t_*}\left\{B(g\odot m\odot z)+C(m\odot z)\right\}\right].
\]

训练的 `t_*` 为首个受监督 token 的前一位置，推理时取最后一个有效 prompt token。后续 cache 生成步的传输为零。普通 LoRA 分支仍参与各位置计算。

关键实现细节如下。

- `A` 用 Kaiming 初始化，`B/C/g/R` 均为零。因此初始 `m=1`，完整 adapter 的初始输出为零。`C` 的梯度先使传输分支开始学习，条件门的梯度启动时序值得进一步实测。
- 当前选用 `uniform` 的分区差传输。另有 attention mass overlap 权重选项，具体公式及边界以 runtime 为准。
- 空模态分区、padding、因果掩码及 cache 有定向测试。稀疏音频或极端长度下仍应覆盖真实模型输入。
- 完整关闭传输需要同时处理静态项和独立 `C` 项；单独令 `g=0` 会留下 `C` 的贡献。现有 off 评估入口实现了完整关闭。
- 实际对照为 O-only rank-12、alpha-24 LoRA，1,769,472 参数；CoTA 为共享 rank 8、alpha 16，1,772,064 参数。两者 alpha/r 缩放均为2，参数量接近，但普通分支维度不同，后续机制消融应显式控制参数分配。
- 当前动态门控通过 `set_transport_gate` 限定为 Omni、O-only、独立 C、text query 与指定残差格式，配置保存为 format 8。Omni 梯度检查点在运行时被显式拒绝，真实配置为 false。Qwen3-ASR 的旧路径不能作为当前动态 Omni 已支持 checkpointing 的证据。
- 实际验证使用 eager attention、非 packing、DynamicCache、单 beam 与一字母答案。向完整语音转写或自由长答案扩展时，需重新设计传输作用位置并验证 cache 与训练监督的一致性。
- 当前方法采用注意力聚合差和 SFT。论文如讨论最优传输、对比学习或因果解释，需要为相应术语提供独立定义及证据。

对应源码重点位于 `swift_asr_ta_runtime.py` 的 `_context_multiplier`、`answer_update`、`set_transport_gate`、`_answer_query`、`_answer_contrast` 和 `attach`。

## 5. 实验目录与两套协议

| 运行目录 | 内容 |
|---|---|
| `/root/autodl-tmp/MMAR/runs/ta-conditional-01` | CoTA 短、长、混合数据实验，等预算 LoRA，对照消融，MMAR 评分与门控诊断 |
| `/root/autodl-tmp/MMAR/runs/ta-conditional-transfer-01` | 同一批混合版 checkpoint 迁移到 IEMOCAP 与 MMAU |
| `/root/autodl-tmp/MMAR/runs/papers28-transfer-01` | 28 篇逐项实现及双目标评估，27 个经验方法加第 1 篇诊断 |
| `/root/autodl-tmp/MMAR/runs/ta-qkvo-01` | 旧静态 TA，Q/K/V 普通 LoRA 加 O 传输，历史证据 |
| `/root/autodl-tmp/MMAR/runs/ta-diagnosis-01` | 旧权重/因果探针和 261 题代理开发集 |
| `/root/autodl-tmp/r1-aqa/runs/omni3b-pilot-01/lora` | 28 篇表中普通 LoRA 的原始训练日志和 checkpoint |

目录名中的 `MMAR` 仅表示保存位置。MMAU 和 MMAR 是两份不同的测试集。

| 设置 | CoTA 主实验 | 28 篇方法实验 |
|---|---|---|
| 训练 | AVQA 原 2048 题加 320 个配对的 640 条，共 2688 条 | AVQA 2048 条 |
| 开发选模 | 变化配对的“两条都正确”率；平分取最早 | AVQA 256 条 accuracy；平分取最早 |
| 常规覆盖 | 36 层 O 投影 | 36 层 Q/K/V/O 和三层 MLP 投影，特殊方法另列 |
| LoRA 对照 | rank12，alpha24 | rank8，alpha16 |
| 有效 batch | 4×累积4=16 | 4×累积4=16 |
| 训练量 | 1轮，168更新 | 最多2轮，256更新 |
| 优化 | AdamW，恒定2e-4，无warmup，weight decay0 | 相同基础配置，方法特有分组及阶段另列 |
| 验证频率 | 每32更新及轮末 | 每32更新及末步 |
| patience | 3 | 3 |
| 梯度检查点 | false | 常规新臂开启，历史和部分特殊方法不同 |
| 主种子 | 42 | 42 |

两套协议分别报告。同一目标集上的跨协议差值只能作为描述性结果。公平对照需重新统一样本、方法预算、实际学习率、选择指标与计算成本。

### CoTA 当前最关键的结果

| 指标 | 混合版匹配 LoRA | CoTA | 同一 CoTA 关闭传输 | 冻结基座 |
|---|---:|---:|---:|---:|
| 长变化配对，两条均正确 /96 | 70 | 73 | 65 | 5 |
| 混合变化配对，两条均正确 /96 | 68 | 68 | 62 | — |
| 真实代理开发正确 /261 | 202 | 201 | 202 | 203 |
| MMAR 官方准确率 | 61.0% | 60.5% | 60.6% | 61.9% |
| IEMOCAP UAR | 63.74% | 62.96% | 62.61% | 59.94% |
| IEMOCAP 准确率 | 62.13% | 60.52% | 60.35% | 57.21% |
| MMAU 官方准确率 | 72.5% | 72.1% | 72.5% | 75.7% |

混合版两组均选第160步，实际训练168步。CoTA 训练295.19秒，峰值 allocated/reserved 为28.66/30.33 GiB；匹配 LoRA 为275.49秒、27.93/29.62 GiB。具体耗时适用于当次设备与输入。

长配对的 CoTA 开关差为 +8.33 个百分点，按来源配对的95%区间为[+2.08,+14.58]；相对独立训练 LoRA 为 +3.12，区间[-3.12,+9.38]。该差异支持研究传输分支的局部作用，仍不足以证明动态门在真实任务上优于匹配基线。

历史短版门控探针在24条输入上发现，保持题干更换音频时，门乘数的平均绝对变化约0.000187。将门学习率提高到0.0032后变为0.002915，同时答案应保持配对也从约0.000103增至0.001737。门敏感性增大后的选择性及任务收益需共同检查。这些探针来自短版 checkpoint，解释混合版时需重新测量。

优先读取 [CoTA 主报告](../research/evidence/raw/ta-conditional-01/REPORT.md)、[主报告逐项核验](../research/evidence/raw/ta-conditional-01/analysis.json)、[跨任务报告](../research/evidence/raw/ta-conditional-transfer-01/REPORT.md)、门控探针（服务器原件：`/root/autodl-tmp/MMAR/runs/ta-conditional-01/gate-probe.json`） 与 提高门学习率的探针（服务器原件：`/root/autodl-tmp/MMAR/runs/ta-conditional-01/gate-probe-fastgate.json`）。

当前 CoTA 主 checkpoint 为 `/root/autodl-tmp/MMAR/runs/ta-conditional-01/train/question_audio-mixed/checkpoint-160`，匹配基线为同目录下 `train/lora-mixed/checkpoint-160`。跨任务 `protocol.json` 封存了两份权重及配置的 SHA-256。

## 6. 28 篇方法的完整结果

下表按用户给出的编号排列。三个指标均为百分比。本表采用上一节的 AVQA2048 协议。第1项为同源已知教师的逐层 Q/O 构造诊断。

| 编号 | 方法 | IEMOCAP UAR | IEMOCAP Acc | MMAU官方Acc |
|---:|---|---:|---:|---:|
| 对照 | 冻结基座 | 59.94 | 57.21 | 75.70 |
| 对照 | 普通LoRA | 61.59 | 59.15 | 72.60 |
| 1 | Expressive Power，已知教师Q/O构造 | 25.00 | 13.70 | 38.60 |
| 2 | LongLoRA | 49.18 | 41.58 | 72.50 |
| 3 | VeRA | 60.72 | 57.94 | 75.30 |
| 4 | DoRA | 61.08 | 56.49 | 72.20 |
| 5 | FourierFT | 26.29 | 22.64 | 31.60 |
| 6 | LoRA+ | 36.85 | 43.11 | 40.80 |
| 7 | LoRA-GA | 25.00 | 13.70 | 39.80 |
| 8 | PiSSA | 25.00 | 13.70 | 39.80 |
| 9 | LoRAM-TrainSmall | 63.60 | 62.21 | 75.10 |
| 10 | RandLoRA | 24.96 | 19.50 | 23.20 |
| 11 | LowRA | 25.00 | 13.70 | 39.80 |
| 12 | AuroRA | 60.14 | 56.49 | 72.90 |
| 13 | EVA | 21.66 | 19.66 | 38.20 |
| 14 | FlyLoRA | 63.27 | 61.08 | 71.70 |
| 15 | GraLoRA | 60.65 | 57.94 | 71.90 |
| 16 | Magnitude-LoRAM | 54.76 | 52.70 | 61.30 |
| 17 | RoLoRA，四客户端联邦 | 62.24 | 59.15 | 73.70 |
| 18 | TopLoRA | 63.31 | 59.31 | 71.60 |
| 19 | Uni-LoRA | 60.00 | 57.45 | 75.80 |
| 20 | aLoRA | 59.86 | 56.89 | 73.40 |
| 21 | BoRA | 59.81 | 55.12 | 71.80 |
| 22 | IGU-LoRA | 60.07 | 57.53 | 75.70 |
| 23 | LoFT | 40.33 | 33.60 | 56.40 |
| 24 | LoRA-Mixer | 59.98 | 56.41 | 73.00 |
| 25 | LoRAGen | 62.22 | 60.60 | 74.00 |
| 26 | RaLoRA | 57.76 | 53.67 | 72.90 |
| 27 | Stable-LoRA | 57.71 | 52.05 | 73.00 |
| 28 | LoKRA | 62.60 | 58.58 | 72.10 |

普通 LoRA 开启与关闭梯度重算的训练 loss、开发分及目标预测一致。LoRAM-TrainSmall 相对普通 LoRA 的 UAR/MMAU 增益为+2.01/+2.50个百分点，配对区间为[+0.48,+3.56]/[+0.30,+4.73]。Uni-LoRA 的 MMAU 75.8%较冻结基座多1题。区间只反映固定单种子 checkpoint 的测试样本变异，尚未做多重比较校正。

完整分数、参数量、峰值、耗时和逐条路径见 [28篇报告](../research/evidence/raw/papers28-transfer-01/REPORT.md)、[结果JSON](../research/evidence/README.md)、[配对区间](../research/evidence/raw/papers28-transfer-01/paired-target-scores.json) 与 [特殊成本](../research/evidence/raw/papers28-transfer-01/SPECIAL_PROTOCOLS.md)。

方法移植的适用范围必须结合来源说明阅读。

- LongLoRA 在 Omni 原生上下文长度内移植 shifted attention，尚未形成长上下文扩展论文的完整复现。
- LoRAM 采用无额外对齐预训练的 LoRAM-Rand 消融，推理恢复到完整维度。
- RoLoRA 在单卡上模拟4个客户端，网络通信时间无实测。
- LoRAGen 使用4个源专家的小规模库，固定生成一个 adapter 用于两目标。教师库规模、任务和基座与原论文不同。
- LowRA 量化252个语言侧投影，平均编码1.75 bit；量化误差与初始化保真仍需诊断。
- 第1篇的已知教师来自同源 AVQA 全参数训练。逐层 Q/O 构造与原作者理论示例分别有证据。
- VeRA、DoRA、FourierFT 与普通 LoRA 复用了同协议历史训练，当前轮复核或补齐目标集评估。

作者代码版本、许可证及论文差异见 [1–10篇来源](../experiments/ms_swift_aqa/papers28/sources-01-10.md)、[11–20篇来源](../experiments/ms_swift_aqa/paper_middle_sources.md)、[21–28篇来源](../experiments/ms_swift_aqa/papers28/sources-21-28.md)。原始28篇列表在 `/root/.codex/attachments/e6b03fbe-dd6b-4383-bd28-009e532374f1/Pasted text.txt`。

## 7. 数据、评估与研究选择的现状

| 数据 | 实际入口 |
|---|---|
| AVQA2048/256和MMAU测试清单 | `/root/autodl-tmp/r1-aqa/data/pilot-01/{train,dev,test}.jsonl` |
| AVQA音频与原始下载 | `/root/autodl-tmp/r1-aqa/avqa` |
| MMAU原始发布与评分脚本 | `/root/autodl-tmp/r1-aqa/mmau` |
| IEMOCAP清单 | `/root/autodl-tmp/iemocap/swift-ser/data/{train,dev,test}.jsonl` |
| IEMOCAP音频 | `/root/autodl-tmp/iemocap/swift-ser/data/IEMOCAP_full_release` |
| CoTA混合训练清单 | `/root/autodl-tmp/MMAR/runs/ta-conditional-01/data/mixed/train.jsonl` |
| CoTA混合配对开发 | `/root/autodl-tmp/MMAR/runs/ta-conditional-01/data/mixed/pair-dev.jsonl` |
| 261题多域代理开发 | `/root/autodl-tmp/MMAR/runs/ta-diagnosis-01/multidomain/dev.jsonl` |
| MMAR历史测试运行与清单 | `/root/autodl-tmp/MMAR/runs/omni3b-avqa-01` |

IEMOCAP 四类为 angry、happy+excited、sad、neutral。现有监督训练为Session1–3共3259条，开发为Session4共1031条，测试为Session5共1241条。当前两套迁移实验均未用这些情感训练数据训练。未来若采用监督情感协议，应重新声明该训练任务并保持所有方法共用数据。

261题代理开发各取87条Clotho-AQA、MUSIC-AVQA音频题与IEMOCAP Session4，涉及210个音频来源；其审计中排除了2条与MMAU高度匹配的语音。它的用途是多域代理检查，来源和重编辑重复仍有验证局限。详见 `ta-diagnosis-01/multidomain/dev-audit.json`。

AVQA2048中A/B/C/D计数为515/510/506/517。Which、Come From、Happening、Where占2031条，Why10条、Before Next5条。源任务与MMAU三域能力、情感任务存在显著差异。现有训练含4条明确View题、开发含1条，以及3条全静音训练音频。下一轮清理应为同协议全部方法生成一份共用新清单并记录哈希。

评估均使用相同 prompt、BF16、eager、batch2、有效长度处理、合法选项单token贪心。MMAU/MMAR将字母还原为原选项文本后调用各自官方评分。IEMOCAP报告UAR及准确率。当前分数的解码条件应写入论文，引用外部榜单时核对其生成方式和训练资源。

规范化PCM审计已检查AVQA train/dev、MMAU及IEMOCAP测试的六对交集为零，训练与开发的原始视频来源分离。该检查的覆盖范围是完全相同的规范化音频，近似重复、重编码及预训练污染尚无全面排查结论。[音频审计](../research/evidence/raw/papers28-transfer-01/split-audio-audit.json)

MMAR的旧阶段脚本根据已观察到的官方分数触发后续实验，具体见 `ta-conditional-01/phase-two-trigger.json`、`phase-three-trigger.json` 和 `third-stage.sh`。MMAU/IEMOCAP也已多次用于比较方案。后续应把这一轮记录为探索性结果，在开发集上进行新选择，并预先记录最终检验方案。所有候选分数和未成功实验均应保留。

## 8. 对“微调后下降”的现有诊断

普通LoRA的冻结源开发分为71.09%，训练后最佳为80.47%；MMAU由75.70%降到72.60%，IEMOCAP UAR由59.94%升到61.59%。MMAU少答对31题，其中音乐少16题、语音少8题、声音少7题。这支持研究源任务适应与目标迁移之间的取舍。

| 普通LoRA步数 | 32 | 64 | 96 | 128 | 160 | 192 |
|---|---:|---:|---:|---:|---:|---:|
| 最近32步平均loss | .2037 | .1843 | .1703 | .1466 | .0837 | .0728 |
| AVQA dev正确数/256 | 205 | 205 | 206 | 203 | 206 | 205 |

第96步即0.75轮最佳；第192步即1.5轮触发连续3次未改善的早停。开发成绩进入平台，末次与最佳只差1题，对“严重过拟合”的证据有限。训练记录没有开发交叉熵loss，目标集也只有选中checkpoint的结果，无法从现有日志定位目标退化的起始步。

| 方法 | 首个更新前batch loss | 诊断重点 |
|---|---:|---|
| 普通LoRA | .2723 | 稳定拟合，开发平台；优先检查优化强度和目标开发集 |
| LoRA+ | .2723 | 实际B学习率.0032，后续loss与梯度骤增，开发49.61%跌到约23% |
| PiSSA | .2774 | 更新后迅速退化，梯度峰值约37830，排查初始化尺度与LR组合 |
| EVA | .2723 | 更新后迅速退化，梯度峰值约247255，排查重分配/缩放/优化设置 |
| LoRA-GA | 13.9457 | 第一轮优化前已偏离，检查非零初始化补偿和BF16保真 |
| LowRA | 9.0786 | 第一轮优化前已偏离，检查1.75bit量化及低秩补偿误差 |

这些臂的首批输入哈希一致。PiSSA/GA/LowRA的最佳开发点均全选A，得到72/256=28.125%，随后两个目标集2241条也全部选A。源训练标签接近均衡。每题通常只监督3个token，答案字母与结束标记共同贡献loss；只学会结束符、答案接近随机时，loss可接近ln(4)/3≈0.462。后期loss降到.46–.50不足以证明答案能力恢复。

现有PiSSA/GA保存真实BF16残差及SHA，并校验训练时最佳dev预测与干净重载预测一致。该证据使单纯的重载故障解释缺乏支持，但初始化后3B的功能保真仍待单独测量。

更多轮数的证据仅对部分方法较积极。LoRAM在第256步仍改善，VeRA在224步后达到最好；普通LoRA、UniLoRA已早停，联邦RoLoRA在160步达到82.81%后降到80.47%。CoTA混合版只训练1轮，延长训练仍是待验证假设，应以稳定的独立开发规则决策。

## 9. 建议 Claude 优先完成的工作

以下是研究建议，具体预算和最终协议应在实验前记录。当前任务是提升CoTA的可解释贡献和可靠性。

### P0 先确认可解释的起点

1. 复核本文引用的CoTA checkpoint、配置和同预算LoRA，确认开启/关闭传输的实际行为。完成真实3B的首答案位置、padding和cache复核。核对`R/g/C/A/B`的梯度、更新范数和每层残差比。
2. 在固定的源开发小集合上比较干净基座与adapter初始化后、尚未更新的logits和loss。CoTA的零更新预期应明确；GA的补偿和LowRA量化分别使用符合其算法的验收条件。
3. 对LoRA+、PiSSA、EVA等不稳定基线做短程优化诊断。使用相同样本和计算预算记录真实分组LR、梯度裁剪前后范数、答案分布，恢复合理基线后再解释主方法差异。
4. 新日志同时记录答案位置交叉熵、完整监督token loss、开发准确率、每类召回与预测字母直方图。`token_acc`包含结束符，应有单独答案指标。
5. 建立后续开发和最终检验规则。若主线选择监督SER，可使用已有Session1–3训练、Session4开发；若选择音频推理迁移，使用来源隔离的音频依赖开发题，并说明既有公开测试已被观察。

### P1 检验门控到底增加了什么

| 假设 | 建议的最小对照 | 需要的证据 |
|---|---|---|
| 动态门具有样本选择性 | 同预算LoRA、静态传输、question门、question+audio门、强制m=1 | 答案变化配对与保持配对均有结果，真实开发收益与门响应共同成立 |
| 独立C才是主要贡献 | 独立C、共享B、匹配参数的额外普通LoRA | 同预算及相近缩放下的训练与推理消融 |
| 问题表示太弱 | 当前embedding均值与轻量上下文表示 | 严格因果、无答案泄漏，新增参数成本有对照 |
| 归一化抹去了有用音频量级 | 当前归一化门与显式幅度特征的小变体 | 对音量/长度变化的选择性和鲁棒性 |
| 答案位置注入太晚 | 当前单位置与少量因果位置注入 | cache行为、训练推理一致性和显存实测 |
| 时长捷径主导 | 同内容跨时长、跨静音占比、跨录音源 | 配对成绩跨时长迁移，并对真实音频有增益 |

每次修改先说明要否定哪个假设。门控热图和范数仅作为诊断，主结论应由控制实验支持。训练式消融与同checkpoint推理关闭各有用途，应同时保留其解释范围。新版 `transport_gate='static'` 表示首答案位置的静态传输；旧静态TA还包括多个文本查询位置，做门控消融时必须固定注入位置。

### P2 优化与训练量

- 先固定其余参数，对普通LoRA及CoTA短扫较低LR，例如5e-5、1e-4及原2e-4；随后再单独比较warmup和衰减。候选预算与选择标准在开发集上预先确定。
- 对门参数记录实际LR。此前提高到.0032使门更敏感，但真实任务提升尚不稳定，应避免把门幅度本身作为目标。
- 延长仍在改善的方案时，保留开发早停与实际曝光样本数。更长时长音频会增加实际计算量，单纯步数一致不能覆盖所有成本差异。
- 最终主比较优先覆盖冻结基座、充分调参的LoRA、DoRA、一个有机制关联的强基线及CoTA。28篇表可作为扩展研究材料，所有低分臂都需披露其配置和诊断状态。

### P3 形成 ACL 可审阅的证据

建议预先选定一条主问题，例如“问题条件化的模态残差能否在参数预算相近时改善依赖音频证据的语言决策，并减少源任务微调对通用能力的损害”。当前结果对其中若干环节只有局部支持，需要允许后续实验否定这个假设。

主比较建议覆盖至少3个训练种子，报告平均值、标准差和样本聚类区间；这是本项目的研究设计建议。IEMOCAP只有Session5两位说话者时应限定结论，若做跨Session评估，应明确外层测试与内层选模边界，并说明既有数据已被探索。

需要完整的新颖性核对。除28篇LoRA外，检索音频条件adapter、动态路由、模态门控、attention差分、音频依赖评测及语言模型中的声学证据使用。逐项比较运算位置、条件输入、参数化、监督信号和计算成本。单独命名CoTA和组合已有模块仍需建立可验证的新贡献。

## 10. 投稿准备及公开政策

ARR的评审表关注主张是否有充分证据、实验方法是否可靠以及工作对ACL读者的价值。方法创新、实证发现和分析均可形成贡献。当前最缺的是匹配对照下的稳定收益及机制证据，不能由增加基线数量代替。[ARR评审表](https://aclrollingreview.org/reviewform)

准备可复现材料时记录数据来源、预处理、完整超参、开发与测试成绩、种子、搜索预算、效率和局限。软件与数据来源应有明确引用，论文中的结论范围与测量范围保持一致。[ARR Responsible NLP Research](https://aclrollingreview.org/responsibleNLPresearch/)

工程已有AI辅助编程、分析和文档记录。投稿时按实际使用情况由人类作者核实和披露，并由作者承担内容、引用及结果的责任。具体投稿周期再核对当期政策。[ACL Publication Ethics](https://www.aclweb.org/adminwiki/index.php/ACL_Policy_on_Publication_Ethics)

IEMOCAP在会话中经Kaggle来源下载，本次尚未见到用户或机构的原始使用许可凭据。投稿与发布数据衍生物前应核查实际授权。官方发布页提供许可和申请入口。[USC IEMOCAP Release](https://sail.usc.edu/iemocap/iemocap_release.htm)

## 11. 可复核状态、命令和产物保护

最近一次CPU全套回归为 **186 passed / 4 skipped**。其中RandLoRA已在其独立PEFT环境另测，LowRA有真实RTX5090 CUDA验收，两项可选ReFT测试因缺少pyreft跳过。命令、输出、失败历史均在28篇目录的`logs/`。这些测试覆盖既有小模型和特定真实模型路径，新结构仍需相应验收。

终态曾因BoRA UAR的Python与NumPy求平均相差1 ULP而误报失败，已保留原始失败记录并补回归。当前要求相同样本数和混淆矩阵，UAR容许2 ULP，再独立重算评分后生成`complete`。[最终复核](../research/evidence/raw/papers28-transfer-01/logs/final-replay-audit.log) · [回归日志](../research/evidence/raw/papers28-transfer-01/logs/final-replay-regression.log) · [终态](../research/evidence/raw/papers28-transfer-01/disposition.json)

最近的源码归档在下方，包含WEST、editable Swift、两版PEFT及所用作者代码。归档日期只能证明该时点的内容；较早运行的方法要结合各自的原始来源记录。

```text
/root/autodl-tmp/MMAR/runs/papers28-transfer-01/sources/83075b150b88cbe603e29d7aee7aeab976fc280b9482606f906d69eb80014b5a
```

以下命令用于接手核查。测试会产生常规缓存；已有checkpoint保持按原目录读取。

```bash
cd /autodl-fs/data/west-ta-baselines
PY=/root/autodl-tmp/iemocap/swift-ser/venv-5090/bin/python
RUN=/root/autodl-tmp/MMAR/runs/papers28-transfer-01
nvidia-smi

env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMBA_NUM_THREADS=2 \
  "$PY" -m pytest -q -rs \
  experiments/ms_swift_aqa/tests \
  experiments/ms_swift_librispeech/tests \
  experiments/ms_swift_iemocap/tests

"$PY" -m experiments.ms_swift_aqa.papers28.provenance verify \
  "$RUN/sources/83075b150b88cbe603e29d7aee7aeab976fc280b9482606f906d69eb80014b5a"
```

原始CoTA启动器位于 `ta-conditional-01/run.py` 和 `run.sh`，部分阶段路径硬编码该旧实验目录。`second-stage.sh`和`third-stage.sh`还包含基于旧MMAR分数的后续触发逻辑。新实验应先提取可配置入口或使用下方共用训练入口，明确新数据、配置及输出目录。

```bash
# 先准备可审阅的新配置与共用清单，再设置 NEW_*。
# 方法ID继续用 ta-lora，论文展示名为 CoTA-LoRA。
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 ENABLE_AUDIO_OUTPUT=0
flock /root/autodl-tmp/r1-aqa/runs/omni3b-tuners-01/gpu.lock \
  "$PY" -m experiments.ms_swift_librispeech.train \
  --config "$NEW_CONFIG" --method ta-lora \
  --train "$NEW_TRAIN" --dev "$NEW_DEV" --output "$NEW_OUTPUT"
```

原始文件名包括 `protocol.json`、`args.json`、`initialization.json`、`logging.jsonl`、`dev/step-*/metrics.json`、`dev/step-*/hypotheses.jsonl`、`result.json`、`reload-dev/`、`reload-test/`、`ser/`。检查训练轨迹时读取这些原件，报告表格仅用于导航。

## 12. 可直接发给 Claude 的启动说明

> 请从这份交接文档及引用的原始文件独立核实当前状态。研究目标是面向ACL改进CoTA-LoRA，并建立可靠、可解释的实验贡献。固定Qwen2.5-Omni-3B和现有语料，单卡运行，保留历史结果。
>
> 优先复核CoTA的真实实现、梯度和首答案注入机制，以及匹配LoRA的公平性。检查严重退化基线在初始化后及前几个优化步的行为。列出已证实的代码问题、由日志支持的优化问题和仍待验证的机制假设。
>
> 根据这些证据提出少量可证伪的改进，每轮记录假设、共用数据、真实参数预算、开发选模规则、实际分组学习率、计算成本和停止条件。先完成定向测试与短实验，再扩大验证。以开发结果决定下一步，并把既有测试暴露和阴性结果写入研究记录。
>
> 交付包含修复代码及回归证据、原始日志与逐题预测、CoTA和匹配强基线的多种子结果、有效及无效消融，以及对ACL贡献是否成立的诚实判断。请优先推进最可能改变结论的验证。
