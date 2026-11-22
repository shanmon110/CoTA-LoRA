# AVQA 音频问答与 MMAU Test-mini 三方法实验

[查看三组实测结果](REPORT.md) · [Qwen2.5-Omni-3B 三组实测结果](REPORT-omni3b.md)

小米 R1-AQA 用清华 AVQA 的训练题和视频音轨训练 Qwen2-Audio-7B-Instruct，采用 GRPO。论文报告约 3.8 万条可用训练样本，在 MMAU 上评估。这里沿用训练源与测试任务，比较本工程的三个 adapter。基座为 Qwen3-ASR-0.6B，同一份官方权重独立初始化，训练目标为生成一个代表正确选项的字母。[R1-AQA](https://github.com/xiaomi-research/r1-aqa) · [清华 AVQA](https://mn.cs.tsinghua.edu.cn/avqa/) · [MMAU](https://github.com/Sakshi113/MMAU)

## 数据和口径

- `prepare_avqa.py` 用固定版本的原始 `train_qa.json` 校验答案下标和原选项。首轮从可获取音轨的记录中选 2,048 道训练题和 256 道开发题，按 YouTube 源视频分组。保留原题与选项顺序。AVQA 部分题目使用视觉线索，三个模型都只接收音频和文字。
- `prepare_mmau.py` 按官方 GitHub UUID 顺序组装完整 Test-mini。题目实际含 2 到 8 个选项；题目和答案取固定版本的官方 JSON，媒体来自官方 Hugging Face 发行包。[MMAU 音频](https://huggingface.co/datasets/gamma-lab-umd/MMAU-test-mini)
- `freeze_data.py` 解码三个 split 的音频，检查 16 kHz 单声道 PCM 指纹。如发现跨集一致音轨，优先保留测试、其次开发，在较低优先级划分剔除对应源视频，并为三组固化同一份清单。不同剪辑和有损重编码仍可能躲过精确指纹。`audit.json` 保存清单哈希及剔除记录。
- 主指标为逐题正确率。模型从该题合法字母中选择下一 token 得分最高者，选中相同的正确答案文字均计对。二级指标把模型选定的原选项文字交给 MMAU 官方 `string_match`；某些词序不同的选项会得到不同判分。小米配方则采用生成答案文本的解码流程。每组都保存逐题预测、分领域正确率、训练时长及显存记录。

## 训练和选模

三组共同使用 seed 42、BF16、非重入梯度检查点、batch 8 × 梯度累积 2、AdamW、恒定学习率 0.0002，训练上限两轮。每 32 次优化器更新及训练终点计算开发集准确率，连续三次未提高早停，同分取较早的 checkpoint。全部方法锁定后，从干净基座复现各自的开发预测，再统一测 Test-mini。[配置](pilot.json)

所有方法在 28 层语言解码器的 Q/K/V/O 和 MLP gate/up/down 投影使用 rank 8、alpha 16。LoRA 和 DoRA 使用原生 PEFT。改进版 TA 在 O 投影对所有有效查询使用音频与文本的上下文差，增加独立、零初始化的上投影来学习这个差，其余六个投影使用普通 LoRA。音频编码器冻结。[实现](../ms_swift_librispeech/swift_asr_ta_runtime.py)

## 下载

本机实验环境和数据根目录如下。下面的命令可在工程根目录独立执行。`download_avqa_training` 同时验证分片、官方题目与媒体映射，并保留解码后的训练音轨；完整归档传输量约 38.65 GB。三组 pilot 只需前面三个命令生成的清单。

```bash
py=/root/autodl-tmp/iemocap/qwen3-asr-capacity/venv/bin/python
root=/root/autodl-tmp/r1-aqa
"$py" -m experiments.ms_swift_aqa.prepare_avqa --root "$root/avqa" --stage pilot --train 2048 --dev 256 --seed 42 --workers 32
"$py" -m experiments.ms_swift_aqa.download_mmau "$root/mmau" --workers 16
"$py" -m experiments.ms_swift_aqa.prepare_mmau "$root/mmau"
"$py" -m experiments.ms_swift_aqa.download_avqa_training --root "$root/avqa"
```

`avqa/full/official_train.jsonl` 按清华的原始训练集重建，包含本次从原始训练集划出的 256 道开发题。若扩大训练并沿用本次开发集，应从全量清单排除这 256 道题所在的全部源视频，再做媒体去重。

三条选入训练的 AVQA 源音轨为数字静音（题号 4400、55627、57909）。三种方法均保留这三题。开发集没有这类样本。

## 运行与核验

在已有数据盘上，从工程根目录运行下列命令。`AQA_ROOT` 默认为 `/root/autodl-tmp/r1-aqa`。同一运行编号支持从已完成步骤继续。另起实验时设置新的 `AQA_OUTPUT`，更换划分时同时设置新的 `AQA_DATA`。

```bash
bash experiments/ms_swift_aqa/run_pilot.sh
```

训练清单、开发清单和完整 MMAU 清单位于 `$AQA_ROOT/data/pilot-01/`，下载记录分别位于 `$AQA_ROOT/avqa/` 和 `$AQA_ROOT/mmau/`。三组日志、adapter 与逐题预测在 `$AQA_ROOT/runs/pilot-01/`。`selection.json` 保存冻结的 checkpoint，`verified-comparison.json` 保存按同题配对复核的结果及按音轨重采样的区间。区间只描述本次训练所得的模型和这批测试音频。一次单种子 pilot 无法估计跨种子方差。

## 复现 Qwen2.5-Omni-3B

使用相同冻结清单与新的运行目录。`AQA_PYTHON` 指向已验收的 Transformers 4.52.3 环境，`AQA_CONFIG` 选择 Omni 的完整 36 层配置。

```bash
export AQA_PYTHON=/root/autodl-tmp/iemocap/swift-ser/venv-5090/bin/python
export AQA_CONFIG=experiments/ms_swift_aqa/omni3b-pilot.json
export AQA_OUTPUT=/root/autodl-tmp/r1-aqa/runs/omni3b-reproduction
export ENABLE_AUDIO_OUTPUT=0
bash experiments/ms_swift_aqa/run_pilot.sh
```

## Omni‑3B 的 ms-swift SFT 方法对照

本轮复用相同的 AVQA 与 MMAU 清单、3B 快照、学习率、有效 batch 16、开发集选模和单字母解码。串行入口先逐题复核上一轮 LoRA、DoRA、改进 TA-LoRA，再依次训练新增方法。每个新方法在干净进程中重载最佳 checkpoint，要求复现训练时全部 256 道开发题的逐题输出，随后评完整 Test-mini。[实验结果与资源开销](REPORT-omni3b-tuners.md) · [队列与各方法参数](omni3b-tuners.json) · [运行记录](/root/autodl-tmp/r1-aqa/runs/omni3b-tuners-01/REPORT.md)

```bash
bash experiments/ms_swift_aqa/run_tuners.sh
```

默认结果目录为 `/root/autodl-tmp/r1-aqa/runs/omni3b-tuners-01`。`status.json` 给出当前方法、阶段和日志，`results/` 保存每种已验证方法的两种分数、显存、用时与配对区间，`failures/` 保存尚未通过的阶段及原始日志。设置 `AQA_OUTPUT` 可以使用新的输出目录。每个方法的最终配置在 `configs/`，训练时开发预测、选中的权重哈希、重载后的逐题输出都在对应方法的目录中。

rsLoRA 和对照 LoRA 共享 A/B 初值，rsLoRA 取 alpha 除以 rank 的平方根。VeRA、BoNE、BOFT、AdaLoRA 和 FourierFT 的目标与 LoRA 相同，覆盖 36 层的七个投影。瓶颈 Adapter 作用于 36 层的 MLP 输出，LLaMA-Pro 增加一个解码块，ReFT 配置为干预 36 层隐藏状态。原生 `lora_llm` 使语言 LoRA 和多模态塔可训练，AVQA 前向只使用音频塔与语言模型。`full` 遵循 Swift 的多模态冻结默认值，更新整个 Thinker 语言模型和输出头。原生 LongLoRA 的 Llama 架构检查也纳入兼容性验证。不同参数规模和训练范围须与准确率一起比较。[ms-swift 参数说明](https://swift.readthedocs.io/en/v4.6/Instruction/Command-line-parameters.html)

物理 batch、梯度重算和 fused AdamW 的显存设置见队列配置。训练遇到 CUDA OOM 时按预设顺序减小物理 batch 并等比例增加梯度累积，所有运行保留相同样本与有效 batch。各方法的协议在对应测试前固定；这次扩展沿用了此前已公布过成绩的 Test-mini，排行属于单种子探索性比较。

本地固定的 PEFT 0.15.2 需要若干适配。VeRA 保留完整的目标正则并在重载时恢复共享随机矩阵。BoNE 显式处理 BF16 输入和 FP32 可训练块。BOFT 用索引排列计算蝶形分块旋转，FourierFT 用稀疏频率计算同一傅里叶更新；CPU 测试对照原稠密公式的输出和梯度。AdaLoRA 存储带零奇异值的完整形状因子及 rank 分配器状态，以精确重载。ReFT 通过单独安装的 PyReFT 0.1.0、PyVene 0.1.8 在实际语音嵌入和生成提示上施加干预。吞吐和显存数值对应本工程的这些实现。[PEFT 适配与测试](peft_suite.py) · [多模态结构适配](structural_plugin.py)

本机 ReFT 的隔离依赖位于 `/root/autodl-tmp/r1-aqa/structural-deps/pyreft-0.1.0`。复用相同 Swift 环境时，可用 `AQA_REFT_DEPS` 指定安装这两个固定版本的目录。各方法都使用同一份 processor。

IA³ 使用 Swift 插件接口和原生 PEFT 缩放向量。Omni 注册信息缺少示例插件所需的 MLP 路径，实验显式选择 36 层的七个语言投影，在注意力投影输出和 MLP 投影输入上缩放。补入 IA³ 时使用 `--extend-plan --only ia3`，运行器保存扩展前的配置并核对已有方法与共用协议。[插件注册表与覆盖范围](/root/autodl-tmp/r1-aqa/runs/omni3b-tuners-01/compatibility/tuner-coverage.json)

同一套提示和评分也用于未微调的基座。复现时，对新的运行目录执行下列命令，使用与训练相同的离线 Python 环境。开发和测试的逐题输出、显存、模型版本和清单哈希保存在该目录的 `frozen-base/` 中。

```bash
"$AQA_PYTHON" -m experiments.ms_swift_aqa.evaluate_frozen --run "$AQA_OUTPUT"
```

FourierFT 的频点位置种子为 777，每个投影 2,000 个频率，缩放 300。当前 Swift 工厂沿用 PEFT 0.15.2 的 `FourierFTConfig.init_weights=False`，频谱系数按标准正态分布初始化。它与 LoRA 载入相同基座，第一次前向含有非零的傅里叶增量。报告其原生默认设置，解释分数时应考虑这一差别。ReFT 使用零权重衰减，关闭语言层梯度重算，以保留前向干预。

## 在 MMAR 上评测已有 checkpoint

复用本地 `/root/autodl-tmp/MMAR/MMAR-meta.jsonl`、`audio/` 和 AVQA 开发集选出的 checkpoint。启动脚本按音频文件名及内容校验 1,000 段 WAV，保留官方题序和选项顺序。模型输入为音频、题干和选项。解码时取每题合法字母中得分最高的一个，在评分时把该字母对应的原选项文字写入 `answer_prediction`。官方元数据中 4 道题的空选项按原样保留。对应的公开基准是 **MMAR Accuracy**。MMAR-Rubrics 另外需要推理文本和评审模型。[官方评测说明](https://github.com/ddlBoJack/MMAR/tree/3bce090a967db576c8ad433b290a8da582d6d2a0#test-your-model-)

```bash
bash experiments/ms_swift_aqa/run_mmar.sh
```

`MMAR_DATA`、`MMAR_OUTPUT`、`AQA_SOURCE_RUN` 和 `AQA_PYTHON` 可分别指定数据集、输出目录、已有实验及解释器。脚本在一块 GPU 上依次运行各组，并用固定版本的官方程序复算全体、七种模态组合和各任务类别的准确率。每个方法目录包含逐题预测、可直接传给官方程序的 JSONL、原始评分日志和显存计时。[本次 MMAR 结果](/root/autodl-tmp/MMAR/runs/omni3b-avqa-01/REPORT.md) · [原始输出](/root/autodl-tmp/MMAR/runs/omni3b-avqa-01) · [统计代码](summarize_mmar.py)

## 验证 TA 音频传输分支的贡献

这组对照固定 Qwen2.5-Omni-3B、单张 RTX 5090 和原 AVQA 的 2,048 个训练样本。每组只用原来的 256 题验证集选 checkpoint。两个训练种子各自配对，所有候选仅适配 36 层 O 投影，LoRA rank 8 与 TA rank 8 共享普通 A/B 初始化。容量对照使用 LoRA rank 12，使其与 TA 的可训练参数只差 288 个 gate 标量。训练和干净重载都使用已有 Swift 入口。

新 TA 配置项 `adapter.transport_weighting` 可选 `uniform` 和 `attention_overlap`。后者按每个注意力头分配给可见音频和文本的概率质量乘以 `4 * m_audio * m_text`，两种模态都有权重时才保留较大的传输增量。使用该选项的权重保存为格式 7。评测时向 `experiments.ms_swift_aqa.transfer_eval` 传入 `--no-transport` 可在干净重载后，仅在内存中置零 gate 和独立传输输出矩阵。逐题预测、原 checkpoint 哈希与消融记录都随分数保存。

实验在环境声、音乐、语音的 261 题上复核两枚种子，并对预先固定的种子 42 运行完整 MMAR 官方答案评分。训练、清单、独立数据和评分器可由汇总命令重新校验。

```bash
"$AQA_PYTHON" -m experiments.ms_swift_aqa.summarize_transfer --out /root/autodl-tmp/MMAR/runs/ta-transfer-01/matched
```

[实验协议](/root/autodl-tmp/MMAR/runs/ta-transfer-01/matched/protocol.json) · [参数预算](/root/autodl-tmp/MMAR/runs/ta-transfer-01/matched/capacity-protocol.json) · [逐组训练命令](/root/autodl-tmp/MMAR/runs/ta-transfer-01/matched/train.sh) · [运行状态](/root/autodl-tmp/MMAR/runs/ta-transfer-01/matched/status.json)
