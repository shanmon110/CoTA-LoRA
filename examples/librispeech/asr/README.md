# LibriSpeech ASR

本目录支持 TouchASU（Qwen3-1.7B + FireRed）和 Qwen2.5-Omni-7B Thinker 两种 LibriSpeech ASR 实验。

## TouchASU 原版 LoRA 与七种适配器

原版 LoRA 使用 [AISHELL 模型配置](../../aishell/asr/conf/qwen3-1.7b-lora_firered.json) 和 [WEST 训练配置](../../conf/training/touch_asu_asr.json)。在已有原版训练之后，[比较配置](conf/touchasu_comparison.json) 依次安排 TA-LoRA、O-LoRA、Q/V-LoRA、TopLoRA、BoRA、DoRA、EVA。各组通过相同的 AISHELL `run.sh --stage train`，从相同的 Qwen3 和 FireRed 基座独立初始化，训练语音投影层及各自的适配器。使用单卡 BF16、ZeRO-2、FlashAttention 2、8,192-token 打包、梯度累积 4、5,000 步。后续各组只写第 2,500、5,000 步的 checkpoint，控制磁盘占用。

在工程根目录、已配置好 FlashAttention、DeepSpeed 及 torchrun 的同一环境下运行以下入口。`BASELINE` 指向原版运行目录，含本地 `touch_asu.json`、带训练配置与清单 SHA-256 的 `inputs.json`、`data/train.jsonl`、`checkpoints/` 和由启动脚本写入的 `exit.code`。正在训练时可以提前排队。数据准备会对照官方转写核实三份清单的条目与说话人互斥性。

```bash
python -m examples.librispeech.asr.local.touchasu_wer prepare \
    --corpus "$LIBRISPEECH" --train "$BASELINE/data/train.jsonl" --output "$EVAL_DATA"
python -m examples.librispeech.asr.local.touchasu_compare init \
    --baseline "$BASELINE" --data "$EVAL_DATA" --output "$COMPARISON"
python -m examples.librispeech.asr.local.touchasu_compare run --gpu-check --output "$COMPARISON"
```

队列等原版训练成功退出，再校验 Trainer 步数并逐组训练。同一组的两个候选在完整 dev-clean 上按英文词错误数选模，平手取较早步数。只用选中的模型跑完整 test-clean，greedy 最多生成 50 个新 token；指标含逐句 ID、参考词数、词错误数及达到上限仍无 EOS 的条数。各组的 `eval/` 保留结果，`status.json` 记录阶段或失败信息，`comparison.json` 在原版与 TA 都完成测试后写入，并随其余组更新。

原版 PEFT LoRA 在 28 层的七类投影上使用 r=64 和 dropout 0.05；共享的 TA 和 O 预设只作用于最后四层注意力输出，r=8、dropout=0。原版与 TA 的 WER 比较涵盖整套方法设置。O 与 TA 的位置、rank、缩放和随机初始化设置一致，适合考察模态分支。现有原版进程在项目训练入口设定 Trainer 种子之前已初始化投影层，无法追溯其精确随机初值。

## Omni 训练入口

与 [AISHELL ASR](../../aishell/asr/run.sh) 一样，[run.sh](run.sh) 按 data、train、decode 分阶段，读取 WEST 的命令行选项，直接调用 `west/bin/train.py`、`west/bin/decode.py` 和 `tools/compute_wer.py`。

## 运行

以下命令从工程根目录执行。已有的 7B 环境使用 Transformers 4.52.3。训练阶段使用一张支持 BF16 的 GPU；请为新实验指定新目录。

```bash
cd examples/librispeech/asr
export PYTHON=/root/.venvs/west-omni-preflight/bin/python
export CUDA_VISIBLE_DEVICES=0
run_dir=/root/autodl-tmp/west-preflight/librispeech-7b-west-01

bash run.sh --stage data
bash run.sh --stage train  --dir "$run_dir"
bash run.sh --stage decode --dir "$run_dir"
```

也可用 `--stage all --dir "$run_dir"` 在新目录顺序运行三个阶段。每一组的完整命令写入 `commands.txt`，运行状态和各组结果写入 `status.json`。

## 数据和配置

[模型配置](conf/qwen2.5-omni-7b.json) 保存 WEST 模型类型、7B 快照位置、基座与 processor 的内容摘要，以及本机已验收的输入清单位置。[生成配置](conf/generation_config.json) 固定 greedy 解码、单 beam、128 个新 token。可通过 `--model-conf` 和 `--decode-conf` 指定配置文件。训练开始时，模型、生成参数、训练策略、各方法的配置和四份 WEST 清单都会保存在实验目录，并记录内容摘要。清单使用 `wav` 和 `txt`，并保留录音 `id` 和 `speaker`，供核验抽样顺序和数据划分。

默认 data 阶段在 CPU 上验证已有清单。从已解压的语料创建新索引时，`--corpus` 指向包含四个 LibriSpeech 子集的目录，`--data-dir` 指定新索引目录。随后在 train 或 all 阶段传入该索引。独立 decode 阶段读取本次实验保存的测试清单和已选 adapter。

```bash
bash run.sh --stage data --corpus /data/LibriSpeech --data-dir /data/west-librispeech-index
```

| 用途 | 子集 | 条数 |
|---|---|---:|
| 训练 | train-clean-100 | 28,539 |
| 选模与早停 | dev-clean | 2,703 |
| 最终测试 | test-clean / test-other | 2,620 / 2,939 |

## 训练和评分

[比较配置](conf/comparison.json) 指定训练策略和按运行顺序排列的方法文件，路径相对于该比较配置。[训练策略](conf/training.json) 设置优化器、学习率、累积步数和验证间隔。rank、alpha 和专用参数来自[共享 adapter 配置](../../conf/adapters/README.md)。默认按 O-LoRA、TA-LoRA、Q/V-LoRA、TopLoRA、BoRA、DoRA、EVA 的顺序从干净基座独立训练。每组 batch size 2，梯度累积 4，最多一轮。每 892 次更新在完整 dev-clean 上生成转写，WER 首次持平或上升即停止，保留较早的最佳权重。

从本 recipe 目录只跑一个方法时可指定共享配置。用 `--training-config FILE` 可覆盖训练策略中的字段，用 `--comparison-conf FILE` 可更改方法集合和顺序。验证和测试共用生成上限。

```bash
bash run.sh --stage train --adapter-config ../../conf/adapters/ta-lora.json \
  --dir /root/autodl-tmp/west-preflight/librispeech-ta-new
```

WEST 训练入口用 `model_type=qwen2_5_omni_adapter` 选择基于 Transformers Trainer 的 [Omni 训练器](../../omni_talora/west_trainer.py)。Trainer 管理优化器更新和学习率调度；Omni collator 处理不同长度的音频，训练步按本次更新中所有有效答案 token 归一化。此协议使用 BF16 基座、FP32 adapter、AdamW 和固定学习率 0.0002。为保留逐条音频的模态与因果 mask，使用单进程 torchrun 和 `pack_size=0`。AISHELL 示例中的 ZeRO-2、多卡分片和 sequence packing 不在这份 Omni recipe 的支持范围内。

本次配置列出的所有方法完成 dev 选模后，`dev-selection.json` 记录最佳 checkpoint 及内容摘要。decode 阶段从干净基座重载，按测试清单顺序输出每条转写，并以 `tools/compute_wer.py --char=0` 计算英文词错误率。每种方法目录分别保存 `test-clean.wer`、`test-other.wer`、评分用参考与预测 JSONL、含截断计数的指标 JSON。脚本校验 WEST 计数与生成验证使用的词级评分一致。
