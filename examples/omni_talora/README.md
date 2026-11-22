# 在 WEST 上训练 TA-LoRA：Qwen2.5-Omni-3B，单任务语音理解

这是独立的监督微调 recipe，不需要 WEST 的 GRPO、WeNet、vLLM 或 TouchASU。目标是从同一个官方基座分别训练 ASR 和 SER；一个训练命令只读一份单任务清单、写一套 adapter。只生成文本，不实例化 Talker 或 token2wav。其他语音任务可以复用 attention 层，但本版只提供这两种数据协议和指标。

## 方法和公平基线

在 Thinker 文本解码器的最后 `--last-n` 层，令 O 为多头 attention 拼接后、进入 `o_proj` 前的值。对每个有效非音频 query/head，沿用该头经过 mRoPE、因果和 padding 掩码的分数，分别在可见音频 key 和其余可见 key 内做 softmax 并加权聚合 value。两组均值的差拼接为 D。任一组为空、query 为音频或 padding 时，D=0。只有 processor **展开后的音频占位 token** 属于音频组，音频边界和指令等属于非音频组。

```text
TA-LoRA: Y = o_proj(O) + (alpha/r) · [(O B) + (D B) ⊙ g] A
O-LoRA:  Y = o_proj(O) + (alpha/r) · (O B) A
```

普通分支与条件分支共享低秩矩阵 B、A；每层只有一个静态可学习向量 g（长度 r，任务内所有样本共享）。A 和 g 零初始化，B 随机初始化，因此初始输出等于冻结模型；相同 B、A 且 g=0 时严格是普通 **O 投影** LoRA，不代表覆盖 Q/K/V 上的 LoRA。第一次更新主要改变 A，不应要求第一次反传 g 已有非零梯度。默认 r=8、alpha=16、最后四层（3B 层号 32–35）：O-LoRA 131,072 个可训练参数，TA-LoRA 131,104 个。两种模式均走 eager attention，便于公平计时；非零 g 的分支依赖输入，不能静态合并进原权重。

## 对比方法与开关

下列开关用于**同一个** `examples.omni_talora.train` 入口；`evaluate` 从 adapter 配置恢复方法，无需重复传参。每个任务、方法、种子使用独立的新目录。所有方法冻结基座，默认目标是最后四层；除 TA 外不计算音频/非音频对比。旧版 `--mode ta`、`--mode lora` 和它们的 adapter 文件格式不变。

| 方法 | 训练开关 | 目标 | 训练前校准 | 实现依据 |
|---|---|---|---|---|
| O-LoRA / TA-LoRA | `--mode lora` / `--mode ta` | O | 无 | 本项目原版实现 |
| TopLoRA | `--mode toplora` | O | 无 | [作者代码](https://github.com/Leopold1423/toplora-neurips25)，论文式 6–7 |
| BoRA | `--mode bora --bora-blocks 2` | O | 无 | [作者代码](https://github.com/Leopold1423/bora-iclr26)，论文式 4–5、8 |
| DoRA | `--mode dora` | O | 无 | [原论文](https://proceedings.mlr.press/v235/liu24bn.html)、[作者实现](https://github.com/NVlabs/DoRA) |
| EVA | `--mode eva --eva-samples 128 --eva-rho 2 --eva-seed 42` | O，可分配零秩 | 仅本次训练集 | [作者代码](https://github.com/ml-jku/EVA)、[PEFT 实现](https://github.com/huggingface/peft/blob/v0.15.2/src/peft/tuners/lora/eva.py) |
| Q/V-LoRA | `--mode lora --target-set qv --match-ta-budget --rank 8` | Q、V | 无 | 普通 LoRA 的目标位置对照，不是独立的新算法 |

- **TopLoRA**：在 O 的输入上额外做 Kaiming 初始化的投影，逐 token 使用带可学习权重的 RMSNorm 和 exp 调制秩通道；不是静态 gate。同 rank 下它比 O-LoRA 多参数。
- **BoRA**：把输入和输出的 feature 维各等分成 b 块，各块对使用可训练的 b×b×r 缩放 logits，按全局平均绝对值归一化后取 exp。b 默认为本 recipe 的 2，必须同时整除输入和输出宽度。分块不跨音频时间步。
- **DoRA**：幅值是每个输出通道的权重行范数，方向是 W 加缩放后的 BA；每次前向重新计算方向范数并停止该范数的梯度，bias 不缩放。未实现权重合并或缓存方向范数。
- **EVA**：用独立随机种子从本任务训练 manifest 无放回选取最多指定数量的音频。默认 `--eva-token-mask labels` 对 O 投影输入收集 attention_mask 有效且 labels 非 -100 的行（答案和 EOS，非位移后的预测位置）；可显式选 `nonpad` 用全部非 padding 位置，包括音频和提示。所有层在同一批数据上收集。采用中心化的分块增量 SVD，以层内解释方差比做跨层秩分配，总预算为层数 × 名义 rank，每层至多约 rho × rank；零秩层不注入。选中右奇异向量初始化 down，up 为零；重分配后保持名义 alpha/rank 的缩放。保存实际校准样本数、seed、掩码策略、rho、逐层 rank 和训练后的权重，重载不再校准。与论文实验的区别：这里只比较 O 投影、每次遍历有限训练子集，不做迭代至收敛或自动调参。短标签 SER 的监督位置较少，若样本不足，增大 `--eva-samples`，不要使用开发集补足。校准成本应单独计入总训练时间。
- **Q/V 预算**：加 `--match-ta-budget` 时，`--rank` 指参考 TA 的 rank，程序按当前基座真实的 Q/V 宽度选使参数差绝对值最小的正整数统一 rank；不加时，`--rank` 就是实际 Q/V rank。按官方 3B 配置，最后四层 TA-r8 为 131,104 个参数；每层 Q 为 2048→2048、V 为 2048→256，因此 Q/V-r5 为 128,000 个，少 3,104 个。最终以启动日志打印的目标、rank 和可训练参数量为准。

在仓库根目录，准备好下文的 manifests 后，以下 Bash 示例可逐个运行（两项任务分别训练，不是联合训练）：

```bash
run_one() {
  local task="$1" tag="$2"
  shift 2
  python -m examples.omni_talora.train --task "$task" --manifest "data/$task/train.jsonl" \
    --output "runs/$task-$tag-s42" --last-n 4 --rank 8 --alpha 16 --seed 42 \
    --gradient-checkpointing "$@" &&
  python -m examples.omni_talora.evaluate --adapter "runs/$task-$tag-s42" --manifest "data/$task/dev.jsonl"
}
run_one asr top --mode toplora
run_one asr block --mode bora --bora-blocks 2
run_one asr dora --mode dora
run_one asr eva --mode eva --eva-samples 128 --eva-rho 2 --eva-token-mask labels
run_one asr qv --mode lora --target-set qv --rank 8 --match-ta-budget --alpha 10
# SER 同样使用上面任一组方法参数，只把第一个参数换成 ser，例如：
run_one ser eva --mode eva --eva-samples 128 --eva-rho 2 --eva-token-mask labels
```

比较的是在同一个 Speech LLM 入口下移植的算法，不是复现作者在其他模型或数据集上的成绩。请记录实际参数量、训练预算及推理耗时；不要从公式对照测试推断 WER/UAR。

预算匹配不自动调整 alpha：例如 TA 默认 scale=16/8=2，而 Q/V-r5 使用相同 alpha 时 scale=16/5。上例为 Q/V 对照传 `--alpha 10`，使 scale=10/5=2。这仍不是完全等参数量的比较。

## 环境与安装

从 WEST 仓库根目录运行。示例环境：Linux、Python 3.11。先安装与本机 NVIDIA 驱动匹配的 PyTorch 2.6.0 和 torchvision 0.21.0 CUDA 构建，再运行：

```bash
python -m pip install -r examples/omni_talora/requirements.txt
python -m examples.omni_talora.train --help
python -m examples.omni_talora.evaluate --help
```

不要在这个最小环境安装整个 WEST 的 `requirements.txt`；既有 WEST 环境只需补齐本例依赖。使用 [官方 3B 权重](https://huggingface.co/Qwen/Qwen2.5-Omni-3B)，可传 `--model-name-or-path /path/to/complete/snapshot`。Hub 加载会记录解析后的提交；本地目录请保留不变。下载完整分片可能仍包含不用的 Talker 权重。CPU 模式使用 FP32（小规模调试），CUDA 使用 BF16 冻结基座、FP32 adapter。

## 数据

JSONL，每行一条音频；相对路径以 **清单所在目录** 为基准。训练、验证、测试分别建文件，不能混用。示例两行属于不同清单：

```json
{"audio":"../LibriSpeech/train-clean-100/19/198/19-198-0000.flac","text":"THE REFERENCE TRANSCRIPT"}
{"audio":"../IEMOCAP/Session1/sentences/wav/Ses01F_impro01/Ses01F_impro01_F000.wav","emotion":"sad"}
```

音频会转单声道、重采样到 16 kHz。训练只监督 assistant 答案和结束标记，不监督提示、音频或 padding。SER 采用 A=angry、B=happy/excited（合并）、C=sad、D=neutral。不要随机拆分 IEMOCAP 语句：例如 Session 1–3 训练、4 验证、5 测试；正式结果可轮换留出 session，再报告跨折结果。LibriSpeech 可先选 train-clean-100，独立报告 dev/test-clean、dev/test-other。

本地已经解压语料时，可以直接生成清单；工具不下载也不改动原音频、不覆盖已有清单。把下面的根目录换成实际路径：

```bash
python -m examples.omni_talora.prepare_manifest --task asr --root /data/LibriSpeech/train-clean-100 --output data/asr/train.jsonl
python -m examples.omni_talora.prepare_manifest --task asr --root /data/LibriSpeech/dev-clean --output data/asr/dev.jsonl
python -m examples.omni_talora.prepare_manifest --task ser --root /data/IEMOCAP_full_release --sessions 1,2,3 --output data/ser/train.jsonl
python -m examples.omni_talora.prepare_manifest --task ser --root /data/IEMOCAP_full_release --sessions 4 --output data/ser/dev.jsonl
python -m examples.omni_talora.prepare_manifest --task ser --root /data/IEMOCAP_full_release --sessions 5 --output data/ser/test.jsonl
```

## 跑第一组实验

先从 ASR 训练集选 2–4 条短语音写成 `data/asr/smoke.jsonl`。在 A6000 先跑一个 **两步** 流程检查；不要把 smoke 清单当正式评测集。

```bash
CUDA_VISIBLE_DEVICES=0 python -m examples.omni_talora.train --task asr --mode ta \
  --manifest data/asr/smoke.jsonl --output runs/smoke-asr-ta \
  --batch-size 1 --accumulation-steps 1 --max-steps 2 --gradient-checkpointing
CUDA_VISIBLE_DEVICES=0 python -m examples.omni_talora.evaluate \
  --adapter runs/smoke-asr-ta --manifest data/asr/smoke.jsonl --max-new-tokens 64
```

训练或评估命令可加 `--smoke-log PATH` 追加 JSONL 日志。每一阶段独立记录耗时、CUDA allocator 的峰值 allocated/reserved 字节数；每批记录音频路径、时长及去 padding 的输入 token 数。EVA 校准与训练分别计量。OOM 时阶段结束记录包含当批样本，CPU 日志的显存字段为 null。

以下四次是**独立训练**。用相同种子、样本顺序、rank、位置、步数和学习率，同时测试两个方法。100 步是启动设置，不是收敛建议。正式训练根据开发集确定预算并固定，不能用测试集调参。

```bash
for task in asr ser; do
  for mode in lora ta; do
    CUDA_VISIBLE_DEVICES=0 python -m examples.omni_talora.train \
      --task "$task" --mode "$mode" --manifest "data/$task/train.jsonl" \
      --output "runs/$task-$mode-s42" --rank 8 --alpha 16 --last-n 4 \
      --batch-size 1 --accumulation-steps 8 --max-steps 100 --lr 2e-4 --seed 42 \
      --gradient-checkpointing
    CUDA_VISIBLE_DEVICES=0 python -m examples.omni_talora.evaluate \
      --adapter "runs/$task-$mode-s42" --manifest "data/$task/dev.jsonl"
  done
done
```

`--max-steps` 统计优化器更新；可跨 epoch 累积，按有效的目标 token 数加权。不同任务可以在两张卡上各启动一个独立进程；本例没有实现 DDP，也不会把两卡显存相加。显存和速度都取决于音频和序列长度，eager attention 对序列长度是平方开销。OOM 时先用短片段、单样本及 checkpointing；梯度累积不能解决单条长音频的峰值。

训练只保存 `adapter.safetensors`、`adapter_config.json` 和训练参数；没有基座副本或恢复训练所需的优化器状态。换任务、模式、种子时使用新输出目录。评估根据 adapter 配置恢复相同基座、注入位置和任务：

- ASR：带动态 KV cache 的 greedy 解码，只评新增 token；输出语料级 WER（比例）、总词错误数和参考词数。`max_new_tokens_without_eos_count` 和 `max_new_tokens_without_eos_ratio` 分别是生成到上限仍未输出 EOS 的样本数及其占全部评估样本的比例。内置规范化不做数字展开，和其他论文比较时要对齐文本规范化。生成长度上限可按开发集设置。
- SER：在真实 assistant 前缀之后比较 A/B/C/D 四个候选 token 的 logits，输出四类宏平均召回率 UAR、准确率及各类样本数。缺类时 UAR 为 null，不把缺类当作零。

## 验证与边界

```bash
python -m pip install pytest
python -m pytest -q test/test_omni_ta*.py
# 可传本地官方 processor/tokenizer 目录（不需要 3B 权重），启用额外兼容性测试：
OMNI_PROCESSOR_DIR=/path/to/processor python -m pytest -q test/test_omni_ta*.py
```

测试含手算 attention 对照、无未来信息、空分区、梯度、padding、动态 KV cache、checkpoint 往返及真实小尺寸随机 Thinker 的音频训练/解码。不支持 sequence packing、beam search、FlashAttention/SDPA、量化训练、语音生成或零开销部署。小模型测试不能证明 3B 上的 WER/UAR 提升，也不是 48 GB / 32 GB 实测显存。

版本基线：WEST 提交 `b9a629d68bf012c9b92a901ba1ed22fbbdc3b3eb`，原 TA/O 版在本次改动前重新验证为 45 项通过。扩展版还包含：五种新增配置在真实小尺寸 Thinker 上的独立更新与重载，各自的 ASR/SER 训练和评估入口，公式梯度、混合精度、EVA 选秩和 GQA 参数预算测试。测试使用合成音频、随机权重及本地官方 processor，不代表 3B 真实任务效果。全仓另有既有测试依赖 torchaudio、librosa 和离线环境中缺少的 Qwen2-1.5B-Instruct，这些不属于本 recipe 的定向套件。
