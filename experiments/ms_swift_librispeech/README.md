# ms-swift · LibriSpeech ASR · Qwen2.5-Omni-3B

本实验用 ms-swift 原生 SFT 和 PEFT LoRA，与同一份 Swift 源码交付的外部 TA-LoRA tuner 做单种子配对比较。语音输入经原生 Omni 模板、processor 和 Thinker 的语音编码器；输出为转写文本。模型、数据、超参数、方法顺序统一由 [config.json](config.json) 管理。运行状态和训练日志位于 [current-run](current-run/)，该链接指向本机数据盘上的正式实验。最终两组完成后，该目录生成 RESULTS.md 和 comparison.json。

## 模型、环境及数据

- 官方模型为 Qwen/Qwen2.5-Omni-3B，commit 为 `f75b40e3da2003cdd6e1829b1f420ca70797c34e`。运行使用 `/root/autodl-tmp/west-preflight/model`。18 个本地文件已按官方对象身份核验。[快照证据](evidence/verified_snapshot.json)
- 已实际加载本地 ms-swift 4.7.0.dev0 交付源码，其中注册了 3B 模型、音频模板和外部 TA tuner。框架包取自这份源码的独立安装副本。关闭音频输出，两组都用同一 text-only Thinker 生成入口。安装、包版本和已知环境差异见[环境说明](environment/README.md)。
- 音频仍在 `/root/autodl-tmp/west-preflight/librispeech/LibriSpeech`。数据转换仅写入带绝对音频路径的 Swift JSONL，每条一个 <audio> 用户提示和一个 assistant 转写。训练、开发、测试采用 train-clean-100（28,539）、dev-clean（2,703）、test-clean（2,620）。转换核对 SHA、音频路径、ID、说话人、音频格式与处理器长度上限。真实波形的扩展 token 与语音特征、padding、仅 assistant 转写及结束标记的监督均已核验。[输入审查](evidence/inputs.json)
- [plugin.py](plugin.py) 和 [swift_asr_ta_runtime.py](swift_asr_ta_runtime.py) 来自 ms-swift 的 `examples/train/multimodal/omni/ta_lora`。新 TA checkpoint 使用与 PEFT 匹配的残差运算和显式目标层，历史 format 1 checkpoint 可按原计算路径读取。原版 LoRA 直接通过 Swift 的 PEFT tuner 安装。为做配对实验，从同一份专用 CPU RNG 状态初始化两组 A/B；保存哈希，并在正式评分前比较。

## 训练协议

两组仅训练 Thinker 最后四层（32–35）的 O 投影适配器；基础 LLM、语音编码器和视觉塔冻结。rank 8、alpha 16、dropout 0；TA 每层增加 8 个 gate。两组都使用 eager attention、BF16 基座、FP32 适配器，单卡、无 packing、关闭梯度检查点。batch 2，累积 4，AdamW，固定学习率 0.0002，seed/data_seed 42，最多完整一轮。前 8 个原生训练 microbatch 的 token、监督和帧 mask 计算摘要，供数据顺序核验。TA 所用语音分区由真实展开的占位符决定，解码时按 KV cache 追加文本位置。[适配器及 GPU 检验](validate_gpu.py)

每 892 次优化器更新以及轮末在**完整 dev-clean** 生成转写。以最低 corpus WER 选模，平手取早，第一次不改善便早停。保留最佳和近期 checkpoint，以及恢复所需的 optimizer、scheduler、Trainer state 和 RNG。两组选择全部固定后，各自在新进程从本地干净 3B 快照装入最佳 adapter，逐句复现首个开发集 batch 的预测，然后才解码 test-clean。

两组均使用同一模板与提示、greedy、单 beam、最多 128 个新 token、相同 EOS。逐行核对生成序列的 prompt 前缀，去掉整个 padded prompt 和 EOS 后解码。归一化使用 Unicode NFKC、casefold、统一撇号，保留词内撇号。WER 为各句英文词编辑距离之和除以参考词总数，同时保存达到上限且无 EOS 的比例、全部预测和参考。训练时间为 Trainer 内的训练墙钟时间（含保存、扣除开发集生成），验证或测试另行计时。peak allocated/reserved 来自真实 CUDA 计数。若早停步数不同，应同时看总时间、优化器步数及每步用时。

## 复跑

以下命令从工程根目录执行。当前机器的环境和正式清单位于 `/root/autodl-tmp/ms-swift-librispeech`。首次准备数据时用一个新目录，后续运行复用已生成的清单。

```bash
p=experiments/ms_swift_librispeech
rt=/root/autodl-tmp/ms-swift-librispeech
# 新机器先按 environment/README.md 创建兼容环境并设置本地路径。
bash "$p/run.sh" prepare --config "$p/config.json" --output "$rt/data-new"
bash "$p/run.sh" compare --config "$p/config.json" --data "$rt/data-new" --output "$rt/runs/pair-new"
```

单独运行时用同一个训练入口，通过 `--method lora` 或 `--method ta-lora` 选择方法，两者共享 JSON 中的训练和解码参数。

```bash
export ENABLE_AUDIO_OUTPUT=0 HF_HUB_OFFLINE=1
"$rt/venv/bin/python" -m experiments.ms_swift_librispeech.train \
  --method ta-lora --config "$p/config.json" --train "$rt/data/train.jsonl" \
  --dev "$rt/data/dev-clean.jsonl" --output "$rt/runs/single-ta"
```

中断后可对同一比较目录再次执行 compare。它先校验配置和清单哈希，再从该组最后一个完整训练 checkpoint 恢复；已完成的组复用其结果。单组入口支持 `--resume CHECKPOINT`。不要并发使用同一个比较目录。

## 已完成的验收

- 真实 3B、RTX 4080 SUPER、同一 batch 的两条训练录音（14.085 秒、24.525 秒），两组初始 logits 逐位相同。两次更新后验证 A/B、TA gate 的梯度、参数范围和从干净基座重载后的 logits。峰值 allocated 为 LoRA 12.31 GiB、TA 13.02 GiB；这组数仅代表该真实 batch。[LoRA](evidence/real-3b-lora.json) · [TA-LoRA](evidence/real-3b-ta-lora.json)
- [smoke.py](smoke.py) 对两组真实 Swift Trainer 分别在第 1 步保存并退出，从 checkpoint 恢复，在第 2 步遇到持平的开发集 WER 并早停，均选中第 1 步。独立进程重载最佳模型后逐句重现预测。该检验只用了两条开发音频和 3-token 解码上限。[实际记录](evidence/swift-trainer-smoke.json)
- [tests](tests/) 检查选择、恢复后的权重与 Adam 动量、TA checkpoint 版本、配对初始化、缓存与 WER 计数。单卡 smoke 和正式成绩分别保存在数据盘的 smoke-01 和 runs/pair-01 中。
