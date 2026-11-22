# IEMOCAP 四分类语音情绪识别
与 ms_swift_librispeech 共用原生 ms-swift 训练入口、Qwen2.5-Omni-3B、匹配初始化的末四层 O 投影 LoRA/TA-LoRA。输入仅为录音和四选一提示，assistant 监督为字母。按共识标注选取 A angry、B happy/excited、C sad、D neutral；其余标签排除。Session 1–3 训练，4 开发，5 只做最终测试；即兴和剧本录音均使用。全程固定 seed 42、batch 2、累积 4、最多一轮；每 100 步及轮末按开发集 UAR 选模，首次持平或下降早停。评分取四个合法单 token 中概率最高者，输出 UAR、准确率、逐类召回和混淆矩阵（行是真值，列是预测，ABCD 顺序）。两组都选定后干净重载，完整复现开发预测，再评估 Session 5。同词异情绪子集仅作事后诊断，公布支持数及准确率。

在工程根目录执行（新运行使用新输出目录）：

```bash
export HF_HUB_OFFLINE=1 ENABLE_AUDIO_OUTPUT=0 OMP_NUM_THREADS=2
py=/root/autodl-tmp/ms-swift-librispeech/venv/bin/python
root=/root/autodl-tmp/iemocap/swift-ser
cfg=experiments/ms_swift_iemocap/config.json
"$py" -m experiments.ms_swift_iemocap.prepare "$cfg" "$root/data"
"$py" -m experiments.ms_swift_iemocap.run --config "$cfg" --data "$root/data" --output "$root/pair-01"
```

已有清单直接复用；原始 ZIP 保存在数据盘。splits.json 记录清单摘要、每类数量、说话人和排除的标签。各组训练日志和 checkpoint 写入 pair-01，两组完成后结果见 comparison.json。

## Qwen3-ASR-0.6B 的普通 LoRA

`qwen3-asr-0.6b-lora-b32.json` 使用 28 层语言解码器的 Q/K/V/O 和 MLP gate/up/down 投影，rank 8、alpha 16、dropout 0。物理 batch 为 32，累积 1，开启非重入梯度检查点。每 25 次更新及轮末按 Session 4 的 UAR 选模，首次持平或下降即早停。从干净基座重载最佳 adapter 后复核 Session 4 的逐条预测。Session 5 留待最终对比。

在工程根目录运行，输出路径须为新的实验目录。5090 上使用可扩展显存段以适应批间的音频长度变化。

```bash
export HF_HUB_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
py=/root/autodl-tmp/iemocap/qwen3-asr-capacity/venv/bin/python
root=/root/autodl-tmp/iemocap/swift-ser
"$py" -m experiments.ms_swift_iemocap.run \
  --config experiments/ms_swift_iemocap/qwen3-asr-0.6b-lora-b32.json \
  --data "$root/data" --output "$root/qwen3-lora-b32-new"
```

训练中断并写出 `failure.json` 后，原命令追加 `--resume-run` 可从该目录最近的完整训练 checkpoint 恢复。各验证点见 `lora/dev/step-*/metrics.json`，最终选择见 `dev-selection.json`。

Qwen3-ASR-0.6B 的 LoRA、DoRA 和 TA-LoRA 对比结果、开发集消融与复现命令见 [TA-LoRA 查询位置与门控实验](ta-query-study.md)。
