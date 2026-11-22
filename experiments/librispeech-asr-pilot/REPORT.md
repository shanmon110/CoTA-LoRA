# LibriSpeech ASR 单种子 pilot

使用已验收的 Qwen2.5-Omni-3B 本地快照，从干净基座独立训练 O-LoRA、TA-LoRA、Q/V-LoRA、TopLoRA、BoRA、DoRA、EVA。七组依次使用同一张 RTX 4080 SUPER。运行目录为 `/root/autodl-tmp/west-preflight/librispeech-asr-pilot-s42-01`。每组评估完成后，驱动在 [status.json](/root/autodl-tmp/west-preflight/librispeech-asr-pilot-s42-01/status.json) 写入完整 dev-clean 结果。每个方法目录保留 adapter、训练日志、分阶段显存与耗时、逐批录音 ID 和评估 JSON。[实际命令](/root/autodl-tmp/west-preflight/librispeech-asr-pilot-s42-01/commands.txt)

## 比较协议

- 在 train-clean-100 的 28,539 条清单上按 seed 42 打乱，batch 1、梯度累积 8、100 次 AdamW 更新，各方法看同序 800 条 SFT 样本。EVA 在更新前单独使用 128 条同任务训练录音做校准，其成本单列。驱动逐组比较实际训练样本的 ID 顺序。
- BF16 基座、FP32 adapter，最后四层，学习率 0.0002，非重入梯度检查点。O 目标名义 rank 8、alpha 16。Q/V 用实际 GQA 投影宽度匹配预算，参考 rank 8、实际 rank 5、alpha 10。各方法缩放都是 2。BoRA 使用两个块，EVA 保留名义缩放。
- 每次评估在新进程恢复 adapter 和固定基座。依同一顺序评估全部 2,703 条 dev-clean 录音，batch 1、greedy、最多新增 128 token，报告语料级 WER、词错误数、参考词数、未生成 EOS 而触及长度上限的数量及比例。只使用开发集，单个种子不提供跨种子显著性证据。

## Review 修复与适用范围

本轮补上 BoRA 的高分块溢出诊断。对于可能溢出的配置，实际增益超出 FP32 表示范围会在块乘前明确报错；有限增益照原公式计算。新增回归同时覆盖同一高分块配置的有效状态和溢出状态。默认 blocks=2、rank=8 保持原计算路径，无新增 GPU 同步。[修复前复现](bora-before.log) · [修复后完整测试](cpu-tests.log)

沿用预检中已通过回归的 BF16 残差精度策略、旧 checkpoint 语义、TA 重算音频布局、混合精度训练和清单预算。训练前校验本地快照内容摘要，评估重载时再次校验。单音频上限 32.645 秒，批内 3,265 个有效 mel 帧，展开输入最多 883 token。两个 split 过滤数均为零。全词表 logits 和 eager attention 的显存开销仍随输入规模增长，本实验仅使用已实测通过的输入预算。[预检报告](../preflight/REPORT.md) · [输入清单与 SHA-256](../preflight/asr-plan/summary.json)
