# AVQA → MMAU Test-mini 实验结果

**改进 TA-LoRA 在 MMAU 的 1,000 题上答对 639 题，DoRA 答对 604 题，普通 LoRA 答对 640 题。** 三组都在 AVQA 的 256 道开发题上达到 186 题正确。实验使用 Qwen3-ASR-0.6B、seed 42 和 2,048 道 AVQA 训练题，三组独立从同一基座做 SFT。[配置与复现](README.md) · [逐题复核结果](/root/autodl-tmp/r1-aqa/runs/pilot-01/verified-comparison.json)

| 方法 | 最佳步 / 实际更新数 | 可训练参数 | 开发准确率 | MMAU 严格准确率 | 官方文本评分¹ | 环境声（333） | 音乐（334） | 语音（333） |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LoRA | 128 / 224 | 5,046,272 | 72.66% | **64.0%** | **64.3%** | 234 | 203 | 203 |
| DoRA | 256 / 256 | 5,390,336 | 72.66% | 60.4% | 60.6% | 225 | 184 | 195 |
| 改进 TA-LoRA | 128 / 224 | 5,275,872 | 72.66% | 63.9% | 64.2% | **236** | **205** | 198 |

¹ 对每题合法选项做一次受限解码，再把选中的原选项文字交给固定版本的 MMAU 官方评分程序。官方词集合规则会给少量不同文本打相同的分。每组的 `reload-test/official.log` 都来自完整官方脚本。官方原 R1-AQA 使用 Qwen2-Audio-7B-Instruct、约 3.8 万条后训练样本和 GRPO，并且生成文本答案；本实验的任务划分与原报告相近，模型、训练量、解码方式和基准版本均有差异。[R1-AQA 原仓库](https://github.com/xiaomi-research/r1-aqa) · [MMAU 评分器](https://github.com/Sakshi113/MMAU/blob/110127f54c0dfba3faa5ec9feee4a7e4148679c5/evaluation.py)

按 904 种不同音轨的 PCM 指纹配对重采样 10,000 次，TA 相对 DoRA 的准确率差为 **+3.50 个百分点**，95% 百分位区间为 **[+1.09, +5.89]**。95 道题只有 TA 答对，60 道只有 DoRA 答对。TA 相对 LoRA 为 **−0.10 个百分点**，区间为 **[−1.70, +1.50]**；两者各自独对 35 道和 36 道。这些区间衡量固定 checkpoint 在这批测试音轨上的抽样波动，训练随机性的影响需要多个种子估计。[配对核算](/root/autodl-tmp/r1-aqa/runs/pilot-01/verified-comparison.json)

## 模型是否使用了音频

对每个最佳 checkpoint 保持同一组 AVQA 开发题、选项、答案及录音集合，仅将每道题的录音交换为另一个源视频的录音。LoRA、DoRA、TA 的准确率分别从 72.66% 降到 36.33%、40.23%、37.11%。这一对照表明正确的音频与题目配对会影响三种模型的判别；无法从这个结果单独归因到 TA 分支。[开发集对照结果](/root/autodl-tmp/r1-aqa/runs/pilot-01/diagnostics/swapped-dev-audio/results.json)

## 训练成本与数据

RTX 5090 单卡。三组的训练计时分别为 149.5 秒、234.4 秒和 176.7 秒（含训练 checkpoint I/O，扣除开发集评分时间）。训练阶段 PyTorch 峰值 allocated 分别为 4.54、4.55 和 4.55 GiB。DoRA 执行了更多更新，时间同时包含算法与早停步数差异。[原始运行目录](/root/autodl-tmp/r1-aqa/runs/pilot-01)

本次训练和开发来自官方 AVQA `train_qa.json` 的源视频互斥子集，测试来自完整 MMAU Test-mini。精确 PCM 检查后跨划分没有相同波形，同一音频的多个测试问题在区间计算时一起采样。训练中有三段两份来源都解码为数字静音，所有方法均使用相同清单。这些检查无法排除不同剪辑或转码的近重复，也无法审计基座预训练语料中的重叠。[清单与音频指纹](/root/autodl-tmp/r1-aqa/data/pilot-01/audit.json) · [AVQA 来源核验](/root/autodl-tmp/r1-aqa/avqa/pilot/provenance.json) · [MMAU 来源核验](/root/autodl-tmp/r1-aqa/mmau/provenance.json) · [完整 AVQA 音频归档状态](/root/autodl-tmp/r1-aqa/avqa/full/progress.json)


<!-- avqa-official-archive-v1 -->
## AVQA 全量训练数据归档

已核对官方 40,425 道训练题。可用 40,425 道，对应 40,242 条独立音频片段。另有 0 道题（0 条片段）缺少音频。已核对冻结的训练、开发和测试清单哈希。[全量可用清单](/root/autodl-tmp/r1-aqa/avqa/full/official_train.jsonl) · [缺失题号](/root/autodl-tmp/r1-aqa/avqa/full/unavailable-questions.jsonl) · [归档校验](/root/autodl-tmp/r1-aqa/archive-verification.json)
