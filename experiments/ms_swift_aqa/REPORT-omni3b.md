# Qwen2.5-Omni-3B 的 AVQA → MMAU 三方法对比

完整 MMAU Test-mini 的严格选项准确率排名为 改进 TA-LoRA 第 1（732/1,000），LoRA 第 2（724/1,000），DoRA 第 3（720/1,000）。训练使用 AVQA 2,048 道题，开发使用独立源视频的 256 道题。每组从干净基座加载最佳 adapter 后复现开发预测，再对同一批 1,000 道测试题评分。

| 3B 方法 | 开发准确率 | 测试严格准确率 | 官方文本评分 | 环境声（333） | 音乐（334） | 语音（333） |
|---|---:|---:|---:|---:|---:|---:|
| LoRA | 80.47% | **72.40%** | 72.60% | 80.78% | 65.57% | 70.87% |
| DoRA | 80.86% | **72.00%** | 72.20% | 79.58% | 64.37% | 72.07% |
| 改进 TA-LoRA | 80.47% | **73.20%** | 73.40% | 81.08% | 65.87% | 72.67% |

三组都从每道题的合法字母中解码。官方评分对选中选项的原文调用固定版本的 MMAU 文本匹配规则，重新计算的 3 份总分与官方程序日志一致。[逐题复核](/root/autodl-tmp/r1-aqa/runs/omni3b-pilot-01/verified-comparison.json) · [各组原始日志和预测](/root/autodl-tmp/r1-aqa/runs/omni3b-pilot-01)

## 配对差异

下表列出严格正确率的百分点差，以及按 904 个音频簇进行 10,000 次配对重采样的 95% 百分位区间。区间描述本次固定 checkpoint 的测试音频波动；一次种子无法估计训练种子间的波动。

| 3B 方法对（前者减后者） | 百分点差（95% 区间） |
|---|---:|
| 改进 TA-LoRA 减 LoRA | +0.80（-1.11 至 +2.77） |
| 改进 TA-LoRA 减 DoRA | +1.20（-0.40 至 +2.83） |
| LoRA 减 DoRA | +0.40（-1.39 至 +2.24） |

| 同一方法 | 0.6B | 3B | 3B 减 0.6B 百分点（95% 区间） |
|---|---:|---:|---:|
| LoRA | 64.00% | 72.40% | +8.40（+5.44 至 +11.43） |
| DoRA | 60.40% | 72.00% | +11.60（+8.57 至 +14.70） |
| 改进 TA-LoRA | 63.90% | 73.20% | +9.30（+6.11 至 +12.53） |

两轮复用相同的训练、开发和测试清单。基座、processor、语言层数（28 到 36）、显存设置和所选 checkpoint 随实验变化。两轮分别使用 Transformers 4.57.6 和 4.52.3，因此跨模型的变化也包含实现差异。[跨基座统计](/root/autodl-tmp/r1-aqa/runs/omni3b-pilot-01/cross-backbone.json) · [0.6B 原报告](REPORT.md)

## 训练、早停与显存

NVIDIA GeForce RTX 5090 单卡 BF16。每组针对 36 层的 Q/K/V/O 和三个前馈投影，rank 8、alpha 16。3B 的物理 batch 4、梯度累积 4；0.6B 对应 8 和 2。3B 梯度检查点关闭。训练上限 2 轮，每 32 步及训练终点评开发准确率，连续三次未提升早停，同分取更早 checkpoint。

| 方法 | 可训练参数 | 最佳步 / 总更新 | 触发早停 | 训练 / 开发评分 / 测试评分（秒） | 训练 allocated / reserved（GiB） | 测试 allocated / reserved（GiB） |
|---|---:|---:|---|---:|---:|---:|
| LoRA | 14,966,784 | 96 / 192 | 是 | 288.4 / 97.5 / 107.0 | 19.30 / 20.13 | 9.65 / 10.06 |
| DoRA | 15,998,976 | 64 / 160 | 是 | 343.0 / 144.6 / 162.0 | 24.42 / 25.68 | 9.89 / 10.07 |
| 改进 TA-LoRA | 15,556,896 | 32 / 128 | 是 | 217.2 / 78.0 / 119.6 | 22.72 / 23.67 | 9.65 / 10.07 |

训练计时包含 checkpoint I/O 并扣除开发评分；显存数值为 PyTorch 峰值。[选模与 checkpoint](/root/autodl-tmp/r1-aqa/runs/omni3b-pilot-01/selection.json) · [容量压力测试](/root/autodl-tmp/r1-aqa/omni3b-preflight-01/gpu-capacity.json) · [逐 split 长度与分位数](/root/autodl-tmp/r1-aqa/omni3b-preflight-01/inputs/capacity.json) · [最长音频的音频塔核验](/root/autodl-tmp/r1-aqa/omni3b-preflight-01/inputs/long-audio-tower.json) · [基座快照哈希](/root/autodl-tmp/r1-aqa/omni3b-preflight-01/inputs/snapshot-audit.json)

## 开发集音频交换对照

| 方法 | 正常音频 | 交换音频 | 正常减交换（百分点） |
|---|---:|---:|---:|
| LoRA | 80.47% | 40.62% | +39.84 |
| DoRA | 80.86% | 36.72% | +44.14 |
| 改进 TA-LoRA | 80.47% | 37.50% | +42.97 |

保持题干、选项和录音集合不变，将每道开发题与其他源视频的录音配对。这一对照衡量预测对题目和录音匹配的敏感性。[完整消融](/root/autodl-tmp/r1-aqa/runs/omni3b-pilot-01/diagnostics/swapped-dev-audio/results.json) · [数据与复现说明](README.md) · [3B 配置](omni3b-pilot.json)
