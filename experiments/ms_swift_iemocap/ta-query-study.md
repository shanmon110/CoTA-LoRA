# IEMOCAP 上 TA-LoRA 的查询位置与门控实验

在 Qwen3-ASR-0.6B 的四分类识别任务上，允许音频位置也使用 TA 分支，使 seed 42 的测试 UAR 从 **72.38% 提高到 74.73%**。原生 DoRA 为 **75.17%**。开发集上扩展查询的 TA 得分最高，测试集上 DoRA 仍领先 0.44 个百分点。

## 第一阶段的选择过程

原 TA 只在文本查询位置向 O 投影加入音频与文本上下文的差。我们对同一个原 TA checkpoint 做 CPU BF16 消融，在完整 Session 4 上开关门控，仅 11/1,031 条预测改变，UAR 分别为 72.21% 和 72.23%。CPU 开门控与原 GPU 推理本身也有 13 条预测不同，因此消融只比较同一设备内的变化。[原 TA 门控消融](/root/autodl-tmp/iemocap/swift-ser/ta-improvement-evidence/gate-ablation/comparison.json)

先调整门控初值或门控乘数。三项都未超过 DoRA 的开发集 UAR，随后按计划将 TA 分支扩到所有有效查询位置，保留因果约束并排除 padding。每个启用该分支的查询都须同时看到至少一个音频键和一个文本键。两轮候选均按完整 Session 4 的 UAR 选择 checkpoint，持平取更早的步数。六个候选中仅冻结后的 `all-valid` 进入最终测试。[第一轮计划](/root/autodl-tmp/iemocap/swift-ser/ta-gate-study-01/plan.json) · [第二轮计划](/root/autodl-tmp/iemocap/swift-ser/ta-query-study-01/plan.json) · [冻结的选择](/root/autodl-tmp/iemocap/swift-ser/ta-query-study-01/selection.json)

| 候选 | 查询位置 | 门控初值 | 门控乘数 | Session 4 UAR | 最佳步 |
|---|---|---:|---:|---:|---:|
| init-quarter | 文本 | 0.25 | 1 | 71.47% | 102 |
| gain16 | 文本 | 0 | 16 | 71.24% | 50 |
| init-one | 文本 | 1 | 1 | 71.32% | 102 |
| all-valid | 文本与音频 | 0 | 1 | **75.17%** | 102 |
| all-valid-gain16 | 文本与音频 | 0 | 16 | 70.74% | 102 |
| all-valid-quarter | 文本与音频 | 0.25 | 1 | 72.26% | 102 |

## 四种方法的完整测试结果

Session 1–3 的 3,259 条用于训练，Session 4 的 1,031 条用于选模，Session 5 的 1,241 条用于测试。四种方法均在 28 层语言解码器覆盖 Q/K/V/O 和 MLP gate/up/down。TA 的专有分支位于 O 投影。训练采用四选一字母回复的 SFT，rank 8、alpha 16、batch 32、累积 1，启用梯度检查点。第一阶段限一轮，每 25 步及轮末检查开发集 UAR，连续三次未提高触发早停。四种方法最终均选择第 102 步。从干净基座加载所选 adapter 后，对下一 token 的四个合法字母取最高分。UAR 为四类召回率的算术平均值。[汇总结果](/root/autodl-tmp/iemocap/swift-ser/ta-query-study-01/verified-comparison.json) · [原始对照配置与运行记录](/root/autodl-tmp/iemocap/swift-ser/qwen3-p3) · [入选 TA 的训练记录](/root/autodl-tmp/iemocap/swift-ser/ta-query-study-01/all-valid/ta-lora/result.json)

| 方法 | 开发 UAR | 测试 UAR | 测试准确率 | 可训练参数 |
|---|---:|---:|---:|---:|
| LoRA | 72.64% | 73.45% | 74.38% | 5,046,272 |
| 原生 DoRA | 73.78% | **75.17%** | **76.55%** | 5,390,336 |
| 原 TA-LoRA | 72.90% | 72.38% | 73.97% | 5,046,496 |
| TA-LoRA（all-valid） | **75.17%** | 74.73% | 75.02% | 5,046,496 |

将 Session 5 相同样本逐条配对，以 31 段对话为单位有放回抽样 20,000 次（seed 42）。all-valid 减 DoRA 的 UAR 差为 −0.44 个百分点，百分位 95% 区间为 [−1.94, +1.35]。准确率差为 −1.53 个百分点，区间为 [−3.38, +0.58]。两区间均跨零，仅反映这组固定模型在该 Session 的对话采样不确定性。[逐类差异与区间](/root/autodl-tmp/iemocap/swift-ser/ta-query-study-01/bootstrap-final.json)

同一 all-valid checkpoint 在 CPU 上分别启用全部查询、仅文本查询和将门控归零，开发 UAR 依次为 74.57%、75.03%、75.57%。这组结果支持进一步检验联合训练中低秩权重的变化，单凭最终得分无法把收益归因于推理时开启 TA 分支。[入选 TA 的查询消融](/root/autodl-tmp/iemocap/swift-ser/ta-improvement-evidence/query-ablation/comparison.json)

六组 TA 配置都使用同一开发集，LoRA 和 DoRA 使用既定配置。表中比较的是这些具体配方的得分。六组中的最高开发分会受到选择过程影响。测试调度脚本起初与 Python 标准库 `select` 重名，第一次调用在加载模型前退出。保留已冻结的选择并修正文件名后才生成完整测试预测。

另一个配对消融保持原生 DoRA 训练后的方向和归一化，将幅值恢复到与 GPU 初始指纹完全匹配的值。同一 CPU 上完整 Session 4 的 UAR 从 73.96% 降至 72.51%，准确率从 76.14% 降至 75.17%。这表明学习幅值在这个已训练模型上确实改变了判别结果。两种不同方法的独立训练轨迹仍会影响差距。[DoRA 幅值消融](/root/autodl-tmp/iemocap/swift-ser/ta-improvement-evidence/dora-magnitude-ablation/comparison.json)

## 复现第一阶段的 all-valid

在工程根目录使用以下命令，运行入口要求新输出目录。配置固定基座 revision `4ce9cc728b473a5aedbe7b6e1ea45646316824dc`，默认仅评估开发集。[实验配置](qwen3-asr-0.6b-ta-all-valid-b32.json) · [本地模型文件标识](/root/autodl-tmp/iemocap/qwen3-asr-capacity/model-snapshot.json) · [划分与清单哈希](/root/autodl-tmp/iemocap/swift-ser/data/splits.json)

```bash
export HF_HUB_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
export PATH=/root/autodl-tmp/iemocap/qwen3-asr-capacity/venv/bin:"$PATH"
py=/root/autodl-tmp/iemocap/qwen3-asr-capacity/venv/bin/python
r=/root/autodl-tmp/iemocap/swift-ser
out="$r/reproduce-all-valid-seed42"
"$py" -m experiments.ms_swift_iemocap.run \
  --config experiments/ms_swift_iemocap/qwen3-asr-0.6b-ta-all-valid-b32.json \
  --data "$r/data" --output "$out"
```

复现已冻结选择的 Session 5 评分时，在上述运行完成后单独执行以下命令。

```bash
"$py" -m experiments.ms_swift_iemocap.run --config "$out/config.json" \
  --data "$r/data" --output "$out" --score test --method ta-lora
```

## 独立种子确认

确认实验已启动，最长三轮。seed 42 用完整 Session 4 在 all-valid 和带独立上投影的 TA 中选定一种，持平取 all-valid。seed 43 按同一协议重新训练 LoRA、DoRA 和选定 TA，先冻结三组 checkpoint，再统一评估完整 Session 5。既往模型已在 Session 5 上评过分，换 seed 能检验训练随机性的影响，仍然使用同一批说话人。[锁定计划](/root/autodl-tmp/iemocap/swift-ser/ta-confirmation-01/plan.json) · [运行状态](/root/autodl-tmp/iemocap/swift-ser/ta-confirmation-01/status.json) · [完成后生成的报告](/root/autodl-tmp/iemocap/swift-ser/ta-confirmation-01/report.json)

## 三轮上限确认实验结果

seed 42 按 Session 4 UAR 选择了 `independent`。原 all-valid 为 73.33%，独立上投影为 76.22%。同协议 LoRA 为 72.60%，DoRA 为 78.63%。[冻结的选择](/root/autodl-tmp/iemocap/swift-ser/ta-confirmation-01/seed42-selection.json)

| seed 43 方法 | 开发 UAR | 开发准确率 | 测试 UAR | 测试准确率 | 最佳步 | 训练结束步 | 可训练参数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| LoRA | 79.57% | 77.89% | 74.39% | 72.04% | 175 | 250 | 5,046,272 |
| 原生 DoRA | 79.97% | 78.66% | 74.69% | 71.88% | 175 | 250 | 5,390,336 |
| TA-LoRA | 80.98% | 79.53% | 74.95% | 72.84% | 175 | 250 | 5,275,872 |

按同一批 31 段对话成对重采样 20,000 次（seed 42）。下表给出 TA 减对照的百分点差及百分位 95% 区间。

| 对照 | 测试 UAR 差 [95% 区间] | 测试准确率差 [95% 区间] |
|---|---:|---:|
| 原生 DoRA | +0.26 [-1.62, +2.76] | +0.97 [-2.00, +4.32] |
| LoRA | +0.56 [-1.21, +2.58] | +0.81 [-1.48, +3.19] |

三个模型使用同一个已冻结的 Session 5 测试清单。区间只衡量固定模型在这些对话上的不确定性。[完整报告](/root/autodl-tmp/iemocap/swift-ser/ta-confirmation-01/report.json) · [配对 DoRA](/root/autodl-tmp/iemocap/swift-ser/ta-confirmation-01/bootstrap-vs-dora.json) · [配对 LoRA](/root/autodl-tmp/iemocap/swift-ser/ta-confirmation-01/bootstrap-vs-lora.json)
