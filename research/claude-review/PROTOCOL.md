# cota-review-01：预先声明的实验协议

写于 2026-11-22，在任何新实验运行之前。目的是让下一轮结果无论正负都能被解释。运行开始后若要修改本文件，请新增一节并注明日期和原因，不要改写已有条款。

固定范围：Qwen2.5-Omni-3B（revision `f75b40e3`）、单卡、现有语料、36 层 O 投影。所有配置由 [`experiments/cota_review/make_configs.py`](../../experiments/cota_review/make_configs.py) 生成并已提交，哈希在 [`matrix.json`](../../experiments/cota_review/matrix.json)。

## 1. 数据角色

| 名称 | 内容 | 用途 | 状态 |
|---|---|---|---|
| TRAIN | 混合训练清单 2688 条（AVQA 2048 + 配对 640） | 训练 | 沿用 `ta-conditional-01/data/mixed/train.jsonl` |
| DEV-PAIR | 混合配对开发 120 对（96 变化 + 24 保持） | 受控诊断：答案只能来自音频 | 沿用；已被观察 |
| DEV-REAL | AVQA dev 256 + 261 题代理开发 | 真实题型的开发检查 | 沿用；已被观察 |
| DEV-AC | **新建**：从未用于训练的 AVQA 题中取约 2000 条，与 TRAIN、AVQA dev 按原视频来源隔离；再按冻结基座划出"强音频依赖"子集 | **主开发指标** | 待在服务器构建 |
| MMAU-mini、MMAR、IEMOCAP-S5 | 现有目标集 | 只作探索性报告 | **已被多轮观察，不得称为最终测试** |
| FINAL | 一个此前未用过的选择题音频基准（候选：MMSU；MMAU 完整测试集若官方仍只接受提交评分也可） | 只在全部方案冻结后评一次 | 未观察；确定前先核实其许可与获取方式 |

DEV-AC 的构建规则（先写定，后构建）：

1. 抽样种子 20261122；按 AVQA 原视频 ID 排除 TRAIN 与 AVQA dev 的全部来源；A/B/C/D 标签分层。
2. 用冻结基座、现有提示和单 token 受限解码各评一次：原音频、等长静音。复用 `evaluate_frozen.py`；静音版本需要新写一个小的清单转换脚本。
3. "强音频依赖"子集 = 静音时答错的题。主指标是这个子集上的准确率；全集准确率一并报告。
4. 清单、哈希、两次冻结评测的逐题结果在任何适配器训练之前提交到运行目录。

为什么需要它：现有目标集里有相当比例的题不听音频也能答对，这类题上任何"让模型更用音频"的方法都没有表现空间；而 DEV-PAIR 只有 96 对，标准误 4.7 pp。约 1000 条强依赖题可以把配对差值的 95% 半宽压到 1.5 pp 左右。

## 2. 训练协议

**阶段 0 沿用旧协议**（1 轮、168 步、恒定 2e-4、每 32 步评测、按 DEV-PAIR 选模），只增加种子。配置：`configs/stage0-legacy/`。目的是测量已发表比较的种子噪声，不做任何改动。

**阶段 1、2 使用协议 v2**，相对旧协议的改动及理由：

| 项 | 旧 | v2 | 理由 |
|---|---|---|---|
| 轮数 / 更新数 | 1 / 168 | 3 / 504 | 旧协议在指标上升途中结束（DIAGNOSIS §1.2） |
| 学习率调度 | 恒定，无预热 | 5% 预热 + 线性衰减到 0 | 恒定学习率下相邻检查点波动 13–19 pp |
| 报告的检查点 | DEV-PAIR 最优 | **最后一个检查点（第 504 步）** | 去掉选模噪声；各臂步数一致 |
| 评测间隔 | 32 | 56（每轮 3 次） | 只用于画曲线 |
| patience | 3 | 100（等于不早停） | 同上 |
| 其余 | — | 不变 | batch 4×4、AdamW、wd 0、BF16、eager、无梯度检查点 |

代码仍会按 DEV-PAIR 记录一个"最优步"并保留对应检查点，作为次要分析报告，不作为主结果。

学习率不沿用 2e-4，而是在阶段 1 对两个主臂用同一组候选 {5e-5, 1e-4, 2e-4, 4e-4} 各跑 3 个种子，按 DEV-AC 强依赖子集的种子平均准确率各自取最优，并列时取较小者。阶段 2 的其余臂使用其参照臂选出的学习率（`make_configs.py --stage2-lr`）。

## 3. 实验臂

全部 36 层 O-only。参数量由生成脚本计算，与旧运行的 `result.json` 一致。

| 臂 | 方法 | 参数量 | 参照臂 | 与参照臂的唯一差别 |
|---|---|---:|---|---|
| `lora-r12` | LoRA r12 | 1,769,472 | — | 同预算基线 |
| `lora-r8` | LoRA r8 | 1,179,648 | `lora-r12` | 普通秩 12→8 |
| `cota-static` | r8 + C，静态门 | 1,769,760 | `lora-r12` | 4 个普通秩换成首答案位置的差分读出 |
| `ctrl-ordinary` | 同上 | 1,769,760 | `cota-static` | 读出的输入换成范数匹配的普通注意力输出 |
| `ctrl-audio-mean` | 同上 | 1,769,760 | `cota-static` | 读出的输入换成 E[V\|audio]，不减文本 |
| `cota-qa` | r8 + C + 条件门 | 1,772,064 | `cota-static` | 加入与旧运行完全相同的条件门 |
| `cota-qa-live` | 同上 | 1,772,064 | `cota-qa` | FP32 传输路径 + 门学习率 ×16 |
| `shared-r12` | r12，无 C，g 初值 1 | 1,769,904 | `lora-r12` | 只通过普通读出 B 加入首答案位置传输 |

`cota-qa-live` 同时改了两处，因为单独任何一处都不足以让门起作用（DIAGNOSIS §1.1）。如果它与 `cota-qa` 出现差别，再拆成两个单因素臂。

种子：阶段 0 用 42–46；阶段 1、2 用 42、43、44。若某个主比较的种子间标准差大于 1 pp，对该比较补到 5 个种子（事先定好，不看方向）。

## 4. 评测与统计

- 所有臂共用同一提示、BF16、eager、batch 2、单 token 受限贪心解码、干净基座重载后评测（现有 `transfer_eval.py`）。
- 同一检查点的消融：`--no-transport`、`--ablation neutral_gate | no_shared | no_independent`。
- **主指标**：DEV-AC 强依赖子集准确率。**次指标**：DEV-AC 全集、DEV-REAL、DEV-PAIR 变化配对两条都对率、保持配对一致率。
- **主比较**（共 2 个，Bonferroni 后每个取 97.5% 区间）：`cota-static − lora-r12`；`cota-static − ctrl-ordinary`。
- 其余比较全部报告，标为次要，不据此下结论。
- 差值与区间：每题先对种子取平均正确率，再按音频来源做聚类配对 bootstrap（10,000 次）；同时报告各种子差值的均值与标准差。
- 每个臂记录：实际参数量、训练秒数、峰值显存、每个因子（A/B/C/g/R）的权重范数、梯度范数和相邻梯度余弦（`adapter-stats.jsonl`）。

## 5. 预先写定的判定

| 结果 | 结论 | 下一步 |
|---|---|---|
| 两个主比较的区间都在 0 以上，且 DEV-REAL 不显著变差 | 差分读出有独立于位置和参数量的作用 | 进入 H3–H5，随后在 FINAL 上评一次 |
| `cota-static > lora-r12` 但与 `ctrl-ordinary` 无差别 | 收益来自"答案位置多一组读出"，与音频/文本差无关 | 方法应改述为位置受限适配，或放弃 CoTA 的机制主张 |
| `cota-static` 与 `lora-r12` 无差别（区间含零且半宽 ≤ 2 pp） | 现有形式没有可测收益 | 停止以 CoTA 为方法主线；转向 [`ACL_ASSESSMENT.md`](ACL_ASSESSMENT.md) §4 的分析型方案 |
| 区间半宽 > 2 pp | 证据不足 | 按 §3 补种子；仍不足则扩大 DEV-AC |
| `cota-qa-live` 在 `neutral_gate` 下不下降 | 条件门没有作用 | 论文与命名中去掉 Conditional |

不得做的事：按 MMAU-mini、MMAR、IEMOCAP-S5 的分数调整方案或挑检查点；只报告最好的种子；在看到 FINAL 之后再改任何设置。

## 6. 运行顺序与预算

旧运行每臂训练约 4–6 分钟（168 步），v2 约为 3 倍；加上开发评测，按每次 25 分钟估算。

| 步骤 | 内容 | 运行数 | 估计 GPU 时间 |
|---|---|---:|---:|
| P0 | 跑回归测试；确认 autocast；构建 DEV-AC；测 `e_prompt` 相似度 | — | < 1 小时 |
| 阶段 0 | 旧协议 2 臂 × 5 种子 | 10 | 约 2 小时 |
| 阶段 1 | 2 臂 × 4 学习率 × 3 种子 | 24 | 约 10 小时 |
| 阶段 2 | 其余 6 臂 × 3 种子 + 两个主臂在选定学习率下的结果复用阶段 1 | 18 | 约 8 小时 |
| 消融评测 | 每个 TA 检查点 4 种开关 × 开发集 | — | 约 3 小时 |

合计约一天单卡。任何一步超出估计一倍以上时停下来检查，而不是继续排队。

### P0 命令

```bash
cd /autodl-fs/data/west-ta-baselines        # 检出本分支后
PY=/root/autodl-tmp/iemocap/swift-ser/venv-5090/bin/python

# 1) 新增与原有的定向测试（本分支作者未能执行，必须先过）
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 "$PY" -m pytest -q -rs \
  experiments/ms_swift_librispeech/tests/test_answer_controls.py \
  experiments/ms_swift_librispeech/tests/test_conditional_gate.py \
  experiments/ms_swift_librispeech/tests/test_adapter_pair.py

# 2) 证据算术
"$PY" -m pytest -q research/claude-review/checks

# 3) 确认训练前向在 autocast 下：在任意一次短训练里临时加一个前向钩子打印
#    torch.is_autocast_enabled() 与 ta_adapter 输入的 dtype，记录到运行目录。
```

### 训练与评测命令模板

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" HF_HUB_OFFLINE=1 ENABLE_AUDIO_OUTPUT=0
RUN=/root/autodl-tmp/MMAR/runs/cota-review-01          # 新目录，不触碰旧运行
DATA=/root/autodl-tmp/MMAR/runs/ta-conditional-01/data/mixed
CFG=experiments/cota_review/configs/stage0-legacy/cota-qa-s43.json
METHOD=ta-lora                                          # lora-* 臂用 lora

flock /root/autodl-tmp/r1-aqa/runs/omni3b-tuners-01/gpu.lock \
  "$PY" -m experiments.ms_swift_librispeech.train --config "$CFG" --method "$METHOD" \
  --train "$DATA/train.jsonl" --dev "$DATA/pair-dev.jsonl" \
  --output "$RUN/stage0-legacy/cota-qa-s43"

# 同一检查点的单因素消融（以强制 m=1 为例）
"$PY" -m experiments.ms_swift_aqa.transfer_eval --config "$CFG" --method ta-lora \
  --checkpoint "$RUN/stage0-legacy/cota-qa-s43/checkpoint-160" \
  --manifest "$DEV_AC" --folder "$RUN/eval/cota-qa-s43/dev-ac-neutral-gate" --ablation neutral_gate
```

每个运行目录保留 `protocol.json`、`initialization.json`、`logging.jsonl`、`adapter-stats.jsonl`、`dev/step-*/`、`result.json`。失败和被否定的运行同样保留并在汇总表中列出。
