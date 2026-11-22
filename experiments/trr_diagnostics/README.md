# D2：模型的回答是否依赖语音的 F0 与能量轮廓

这是 TRR 方案动工前的第一项诊断。它只回答一个行为层面的问题：**同一句话，把音高和能量轮廓拉平之后，模型的答案变不变。** 不训练、不选模、不改任何已有 checkpoint 和日志。

## 它能说明什么，不能说明什么

- 能说明：冻结基座（或某个已训练的 adapter）在情绪四分类上，是否在用 F0/能量；录音相对转写文本多带来多少；声码器本身引入多少损伤。
- 不能说明：信息是否"被编码"（那是 D1、D3 探针的事）；节奏、停顿、音质是否被使用（孪生只改 F0 和慢变能量）；任何方法是否有效。
- 数据是 IEMOCAP Session 4（开发集，1031 条四分类）。Session 5 不参与。IEMOCAP 在本项目里已被多轮观察，本诊断同属探索性分析。

## 条件

每个条件问同一个问题、用同一批语句，只换输入：

| 条件 | 输入 | 作用 |
|---|---|---|
| `orig` | 原始录音 | 参照 |
| `sham` | WORLD 分析后原样合成 | 只含声码器损伤，是各拉平条件的对照 |
| `flat_f0` | 浊音帧 F0 置为该句中位数 | 去掉音高轮廓 |
| `flat_energy` | 慢变能量（250 ms 平滑）拉到中位数 | 去掉响度轮廓 |
| `flat_both` | 两者都拉平 | TRR v1 的反事实输入 a⁻ |
| `text` | 只给转写文本，不给音频 | 词汇信息单独能答对多少 |
| `question_only` | 只给问题 | 提示本身的类别先验 |

所有孪生与原录音等长、等 RMS；谱包络、非周期性、清浊判决不动。转写中的 `[LAUGHTER]` 等方括号标注在 `text` 条件里删除，否则会泄露答案；删除后没有词的语句从全部条件中剔除，保证各条件逐条配对。

另有一个可懂度检查：在 300 条子集上让模型转写 `orig`、`sham`、`flat_both`，比较 WER。拉平若损伤了词的可懂度，`flat_both` 上的掉分就不能归因于韵律。

## 预先声明的对比与判据

写在 [d2-iemocap-session4.json](d2-iemocap-session4.json) 的 `contrasts` 与 `decision` 里，`prepare` 时连同配置摘要一起固化，跑完后不再改。区间是按对话聚类的 95% bootstrap（10000 次，seed 42，与本项目此前 IEMOCAP 分析的做法一致）。

| 判据 | 对比 | 规则 |
|---|---|---|
| 声码器损伤可接受 | `sham − orig` 的 UAR | 绝对值 ≤ 2 pp 为通过 |
| 可懂度保持 | `flat_both − sham` 的 WER | 升高 ≤ 2 pp 为通过 |
| 是否使用 F0/能量（主对比） | `flat_both − sham` 的 UAR | 下降 ≥ 3 pp 且区间不含 0 → `uses`；区间整体落在 ±3 pp 内 → `at_most_small`；其余 → `inconclusive` |
| 录音是否超过转写 | `orig − text` 的 UAR | 区间下界 > 0 → `yes`；上界 ≤ 0 → `no`；其余 → `inconclusive` |
| 孪生的声学验收 | 重新分析各孪生 | 拉平项的轮廓标准差降到原来的 20%（F0）/50%（能量）以内，未编辑项保持在 0.8–1.25 倍 |

1031 条、约 30 段对话的规模下，配对差的区间半宽大致在 2 pp 上下，所以 3 pp 以下的效应本诊断分辨不了，会落到 `inconclusive`。

## 结果怎么读

前两项不通过时，后面的结论都不成立，应先换声码器或调参数，而不是解释掉分。

| `f0_energy_use` | `audio_beyond_transcript` | 含义 | 下一步 |
|---|---|---|---|
| `uses` | 任意 | 该模型在这个任务上已经在用 F0/能量 | 这个任务上不存在针对 F0/能量的 "Heard but Not Heeded"；TRR 的前提在此不成立，换任务或停止 |
| `at_most_small` | `no` | 行为上接近只看文字 | 候选的 HbNH 情形；接着做 D1/D3，确认信息确实被编码而只是没被读出 |
| `at_most_small` | `yes` | 录音有帮助，但不是靠 F0/能量轮廓 | TRR v1 的孪生对准了错误的因素；考虑其他孪生（节奏、音质）或重新定义范围 |
| `inconclusive` | 任意 | 效应在本诊断的分辨率以下或区间太宽 | 不据此做决定；可加入 Session 1–3 扩大样本后重做（需另行声明） |

对冻结基座和已训练的 LoRA 各跑一次才有决策价值：如果基座是 `at_most_small` 而普通 LoRA 已经是 `uses`，说明常规微调就能打开这条通路，TRR 必须在这些样本上超过 LoRA 才有意义。

分层结果（转写单独答对/答错、同词异标签、即兴/剧本、说话人）只报告准确率，属于事后描述，不参与判据。Session 4 只有两位说话人，结论不能外推到说话人层面。

## 运行

以下命令在服务器上执行。建议把分支克隆到新目录，不动现有工作区。

```bash
git clone --branch research/trr-diagnostics --depth 1 https://github.com/shanmon110/CoTA-LoRA.git /root/autodl-tmp/trr-diagnostics/src
cd /root/autodl-tmp/trr-diagnostics/src
PY=/root/autodl-tmp/iemocap/swift-ser/venv-5090/bin/python

# 唯一的新依赖。--no-deps 保证不改动现有 numpy 等版本。
"$PY" -m pip install --no-deps pyworld

# 1. 单元测试（CPU，几秒）
env CUDA_VISIBLE_DEVICES= "$PY" -m pytest -q experiments/trr_diagnostics/tests

# 2. 在本机用合成滑音验证 WORLD 的分析—合成链路；全部 true 才继续
"$PY" -m experiments.trr_diagnostics.twin --self-check

# 3. 冒烟：16 条，确认 prepare / run / report 能跑通。数字无意义。
export HF_HUB_OFFLINE=1 ENABLE_AUDIO_OUTPUT=0 OMP_NUM_THREADS=2
CFG=experiments/trr_diagnostics/d2-iemocap-session4.json
SMOKE=/root/autodl-tmp/trr-diagnostics/d2-smoke-01
"$PY" -m experiments.trr_diagnostics.d2 prepare --config "$CFG" --output "$SMOKE" --limit 16
"$PY" -m experiments.trr_diagnostics.d2 run --prepared "$SMOKE" \
  --model-config experiments/ms_swift_iemocap/config.json --method frozen-base --arm frozen-base

# 4. 正式：全部 Session 4
OUT=/root/autodl-tmp/trr-diagnostics/d2-iemocap-s4-01
"$PY" -m experiments.trr_diagnostics.d2 prepare --config "$CFG" --output "$OUT" --workers 8
"$PY" -m experiments.trr_diagnostics.d2 run --prepared "$OUT" \
  --model-config experiments/ms_swift_iemocap/config.json --method frozen-base --arm frozen-base \
  2>&1 | tee "$OUT/frozen-base.log"
```

对已训练的 adapter 再跑一臂（`--model-config` 必须是训练它时用的配置，`--arm` 自己起名）：

```bash
"$PY" -m experiments.trr_diagnostics.d2 run --prepared "$OUT" \
  --model-config /root/autodl-tmp/MMAR/runs/ta-conditional-01/configs/lora-mixed.json \
  --method lora --checkpoint /root/autodl-tmp/MMAR/runs/ta-conditional-01/train/lora-mixed/checkpoint-160 \
  --arm lora-mixed 2>&1 | tee "$OUT/lora-mixed.log"
```

`run` 可以重复执行：已有 `metrics.json` 的条件不会重算，中断后直接重跑同一条命令即可。`text`、`question_only` 和 ASR 条件失败时会记录 `failure.json` 并继续，音频条件失败则立即停止。只重新生成报告用 `d2 report --prepared "$OUT" --arm frozen-base`。

## 输出

```
$OUT/config.json            配置快照
$OUT/prepare.json           样本数、剔除原因、孪生参数、声学验收、各清单摘要、代码提交号
$OUT/qc.jsonl               逐条的 F0/能量轮廓统计
$OUT/manifests/*.jsonl      各条件清单
$OUT/audio/<条件>/*.wav      孪生音频
$OUT/runs/<臂>/<条件>/       predictions.jsonl、metrics.json
$OUT/runs/<臂>/report.md     汇总表与判据结论（report.json 为同内容的机器可读版）
```

请把 `prepare.json`、`runs/*/report.md`、`runs/*/report.json` 和各条件的 `predictions.jsonl` 回传或提交到结果分支。逐题预测是后续分析的输入，汇总数字不能替代它。

## 成本估计

| 步骤 | 估计 | 依据 |
|---|---|---|
| `prepare`，8 进程 | 5–15 分钟 | 未实测。约 77 分钟音频（按每条 4.5 秒估），WORLD 各步通常快于实时。冒烟步骤会给出本机实际速度 |
| 每个音频条件 | 约 1 分钟 | 实测：冻结基座评 Session 5 的 1241 条用 74 秒（`ser-verified.json`），按 1031 条折算 |
| `text`、`question_only` | 各不到 1 分钟 | 推断：序列更短 |
| ASR 检查，3 × 300 条 | 合计 5–15 分钟 | 未实测，逐条生成最多 128 token |
| 显存峰值 | 约 10 GB | 实测：同一评测 10.1 GB（`ser-verified.json`），5090 的 32 GB 足够 |

## 验证状态

- 已在无 GPU 环境通过：35 个单元测试，覆盖拉平与增益的数值、统计量、清单构造、`run` 的编排（用替身打分函数）、报告与判据。
- 已静态核对：ms-swift 快照中 Omni 模板的 `_encode` 与 collator 对没有音频的输入有分支处理。
- **未验证**：`pyworld` 的真实调用（这里无法安装）；真实模型上的任何一步。步骤 2 和步骤 3 就是为在服务器上补这两项验证而设的。

## 局限

- 孪生只改 F0 和慢变能量，时长、停顿、音质不变；结论只对这两个因素成立。
- WORLD 的分析—合成有可闻损伤，`sham` 用来把它从拉平效应里分离出来，但不能排除"拉平后的合成比原样合成更不自然"带来的额外影响；可懂度检查只覆盖词的层面。
- IEMOCAP 是表演语料，Session 4 仅两位说话人。
- `text` 条件的提示只有一种写法；它是参照，不是文本上限。
- bootstrap 区间反映样本变异，不是训练种子变异。
