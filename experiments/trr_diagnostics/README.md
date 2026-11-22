# D2：模型的回答是否依赖语音的 F0 与能量轮廓

这是 TRR 方案动工前的第一项诊断。它只回答一个行为层面的问题：**同一句话，把音高和能量轮廓拉平之后，模型的答案变不变。** 不训练、不选模、不改任何已有 checkpoint 和日志。

**2026-11-23 整合状态：** 已接入现有工程，50 个 CPU 测试通过，真实 WORLD 合成滑音自检通过；16 条 Session 4 真实音频的声学验收失败，尚未运行 3B 模型。当前不能据此判断模型是否使用韵律。实测、修复和日志见 [INTEGRATION.md](INTEGRATION.md)。

## 它能说明什么，不能说明什么

- 能说明：在声学验收及损伤控制通过的前提下，冻结基座（或某个已训练的 adapter）在情绪四分类上对这组 F0/能量干预的行为敏感性；录音相对转写文本的表现差异；声码器引入的表现变化。
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

孪生保持原录音长度，并尝试匹配 RMS；防削波缩放可能破坏 RMS 匹配，因此必须检查实际落盘音频，不能仅根据算法步骤声称等 RMS。WORLD 合成使用原谱包络、非周期性、清浊判决；输出端仍需重新分析验收。转写中的 `[LAUGHTER]` 等方括号标注在 `text` 条件里删除，否则会泄露答案；删除后没有词的语句从全部条件中剔除，保证各条件逐条配对。

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

另检查落盘音频无削波且 RMS 相对原音频的误差不超过 2%（`decision.qc.max_rms_relative_error` 可显式配置，默认 0.02）。当前清单 1031 条，删除无词转写后为 1026 条。区间宽度由实际配对预测及对话聚类结构决定，不能在模型评测前保证 2 pp 的精度。

## 结果怎么读

声学验收失败、声码器/可懂度控制缺失或失败、主对比未完成、以及任何 `--limit` 冒烟，均将 `verdicts.f0_energy_use` 标为 `not_interpretable`。`raw_verdicts` 仅保留阈值计算用于排查；不能把它当成有效结论。先定位干预与控制问题，再解释掉分。控制通过也不能排除全部合成伪影，见局限。

| `f0_energy_use` | `audio_beyond_transcript` | 含义 | 下一步 |
|---|---|---|---|
| `not_interpretable` | 任意 | 验收或控制不满足，或仅为冒烟 | 按 `interpretation.reasons` 补齐检查；不能判断有无韵律贡献 |
| `uses` | 任意 | 该模型在这个任务上已经在用 F0/能量 | 这个任务上不存在针对 F0/能量的 "Heard but Not Heeded"；TRR 的前提在此不成立，换任务或停止 |
| `at_most_small` | `no` | 行为上接近只看文字 | 候选的 HbNH 情形；接着做 D1/D3，确认信息确实被编码而只是没被读出 |
| `at_most_small` | `yes` | 录音有帮助，但不是靠 F0/能量轮廓 | TRR v1 的孪生对准了错误的因素；考虑其他孪生（节奏、音质）或重新定义范围 |
| `inconclusive` | 任意 | 效应在本诊断的分辨率以下或区间太宽 | 不据此做决定；可加入 Session 1–3 扩大样本后重做（需另行声明） |

对冻结基座和已训练的 LoRA 各跑一次才有决策价值：如果基座是 `at_most_small` 而普通 LoRA 已经是 `uses`，说明常规微调就能打开这条通路，TRR 必须在这些样本上超过 LoRA 才有意义。

分层结果（转写单独答对/答错、同词异标签、即兴/剧本、说话人）只报告准确率，属于事后描述，不参与判据。Session 4 只有两位说话人，结论不能外推到说话人层面。

## 运行

代码已整合到现有工程，无需重新克隆。下面先做 CPU 检查；只有真实音频验收合格且 GPU 可用时才进入模型评测。本轮 16 条真实音频尚未合格，命令不是正式评测已获验收的声明。

```bash
cd /autodl-fs/data/west-ta-baselines
PY=/root/autodl-tmp/iemocap/swift-ser/venv-5090/bin/python

# 唯一的新依赖，固定版本并装到独立目录；本机本轮已安装。
# 在新环境中才需要执行：
# "$PY" -m pip install --no-deps --target /root/autodl-tmp/trr-diagnostics/deps \
#   -r experiments/trr_diagnostics/requirements.txt
export PYTHONPATH="/root/autodl-tmp/trr-diagnostics/deps${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# 1. 单元测试（CPU）
env CUDA_VISIBLE_DEVICES= "$PY" -m pytest -q experiments/trr_diagnostics/tests

# 2. 在本机用合成滑音验证 WORLD 的分析—合成链路；全部 true 才继续
"$PY" -m experiments.trr_diagnostics.twin --self-check

# 3. 真实音频冒烟。每次 prepare 使用新目录，保持旧日志可追溯。
export HF_HUB_OFFLINE=1 ENABLE_AUDIO_OUTPUT=0
CFG=experiments/trr_diagnostics/d2-iemocap-session4.json
SMOKE=/root/autodl-tmp/trr-diagnostics/d2-smoke-01
"$PY" -m experiments.trr_diagnostics.d2 prepare --config "$CFG" --output "$SMOKE" --limit 16 --workers 1
"$PY" -c 'import json,sys; p=json.load(open(sys.argv[1])); sys.exit(0 if p["qc"]["passed"] else "STOP: acoustic QC failed")' "$SMOKE/prepare.json"
```

上一步通过后，在有 GPU 的环境中单独执行模型冒烟；仅证明调用链，指标不作研究结论：

```bash
"$PY" -m experiments.trr_diagnostics.d2 run --prepared "$SMOKE" \
  --model-config experiments/ms_swift_iemocap/config.json --method frozen-base --arm frozen-base

# 4. 全部 Session 4；先确认完整 prepared 的声学验收。
OUT=/root/autodl-tmp/trr-diagnostics/d2-iemocap-s4-01
"$PY" -m experiments.trr_diagnostics.d2 prepare --config "$CFG" --output "$OUT" --workers 8
"$PY" -c 'import json,sys; p=json.load(open(sys.argv[1])); sys.exit(0 if p["qc"]["passed"] else "STOP: acoustic QC failed")' "$OUT/prepare.json"
```

完整 prepared 验收通过后：

```bash
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

`run` 可以重复执行：完成条件在配置、音频、清单、预测文件及预测 ID/参考答案校验通过后才复用；模型配置、adapter 配置/权重、prepared 元数据或执行代码变化会拒绝混用同一臂。中断后重跑同一条命令即可，没有 `metrics.json` 完成标记的预测不会进入报告。`text`、`question_only` 和 ASR 条件失败时记录 `failure.json` 并继续，音频条件失败则立即停止；ASR 缺失时报告不可解释。只重新生成报告用 `d2 report --prepared "$OUT" --arm frozen-base`。

新版 prepared 格式为 v2，包含原音频及孪生的 SHA-256。旧版目录不满足校验时，请在新目录重新 `prepare`，不要修改旧元数据冒充 v2。`prepare` 成功生成文件不等于声学验收通过，必须查看 `prepare.json` 的 `qc.passed`。

## 输出

```
$OUT/config.json            配置快照
$OUT/prepare.json           样本数、剔除原因、孪生参数、声学验收、清单/音频/代码摘要
$OUT/qc.jsonl               逐条的 F0/能量轮廓统计
$OUT/manifests/*.jsonl      各条件清单
$OUT/audio/<条件>/*.wav      孪生音频
$OUT/runs/<臂>/<条件>/       predictions.jsonl、metrics.json
$OUT/runs/<臂>/report.md     汇总表与判据结论（report.json 为同内容的机器可读版）
```

请把 `prepare.json`、`runs/*/report.md`、`runs/*/report.json` 和各条件的 `predictions.jsonl` 回传或提交到结果分支。逐题预测是后续分析的输入，汇总数字不能替代它。

## 成本估计

以下是上游方案按历史 GPU 评测推算的估计，**不是本轮 D2 的 GPU 实测**。本轮只有 CPU：新版 16 条 prepare 用时 45.14 秒（单 worker，与 CPU 测试并发），不能直接外推满负载 GPU 时间或显存。

| 步骤 | 估计 | 依据 |
|---|---|---|
| `prepare`，8 进程 | 5–15 分钟 | 未实测。约 77 分钟音频（按每条 4.5 秒估），WORLD 各步通常快于实时。冒烟步骤会给出本机实际速度 |
| 每个音频条件 | 约 1 分钟 | 实测：冻结基座评 Session 5 的 1241 条用 74 秒（`ser-verified.json`），按 1031 条折算 |
| `text`、`question_only` | 各不到 1 分钟 | 推断：序列更短 |
| ASR 检查，3 × 300 条 | 合计 5–15 分钟 | 未实测，逐条生成最多 128 token |
| 显存峰值 | 约 10 GB | 实测：同一评测 10.1 GB（`ser-verified.json`），5090 的 32 GB 足够 |

## 验证状态

- 上游 35 个测试通过；新增 15 个回归用例先复现失败、修复后通过；整合后总计 **50 passed，0 failed，0 skipped**（1 条第三方 pytest 弃用警告）。模型加载和打分的编排测试使用替身，不代表真实推理通过。
- 真实 `pyworld==0.3.5` 合成滑音自检 12 项全部通过；真实 Session 4 的 16 条音频已生成，**声学验收失败**，见 [实测汇总](validation/real-audio-smoke.json)。
- 已静态核对：ms-swift 快照中 Omni 模板的 `_encode` 与 collator 对没有音频的输入有分支处理。
- **未验证**：真实 3B 模型加载、音频/纯文本推理、adapter 重载、ASR 控制、GPU 峰值和全量 Session 4 的模型报告。当前执行环境未暴露 GPU，未启动这些步骤。

## 局限

- 孪生只改 F0 和慢变能量，时长、停顿、音质不变；结论只对这两个因素成立。
- WORLD 的分析—合成有可闻损伤，`sham` 用来把它从拉平效应里分离出来，但不能排除"拉平后的合成比原样合成更不自然"带来的额外影响；可懂度检查只覆盖词的层面。
- IEMOCAP 是表演语料，Session 4 仅两位说话人。
- `text` 条件的提示只有一种写法；它是参照，不是文本上限。
- bootstrap 区间反映样本变异，不是训练种子变异。
