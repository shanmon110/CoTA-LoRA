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
| `sham` | 提取音高后原样重合成（Praat 叠接相加） | 只含重合成损伤，是各拉平条件的对照 |
| `flat_f0` | 浊音帧 F0 置为该句中位数 | 去掉音高轮廓 |
| `flat_energy` | 慢变能量（250 ms 平滑）拉到中位数 | 去掉响度轮廓 |
| `flat_both` | 两者都拉平 | TRR v1 的反事实输入 a⁻ |
| `text` | 只给转写文本，不给音频 | 词汇信息单独能答对多少 |
| `question_only` | 只给问题 | 提示本身的类别先验 |

所有孪生与原录音等长、等 RMS。某条语句只要有一个孪生会削波，这条语句的全部孪生一起按同一比例降低，所以孪生之间响度始终一致，只在被编辑的轮廓上不同。转写中的 `[LAUGHTER]` 等方括号标注在 `text` 条件里删除，否则会泄露答案；删除后没有词的语句从全部条件中剔除，保证各条件逐条配对。

另有一个可懂度检查：在 300 条子集上让模型转写 `orig`、`sham`、`flat_both`，比较 WER。重合成或拉平若损伤了词的可懂度，后面的掉分就不能归因于韵律。

默认后端是 Praat 的叠接相加（PSOLA）：它移动录音里原有的基音周期，而不是用声码器参数重建整段信号。WORLD 仍可通过 `"backend": "world"` 使用，但不建议（见文末"第一次冒烟的发现"）。

## 预先声明的对比与判据

写在 [d2-iemocap-session4.json](d2-iemocap-session4.json) 的 `contrasts` 与 `decision` 里，`prepare` 时连同配置摘要一起固化，跑完后不再改。区间是按对话聚类的 95% bootstrap（10000 次，seed 42，与本项目此前 IEMOCAP 分析的做法一致）。

| 判据 | 对比 | 规则 |
|---|---|---|
| 重合成不改变答案 | `sham − orig` 的 UAR | 绝对值 ≤ 2 pp 为通过 |
| 重合成不损伤词 | `sham − orig` 的 WER | 升高 ≤ 2 pp 为通过 |
| 拉平不损伤词 | `flat_both − sham` 的 WER | 升高 ≤ 2 pp 为通过 |
| 是否使用 F0/能量（主对比） | `flat_both − sham` 的 UAR | 下降 ≥ 3 pp 且区间不含 0 → `uses`；区间整体落在 ±3 pp 内 → `at_most_small`；其余 → `inconclusive` |
| 录音是否超过转写 | `orig − text` 的 UAR | 区间下界 > 0 → `yes`；上界 ≤ 0 → `no`；其余 → `inconclusive` |
| 孪生的声学验收 | 重新分析各孪生 | 拉平项的轮廓标准差降到原来的 20%（F0）/50%（能量）以内，未编辑项保持在 0.8–1.25 倍；孪生之间响度一致、无削波。能量在原录音的有声段上测量 |

另检查落盘音频无削波且 RMS 相对原音频的误差不超过 2%（`decision.qc.max_rms_relative_error` 可显式配置，默认 0.02）。当前清单 1031 条，删除无词转写后为 1026 条。区间宽度由实际配对预测及对话聚类结构决定，不能在模型评测前保证 2 pp 的精度。

## 结果怎么读

前三项和声学验收不通过时，后面的结论都不成立，应先修音频干预，而不是解释掉分。其中 UAR 那一项需要全量才有分辨力；WER 两项和声学验收在 16 条冒烟上就能看出大问题。

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

# 唯一的新依赖，装到独立目录，不改动现有环境的 numpy 等版本。
"$PY" -m pip install --no-deps --target /root/autodl-tmp/trr-diagnostics/deps \
  -r experiments/trr_diagnostics/requirements.txt
export PYTHONPATH="/root/autodl-tmp/trr-diagnostics/deps${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# 1. 单元测试（CPU）
env CUDA_VISIBLE_DEVICES= "$PY" -m pytest -q experiments/trr_diagnostics/tests

# 2. 在本机用合成滑音验证 Praat 重合成链路已装好、接对；全部 true 才继续
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
| `prepare`，8 进程 | 未知 | WORLD 后端实测 16 条（94 秒音频）用 45 秒；Praat 后端未实测，冒烟会给出本机速度 |
| 每个音频条件 | 约 1 分钟 | 实测：冻结基座评 Session 5 的 1241 条用 74 秒（`ser-verified.json`），按 1026 条折算 |
| `text`、`question_only` | 各不到 1 分钟 | 推断：序列更短 |
| ASR 检查，3 × 300 条 | 合计 5–15 分钟 | 未实测，逐条生成最多 128 token |
| 显存峰值 | 约 9 GiB | 实测：2026-11-23 冒烟两臂峰值 9.1 GiB |

## 验证状态

- 已在无 GPU 环境通过全部单元测试（含服务器侧整合时新增的回归用例），覆盖拉平与增益的数值、孪生间响度一致、统计量、清单构造、`run` 的编排（用替身打分函数）、报告与判据的门控。
- WORLD 后端：真实 `pyworld==0.3.5` 的合成滑音自检通过；16 条真实音频的声学验收失败，见 [实测汇总](validation/real-audio-smoke.json)。
- 已静态核对：ms-swift 快照中 Omni 模板的 `_encode` 与 collator 对没有音频的输入有分支处理。
- 已在服务器上跑通（2026-11-23，WORLD 后端，16 条）：`prepare`、冻结基座与 LoRA 两臂的全部 10 个条件，含无音频输入和 ASR。
- **未验证**：Praat 后端的真实调用（这里无法安装 `praat-parselmouth`）；它是否把重合成损伤降到阈值内。步骤 2 和步骤 3 用来在服务器上回答这两点。

## 第一次冒烟的发现（2026-11-23，WORLD 后端，16 条）

| 现象 | 数值 | 处理 |
|---|---|---|
| 原样重合成就损伤词的可懂度 | 冻结基座 WER：原录音 8.3%，`sham` 24.9%，`flat_both` 32.7%；`sham − orig` 的区间为 [+5.1, +26.1] pp | 默认后端改为 Praat 叠接相加 |
| F0 估计在这批录音上不稳 | 原录音的 F0 标准差中位数 4.6 半音，个别 8–10 半音；拉平后重新分析仍有 0.4–4 半音 | 同上。Praat 的音高跟踪同时用于编辑和验收 |
| 孪生之间响度不一致 | 2 条近满幅录音的 `sham`、`flat_f0` 被单独压低 25–33% | 改为同一语句的全部孪生共用一个增益 |
| 能量验收偏高 | `flat_energy` 残留为原来的 0.70 | 原实现在孪生自己的有声段上测量，而拉平会把安静段抬进有声段；改为统一在原录音的有声段上测量。±12 dB 的增益上限未改，是否足够要看重跑结果 |

那次运行的模型层面数字（UAR、各对比）不作任何解读。

## 第一次全量运行的结果与第二版协议（2026-11-23）

第一版协议（`d2-iemocap-session4.json`，Praat 后端）在 Session 4 的 1026 条上跑完了冻结基座和 LoRA-mixed：

- 声学验收、重合成不改变答案、重合成不损伤词三项都通过。
- 拉平损伤了词（WER 升高 3.0–4.0 pp，区间不含 0），按门控主对比不可解读；不看门控时两臂也都落在"不确定"。
- 只给问题时模型全选 A，给原录音时基座仍有 559/1026 选 A。选项顺序固定，位置偏好与情绪判断混在一起。
- 事后分析：用每条原录音的 F0 中位数、F0 标准差、能量标准差、响度四个数（说话人内标准化）做留一对话的逻辑回归，UAR 49.6%，接近基座听原录音的 52.0%。类别差异集中在音高高度和响度上，轮廓起伏的差异很小，而第一版的孪生恰好保留了高度、只抹掉了起伏。

因此在任何新条件被打分之前声明第二版协议 `d2b-iemocap-session4.json`，与第一版共用数据、孪生参数、bootstrap 和判据阈值，改三处：

| 改动 | 内容 |
|---|---|
| 高度孪生 | `level_f0`：整体平移 F0，使该句中位数等于该说话人的常用音高；`level_energy`：把 RMS 设为该说话人的常用响度；`level_both`：两者；`neutral`：F0 恒定在常用音高、能量拉平、响度取常用值。参考值是该说话人在本集合内各句中位数的中位数，不使用标签 |
| 选项轮换 | 每条语句在四种循环排列的选项顺序下各问一次，报告按类别计分而不是按字母；另列各字母被选中的比例和各排列下的 UAR |
| 逐孪生的可懂度检查 | 每一种孪生都做 ASR；每个主对比只在它自己的孪生没有损伤词时才给结论 |

主对比有三个，各自独立判定：`flat_both − sham`（轮廓起伏）、`level_both − sham`（高度与响度）、`neutral − sham`（两者都去掉）。判据阈值不变。

读法：如果基座在 `level_both` 和 `neutral` 上都不掉分，而四个声学统计量就能区分类别，这是"信息在、行为上没用"的直接证据，TRR 的反事实应改为高度归一化；如果明显掉分，说明模型已经在用，TRR 在这个任务上的前提不成立。

运行方式与第一版相同，只把 `--config` 换成 `experiments/trr_diagnostics/d2b-iemocap-session4.json`。分类条件的耗时约为第一版的 4 倍：按第一次全量的实测（每个音频条件 155 秒、每个 ASR 条件 105 秒），每臂约 1.5 小时。

## 局限

- 孪生只改 F0 和慢变能量（起伏与高度），时长、停顿、音质不变；结论只对这两个因素成立。
- 高度孪生的参考值按说话人取，Session 4 只有两位说话人；参考值来自同一批语句，模型在单句输入下并不知道说话人的常用音高。
- 重合成有损伤，`sham` 用来把它从拉平效应里分离出来，但不能排除"拉平后的合成比原样合成更不自然"带来的额外影响；可懂度检查只覆盖词的层面。
- IEMOCAP 是表演语料，Session 4 仅两位说话人。
- `text` 条件的提示只有一种写法；它是参照，不是文本上限。
- bootstrap 区间反映样本变异，不是训练种子变异。
