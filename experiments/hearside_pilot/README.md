# 听侧适配小实验（hear-side pilot）

**状态（2026-11-23）：代码和 CPU 单元测试已完成，尚未在真实模型上运行。本目录不含任何实验结果。**

## 要回答的问题

D2/D2b 诊断在 IEMOCAP Session 4 上得到两件事：

- 冻结基座对音高、能量的改动有反应，但各类别此消彼长，UAR 上相互抵消。
- 普通监督 LoRA（pair-01）在训练时的选项顺序下 UAR 67.7%，且确实依赖韵律；把同样的四个选项换个顺序，UAR 降到 40–45%，低于冻结基座的 52–60%。它学到的一部分是“音频 → 答案字母”的捷径。

由此提出一个可以被否定的假设：**把适配器的更新只加在音频 token 的位置上，让冻结的语言模型自己读题、作答，可以学会用声学信息而不绑定答题格式。** 小实验同时检验一个更便宜的解释：训练时打乱选项顺序就够了，位置限制没有独立作用。

这不是新结构的主张。位置受限的低秩更新本身很简单，相关工作的核查尚未完成（见“局限”）。小实验只决定这个方向值不值得继续。

## 四个臂

| 臂 | 更新加在哪些位置 | 训练时的选项顺序 | 作用 |
|---|---|---|---|
| `A-all` | 全部位置 | 固定 | 普通低秩适配，参照 |
| `B-audio` | 只在音频 token 位置 | 固定 | 待检验的做法 |
| `C-post-audio` | 只在最后一个音频 token 之后（问题、答案） | 固定 | 机制对照：只改“答”的一侧 |
| `D-all-shuffled` | 全部位置 | 四种循环顺序，类内均衡 | 便宜的替代解释 |

四个臂只在表中两列上不同，其余完全相同：

- Thinker 全部 36 层的注意力 O 投影，r=8、alpha=16，`gate_multiplier=0`（没有传输项，就是 B·A·x）。每臂 1,179,936 个可训练参数，其中每层 8 个 gate 参数不起作用。
- 用 36 层而不是 pair-01 的最后 4 层，是因为最后一层在音频位置上的更新到不了答案位置（单元测试 `test_audio_updates_in_the_last_layer_cannot_reach_a_text_answer`），听侧更新必须落在更早的层。
- 训练配方取自服务器上的 `pair-01/config.json`（学习率 2e-4 常数、batch 2 × 累积 4、1 个 epoch、seed 42），只把评估间隔改为 88 步。
- **Sessions 1–2 训练（2108 句左右，264 步），Session 3 选 checkpoint（第 88、176、264 步三次），Session 4 只评测，Session 5 不读取。** `data.py` 发现源清单含 Session 4/5 会直接报错。pair-01 当时用 Session 4 选 checkpoint，这里不再这样做。
- `D` 的训练集和选模集与其他臂是同一批语句、同一顺序，只是每句的选项被循环移位（`data.py`）。选模用的是训练器现有的指标，按答案字母计 UAR；对 `D` 来说字母与类别不再一一对应，这个数只用来挑 checkpoint，不与其他臂的 Session 3 成绩比较。

位置限制在 `experiments/ms_swift_librispeech/swift_asr_ta_runtime.py` 里实现（`update_positions`，checkpoint 格式 10）。不设这个选项时，原有格式 1–9 的读写和前向不变。

同一次提交里还有一处与位置无关的改动：乘子为 0 且没有独立 C 矩阵时，传输项恒为零，runtime 不再计算它的注意力对比项。输出与 A、B 的梯度逐位不变（`test_zero_multiplier_skips_the_contrast_without_changing_outputs_or_gradients`），省掉每层两张 FP32 注意力图及其反向缓存；gate 参数此时没有梯度（原先是零梯度）。四个臂都走这条路径。

## 评测

`pilot-iemocap-session4.json`，沿用 D2b 的语句、问题和四种选项顺序（1026 句 × 4 = 4104 行），按类别计分。

| 输入 | 内容 |
|---|---|
| `orig` | 原始录音，训练时的问法 |
| `orig_reworded` | 原始录音，另一种问法和版式（所有臂都没见过） |
| `text` / `question_only` | 只给转写 / 只给问题 |
| `asr-orig` | 300 句子集的转写，WER 在去掉“The transcript is:”这类引导语后计算 |

“训练顺序”指 A=angry、B=happy、C=sad、D=neutral 这一种，“其他顺序”指另外三种循环移位。区间是按对话聚类的 95% bootstrap（10000 次，seed 42）。

`B-audio` 对不含音频的输入不做任何更新，所以它在 `text` 和 `question_only` 上与冻结基座逐位相同。这是构造决定的，不是实验发现。

## 预先声明的判据

写在 `summarize.py` 的 `CRITERIA` 和 `decide` 里，训练前固定。

| 检查 | 对比 | 通过条件 |
|---|---|---|
| 听侧学得会 | `B − A`，`orig`，训练顺序 | UAR 差 ≥ −3 pp |
| 听侧没有丢掉读题能力 | `B − 冻结基座`，`orig`，其他顺序 | UAR 差 ≥ 0 |
| 听侧不绑定格式 | `B − A`，`orig`，其他顺序 | UAR 差 ≥ +10 pp |
| 打乱选项不能替代 | `B − D` | 改写问法（全部顺序）UAR 差的区间下界 > 0，**或** WER 差的区间上界 < 0 |

四项全过记为 `continue`，任何一项不过记为 `stop`，缺臂记为 `undecided`。

读结果时的三种情形：

1. 前三项通过、第四项不过：打乱选项就能得到同样的效果，位置限制没有独立价值，这条线停止。
2. 四项全过：位置限制本身就有用。能写的是“简单做法 + 诊断”，不是新结构。
3. 第一项或第二项不过：只改音频位置学不会，或者学会了却损害读题。这时才有理由讨论额外的组件，也可能说明假设本身不成立。

`C-post-audio` 不进判据。它用来看格式绑定是否来自“答”的一侧：如果 C 与 A 一样绑定格式而 B 不绑定，机制解释才站得住。

## 运行

服务器上一条命令（克隆 `main` 的当前版本并在后台运行）：

```bash
R=/root/autodl-fs/CoTA-LoRA; S=$R/src-$(date +%Y%m%d-%H%M%S)
git clone --depth 1 https://github.com/shanmon110/CoTA-LoRA.git "$S" && cd "$S" \
  && nohup bash experiments/hearside_pilot/run_pilot.sh > "$R/hearside-pilot.out" 2>&1 &
```

`run_pilot.sh` 依次做：环境检查 → 单元测试（不通过就停，不训练）→ 生成清单和配置 → 评测冻结基座 → 每个臂训练后立刻评测 → 汇总 → 打包。所有输出在 `$R/hearside-pilot-<时间>/`，结束时（包括中途失败）生成 `$R/hearside-pilot-<时间>-for-claude.tar.gz`，只含日志、清单、预测和汇总，不含权重和音频。

看进度：`tail -f /root/autodl-fs/CoTA-LoRA/hearside-pilot.out`。中断后续跑：`RUN=<同一个输出目录> bash experiments/hearside_pilot/run_pilot.sh`，已完成的步骤不会重做，被打断的训练目录改名保留后重新训练。

耗时是估计值，约 4 小时：评测部分按本机实测推算（4 种顺序的 `orig` 约 10–11 分钟、`text` 和 `question_only` 各约 1 分钟、`asr-orig` 约 2 分钟，5 个模型）；36 层的训练耗时和显存没有实测过。

## 验证状态

| 内容 | 状态 |
|---|---|
| 位置掩码、格式 10 读写、缓存生成、零乘子跳过对比项（`test_update_positions.py`，13 项） | 在一台 CPU 机器上通过：torch 2.15.0、transformers 4.52.3、peft 0.15.2，ms-swift 用只含 `Tuner`、`tuners_map`、`import_external_file` 的替身 |
| 同一环境下 `ms_swift_librispeech/tests` 65 项、`ms_swift_iemocap/tests` 10 项、`research/claude-review/checks` 7 项 | 通过 |
| 故意改坏掩码的 5 种写法 | 全部被上述测试发现 |
| `experiments/ms_swift_aqa/tests` | 未运行（需要真实的 ms-swift 和 numba） |
| 不依赖 torch 的测试（`trr_diagnostics` 57 项、本目录 14 项） | 通过 |
| `run_pilot.sh` 的流程（正常、单元测试失败、单臂训练失败、续跑） | 用替身解释器和玩具语料走通；真实训练和评测未运行 |
| 服务器环境（torch 2.12.1、真实 ms-swift、GPU）下的单元测试、训练、评测 | **未运行**，是启动脚本的第一步 |
| 36 层训练的显存与耗时 | 未测 |

## 局限

- 每臂一个训练种子。汇总里的区间只反映评测噪声，不反映重新训练的波动。结果值得继续时，下一步先加种子。
- IEMOCAP 在本项目里已被多轮观察，Session 4 用于此前的诊断，也被 pair-01 用来选过 checkpoint。本实验属于探索性分析，结论需要在 Session 5 或另一个语料上按预先声明的流程确认。
- Session 4 只有两位说话人。
- 只有一种改写问法、一个任务、264 步训练。
- 相关工作核查未完成。此前检索到、但尚未精读的方向包括：只在语音位置加 LoRA 的做法、音频语言模型的选项顺序偏置、微调造成的格式特化。在核查完成之前，不对新颖性作任何表述。
- 阴性结果同样保留在输出目录和汇总里。
