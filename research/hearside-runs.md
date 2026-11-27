# 听侧实验索引：运行 → 声明 → 读数

2026-11-27 建立。这份文件把听侧这一线的每一次运行、它事先写下的声明、以及最终读数放在一处，方便核对"哪一条结论来自哪一次运行"。每一行的读数都是事先声明的判据在代码里的实现（`experiments/hearside_pilot/` 下各模块的 `judge` 函数），不是事后叙述。

原始预测、逐题结果、日志与训练元数据在 `experiments/hearside_pilot/results/<运行名>/`；服务器上的完整运行文件夹在 `/root/autodl-fs/CoTA-LoRA/<运行名>/`。

## 运行顺序

| # | 运行 | 服务器文件夹 | 声明 | 读数 |
|---|---|---|---|---|
| 1 | 试点：五种适配在 IEMOCAP Session 4 | `hearside-pilot-20261123-160928` | `research/related-work-hearside-20261123.md` | 后 4 层的增益绑定在训练时的选项顺序上；全深度的增益可迁移 |
| 2 | 跟进：另两个种子、四种新问法、余量检验 | `hearside-followup-20261123-212622`、`hearside-next-20261124-031913` | `research/headroom-test-input-conditional-20261124.md` | 绑定在另两个种子上也成立；新问法上没有增益 |
| 3 | 跨语料：CREMA-D | `hearside-cross-20261124-213539` | `research/cross-corpus-cremad-20261124.md` | 同一读法复现：后 4 层 `format-bound`，全深度 `transfers` |
| 4 | 层段对照：前 4 层、中段 4 层 | `hearside-bands-20261125-045245` | `research/layer-bands-20261125.md` | 浅层段 `transfers`，后 4 层 `format-bound`；顺序绑定 **specific to the last layers** |
| 5 | 线性探针 | `hearside-probes-20261125-093959` | `research/probes-20261125.md` | Session 4 上感知落差 `present`，CREMA-D 上 `unclear`；两个语料读法不一致，不算确认 |
| 6 | 字母偏置替身（第二步） | `hearside-letters-20261125-170918` | `research/letter-bias-20261125.md` | 两个语料都 `mostly a letter bias` |
| 7 | 第二个任务：AVQA 训练、MMAU 测试 | `hearside-aqa-20261125-224035` | `research/second-task-and-grpo-20261125.md` 第 1a 节 | 四个臂全部 `no gain`；主张在这个任务上 `unclear` |

## 每条声明文档里的读数是什么

| 声明 | 判据在哪 | 读法词汇 |
|---|---|---|
| `research/related-work-hearside-20261123.md` | `experiments/hearside_pilot/followup.py` 的 `judge`（试点本身的读数由同一套判据在跟进轮里给出） | `transfers` / `format-bound` / `no gain` |
| `research/headroom-test-input-conditional-20261124.md` | `experiments/hearside_pilot/followup.py`、`headroom.py` | 见各文档 |
| `research/cross-corpus-cremad-20261124.md` | `experiments/hearside_pilot/crosscorpus.py` 的 `judge` | `transfers` / `format-bound` / `replicated` / `not replicated` / `gap` / `harmed` / `base at ceiling` |
| `research/layer-bands-20261125.md` | `experiments/hearside_pilot/bands.py` 的 `judge` | 同上，加上顺序绑定的 `specific to the last layers` / `in every shallow band` / `mixed` |
| `research/probes-20261125.md` | `experiments/hearside_pilot/probes.py` 的 `judge` | `present` / `unclear` / `yes` / `no`；`no more than a linear readout of the base` / `between` |
| `research/letter-bias-20261125.md` | `experiments/hearside_pilot/letters.py` 的 `judge` | `mostly a letter bias` / `not a letter bias` / `partly a letter bias` |
| `research/second-task-and-grpo-20261125.md` 1a | `experiments/hearside_pilot/aqa.py` 的 `judge` | `no gain` / `transfers` / `prompt-bound` / `partial`；主张 `holds` / `does not extend` / `unclear` |

## 还没做的

- `research/second-task-and-grpo-20261125.md` 第 1b 节（公开的 GRPO 模型 `mispeech/r1-aqa`）：仍是草案，等模型许可确认和用户对放宽到 7B 的决定。
- `research/headroom-test-input-conditional-20261124.md` 的余量检验：已在第三轮跑完，三种分组全部读作 `no`，总体 `closed`；声明文档的状态行已回填。
- 层段对照的"固定训练步数"对照：现有比较用的是各臂各自最好的检查点，步数不同（见 `results/bands-20261125-045245/NOTES.md` 末节）。

## 这条线目前能支持什么

- 在封闭集情感分类上，只适配靠近输出的层得到的增益绑定在训练时的提示上（选项顺序、问法）；适配前面的层，增益可迁移。这个读法在 IEMOCAP Session 4 和未参与训练的 CREMA-D 上都复现，两个语料的顺序差一致指向"只发生在最后几层"。
- 在同一任务上，**后 4 层的这种行为在很大程度上可以用"给字母加常数"解释**：三个数的替身重现了后 4 层的增益、损失，以及换顺序后偏向固定字母的具体签名。这是充分性，不是机制上的同一性。
- 在第二个任务（音频问答）上，**没有观测到任何增益**，因此这条线的主张在那里既没有成立也没有被推翻，读作 `unclear`。
