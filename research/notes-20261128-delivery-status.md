# 本次接手的交付清单：已完成 / 未完成 / 明确不做

日期：2026-11-28。分支 `research/analysis-20261127`（已快进合并到 `main`，`main` 已推到 GitHub）。
这一份只回答"交付了什么、什么还没做"，不重复结论；结论在
`notes-20261127-diagnosis-and-plan.md` §4.2.2 和 `notes-20261128-session5-confirmation.md`。

## 1. 交接文档要求的五项交付，逐项对照

| 交付项 | 状态 | 在哪里 |
|---|---|---|
| 独立重算全部头条数字 | **完成** | `research/notes-20261127-recompute.md` |
| 现状诊断（最强的 3–5 条、最弱的 3–5 条） | **完成** | `research/notes-20261127-diagnosis-and-plan.md` §2.1 / §2.2 |
| 下一步建议（优先级、预声明判据、成本） | **完成**，其中 P1' / P0b / P3 / P4 已执行 | 同上 §3 |
| 需要推翻的结论，明确说清 | **完成** | 同上 §4.1 |
| 是否应改成分析论文 + 骨架 | **完成** | 同上 §4.3 |

## 2. 本轮新跑的五条实验（全部无训练）

| 编号 | 内容 | 成本 | 结果 |
|---|---|---|---|
| **P1'** | IEMOCAP Session 5 上确认标签自由类别边缘校正 | 8 次推理 | **CONFIRMED**（三种子各自成立） |
| **P0b** | AQA 上槽位 vs 类：负迁移是不是位置捷径 | 纯读已跟踪预测 | **不是**位置捷径；后 4 层臂额外有偏好字母偏置 |
| **P0 步骤 1** | AQA 负迁移落在需要音频的题上还是语言漂移 | 纯读已跟踪预测 | **是音频通路的伤害**，12 个臂一致，区间不含 0 |
| **P3** | CREMA-D 同句同演员不同情感的韵律对照 | 纯读已跟踪预测 | **韵律线保住**（判据 HELD），展平对照仍禁用 |
| **P4** | 五种文本打分规则并排 | 纯读已跟踪预测 | **规则会改变结论**，必须同报严格与宽松两列 |

预声明判据写在跑之前：`research/notes-20261128-session5-confirmation.md`（提交 `17db41b`）。

## 3. 本轮新增/修复的代码

| 文件 | 作用 |
|---|---|
| `experiments/hearside_pilot/session5.py` | P1' 的判据与逐臂分析 |
| `experiments/hearside_pilot/session5_manifest.py` | Session 5 清单构造（与分析模块分开，避免互相覆盖） |
| `experiments/hearside_pilot/run_session5.sh` | P1' 的运行脚本（先跑单元测试，结果不合格就不跑） |
| `experiments/hearside_pilot/aqa_logits.py` | AQA 选项 logit 提取（与已跟踪答案 8/8 一致） |
| `experiments/hearside_pilot/avqa_holdout.py` | 按源视频切分的 AVQA 留出集（评测折与储备折**零视频重叠**） |
| `experiments/hearside_pilot/run_queue.sh` | 六步队列，失败不阻塞后续步骤 |
| `experiments/hearside_pilot/run_avqa_same_dist.sh` | P0：同分布 AVQA 评测 |
| `experiments/hearside_pilot/run_aqa_surrogate.sh` | P5：AQA 替身检验 |
| `research/tools/session5_verdict.py` | 逐种子读数 + 逐臂类别边缘 |
| `research/tools/aqa_slot_vs_class.py` | P0b |
| `research/tools/avqa_vs_mmau.py` | P0 的比较与 bootstrap |
| `research/tools/prosody_control.py` | P3 |
| `research/tools/aqa_surrogate.py` | P5 |
| `research/tools/text_rules2.py` | P4 |
| `research/tools/aqa_audio_strata.py` | P0 步骤 1 |

## 4. 正在跑 / 排队中

`experiments/hearside_pilot/run_queue.sh`，日志 `/root/autodl-fs/CoTA-LoRA/queue.out`，
结果 `/root/autodl-fs/CoTA-LoRA/queue-20261128-174926/results/`。

- **Q1–Q4：已完成**（54 秒，全部零 GPU）。
- **Q5（P0 步骤 2，同分布 AVQA）**：运行中。13 个臂 × 3,001 题 × 4 个选项序
  ≈ 12,004 行/臂，实测约 37 分钟/臂，**总计约 8 小时**。
  这是第 5 章剩下的**唯一空洞**：跨语料 vs 跨任务的归因。
- **Q6（P5，AQA 替身）**：排在 Q5 之后。
- **P2 故意不做**，它自己的前提是"只在 P0 给出明确归因后才做"——Q5 落地后再决定。

## 5. 明确**没有**做的事（必须记下来）

- **没有训练任何模型。** 本轮全部是一次性推理或只读分析。
- **没有用 Session 5 的标签做任何拟合**，标签只在最后算 UAR 时使用。
- **没有因为结果不利改判据**；Session 5 的判据在跑之前写下。
- **没有在 `main` 上做过实验**；所有提交在 `research/analysis-20261127`，
  最后快进合并到 `main`（`origin/main` 是祖先，无冲突）。
- **没有跑 28 篇表的复现**，那需要 28 次训练，本轮范围之外。
- **D2b 的展平韵律对照的数字仍然不得用于任何韵律结论**（运行的闸门禁止），
  韵律线只由 P3 的 CREMA-D 对照支撑。

## 6. 已知的未决项

- **`git stash` 里有一条冗余条目** `2e8bb1852687c091683055ed4dd16045c5374a04`
  （"cota-analysis-20261127-premerge"）。它只含一个文件
  `research/tools/emotion_prior_all_arms.py`，该文件在分支上已有更好的版本
  （逐臂读数 + 簇 bootstrap，而非整语料拟合）。**内容已确认不同但已过时**，
  自动 drop 被权限分类器拦下，需要人工执行：
  ```
  git stash drop 2e8bb1852687c091683055ed4dd16045c5374a04
  ```
- **GitHub 推送在本次会话中多次遇到 TLS 连接中断**（`GnuTLS recv error (-110)`）。
  **所有提交最终都已推送**，`main` 与 `origin/main` 同步在 `411e15f`。
  如果你在别处看到本地领先远端，是网络问题，不是漏推。
- **本地 `integration/trr-diagnostics-20261123` 分支落后 `origin/main` 52 个提交**，
  它在另一个 worktree（`/root/autodl-tmp/trr-diagnostics/integration-src`）。
  本轮没有动它。
- **`research/evidence/` 的历史记录没有改**，符合仓库约定。
