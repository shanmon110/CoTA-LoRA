# 跨语料实验声明：IEMOCAP 上训练的适配器在 CREMA-D 上（2026-11-24）

写于任何模型在 CREMA-D 上被评测之前。实现：`experiments/hearside_pilot/crosscorpus.py`、`run_cross.sh`。

## 为什么做

前三轮都在 IEMOCAP Session 4 上，属于探索性结果。这一轮有两个用途，用同一批评测完成：

1. **确认。** “只改最后 4 层得到的增益是格式性的，全深度得到的增益可迁移”能否在一个没参与过任何选择的语料上重现。
2. **最后一次找缺口。** 普通的全深度 LoRA（`D-all-shuffled`）在没见过的语料上是否还有增益。没有的话，跨语料迁移就是一个值得设计方法的场景；有的话，方法路线在现有范围内可以放下。

## 语料与选取

CREMA-D（Cao 等，2014）：91 位演员，每人用多种情感朗读同样的 12 个句子，共 7,442 段，六种情感。句子文本对情感标签不含信息，所以不设“只给转写”的对照。原论文报告的人类仅凭音频识别预期情感的正确率为 40.9%（六类），因此绝对分数预期明显低于 IEMOCAP。

- 只取与训练类别对应的四类：ANG→angry，HAP→happy or excited，SAD→sad，NEU→neutral。disgust、fear 不用。
- 标签用演员的预期情感（文件名中的代码），不用众包感知标签。
- 每位演员每类取 3 段，固定种子 42 抽取，预计 1,092 段。强度等级不筛选。
- 不训练，不在 CREMA-D 上选 checkpoint 或调任何设置。

## 评测

冻结基座和此前训练的 13 个适配器（`A-all`、`B-audio`、`D-all-shuffled`、`A-last4` 各三个种子，`C-post-audio` 一个）回答：

- 训练时的四选一字母题，四种选项顺序（生成设置与 Session 4 相同）；
- `formats.py` 的三种问法（yesno、word、open），解析规则和词表不变。

分数为按类平均召回率。区间：对演员（91 个）和训练种子联合重采样，10000 次，种子 42。

## 预先声明的判读

| 名称 | 条件 | 读法 |
|---|---|---|
| `full_depth_gain_on_unseen_corpus` | `D-all-shuffled` 减基座，字母题全部顺序。基座 ≥ 85% → `base at ceiling`；区间上界 < 0 → `harmed`；增益 ≥ 8.7 pp（语料内增益 17.4 的一半）且区间下界 > 0 → `transfers`；其余 → `gap` | `gap` 或 `harmed`：跨语料是一个普通做法不够用的场景，值得设计方法，但仍需第三个语料确认。`transfers`：这条找缺口的路没有结果 |
| `last_layers_bind_to_option_order_on_unseen_corpus` | `A-last4` 的顺序差减 `A-all` 的顺序差 ≥ 10 pp 且区间下界 > 0 → `replicated`，否则 `not replicated` | |
| `last_layers_gain_is_format_specific_on_unseen_corpus` | 只看 `A-all` 相对基座有增益（区间下界 > 0）的问法；在所有这些问法上，`A-last4` 减基座的区间上界都低于 `A-all` 增益的一半 → `replicated`；没有这样的问法 → `not testable`；否则 `not replicated` | |

需要事先说明的一点：如果 `A-last4` 在新语料的训练顺序上本来就没有增益，顺序差会很小，第二条会判为 `not replicated`。这种情况下应读作“后段适配在新语料上什么都没带来”，而不是“后段适配没有格式捷径”；汇总表里训练顺序、其他顺序两列会显示是哪一种。

其余对比（B − A、D − A、C 相对基座等）只作描述，不设判据。

## 已知局限

- 表演语音、固定句子，与 IEMOCAP 的对话语音差别很大；“迁移”同时跨了说话人、录音条件、语体和标注方式，无法区分是哪一项造成差异。
- 预期情感标签与听感不总一致，绝对分数有上限。
- 四类中的 happy 在 IEMOCAP 一侧是 happy 与 excited 合并。
- 仍是同一个基座模型（Qwen2.5-Omni-3B）、同一个训练语料。
