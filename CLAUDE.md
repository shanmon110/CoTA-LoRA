# Claude 接手 CoTA-LoRA

用户希望以 CoTA-LoRA 为主线形成 ACL 方法或分析贡献。请独立核实源码与证据，允许推翻目前的机制解释。当前材料不足以声称稳定优于匹配 LoRA 或达到 SOTA。

## 阅读顺序

1. [完整交接](docs/CoTA-LoRA_CLAUDE_HANDOFF.md)，尤其第 4、5、8、9 节。
2. [28 篇论文索引](research/papers/README.md) 和 [实验记录](research/evidence/README.md)。
3. [CoTA runtime](experiments/ms_swift_librispeech/swift_asr_ta_runtime.py)、[插件](experiments/ms_swift_librispeech/plugin.py)、[训练入口](experiments/ms_swift_librispeech/train.py)。
4. [论文方法插件](experiments/ms_swift_aqa/paper_plugin.py)、[逐方法配置](experiments/ms_swift_aqa/papers28/configs)、[统一运行与重载验证](experiments/ms_swift_aqa/papers28/run.py)。
5. [复核与环境说明](research/REPRODUCIBILITY.md)。

## 需要准确理解的现状

- 当前 CoTA 为 Omni 36 层 O-only、首答案查询、问题 embedding 均值与音频—文本 attention 聚合差条件门。它不包含最优传输求解器或对比学习损失。
- 当前动态版拒绝 Omni 梯度检查点；旧的 QKVO 静态 TA 和 Qwen3-ASR 路径不能证明这个动态版本已经支持对应功能。
- 匹配 LoRA 为 r12/alpha24，CoTA 为 r8/alpha16，scale 都为 2。不要重复把这组比较解释为缩放不匹配。
- 完整关闭传输需关闭共享门项和独立 C 项；仅 `g=0` 仍会保留独立传输。
- 28 篇表与 CoTA 主比较使用不同训练数据和模块覆盖。第 1 篇另有理论与教师构造诊断；特殊训练协议、额外成本及偏离论文之处都有记录。
- PiSSA、LoRA-GA、LowRA 在当前目标集全部选 A；LoRA+、EVA 也明显退化。应先排查初始化/量化/优化，再解释方法差异。
- MMAR、MMAU、IEMOCAP 已被探索性观察。请记录后续选择流程，保留阴性结果。

## 建议先做的研究

先复核首答案位置、模态 mask、cache、真实可训练集合及 R/g/C/A/B 的更新。对严重退化方法比较干净基座、初始化后、首个更新后及少量更新的 logits/loss/答案分布。再做位置一致、预算接近的静态门、强制 m=1、独立 C、额外普通 LoRA 等对照，判断动态条件门的独立作用。

保留历史 checkpoint、日志和原始清单；新实验使用新目录。用户此前采用固定 3B、单卡、现有语料的研究范围。本文件不启动任务、不更改协议；后续具体训练安排由用户与接手者确定。

## 审查与运行边界

源码可以直接阅读；真实模型实验还需要服务器中的权重与合法数据。不要把聚合记录当成能替代逐题预测的完整评分输入。原实验记录中的绝对路径和 SHA 保留用于追踪；[导出清单](research/EXPORT_MANIFEST.json) 核对的是此 GitHub 快照。

交付应包括：已证实的问题、修复与回归证据、预先声明的公平协议、原始日志和逐题结果、多种子及机制消融、对 ACL 贡献是否成立的诚实判断。
