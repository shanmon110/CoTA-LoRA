# 28 篇 LoRA 工作在 Qwen2.5-Omni-3B 上的评测

本目录对应用户提供的 28 篇清单。每篇的编号、题名和会议原文保存在 [catalog.json](catalog.json)。训练复用本工程的 ms-swift SFT、音频处理、生成和开发集选模。特殊投影通过 [Swift tuner 插件](../paper_plugin.py) 注册。原生 PEFT 可表达的方法沿用各自实现。研究性训练流程与必要的算法差异见分组来源说明。[第 1–10 篇](sources-01-10.md) · [第 11–20 篇](../paper_middle_sources.md)

## 实验条件

- 模型为 Qwen2.5-Omni-3B，revision `f75b40e3da2003cdd6e1829b1f420ca70797c34e`。语言侧为 36 层 Thinker。快照清单包括 processor。训练环境为 ms-swift 4.7.0.dev0、PEFT 0.15.2、Transformers 4.52.3。RandLoRA 单独加载 PEFT 0.17.1。
- 训练和开发分别使用 AVQA 的 2,048 和 256 道题，数据按来源隔离。选择开发集准确率最高、同分最早的 checkpoint。单个 checkpoint 经独立重载，先复验开发逐题输出，再用于 MMAU 和 IEMOCAP。LoRAGen 的固定生成流程有独立来源和成本记录。
- MMAU 使用完整 Test-mini 的 1,000 题，合法选项字母贪心解码，映射回原始选项文本并执行官方评分器。IEMOCAP 使用 Session 5 的 1,241 条四分类话语，happy/excited 合并，主要指标为四类召回的算术平均 UAR。两个目标集仅用于迁移评估。训练数据和协议与上一轮 2,688 条长短混合 AVQA 实验有别。
- 常规配置为 rank 8、alpha 16，Q/K/V/O 与 MLP 三个线性层。实际物理 batch 为 4，梯度累积 4，生成 batch 为 2，AdamW、固定基础学习率 0.0002、种子 42，最多两轮。每 32 步验证，patience=3。同一张 RTX 5090 通过文件锁串行使用。必要的方法参数、投影范围和重算开关见 [配置目录](configs)。
- 第 1 篇的构造实验需要已知目标权重，运行作者提供的数值验证。第 9 篇采用无额外对齐预训练的 LoRAM-Rand 消融。LongLoRA 在原生上下文长度以内检验训练期移位分组注意力及可训练的嵌入和归一化。RoLoRA 在单卡上模拟 4 个不共享训练录音的客户端。LowRA 只量化 252 个语言侧投影，除编码以外还需存放码本、尺度及 adapter。额外阶段的耗时和容量单列。冻结基座、普通 LoRA 给出对照。

## 运行

第 1 篇另有真实模型的逐层诊断。它从同一 AVQA 训练/开发划分得到的全参教师读取已选权重，在 36 层的 72 个方形 Q/O 投影上应用作者的已知目标线性构造，生成一个 rank 8 adapter。此臂单列教师依赖、构造成本及两个目标集的分数。普通 LoRA 额外跑一组开启梯度重算的配置，用于与同重算设置的实验比较训练资源。

从仓库根目录使用已有环境。每次新实验指定独立目录，输出、配置、checkpoint 身份和逐题预测均保留在该目录。选取的配置文件名对应方法 ID。

```bash
out=/root/autodl-tmp/MMAR/runs/papers28-transfer-new
experiments/ms_swift_aqa/papers28/run.sh --out "$out" --archived lora vera dora fourierft
experiments/ms_swift_aqa/papers28/run.sh --out "$out" lora-plus bora toplora
python -m experiments.ms_swift_aqa.papers28.report --out "$out"
```

`--archived` 复用 AVQA 2,048 题同协议的开发选定权重、复核其原始逐题输出，补齐目标集预测。省略该选项可从干净基座训练。原始训练与推理文件、哈希、计时及日志路径记录在各臂的结果 JSON。LowRA 需要已封存的压缩基座；LoRAGen 需要来自同一源训练清单的专家库，先运行 LoRA-Mixer。独立的第 1 篇 CPU 复现入口见 [作者实验说明](sources-01-10.md)。

`REPORT.md` 展示有完整评测的条目及其余条目的阶段。每个新结果必须通过独立重载、开发集预测复验（固定生成模型单列）、完整测试覆盖、参考答案顺序检查和官方 MMAU 评分。复验失败写入 `status-方法.json` 和 `logs/`。

准备好同一协议的官方数据、模型和 LowRA 压缩基座后，可用以下入口完成整张表。它复用同目录已核实的结果，等待 LoRA-Mixer 训练出源任务专家库，再生成 LoRAGen 的权重。若已有同实验的训练进程，可通过 `--worker-pids` 传入 PID，同时更新报告并在这些进程退出后补跑缺项。

```bash
out=/root/autodl-tmp/MMAR/runs/papers28-transfer-new
mkdir -p "$out"
/root/autodl-tmp/iemocap/swift-ser/venv-5090/bin/python -m experiments.ms_swift_aqa.papers28.finish_queue --out "$out"
```

每个方法的作者源码版本、采用的实现路径和偏离原协议之处可按论文编号核对。[第 1 至 10 篇](sources-01-10.md) · [第 11 至 20 篇](../paper_middle_sources.md) · [第 21 至 28 篇](sources-21-28.md)

运行目录的 `sources/` 提供内容寻址的源码归档和 SHA-256 清单，包含 WEST、editable ms-swift、两版 PEFT 及使用的作者代码。新启动的实验驱动在阶段日志首行记录源码快照。较早启动的进程和历史复用结果以各自日志中实际记录的来源为准。

`python -m experiments.ms_swift_aqa.papers28.data_audit --out "$out"` 会重新解码四份清单的全部音频，核对原始 PCM 索引、AVQA 训练与开发来源、六对清单的 PCM 交集及 IEMOCAP 最终 Session，输出 `split-audio-audit.json`。队列结束时，`disposition.json` 记录每一项最终验收情况。
