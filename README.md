# CoTA-LoRA research handoff

基于 ms-swift 的 Qwen2.5-Omni-3B 音频微调研究仓库。所有微调方法的实现、训练和实验统一建立在 ms-swift 之上，不引入其他微调框架。

**分支约定：只有 `main`。** 开发、维护和服务器同步都使用 `main`。

**当前证据尚未证明 CoTA-LoRA 稳定优于匹配 LoRA。** 28 篇方法移植的工程验收，与原论文完整实验复现是不同层次的结论。本仓库保留成功与失败结果、协议差异及未解决的问题。

## 从这里开始

1. [Claude 接手入口](CLAUDE.md)：阅读顺序、代码导航和当前优先事项。
2. [完整交接文档](docs/CoTA-LoRA_CLAUDE_HANDOFF.md)：真实公式、全部结果、训练诊断、ACL 证据缺口。
3. [28 篇论文与实现索引](research/papers/README.md)：原文、作者代码、固定版本、实现文件和复现边界。
4. [实验材料](research/evidence/README.md)：原始报告、聚合结果、配置、训练曲线和验证日志。
5. [环境与复核方法](research/REPRODUCIBILITY.md)：本机路径、源码快照、测试和移植边界。
6. [第三方源码与许可](third_party/README.md)：ms-swift、PEFT 及评估代码的可读快照。

## 目录

```text
experiments/ms_swift_librispeech/  共用 SFT、CoTA runtime、插件和 checkpoint
experiments/ms_swift_aqa/          AQA 数据处理、评估、28 篇方法适配
  papers28/configs/               逐方法配置
  branches/                       BoRA、TopLoRA 复用的投影分支（原 west_adapters）
experiments/ms_swift_iemocap/      情感评估与数据处理
experiments/cota_review/           CoTA 对照臂的配置生成与预先声明的矩阵
experiments/trr_diagnostics/       D2 诊断：回答是否依赖 F0/能量轮廓
research/claude-review/          独立审查：诊断、协议、ACL 判断
docs/                            交接与历史研究计划
research/papers/                 28 篇论文索引与 BibTeX
research/evidence/               实验记录，不含原始题目或音频
research/validation/             本次导出后的复核记录
third_party/                    实验依赖的源码快照及许可
```

28 项实现、配置、作者来源和结果通过索引对应，复用同一套训练流程。

### 2026-11-23 的清理

本仓库最初从 WEST 工具包的工程目录导出。WEST 的模型、训练器、recipes、C++ runtime、demo 及其测试、工具和依赖清单已移除，两个只能靠 WEST 运行的早期试验目录（`experiments/librispeech-asr-pilot`、`experiments/preflight`）一并移除。ms-swift 实验仍在使用的两处旧代码已迁入 `experiments/`：

- `west_adapters/{base,bora,toplora}.py` → `experiments/ms_swift_aqa/branches/`（内容未改）
- `examples/omni_talora/prepare_manifest.py` 中的 IEMOCAP 读取函数 → `experiments/ms_swift_iemocap/prepare.py`

被移除的代码可从提交 `5bf7aa3` 及更早的历史取回。`research/evidence/` 的历史记录原样保留，其中仍会出现旧路径。`research/EXPORT_MANIFEST.json` 描述的是提交 `043b6fc` 的导出快照，对当前目录树运行 `tools/verify_handoff_export.py` 会报告这些已移除或已修改的文件。

## 方法与结果概览

CoTA-LoRA：**Conditional Transport-Aware Low-Rank Adaptation，条件传输感知低秩适配**。当前动态门控位于 36 层语言解码器的 O 投影，仅在预测首个答案 token 的查询位置加入传输分支。代码及旧 checkpoint 继续使用 `ta-lora` / `ta_lora` / `question_audio`，以保留加载语义。

| 同一混合训练协议 | IEMOCAP UAR | MMAU Acc | MMAR Acc |
|---|---:|---:|---:|
| 冻结基座 | 59.94% | 75.7% | 61.9% |
| 匹配 LoRA，r12/alpha24 | 63.74% | 72.5% | 61.0% |
| CoTA，r8/alpha16 | 62.96% | 72.1% | 60.5% |

两种 adapter 的缩放均为 2，参数量约 177 万。28 篇方法表采用另一套 AVQA2048、Q/K/V/O+MLP 协议；该表的 LoRAM 等强基线属于原作者方法。请先读协议，再比较数值。

## 本仓库的范围

- 包含项目源码、测试、配置、交接、论文链接与引用、可分发的依赖源码、聚合实验记录和内容哈希。
- 模型权重、adapter 权重、音频、训练/测试样本文本、环境安装目录保留在原服务器。原路径与标识见交接和各协议；需要它们才能重跑真实实验。
- 历史配置保留当时的绝对路径；这是可审查的研究快照，迁移到新机器需配置实际路径和依赖。不要直接启动旧的自动实验队列。
- 当前测试集已被多轮探索性观察。新研究需预先确定开发选模与最终检验安排。

根目录保留 [Apache-2.0 LICENSE](LICENSE)：迁入 `experiments/ms_swift_aqa/branches/` 的代码源自 WEST 工程。第三方目录遵循其各自许可。论文通过原始发布链接提供，不将所有论文或作者代码重新声明为本项目成果。
