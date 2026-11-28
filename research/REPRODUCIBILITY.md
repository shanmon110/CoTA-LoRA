# 环境、源码与复核

这个目录是为代码审查和研究交接整理的快照。原工程 `/autodl-fs/data/west-ta-baselines` 未因导出而修改；导出目录为 `/root/autodl-tmp/github-export/cota-lora-handoff-20261122`。

## 已固定的内容

- 当前项目文件原始 SHA 见 [project-source-manifest.json](project-source-manifest.json)。导出 README、忽略规则和交接链接有文档层面的调整，代码及配置保持原字节。
- 第三方源码来自既有归档 `83075b150b88cbe603e29d7aee7aeab976fc280b9482606f906d69eb80014b5a`。见 [第三方说明](../third_party/README.md)。该归档不能反向证明更早训练用过完全相同的全部源码。
- 模型为 Qwen/Qwen2.5-Omni-3B，revision `f75b40e3da2003cdd6e1829b1f420ca70797c34e`。逐文件标识见 [verified_snapshot.json](../experiments/ms_swift_librispeech/evidence/verified_snapshot.json)。
- 论文方法的原作者 commit、文件 hash、特有配置和适配差异见 [论文索引](papers/README.md)。
- 实验记录逐文件来源和哈希见 [证据目录](evidence/README.md)。聚合派生材料会明确标记，不冒充原始文件。

## 原服务器环境

Python 入口：`/root/autodl-tmp/iemocap/swift-ser/venv-5090/bin/python`。

| 项目 | 记录版本 |
|---|---|
| Python | 3.12.3 |
| Torch | 2.12.1+cu130 |
| Transformers | 4.52.3 |
| ms-swift | 4.7.0.dev0，editable 源码 |
| PEFT | 0.15.2 |
| RandLoRA 专用 PEFT | 0.17.1，单独依赖路径 |
| Accelerate / TRL | 1.7.0 / 0.18.2 |

完整环境快照保留在第三方 provenance 文件中。这里的历史版本记录不等于跨平台安装已验收；Torch/CUDA 与显卡驱动应由接手环境单独核对。原根目录的 WEST `requirements.txt` 已于 2026-11-23 移除；ms-swift 实验的依赖约束见 `experiments/ms_swift_librispeech/environment/`。

## CPU 定向测试

在已有兼容环境中，从仓库根运行：

```bash
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMBA_NUM_THREADS=2 \
  python -m pytest -q -rs \
  experiments/ms_swift_aqa/tests \
  experiments/ms_swift_librispeech/tests \
  experiments/ms_swift_iemocap/tests
```

RandLoRA 需单独 PEFT 0.17.1，LowRA 的 CUDA 测试需要真实 GPU，可选 ReFT 测试需要 pyreft。不要把这些跳过项解释为通过。本次导出后的实际日志见 [validation](validation)。

核对导出内容本身：

```bash
python tools/verify_handoff_export.py
```

该命令只读取清单与文件，检查字节哈希、可执行位和符号链接，不加载模型、不训练。清单覆盖导出产物，排除清单自身及 Git 元数据。

## 分析线（2026-11-27 起，分支 `research/analysis-20261127`，已合并入 `main`）

这条线的**分析工具全部是只读的**：读已跟踪的逐题预测或已保存的 logit，不训练、不改权重。
可复现的入口有四类：

```bash
# 1. 听侧的全部单元测试（本轮 82 passed）
env CUDA_VISIBLE_DEVICES= /root/autodl-tmp/iemocap/swift-ser/venv-5090/bin/python \
  -m pytest -q -p no:cacheprovider experiments/hearside_pilot/tests/

# 2. 四条零成本分析（全部只读已跟踪的预测）
python3 -I research/tools/session5_verdict.py   <session5 logits folder>
python3 -I research/tools/aqa_slot_vs_class.py  <aqa run folder>
python3 -I research/tools/text_rules2.py
python3 -I research/tools/prosody_control.py     <cremad run folder>
python3 -I research/tools/aqa_audio_strata.py    <aqa run folder>

# 3. 两条 GPU 步骤（一次性推理，不训练）
bash experiments/hearside_pilot/run_avqa_same_dist.sh     # P0 步骤 2
bash experiments/hearside_pilot/run_aqa_surrogate.sh      # P5

# 4. 全部剩下的工作，按顺序排队，失败不阻塞后续步骤
nohup bash experiments/hearside_pilot/run_queue.sh > /root/autodl-fs/CoTA-LoRA/queue.out 2>&1 &
```

**依赖说明。** `research/tools/` 下的工具只用 numpy（部分用 scipy 的分位数），
系统 python 里**没有 scipy**，要跑请用 `/root/autodl-tmp/iemocap/swift-ser/venv-5090/bin/python`
（numpy 1.26.4 / scipy 1.17.1 / torch 2.12.1+cu130）。该环境**没有 parselmouth**，
所以 `experiments/trr_diagnostics/d2.py prepare`（需要韵律孪生）跑不了——
本轮的 Session 5 清单因此用了一个更轻的构造器 `experiments/hearside_pilot/session5_manifest.py`，
它**只读清单里有的 id**，并对冲突的转写报错而不是静默接受。

**预声明文档。** 每条有判据的实验，判据都写在跑之前，单独成文：

- `research/notes-20261128-session5-confirmation.md`（P1' 的判据，提交 `17db41b`）
- `research/notes-20261127-diagnosis-and-plan.md` §3（P0 / P0b / P1 / P1' / P2 / P3 / P4 / P5）

**原始记录的位置。** 运行结果不在 Git 里（在服务器上）：

- `/root/autodl-fs/CoTA-LoRA/queue-20261128-174926/results/` —— 本轮四条零成本分析的输出
- `/root/autodl-fs/CoTA-LoRA/hearside-session5-20261128-011343/` —— Session 5 的 8 个模型推理
- `/root/autodl-fs/CoTA-LoRA/hearside-avqa-same-dist-*/` —— P0 步骤 2（运行中）

## 迁移到其他机器

1. 先阅读和复核，按来源准备合法模型与数据。
2. 复制一份方法配置到新实验目录，调整 `model`、`swift_source`、清单和输出路径；保留原配置用于追踪。
3. 检查当前导入的 Swift/PEFT 与声明的版本一致。`third_party/snapshot` 为阅读快照，不会自动插入 Python import 路径。
4. 复核模型/数据标识，再运行短 CPU/GPU 定向测试。真实训练、验证、保存、重载及最终评估均需自己的新日志。
5. 旧队列脚本可能含原机器绝对目录与历史测试分数触发逻辑，不要将其直接当成新一轮科研协议。

原始音频、训练/测试题目、权重和完整逐题预测未放入 GitHub；原协议和清单哈希用于定位它们。IEMOCAP 的授权应由使用者按实际来源核查。
