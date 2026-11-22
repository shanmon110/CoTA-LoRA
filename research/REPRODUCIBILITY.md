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

完整环境快照保留在第三方 provenance 文件中。这里的历史版本记录不等于跨平台安装已验收；Torch/CUDA 与显卡驱动应由接手环境单独核对。根目录 `requirements.txt` 原为 WEST recipe，不能单独视为当前 ms-swift 实验锁文件。

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

## 迁移到其他机器

1. 先阅读和复核，按来源准备合法模型与数据。
2. 复制一份方法配置到新实验目录，调整 `model`、`swift_source`、清单和输出路径；保留原配置用于追踪。
3. 检查当前导入的 Swift/PEFT 与声明的版本一致。`third_party/snapshot` 为阅读快照，不会自动插入 Python import 路径。
4. 复核模型/数据标识，再运行短 CPU/GPU 定向测试。真实训练、验证、保存、重载及最终评估均需自己的新日志。
5. 旧队列脚本可能含原机器绝对目录与历史测试分数触发逻辑，不要将其直接当成新一轮科研协议。

原始音频、训练/测试题目、权重和完整逐题预测未放入 GitHub；原协议和清单哈希用于定位它们。IEMOCAP 的授权应由使用者按实际来源核查。
