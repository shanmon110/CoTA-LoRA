# D2 integration and validation — 2026-11-23

## 结论

**代码已整合；暂不能用这轮 D2 判断模型是否利用 F0/能量。** CPU 测试与真实 WORLD 自检通过，但 16 条真实开发集音频没有通过预设的声学验收。当前执行环境也未暴露 GPU，所以没有真实 3B 推理结果、WER/UAR 或显存实测。未启动训练或完整 Session 4 模型评测。

## 来源与范围

- 上游：[PR #2](https://github.com/shanmon110/CoTA-LoRA/pull/2)，分支 `research/trr-diagnostics`，精确提交 `ee2e388429fbb80717f6fa06a07d14e3eba5a71f`。
- 独立整合分支：`integration/trr-diagnostics-20261123`，保留上游提交的父子关系。原 `main`、PR #1、PR #2 未被本轮覆盖。
- 原工程：`/autodl-fs/data/west-ta-baselines`（无 `.git`）；本轮仅新增 `experiments/trr_diagnostics/`。
- 审查工作树：`/root/autodl-tmp/trr-diagnostics/integration-src`，复用已有 Git 导出仓库建立，没有重新 clone WEST。
- Git fetch 的 TLS 连接失败后，通过 GitHub API/raw 获取内容；逐文件 Git blob SHA、完整 tree SHA 和签名 commit SHA 均与 GitHub 核对一致，恢复的是原提交。
- 继续复用既有 IEMOCAP/ASR scorer、冻结基座加载器和 adapter 加载器。三个相关实验目录的 102 个 Python 文件与原工程逐文件一致，未修改训练、模型实现或 checkpoint。

## 实际修复

| 问题 | 影响与修复 | 覆盖 |
|---|---|---|
| 已完成条件断点续跑直接跳过 | 配置、清单、音频或预测变化后可能混用结果。新增 prepared v2 音频摘要、配置/清单摘要校验、完成预测摘要及有序 ID/参考答案检查 | 变更四类输入后拒绝续跑 |
| adapter 配置未随权重固定 | 同一权重配不同 alpha/rank 等配置可能改变结果。身份记录同时固定 adapter 配置与权重、模型配置、prepared 元数据及执行代码摘要 | 固定权重但改变 alpha 后拒绝复用 |
| 报告接受未完成或被修改的预测 | 中断残留文件可能被当作完整评测。只接收带匹配 `metrics.json` 完成标记的预测；报告也校验 prepared 和输出 | SER/ASR 修改、无完成标记测试 |
| 控制失败仍给出主效应结论 | QC、声码器损伤、可懂度控制缺失/失败以及冒烟时，最终 `f0_energy_use` 统一为 `not_interpretable`；阈值原始结果单列为描述值 | QC/ASR/冒烟及损伤控制测试 |
| 合成前数值不能代表实际音频 | 改为重新读取写出的 PCM 做声学 QC，增加 RMS 偏差和削波检查，暴露 peak guard 的响度混杂 | RMS 变化回归及真实音频 |
| session 未按源清单强校验 | 生成之前核对 utterance ID、session 字段与配置的开发 session 一致，拒绝混入其他 session | 错误 session 拒绝生成 |
| McNemar 大整数过早转 float | 约 1024 个以上不一致配对可抛 `OverflowError`。保持整数分子直至除法 | 1031 配对，与 SciPy 精确检验比较 |

`run` 还增加了 GPU 名称及 peak reserved 记录（原先只有 allocated）；这些字段尚未在真实 GPU 上验证。README 更新了结论适用范围和本机命令。没有改变 WORLD 干预算法、原来的 20%/50% 声学阈值或 3 pp/2 pp 行为判据。RMS 检查使用既有合成自检的 2% 容差，默认值为 0.02，也可在新配置中显式固定。

## 验证记录

| 验证 | 实际结果 | 证据 |
|---|---|---|
| 上游原始测试 | 35 passed | 本机 `upstream-tests.log` |
| 新回归先复现 | 原实现上 15 failed | 本机 `regressions-before.log` |
| 修复后新回归 | 15 passed | 本机 `regressions-after.log` |
| 完整 D2 定向测试 | **50 passed，0 failed，0 skipped**；34.62 s | [测试日志](validation/cpu-tests.txt) |
| 真实 WORLD 合成滑音 | **12/12 检查通过**，pyworld 0.3.5 | [自检 JSON](validation/world-self-check.json) |
| 真实 Session 4 音频 | 16 条生成完成，**声学 QC 失败** | [汇总 JSON](validation/real-audio-smoke.json) |

测试有 1 条第三方 anyio pytest 插件弃用警告，不是测试失败。`run` 编排及报告的单元测试用替身模型/scorer；真实模型推理没有被这些测试证明。

本机原始日志目录：`/root/autodl-tmp/trr-diagnostics/integration-evidence-20261123/`。

- `pyworld-install.log`：安装到独立的 `/root/autodl-tmp/trr-diagnostics/deps`，使用 `--no-deps`，未升级既有环境。
- `world-self-check-upstream.json`：真实 WORLD 自检；`twin.py` 从上游到整合版未改动。
- `prepare-upstream.log`、`prepare-integrated.log`：原版与整合版的真实音频生成。整合版检查的是实际落盘 PCM，QC 数值可能稍有不同。
- `integrated-tests.log`：完整 50 项通过日志。
- `environment.json`：Python/包版本及本轮资源可见性。

## 真实音频结果

只读取 IEMOCAP Session 4 的开发清单与转写。源清单 1031 条；5 条删除非词标注后没有词，从所有条件共同排除，剩余 1026 条。按固定种子选 16 条冒烟，每类 4 条，覆盖 11 段对话，共 94.2427 秒原音频。没有读取 Session 5 内容。

源清单 SHA-256：`423156b0b83cc0b060e3a7c5429ee0c908d295076021a0ea6f85a42002e50050`。

输出：`/root/autodl-tmp/trr-diagnostics/d2-smoke-integrated-20261123/`。原版对照保存在 `d2-smoke-upstream-20261123/`。音频和含转写的逐条清单留在本机；仓库仅放聚合 QC 和测试证据。

| 条件 | F0 轮廓标准差/原音频（中位数） | 能量轮廓标准差/原音频（中位数） | 最大 RMS 相对偏差 |
|---|---:|---:|---:|
| sham | 0.9891 | 1.0093 | **25.31%** |
| flat_f0 | **0.3422** | 1.0054 | **33.32%** |
| flat_energy | 0.9862 | **0.6959** | 0.0005% |
| flat_both | **0.3614** | **0.6869** | 1.68% |

F0 拉平要求 ≤0.2，能量拉平要求 ≤0.5，未改项需在 0.8–1.25；RMS 误差需 ≤2%。防削波处理避免了削波，但在 sham 与 flat_f0 引入了不可忽略的响度变化。合成滑音通过不足以保证真实语音通过。这些是声学验证值，不是模型识别结果。

整合版 prepare 实测 45.14 s，1 worker，期间有 CPU 测试并行；原版为 23.77 s。不能用这个差值估计算法性能回退，也不能外推 GPU 评测耗时。

## 环境与未完成项

- Python 3.12.3、NumPy 1.26.4、SciPy 1.17.1、Transformers 4.52.3、ms-swift 4.7.0.dev0；包声明的 PyTorch 为 2.12.1+cu130。这是包元信息，不是 CUDA 可用性证明。
- 当前容器无 `/dev/nvidia*`，`/usr/bin/nvidia-smi` 是 0 字节且不可执行，内存限制 2 GiB。**没有可用 GPU，未尝试加载 3B。** 这不表示原训练服务器的物理 GPU 已消失。
- 既有基座路径为 `/root/autodl-tmp/west-preflight/model`，配置固定 revision `f75b40e3da2003cdd6e1829b1f420ca70797c34e`；后续仍需由现有 loader 校验并真实加载。
- 已有 LoRA 候选为 `/root/autodl-tmp/MMAR/runs/ta-conditional-01/train/lora-mixed/checkpoint-160`，搭配原 `configs/lora-mixed.json`；本轮未重载或给它产生新成绩。
- 未证明：3B 的七种输入条件、ASR 控制、adapter 干净重载和真实推理断点续跑、显存/速度、全 Session 4 报告。无新的 WER/UAR。

## 下一步

先定位真实语音重分析中的 F0 残留、能量拉平不足和防削波引起的 RMS 偏差；保留失败样本及原阈值，不按模型分数放宽验收。若需要改变干预算法，应建立新的版本/配置与输出目录，重新做声学检查。

真实音频验收通过后，在可见 GPU 的环境按 [README](README.md) 对冻结基座及既有 LoRA 做小规模推理验收，再考虑完整 Session 4。单次失败的干预不能被解释为“模型不用韵律”，也不能据此推进 TRR 或声称方法提升。
