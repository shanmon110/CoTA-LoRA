# 本分支的改动与验证状态

分支：`research/claude-review`，基于 `main` 的 `043b6fc`。历史证据、旧配置、旧检查点格式均未改动。

## 状态一览

| 内容 | 本地已修改 | 已提交到 GitHub | 已验证 |
|---|---|---|---|
| 诊断、协议、ACL 判断（本目录的 Markdown） | 是 | 随本分支 | 数字由 `checks/` 重算；文献为摘要级核对 |
| `checks/numerics.py` 及测试 | 是 | 随本分支 | **已在审查环境运行，7 passed**（仅 NumPy） |
| `experiments/cota_review/` 配置生成器与 59 个生成文件 | 是 | 随本分支 | 生成器已运行，`--check` 通过；阶段 0 配置与旧配置逐键比较只多 `diagnostics`；**配置本身未用于训练** |
| 运行时、插件、训练入口、评测入口的改动 | 是 | 随本分支 | **未执行**。只做了 `py_compile` 和导入名静态检查 |
| `tests/test_answer_controls.py`（11 个测试函数） | 是 | 随本分支 | **未执行**（审查环境没有 PyTorch） |
| 任何模型训练或评测 | 否 | 否 | 否 |

**在服务器上跑通测试之前，请把所有新选项视为未经验证。** 命令见 [`PROTOCOL.md`](PROTOCOL.md) 的 P0。

## 代码改动

默认行为保持不变：不设置新键的配置走的是与原来相同的代码路径，写出的仍是 format 8 检查点；已有的 format 1–8 检查点按原规则加载。

### `experiments/ms_swift_librispeech/swift_asr_ta_runtime.py`

- 新增 `transport_signal`（`contrast` 默认 / `audio_mean` / `ordinary` / `ordinary_matched`）。后三个是对照：位置、参数、初始化与 CoTA 完全相同，只改读出的输入。
- 新增 `transport_precision`（`autocast` 默认 / `fp32`）。`fp32` 让门控系数、乘数和两个读出都在 FP32 中计算，避免 BF16 把小乘数舍掉；不走 autocast 的推理本来就是这个算术，所以它同时消除了传输项的训练/推理数值差异。
- `set_transport_gate` 不再要求必须有独立读出 `C`。没有 `C` 的首答案位置传输（只经 `B·g`）写成 format 9，旧代码读到会直接报错而不会误读。
- `set_transport_gate` 不再把 format 版本强行写回 8（改为取较大值），否则 format 9 的检查点重载时会被降级。
- `_answer_contrast` 增加两个带默认值的参数，返回值形状不变。

### `experiments/ms_swift_librispeech/plugin.py`

- 接受 format 9：要求 `transport_signal`、`transport_precision` 取值合法；format < 9 出现这两个键则拒绝；format 8 仍要求有独立读出。

### `experiments/ms_swift_librispeech/train.py`

- 读取 `adapter.transport_signal`、`adapter.transport_precision`。
- 只要设置了 `transport_gate` 就重新审计可训练集合（原来只在有 `C` 时重算，导致无 `C` 的新臂参数量少计）。
- 可选的 `diagnostics.adapter_stats_every`：启用后写 `adapter-stats.jsonl`。

### `experiments/ms_swift_librispeech/adapter_stats.py`（新）

每 N 次更新记录各因子（A、B、g、C、R，或 PEFT 的 lora_A / lora_B / DoRA 幅度）的权重范数、梯度范数、以及与上一次记录的梯度余弦。用来直接检验"门的梯度是否有稳定方向"。依赖 `TrainerCallback.on_pre_optimizer_step`，需确认 transformers 4.52.3 提供该事件。

### `experiments/ms_swift_aqa/transfer_eval.py`

- 新增 `ablate_for_eval` 与 `--ablation`：`neutral_gate`（清零 `R`，乘数恰为 1）、`no_shared`（清零 `g`）、`no_independent`（清零 `C`）。原有 `--no-transport` 不变。只改内存中的模型，不写回检查点。

### `experiments/ms_swift_librispeech/tests/test_answer_controls.py`（新）

覆盖：默认配置仍为 format 8；新选项的取值与前置条件校验；各对照在初始化时零更新、保存重载后逐位一致；format 9 元数据不一致时拒绝；对照信号只改变答案位置；BF16 autocast 下小乘数在 format 8 路径被完全抹掉而 FP32 路径保留；无 autocast 时两条路径一致；共享读出对照的梯度；三种单因素消融；统计回调。

## 与导出清单的差异

`research/EXPORT_MANIFEST.json` 记录的是交接快照 `043b6fc`，本分支没有改写它。因此在本分支上运行 `python tools/verify_handoff_export.py` 会报告下面 4 个文件内容不同，这是预期的；其余 2382 项不变。清单中的原始 SHA-256：

| 文件 | 交接快照中的 SHA-256 |
|---|---|
| `experiments/ms_swift_aqa/transfer_eval.py` | `31356ea03038c09776e2d18196f587b0537ce6ca18f58183b05cd13485bd6bd9` |
| `experiments/ms_swift_librispeech/plugin.py` | `19d52af2f56b571af971475dcdfcee92312e869e893b77841d9e50ba65980f1a` |
| `experiments/ms_swift_librispeech/swift_asr_ta_runtime.py` | `3e2af6767e0fb0993cac0bd93763643ba12750a9d8b8864dc4342012d405ed80` |
| `experiments/ms_swift_librispeech/train.py` | `c932d8d192c27b0ef2b814e17204438c4e71da6c288a5b111537e4e05af7d431` |

原文件可用 `git show 043b6fc:<路径>` 取回。合并时如果希望清单重新覆盖全仓库，建议另存一份新清单，而不是覆盖原清单。

## 尚未做、且不应被当作已完成的事

- 没有复跑仓库原有的 186 项 torch 测试，也没有运行新增测试。
- 没有任何 GPU 训练、评测或显存与耗时测量。
- 没有构建 DEV-AC，也没有确认候选最终测试集的许可。
- 没有修复 LoRA-GA、PiSSA、EVA、LoRA+ 等基线，只给出了诊断与修法（DIAGNOSIS §1.6、§2.2）。
- 相关工作只做了检索级核对。
