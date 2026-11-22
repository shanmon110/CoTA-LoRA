# WEST + TA-LoRA 单种子 pilot 上机验收

**暂缓 14 组单种子 pilot。** ASR 实卡重算问题已修复，典型和最长样本均已通过。SER 仍缺 USC 授权的 IEMOCAP Session 1–4 及真实数据验收。

## 范围与环境

在现有 ZIP 工程中沿用 Thinker 训练与评估入口。设备为一张 RTX 4080 SUPER，启动前占用 0 / 32,760 MiB，驱动 580.105.08。独立环境位于 `/root/.venvs/west-omni-preflight`，由 `python -m venv --system-site-packages` 建立。复用 CUDA 13.0 的 torch 2.12.1 和 torchvision 0.27.1，venv 内固定 Transformers 4.52.3、Accelerate 1.7.0、NumPy 1.26.4。BF16 矩阵乘和反传有限；`pip check` 通过。具体版本与设备记录见 [environment.txt](logs/environment.txt)、[gpu.csv](logs/gpu.csv)、[安装日志](logs/install.log)。

模型和数据放在 `/root/autodl-tmp/west-preflight`。官方模型为 `Qwen/Qwen2.5-Omni-3B@f75b40e3da2003cdd6e1829b1f420ca70797c34e`。本地 `model/` 共 18 个文件、11,989,103,936 字节；通过镜像传输的三片权重分别匹配固定提交的 LFS SHA-256。覆盖 Thinker 权重及 processor 输入的工程内容摘要为 `b09075962102a1d421f3d891bec565d2c40eedd0a8c5dc0668d749b9a6b00238`。七法共用此快照和摘要。[模型校验日志](logs/model-download.log)

## 修复与回归

- **混合精度。** 新训练配置持久化 `residual_precision=fp32_add_then_cast`。所有方法在 FP32 中相加，再一次转回原始输出 dtype。旧 O/TA 缺字段时回到先 cast 增量的历史路径，旧 projection 继续使用原来的 FP32 相加语义。用非零权重、真实 tiny eager attention 和独立构造的旧格式 safetensors 验证。半 ULP 边界用例中，旧 O-LoRA 的两个输出为 1、2，新策略及同权 projection 得到 1.0078125、2.015625。新 O/TA 零 gate 与同权 EVA 低秩投影保存前后逐位一致。DoRA 的幅值、方向和归一化计算沿用原实现。
- **实卡发现的 TA 重算错误。** 首次真实 TA 在 14 秒、479 token 样本的第一次 backward 进入 GPU autograd 工作线程时，原 `ContextVar` 为空，触发 `TA mode requires audio_context`。该次峰值为 9.72 / 10.26 GiB（allocated / reserved）。在 TA 启用非重入 checkpoint 的入口通过 `context_fn` 捕获每次原始 forward 的布局，并在对应重算作用域内恢复。新增 CPU 跨线程、不同 microbatch 的非零参数梯度回归，对照关闭 checkpoint 的结果。原始异常和指标保留在 [失败栈](logs/asr-gpu-01/ta-0-train.log) 与 [失败阶段记录](logs/asr-gpu-01/ta-0-train.jsonl)。
- **输入协议。** `examples.omni_talora.preflight` 按单样本读取音频头、换算重采样后的 mel 有效帧，使用实际 chat template/tokenizer 计算展开后长度。训练计完整 SFT 序列，验证计 prompt。小模型测试逐样本与 processor 输出的两种 mask 对照。检查 train/dev 间和 split 内的重复 ID、同一音频文件；SER 限定 train 为 Session 1–3、dev 为 Session 4。可输出逐 ID 原因、split 汇总、带哈希的新清单和批帧违规记录。运行时同一 collator 在训练、EVA 校准和评估入口硬性检查单条时长、批内有效帧之和与展开输入 token 数；超限报错并带样本 ID。新 checkpoint 记录启用的预算，评估默认恢复并拒绝冲突覆盖。
- **基座身份。** 本地训练读取模型分片、索引、模型与 processor 配置、词表和模板的字节内容，按文件名绑定 SHA-256。训练可通过 `--base-sha256` 钉住目标，评估在加载前核对归档的标识。无此字段的历史文件继续按原协议重载。用同尺寸替换权重或 tokenizer 的回归用例验证拒绝行为。
- **ASR 诊断。** 评估 JSON 保留 WER 等字段，增加 `max_new_tokens`、`max_new_tokens_without_eos_count`、`max_new_tokens_without_eos_ratio`。只检查 padded prompt 之后的续写，最后一位为 EOS 也算自然终止。ASR 七法计划统一上限 **128**。
- **Q/V 对照。** 明确使用 `--mode lora --target-set qv --rank 8 --match-ta-budget --alpha 10 --last-n 4`。先以 meta 模块核对，再在这张 GPU 加载完整 BF16 3B 权重、实际安装 Q/V 分支，核对 rank **5**、所有分支 scale **2.0**、可训练参数 **128,000**，参考 TA 为 **131,104**。这里没有为 Q/V 执行 optimizer.step。训练入口也输出请求/解析 rank、alpha 与有效 scale。见 [GPU 实际加载日志](logs/qv-real-3b.log) 和 [meta 核查日志](logs/qv-budget.json)。
- **受控推迟。** BoRA 固定 blocks=2，保留原增益公式。非默认高分块的指数溢出仍是已知边界。全词表 logits、音频塔稠密 eager attention 和 TA 条件 attention 的算子本轮沿用原实现；适用上限以本报告的实测长度为准。

## 定向测试

在禁用 CUDA、禁用 Hub 在线读取且指定官方 processor 的条件下，运行 `/root/.venvs/west-omni-preflight/bin/python -m pytest -q -ra test/test_omni_ta*.py`。修复 GPU 重算后最终 **121 passed，0 skipped，0 failed，另有 22 个子测试通过**。唯一警告来自外部 anyio pytest 插件的弃用名称。早期在新测试文件创建、对应模块尚在开发的时间窗口发生过一次收集错误；最终全套已通过。[最终原始输出](logs/cpu-after-recompute.log) · [早期收集日志](logs/initial-tests.log)

## SER 数据与验收

本地搜索未找到 IEMOCAP 授权副本。USC 官方要求阅读许可并提交 release form 获取数据。本次未取得授权下载地址。Session 5 未读取。SER 的 train/dev 数量、过滤数、哈希、预算和三阶段显存均为**未测**。小模型合成 SER CLI 已纳入定向测试。获得授权后，应以 Session 1–3 和 4 的清单完成同样的选样、预算核对和 GPU 验收。[USC 官方发布页](https://sail.usc.edu/iemocap/iemocap_release.htm) · [获取条件记录](logs/iemocap-access.txt)

## ASR 清单和统一预算

ASR 使用完整 LibriSpeech train-clean-100 和 dev-clean，均通过 OpenSLR 发布的归档 MD5。训练与验证共 31,242 个唯一 utterance ID，文件身份检查通过。先用固定提交的官方 processor 做无阈值统计，再以两份清单的观测上界确定七法共用预算。固定 **batch=1、单条最长 32.645 秒、批内最多 3,265 个有效 mel 帧、单条最多 883 个展开输入 token**。ASR 解码可另追加 128 token。七法使用同一对计划清单，每份新清单保留所有样本，音频路径已规范化。[下载校验](logs/librispeech-download.log) · [无阈值统计](asr-stats/summary.json) · [预算审计](asr-plan/summary.json) · [逐 ID 测量与超限原因](asr-plan/samples.jsonl)

| Split | 数量 | 时长中位 / 最大（秒） | 帧中位 / 最大 | 输入 token 中位 / 最大 | 超限 / 过滤 |
|---|---:|---:|---:|---:|---:|
| train-clean-100 | 28,539 | 13.995 / 24.525 | 1,400 / 2,453 | 479 / 804 | 0 / 0 |
| dev-clean | 2,703 | 5.920 / 32.645 | 592 / 3,265 | 215 / 883 | 0 / 0 |

| Split | 原始清单 SHA-256 | 七法共用清单 SHA-256 |
|---|---|---|
| train | `f2e0ffdfc3dafcb890ddc0fa5ca1704da5a8076c09801293443f67030e138cb0` | [train.jsonl](asr-plan/train.jsonl) · `e33bd6e2b6c75cbcd6706fb95949843624f7d26dfd14c723763a5d61c90c8411` |
| dev | `e155b67e573f9d2e43c28091627ee7e08b4ef9599aac8691ca316eaee92fdd45` | [dev.jsonl](asr-plan/dev.jsonl) · `eea0acf0f504eb8ca2a095591d2887b6b9653fc6a4a45ae389de5bca3a003b04` |

按时长与序列长度相对中位数的距离选典型样本，同时选择最长序列与最大帧数的样本。每个 split 内两个极值恰好为同一条录音。EVA 使用 12 条独立训练记录，包含训练侧典型和极值。记录选择算法、各录音 ID、时长、帧数、展开长度及 smoke 清单哈希见 [asr-selection.json](asr-selection.json)。

## ASR 实卡 smoke

全部在同一卡顺序执行。每个训练进程从干净 BF16 基座开始，最后四层、rank 8、alpha 16、种子 42、batch 1、累积 1、非重入梯度检查点，AdamW 两次更新。O 和 TA 对典型与最长训练样本分别生成独立新 adapter，再以独立进程重载和评估对应的 dev 样本。EVA 校准 12 条后训练两步，同一归档分别在典型与最长 dev 输入上重载。[O-LoRA 和首次 TA 命令](logs/asr-gpu-01/commands.txt) · [修复后的 TA 命令](logs/asr-gpu-ta-replay/commands.txt) · [EVA 命令](logs/asr-gpu-eva/commands.txt) · [受限运行入口](run_asr_smoke.py)

| 方法 | 阶段 | 样本时长（秒） | 展开输入 token | 峰值 allocated（GiB） | 峰值 reserved（GiB） | 原始记录 |
|---|---|---:|---:|---:|---:|---|
| O-LoRA | 训练 | 14 | 479 | 9.726 | 10.230 | [JSONL](logs/asr-gpu-01/lora-0-train.jsonl) |
| O-LoRA | 重载评估 | 5.92 | 215 | 8.961 | 9.316 | [JSONL](logs/asr-gpu-01/lora-0-dev0.jsonl) |
| O-LoRA | 训练 | 24.525 | 804 | 10.287 | 10.949 | [JSONL](logs/asr-gpu-01/lora-1-train.jsonl) |
| O-LoRA | 重载评估 | 32.645 | 883 | 9.181 | 9.740 | [JSONL](logs/asr-gpu-01/lora-1-dev1.jsonl) |
| TA-LoRA | 训练 | 14 | 479 | 9.726 | 10.262 | [JSONL](logs/asr-gpu-ta-replay/ta-0-train.jsonl) |
| TA-LoRA | 重载评估 | 5.92 | 215 | 8.961 | 9.316 | [JSONL](logs/asr-gpu-ta-replay/ta-0-dev0.jsonl) |
| TA-LoRA | 训练 | 24.525 | 804 | 10.287 | 11.068 | [JSONL](logs/asr-gpu-ta-replay/ta-1-train.jsonl) |
| TA-LoRA | 重载评估 | 32.645 | 883 | 9.191 | 9.836 | [JSONL](logs/asr-gpu-ta-replay/ta-1-dev1.jsonl) |
| EVA | 校准 | 4.52–24.525 | 193–804 | 9.126 | 9.783 | [JSONL](logs/asr-gpu-eva/eva-0-train.jsonl) |
| EVA | 训练 | 11.43–14 | 410–479 | 9.726 | 11.295 | [JSONL](logs/asr-gpu-eva/eva-0-train.jsonl) |
| EVA | 重载评估 | 5.92 | 215 | 8.961 | 9.316 | [JSONL](logs/asr-gpu-eva/eva-0-dev0.jsonl) |
| EVA | 重载评估 | 32.645 | 883 | 9.181 | 9.740 | [JSONL](logs/asr-gpu-eva/eva-0-dev1.jsonl) |

阶段开始前同步并重置 CUDA allocator 的峰值计数，数值包含该阶段驻留的基座和仍保留的 allocator 缓存。EVA 校准是整个 12 条样本阶段的峰值，训练是两个实际抽到的 microbatch 的阶段峰值。六次单样本重载评估的无 EOS 撞上限计数均为 0/1，仅用于验证解码诊断。

EVA 的实际逐层 rank 为 6、8、10、8，总和 32，可训练参数 131,072。独立评估进程只允许清单读取器和音频解码器访问指定 dev 清单及对应音频，任何校准调用会立即失败；两次评估通过。[典型样本评估输出](logs/asr-gpu-eva/eva-0-dev0.log) · [最长样本评估输出](logs/asr-gpu-eva/eva-0-dev1.log)

各归档的非零参数计数、SHA-256 和本地新 checkpoint 路径见 [O-LoRA](logs/asr-gpu-01/saved-adapters.jsonl)、[TA-LoRA](logs/asr-gpu-ta-replay/saved-adapters.jsonl)、[EVA](logs/asr-gpu-eva/saved-adapters.jsonl)。本轮检查了训练后非零 up 和 TA 的非零 gate。

**结论：暂缓 14 组单种子 pilot。** ASR 已通过约定的真实 3B 验收。剩余阻塞是取得合法的 IEMOCAP Session 1–4，并据真实清单完成 SER 的划分审计、统一预算、O/TA 两步训练重载、EVA 校准和独立评估。本轮未启动正式对比。
