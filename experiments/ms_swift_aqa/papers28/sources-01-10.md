# 论文 1–10 的实现与复核

本组沿用统一的 AVQA 训练与开发集。MMAU Test-mini 和 IEMOCAP Session 5 用于冻结所选 checkpoint 后的迁移评估。下表注明每篇论文如何接到当前工程。实际完成的目标集成绩以逐项 result.json 为准。

| 编号 | 方法与实现来源 | 本工程使用的条件 |
|---|---|---|
| 1 | Expressive Power。作者的构造法从已知冻结和目标网络权重解析地构造 LoRA 因子。[作者代码](https://github.com/UW-Madison-Lee-Lab/Expressive_Power_of_LoRA/tree/eaeb7d3f265853343b234172ed44cd7623c5212a) | 作者的已知目标构造在两个合成网络复算。另以同批 AVQA 全参数教师的 72 个方形 Q/O 矩阵作 depth=1 的独立压缩控制，并用生成的适配器转移评估。作者完整 Transformer 证明的结构假设与 Qwen 的分组查询注意力、非方形 SwiGLU 不同。 |
| 2 | LongLoRA。作者训练时使用 shifted short attention，推理时恢复全注意力。[作者代码](https://github.com/JIA-Lab-research/LongLoRA/tree/d4eb344c5ccc9e91c0812a2b2aeea69070df8c33) | 原生 Swift 的 longlora 入口检查 Llama 类型。Omni 移植需要保留因果 mask、GQA 和多模态 RoPE，单列上下文长度及可训练 embedding/norm 的选择。 |
| 3 | VeRA。作者项目页推荐 PEFT 入口。[项目页](https://dkopi.github.io/vera/) | 使用本机 PEFT 0.15.2 的共享冻结 A/B 和每层可训练向量。沿用已有 peft_suite 的精确目标选择和共享 buffer 重载适配。 |
| 4 | DoRA。采用权重幅值和方向的原生 PEFT 实现。[作者代码](https://github.com/NVlabs/DoRA/tree/7e2f10abbe8efe212c8fca1d983ae1d04ef13a18) | 通过 use_dora 开启，审计可训练的幅值向量、A/B 矩阵和全部目标投影。 |
| 5 | FourierFT。作者仓库当前将官方实现指向 HF PEFT。[作者说明](https://github.com/Chaos96/fourierft/tree/98c7893abbc204ae34b0c4fe59d5ae1fe2b45bce) | 原生 PEFT 频率采样和系数，沿用工程已通过稠密输出与梯度对照的稀疏计算。 |
| 6 | LoRA+。原论文提出给 B 和 A 使用不同学习率。[作者代码](https://github.com/nikhilgsh/loraplus/tree/a0c44bfec0eff97b9574469366bc6b0ccc28838d) | 使用 Swift 原生 lorap_lr_ratio=16。每个 A 与普通 LoRA 使用相同初始化，记录实际 AdamW 参数成员和学习率。 |
| 7 | LoRA-GA。复用作者公开的 ArB2r 奇异向量选择、stable 缩放和冻结残差公式。[作者实现](https://github.com/Outsider565/LoRA-GA/blob/c4cd5372c75b290924214b348008891f744512ef/peft/src/peft/tuners/lora/layer.py) | 从当前 AVQA 训练集按种子取 16 条，各 1 条组成一个批次。使用 Swift 的音频嵌入、SFT 标签和 compute_loss，从冻结基座计算梯度，转移至 CPU 累加，按投影初始化。stable_gamma=16。保存被减去初始更新后的 BF16 残差、哈希和训练后 FP32 因子。 |
| 8 | PiSSA。作者已将奇异分量初始化并入 PEFT。[作者代码](https://github.com/MuLabPKU/PiSSA/tree/a6e4c9c1d8bc1b73c0fd2f524be8e00e26043395) | 原生 pissa_niter_16 随机快速 SVD。保存实际 BF16 残差和最终因子，在新基座上按原运算顺序重载。残差在同一次训练的各 checkpoint 之间用硬链接共享，每个 checkpoint 可以独立复制。 |
| 9 | LoRAM，本文使用 **LoRAM-Rand、无持续预训练对齐**。这是论文附录 C/J 报告的消融条件。[作者代码](https://github.com/junzhang-zj/LoRAM/tree/ce64e68bbb3a24d21d1a85e883f71323879acb88/loram) · [论文与附录](https://arxiv.org/html/2502.13533) | 作者公开脚本面向 Llama-3-70B。移植同一裁剪和补零恢复操作，保留前 4 和后 2 层，在中间层随机保留一半 MLP 中间通道与成组 KV/Q 头。50% 比例适配 3B 的两个 KV 头。训练实际使用较小张量，验证和测试使用原尺寸基座加恢复后的适配器。checkpoint 同时保存训练用窄因子、掩码和 recovered 目录。论文主方案包含裁剪后的持续预训练对齐，当前固定语料实验只评估无对齐条件。 |
| 10 | RandLoRA。作者推荐合入 PEFT 的实现。[作者代码](https://github.com/PaulAlbert31/RandLoRA/tree/371d82f4ef713fc4c35d6cb581d37eef558dbb31) · [上游合入 PR](https://github.com/huggingface/peft/pull/2464) | 只有这一组使用隔离的 PEFT 0.17.1。原生 r=32，alpha=640，多组共享随机基形成可满秩更新。固定 PRNG key=42，在干净进程重建共享基并验证训练时写入的 SHA-256。 |

第 6–10 项的启动配置放在 configs 下。LoRA+、LoRA-GA 和 PiSSA 使用现有训练环境。RandLoRA 单独追加依赖路径，基础环境中的 PEFT 保持 0.15.2。可按以下命令安装与验证。

```bash
PY=/root/autodl-tmp/iemocap/swift-ser/venv-5090/bin/python
RAND=/root/autodl-tmp/r1-aqa/paper-envs/peft-0.17.1
"$PY" -m pip install --no-deps --target "$RAND" peft==0.17.1
OMP_NUM_THREADS=1 "$PY" -m pytest -q experiments/ms_swift_aqa/tests/test_paper_early{,_loram}.py
PYTHONPATH="$RAND${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=1 "$PY" -m pytest -q experiments/ms_swift_aqa/tests/test_paper_early.py -k randlora
```

## 论文 1 的独立复核

expressive.py 固定作者提交并检查三个 Python 文件的 SHA-256。第一个实验原样运行作者 README 的 Example 1。第二个运行同深度 Transformer 的解析构造，宽度 8、两头、rank 4、测试输入 5,000 个。作者脚本在该路径有两个记录变量的拼写问题，复核脚本将改动另存为 transformer.diff，可与保留的原始文件对照。

本机实测，FNN 的普通与分布偏移 MSE 分别为 0.0329、3.0024。Transformer 两项在作者保留四位小数的打印精度下均为 0.0000。完整运行记录在 /root/autodl-tmp/MMAR/runs/papers28-transfer-01/theory-01/，其中包含源文件、哈希、命令、diff、原始日志和 result.json。

```bash
PY=/root/autodl-tmp/iemocap/swift-ser/venv-5090/bin/python
DEPS=/root/autodl-tmp/r1-aqa/paper-envs/expressive/deps
"$PY" -m pip install --target "$DEPS" wandb==0.19.11
"$PY" -m experiments.ms_swift_aqa.papers28.expressive \
  --output /root/autodl-tmp/r1-aqa/expressive-replay \
  --extra-dependencies "$DEPS"
```

已有作者源码时可以通过 --cached-sources 指定其目录以离线复算。该脚本关闭在线日志和 GPU。

### 真实 Omni 同源已知教师控制

已在相同 2048 条 AVQA 训练、256 条开发数据上完成全参数微调的 Omni 3B 提供了可读取的目标权重。构造过程从归档的 full/selection.json 读取开发集选定的第 32 步。程序重新计算每个教师分片、基座分片、配置、训练及开发清单的 SHA-256，核对教师的训练协议和最佳 checkpoint。测试数据不参与构造。

36 个 Thinker 语言层各取 2048 × 2048 的 Q/O 投影，共 72 个独立 depth=1 的线性构造。从真实 BF16 矩阵转换至 FP32 计算差矩阵的完整 SVD，按作者 helper.py 公式保留 8 个右奇异向量。PEFT 的 alpha=16、r=8 对应 scale=2，导出的 B 因子除以 2；在可精确表示的精度内，有效更新等于差矩阵最佳秩 8 近似。其余语言投影、偏置、归一化层和音频/视觉权重来自冻结基座。构造本身使用 0 步优化，记录的 2,359,296 个参数为导出因子容量，教师的原始训练成本独立存在。这个控制测量给定真实教师矩阵后，局部构造对端到端任务保留了多少能力。完整 Transformer 的跨层表达性定理仍由上一节中的合成网络实验复核。

```bash
OUT=/root/autodl-tmp/MMAR/runs/papers28-transfer-01
PY=/root/autodl-tmp/iemocap/swift-ser/venv-5090/bin/python
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 "$PY" -u \
  -m experiments.ms_swift_aqa.papers28.known_target \
  --protocol experiments/ms_swift_aqa/papers28/protocol.json \
  --output "$OUT/theory-01/omni-control" --threads 4
```

新目录中的 construction.json 包含教师选择、快照与数据标识、分层保留能量、Frobenius 相对误差和构造计时；checkpoint-constructed/ 可通过原生 PEFT 的干净基座加载。上述两项均在完整构造后发布。该实验的统一评估和作者合成实验的 theory-01/result.json 单独留存。
