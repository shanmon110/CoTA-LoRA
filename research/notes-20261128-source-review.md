# 源码级复核：可训练集合、门项、首答案查询、cache

日期：2026-11-28。这一份是 `CLAUDE.md`「建议先做的研究」里第一组的执行结果：
**只读源码和已保存的 checkpoint，不训练、不加载模型**。

工具：`research/tools/checkpoint_audit.py`（直接读 safetensors，不加载 torch 模型）。

## 结论先说

**仓库里所有用于本文分析的臂，`gate_multiplier` 都是 0.0，传输项是一个可证明的恒等 no-op。**
这不是 bug，是**有意的、文档写明的**设计：这些臂测的是"把 LoRA 放在哪些层"，
**不是**测传输机制本身。交接文档的「当前 CoTA 为 … 音频—文本 attention 聚合差条件门」
描述的是**代码里存在的能力**，**不是这些臂实际训练过的东西**。

**这一条必须写进论文的实验设置**：如果论文说"CoTA-LoRA 的层段效应"，
读者会以为测的是 CoTA 的传输机制；**实际测的是一个不带传输项的低秩更新**。
不写清楚这一点，审稿人会认为结果被误述。

## 1. 可训练集合

`experiments/ms_swift_librispeech/plugin.py` 的 `_params` 决定哪些参数进优化器。
以 `A-last4-s43` 为例（`initialization.json` 记录）：

```
trainable_parameters: 131104
trainable_names:      每层 3 个 —— down (A), up (B), gate (g)
                      thinker.model.layers.{32,33,34,35}.self_attn.ta_adapter.{down,up,gate}
```

- **R / g / C / A / B 的对应**：
  - `down` = A（rank × in_features，Kaiming 初始化）
  - `up` = B（out_features × rank，**零初始化**）
  - `gate` = g（rank，**零初始化**）
  - **独立 C（`transport_up`）不存在**：所有臂的 `independent_transport_up` 都是假
  - **上下文门（`gate_weight`）不存在**：`transport_gate` 是 `None`
- **参数计数自洽**：4 层 × (8×2048 + 2048×8 + 8) = 4 × 32,776 = **131,104** ✓
- **多层臂**（A-all 等，36 层）：36 × 32,776 = **1,179,936** ✓
  与 `experiments/hearside_pilot/README.md` 第 27 行写的数完全一致。

**所以本文分析里的"CoTA"实际是**：`up @ down @ ordinary` 的一个 r=8 低秩更新，
加一个**没有梯度的**门项。

## 2. 门项为什么不动 —— 数值验证

源码路径：

- `swift_asr_ta_runtime.py:470` —
  `unused = self.ta_adapter.gate_multiplier == 0. and self.ta_adapter.transport_up is None`
  → `contrast = None if unused else _transport(...)`
- `swift_asr_ta_runtime.py:59-64` — `contrast is None` 时走**不含 `gate` 的**早退分支：
  `return self.scale * F.linear(F.linear(ordinary.float(), self.down), self.up)`

**门项只通过 `gate * extra` 进入计算图**，而 `extra` 就是被跳过的那一项。
所以 `gate_multiplier == 0` 时，**前向图里根本没有 `gate` 这个节点**，
`dL/dgate` 是**不存在**，不是"零"。

数值验证（复刻 format 4 的前向，`down`/`up` 取非零值以保证 loss 非零）：

| `gate_multiplier` | loss | `grad(gate)` | `|grad(up)|` | `|grad(down)|` |
|---|---:|---|---:|---:|
| 0.0 | 264.6608 | **None（不在计算图里）** | 1067.006 | 105.081 |
| 1.0 | 264.6608 | 77.271 | 1067.006 | 105.081 |

并且：`gate_multiplier=0` 且 `gate=0.7`（非零）时，
`forward(contrast=None)` 与 `forward(contrast=v)` 的最大差 **0.000e+00**（逐位相同）。

**结论**：`gate_multiplier=0` 让传输项**逐位**成为恒等 no-op。
这与仓库 README 第 35 行的记载一致（"gate 参数此时没有梯度（原先是零梯度）"），
也对应单元测试 `test_zero_multiplier_skips_the_contrast_without_changing_outputs_or_gradients`。

**checkpoint 侧的独立证实**：所有臂保存的 `gate` 张量**全为 0**
（`nonzero 0/8`，`min = max = mean = +0.000000`）——
如果门项真的参与了训练，它不会一个比特都没动。

## 3. 首答案查询

`_answer_query`（`swift_asr_ta_runtime.py:286-311`）在 `transport_gate` 不为 `None` 时才被调用
（`layout()` 第 572-576 行：`mode is None` 就直接返回普通 layout）。
**所有臂的 `transport_gate` 都是 `None`**，所以**首答案查询机制在这些臂里根本没有启用**。

代码本身的正确性（供将来启用时参考）：
- `positions = first_label - 1`，即第一个被监督 token 的**前一个**位置；
- 有硬校验：`(first_label > 0)` 且该位置的 label 必须等于 `input_ids` 的对应值，
  否则 `raise ValueError('Each training row needs a visible prompt and a supervised next token')`；
- 还有 `not (valid & ~audio).gather(...)` 的校验，保证该位置是**有效文本 token**；
- `labels` 必须是**完整未移位**的（`labels.shape != input_ids.shape` 直接报错）。

**监督的形状**（`initialization.json` 的 `supervision` 字段）：
`sequence_tokens: 312`、`audio_placeholders: 248`、`supervised_tokens: 3`、`decoded_target: "B"`。
**每条训练样本只监督 3 个 token**——这是"短答案封闭式多选"任务，
也解释了为什么 4,916 行 IEMOCAP 只训 264 步。

## 4. 模态 mask

`_layout`（第 512-528 行）建立 `audio = input_ids.eq(audio_token_id) & valid` 和 `valid`。
**它要求**：
- `input_ids` 是**展开后**的 `[batch, sequence]`（打包/无填充被拒绝）；
- `attention_mask` 必须是与 `input_ids` 同形状的 **2-D 二值**掩码；
- 拒绝 packing、拒绝外部传入的 KV cache、拒绝视觉输入。

**传输项的模态划分**（`_transport`，第 385-419 行）：
`visible` = `valid` 且位置 ≤ 当前 query 位置；`audio` 与 `~audio` 分别求条件概率，
差值乘到 value 上。**没有独立的"模态 mask"超参**——划分完全由 audio token id 决定。
所以「模态 mask」在当前实现里**不是可调项**，是**构造性的**。

## 5. cache

- 只支持 `DynamicCache`（`type(past_key_value) is not DynamicCache` 直接报错）；
- `generate` 被包装成**保留原始 prompt layout**（`frames.set(frames.get() + (layout(prompt, kwargs),))`），
  注释写明"GenerationMixin 反复调用 Thinker.__call__（而且不走 outer.forward）"；
- 生成出来的文本槽位在 `_transport` 里被显式补齐：
  `F.pad(audio, (0, k - audio.size(1)), value=False)` / `F.pad(valid, ..., value=True)`，
  注释是"prompt 之外的槽位是生成的文本；把补齐的 prompt key 单独掩掉"。
- **梯度检查点被显式拒绝**（Omni 路径）：`swift_asr_ta_runtime.py:581-582`
  "disable gradient checkpointing: recomputation would run outside the TA layout scope"。
  Qwen3-ASR 路径则用 non-reentrant replay 单独处理。
  **这与交接文档「当前动态版拒绝 Omni 梯度检查点」一致，且已在本仓库核实。**

## 6. 这一组复核对论文的影响

1. **必须写清楚**：本文分析测的是**不带传输项的低秩更新**的层段效应，
   不是 CoTA 传输机制的层段效应。
2. **「CoTA-LoRA」这个名称在本文的实验表里要加限定**，
   否则就是**误述方法**——审稿人按名称理解会去查传输项。
3. **传输机制本身在本文里没有被测过**。仓库里 `experiments/cota_review/configs/`
   有一批 `gate_multiplier: 1.0` 的配置（stage0-legacy / stage1-lr），
   **但那些不是本文分析用的臂**，也没有在本轮核对过的结果里。
   **如果论文要谈传输机制，必须另做实验并单独报告，不能拿本轮的层段结果去支持。**
4. **一个可以顺手做、零成本的事**：既然 `gate_multiplier=0` 与 `=1` 在数值上等价于
   「无传输项」vs「有传输项」，而**本文所有臂都是 0**，
   那么本文的**层段结论严格来说是"低秩更新的层段结论"**，
   这一点在措辞上必须精确。

## 7. 没有做的事

- **没有重新训练任何臂**，没有验证"传输项若启用会怎样"。
- **没有核对 `experiments/cota_review/` 那批 `gate_multiplier=1.0` 的结果**——
  它们不在本文分析链上，本轮只核对了它们的配置存在。
- **没有检查 `down`/`up` 的实际位移量**：仓库只记了 `ab_init_sha256`（A+B 拼接的哈希），
  **没有保存初始张量本身**，所以"移动了多少"算不出来。
  本轮给出的是**每个因子的最终范数**（`down` |·| ≈ 1.69–2.02，`up` |·| ≈ 0.27–0.62，
  `gate` = 0），**不是位移**。这一点必须写清楚，不能拿范数当位移讲。
