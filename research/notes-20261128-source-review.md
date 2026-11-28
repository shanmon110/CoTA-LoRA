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

### 2b. **来自训练本身的独立证据**（比读源码更强）

每个 checkpoint 里都存了 `optimizer.pt`，那是**训练循环自己写的**。
Adam **只会为它真正更新过的参数分配状态**（`exp_avg`、`exp_avg_sq`）。
所以：**门项有没有优化器状态，直接回答"训练时它有没有拿到过梯度"**，
而且这个证据**完全不依赖对前向代码的解读**。

工具：`research/tools/gate_optimizer_state.py`。以 `A-last4-s43`（264 步）为例：

```
index parameter                                                      optimizer state?
    0 thinker.model.layers.32.self_attn.ta_adapter.down                           yes
    1 thinker.model.layers.32.self_attn.ta_adapter.up                             yes
    2 thinker.model.layers.32.self_attn.ta_adapter.gate                          NONE
    3 thinker.model.layers.33.self_attn.ta_adapter.down                           yes
    4 thinker.model.layers.33.self_attn.ta_adapter.up                             yes
    5 thinker.model.layers.33.self_attn.ta_adapter.gate                          NONE
    6 thinker.model.layers.34.self_attn.ta_adapter.down                           yes
    7 thinker.model.layers.34.self_attn.ta_adapter.up                             yes
    8 thinker.model.layers.34.self_attn.ta_adapter.gate                          NONE
    9 thinker.model.layers.35.self_attn.ta_adapter.down                           yes
   10 thinker.model.layers.35.self_attn.ta_adapter.up                             yes
   11 thinker.model.layers.35.self_attn.ta_adapter.gate                          NONE
```

**全部 44 个臂、全部 144 个门参数：0 个有优化器状态；
而 100% 的 `down`/`up` 因子都有**（多步之后仍非零）。

**所以门项在每一个臂里都从未被更新过。** 这不是"学得小"，是"没有被步进过"。

**checkpoint 张量侧的旁证**：所有臂保存的 `gate` 张量**全为 0**
（`nonzero 0/8`，`min = max = mean = +0.000000`）——
如果门项真的参与了训练，它不会一个比特都没动。

## 2c. **历史臂连独立 C 都没有**（审计确认，44 个臂全部）

`checkpoint_audit.py` 对全部历史臂的结果：

```
   --- AQ-all
       gate_multiplier      0.0
       transport_gate       None
       independent_C        None
       saved factors        down x36, gate x36, up x36
```

**历史臂保存的因子只有 `down`、`up`、`gate` 三个。**
没有独立的 `transport_up`（C），没有 `gate_weight`。

这一点很重要，因为它把"传输机制缺席"的证据从一条变成两条**互相独立**的：

1. **共享门项**：`gate_multiplier = 0.0` ⇒ 门项乘子为零 ⇒ 传输对比项不进前向。
2. **独立 C 项**：C 因子根本不存在 ⇒ 即使门项打开，也没有第二条传输通路。

在 `_SharedAtoms._project` 里，传输分支是

```python
def _project(self, coeff, extra):
    update = F.linear(coeff, self.up)
    if self.transport_up is not None:
        update = update + F.linear(extra, self.transport_up)
    return update
```

**C 这一条通路根本不过门项。**
所以历史臂是**双重缺席**：共享门项被乘子关掉，独立 C 项压根没建。

**因此历史臂在数学上与"普通 r=8 LoRA"完全等价。**
CLAUDE.md 说"完整关闭传输需关闭共享门项和独立 C 项"——历史臂**两项都关了**。
这不是"部分关闭"，是**完整关闭**。

### 2d. **传输项真的有效吗？**（在数值上真正生效的配置下实测）

新的三臂比较（§7）要跑 `cota-qa-live`：multiplier 1 + FP32 传输通路 + 16× 门学习率。
在 16 行的小冒烟训练上实测：

- **36/36 个门参数拿到了优化器状态**（对比：历史臂 0/144）；
- 门权重**从全零变成了非零**（8 步后 max |gate| ≈ 1.2e-3）；
- `transport_up`（C）**从零初始化长到 |.| ≈ 15**（普通 `up` 是 0.026 量级）。

**但有一个必须记下来的警告：**
在同一个冒烟跑里，**被选中评测的 checkpoint-1 的门项仍然是全零**
（开发集分数打平，`select()` 取**第一个**最大值）。
所以"门项可训练"不等于"被评测的那个 checkpoint 的门项是活的"。

**为此在比较脚本里加了显式检查**（`run_comparison.sh` 的 gate-live 段）：
读**被选中 checkpoint** 的 `gate` 张量，若仍为数值零就**打印 WARNING**，
并且这条 WARNING 本身就是结果——那意味着这个臂测的还是和历史臂一样的东西。

**另一个更根本的观察：** `gate` 的 |.| ≈ 1e-3，而独立 C 的 |.| ≈ 15。
按 `_project` 的结构，**C 这一条通路上没有门项**，
所以真正起作用的是 C，不是门。
**门项是被乘在 `A·contrast` 上的那一项**（`coeff + gate * extra`），
它的量级比 C 小几个数量级。
**所以"传输机制"真正起作用的是独立 C，而不是条件门。**
这一点必须在论文里说清楚，否则会把功劳记错地方。

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

## 7. 各因子**实际移动了多少**（本次新增，且是**验证过的**）

仓库只保存了初始化的 SHA256，**没有保存初始张量**。
但初始化是**完全可复现**的：`adapter_audit.initialize_pair` 用**独立的 CPU RNG**
（`torch.Generator().manual_seed(seed)`）逐层抽 `kaiming_uniform_(a=sqrt(5))`，
初始 B 恰好为零（PEFT 约定）。

**所以可以重建初始 A，然后用记录的 SHA 检验重建是否正确**——
这是**验证**，不是假设。工具：`research/tools/adapter_displacement.py`。

**对全部 30 个臂逐一验证：SHA 全部精确匹配**（`VERIFIED`），
所以下面的位移量是**真实位移**，不是估计。

| 臂 | 层数 | 相对 A 位移（均值） | 区间 | 乘积 \|BA\|（均值） |
|---|---:|---:|---|---:|
| A-first4 | 4 | **0.529** | [0.467, 0.571] | 1.061 |
| A-first4-s43 | 4 | 0.497 | [0.403, 0.607] | 1.027 |
| A-first4-s44 | 4 | 0.451 | [0.411, 0.488] | 0.891 |
| A-last4 | 4 | **0.442** | [0.361, 0.485] | 0.975 |
| A-last4-s43 | 4 | 0.446 | [0.384, 0.501] | 0.934 |
| A-last4-s44 | 4 | 0.300 | [0.272, 0.319] | 0.547 |
| A-mid4 | 4 | 0.387 | [0.355, 0.463] | 0.755 |
| A-mid4-s43 | 4 | 0.370 | [0.348, 0.406] | 0.763 |
| A-mid4-s44 | 4 | 0.312 | [0.272, 0.367] | 0.624 |
| A-all | 36 | **0.236** | [0.129, 0.346] | 0.478 |
| A-all-s43 | 36 | 0.231 | [0.118, 0.337] | 0.484 |
| A-all-s44 | 36 | 0.206 | [0.105, 0.310] | 0.405 |
| AQ-all | 36 | 0.205 | [0.136, 0.331] | 0.367 |
| AQ-all-s43 | 36 | 0.255 | [0.146, 0.378] | 0.474 |
| AQ-all-s44 | 36 | 0.179 | [0.096, 0.271] | 0.320 |
| AQ-first4 | 4 | **0.718** | [0.658, 0.800] | 1.307 |
| B-audio | 36 | 0.325 | [0.000, 0.584] | 0.712 |
| C-post-audio | 36 | 0.219 | [0.133, 0.306] | 0.454 |
| D-all-shuffled | 36 | 0.232 | [0.114, 0.362] | 0.467 |

### 这条结果怎么用

**1. 层段效应有一个此前没有量化的混杂解释。**
4 层臂的 A 相对位移是 36 层臂的 **2–3 倍**（0.30–0.72 对 0.18–0.33），
乘积范数也是 **2–3 倍**。**层段臂不只是"参数少"，它们移动得更远。**
这是**参数量不同**的必然后果（同样的梯度、同样多的步数，更少的参数移动更多），
**但它意味着「后 4 层」和「多层」不只是容量不同，优化的难度也不同。**
论文必须写清楚：层段对照里**同时变了参数量和位移量**。

**2. 但它不能解释层段效应的方向。**
更大的位移通常意味着**更强的拟合**。而实测是：**位移最大的 `A-first4`（0.53–0.72），
在 IEMOCAP 上校正后比基座低 −5 到 −6 pp**；**位移最小的多层臂（0.18–0.24）
保留了 −5 到 −6 pp 的优势之外的判别能力**（第 4 章）。
**所以「移动得更远 = 学到更多」在这里不成立**——这条混杂**不足以**解释层段差异，
反而让「后 4 层移动得更远却更差」更值得解释。

**3. 乘积位移等于乘积本身**（`\|BA_trained − BA_init\| = \|BA_trained\|`，因为 `BA_init = 0`），
这一点是平凡的（初始 B 是零）。**真正可解释的量是乘积的最终范数**，
上面表里已经给了。

**4. `B-audio` 的区间下界是 0.000**——这个臂的更新只加在音频 token 位置上，
所以**大部分层在大部分训练样本上根本没被更新**，相对位移是 0。
**这是设计使然**（`update_positions='audio'`），不是缺陷。

**5. 一个必须写清的限制**：上面是**位移的量**，**不是位移的方向**。
方向要靠第 4 章的类别边缘分析和第 5 章的负翻转分析来读，
本节**不能**单独用来支持任何机制主张。

## 8. 一件**已设计但从未训练**的对照（影响论文能声称什么）

`experiments/cota_review/matrix.json` 里有 8 个臂的配置，
**但 `experiments/cota_review/` 下只有 `configs/` 和 `matrix.json`，没有任何 `result.json`**，
服务器上也没有对应的 `train/` 目录。**这些臂一个都没有训练过。**

| 臂 | 方法 | r/alpha | gate 乘子 | transport_gate | 独立 C | 可训练参数 |
|---|---|---|---:|---|---|---:|
| `cota-qa` | ta-lora | 8/16 | **1.0** | question_audio | 是 | 1,772,064 |
| `cota-qa-live` | ta-lora | 8/16 | **1.0** | question_audio | 是 | 1,772,064 |
| `cota-static` | ta-lora | 8/16 | **1.0** | static | 是 | 1,769,760 |
| `ctrl-audio-mean` | ta-lora | 8/16 | **1.0** | static | 是 | 1,769,760 |
| `ctrl-ordinary` | ta-lora | 8/16 | **1.0** | static | 是 | 1,769,760 |
| `lora-r12` | **lora** | **12/24** | — | — | — | 1,769,472 |
| `lora-r8` | **lora** | 8/16 | — | — | — | 1,179,648 |
| `shared-r12` | ta-lora | 12/24 | **1.0** | static | 否 | 1,769,904 |

三点必须写下来：

1. **`cota-qa-live` 是"把门项做成数值有效"的那个臂**：
   它的说明是"gate made numerically effective: FP32 transport path and 16x gate learning rate"。
   **这正是 §2 证明本文所有臂缺的那一块**——
   本文所有臂的 `gate_multiplier` 是 0，而 `cota-qa-live` 是 1，**并且**用了 FP32 传输路径
   （`transport_precision='fp32'`），**并且**给门项 16 倍学习率
   （`transport_gate_lr_multiplier: 16`）。
   **这三条正好对应源码第 90–101 行注释里说的那个问题**：
   BF16 autocast 会把门项乘子抹掉（`|m − 1| < 2^-9` 就没了），所以必须走 FP32 路径。

2. **匹配 LoRA 对照（`lora-r12`，r12/alpha24）也从未训练。**
   这是 `CLAUDE.md` 明确要求的名字（"匹配 LoRA 为 r12/alpha24，CoTA 为 r8/alpha16，
   scale 都为 2"），**本轮核对的 30 个 checkpoint 里没有它**。
   **所以本文的层段结论是"CoTA 配置的低秩更新 vs 冻结基座"，不是"CoTA vs 匹配 LoRA"。**
   论文**不能**声称优于或持平于匹配 LoRA。

3. **`lora-r8`（r8/alpha16，可训练参数 1,179,648）与本文的多层臂
   （`A-all`：1,179,936）几乎完全同参数量**——
   差 288 个参数，正好是 36 层的 8 个 gate。
   **这就是一个现成的、参数匹配的对照臂**，而且它的配置已经写好了。
   **如果只补一个训练，就该补这个**（在 IEMOCAP Session 4 上，1 个臂 ≈ 0.15–0.22 GPU 小时）。

**结论**：本文现在的定位是**分析论文**，证据全部来自**已训练的那 30 个臂**；
**`cota_review` 那一族是"方法侧的对照实验"，一次都没跑过**。
论文如果想声称任何关于 CoTA 传输机制的东西，
**必须先训练 `cota-qa-live` 和 `lora-r12`（以及 `lora-r8` 作为参数匹配对照）**，
这是下一步唯一的、真正需要 GPU 训练的建议（成本约 1 GPU 小时 × 3 个臂）。

## 9. 没有做的事

- **没有重新训练任何臂**，没有验证"传输项若启用会怎样"。
- **没有核对 `experiments/cota_review/` 那批 `gate_multiplier=1.0` 的结果**——
  它们不在本文分析链上，本轮只核对了它们的配置存在。
- **没有检查 `down`/`up` 的实际位移量**：仓库只记了 `ab_init_sha256`（A+B 拼接的哈希），
  **没有保存初始张量本身**，所以"移动了多少"算不出来。
  本轮给出的是**每个因子的最终范数**（`down` |·| ≈ 1.69–2.02，`up` |·| ≈ 0.27–0.62，
  `gate` = 0），**不是位移**。这一点必须写清楚，不能拿范数当位移讲。
- **位移的方向没有分析**：本轮只算了位移的**量**，没有算它指向哪里。
  方向由第 4、5 章的分析负责。
- **没有检查 `experiments/cota_review/` 的结果**（见 §6.3）。
