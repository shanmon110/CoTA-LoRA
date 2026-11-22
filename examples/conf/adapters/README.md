# WEST adapter configurations

LibriSpeech (Qwen2.5-Omni Thinker) and AISHELL / AISHELL-2 (TouchASU with a Qwen2 or Qwen3 text decoder) use these same method files. In a WEST model JSON, set `lora_config` to the contents of one file. To reuse a preset across model files, pass `--adapter_config FILE` to `west/bin/train.py`, or `--adapter-config FILE` to the three example scripts. The external file replaces the entire inline LoRA configuration, including its target list. Existing TouchASU configs without a `method` field still use PEFT and keep their configured rank, dropout and target modules. A null TouchASU `lora_config` keeps the original projector-only training.

| Method JSON | Target | Preset-specific settings |
| --- | --- | --- |
| [o-lora.json](o-lora.json) | Attention output | Ordinary low-rank residual |
| [ta-lora.json](ta-lora.json) | Attention output | Modality-conditioned residual with learned gate |
| [qv-lora.json](qv-lora.json) | Query and value projections | `match_ta_budget` uses actual projection widths. With this Omni-7B config, reference rank 8 resolves to rank 5 and `lora_alpha=10` gives scale 2 |
| [toplora.json](toplora.json) | Output projection | Token-dependent scaling |
| [bora.json](bora.json) | Output projection | `bora_blocks=2`, must divide both projection widths |
| [dora.json](dora.json) | Output projection | Learned row magnitude and low-rank direction |
| [eva.json](eva.json) | Output projection | `eva_samples=128`, `eva_seed=42`, `eva_rho=2`, `eva_token_mask="labels"` |

The common fields are `method`, `r`, `lora_alpha`, `last_n` and `target_modules`. The published presets target the last four text decoder layers. Set `last_n` within the decoder's actual depth. Tagged configs support `lora_dropout=0`, `task_type="CAUSAL_LM"` and `inference_mode=false`. Invalid targets, unsupported settings and method-specific keys belonging to a different method raise errors. EVA also accepts `eva_token_mask="nonpad"`. EVA fits on the current training input and saves the resulting per-layer ranks with the weights.

## Training and checkpoints

Model and encoder selection stay in the original model JSON. Set training fields in its optional `training_args` object or pass a separate JSON object using `--training_config FILE`. The external training file overlays inline defaults. Explicit training arguments such as `--max_steps` take precedence. Put method parameters in the adapter configuration. The examples use the same `west/bin/train.py` and `west/bin/decode.py` entry points.

* **TouchASU** freezes the encoder and base LLM and trains the speech projector with the selected adapter. The AISHELL scripts default to [BF16 AMP with unpacked utterances](../training/touch_asu_adapters.json) for tagged models. For the native packed recipe, set `adapter_attention_backend` to `flash_attention_2` in the model JSON and pass the [original training profile](../training/touch_asu_asr.json). This selects 8,192-token packing with four accumulated microbatches and BF16 DeepSpeed ZeRO-2. The adapter computes its low-rank residual in FP32 under both profiles. ZeRO-2 uses BF16 model parameters and FP32 optimizer master weights. WEST records the precision profile in the checkpoint and verifies it on resume. Keep Trainer's `per_device_train_batch_size=1` because the WEST data reader builds each batch. EVA calibrates on individual eligible training utterances before packing starts. Decode the full WEST Trainer checkpoint, which also contains the trained projector.
* **Omni** freezes the entire base Thinker and trains the adapter. It saves an adapter-only checkpoint bound to its base and processor. The multi-method recipe selects its adapter files from the comparison configuration or an explicit single-method override. For the complete ASR protocol, see the [LibriSpeech recipe](../../librispeech/asr/README.md).

## Adding another method or example

A dataset using either supported model host can reuse a method file and supply its own model, data and training configuration. A different model architecture needs a host integration for its decoder projections. Attention methods also need the host's attention implementation and, for TA-LoRA, the actual speech spans throughout forward, generation and checkpoint recomputation.

A projection method subclasses `AdapterMethod` and supplies a unique configuration name and checkpoint identity, its branch, and any extra options or tensor shapes. Register it in `west_adapters/registry.py` so training and fresh checkpoint loading resolve the same method. See [BoRA](../../../west_adapters/bora.py) for an example using shared low-rank weights. An attention method supplies `attention_delta` to consume the native attention tensors and declares `needs_audio_context` when it uses speech spans. See [TA-LoRA](../../../west_adapters/ta_lora.py). Add a preset and a train/save/reload test. A LibriSpeech comparison lists its chosen presets in [comparison.json](../../librispeech/asr/conf/comparison.json).
