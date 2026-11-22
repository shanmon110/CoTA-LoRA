"""Omni ASR integration for WEST's Transformers Trainer entry point."""
from contextlib import ExitStack
from dataclasses import dataclass
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader
from transformers import Trainer, TrainerCallback
from transformers.trainer_callback import PrinterCallback, ProgressCallback

from .attention import audio_context, checkpoint_audio_context
from .budget import Budget
from .checkpoint import load_adapter, save_adapter
from .data import load_manifest
from .diagnostics import SmokeLog
from .evaluate import evaluate
from .runtime import move_inputs
from .train import prepare_training, validate_output


@dataclass
class OmniArguments:
    data_path: str
    dev_data_path: str
    pack_size: int = 0
    dev_patience: int = 1
    max_new_tokens: int = 128
    max_audio_seconds: float | None = None
    max_batch_frames: int | None = None
    max_sequence_length: int | None = None
    smoke_log: str | None = None


@dataclass
class LegacyAdapterArguments:
    """Flags accepted by existing recipes. New methods declare options in their plugins."""
    mode: str = "lora"
    target_set: str = "o"
    rank: int = 8
    alpha: float = 16.0
    last_n: int = 4
    match_ta_budget: bool = False
    bora_blocks: int = 2
    eva_samples: int = 128
    eva_rho: float = 2.
    eva_seed: int = 42
    eva_token_mask: str = "labels"


def adapter_options(spec, legacy, explicit, training_defaults):
    if spec is None:
        return legacy
    from west_adapters import resolve_lora_config

    resolved = resolve_lora_config(spec)
    if training_defaults.keys() & (legacy.keys() | resolved.keys()):
        raise ValueError("Put adapter parameters in lora_config or --adapter_config")
    # Accept existing CLI hyperparameter overrides if they belong to this
    # plugin. Never change which method the selected configuration represents.
    flags = explicit & legacy.keys()
    unsupported = flags - resolved.keys()
    if unsupported:
        raise ValueError(f"Invalid adapter overrides for {spec['method']}: {sorted(unsupported)}")
    for name in flags & {"mode", "target_set"}:
        if legacy[name] != resolved[name]:
            raise ValueError(f"--{name} conflicts with configured method {spec['method']}")
    names = {"rank": "r", "alpha": "lora_alpha"}
    overrides = {names.get(k, k): legacy[k] for k in flags - {"mode", "target_set"}}
    return resolve_lora_config({**spec, **overrides})


class OmniTrainer(Trainer):
    """Keep the ASR sample order and normalize across all answer tokens in each update."""
    tokens = 0
    summed_loss = 0.

    def get_train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.args.per_device_train_batch_size,
                          shuffle=True, collate_fn=self.data_collator,
                          generator=torch.Generator().manual_seed(self.args.seed))

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        inputs = move_inputs(inputs, self.args.device, self.model.audio_tower.dtype)
        count = int(inputs["labels"][:, 1:].ne(-100).sum().item())
        if count == 0:
            raise ValueError("No supervised next tokens in microbatch")
        with audio_context(inputs["input_ids"], inputs["attention_mask"], self.model.config.audio_token_id):
            loss = model(**inputs, use_cache=False).loss.float()
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f"Nonfinite loss at update {self.state.global_step + 1}")
            self.accelerator.backward(loss * count)
        self.tokens += count
        self.summed_loss += loss.detach() * count
        if self.accelerator.sync_gradients:
            grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
            torch._foreach_div_(grads, self.tokens)
            if not torch.stack([g.norm() for g in grads]).isfinite().all().item():
                raise FloatingPointError(f"Nonfinite gradient at update {self.state.global_step + 1}")
            self.update_metrics = dict(loss=(self.summed_loss / self.tokens).item(), supervised_tokens=self.tokens)
            self.tokens, self.summed_loss = 0, 0.
        return loss.detach() / self.args.gradient_accumulation_steps


def train_omni(a, args, model_spec, adapter):
    if args.world_size != 1 or args.n_gpu > 1 or args.deepspeed or args.fp16 or a.pack_size != 0:
        raise ValueError("Omni ASR requires one visible device, unpacked audio and BF16 on GPU")
    if args.device.type == "cuda" and not args.bf16:
        raise ValueError("Pass --bf16 True to use the Omni BF16 base on GPU")
    if (args.dataloader_num_workers or args.dataloader_drop_last or args.save_strategy != "no"
            or args.eval_strategy != "no" or args.load_best_model_at_end):
        raise ValueError("Omni dev WER and adapter saving require --eval_strategy no --save_strategy no "
                         "and a complete, single-worker DataLoader")
    if not args.eval_steps or int(args.eval_steps) != args.eval_steps or args.eval_steps < 1:
        raise ValueError("Use an integer --eval_steps greater than zero for generated dev WER")
    if (min(adapter["rank"], adapter["last_n"], a.dev_patience, a.max_new_tokens) < 1
            or not math.isfinite(adapter["alpha"]) or adapter["alpha"] <= 0
            or (adapter.get("match_ta_budget") and adapter["target_set"] != "qv")):
        raise ValueError("Invalid Omni adapter or validation options")
    output = Path(args.output_dir)
    validate_output(output)
    budget = Budget.from_args(a)
    setup = SimpleNamespace(**vars(a), task="asr", seed=args.seed, device=str(args.device),
                            model_name_or_path=model_spec["model"], revision=model_spec.get("revision"),
                            base_sha256=model_spec.get("base_sha256"), manifest=a.data_path,
                            batch_size=args.per_device_train_batch_size)
    model, processor, records, collate, config, smoke = prepare_training(setup, budget, adapter)
    dev = load_manifest(a.dev_data_path, "asr")
    train_files = {r["audio"] for r in records}
    if any(r["audio"] in train_files for r in dev):
        raise ValueError("Dev audio overlaps training data")
    if args.gradient_checkpointing:
        args.gradient_checkpointing_kwargs = {"use_reentrant": False}
        from west_adapters.registry import checkpoint_method
        if checkpoint_method(config).needs_audio_context:
            args.gradient_checkpointing_kwargs["context_fn"] = checkpoint_audio_context

    class DevSelection(TrainerCallback):
        best_errors = math.inf
        best_step = None
        misses = 0

        def on_step_end(self, args, state, control, **kwargs):
            end = state.global_step
            print(json.dumps(dict(step=end, **trainer.update_metrics)), flush=True)
            if end % args.eval_steps != 0 and end < state.max_steps:
                return
            stages.close()
            save_adapter(model, output / f"step-{end}", config)
            dev_log = SmokeLog(str(output / f"dev-{end}.jsonl"), args.device)
            with dev_log.stage("evaluate"):
                score = evaluate(model, processor, dev, task="asr", device=args.device,
                                 batch_size=args.per_device_eval_batch_size, max_new_tokens=a.max_new_tokens,
                                 budget=budget, smoke=dev_log)
            if score["word_errors"] < self.best_errors:
                self.best_errors, self.best_step, self.misses = score["word_errors"], end, 0
            else:
                self.misses += 1
            stop = self.misses >= a.dev_patience and end < state.max_steps
            decision = dict(validation_step=end, score=score, best_step=self.best_step,
                            non_improving_checks=self.misses, early_stopped=stop)
            with (output / "dev-history.jsonl").open("a") as stream:
                print(json.dumps(decision), file=stream)
            print(json.dumps(decision), flush=True)
            if stop:
                control.should_training_stop = True
            elif end < state.max_steps:
                stages.enter_context(smoke.stage("train"))

    selection = DevSelection()
    trainer = OmniTrainer(model=model, args=args, train_dataset=records, data_collator=collate,
                          callbacks=[selection])
    # stdout is a JSON-lines contract consumed by the seven-arm recipe.
    trainer.remove_callback(PrinterCallback)
    trainer.remove_callback(ProgressCallback)
    # The snapshot is loaded directly in BF16. Preserve Omni's explicit
    # per-operation dtype rules and avoid casting vocabulary logits to FP32 a
    # second time via Accelerate's AMP output wrapper.
    trainer.accelerator.native_amp = False
    if trainer.accelerator.gradient_accumulation_steps != 1:
        raise ValueError("Transformers must own accumulation; omit Accelerate num_steps")
    with ExitStack() as stages:
        stages.enter_context(smoke.stage("train"))
        trainer.train()
    load_adapter(model, output / f"step-{selection.best_step}")
    save_adapter(model, output, config)
    trainer.save_state()
    with (output / "train_args.json").open("x") as stream:
        json.dump(dict(adapter={**vars(a), **adapter}, trainer=args.to_dict()), stream, indent=2, default=str)
