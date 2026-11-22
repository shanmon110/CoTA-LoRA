# Copyright (c) 2025 Binbin Zhang(binbzha@qq.com)
# This code is based on the QWen2 from
# https://github.com/QwenLM/Qwen2/blob/main/examples/sft/finetune.py

import argparse
import json
import logging
import os
import pathlib
import sys
from dataclasses import dataclass, field
from typing import Any, Union

import torch
from torch import nn
from transformers import (AutoConfig, AutoModel, HfArgumentParser, Trainer,
                          TrainerCallback, TrainingArguments, set_seed)
from transformers.trainer_utils import get_last_checkpoint


@dataclass
class TrainingArguments(TrainingArguments):
    optim: str = field(default="adafactor")
    model_config_or_dir: str = field(default='')
    adapter_config: str = field(default='', metadata={'help': 'JSON LoRA configuration for a new model.'})
    training_config: str = field(default='', metadata={'help': 'JSON defaults for training and data arguments.'})


class MyTrainer(Trainer):

    def training_step(self,
                      model: nn.Module,
                      inputs: dict[str, Union[torch.Tensor, Any]],
                      num_items_in_batch=None) -> torch.Tensor:
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()
        inputs = self._prepare_inputs(inputs)
        # loss compute will aplly `shift_labels`
        # See https://github.com/huggingface/transformers/blob/v4.52.2/src/transformers/trainer.py#L3825  # noqa
        with self.compute_loss_context_manager():
            loss, outputs = self.compute_loss(model,
                                              inputs,
                                              return_outputs=True)
        # Compute accuracy (ignoring ignore_index)
        if "labels" in inputs:
            logits = outputs.logits[..., :-1, :].contiguous()
            labels = inputs["labels"][..., 1:].contiguous()
            preds = torch.argmax(logits, dim=-1)
            mask = labels != -100
            correct = (preds == labels) & mask
            accuracy = correct.sum().item() / mask.sum().item()
            interval = \
                self.args.logging_steps * self.args.gradient_accumulation_steps
            if self.state.global_step % interval == 0:
                self.log({
                    "train_loss": loss.detach().cpu().item(),
                    "train_accuracy": accuracy
                })

        if self.args.n_gpu > 1:
            loss = loss.mean()
        loss = loss / self.args.gradient_accumulation_steps

        self.accelerator.backward(loss)
        return loss.detach()


class ProfilerCallback(TrainerCallback):

    def __init__(self, log_dir="./logs"):
        self.log_dir = log_dir
        self.profiler = None
        self.started = False
        self.global_step = 0

    def on_step_begin(self, args, state, control, **kwargs):
        self.global_step = state.global_step

        if not self.started:
            self.profiler = torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=999,
                                                 warmup=0,
                                                 active=1,
                                                 repeat=0,
                                                 skip_first=5),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    self.log_dir),
                record_shapes=True,
                profile_memory=True,
                with_stack=False,
                with_modules=True)
            self.profiler.__enter__()
            self.started = True

    def on_step_end(self, args, state, control, **kwargs):
        if self.started and self.profiler is not None:
            self.profiler.step()

    def on_train_end(self, args, state, control, **kwargs):
        if self.profiler is not None:
            self.profiler.__exit__(None, None, None)


def read_config(path):
    with open(path, encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def parse_configured_args(parser, defaults, argv=None):
    """Use JSON defaults and keep HfArgumentParser's normal explicit CLI overrides."""
    fields = {action.dest: action for action in parser._actions}
    reserved = {"help", "model_config_or_dir", "training_config", "adapter_config"}
    invalid = (defaults.keys() - fields.keys()) | (defaults.keys() & reserved)
    if invalid:
        raise ValueError(f"Unknown or reserved training configuration fields: {sorted(invalid)}")
    for name, value in defaults.items():
        if value is None and fields[name].required:
            raise ValueError(f"Required argument {name} cannot be null")
        fields[name].required = False
    parser.set_defaults(**defaults)
    return parser.parse_args_into_dataclasses(args=argv)


def validate_tagged_training(config, args, *, restoring=False):
    if not getattr(type(config), "supports_west_adapters", False):
        raise ValueError(f"Tagged adapter methods are not implemented for {config.model_type}")
    if args.per_device_train_batch_size != 1:
        raise ValueError("WEST supplies prebatched examples; use per_device_train_batch_size=1 and set batch_size")
    if args.fsdp:
        raise ValueError("Tagged TouchASU adapters do not support FSDP")
    precision = "fp32"
    if args.deepspeed:
        ds = args.hf_deepspeed_config  # TrainingArguments resolves the recipe's 'auto' values
        if not (args.bf16 and ds.is_zero2() and ds.get_value("bf16.enabled") is True):
            raise ValueError("Tagged TouchASU DeepSpeed training requires BF16 and ZeRO stage 2")
        precision = "bf16_zero2"
    saved = getattr(config, "adapter_parameter_precision", "fp32" if restoring else None)
    if saved is not None and saved != precision:
        raise ValueError(f"Cannot change adapter_parameter_precision from {saved!r} to {precision!r}")
    config.adapter_parameter_precision = precision


def tagged_lora(lora):
    return isinstance(lora, dict) and "method" in lora


def resume_native_config(config, checkpoint, pretrained_source=None):
    """Recover fitted ranks before weight loading and prevent combining unrelated adapters."""
    checkpoint = pathlib.Path(checkpoint)
    path = checkpoint / "config.json"
    saved = read_config(path) if path.is_file() else {}
    lora = getattr(config, "lora_config", None)
    if not (tagged_lora(lora) or tagged_lora(saved.get("lora_config"))):
        return config
    if saved.get("lora_config") != lora:
        raise ValueError(f"Cannot resume with a changed lora_config: {checkpoint}")
    if pretrained_source and pathlib.Path(pretrained_source).resolve() not in (
            checkpoint.resolve(), checkpoint.resolve().parent):
        raise ValueError(f"Pretrained source and output checkpoint belong to different runs: {checkpoint}")
    # TouchASU derives hidden_size from its bound LLM during construction.
    for key in ("model_type", "llm_model_name_or_path", "wenet_model_name_or_path",
                "encoder_ds_rate", "encoder_projector_ds_rate", "projector_hidden_size"):
        if saved.get(key) != getattr(config, key, None):
            raise ValueError(f"Cannot resume with a changed {key}: {checkpoint}")
    # Legacy tagged checkpoints used eager attention and retained FP32 adapters.
    for key, default in (("adapter_attention_backend", "eager"), ("adapter_parameter_precision", "fp32")):
        if saved.get(key, default) != getattr(config, key, default):
            raise ValueError(f"Cannot resume with a changed {key}: {checkpoint}")
    return AutoConfig.from_pretrained(checkpoint)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    # The Omni Thinker uses its own multimodal collator. Keep unrelated registry
    # imports (including WeNet speech encoders) on the existing WEST model path.
    route = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    route.add_argument("--model_config_or_dir", "--model-config-or-dir", default="")
    route.add_argument("--adapter_config", "--adapter-config")
    route.add_argument("--training_config", "--training-config")
    cli, _ = route.parse_known_args()
    model_file = cli.model_config_or_dir
    model_spec = read_config(model_file) if os.path.isfile(model_file) else {}
    if cli.adapter_config and not os.path.isfile(model_file):
        raise ValueError("--adapter_config requires a model JSON; pretrained checkpoints restore their own adapters")
    lora = read_config(cli.adapter_config) if cli.adapter_config else model_spec.get("lora_config")
    training_defaults = model_spec.get("training_args", {})
    if not isinstance(training_defaults, dict):
        raise ValueError("training_args must be a JSON object")
    if cli.training_config:
        training_defaults = {**training_defaults, **read_config(cli.training_config)}
    if model_spec.get("model_type") == "qwen2_5_omni_adapter":
        from examples.omni_talora.west_trainer import (LegacyAdapterArguments, OmniArguments,
                                                       adapter_options, train_omni)
        if "adapter_config" in model_spec:
            raise ValueError("Use lora_config in the Omni model JSON or --adapter_config on the command line")
        parser = HfArgumentParser((OmniArguments, TrainingArguments, LegacyAdapterArguments), allow_abbrev=False)
        options, training_args, legacy = parse_configured_args(parser, training_defaults)
        explicit = {parser._option_string_actions[flag].dest for arg in sys.argv[1:]
                    if (flag := arg.split("=", 1)[0]) in parser._option_string_actions}
        adapter = adapter_options(lora, vars(legacy), explicit, training_defaults)
        train_omni(options, training_args, model_spec, adapter)
        return

    from west.dataset.dataset import DataArguments, SpeechDataset
    from west.dataset.extractor import Extractor

    parser = HfArgumentParser((DataArguments, TrainingArguments))
    data_args, training_args = parse_configured_args(parser, training_defaults)
    checkpoint = get_last_checkpoint(training_args.output_dir) if os.path.isdir(training_args.output_dir) else None
    if os.path.isfile(training_args.model_config_or_dir):  # init from config
        config = AutoConfig.from_pretrained(training_args.model_config_or_dir)
        if cli.adapter_config:
            if not hasattr(config, "lora_config"):
                raise ValueError(f"{config.model_type} does not expose lora_config")
            config.lora_config = lora
        if hasattr(config, "training_args"):
            del config.training_args
        # A model JSON starts fresh from the configured base. Fitted layer
        # ranks are paired with checkpoint tensors and restored together below.
        if tagged_lora(getattr(config, "lora_config", None)) and hasattr(config, "resolved_adapter_config"):
            del config.resolved_adapter_config
        if tagged_lora(getattr(config, "lora_config", None)):
            validate_tagged_training(config, training_args)
            set_seed(training_args.seed)
        if checkpoint:
            config = resume_native_config(config, checkpoint)
        model = AutoModel.from_config(config)
    else:  # load from pretrained
        source = training_args.model_config_or_dir
        if checkpoint:
            config = resume_native_config(AutoConfig.from_pretrained(source), checkpoint, source)
            if tagged_lora(getattr(config, "lora_config", None)):
                source = checkpoint
        model = AutoModel.from_pretrained(source)
        config = model.config
        if tagged_lora(getattr(config, "lora_config", None)):
            validate_tagged_training(config, training_args, restoring=True)
    p = (getattr(data_args, 'spk_prompt_wav_map_path', None) or '').strip()
    tokenizer = model.init_tokenizer()
    extractor_args = {"spk_prompt_wav_map_path": p} if p else {}
    extractor = Extractor.get_class(model.model_type)(tokenizer, config, **extractor_args)

    print("Loading data...")
    train_dataset = SpeechDataset(extractor, data_args)
    if getattr(model, "needs_adapter_calibration", False):
        model.to(training_args.device)
        model.calibrate_adapter(train_dataset)
    if tagged_lora(getattr(config, "lora_config", None)) and training_args.process_index == 0:
        layout = model.config.resolved_adapter_config
        summary = {
            "method": config.lora_config["method"],
            "rank": layout["rank"],
            "alpha": layout["alpha"],
            "scale": layout["alpha"] / layout["rank"],
            "last_n": layout["last_n"],
            "adapter_parameters": sum(p.numel() for p in model.llm.parameters() if p.requires_grad),
            "projector_parameters": sum(p.numel() for p in model.projector.parameters() if p.requires_grad),
            "adapter_parameter_precision": getattr(model.config, "adapter_parameter_precision", "fp32"),
            "adapter_attention_backend": getattr(config, "adapter_attention_backend", "eager"),
        }
        if config.lora_config["method"] == "eva":
            summary.update(layer_ranks=layout["layer_ranks"], calibration_samples=layout["eva_samples"])
        logging.info("TouchASU adapter %s", json.dumps(summary, sort_keys=True))
    # Start trainer
    trainer = MyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=lambda x: x[0],
        callbacks=[ProfilerCallback(log_dir=training_args.logging_dir)],
    )
    if checkpoint:
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        trainer.train()
    trainer.save_state()


if __name__ == "__main__":
    # WEST recipes also execute a symlinked west/bin/train.py from their own directory.
    if not __package__:
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    main()
