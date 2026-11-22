"""Run WEST's public training entry on the tiny Thinker and the official processor."""
import json
import os
from pathlib import Path
import runpy
import sys

import numpy as np
import pytest
import soundfile as sf
import torch
from safetensors.torch import load_file

if __package__:
    from .test_omni_ta_integration import write_tiny_checkpoint
else:
    from test_omni_ta_integration import write_tiny_checkpoint


@pytest.mark.skipif(not os.getenv("OMNI_PROCESSOR_DIR"), reason="requires local official processor")
@pytest.mark.parametrize("patience, updates, examples", [(1, 2, 8), (3, 3, 9)])
def test_native_trainer_validation_and_tail(tmp_path, monkeypatch, patience, updates, examples):
    base = tmp_path / "base"
    base.mkdir()
    write_tiny_checkpoint(base, Path(os.environ["OMNI_PROCESSOR_DIR"]))
    config = tmp_path / "model.json"
    config.write_text(json.dumps({"model_type": "qwen2_5_omni_adapter", "model": str(base)}))
    rows = []
    for i in range(10):
        wav = tmp_path / f"{i}.wav"
        sf.write(wav, np.sin(np.arange(2400) * (i+1) * .01), 16000)
        rows.append(json.dumps({"audio": str(wav), "text": "hi there", "id": str(i)}))
    train, dev = tmp_path / "train.jsonl", tmp_path / "dev.jsonl"
    train.write_text("\n".join(rows[:-1]))
    dev.write_text(rows[-1])
    output = tmp_path / "adapter"
    trace = tmp_path / "train-events.jsonl"
    script = Path(__file__).resolve().parents[1] / "west/bin/train.py"
    # lr=0 keeps generations identical, testing WER ties without guessing the tiny model's output.
    monkeypatch.setattr(sys, "argv", [str(script), "--model_config_or_dir", str(config), "--use_cpu", "true",
        "--data_path", str(train), "--dev_data_path", str(dev), "--output_dir", str(output),
        "--per_device_train_batch_size", "2", "--gradient_accumulation_steps", "2", "--eval_steps", "1",
        "--save_strategy", "no", "--report_to", "none", "--disable_tqdm", "True", "--num_train_epochs", "1",
        "--mode", "ta", "--last_n", "1", "--rank", "2", "--alpha", "4", "--gradient_checkpointing", "True",
        "--dev_patience", str(patience), "--max_new_tokens", "1", "--lr_scheduler_type", "constant",
        "--optim", "adamw_torch", "--learning_rate", "0", "--smoke_log", str(trace)])
    runpy.run_path(str(script), run_name="__main__")
    decisions = [json.loads(s) for s in (output / "dev-history.jsonl").read_text().splitlines()]
    assert len(decisions) == updates
    assert decisions[-1]["early_stopped"] == (patience == 1)
    assert decisions[-1]["best_step"] == 1
    actual = [s["id"] for e in map(json.loads, trace.read_text().splitlines())
              if e["event"] == "batch" and e["stage"] == "train" for s in e["samples"]]
    g = torch.Generator().manual_seed(42)
    expected = [str(i) for batch in torch.utils.data.DataLoader(range(9), batch_size=2,
                shuffle=True, generator=g) for i in batch.tolist()]
    assert actual == expected[:examples]
    assert json.loads((output / "trainer_state.json").read_text())["global_step"] == updates
    first = load_file(output / "step-1" / "adapter.safetensors")
    assert all(torch.equal(v, first[k]) for k, v in load_file(output / "adapter.safetensors").items())


def test_native_accumulation_matches_pooled_sgd_including_short_tail(tmp_path):
    from types import SimpleNamespace
    from transformers import TrainingArguments
    from examples.omni_talora.train import train_updates
    from examples.omni_talora.west_trainer import OmniTrainer
    if __package__:
        from .test_omni_ta_train import Classifier, batch
    else:
        from test_omni_ta_train import Classifier, batch

    native, pooled = Classifier(), Classifier()
    native.config = SimpleNamespace(audio_token_id=8)
    native.audio_tower = SimpleNamespace(dtype=torch.float32)
    labels = [[1], [0, 2, 0], [2, 0]]
    args = TrainingArguments(output_dir=str(tmp_path), use_cpu=True, optim="sgd", learning_rate=.1,
                             lr_scheduler_type="constant", num_train_epochs=1, per_device_train_batch_size=1,
                             gradient_accumulation_steps=2, weight_decay=0, seed=42,
                             save_strategy="no", disable_tqdm=True, report_to="none")
    trainer = OmniTrainer(model=native, args=args, train_dataset=labels, data_collator=lambda rows: batch(rows[0]))
    trainer.train()
    loader = torch.utils.data.DataLoader(range(3), batch_size=1, shuffle=True,
                                         generator=torch.Generator().manual_seed(42))
    order = [i.item() for i in loader]
    opt = torch.optim.SGD(pooled.parameters(), lr=.1)
    for answers in (labels[order[0]] + labels[order[1]], labels[order[2]]):
        train_updates(pooled, [batch(answers)], opt, device="cpu", audio_token_id=8,
                      max_steps=1, accumulation_steps=1)
    assert trainer.state.global_step == 2
    torch.testing.assert_close(native.logits, pooled.logits)


def test_native_config_precedence_and_unknown_keys(tmp_path, monkeypatch):
    from examples.omni_talora import west_trainer

    # Exercise the public argument parser without allocating a speech model.
    script = Path(__file__).resolve().parents[1] / "west/bin/train.py"
    captures = []
    monkeypatch.setattr(west_trainer, "train_omni", lambda *args: captures.append(args))
    model = tmp_path / "model.json"
    model.write_text(json.dumps({
        "model_type": "qwen2_5_omni_adapter", "model": "tiny",
        "lora_config": {"method": "ta-lora", "r": 2, "lora_alpha": 4, "last_n": 1},
        "training_args": {"data_path": "train.jsonl", "dev_data_path": "dev.jsonl",
                          "output_dir": str(tmp_path / "out"), "gradient_accumulation_steps": 2,
                          "learning_rate": 0.04, "use_cpu": True, "report_to": "none"},
    }))
    adapter = tmp_path / "adapter.json"
    adapter.write_text(json.dumps({"method": "bora", "r": 8, "lora_alpha": 16,
                                   "bora_blocks": 2, "last_n": 4}))
    training = tmp_path / "train.json"
    training.write_text(json.dumps({"learning_rate": 0.02, "gradient_accumulation_steps": 4}))
    monkeypatch.setattr(sys, "argv", [str(script), "--model_config_or_dir", str(model),
                        "--adapter_config", str(adapter), "--training_config", str(training),
                        "--learning_rate", "0.01"])
    runpy.run_path(str(script), run_name="__main__")
    a, args, _, opts = captures.pop()
    assert (opts["mode"], opts["rank"], opts["alpha"], opts["last_n"], opts["bora_blocks"]) == ("bora", 8, 16., 4, 2)
    assert args.learning_rate == .01 and args.gradient_accumulation_steps == 4
    assert (a.data_path, a.dev_data_path, args.output_dir) == ("train.jsonl", "dev.jsonl", str(tmp_path / "out"))
    training.write_text(json.dumps({"learnng_rate": .0002}))
    with pytest.raises(ValueError, match="learnng_rate"):
        runpy.run_path(str(script), run_name="__main__")

    monkeypatch.setattr(sys, "argv", [str(script), "--model_config_or_dir", str(model)])
    runpy.run_path(str(script), run_name="__main__")
    _, args, _, opts = captures.pop()
    assert (opts["mode"], opts["rank"], args.learning_rate) == ("ta", 2, .04)

    monkeypatch.setattr(sys, "argv", [*sys.argv, "--mode", "lora"])
    with pytest.raises(ValueError, match="conflicts with configured method"):
        runpy.run_path(str(script), run_name="__main__")


def test_new_plugin_options_do_not_require_cli_fields(monkeypatch):
    from west_adapters import get_method
    from examples.omni_talora.west_trainer import LegacyAdapterArguments, adapter_options

    # Stand in for a newly registered plugin with no matching dataclass field.
    monkeypatch.setitem(get_method("o-lora").options, "extra_option", 11)
    legacy = vars(LegacyAdapterArguments(rank=2))
    opts = adapter_options({"method": "o-lora", "extra_option": 17}, legacy, {"rank"}, {})
    assert opts["extra_option"] == 17 and opts["rank"] == 2
    with pytest.raises(ValueError, match="eva_samples"):
        adapter_options({"method": "o-lora"}, legacy, {"eva_samples"}, {})


def test_native_models_keep_architectural_adapter_config(tmp_path, monkeypatch):
    """Exercise the public path without loading gated acoustic dependencies."""
    from dataclasses import dataclass
    from types import ModuleType, SimpleNamespace
    from transformers import AutoConfig, AutoModel

    @dataclass
    class DataArguments:
        data_path: str

    native_data = ModuleType("west.dataset.dataset")
    native_data.DataArguments, native_data.SpeechDataset = DataArguments, None
    extractor = ModuleType("west.dataset.extractor")
    extractor.Extractor = None
    monkeypatch.setitem(sys.modules, native_data.__name__, native_data)
    monkeypatch.setitem(sys.modules, extractor.__name__, extractor)
    monkeypatch.setitem(sys.modules, "west_adapters", None)
    spec = {"model_type": "goat_slm", "adapter_config": {"num_hidden_layers": 2}}
    model_file = tmp_path / "model.json"
    model_file.write_text(json.dumps(spec))
    monkeypatch.setattr(AutoConfig, "from_pretrained",
                        lambda path: SimpleNamespace(**json.loads(Path(path).read_text())))

    class ReachedNativeModel(Exception):
        pass

    def from_config(config):
        assert vars(config) == spec
        raise ReachedNativeModel

    monkeypatch.setattr(AutoModel, "from_config", from_config)
    script = Path(__file__).resolve().parents[1] / "west/bin/train.py"
    monkeypatch.setattr(sys, "argv", [str(script), "--model_config_or_dir", str(model_file),
                        "--data_path", "data.jsonl", "--output_dir", str(tmp_path / "out"), "--use_cpu"])
    with pytest.raises(ReachedNativeModel):
        runpy.run_path(str(script), run_name="__main__")


def test_tagged_west_prebatched_and_mixed_precision_limits():
    from accelerate.utils.deepspeed import HfDeepSpeedConfig
    from types import SimpleNamespace
    script = Path(__file__).resolve().parents[1] / "west/bin/train.py"
    validate = runpy.run_path(str(script))["validate_tagged_training"]
    config = type("Supported", (), {"model_type": "touch_asu", "supports_west_adapters": True})()
    args = SimpleNamespace(per_device_train_batch_size=1, deepspeed=None, fsdp=[])
    validate(config, args)
    args.per_device_train_batch_size = 2
    with pytest.raises(ValueError, match="prebatched"):
        validate(config, args)
    args.per_device_train_batch_size = 1
    args.bf16 = True
    args.deepspeed = {"zero_optimization": {"stage": 2}, "bf16": {"enabled": True}}
    args.hf_deepspeed_config = HfDeepSpeedConfig(args.deepspeed)
    with pytest.raises(ValueError, match="adapter_parameter_precision"):
        validate(config, args)
    del config.adapter_parameter_precision
    validate(config, args)
    assert config.adapter_parameter_precision == "bf16_zero2"
    args.hf_deepspeed_config = HfDeepSpeedConfig({"zero_optimization": {"stage": 3}, "bf16": {"enabled": True}})
    with pytest.raises(ValueError, match="DeepSpeed"):
        validate(config, args)
    args.deepspeed, args.fsdp = None, ["full_shard"]
    with pytest.raises(ValueError, match="FSDP"):
        validate(config, args)


def test_native_resume_guards_both_methods_and_fitted_checkpoint(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from transformers import AutoConfig
    script = Path(__file__).resolve().parents[1] / "west/bin/train.py"
    resume = runpy.run_path(str(script))["resume_native_config"]
    checkpoint = tmp_path / "checkpoint-10"
    checkpoint.mkdir()
    tagged = SimpleNamespace(model_type="touch_asu", hidden_size=0, lora_config={"method": "eva"})
    legacy = SimpleNamespace(model_type="touch_asu", lora_config={"r": 2})
    config_file = checkpoint / "config.json"
    config_file.write_text(json.dumps(dict(vars(tagged), hidden_size=32,
                                           resolved_adapter_config={"layer_ranks": {"v": 2}})))
    monkeypatch.setattr(AutoConfig, "from_pretrained",
                        lambda path: SimpleNamespace(**json.loads((path / "config.json").read_text())))
    assert resume(tagged, checkpoint).resolved_adapter_config == {"layer_ranks": {"v": 2}}
    for key, changed in [("adapter_attention_backend", "flash_attention_2"),
                         ("adapter_parameter_precision", "bf16_zero2")]:
        setattr(tagged, key, changed)
        with pytest.raises(ValueError, match=key):
            resume(tagged, checkpoint)
        delattr(tagged, key)
    with pytest.raises(ValueError, match="changed lora_config"):
        resume(legacy, checkpoint)
    with pytest.raises(ValueError, match="different runs"):
        resume(tagged, checkpoint, tmp_path / "other" / "checkpoint-10")
    config_file.write_text(json.dumps(vars(legacy)))
    with pytest.raises(ValueError, match="changed lora_config"):
        resume(tagged, checkpoint)
    assert resume(legacy, checkpoint) is legacy
