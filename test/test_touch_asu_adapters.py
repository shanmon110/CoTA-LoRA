"""Exercise native WEST adapters with real tiny Qwen decoders and a stub speech encoder."""
import importlib
import json
import sys
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import AutoModelForCausalLM, Qwen2Config, Qwen3Config


class SpeechEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 4)
        self.encoder.output_size = lambda: 4
        self.decoder = self.ctc = nn.Identity()

    def _forward_encoder(self, features, lengths):
        mask = torch.arange(features.size(1), device=features.device) < lengths[:, None]
        return self.encoder(features), mask[:, None, :]


@contextmanager
def isolated_model(monkeypatch, tmp_path):
    # WEST registers many optional model backends in its package __init__.
    # Load only the TouchASU implementation; this fixture needs no WeNet weights.
    root = Path(__file__).resolve().parents[1]
    existing_modules = set(sys.modules)
    for name in ("west", "west.models", "west.models.touch_asu"):
        pkg = ModuleType(name)
        pkg.__path__ = [str(root.joinpath(*name.split(".")))]
        monkeypatch.setitem(sys.modules, name, pkg)
    wenet = ModuleType("wenet")
    wenet.load_model = lambda _: SpeechEncoder()
    monkeypatch.setitem(sys.modules, "wenet", wenet)
    for key in list(sys.modules):
        if key.startswith("west.models.touch_asu."):
            monkeypatch.delitem(sys.modules, key)
    mod = importlib.import_module("west.models.touch_asu.modeling_touch_asu")

    def create(config_cls=Qwen3Config, method="ta-lora", dtype=torch.float32, **lora):
        path = tmp_path / config_cls.model_type
        if not path.exists():
            config = config_cls(hidden_size=16, num_attention_heads=4, num_key_value_heads=2,
                                head_dim=4, intermediate_size=32, vocab_size=40, num_hidden_layers=2,
                                bos_token_id=1, pad_token_id=39, eos_token_id=2, tie_word_embeddings=True)
            AutoModelForCausalLM.from_config(config).to(dtype).save_pretrained(path)
        spec = dict(r=1, lora_alpha=2, last_n=2, method=method, **lora) if method else lora
        host_config = mod.TouchASUConfig(
            llm_model_name_or_path=str(path), encoder_projector_ds_rate=2, projector_hidden_size=12,
            lora_config=spec, bos_token_id=1, pad_token_id=39, eos_token_id=2, torch_dtype=dtype)
        return mod.TouchASU._from_config(host_config)

    try:
        yield create
    finally:
        for key in tuple(sys.modules):
            if key.startswith("west.models.touch_asu.") or (key.startswith("west.")
                                                            and key not in existing_modules):
                sys.modules.pop(key)


@pytest.fixture
def native_model(monkeypatch, tmp_path):
    with isolated_model(monkeypatch, tmp_path) as create:
        yield create


def speech_batch():
    ids = torch.tensor([[1, 0, 0, 0, 5, 6, 7, 2], [1, 3, 0, 0, 4, 2, 39, 39]])
    return dict(input_ids=ids, labels=ids.masked_fill(ids.eq(39), -100),
                attention_mask=ids.ne(39), audio_offsets=torch.tensor([1, 2]),
                audio_features=torch.randn(2, 5, 4), audio_features_lengths=torch.tensor([5, 4]),
                batch_idx=torch.tensor([0, 1]), has_audio=torch.tensor([1, 1]))


class TrainingData:
    def __init__(self, batch):
        self.data_lists = [0, 1]
        self.inference = False
        self.extractor = SimpleNamespace(inference=False, extract=lambda raw: raw)
        self.batch = batch

    def _read_one(self):
        assert self.inference
        yield from self.data_lists

    def _batch(self, rows):
        result = {k: v[rows] for k, v in self.batch.items()}
        result["batch_idx"] = torch.zeros(len(rows), dtype=torch.long)
        return result


@pytest.mark.parametrize("method", ["o-lora", "ta-lora", "qv-lora", "toplora", "bora", "dora", "eva"])
def test_native_step_reload_generation(native_model, method, tmp_path):
    torch.manual_seed(29)
    options = {"eva_samples": 5} if method == "eva" else {}
    model = native_model(method=method, **options)
    batch = speech_batch()
    if method == "eva":
        with pytest.raises(RuntimeError, match="training calibration"):
            model(**batch)
        model.calibrate_adapter(TrainingData(batch))
        assert model.config.resolved_adapter_config["eva_samples"] == 2
        ranks = model.config.resolved_adapter_config["layer_ranks"]
        assert sum(ranks.values()) == 2
    llm_params = [p for n, p in model.llm.named_parameters() if p.requires_grad]
    assert llm_params and not any(p.requires_grad for p in model.encoder.parameters())
    assert all(p.requires_grad for p in model.projector.parameters())
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
    model.train()
    model(**batch, use_cache=False).loss.backward()
    assert any(p.grad is not None and p.grad.abs().max() > 0 for p in llm_params)
    opt.step()
    model.eval()
    with torch.no_grad():
        reference = model(**batch, use_cache=False).logits
        generated = model.generate(**batch, max_new_tokens=2)
    path = tmp_path / "checkpoint-1"
    model.save_pretrained(path)
    loaded = type(model).from_pretrained(path)
    assert not loaded.needs_adapter_calibration
    with torch.no_grad():
        torch.testing.assert_close(loaded(**batch, use_cache=False).logits, reference)
        assert torch.equal(loaded.generate(**batch, max_new_tokens=2), generated)


@pytest.mark.parametrize("config_cls", [Qwen2Config, Qwen3Config])
def test_gate_zero_actual_spans_and_checkpoint_replay(native_model, config_cls):
    torch.manual_seed(18)
    batch = speech_batch()
    ta = native_model(config_cls=config_cls)
    ordinary = native_model(config_cls=config_cls, method="o-lora")
    for layer in ta.llm.model.layers:
        layer.self_attn.ta_adapter.up.data.normal_(std=.1)
    ordinary.load_state_dict({k: v for k, v in ta.state_dict().items()
                              if not k.endswith("ta_adapter.gate")}, strict=True)
    _, _, context = ta._decoder_inputs(batch["input_ids"], batch["attention_mask"], None,
                                       batch["audio_offsets"], batch["audio_features"],
                                       batch["audio_features_lengths"], batch["batch_idx"],
                                       batch["has_audio"])
    from west_adapters.context import current_audio_layout
    with context:
        mask, valid = current_audio_layout()
        # Third zero in row 0 is an unfilled placeholder, not an acoustic embedding.
        assert mask.nonzero().tolist() == [[0, 1], [0, 2], [1, 2], [1, 3]]
        assert torch.equal(valid, batch["attention_mask"])
    assert current_audio_layout() is None
    from transformers.cache_utils import DynamicCache
    cache = DynamicCache()
    with pytest.raises(ValueError, match="projected audio layout"):
        ta.llm(input_ids=batch["input_ids"], past_key_values=cache, use_cache=True)
    assert cache.get_seq_length() == 0
    reference = ordinary(**batch, use_cache=False).logits
    result = ta(**batch, use_cache=False).logits
    torch.testing.assert_close(result, reference, atol=0, rtol=0)
    ta.gradient_checkpointing_enable()
    ta.train()
    ta(**batch, use_cache=False).loss.backward()
    for layer in ta.llm.model.layers:
        assert layer.self_attn.ta_adapter.gate.grad.abs().max() > 0


@pytest.mark.parametrize("method", ["ta-lora", "qv-lora"])
def test_packed_utterance_cannot_read_earlier_audio(native_model, method):
    model = native_model(method=method).eval()
    batch = speech_batch()
    batch["input_ids"] = torch.tensor([[1, 0, 0, 3, 2, 1, 0, 0, 4, 2]])
    batch["position_ids"] = torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4]])
    batch["batch_idx"] = torch.tensor([0, 0])
    batch["audio_offsets"] = torch.tensor([1, 6])
    del batch["attention_mask"], batch["labels"]
    if method == "ta-lora":
        for layer in model.llm.model.layers:
            layer.self_attn.ta_adapter.gate.data.fill_(1)
            layer.self_attn.ta_adapter.up.data.normal_()
    with torch.no_grad():
        initial = model(**batch, use_cache=False).logits
        batch["audio_features"][0] += 11
        later = model(**batch, use_cache=False).logits
    assert not torch.equal(later[:, :5], initial[:, :5])
    torch.testing.assert_close(later[:, 5:], initial[:, 5:], atol=0, rtol=0)


def test_untagged_config_uses_peft(native_model, monkeypatch):
    peft = pytest.importorskip("peft")
    original = AutoModelForCausalLM.from_pretrained

    def cpu_load(*args, **kwargs):
        assert kwargs["attn_implementation"] == "flash_attention_2"
        kwargs["attn_implementation"] = "eager"
        return original(*args, **kwargs)

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", cpu_load)
    preset = Path(__file__).resolve().parents[1] / "examples/aishell/asr/conf/qwen3-1.7b-lora_firered.json"
    options = json.loads(preset.read_text())["lora_config"]
    model = native_model(method=None, **options)
    assert isinstance(model.llm, peft.PeftModel)
    cfg = model.llm.peft_config["default"]
    assert (cfg.r, cfg.lora_alpha, cfg.lora_dropout) == (64, 16, .05)
    assert not cfg.inference_mode and cfg.task_type == "CAUSAL_LM"
    assert cfg.target_modules == {"q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"}
    for layer in model.llm.base_model.model.model.layers:
        installed = {name.rsplit(".", 1)[-1] for name, mod in layer.named_modules()
                     if hasattr(mod, "lora_A")}
        assert installed == cfg.target_modules
    model(**speech_batch(), use_cache=False).loss.backward()
    assert all(p.grad is not None for p in model.llm.parameters() if p.requires_grad)


@pytest.mark.parametrize("method", ["ta-lora", "dora"])
def test_bf16_full_reload_preserves_float_adapters(native_model, method, tmp_path):
    model = native_model(method=method, dtype=torch.bfloat16).eval()
    assert model.llm.model.embed_tokens.weight.dtype == torch.bfloat16
    with torch.no_grad():
        for name, p in model.llm.named_parameters():
            if name.endswith("adapter.up"):
                p.normal_(std=.1)
    batch = speech_batch()
    path = tmp_path / "bf16"
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        logits = model(**batch, use_cache=False).logits
        model.save_pretrained(path)
        restored = type(model).from_pretrained(path)
        assert all(p.dtype == torch.float32 for p in restored.llm.parameters() if p.requires_grad)
        torch.testing.assert_close(restored(**batch, use_cache=False).logits, logits, atol=0, rtol=0)


@pytest.mark.parametrize("method", ["o-lora", "ta-lora", "qv-lora", "toplora", "bora", "dora", "eva"])
def test_zero2_bf16_storage_and_packed_gradients(native_model, method, tmp_path):
    # DeepSpeedEngine._configure_distributed_model calls module.bfloat16()
    # before ZeRO partitions its FP32 optimizer weights.
    torch.manual_seed(28)
    model = native_model(method=method, dtype=torch.bfloat16)
    batch = speech_batch()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        if method == "eva":
            model.calibrate_adapter(TrainingData(batch))
    model.bfloat16()
    trainable = {n: p for n, p in model.llm.named_parameters() if p.requires_grad}
    with torch.no_grad():
        for name, p in trainable.items():
            if name.endswith((".up", ".gate")):
                p.normal_(std=.2)
    # Two native utterances in one pack, with independent position resets.
    batch.update(input_ids=torch.tensor([[1, 0, 0, 3, 2, 1, 0, 0, 4, 2]]),
                 position_ids=torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4]]),
                 audio_offsets=torch.tensor([1, 6]), batch_idx=torch.tensor([0, 0]))
    del batch["attention_mask"]
    batch["labels"] = batch["input_ids"].masked_fill(batch["position_ids"] == 0, -100)
    model.train()
    model.gradient_checkpointing_enable()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        model(**batch, use_cache=False).loss.backward()
    assert all(p.dtype == torch.bfloat16 and p.grad is not None and torch.isfinite(p.grad).all()
               for p in trainable.values())
    assert all(p.grad is not None for p in model.projector.parameters())
    assert all(p.grad is None for p in model.encoder.parameters())
    model.eval()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        expected = model(**batch, use_cache=False).logits
        model.save_pretrained(tmp_path / "zero2")
        loaded = type(model).from_pretrained(tmp_path / "zero2")
        assert not loaded.needs_adapter_calibration
        torch.testing.assert_close(loaded(**batch, use_cache=False).logits, expected, atol=0, rtol=0)


def test_native_training_entry_and_clean_decode(native_model, tmp_path, monkeypatch):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import AutoModel, PreTrainedTokenizerFast
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    # The JSONL path does not use the optional tar shard library.
    monkeypatch.setitem(sys.modules, "webdataset", ModuleType("webdataset"))
    from west.bin import decode, train
    from west.dataset.extractor import Extractor

    monkeypatch.setattr(Extractor, "_registry", dict(Extractor._registry))
    importlib.import_module("west.models.touch_asu.extractor_touch_asu")
    base = native_model(method="eva", eva_samples=2)
    cls, cfg = type(base), type(base.config)
    monkeypatch.setitem(CONFIG_MAPPING._extra_content, "touch_asu", cfg)
    monkeypatch.setitem(AutoModel._model_mapping._extra_content, cfg, cls)
    monkeypatch.setattr(sys.modules["wenet"], "load_feature",
                        lambda _: (lambda path: torch.arange(128).reshape(32, 4).float() / 128, 4),
                        raising=False)
    # A local tokenizer exercises the real raw JSONL reader, TouchASU
    # extractor, SpeechDataset, Trainer, AutoModel and decode entry point.
    vocab = {"[UNK]": 0, "[BOS]": 1, "[EOS]": 2, "[PAD]": 39}
    vocab.update({f"w{i}": i for i in range(3, 39)})
    vocab.update({"the": 3, "Speech": 4, "user": 5, "assistant": 6})
    for i in range(3, 7):
        del vocab[f"w{i}"]
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, bos_token="[BOS]", eos_token="[EOS]",
                            unk_token="[UNK]", pad_token="[PAD]").save_pretrained(base.config.llm_model_name_or_path)
    config = base.config.to_dict()
    config.pop("hidden_size")  # Standard TouchASU recipes can derive this from the LLM.
    config.update(encoder_ds_rate=1, min_speech_frames=1, max_new_tokens=2)
    spec = tmp_path / "model.json"
    adapter = tmp_path / "adapter.json"
    strategy = tmp_path / "strategy.json"
    output = tmp_path / "output"
    train_file = tmp_path / "train.jsonl"
    test_file = tmp_path / "heldout.jsonl"
    record = json.dumps({"wav": "stub.wav", "txt": "w8 w9 w10 w11"})
    train_file.write_text((record + "\n") * 2)
    test_file.write_text(record + "\n")
    adapter.write_text(json.dumps(config.pop("lora_config")))
    strategy.write_text(json.dumps(dict(max_steps=1, save_steps=1, logging_steps=1, use_cpu=True, pack_size=96,
                                        logging_strategy="steps", report_to="none", optim="adamw_torch",
                                        lr_scheduler_type="constant", learning_rate=0.001,
                                        gradient_checkpointing=True, per_device_train_batch_size=1)))
    spec.write_text(json.dumps(config))
    monkeypatch.setattr(sys, "argv", ["train.py", "--model_config_or_dir", str(spec),
                                     "--adapter_config", str(adapter), "--training_config", str(strategy),
                                     "--data_path", str(train_file), "--output_dir", str(output)])
    train.main()
    checkpoint = output / "checkpoint-1"
    persisted = json.loads((checkpoint / "config.json").read_text())
    assert persisted["resolved_adapter_config"]["eva_samples"] == 2
    assert sum(persisted["resolved_adapter_config"]["layer_ranks"].values()) == 2
    monkeypatch.setattr(cls, "calibrate_adapter", lambda *args: pytest.fail("repeated calibration"))
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--max_steps", "2"])
    train.main()
    resumed = output / "checkpoint-2"
    assert json.loads((resumed / "config.json").read_text())["resolved_adapter_config"] == persisted["resolved_adapter_config"]
    from safetensors.torch import load_file
    first = load_file(checkpoint / "model.safetensors")
    second = load_file(resumed / "model.safetensors")
    assert any(not torch.equal(first[k], second[k]) for k in first if ".baseline_adapter." in k)
    train_file.unlink()
    result = tmp_path / "result.jsonl"
    monkeypatch.setattr(sys, "argv", ["decode.py", "--model_config_or_dir", str(resumed),
                                     "--data_path", str(test_file), "--result_path", str(result)])
    decode.main()
    assert isinstance(json.loads(result.read_text())["txt"], str)


def distributed_fit(rank, tmp_path):
    from torch import distributed as dist

    dist.init_process_group("gloo", init_method=f"file://{tmp_path}/init", rank=rank,
                            world_size=2, timeout=timedelta(seconds=45))
    try:
        with pytest.MonkeyPatch.context() as patch, isolated_model(patch, tmp_path) as create:
            model = create(method="eva", eva_samples=2)
            data = TrainingData(speech_batch())
            if rank == 1:
                data._read_one = lambda: pytest.fail("worker re-read calibration data")
            model.calibrate_adapter(data)
            rows = [p.detach() for p in model.llm.parameters() if p.requires_grad]
            assert not model.needs_adapter_calibration
            torch.save((model.config.resolved_adapter_config, rows), tmp_path / f"rank{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_distributed_eva_broadcasts_fitted_layout(native_model, tmp_path):
    from torch import distributed as dist
    from torch.multiprocessing import spawn

    if not dist.is_gloo_available():
        pytest.skip("requires CPU distributed Gloo")
    native_model(method="eva", eva_samples=2)
    spawn(distributed_fit, args=(tmp_path,), nprocs=2)
    left, right = [torch.load(tmp_path / f"rank{i}.pt", weights_only=True) for i in (0, 1)]
    assert left[0] == right[0]
    for expected, observed in zip(left[1], right[1], strict=True):
        torch.testing.assert_close(observed, expected, atol=0, rtol=0)


@pytest.mark.parametrize("beams", [1, 2])
def test_generation_keeps_utterance_layout_when_batching(native_model, beams):
    torch.manual_seed(172)
    model = native_model().eval()
    batch = speech_batch()
    for layer in model.llm.model.layers:
        layer.self_attn.ta_adapter.gate.data.fill_(1)
        layer.self_attn.ta_adapter.up.data.normal_()
    opts = dict(num_beams=beams, min_new_tokens=3, max_new_tokens=3, do_sample=False)
    with torch.no_grad():
        actual = model.generate(**batch, **opts)
        for i, length in enumerate((8, 6)):
            row = TrainingData(batch)._batch([i])
            for key in ("input_ids", "attention_mask", "labels"):
                row[key] = row[key][:, :length]
            expected = model.generate(**row, **opts)
            assert torch.equal(expected[0], actual[i])
