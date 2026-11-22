"""Baseline SFT and held-out inference on a tiny native Thinker; no Hub weights."""

import copy
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from transformers import Qwen2_5OmniThinkerForConditionalGeneration

from examples.omni_talora.attention import audio_context
from examples.omni_talora.checkpoint import load_adapter, read_config, save_adapter
from examples.omni_talora.train import train_updates
if __package__:
    from .test_omni_ta_integration import tiny_config, write_tiny_checkpoint
else:  # pytest's default import mode adds test/ itself to sys.path
    from test_omni_ta_integration import tiny_config, write_tiny_checkpoint


MODES = ("toplora", "bora", "dora", "eva", "lora")


def tiny_audio_batch():
    ids = torch.tensor([[1, 4, 5, 5, 6, 8, 13, 2], [1, 4, 5, 5, 6, 9, 11, 2]])
    return dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                input_features=torch.randn(2, 8, 8),
                feature_attention_mask=torch.ones(2, 8, dtype=torch.long),
                labels=torch.tensor([[-100] * 6 + [13, 2], [-100] * 6 + [11, 2]]))


@pytest.mark.parametrize("mode", MODES)
def test_baseline_update_freezes_native_weights_and_reloads_logits(mode, tmp_path):
    from examples.omni_talora.adapters import install_from_config

    torch.manual_seed(1234)
    model = Qwen2_5OmniThinkerForConditionalGeneration(tiny_config())
    fresh = copy.deepcopy(model)
    native = [(p, p.detach().clone()) for p in model.parameters()]
    batch = tiny_audio_batch()
    inputs = {k: v for k, v in batch.items() if k != "labels"}

    def logits(m):
        m.eval()
        with torch.no_grad(), audio_context(inputs["input_ids"], inputs["attention_mask"], 5):
            return m(**inputs, use_cache=False).logits

    original_logits = logits(fresh)
    target_set = "qv" if mode == "lora" else "o"
    ranks = {f"model.layers.{layer}.self_attn.{kind}_proj": 2
             for layer in (0, 1) for kind in target_set}
    config = dict(model_name_or_path="tiny", revision=None, task="asr", rank=2, alpha=4.,
                  last_n=2, mode=mode, target_set=target_set, layer_ranks=ranks)
    if mode == "bora":
        config["bora_blocks"] = 2
    directions = None
    if mode == "eva":
        from examples.omni_talora.eva import calibrate_eva
        config.update(rank=1, eva_samples=2, eva_seed=17, eva_rho=2.)
        directions = calibrate_eva(model, [batch], last_n=2, rank=1, rho=2., device="cpu")
        config["layer_ranks"] = {name: v.shape[0] for name, v in directions.items()}
    install_from_config(model, config, directions=directions)
    torch.testing.assert_close(logits(model), original_logits, atol=2.e-6, rtol=1.e-5)
    params = [p for p in model.parameters() if p.requires_grad]
    assert params and all(not p.requires_grad for p, _ in native)
    opt = torch.optim.AdamW(params, lr=.025, weight_decay=0.)
    train_updates(model, [batch], opt, device="cpu", audio_token_id=5,
                  max_steps=2, accumulation_steps=1)
    assert all(torch.equal(p, pristine) for p, pristine in native)
    trained_logits = logits(model)
    assert not torch.allclose(trained_logits, original_logits, atol=1.e-5, rtol=1.e-5)
    save_adapter(model, tmp_path, config)
    saved = read_config(tmp_path)
    assert saved == config
    install_from_config(fresh, saved)
    load_adapter(fresh, tmp_path)
    torch.testing.assert_close(logits(fresh), trained_logits, atol=1.e-6, rtol=1.e-6)


@pytest.mark.parametrize("mode", ["ta", "lora"])
def test_legacy_o_archives_reconstruct_without_new_config_keys(mode, tmp_path):
    """Construct the original wire format independently of today's writer."""
    from safetensors.torch import save_file
    from examples.omni_talora.adapters import install_from_config

    torch.manual_seed(543)
    model = Qwen2_5OmniThinkerForConditionalGeneration(tiny_config())
    config = dict(model_name_or_path="tiny", revision=None, task="asr", rank=2,
                  alpha=4., last_n=1, mode=mode)
    digest = hashlib.sha256(json.dumps(config, separators=(",", ":"), sort_keys=True).encode())
    prefix = "model.layers.1.self_attn.ta_adapter."
    state = {prefix + "down": torch.randn(2, 16), prefix + "up": torch.randn(16, 2)}
    if mode == "ta":
        state[prefix + "gate"] = torch.randn(2)
    save_file(state, tmp_path / "adapter.safetensors",
              metadata={"format": "west-omni-talora-v1", "config_sha256": digest.hexdigest()})
    (tmp_path / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    install_from_config(model, read_config(tmp_path))
    assert load_adapter(model, tmp_path) == config
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    assert trainable.keys() == state.keys()
    for key in state:
        torch.testing.assert_close(trainable[key], state[key], atol=0, rtol=0)


@pytest.fixture(scope="module")
def tiny_processor_model(tmp_path_factory):
    src = os.getenv("OMNI_PROCESSOR_DIR")
    if not src:
        pytest.skip("requires local official processor files (no pretrained weights)")
    directory = tmp_path_factory.mktemp("tiny-baseline-model")
    write_tiny_checkpoint(directory, Path(src))
    return directory


@pytest.mark.parametrize("task", ["asr", "ser"])
@pytest.mark.parametrize("mode", MODES)
def test_baseline_clis_train_and_evaluate_without_calibrating_on_test(
        mode, task, tiny_processor_model, tmp_path, monkeypatch, capsys):
    from examples.omni_talora import runtime
    from examples.omni_talora.evaluate import main as evaluate
    from examples.omni_talora.train import main as train

    labels = [{"text": "hello world"}, {"text": "good bye"}] if task == "asr" else [
        {"emotion": "sad"}, {"emotion": "angry"}]
    rows = []
    for i in range(3):
        audio = f"clip-{i}.wav"
        sf.write(tmp_path / audio, np.sin(np.arange(2400) * (.03 + .01 * i)), 16000)
        rows.append(json.dumps({"audio": audio, **labels[i % 2]}))
    train_manifest = tmp_path / "train.jsonl"
    train_manifest.write_text("\n".join(rows[:2]) + "\n", encoding="utf-8")
    held_out = tmp_path / "held-out.jsonl"
    held_out.write_text(rows[2] + "\n", encoding="utf-8")
    output = tmp_path / "adapter"
    extra = {"bora": ["--bora-blocks", "2"],
             "eva": ["--eva-samples", "2", "--eva-rho", "2", "--eva-seed", "17"],
             "lora": ["--target-set", "qv", "--match-ta-budget"]}.get(mode, [])
    nominal_rank = {"lora": "4", "eva": "1"}.get(mode, "2")
    train(["--model-name-or-path", str(tiny_processor_model), "--device", "cpu",
           "--task", task, "--mode", mode, "--manifest", str(train_manifest),
           "--output", str(output), "--last-n", "2", "--rank", nominal_rank,
           "--max-steps", "1", "--accumulation-steps", "1", *extra])
    reports = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    summary, = (report for report in reports if "trainable_parameters" in report)
    # Two 16×16 O projections. In addition to BA, Top learns a
    # projector/RMS weight, BoRA 2×2 block factors, DoRA channel magnitudes.
    assert summary["trainable_parameters"] == {
        "toplora": 196, "bora": 144, "dora": 160, "eva": 64, "lora": 224}[mode]
    config = read_config(output)
    target_set = "qv" if mode == "lora" else "o"
    assert (config["task"], config["mode"], config["target_set"]) == (task, mode, target_set)
    expected = {f"model.layers.{i}.self_attn.{kind}_proj" for i in (0, 1) for kind in target_set}
    assert config["layer_ranks"].keys() == expected
    if mode == "lora":
        # TA O costs (16+16+1)*4=132 per layer; Q/V costs (32+24)*r.
        assert set(config["layer_ranks"].values()) == {2}
        budget, = (report for report in reports if "ta_parameters" in report)
        assert (budget["ta_parameters"], budget["qv_parameters"], budget["difference"]) == (264, 224, -40)
    if mode == "bora":
        assert config["bora_blocks"] == 2
    if mode == "eva":
        assert (config["eva_samples"], config["eva_seed"], config["eva_rho"]) == (2, 17, 2.)
        assert sum(config["layer_ranks"].values()) == 2

    # A saved EVA adapter must not depend on train files or fit a new basis on
    # eval examples. Each task needs exactly one native Thinker forward here
    # (one example; ASR greedily generates just one token).
    train_manifest.unlink()
    for i in (0, 1):
        (tmp_path / f"clip-{i}.wav").unlink()
    forward_calls = []
    loader = runtime.load_base

    def monitored_load(*args, **kwargs):
        model, processor = loader(*args, **kwargs)
        model.register_forward_pre_hook(lambda *_: forward_calls.append(None))
        return model, processor

    monkeypatch.setattr(runtime, "load_base", monitored_load)
    monkeypatch.setattr(sys, "argv", ["evaluate", "--device", "cpu", "--adapter", str(output),
                                      "--manifest", str(held_out), "--max-new-tokens", "1"])
    evaluate()
    metrics = json.loads(capsys.readouterr().out)
    assert (metrics["task"], metrics["utterances"]) == (task, 1)
    assert len(forward_calls) == 1
    if task == "asr":
        assert metrics["reference_words"] == 2
    else:
        assert metrics["per_class"]["C"]["support"] == 1
