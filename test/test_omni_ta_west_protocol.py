"""Audit early stopping and the dev-to-test checkpoint handoff on a CPU fixture."""
import json
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader

from examples.librispeech.asr.local import protocol


def write_rows(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def events(path, stage, ids):
    write_rows(path, [
        {"stage": stage, "event": "batch", "samples": [{"id": i} for i in ids]},
        {"stage": stage, "event": "stage_end", "status": "ok"},
    ])


def test_protocol_selects_earlier_tied_candidate_and_seals_before_test(tmp_path, monkeypatch):
    preset = protocol.ROOT / "examples/conf/adapters/o-lora.json"
    strategy = tmp_path / "strategy.json"
    strategy.write_text(json.dumps({"eval_steps": 1, "per_device_eval_batch_size": 3}))
    config = tmp_path / "config.json"
    config.write_text(json.dumps(dict(model_type="qwen2_5_omni_adapter", model=str(tmp_path),
                                      plan="plan", test_manifests="test", base_sha256="base")))
    source = tmp_path / "audio.flac"
    source.touch()
    paths, ids = {}, {}
    for split, size in (("train", 32), ("dev", 2), ("test-clean", 2), ("test-other", 2)):
        paths[split] = tmp_path / (split + ".jsonl")
        ids[split] = [f"{split}-{i}" for i in range(size)]
        write_rows(paths[split], [dict(id=i, audio=source.name, speaker=i, text="HELLO") for i in ids[split]])
    hashes = {s: protocol.sha(p) for s, p in paths.items()}
    monkeypatch.setattr(protocol, "prepare_splits", lambda *_: (paths, hashes, ids, {}))
    run = tmp_path / "experiment"
    protocol.init(run, config, "-", adapter=preset, training=strategy)
    state = json.loads((run / "status.json").read_text())
    train = protocol.read_rows(run / "data/train.jsonl")
    assert train[0] == dict(id="train-0", speaker="train-0", txt="HELLO", wav=str(source))
    assert state["methods"] == ["o-lora"] and state["eval_batch_size"] == 3
    assert state["candidates"] == [1, 2, 3, 4]
    saved = run / "adapters/o-lora.json"
    original = saved.read_bytes()
    saved.write_text("{}")
    with pytest.raises(ValueError, match="config changed"):
        protocol.start(run, state, "o-lora")
    saved.write_bytes(original)
    protocol.start(run, state, "o-lora")
    d = run / "o-lora"
    a = d / "adapter"
    a.mkdir()
    write_rows(d / "train.log", [
        dict(trainable_parameters=5, effective_scale=2, resolved_rank=8),
        dict(step=1, loss=.4), dict(step=2, loss=.3),
    ])
    indices = DataLoader(range(32), batch_size=2, shuffle=True, generator=torch.Generator().manual_seed(42))
    order = [ids["train"][i] for batch in indices for i in batch.tolist()]
    events(d / "train.jsonl", "train", order[:16])
    score = dict(wer=.5, word_errors=1, reference_words=2, utterances=2, max_new_tokens=128)
    for i in (1, 2):
        path = a / f"step-{i}"
        path.mkdir()
        (path / "adapter.safetensors").write_bytes(f"weights-{i}".encode())
        (path / "adapter_config.json").write_text(json.dumps(dict(mode="lora", last_n=4, rank=8, alpha=16)))
        events(a / f"dev-{i}.jsonl", "evaluate", ids["dev"])
    history = [dict(validation_step=i, score=score, best_step=1, early_stopped=i == 2) for i in (1, 2)]
    write_rows(a / "dev-history.jsonl", history)
    protocol.finish(run, state, "o-lora")
    with pytest.raises(ValueError, match="sealed"):
        protocol.checkpoint(run, state, "o-lora", "test-clean")
    protocol.seal(run, state)
    # A clean evaluation needs only the selected weights and its held-out split.
    (run / "data/train.jsonl").unlink()
    (run / "data/dev.jsonl").unlink()
    assert protocol.checkpoint(run, state, "o-lora", "test-clean") == a / "step-1"
    first = SimpleNamespace(action="checkpoint", method="o-lora", split="test-clean")
    protocol.save(run / "status.json", state)
    protocol.dispatch(run, first)
    with pytest.raises(ValueError, match="already claimed"):
        protocol.dispatch(run, first)
    for split in protocol.TEST_SPLITS:
        (d / f"{split}.json").write_text(json.dumps(score))
        events(d / f"{split}.jsonl", "evaluate", ids[split])
        for side in ("hyp", "ref"):
            write_rows(d / f"{split}.{side}.jsonl", [dict(id=i, txt="hello") for i in ids[split]])
        wer = d / f"{split}.wer"
        wer.write_text("Overall -> 0.00 % N=2 C=2 S=0 D=0 I=0\n")
        with pytest.raises(ValueError, match="counts disagree"):
            protocol.score_test(run, state, "o-lora", split)
        wer.write_text("Overall -> 50.00 % N=2 C=1 S=0 D=1 I=0\n")
        protocol.score_test(run, state, "o-lora", split)
    assert set(state["results"]["o-lora"]["test"]) == set(protocol.TEST_SPLITS)
    state["state"] = "completed"
    protocol.save(run / "status.json", state)
    with pytest.raises(ValueError, match="already claimed"):
        protocol.dispatch(run, first)
    (a / "step-1/adapter_config.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="adapter changed"):
        protocol.checkpoint(run, state, "o-lora", "test-other")


def test_comparison_loads_shared_presets_and_training_overrides(tmp_path):
    strategy, presets = protocol.read_comparison({}, protocol.COMPARISON, None, None)
    assert list(presets) == ["o-lora", "ta-lora", "qv-lora", "toplora", "bora", "dora", "eva"]
    assert (strategy["per_device_train_batch_size"], strategy["gradient_accumulation_steps"]) == (2, 4)
    qv = protocol.resolve_lora_config(presets["qv-lora"])
    assert (qv["rank"], qv["alpha"], qv["match_ta_budget"]) == (8, 10, True)

    preset = tmp_path / "ta.json"
    preset.write_text(json.dumps({"method": "ta-lora", "last_n": 1, "r": 3, "lora_alpha": 9}))
    override = tmp_path / "strategy.json"
    override.write_text(json.dumps({"dev_patience": 3}))
    strategy, presets = protocol.read_comparison(
        {"training_args": {"seed": 7, "dev_patience": 2}}, protocol.COMPARISON, preset, override)
    assert list(presets) == ["ta-lora"] and presets["ta-lora"]["r"] == 3
    assert (strategy["seed"], strategy["dev_patience"]) == (7, 3)
