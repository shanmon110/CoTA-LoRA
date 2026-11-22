"""Regression tests for scheduling the native WEST LoRA comparison."""
import importlib.util
import json
from pathlib import Path

import pytest


MODULE = "examples.librispeech.asr.local.touchasu_compare"


def comparison():
    assert importlib.util.find_spec(MODULE), "the native TouchASU comparison runner is missing"
    return __import__(MODULE, fromlist=["compare_scores"])


def test_never_handoff_to_an_incomplete_baseline(tmp_path):
    module = comparison()
    assert not module.baseline_completed(tmp_path, [2, 4])
    (tmp_path / "exit.code").write_text("0\n")
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints/trainer_state.json").write_text('{"global_step":3}')
    with pytest.raises(ValueError, match="global_step"):
        module.baseline_completed(tmp_path, [2, 4])
    (tmp_path / "checkpoints/trainer_state.json").write_text('{"global_step":4}')
    with pytest.raises(FileNotFoundError):
        module.baseline_completed(tmp_path, [2, 4])
    for step in (2, 4):
        checkpoint = tmp_path / "checkpoints" / f"checkpoint-{step}"
        checkpoint.mkdir()
        (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": step}))
    assert module.baseline_completed(tmp_path, [2, 4])


def test_failed_predecessor_halts_queue(tmp_path):
    (tmp_path / "exit.code").write_text("1\n")
    with pytest.raises(RuntimeError, match="exit"):
        comparison().baseline_completed(tmp_path, [2, 4])


def test_wer_ranking_requires_identical_protocol():
    metric = {"wer": 0.25, "word_errors": 2, "reference_words": 8, "utterances": 2,
              "generation_sha256": "greedy", "manifest_sha256": "test-clean"}
    better = dict(metric, wer=0.125, word_errors=1)
    pair = comparison().compare_scores(metric, better)
    assert pair == {"lower_test_wer": "ta-lora", "ta_minus_peft_percentage_points": -12.5}
    assert comparison().compare_scores(metric, metric)["lower_test_wer"] == "tie"
    for key in ("generation_sha256", "manifest_sha256", "reference_words", "utterances"):
        with pytest.raises(ValueError):
            comparison().compare_scores(metric, dict(better, **{key: "changed"}))
    with pytest.raises(ValueError):
        comparison().compare_scores(metric, dict(better, wer=0))


def test_plan_only_changes_saving_and_backend_paths(tmp_path):
    module = comparison()
    baseline = tmp_path / "baseline"
    data = tmp_path / "held-out"
    (baseline / "data").mkdir(parents=True)
    data.mkdir()
    train = baseline / "data/train.jsonl"
    train.write_text('{"id":"training-utterance"}\n')
    paths = {"dev-clean": "dev", "test-clean": "test"}
    for split, row in paths.items():
        (data / f"{split}.jsonl").write_text(json.dumps({"id": row}) + "\n")
    splits = {s: {"sha256": module.sha(data / f"{s}.jsonl")} for s in paths}
    splits["train-clean-100"] = {"sha256": module.sha(train)}
    (data / "splits.json").write_text(json.dumps(splits))
    native_model = {"model_type": "touch_asu", "lora_config": {"r": 64},
                    "llm_model_name_or_path": "/base", "wenet_model_name_or_path": "/speech"}
    (baseline / "touch_asu.json").write_text(json.dumps(native_model))
    spec = module.read(module.DEFAULT_CONFIG)
    strategy = module.DEFAULT_CONFIG.parent / spec["training_config"]
    (baseline / "inputs.json").write_text(json.dumps(
        {"manifest_sha256": module.sha(train), "strategy_sha256": module.sha(strategy)}))
    output = tmp_path / "comparison"
    module.initialize(output, baseline, data, module.DEFAULT_CONFIG, None)
    plan = module.read(output / "plan.json")
    assert plan["methods"] == ["ta-lora", "o-lora", "qv-lora", "toplora", "bora", "dora", "eva"]
    assert plan["steps"] == [2500, 5000]
    tagged = module.read(output / "model.json")
    assert tagged == dict(native_model, lora_config=None, adapter_attention_backend="flash_attention_2")
    new, original = module.read(output / "training.json"), module.read(strategy)
    assert new["save_steps"] == 2500
    for field in ("deepspeed", "accelerator_config"):
        assert Path(new[field]).read_bytes() == (module.RECIPE / original[field]).read_bytes()
    assert {k for k in new if new[k] != original[k]} == {"save_steps", "deepspeed", "accelerator_config"}
    module.check_inputs(output, plan)
    train.write_text("changed\n")
    with pytest.raises(ValueError, match="inputs changed"):
        module.check_inputs(output, plan)
