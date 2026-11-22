"""Native TouchASU WER selection, generation boundaries, and held-out identities."""
import json

import pytest

from examples.librispeech.asr.local import touchasu_wer as wer


def dump(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.fixture
def comparison(tmp_path, monkeypatch):
    checkpoints = tmp_path / "checkpoints"
    for step in (10, 20):
        ck = checkpoints / f"checkpoint-{step}"
        ck.mkdir(parents=True)
        (ck / "model.safetensors").write_bytes(bytes([step]))
        (ck / "config.json").write_text('{"model_type":"touch_asu"}')
    data = tmp_path / "data"
    data.mkdir()
    splits = {}
    for split, speaker in (("dev-clean", "1"), ("test-clean", "2")):
        rows = [dict(id=f"{speaker}-00-{i}", speaker=speaker, wav=f"{speaker}-{i}.flac",
                     txt=txt) for i, txt in enumerate(("It's GOOD.", "Word"))]
        path = data / f"{split}.jsonl"
        dump(path, rows)
        splits[split] = {"sha256": wer.sha(path), "utterances": len(rows)}
    (data / "splits.json").write_text(json.dumps(splits))
    generation = tmp_path / "generation.json"
    generation.write_text(json.dumps({"do_sample": False, "max_new_tokens": 2, "eos_token_id": [2, 3]}))
    calls = []

    def decode(checkpoint, dataset_path, rows, config, device):
        calls.append((checkpoint.name, dataset_path.stem))
        for i, row in enumerate(rows):
            yield dict(id=row["id"], txt=("IT’S GOOD!" if i == 0 else ""), generated_tokens=2,
                       capped_without_eos=(i == 0))

    monkeypatch.setattr(wer, "decode", decode)
    common = ["--checkpoints", str(checkpoints), "--steps", "20", "10", "--data", str(data),
              "--output", str(tmp_path / "out"), "--generation", str(generation)]
    return tmp_path, calls, common


def test_select_then_test(comparison):
    root, calls, args = comparison
    with pytest.raises(FileNotFoundError):
        wer.main(["test", *args])
    wer.main(["dev", *args])
    assert calls == [("checkpoint-10", "dev-clean"), ("checkpoint-20", "dev-clean")]
    selection = json.loads((root / "out/dev-selection.json").read_text())
    assert selection["selected_step"] == 10
    for _ in range(2):
        wer.main(["test", *args])
    assert calls[-1] == ("checkpoint-10", "test-clean") and len(calls) == 3
    result = json.loads((root / "out/test-clean/metrics.json").read_text())
    assert result["wer"] == pytest.approx(1 / 3)
    assert result["reference_words"] == 3
    assert result["capped_without_eos_rate"] == 0.5
    assert result["checkpoint_sha256"] == selection["score"]["checkpoint_sha256"]


def test_changed_protocol_cannot_reuse_selection(comparison):
    root, _, args = comparison
    wer.main(["dev", *args])
    (root / "generation.json").write_text('{"max_new_tokens":1,"eos_token_id":2,"do_sample":false}')
    with pytest.raises(ValueError):
        wer.main(["test", *args])


def test_partial_or_permuted_hypotheses_cannot_be_scored(comparison, monkeypatch):
    root, _, args = comparison
    monkeypatch.setattr(wer, "decode", lambda *_: iter([{"id": "wrong", "txt": "hi"}]))
    with pytest.raises(ValueError):
        wer.main(["dev", *args])
    assert not (root / "out/dev-selection.json").exists()
    assert not (root / "out/dev-clean/step-10").exists()


def test_generated_token_boundaries():
    # inputs_embeds-only decoder generation returns the new tokens starting at column zero.
    assert wer.generated_hypothesis([4, 3], [2, 3], 2) == ([4], False)
    assert wer.generated_hypothesis([6, 7], [2, 3], 2) == ([6, 7], True)
    assert wer.generated_hypothesis([2], [2, 3], 2) == ([], False)
    with pytest.raises(ValueError):
        wer.generated_hypothesis([1, 3, 5, 7], [2, 3], 2)


@pytest.mark.parametrize("method", ["o-lora", "peft-lora"])
def test_actual_tiny_touchasu_checkpoint_decodes_only_new_tokens(tmp_path, monkeypatch, method):
    import importlib
    import sys

    import torch
    from transformers import AutoModel, GenerationConfig
    from test_touch_asu_adapters import isolated_model

    with isolated_model(monkeypatch, tmp_path) as create:
        if method == "peft-lora":
            from transformers import Qwen3ForCausalLM
            def cpu_sdpa(cls, config, **kwargs):
                config._attn_implementation = "sdpa"
                return config
            monkeypatch.setattr(Qwen3ForCausalLM, "_check_and_enable_flash_attn_2", classmethod(cpu_sdpa))
            host = create(method=None, r=2, lora_alpha=3, target_modules=["q_proj", "v_proj"],
                          task_type="CAUSAL_LM")
        else:
            host = create(method=method)
        # Published Qwen3 defaults to sampling, while the shared ASR profile requires greedy.
        GenerationConfig(do_sample=True, temperature=0.6, top_k=20, top_p=0.95).save_pretrained(
            host.config.llm_model_name_or_path)
        checkpoint = tmp_path / "checkpoint-1"
        host.save_pretrained(checkpoint)
        modes = []
        detect_mode = GenerationConfig.get_generation_mode
        def capture_mode(config, *args, **kwargs):
            mode = detect_mode(config, *args, **kwargs)
            modes.append(mode.value)
            return mode
        monkeypatch.setattr(GenerationConfig, "get_generation_mode", capture_mode)
        kind = type(host)
        monkeypatch.setattr(AutoModel, "from_pretrained", kind.from_pretrained)
        monkeypatch.setattr(sys.modules["wenet"], "load_feature",
                            lambda _: (lambda _: torch.zeros(2, 4), 4), raising=False)
        importlib.import_module("west.models.touch_asu.extractor_touch_asu")

        class MiniTokenizer:
            pad_token_id = 39

            def encode(self, text):
                assert "secret" not in text
                return [1, 4] if text else []

            def decode(self, tokens, **kwargs):
                assert len(tokens) <= 2
                return " ".join(map(str, tokens))

        monkeypatch.setattr(kind, "init_tokenizer", lambda _: MiniTokenizer())
        manifest = tmp_path / "dev-clean.jsonl"
        rows = [dict(id="1-2-3", wav="mock.flac", txt="secret reference")]
        dump(manifest, rows)
        score, = wer.decode(checkpoint, manifest, rows,
                            {"max_new_tokens": 2, "do_sample": False, "eos_token_id": [2, 3]}, "cpu")
        assert score["id"] == "1-2-3" and 1 <= score["generated_tokens"] <= 2
        assert modes == ["greedy_search"]


def test_previously_unknown_generation_mode_cannot_be_used_as_greedy_score(comparison):
    root, _, args = comparison
    wer.main(["dev", *args])
    location = root / "out/dev-clean/step-10/metrics.json"
    old = json.loads(location.read_text())
    assert old.pop("generation_mode") == "greedy_search"
    location.write_text(json.dumps(old))
    saved = root / "out/dev-selection.json"
    selection = json.loads(saved.read_text())
    selection["score"].pop("generation_mode")
    saved.write_text(json.dumps(selection))
    with pytest.raises(ValueError, match="Changed evaluation"):
        wer.main(["dev", *args])
    with pytest.raises(ValueError):
        wer.main(["test", *args])
