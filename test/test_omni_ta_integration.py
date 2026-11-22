"""Exercise a complete native Thinker (including its audio tower), without model downloads."""

import copy
import json
import os
import sys
from pathlib import Path

import torch
import pytest
from safetensors.torch import save_file
from transformers import Qwen2_5OmniConfig
from transformers import Qwen2_5OmniThinkerConfig, Qwen2_5OmniThinkerForConditionalGeneration

from examples.omni_talora.attention import audio_context, install_adapter
from examples.omni_talora.checkpoint import load_adapter, save_adapter
from examples.omni_talora.train import train_updates
from examples.omni_talora.runtime import load_base


def tiny_config():
    return Qwen2_5OmniThinkerConfig(
        audio_config=dict(num_mel_bins=8, d_model=16, output_dim=16, encoder_layers=1,
                          encoder_attention_heads=2, encoder_ffn_dim=32, max_source_positions=16, n_window=8),
        vision_config=dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2,
                           out_hidden_size=16, fullatt_block_indexes=[0]),
        text_config=dict(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=128,
                         rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]}),
        audio_token_index=5, audio_start_token_id=4, audio_end_token_id=6,
        vision_start_token_id=7, pad_token_id=0, eos_token_id=2, bos_token_id=1,
        attn_implementation="eager")


def test_real_audio_sft_two_updates_then_reload_cached_decode(tmp_path):
    torch.manual_seed(1234)
    model = Qwen2_5OmniThinkerForConditionalGeneration(tiny_config())
    restored = copy.deepcopy(model)
    args = dict(rank=2, alpha=4., last_n=2, mode="ta")
    install_adapter(model, **args)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    ids = torch.tensor([[1, 4, 5, 5, 6, 8, 13, 2]])
    features = dict(input_features=torch.randn(1, 8, 8),
                    feature_attention_mask=torch.ones(1, 8, dtype=torch.long))
    train_batch = dict(**features, input_ids=ids, attention_mask=torch.ones_like(ids),
                       labels=torch.tensor([[-100] * 6 + [13, 2]]))
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=.05)
    train_updates(model, [train_batch], opt, device="cpu", audio_token_id=5, max_steps=2,
                  accumulation_steps=1)
    assert torch.count_nonzero(model.model.layers[-1].self_attn.ta_adapter.gate)
    assert all(torch.equal(p, dict(model.named_parameters())[n]) for n, p in restored.named_parameters())
    config = dict(**args, revision=None, model_name_or_path="local-tiny", task="asr")
    save_adapter(model, tmp_path, config)
    install_adapter(restored, **args)
    assert load_adapter(restored, tmp_path) == config
    prompt = ids[:, :6]
    mask = torch.ones_like(prompt)
    with torch.inference_mode(), audio_context(prompt, mask, 5):
        def scores(m, cached):
            m.eval()
            output = m.generate(input_ids=prompt, attention_mask=mask, **features,
                                do_sample=False, num_beams=1, use_cache=cached,
                                min_new_tokens=3, max_new_tokens=3, pad_token_id=0, eos_token_id=2,
                                suppress_tokens=[4, 5, 6], return_dict_in_generate=True, output_scores=True)
            return output.sequences, torch.stack(output.scores)
        seq, logits = scores(model, True)
        assert seq.size(1) == prompt.size(1) + 3
        for other, cached in [(restored, True), (model, False)]:
            seq2, logits2 = scores(other, cached)
            torch.testing.assert_close(seq, seq2)
            torch.testing.assert_close(logits, logits2, rtol=1.e-4, atol=2.e-5)


def write_tiny_checkpoint(tmp_path, src):
    """Publish a random local model beside the official processor-only files."""
    # Official upstream distributes one root config and weights prefixed with
    # thinker.*, talker.*, token2wav.*; native Thinker.save_pretrained alone
    # would not exercise the prefix stripping or nested config selection.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(src)
    cfg = tiny_config()
    cfg.audio_token_id = tokenizer.convert_tokens_to_ids(tokenizer.audio_token)
    cfg.text_config.vocab_size = len(tokenizer)
    cfg.audio_config.num_mel_bins = 128
    pristine = Qwen2_5OmniThinkerForConditionalGeneration(cfg)
    Qwen2_5OmniConfig(thinker_config=cfg.to_dict()).save_pretrained(tmp_path)
    for item in src.iterdir():
        if item.is_file() and item.name != "config.json" and item.suffix in (".json", ".txt", ".jinja"):
            (tmp_path / item.name).symlink_to(item.resolve())
    state = {"thinker." + k: v for k, v in pristine.state_dict().items()}
    state["talker.test_only"] = state["token2wav.test_only"] = torch.ones(1)
    save_file({k: v.clone().contiguous() for k, v in state.items()}, str(tmp_path / "model.safetensors"),
              metadata={"format": "pt"})
    return state


@pytest.mark.skipif(not os.getenv("OMNI_PROCESSOR_DIR"), reason="requires local official processor (no weights)")
def test_official_processor_full_checkpoint_layout_and_separate_task_clis(tmp_path, monkeypatch, capsys):
    import numpy as np
    import soundfile
    from examples.omni_talora.train import main as train
    from examples.omni_talora.evaluate import main as evaluate

    state = write_tiny_checkpoint(tmp_path, Path(os.environ["OMNI_PROCESSOR_DIR"]))
    model, processor = load_base(str(tmp_path), None, "cpu")
    assert len(model.model.layers) == 2
    assert processor.tokenizer.padding_side == "left"
    for key, weight in model.state_dict().items():
        torch.testing.assert_close(weight, state["thinker." + key], rtol=0, atol=0)
    soundfile.write(tmp_path / "clip.wav", np.sin(np.arange(2400) * .04), 16000)
    for task, label in [("asr", {"text": "hello"}), ("ser", {"emotion": "sad"})]:
        manifest = tmp_path / (task + ".jsonl")
        manifest.write_text(json.dumps({"audio": "clip.wav", **label}) + "\n")
        dest = tmp_path / ("run-" + task)
        train(["--model-name-or-path", str(tmp_path), "--device", "cpu", "--task", task,
               "--manifest", str(manifest), "--output", str(dest), "--last-n", "1", "--rank", "2",
               "--max-steps", "2", "--accumulation-steps", "1", "--gradient-checkpointing"])
        capsys.readouterr()
        monkeypatch.setattr(sys, "argv", ["evaluate", "--device", "cpu", "--adapter", str(dest),
                                          "--manifest", str(manifest), "--max-new-tokens", "2"])
        evaluate()
        metrics = json.loads(capsys.readouterr().out)
        assert metrics["task"] == task
        assert metrics["utterances"] == 1
        if task == "asr":
            assert metrics["reference_words"] == 1
        else:
            assert metrics["per_class"]["C"]["support"] == 1


@pytest.mark.skipif(not os.getenv("OMNI_PROCESSOR_DIR"), reason="requires local official processor")
def test_online_dev_early_stop_restores_the_best(tmp_path, monkeypatch):
    import numpy as np
    import soundfile
    from safetensors.torch import load_file
    from examples.omni_talora import evaluate
    from examples.omni_talora.train import main as train

    write_tiny_checkpoint(tmp_path, Path(os.environ["OMNI_PROCESSOR_DIR"]))
    for split in ("train", "dev"):
        soundfile.write(tmp_path / f"{split}.wav", np.sin(np.arange(2400) * .04), 16000)
        (tmp_path / f"{split}.jsonl").write_text(json.dumps({"id": split, "audio": f"{split}.wav", "text": "hello"}) + "\n")
    run_eval = evaluate.evaluate
    errors = iter([3, 2, 2])

    def score(*args, **kwargs):
        # Run genuine validation and cached generation before injecting a known ranking.
        assert args[0].training
        assert [r["id"] for r in args[2]] == ["dev"]
        kwargs["max_new_tokens"] = 2
        result = run_eval(*args, **kwargs)
        result["word_errors"] = next(errors)
        result["wer"] = result["word_errors"] / result["reference_words"]
        return result

    monkeypatch.setattr(evaluate, "evaluate", score)
    out = tmp_path / "early"
    train(["--model-name-or-path", str(tmp_path), "--task", "asr", "--device", "cpu",
           "--output", str(out), "--manifest", str(tmp_path / "train.jsonl"),
           "--dev-manifest", str(tmp_path / "dev.jsonl"), "--dev-interval", "1",
           "--dev-patience", "1", "--rank", "2", "--last-n", "1", "--gradient-checkpointing",
           "--max-steps", "5", "--accumulation-steps", "1"])
    decisions = [json.loads(line) for line in (out / "dev-history.jsonl").read_text().splitlines()]
    assert [d["best_step"] for d in decisions] == [1, 2, 2]
    assert [d["early_stopped"] for d in decisions] == [False, False, True]
    assert not (out / "step-4").exists()
    assert all(torch.equal(v, load_file(out / "step-2/adapter.safetensors")[k])
               for k, v in load_file(out / "adapter.safetensors").items())
