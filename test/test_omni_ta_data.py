"""Real HF processor/Whisper, tiny offline tokenizer by default.

Set OMNI_PROCESSOR_DIR to a local Qwen2.5-Omni processor snapshot to exercise its
published tokenizer and chat template instead. No weights, network or GPU needed.
"""
import json
import os

import numpy as np
import pytest
import soundfile as sf
import torch

from examples.omni_talora.data import OmniCollator, answer_for, load_manifest


@pytest.fixture(scope="module")
def processor():
    from transformers import (Qwen2_5OmniProcessor, Qwen2TokenizerFast,
                              Qwen2VLImageProcessor, Qwen2VLVideoProcessor,
                              WhisperFeatureExtractor)
    if os.environ.get("OMNI_PROCESSOR_DIR"):
        return Qwen2_5OmniProcessor.from_pretrained(
            os.environ["OMNI_PROCESSOR_DIR"], local_files_only=True)

    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    backend = Tokenizer(models.BPE({t: i for i, t in enumerate(alphabet)}, []))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    mm = {k + "_token": "<|" + v + "|>" for k, v in (
        ("audio", "AUDIO"), ("image", "IMAGE"), ("video", "VIDEO"),
        ("audio_bos", "audio_bos"), ("audio_eos", "audio_eos"),
        ("vision_bos", "vision_bos"), ("vision_eos", "vision_eos"))}
    tok = Qwen2TokenizerFast(tokenizer_object=backend, eos_token="<|im_end|>",
                             extra_special_tokens=mm, model_max_length=32768,
                             additional_special_tokens=["<|im_start|>", *mm.values()])
    template = (
        "{% for m in messages %}<|im_start|>{{ m.role }}\n"
        "{% if m.content is string %}{{ m.content }}{% else %}"
        "{% for item in m.content %}{% if item.type == 'audio' %}"
        "<|audio_bos|><|AUDIO|><|audio_eos|>{% else %}{{ item.text }}{% endif %}"
        "{% endfor %}{% endif %}<|im_end|>\n{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
    return Qwen2_5OmniProcessor(
        Qwen2VLImageProcessor(), Qwen2VLVideoProcessor(),
        WhisperFeatureExtractor(feature_size=128, chunk_length=1), tok, template)


@pytest.fixture
def records(tmp_path):
    # 1600 frames @8 kHz stereo -> 3200 @16 kHz -> 20 mel frames, 5 tokens;
    # second clip: 16 kHz -> 40 mel frames, 10 tokens.
    clips = [("a.wav", 8000, 1600), ("b.wav", 16000, 6400)]
    for name, sr, n in clips:
        signal = np.sin(np.arange(n) * (2 * np.pi * 440 / sr)) * 0.1
        sf.write(tmp_path / name, np.stack([signal, signal], axis=-1), sr)
    rows = [{"id": "x", "audio": "a.wav", "text": "你好。", "emotion": "angry"},
            {"id": "y", "audio": "b.wav", "text": "A longer answer.", "emotion": "excited"}]
    path = tmp_path / "data.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return load_manifest(path, "asr")


def test_manifest_returns_resolved_audio_and_refs(records, tmp_path):
    assert len(records) == 2 and records[0]["audio"] == str(tmp_path / "a.wav")
    assert records[0]["id"] == "x" and answer_for(records[0], "asr") == "你好。"
    assert [answer_for({"emotion": e}, "ser") for e in
            ["angry", "happy", "excited", "happy/excited", "sad", "neutral"]] == list("ABBBCD")


@pytest.mark.parametrize("row,task", [
    ({"audio": "a.wav", "text": ""}, "asr"),
    ({"audio": "a.wav", "emotion": "fear"}, "ser"),
    ({"audio": "a.wav", "text": "a<|im_end|>b"}, "asr"),
    ({"audio": "a.wav", "text": "<|audio_placeholder|>"}, "asr"),
    ({"audio": "missing.wav", "text": "hello"}, "asr"),
])
def test_bad_manifest_reports_source_line(records, tmp_path, row, task):
    path = tmp_path / "bad.jsonl"
    path.write_text("\n" + json.dumps(row) + "\n")
    with pytest.raises(ValueError, match=r"bad\.jsonl:2"):
        load_manifest(path, task)


@pytest.mark.parametrize("task,expected", [("asr", ["你好。", "A longer answer."]),
                                           ("ser", ["A", "B"])])
def test_labels_are_only_answer_and_eos_and_prompts_are_left_padded(
        records, processor, task, expected):
    collate = OmniCollator(processor, task)
    train, prompt = collate(records), collate.prompts(records)
    fields = {"input_ids", "attention_mask", "feature_attention_mask", "input_features"}
    assert set(prompt) == fields and set(train) == fields | {"labels"}
    assert prompt["attention_mask"][:, -1].tolist() == [1, 1]
    assert prompt["attention_mask"][:, 0].tolist() == [0, 1]
    assert prompt["feature_attention_mask"].sum(-1).tolist() == [20, 40]
    assert torch.equal(train["input_features"], prompt["input_features"])
    audio_id = processor.tokenizer.convert_tokens_to_ids(processor.audio_token)
    for i, answer in enumerate(expected):
        prefix = prompt["input_ids"][i][prompt["attention_mask"][i].bool()]
        assert prefix.eq(audio_id).sum().item() == [5, 10][i]
        active = train["attention_mask"][i].bool()
        full, label = train["input_ids"][i][active], train["labels"][i][active]
        assert torch.equal(full[:len(prefix)], prefix)
        target = label[label != -100]
        assert torch.equal(target, full[len(prefix):len(prefix) + len(target)])
        assert label[:len(prefix)].eq(-100).all()
        assert train["labels"][i][~active].eq(-100).all()
        assert target[-1].item() == processor.tokenizer.eos_token_id
        assert processor.tokenizer.decode(target[:-1]) == answer


def test_audio_and_text_overflow_are_rejected(processor, records, monkeypatch):
    collate = OmniCollator(processor, "asr")
    with monkeypatch.context() as patch:
        patch.setattr(processor.feature_extractor, "n_samples", 1000)
        with pytest.raises(ValueError, match="audio"):
            collate(records)
    monkeypatch.setattr(processor.tokenizer, "model_max_length", 8)
    with pytest.raises(ValueError, match="context"):
        collate(records)


def test_detects_processor_placeholder_mismatch(processor, records, monkeypatch):
    original = processor.replace_multimodal_special_tokens

    def corrupt(*args, **kwargs):
        samples = original(*args, **kwargs)
        samples[0] = samples[0].replace(processor.audio_token, "", 1)
        return samples

    monkeypatch.setattr(processor, "replace_multimodal_special_tokens", corrupt)
    with pytest.raises(ValueError, match="placeholder"):
        OmniCollator(processor, "asr").prompts(records)


def test_mismatched_assistant_header_is_rejected(processor, records, monkeypatch):
    template = processor.chat_template.replace(
        "<|im_start|>assistant", "<|im_start|>DIFFERENT")
    monkeypatch.setattr(processor, "chat_template", template)
    with pytest.raises(ValueError, match="prefix"):
        OmniCollator(processor, "asr")(records)
