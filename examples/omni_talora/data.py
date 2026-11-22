"""One local audio clip per JSONL record, for Qwen2.5-Omni Thinker SFT.

Use the processor from the same checkpoint as the Thinker (Transformers 4.52.3).
Training targets include the assistant's answer and its end-of-turn token.
"""
import json
import math
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

from .budget import Budget

SAMPLE_RATE = 16_000
IGNORE_INDEX = -100
SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of "
    "perceiving auditory and visual inputs, as well as generating text and speech.")
EMOTION_TO_ANSWER = {
    "angry": "A", "happy": "B", "excited": "B", "happy/excited": "B",
    "sad": "C", "neutral": "D",
}
TASK_PROMPTS = {
    "asr": "Transcribe the speech verbatim in its original language. Reply with the transcript only.",
    "ser": ("Which emotion does the speaker express? A: angry; B: happy or excited; "
            "C: sad; D: neutral. Reply with exactly one letter: A, B, C, or D."),
}


def answer_for(record, task):
    """Return the gold assistant text; never substitute a default for a missing label."""
    if task not in TASK_PROMPTS:
        raise ValueError(f"unknown task {task!r}; expected 'asr' or 'ser'")
    key = "text" if task == "asr" else "emotion"
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a nonempty string")
    value = value.strip()
    if task == "ser":
        if value.lower() not in EMOTION_TO_ANSWER:
            raise ValueError(f"unsupported emotion {value!r}; expected {list(EMOTION_TO_ANSWER)}")
        return EMOTION_TO_ANSWER[value.lower()]
    # This also rejects the processor's *internal* placeholders, which are not
    # listed in tokenizer.all_special_tokens but are replaced by HF after parsing.
    if re.search(r"<\|[^<>\r\n]*\|>", value):
        raise ValueError("transcript contains a reserved Qwen control token")
    return value


def load_manifest(path, task, *, validate_row=None):
    """Load JSON Lines, retaining extra metadata (e.g. id/speaker/session)."""
    if task not in TASK_PROMPTS:
        raise ValueError(f"unknown task {task!r}")
    path = Path(path).expanduser().resolve()
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("expected a JSON object")
                if task == "asr":
                    for key, alias in (("audio", "wav"), ("text", "txt")):
                        if alias in row:
                            if key in row and row[key] != row[alias]:
                                raise ValueError(f"conflicting {key} and {alias} fields")
                            row[key] = row[alias]
                audio = row.get("audio")
                if not isinstance(audio, str) or not audio.strip():
                    raise ValueError("audio must be a local file path")
                if validate_row is not None:
                    validate_row(row)
                local = (path.parent / Path(audio).expanduser()).resolve()
                if not local.is_file():
                    raise ValueError(f"audio file does not exist: {local}")
                answer_for(row, task)
                records.append({**row, "audio": str(local)})
            except (ValueError, OSError) as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
    if not records:
        raise ValueError(f"{path}: empty manifest")
    return records


def audio_lengths(info, hop):
    samples = (info.frames * SAMPLE_RATE + info.samplerate - 1) // info.samplerate
    frames = (samples + hop - 1) // hop
    return info.frames / info.samplerate, samples, frames


def _load_audio(path, sample_limit):
    with sf.SoundFile(path) as stream:
        rate = stream.samplerate
        if stream.frames == 0 or stream.frames * SAMPLE_RATE > sample_limit * rate:
            raise ValueError(f"{path}: empty or overlength audio; limit {sample_limit / SAMPLE_RATE:g}s")
        waveform = stream.read(dtype="float32", always_2d=True).mean(axis=1)
    if rate != SAMPLE_RATE:
        divisor = math.gcd(rate, SAMPLE_RATE)
        waveform = resample_poly(waveform, SAMPLE_RATE // divisor, rate // divisor)
    if not np.isfinite(waveform).all():
        raise ValueError(f"{path}: audio contains nonfinite samples")
    return waveform


class OmniCollator:
    """Right-pad teacher-forced sequences; left-pad generation prompts."""

    def __init__(self, processor, task, budget=None):
        if task not in TASK_PROMPTS:
            raise ValueError(f"unknown task {task!r}")
        self.budget = budget or Budget()
        self.processor = processor
        self.task = task
        self.tokenizer = processor.tokenizer
        self.audio_token = processor.audio_token
        self.audio_id = self.tokenizer.convert_tokens_to_ids(self.audio_token)
        if processor.feature_extractor.sampling_rate != SAMPLE_RATE:
            raise ValueError("expected the 16 kHz Omni audio feature extractor")
        if self.tokenizer.eos_token is None or self.audio_id is None:
            raise ValueError("processor is missing assistant EOS or audio token")

    def _user(self, record):
        return {"role": "user", "content": [
            {"type": "audio", "audio": record["audio"]},
            {"type": "text", "text": TASK_PROMPTS[self.task]},
        ]}

    def prompt_text(self, record):
        """Render the actual checkpoint template, with unexpanded audio placeholder."""
        return self._render(record)

    def _render(self, record, answer=None):
        messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                    self._user(record)]
        if answer is not None:
            messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
        # 4.52.3 wraps even one conversation into a batch.
        [text] = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=answer is None)
        return text

    def _encode(self, records, texts, side):
        if not records:
            raise ValueError("empty audio batch")
        if any(t.count(self.audio_token) != 1 for t in texts):
            raise ValueError("expected exactly one audio placeholder per rendered input")
        extractor = self.processor.feature_extractor
        if self.budget.max_audio_seconds is not None or self.budget.max_batch_frames is not None:
            total_frames = 0
            for row in records:
                seconds, _, frames = audio_lengths(sf.info(row["audio"]), extractor.hop_length)
                self.budget.check(row.get("id", row["audio"]), seconds=seconds)
                total_frames += frames
            self.budget.check([r.get("id", r["audio"]) for r in records], frames=total_frames)
        waves = [_load_audio(row["audio"], extractor.n_samples) for row in records]
        hop = extractor.hop_length
        frames = [(len(wave) + hop - 1) // hop for wave in waves]
        # HF 4.52.3 always pads audio to max_length. Supply a batch-specific
        # window with an STFT margin instead of padding short clips to 300 s.
        margin = math.ceil(extractor.n_fft / (2 * hop))
        window = min(extractor.n_samples, (max(frames) + margin) * hop)
        encoded = self.processor(
            text=texts, audio=waves, return_tensors="pt",
            text_kwargs={"padding": True, "padding_side": side,
                         "truncation": False, "add_special_tokens": False},
            audio_kwargs={"sampling_rate": SAMPLE_RATE, "max_length": window,
                          "return_attention_mask": True, "truncation": False})
        batch = {k: encoded[k] for k in (
            "input_ids", "attention_mask", "input_features", "feature_attention_mask")}
        mask = batch["feature_attention_mask"]
        if (mask.sum(-1).tolist() != frames or mask.shape[-1] != batch["input_features"].shape[-1]):
            raise ValueError("audio feature frames disagree with untruncated waveform lengths")
        # Encoder's stride-2 convolution followed by stride-2 pooling. Test
        # per row: a batch-wide count would miss swapped clips of unequal length.
        needed = [((f + 1) // 2) // 2 for f in frames]
        actual = ((batch["input_ids"] == self.audio_id) & batch["attention_mask"].bool()).sum(-1)
        if any(n == 0 for n in needed) or actual.tolist() != needed:
            raise ValueError(f"audio placeholder/feature mismatch: expected {needed}, got {actual.tolist()}")
        lengths = batch["attention_mask"].sum(-1).tolist()
        for row, length in zip(records, lengths):
            self.budget.check(row.get("id", row["audio"]), tokens=length)
        if max(lengths) > self.tokenizer.model_max_length:
            raise ValueError(f"input exceeds tokenizer context limit {self.tokenizer.model_max_length}")
        return batch, needed

    def prompts(self, batch):
        """Inference inputs (including audio but no labels/answers), padded on the left."""
        return self._encode(batch, [self.prompt_text(row) for row in batch], "left")[0]

    def training_text(self, row):
        answer = answer_for(row, self.task)
        if any(t and t in answer for t in self.tokenizer.all_special_tokens):
            raise ValueError("assistant answer contains a tokenizer special token")
        prompt = self.prompt_text(row)
        full = self._render(row, answer)
        expected = prompt + answer + self.tokenizer.eos_token
        if not full.startswith(expected) or full[len(expected):].strip():
            raise ValueError("chat template prefix, assistant text or end-of-turn does not match")
        return prompt, full

    def __call__(self, batch):
        rendered = [self.training_text(row) for row in batch]
        prompts = [prompt for prompt, _ in rendered]
        fulls = [full for _, full in rendered]
        encoded, counts = self._encode(batch, fulls, "right")
        # Check the *tokenized* generation prefix against the actual expanded
        # full sequence. Reuse mel features; no second audio decode or STFT.
        expanded = [s.replace(self.audio_token, self.audio_token * n) for s, n in zip(prompts, counts)]
        prefixes = self.tokenizer(expanded, padding=False, truncation=False, add_special_tokens=False)["input_ids"]
        labels = torch.full_like(encoded["input_ids"], IGNORE_INDEX)
        for i, prefix in enumerate(prefixes):
            ids = encoded["input_ids"][i][encoded["attention_mask"][i].bool()]
            start = len(prefix)
            if ids[:start].tolist() != prefix:
                raise ValueError(f"sample {i}: generation prompt is not a token prefix of the full input")
            ends = (ids[start:] == self.tokenizer.eos_token_id).nonzero().flatten()
            if len(ends) != 1 or ends.item() < 1:
                raise ValueError(f"sample {i}: expected a nonempty answer and one assistant EOS")
            stop = start + ends.item() + 1
            labels[i, start:stop] = ids[start:stop]
        return {**encoded, "labels": labels}
