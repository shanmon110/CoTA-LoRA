"""Utterance modality layout carried across attention and checkpoint replay."""
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar

import torch

_layout = ContextVar("omni_talora_layout", default=None)


@contextmanager
def _use_audio_layout(layout):
    token = _layout.set(layout)
    try:
        yield
    finally:
        _layout.reset(token)


def current_audio_layout():
    return _layout.get()


def audio_mask_context(audio, valid):
    if audio.ndim != 2 or audio.shape != valid.shape:
        raise ValueError("Expected matching two-dimensional modality and padding masks")
    return _use_audio_layout((audio.bool() & valid.bool(), valid.bool()))


def checkpoint_audio_context():
    """Capture this forward's layout for recomputation on an autograd worker."""
    return nullcontext(), _use_audio_layout(_layout.get())


@contextmanager
def audio_context(input_ids, attention_mask, audio_token_id):
    """Mark *expanded audio placeholder IDs*; BOS/EOS remain non-audio.

    For generation, pass just the padded prompt; appended KV slots are text.
    ContextVar tokens also restore outer contexts on exceptions.
    """
    if input_ids.ndim != 2 or input_ids.numel() == 0:
        raise ValueError("expected nonempty, unpacked [batch, sequence] input_ids")
    if attention_mask is not None and attention_mask.shape != input_ids.shape:
        raise ValueError("expected a 2-D padding mask matching input_ids")
    valid = torch.ones_like(input_ids, dtype=torch.bool) if attention_mask is None else attention_mask.bool()
    with _use_audio_layout((input_ids.eq(audio_token_id) & valid, valid)):
        yield
