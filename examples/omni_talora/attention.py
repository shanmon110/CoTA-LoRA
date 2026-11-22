"""TA-LoRA for the Transformers 4.52.3 Qwen2.5-Omni Thinker.

Only for unpacked audio/text prompts and greedy text generation (dynamic KV
cache). Put the entire forward/backward or generate call in `audio_context`.
Load the frozen model in its intended dtype *before* installing adapters.
"""

import math
from types import MethodType

import torch
from torch.nn import functional as F
from transformers import __version__ as transformers_version
from transformers.cache_utils import DynamicCache
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
    Qwen2_5OmniAttention,
    apply_multimodal_rotary_pos_emb,
    repeat_kv,
)

from west_adapters.base import FP32_ADD_THEN_CAST, add_fp32_residual
from west_adapters.context import _layout, audio_context, checkpoint_audio_context, current_audio_layout
from west_adapters.registry import checkpoint_method


def _SharedAtoms(proj, rank, alpha, mode, residual_precision):
    method = checkpoint_method({"mode": mode})
    return method.atoms(proj, rank, alpha, residual_precision)


def _adapter_forward(self, hidden_states, attention_mask=None, position_ids=None,
                     past_key_value=None, output_attentions=False, use_cache=False,
                     cache_position=None, position_embeddings=None):
    # Adapted from Qwen2_5OmniAttention.forward (Transformers v4.52.3,
    # Copyright 2025 The Qwen team, Alibaba Group and HuggingFace Inc., Apache-2.0).
    # Keep the original eager numerics/return contract, adding a branch before
    # o_proj. Cache update happens exactly once, before repeating grouped heads.
    layout = current_audio_layout()
    if self._adapter_needs_audio_context:
        if layout is None:
            raise RuntimeError("TA mode requires audio_context around forward/backward or generate")
        if past_key_value is not None and type(past_key_value) is not DynamicCache:
            raise ValueError("TA mode supports DynamicCache only (no static/sliding/beam cache)")
    b, q, _ = hidden_states.shape

    def heads(projection):
        return projection(hidden_states).view(b, q, -1, self.head_dim).transpose(1, 2)

    query, key, value = heads(self.q_proj), heads(self.k_proj), heads(self.v_proj)
    cos, sin = position_embeddings
    query, key = apply_multimodal_rotary_pos_emb(query, key, cos, sin, self.rope_scaling["mrope_section"])
    if past_key_value is not None:
        key, value = past_key_value.update(key, value, self.layer_idx,
                                           {"sin": sin, "cos": cos, "cache_position": cache_position})
    key = repeat_kv(key, self.num_key_value_groups)
    value = repeat_kv(value, self.num_key_value_groups)
    logits = (query @ key.transpose(2, 3)) / math.sqrt(self.head_dim)
    mask = None
    if attention_mask is not None:
        mask = attention_mask[:, :, :, :key.size(2)]
        logits = logits + mask
    if query.dtype == torch.float16:
        logits = torch.where(torch.isinf(logits), torch.zeros_like(logits), logits)
    weights = F.softmax(logits, dim=-1, dtype=torch.float32).to(query.dtype)
    weights = F.dropout(weights, p=self.attention_dropout, training=self.training)
    ordinary = (weights @ value).transpose(1, 2).contiguous().reshape(b, q, -1)
    base = self.o_proj(ordinary)
    delta = self.ta_adapter.attention_delta(ordinary, logits, value, mask, cache_position, layout)
    if self.ta_adapter.residual_precision == FP32_ADD_THEN_CAST:
        out = add_fp32_residual(base, delta)
    else:  # Historical O/TA checkpoints rounded the delta before addition.
        out = base + delta.to(base.dtype)
    return out, (weights if output_attentions else None), past_key_value


def install_adapter(thinker, *, last_n=4, rank=8, alpha=16., mode="ta", residual_precision=None):
    """Freeze the Thinker and adapt exactly its last N language-attention layers.

    Returns those attention modules, in layer order. Native parameter names
    remain unchanged; only `self_attn.ta_adapter.{down,up[,gate]}` are trainable.
    """
    if transformers_version != "4.52.3":
        raise ValueError(f"expected transformers==4.52.3; got {transformers_version}")
    layers = thinker.model.layers
    if type(last_n) is not int or not 1 <= last_n <= len(layers):
        raise ValueError("last_n must be in [1, number of Thinker layers]")
    if type(rank) is not int or rank < 1 or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("expected positive rank and finite, positive alpha")
    method = checkpoint_method({"mode": mode})
    if not method.attention:
        raise ValueError(f"{method.name} requires a projection installer")
    if residual_precision not in (None, FP32_ADD_THEN_CAST):
        raise ValueError(f"unknown residual_precision: {residual_precision!r}")
    if any(hasattr(layer.self_attn, "ta_adapter") for layer in layers):
        raise ValueError("adapter already installed")
    targets = [layer.self_attn for layer in layers[-last_n:]]
    for attn in targets:
        if type(attn) is not Qwen2_5OmniAttention or attn.config._attn_implementation != "eager":
            raise ValueError("load the native Qwen2.5-Omni Thinker with attn_implementation='eager'")
    thinker.requires_grad_(False)
    for attn in targets:
        attn._adapter_needs_audio_context = method.needs_audio_context
        attn.ta_adapter = method.atoms(attn.o_proj, rank, alpha, residual_precision)
        attn.forward = MethodType(_adapter_forward, attn)
    return targets
