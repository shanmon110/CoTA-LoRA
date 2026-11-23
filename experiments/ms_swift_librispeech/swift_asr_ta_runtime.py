"""Instance-local TA-LoRA for the ms-swift Qwen2.5-Omni and Qwen3-ASR Thinkers."""

import math
import torch
import torch.nn.functional as F
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from functools import wraps
from torch import nn
from transformers import __version__ as transformers_version
from transformers.cache_utils import DynamicCache
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (Qwen2_5OmniAttention,
                                                                    apply_multimodal_rotary_pos_emb, repeat_kv)
from types import MethodType
from typing import NamedTuple


TRANSPORT_GATES = ('static', 'question', 'question_audio')
# Answer-query controls (checkpoint format 9). The defaults reproduce format 8 exactly.
#   contrast          E[V|audio] - E[V|text], the CoTA signal
#   audio_mean        E[V|audio] only, i.e. without subtracting the text mean
#   ordinary          the ordinary attention output at the same query: same position, parameters and
#                     initialisation as CoTA, no modality partition
#   ordinary_matched  as ordinary, rescaled per row to the norm of the contrast
TRANSPORT_SIGNALS = ('contrast', 'audio_mean', 'ordinary', 'ordinary_matched')
# 'autocast' rounds the gated rank coefficients to the autocast dtype before the readouts (format 8).
# 'fp32' keeps the whole transport term in FP32 until it is added to the ordinary update.
TRANSPORT_PRECISIONS = ('autocast', 'fp32')
# Where the whole adapter update is added (checkpoint format 10). 'all' is every earlier format.
#   audio       only at audio-token positions: the adapter can change what the audio positions expose,
#               but text and answer positions are computed by the frozen base
#   post_audio  only at positions after the row's last audio token (question, answer and generated
#               tokens): audio positions and everything before them are exactly the frozen base
UPDATE_POSITIONS = ('all', 'audio', 'post_audio')


class _SharedAtoms(nn.Module):

    def __init__(self, projection, rank, alpha, format_version):
        super().__init__()
        opts = dict(device=projection.weight.device, dtype=torch.float32)
        self.down = nn.Parameter(torch.empty(rank, projection.in_features, **opts))
        self.up = nn.Parameter(torch.zeros(projection.out_features, rank, **opts))
        self.gate = nn.Parameter(torch.zeros(rank, **opts))
        self.gate_multiplier = 1.
        self.transport_queries = 'text'
        self.transport_weighting = 'uniform'
        self.transport_gate = None
        self.transport_signal = 'contrast'
        self.transport_precision = 'autocast'
        self.update_positions = 'all'
        self.register_parameter('transport_up', None)
        self.register_parameter('gate_weight', None)
        nn.init.kaiming_uniform_(self.down, a=math.sqrt(5))
        self.scale = alpha / rank
        self.format_version = format_version

    def forward(self, ordinary, contrast):
        if contrast is None:
            # The transport term is exactly zero (zero multiplier, no independent C): a plain low-rank update.
            if self.format_version == 2:
                return F.linear(F.linear(ordinary.to(self.down.dtype), self.down), self.up) * self.scale
            with torch.autocast(device_type=ordinary.device.type, enabled=False):
                return self.scale * F.linear(F.linear(ordinary.float(), self.down), self.up)
        gate = self.gate if self.gate_multiplier == 1. else self.gate * self.gate_multiplier
        if self.format_version == 2:
            # Follow PEFT Linear.forward: feed A in its parameter dtype, honor
            # the enclosing autocast, multiply B's output by scaling.
            coeff = F.linear(ordinary.to(self.down.dtype), self.down)
            extra = F.linear(contrast.to(self.down.dtype), self.down)
            coeff = coeff + (gate * extra).to(coeff.dtype)
            return self._project(coeff, extra) * self.scale
        with torch.autocast(device_type=ordinary.device.type, enabled=False):
            ordinary_coeff = F.linear(ordinary.float(), self.down)
            extra = F.linear(contrast, self.down)
            return self.scale * self._project(ordinary_coeff + gate * extra, extra)

    def _project(self, coeff, extra):
        update = F.linear(coeff, self.up)
        if self.transport_up is not None:
            update = update + F.linear(extra, self.transport_up)
        return update

    def _context_multiplier(self, question, audio):
        features = F.normalize(question.float(), dim=-1)
        if self.transport_gate == 'question_audio':
            features = (features + F.normalize(audio.float(), dim=-1)) / math.sqrt(2)
        return 2 * torch.sigmoid(F.linear(features, self.gate_weight))

    def _fp32_transport(self, contrast, question, device_type):
        # Under BF16 autocast the format-8 path rounds each gated coefficient back to 8 significant
        # bits, which erases any multiplier with |m - 1| < 2**-9. Here the rank coefficients, the
        # multiplier and both readouts stay in FP32, matching what inference without autocast computes.
        with torch.autocast(device_type=device_type, enabled=False):
            extra = F.linear(contrast.float(), self.down)
            if self.gate_weight is not None:
                extra = extra * self._context_multiplier(F.linear(question, self.down), extra)
            transport = F.linear(extra * self.gate * self.gate_multiplier, self.up)
            if self.transport_up is not None:
                transport = transport + F.linear(extra, self.transport_up)
        return transport

    def answer_update(self, ordinary, contrast, query_indices, question):
        ordinary_update = F.linear(F.linear(ordinary.to(self.down.dtype), self.down), self.up)
        if self.transport_precision == 'fp32':
            transport = self._fp32_transport(contrast, question, ordinary.device.type)
        else:
            extra = F.linear(contrast.to(self.down.dtype), self.down)
            if self.gate_weight is not None:
                with torch.autocast(device_type=ordinary.device.type, enabled=False):
                    query_features = F.linear(question, self.down)
                    multiplier = self._context_multiplier(query_features, extra)
                extra = (extra * multiplier).to(extra.dtype)
            shared = F.linear((extra * self.gate * self.gate_multiplier).to(extra.dtype), self.up)
            transport = shared if self.transport_up is None else shared + F.linear(extra, self.transport_up)
        index = query_indices[:, None, None].expand(-1, 1, ordinary_update.shape[-1])
        update = ordinary_update.scatter_add(1, index, transport.to(ordinary_update.dtype).unsqueeze(1))
        return self.scale * update


def enable_transport_up(outer):
    """Add an independent, zero-initialized output matrix for the shared contrast coefficients."""
    cfg = outer._ta_lora_config
    if cfg['tuner_type'] != 'ta_lora':
        raise ValueError('independent_transport_up requires the ta_lora tuner')
    cfg.setdefault('residual_format_version', min(cfg['format_version'], 2))
    cfg.setdefault('gate_multiplier', 1.)
    cfg.setdefault('transport_queries', 'text')
    cfg.update(format_version=max(cfg['format_version'], 6), independent_transport_up=True)
    for layer in outer.thinker.model.layers:
        branch = getattr(layer.self_attn, 'ta_adapter', None)
        if branch is not None and branch.transport_up is None:
            branch.transport_up = nn.Parameter(torch.zeros_like(branch.up), requires_grad=branch.up.requires_grad)


def set_gate_multiplier(outer, multiplier):
    """Reparameterize the effective gate as multiplier * raw_gate, preserving raw weights."""
    if type(multiplier) not in (float, int) or not math.isfinite(multiplier) or multiplier < 0:
        raise ValueError('gate_multiplier must be a finite, nonnegative number')
    cfg = outer._ta_lora_config
    if multiplier != 1. and cfg['format_version'] < 4:
        # Readers of earlier formats ignore extra metadata and would silently omit the multiplier.
        cfg['residual_format_version'] = min(cfg['format_version'], 2)
        cfg['format_version'] = 4
    if cfg['format_version'] >= 4:
        cfg['gate_multiplier'] = float(multiplier)
    for layer in outer.thinker.model.layers:
        branch = getattr(layer.self_attn, 'ta_adapter', None)
        if branch is not None:
            branch.gate_multiplier = float(multiplier)


def set_transport_queries(outer, scope):
    """Select text-only or all valid queries; both modes require visible keys of each modality."""
    if scope not in ('text', 'all_valid'):
        raise ValueError('transport_queries must be text or all_valid')
    cfg = outer._ta_lora_config
    if scope == 'all_valid' and cfg['format_version'] < 5:
        cfg.setdefault('residual_format_version', min(cfg['format_version'], 2))
        cfg.setdefault('gate_multiplier', 1.)
        cfg['format_version'] = 5
    if cfg['format_version'] >= 5:
        cfg['transport_queries'] = scope
    for layer in outer.thinker.model.layers:
        branch = getattr(layer.self_attn, 'ta_adapter', None)
        if branch is not None:
            branch.transport_queries = scope


def set_transport_weighting(outer, weighting):
    """Persist the per-head weighting of the audio/text contrast."""
    if weighting not in ('uniform', 'attention_overlap'):
        raise ValueError('transport_weighting must be uniform or attention_overlap')
    cfg = outer._ta_lora_config
    if weighting == 'attention_overlap' and cfg['format_version'] < 7:
        cfg.setdefault('residual_format_version', min(cfg['format_version'], 2))
        cfg.setdefault('gate_multiplier', 1.)
        cfg.setdefault('transport_queries', 'text')
        cfg.setdefault('independent_transport_up', False)
        cfg['format_version'] = 7
    if cfg['format_version'] >= 7:
        cfg['transport_weighting'] = weighting
    for layer in outer.thinker.model.layers:
        branch = getattr(layer.self_attn, 'ta_adapter', None)
        if branch is not None:
            branch.transport_weighting = weighting


def set_transport_gate(outer, mode):
    """Opt in to one answer query and optionally a learned prompt-conditioned multiplier."""
    cfg = outer._ta_lora_config
    residual = cfg.get('residual_format_version', min(cfg['format_version'], 2))
    if (mode not in TRANSPORT_GATES or cfg['tuner_type'] != 'ta_lora'
            or outer.config.model_type != 'qwen2_5_omni'
            or cfg.get('transport_queries', 'text') != 'text' or residual != 2
            or cfg.get('update_positions', 'all') != 'all'
            or any(not name.endswith('.self_attn.o_proj') for name in cfg['targets'])):
        raise ValueError('Answer-query transport requires Qwen2.5-Omni O-only TA with text queries')
    if cfg.get('transport_gate', mode) != mode:
        raise ValueError('Cannot change the transport gate mode after configuration')
    independent = cfg.get('independent_transport_up', False)
    cfg.setdefault('residual_format_version', residual)
    cfg.setdefault('gate_multiplier', 1.)
    cfg.setdefault('transport_queries', 'text')
    cfg.setdefault('independent_transport_up', independent)
    # Format 8 always has the independent readout C. The shared-only control (transport through
    # B*g alone) is written as format 9 so that format-8 readers reject it instead of misreading it.
    cfg.update(format_version=max(cfg['format_version'], 8 if independent else 9), transport_gate=mode)
    cfg.setdefault('transport_weighting', 'uniform')
    if cfg['format_version'] >= 9:
        cfg.setdefault('transport_signal', 'contrast')
        cfg.setdefault('transport_precision', 'autocast')
    for layer in outer.thinker.model.layers:
        branch = getattr(layer.self_attn, 'ta_adapter', None)
        if branch is not None:
            branch.transport_gate = mode
            if mode != 'static' and branch.gate_weight is None:
                rank = branch.down.shape[0]
                branch.gate_weight = nn.Parameter(branch.down.new_zeros(rank, rank),
                                                  requires_grad=branch.up.requires_grad)


def set_update_positions(outer, scope):
    """Restrict the whole low-rank update to audio-token positions or to the positions after them."""
    if scope not in UPDATE_POSITIONS:
        raise ValueError(f'update_positions must be one of {UPDATE_POSITIONS}')
    cfg = outer._ta_lora_config
    if scope == 'all':
        if cfg['format_version'] == 10:
            raise ValueError('A position-restricted adapter cannot be changed to update all positions')
    else:
        if (cfg['tuner_type'] not in ('ta_lora', 'ta_gate_off') or outer.config.model_type != 'qwen2_5_omni'
                or cfg.get('transport_gate') is not None
                or any(not name.endswith('.self_attn.o_proj') for name in cfg['targets'])):
            raise ValueError('Position-restricted updates require Qwen2.5-Omni O-only TA without answer-query transport')
        if cfg.get('update_positions', scope) != scope:
            raise ValueError('Cannot change the update positions after configuration')
        # Earlier readers ignore unknown metadata and would apply the update everywhere, so the
        # restriction gets its own format. Format 10 carries every key of the static formats.
        cfg.setdefault('residual_format_version', min(cfg['format_version'], 2))
        cfg.setdefault('gate_multiplier', 1.)
        cfg.setdefault('transport_queries', 'text')
        cfg.setdefault('independent_transport_up', False)
        cfg.setdefault('transport_weighting', 'uniform')
        cfg.update(format_version=10, update_positions=scope)
    for layer in outer.thinker.model.layers:
        branch = getattr(layer.self_attn, 'ta_adapter', None)
        if branch is not None:
            branch.update_positions = scope


def _set_answer_option(outer, key, value, choices):
    cfg = outer._ta_lora_config
    if value not in choices:
        raise ValueError(f'{key} must be one of {choices}')
    if cfg.get('transport_gate') is None:
        raise ValueError(f'{key} requires answer-query transport; set transport_gate first')
    if value != choices[0] or cfg['format_version'] >= 9:
        cfg['format_version'] = 9
        cfg.setdefault('transport_signal', 'contrast')
        cfg.setdefault('transport_precision', 'autocast')
        cfg[key] = value
    for layer in outer.thinker.model.layers:
        branch = getattr(layer.self_attn, 'ta_adapter', None)
        if branch is not None:
            setattr(branch, key, value)


def set_transport_signal(outer, signal):
    """Choose what the answer-query readout sees; non-default values are matched controls."""
    _set_answer_option(outer, 'transport_signal', signal, TRANSPORT_SIGNALS)


def set_transport_precision(outer, precision):
    """Choose where the gated rank coefficients are rounded during autocast training."""
    _set_answer_option(outer, 'transport_precision', precision, TRANSPORT_PRECISIONS)


class _AnswerFrame(NamedTuple):
    audio: torch.Tensor
    valid: torch.Tensor
    positions: torch.Tensor
    question: torch.Tensor | None


def _answer_query(input_ids, valid, labels, audio_token_id, embed=None):
    """Find the logits immediately before the first target without using the answer text."""
    audio = input_ids.eq(audio_token_id) & valid
    indices = torch.arange(input_ids.shape[1], device=input_ids.device)
    if labels is None:
        positions = indices.masked_fill(~valid, -1).amax(dim=1)
    else:
        if labels.shape != input_ids.shape:
            raise ValueError('Answer-query transport requires full, unshifted labels')
        first_label = labels.ne(-100).int().argmax(dim=1)
        gold = labels.gather(1, first_label[:, None])
        if not ((first_label > 0) & gold.squeeze(1).eq(input_ids.gather(1, first_label[:, None]).squeeze(1))).all():
            raise ValueError('Each training row needs a visible prompt and a supervised next token')
        positions = first_label - 1
    if (positions < 0).any() or not (valid & ~audio).gather(1, positions[:, None]).all():
        raise ValueError('Expected a valid text token immediately before the answer')
    question = None
    if embed is not None:
        prefix = valid & (indices <= positions[:, None])
        audio_end = indices.masked_fill(~(audio & prefix), -1).amax(dim=1)
        question_mask = prefix & (indices > audio_end[:, None])
        with torch.no_grad():
            embeddings = embed(input_ids).float()
            total = embeddings.masked_fill(~question_mask[..., None], 0).sum(dim=1)
            question = total / question_mask.sum(dim=1, keepdim=True).clamp_min(1)
    return _AnswerFrame(audio, valid, positions, question)


def _update_mask(audio, cache_position, queries, keys, scope):
    """[batch, queries, 1] mask of the query positions at which the adapter update is added.

    audio is the prompt layout [batch, prompt]; slots beyond it are generated text.
    """
    pos = (torch.arange(keys - queries, keys, device=audio.device) if cache_position is None
           else cache_position.to(audio.device))
    if pos.ndim != 1 or pos.numel() != queries:
        raise ValueError('expected one absolute cache position per query')
    prompt = audio.shape[1]
    if scope == 'audio':
        inside = pos < prompt
        mask = torch.zeros(audio.shape[0], queries, dtype=torch.bool, device=audio.device)
        mask[:, inside] = audio[:, pos[inside]]
    elif scope == 'post_audio':
        # Strictly after the row's last audio token. A row without audio has nothing before it.
        last = torch.arange(prompt, device=audio.device).masked_fill(~audio, -1).amax(dim=1)
        mask = pos[None, :] > last[:, None]
    else:
        raise ValueError(f'unknown update position scope {scope!r}')
    return mask[..., None]


def _conditional_probs(logits, members):
    # An all -inf row produces NaNs in backward even if its output is subsequently masked.
    present = members.any(dim=-1, keepdim=True)
    safe = torch.where(present, logits.masked_fill(~members, -torch.inf), 0.)
    return safe.softmax(dim=-1, dtype=torch.float32), present


def _answer_contrast(logits, value, mask, cache_position, frame, joint_probabilities,
                     signal='contrast', ordinary=None):
    """Evaluate contrast for one query in each row, with zero on subsequent cache calls."""
    batch, _, query_len, keys = logits.shape
    positions = (torch.arange(keys - query_len, keys, device=logits.device)
                 if cache_position is None else cache_position)
    matches = frame.positions[:, None] == positions
    query_indices = matches.int().argmax(dim=1)
    rows = torch.arange(batch, device=logits.device)
    audio, valid = frame.audio[:, :keys], frame.valid[:, :keys]
    if keys > audio.shape[1]:
        pad = (0, keys - audio.shape[1])
        audio, valid = F.pad(audio, pad, value=False), F.pad(valid, pad, value=True)
    visible = valid & (torch.arange(keys, device=logits.device) <= frame.positions[:, None])
    if mask is not None:
        visible = visible & mask.expand(batch, -1, -1, -1)[rows, 0, query_indices].eq(0)
    audio_keys = (visible & audio)[:, None, :]
    text_keys = (visible & ~audio)[:, None, :]
    selected = logits[rows, :, query_indices]
    with torch.autocast(device_type=logits.device.type, enabled=False):
        audio_probs, has_audio = _conditional_probs(selected, audio_keys)
        text_probs, has_text = _conditional_probs(selected, text_keys)
        weights = audio_probs if signal == 'audio_mean' else audio_probs - text_probs
        contrast = weights.unsqueeze(2) @ value.float()
        if joint_probabilities is not None:
            joint = joint_probabilities[rows, :, query_indices]
            audio_mass = (joint * audio_keys).sum(dim=-1, keepdim=True)
            text_mass = (joint * text_keys).sum(dim=-1, keepdim=True)
            contrast = contrast * (4 * audio_mass * text_mass).unsqueeze(-1)
        eligible = matches.any(dim=1)[:, None, None] & has_audio & has_text
        contrast = contrast.squeeze(2).masked_fill(~eligible, 0).flatten(1)
        if signal in ('ordinary', 'ordinary_matched'):
            # Same query, eligibility, parameters and initialisation; no modality partition.
            control = ordinary[rows, query_indices].float().masked_fill(~eligible[:, 0], 0)
            if signal == 'ordinary_matched':
                scale = contrast.norm(dim=-1, keepdim=True) / control.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                control = control * scale
            contrast = control
    return contrast, query_indices


def _transport(logits, value, mask, cache_position, layout, transport_queries='text', joint_probabilities=None):
    b, _, q, k = logits.shape
    audio, valid = layout
    if audio.shape[0] != b:
        raise ValueError('TA-LoRA does not support batch expansion or beam search')
    audio, valid = audio[:, :k].to(logits.device), valid[:, :k].to(logits.device)
    if k > audio.size(1):
        # Slot indices beyond the prompt are generated text; mask padded prompt keys separately.
        pad = (0, k - audio.size(1))
        audio, valid = F.pad(audio, pad, value=False), F.pad(valid, pad, value=True)
    pos = torch.arange(k - q, k, device=logits.device) if cache_position is None else cache_position.to(logits.device)
    if pos.ndim != 1 or pos.numel() != q:
        raise ValueError('expected one absolute cache position per query')
    visible = valid[:, None, None, :] & (torch.arange(k, device=pos.device) <= pos[:, None])
    if mask is not None:
        visible = visible & mask.eq(0)  # HF 4.52.3 eager uses an inverted, additive mask.
    queries = valid if transport_queries == 'all_valid' else valid & ~audio
    eligible_query = queries.index_select(1, pos)[:, None, :, None]
    with torch.autocast(device_type=logits.device.type, enabled=False):
        if joint_probabilities is None:
            p_audio, has_audio = _conditional_probs(logits, visible & audio[:, None, None, :])
            p_text, has_text = _conditional_probs(logits, visible & ~audio[:, None, None, :])
            diff = (p_audio - p_text).masked_fill(~(eligible_query & has_audio & has_text), 0.)
            contrast = diff @ value.float()
        else:
            # 4*m_a*m_t*(E[v|audio]-E[v|text]) without dividing by either mass.
            # The appended 1 accumulates each modality's probability in the same matmul.
            joint = joint_probabilities.masked_fill(~visible, 0.)
            with_mass = F.pad(value.float(), (0, 1), value=1.)
            audio_keys = audio[:, None, :, None]
            a = joint @ with_mass.masked_fill(~audio_keys, 0.)
            t = joint @ with_mass.masked_fill(audio_keys, 0.)
            contrast = 4 * (a[..., :-1] * t[..., -1:] - t[..., :-1] * a[..., -1:])
            contrast = contrast.masked_fill(~eligible_query, 0.)
    return contrast.transpose(1, 2).reshape(b, q, -1)


def _attention(self,
               hidden_states,
               attention_mask=None,
               position_ids=None,
               past_key_value=None,
               output_attentions=False,
               use_cache=False,
               cache_position=None,
               position_embeddings=None):
    # Eager attention adapted from Transformers v4.52.3 Qwen2_5OmniAttention.forward
    # (Copyright 2025 The Qwen team, Alibaba Group and HuggingFace Inc.; Apache-2.0).
    frames = self._ta_frames.get()
    if not frames or frames[-1] is None:
        raise RuntimeError('call the ms-swift outer model or its generate method so TA can see expanded input_ids')
    if past_key_value is not None and type(past_key_value) is not DynamicCache:
        raise ValueError('TA-LoRA supports only an append-only DynamicCache')
    b, q, _ = hidden_states.shape

    def heads(proj):
        return proj(hidden_states).view(b, q, -1, self.head_dim).transpose(1, 2)

    query, key, value = heads(self.q_proj), heads(self.k_proj), heads(self.v_proj)
    cos, sin = position_embeddings
    query, key = apply_multimodal_rotary_pos_emb(query, key, cos, sin, self.rope_scaling['mrope_section'])
    if past_key_value is not None:
        key, value = past_key_value.update(key, value, self.layer_idx,
                                           dict(sin=sin, cos=cos, cache_position=cache_position))
    key, value = repeat_kv(key, self.num_key_value_groups), repeat_kv(value, self.num_key_value_groups)
    logits = (query @ key.transpose(2, 3)) / math.sqrt(self.head_dim)
    mask = None if attention_mask is None else attention_mask[:, :, :, :key.shape[2]]
    if mask is not None:
        logits = logits + mask
    if query.dtype == torch.float16:
        logits = torch.where(torch.isinf(logits), torch.zeros_like(logits), logits)
    probabilities = logits.softmax(dim=-1, dtype=torch.float32)
    weights = probabilities.to(query.dtype)
    joint = probabilities if self.ta_adapter.transport_weighting == 'attention_overlap' else None
    del probabilities
    weights = F.dropout(weights, p=self.attention_dropout, training=self.training)
    ordinary = (weights @ value).transpose(1, 2).contiguous().reshape(b, q, -1)
    base = self.o_proj(ordinary)
    if isinstance(frames[-1], _AnswerFrame):
        contrast, indices = _answer_contrast(logits, value, mask, cache_position, frames[-1], joint,
                                             self.ta_adapter.transport_signal, ordinary)
        update = self.ta_adapter.answer_update(ordinary, contrast, indices, frames[-1].question)
    else:
        # With a zero multiplier and no independent C the contrast is multiplied by zero. Skipping it leaves
        # the update unchanged and saves two FP32 attention maps per layer, which backward would also keep.
        unused = self.ta_adapter.gate_multiplier == 0. and self.ta_adapter.transport_up is None
        contrast = None if unused else _transport(logits, value, mask, cache_position, frames[-1],
                                                  self.ta_adapter.transport_queries, joint)
        update = self.ta_adapter(ordinary, contrast)
    if self.ta_adapter.update_positions != 'all':
        if frames[-1][0].shape[0] != b:
            raise ValueError('TA-LoRA does not support batch expansion or beam search')
        where = _update_mask(frames[-1][0], cache_position, q, key.shape[2], self.ta_adapter.update_positions)
        update = update * where.to(device=update.device, dtype=update.dtype)
    if self.ta_adapter.format_version == 2:
        output = (base + update).to(base.dtype)  # Same addition and final cast as PEFT LoRA.
    else:
        output = base + update.to(base.dtype)  # Original integrated tuner checkpoint semantics.
    return output, (weights if output_attentions else None), past_key_value


def _qwen3_attention(self, hidden_states, position_embeddings, attention_mask=None,
                     past_key_values=None, cache_position=None, **kwargs):
    from qwen_asr.core.transformers_backend.modeling_qwen3_asr import apply_rotary_pos_emb
    shape = (*hidden_states.shape[:-1], -1, self.head_dim)
    q = self.q_norm(self.q_proj(hidden_states).view(shape)).transpose(1, 2)
    k = self.k_norm(self.k_proj(hidden_states).view(shape)).transpose(1, 2)
    v = self.v_proj(hidden_states).view(shape).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, *position_embeddings)
    if past_key_values is not None:
        k, v = past_key_values.update(k, v, self.layer_idx, dict(
            cos=position_embeddings[0], sin=position_embeddings[1], cache_position=cache_position))
    k, v = repeat_kv(k, self.num_key_value_groups), repeat_kv(v, self.num_key_value_groups)
    logits = (q @ k.transpose(2, 3)) * self.scaling
    mask = None if attention_mask is None else attention_mask[..., :k.shape[-2]]
    if mask is not None:
        logits = logits + mask
    probabilities = logits.softmax(-1, dtype=torch.float32)
    weights = F.dropout(probabilities.to(q.dtype), p=self.attention_dropout, training=self.training)
    joint = probabilities if self.ta_adapter.transport_weighting == 'attention_overlap' else None
    del probabilities
    ordinary = (weights @ v).transpose(1, 2).reshape(*hidden_states.shape[:2], -1)
    contrast = _transport(logits, v, mask, cache_position, self._ta_frames.get()[-1], self.ta_adapter.transport_queries, joint)
    base = self.o_proj(ordinary)
    return (base + self.ta_adapter(ordinary, contrast)).to(base.dtype), weights


def _layout(input_ids, kwargs, token_id):
    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2 or not input_ids.numel():
        raise ValueError('TA-LoRA requires unpacked, expanded [batch, sequence] input_ids before embedding')
    if any(kwargs.get(k) is not None for k in ('cu_seq_lens_q', 'cu_seq_lens_k', 'max_length_q', 'max_length_k')):
        raise ValueError('TA-LoRA does not support packing or padding-free batches')
    vision_keys = ('pixel_values', 'pixel_values_videos', 'image_grid_thw', 'video_grid_thw')
    if any(kwargs.get(k) is not None for k in vision_keys):
        raise ValueError('this TA-LoRA recipe supports only audio/text inputs')
    if kwargs.get('past_key_values') is not None:
        raise ValueError('use outer.generate for cached decoding; externally supplied KV caches are unsupported')
    mask = kwargs.get('attention_mask')
    if mask is not None and mask.shape != input_ids.shape:
        raise ValueError('TA-LoRA requires a 2-D padding mask matching input_ids; disable packing')
    if mask is not None and mask.dtype != torch.bool and not torch.all((mask == 0) | (mask == 1)):
        raise ValueError('TA-LoRA requires a binary padding mask; packed sequence IDs are unsupported')
    valid = torch.ones_like(input_ids, dtype=torch.bool) if mask is None else mask.bool()
    return input_ids.eq(token_id) & valid, valid


def attach(outer, *, last_n=4, rank=8, alpha=16., format_version=2):
    """Freeze outer; add FP32 atoms to its last N eager language attentions; return outer."""
    if format_version not in (1, 2):
        raise ValueError('unknown TA checkpoint format')
    qwen3 = outer.config.model_type == 'qwen3_asr'
    required_version = '4.57.6' if qwen3 else '4.52.3'
    if transformers_version != required_version:
        raise ValueError(f'TA-LoRA requires transformers=={required_version}; got {transformers_version}')
    thinker = getattr(outer, 'thinker', None)
    if thinker is None or not hasattr(thinker, 'model') or not hasattr(thinker.model, 'layers'):
        raise ValueError('TA-LoRA requires the ms-swift Qwen2.5-Omni outer model with a Thinker')
    layers = thinker.model.layers
    if type(last_n) is not int or not 1 <= last_n <= len(layers):
        raise ValueError('last_n must be in [1, number of Thinker layers]')
    if type(rank) is not int or rank < 1 or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError('rank and alpha must be positive and finite')
    if any(hasattr(layer.self_attn, 'ta_adapter') for layer in layers):
        raise ValueError('TA-LoRA is already attached')
    targets = [layer.self_attn for layer in layers[-last_n:]]
    expected_attention = Qwen2_5OmniAttention
    if qwen3:
        from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRTextAttention
        expected_attention = Qwen3ASRTextAttention
    for attn in targets:
        if type(attn) is not expected_attention or attn.config._attn_implementation != 'eager':
            raise ValueError('TA-LoRA requires the matching native Thinker eager attention')

    # The trainer registers Template.pre_forward_hook on outer, converting IDs to
    # embeddings; the loader's use_submodel_func then bypasses thinker.__call__.
    # Scope masks to *this model and call*; always clear after template/model exceptions.
    frames = ContextVar('swift_ta_frames', default=())
    token_id = thinker.config.audio_token_id
    original_generate = outer.generate

    def layout(prompt, kwargs):
        result = _layout(prompt, kwargs, token_id)
        mode = getattr(outer, '_ta_lora_config', {}).get('transport_gate')
        if mode is None:
            return result
        embed = None if mode == 'static' else thinker.get_input_embeddings()
        return _answer_query(prompt, result[1], kwargs.get('labels'), token_id, embed)

    def start(module, args, kwargs):
        old = frames.get()
        frames.set(old + (None, ))  # always_call pops this frame even if validation/template raises.
        if not qwen3 and thinker.model.training and thinker.model.gradient_checkpointing:
            raise ValueError('disable gradient checkpointing: recomputation would run outside the TA layout scope')
        if module.training and getattr(outer, '_ta_lora_config', {}).get('transport_gate') and kwargs.get('labels') is None:
            raise ValueError('Answer-query training needs labels before embedding')
        prompt = kwargs.get('input_ids', args[0] if args else None)
        frames.set(old + (layout(prompt, kwargs), ))

    def finish(module, args, output):
        frames.set(frames.get()[:-1])

    @wraps(original_generate)
    def generate(*args, **kwargs):
        # GenerationMixin runs Thinker.__call__ repeatedly (and not outer.forward).
        # Keep the original prompt layout for every step; DynamicCache appends text slots.
        cfg = kwargs.get('generation_config') or thinker.generation_config
        beams = kwargs.get('num_beams', cfg.num_beams)
        returns = kwargs.get('num_return_sequences', cfg.num_return_sequences)
        if beams != 1 or returns != 1:
            raise ValueError('TA-LoRA requires num_beams=1 and num_return_sequences=1 (no beam expansion)')
        prompt = kwargs.get('input_ids', kwargs.get('inputs', args[0] if args else None))
        token = frames.set(frames.get() + (layout(prompt, kwargs), ))
        try:
            return original_generate(*args, **kwargs)
        finally:
            frames.reset(token)

    if qwen3:
        original_enable = outer.gradient_checkpointing_enable

        @contextmanager
        def replay(captured):
            token = frames.set(captured)
            try:
                yield
            finally:
                frames.reset(token)

        def contexts():
            return nullcontext(), replay(frames.get())

        @wraps(original_enable)
        def enable(gradient_checkpointing_kwargs=None):
            opts = dict(gradient_checkpointing_kwargs or {})
            if opts.get('use_reentrant') is not False:
                raise ValueError('TA modality replay requires non-reentrant checkpointing')
            return original_enable(dict(opts, context_fn=contexts))

        outer.gradient_checkpointing_enable = enable
    outer.requires_grad_(False)
    for attn in targets:
        attn.ta_adapter = _SharedAtoms(attn.o_proj, rank, alpha, format_version)
        attn._ta_frames = frames
        attn.forward = MethodType(_qwen3_attention if qwen3 else _attention, attn)
    outer.register_forward_pre_hook(start, prepend=True, with_kwargs=True)
    outer.register_forward_hook(finish, always_call=True)
    outer.generate = generate
    return outer
