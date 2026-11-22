"""Bridge Qwen text attention and WEST's projected speech spans to the adapter registry."""
import copy
import random
from types import MethodType

import torch
from torch.nn import functional as F
from transformers import AttentionInterface
from transformers.cache_utils import DynamicCache
from transformers.integrations.flash_attention import flash_attention_forward
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, repeat_kv
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

from west_adapters.base import add_fp32_residual
from west_adapters.context import current_audio_layout
from west_adapters.registry import checkpoint_method
from west_adapters.ta_lora import _transport


def _eager_adapter_attention(module, query, key, value, attention_mask,
                             scaling, dropout=0., adapter_cache_position=None, adapter_layout=None, **kwargs):
    # HF has already applied Qwen3's Q/K norms, rotary embeddings and cache
    # update. Keep the native GQA scaling and use the same logits for transport.
    value = repeat_kv(value, module.num_key_value_groups)
    key = repeat_kv(key, module.num_key_value_groups)
    logits = (query @ key.transpose(-2, -1)) * scaling
    mask = None
    if attention_mask is not None:
        mask = attention_mask[:, :, :, :key.size(-2)]
        logits = logits + mask
    weights = F.softmax(logits, dim=-1, dtype=torch.float32).to(query.dtype)
    weights = F.dropout(weights, p=dropout, training=module.training)
    output = (weights @ value).transpose(1, 2).contiguous()
    delta = module.ta_adapter.attention_delta(output.flatten(2), logits, value, mask,
                                              adapter_cache_position, adapter_layout)
    return output, (weights, delta)


AttentionInterface.register("west_touch_asu_adapter", _eager_adapter_attention)


def _segmented_transport(query, key, value, scaling, layout, cache_position,
                         position_ids, attention_mask, sliding_window=None):
    """Compute conditional attention only for text queries in their own utterance.

    Each score operation spans at most 32 query rows, including for one long utterance.
    The host-side index metadata is linear in input length. The conditional
    softmax and value reduction share the eager reference's FP32 arithmetic.
    """
    batch, heads, qlen, width = query.shape
    klen = key.size(2)
    device = query.device
    audio, valid = layout
    if audio.size(0) != batch:
        raise ValueError("TouchASU transport layout and attention batch must match")
    audio, valid = audio[:, :klen], valid[:, :klen]
    if audio.size(1) < klen:
        extension = (0, klen - audio.size(1))
        audio = F.pad(audio, extension, value=False)
        valid = F.pad(valid, extension, value=True)
    if attention_mask is not None:
        valid = valid & attention_mask[:, :klen].bool()
    if cache_position is None:
        cache_position = torch.arange(klen - qlen, klen, device=device)
    if cache_position.shape != (qlen,):
        raise ValueError("expected one absolute cache position per query")

    # Training packs restart position_ids at zero and pass no padding mask.
    # Generated tokens use absolute DynamicCache slots even with left padding.
    positions = cache_position.cpu().long()
    modality = (audio & valid).cpu()
    text_query = (valid & ~audio).cpu().index_select(1, positions)
    packed = None
    if position_ids is not None and attention_mask is None and qlen == klen:
        possible_pack = position_ids.detach().expand(batch, -1).cpu()
        if (possible_pack.diff(dim=-1) < 0).any():
            packed = possible_pack

    query_indices, contrasts = [], []
    groups = heads // key.size(1)
    for row in range(batch):
        starts = [0] if packed is None else packed[row].eq(0).nonzero().flatten().tolist()
        for start, stop in zip(starts, starts[1:] + [klen]):
            if stop - start < 2:
                continue
            audio_positions = modality[row, start:stop].nonzero().flatten()
            if not len(audio_positions):
                continue
            candidates = (text_query[row] & (positions > start + audio_positions[0])
                          & (positions < stop)).nonzero().flatten()
            if not len(candidates):
                continue
            # GQA expands K/V within an utterance, independent of pack length.
            keys = repeat_kv(key[row:row + 1, :, start:stop], groups)
            values = repeat_kv(value[row:row + 1, :, start:stop], groups)
            members = (audio[row:row + 1, start:stop], valid[row:row + 1, start:stop])
            for indices in candidates.split(32):
                indices = indices.to(device)
                local_pos = cache_position.index_select(0, indices) - start
                queries = query[row:row + 1].index_select(2, indices)
                logits = (queries @ keys.transpose(-1, -2)) * scaling
                mask = None
                if sliding_window is not None:
                    # FlashAttention window_size=(w, w) includes key query_pos-w.
                    excluded = torch.arange(stop - start, device=device) < local_pos[:, None] - sliding_window
                    mask = torch.zeros_like(excluded, dtype=logits.dtype).masked_fill_(excluded, -torch.inf)[None, None]
                contrasts.append(_transport(logits, values, mask, local_pos, members).squeeze(0))
                query_indices.append(indices + row * qlen)
    result = query.new_zeros((batch * qlen, heads * width), dtype=torch.float32)
    if contrasts:
        result.index_copy_(0, torch.cat(query_indices), torch.cat(contrasts))
    return result.view(batch, qlen, heads * width)


def _flash_adapter_attention(module, query, key, value, attention_mask, scaling,
                             dropout=0., adapter_cache_position=None, adapter_layout=None,
                             sliding_window=None, **kwargs):
    output, _ = flash_attention_forward(module, query, key, value, attention_mask,
                                        scaling=scaling, dropout=dropout, sliding_window=sliding_window,
                                        **kwargs)
    contrast = None
    if module._adapter_needs_audio_context:
        contrast = _segmented_transport(query, key, value, scaling, adapter_layout,
                                        adapter_cache_position, kwargs.get("position_ids"), attention_mask,
                                        sliding_window)
    delta = module.ta_adapter(output.flatten(2), contrast)
    return output, (None, delta)


AttentionInterface.register("west_touch_asu_flash_adapter", _flash_adapter_attention)


def _adapter_forward(self, hidden_states, position_embeddings, attention_mask, past_key_value=None,
                     cache_position=None, **kwargs):
    if past_key_value is not None and not isinstance(past_key_value, DynamicCache):
        raise ValueError("TouchASU adapted attention requires append-only DynamicCache")
    layout = current_audio_layout()
    if self._adapter_needs_audio_context and layout is None:
        raise ValueError("TouchASU TA-LoRA requires a projected audio layout")
    base, (weights, delta) = type(self).forward(
        self, hidden_states, position_embeddings, attention_mask, past_key_value, cache_position,
        adapter_cache_position=cache_position, adapter_layout=layout, **kwargs)
    return add_fp32_residual(base, delta), weights


def install_attention(llm, *, last_n, rank, alpha, mode, residual_precision):
    method = checkpoint_method({"mode": mode, "target_set": "o"})
    selected = [layer.self_attn for layer in llm.model.layers[-last_n:]]
    if any(type(m) not in (Qwen2Attention, Qwen3Attention) for m in selected):
        raise ValueError("TouchASU O/TA-LoRA supports Qwen2 and Qwen3 text attention")
    llm.requires_grad_(False)
    for module in selected:
        module._adapter_needs_audio_context = method.needs_audio_context
        module.ta_adapter = method.atoms(module.o_proj, rank, alpha, residual_precision)
        module.config = copy.copy(module.config)
        module.config._attn_implementation = ("west_touch_asu_flash_adapter"
                                              if llm.config._attn_implementation == "flash_attention_2"
                                              else "west_touch_asu_adapter")
        module.forward = MethodType(_adapter_forward, module)
    return selected


def packed_attention_mask(position_ids, dtype, sliding_window=None):
    """WEST packs utterances into one row and restarts positions per utterance."""
    segment = position_ids.eq(0).long().cumsum(-1)
    allowed = segment[:, :, None].eq(segment[:, None, :]).tril()
    if sliding_window is not None:
        allowed = allowed.triu(1 - sliding_window)
    mask = torch.zeros_like(allowed, dtype=dtype)
    return mask.masked_fill_(~allowed, torch.finfo(dtype).min)[:, None]


def calibration_batches(dataset, options):
    """Read a finite, seeded sample from the current WEST training manifest."""
    # Use the native JSONL/shard reader for exactly one pass. The extractor
    # retains its training filters and labels, independent of the reader flag.
    reader = copy.copy(dataset)
    reader.inference = True
    reader.data_lists = list(dataset.data_lists)
    random.Random(options["eva_seed"]).shuffle(reader.data_lists)
    observed = 0
    for raw in reader._read_one():
        item = reader.extractor.extract(raw)
        if item is not None:
            yield reader._batch([item])
            observed += 1
            if observed == options["eva_samples"]:
                return
    if not observed:
        raise ValueError("No eligible training utterances for EVA calibration")
