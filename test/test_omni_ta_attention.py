"""CPU contract tests against actual Transformers 4.52.3 Thinker attention."""

import math

import pytest
import torch
from torch import nn
from transformers.cache_utils import DynamicCache
from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import Qwen2_5OmniTextConfig
from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniAttention

from examples.omni_talora.attention import audio_context, install_adapter


def tiny_thinker(n_layers=1, dtype=torch.float32):
    cfg = Qwen2_5OmniTextConfig(
        hidden_size=12, num_attention_heads=2, num_key_value_heads=1,
        num_hidden_layers=n_layers, rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 1]},
    )
    cfg._attn_implementation = "eager"
    thinker = nn.Module()
    thinker.model = nn.Module()
    thinker.model.layers = nn.ModuleList()
    for i in range(n_layers):
        layer = nn.Module()
        layer.self_attn = Qwen2_5OmniAttention(cfg, layer_idx=i)
        thinker.model.layers.append(layer)
    return thinker.to(dtype).eval()


def run_attn(attn, x, valid, *, start=0, cache=None, identity_rope=False, hide_key=None):
    b, q, _ = x.shape
    pos = torch.arange(start, start + q)
    phase = torch.zeros((3, b, q, 6), dtype=x.dtype)
    if not identity_rope:
        phase = (pos[None, None, :, None] + 1) * torch.tensor([0.4, 0.1, 0.2], dtype=x.dtype)[:, None, None, None]
        phase = phase.expand(3, b, q, 6)
    visible = (torch.arange(valid.size(1)) <= pos[:, None]) & valid[:, None, :].bool()
    if hide_key is not None:
        visible[:, :, hide_key] = False
    mask = torch.zeros(visible.shape, dtype=x.dtype).masked_fill(~visible, torch.finfo(x.dtype).min)
    return attn(x, attention_mask=mask[:, None], cache_position=pos, position_embeddings=(phase.cos(), phase.sin()),
                past_key_value=cache, output_attentions=True)


@pytest.mark.parametrize("mode", ["ta", "lora"])
def test_only_last_layers_train_and_zero_initialization_preserves_eager_attention(mode):
    thinker = tiny_thinker(3)
    attn = thinker.model.layers[-1].self_attn
    x = torch.randn(1, 5, 12)
    ids, valid = torch.tensor([[0, 1, 99, 99, 2]]), torch.tensor([[0, 1, 1, 1, 1]])
    original_names = set(dict(thinker.named_parameters()))
    base, probs, _ = run_attn(attn, x, valid)
    install_adapter(thinker, last_n=2, rank=2, alpha=3.0, mode=mode)
    trainable = {name for name, p in thinker.named_parameters() if p.requires_grad}
    expected_suffixes = ["down", "up", "gate"] if mode == "ta" else ["down", "up"]
    assert trainable == {f"model.layers.{i}.self_attn.ta_adapter.{n}" for i in [1, 2] for n in expected_suffixes}
    assert set(dict(thinker.named_parameters())) == original_names | trainable
    with audio_context(ids, valid, 99):
        actual, actual_probs, _ = run_attn(attn, x, valid)
    torch.testing.assert_close(actual, base, rtol=0, atol=0)
    torch.testing.assert_close(actual_probs, probs, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["ta", "lora"])
def test_zero_gate_matches_nonzero_ordinary_o_lora(mode):
    thinker = tiny_thinker()
    attn = thinker.model.layers[0].self_attn
    ids, valid = torch.tensor([[3, 99, 4, 5]]), torch.ones(1, 4)
    x = torch.randn(1, 4, 12)
    pre_o = []
    hook = attn.o_proj.register_forward_pre_hook(lambda mod, args: pre_o.append(args[0].detach()))
    base = run_attn(attn, x, valid)[0]
    hook.remove()
    install_adapter(thinker, last_n=1, rank=2, alpha=3, mode=mode)
    a = attn.ta_adapter
    with torch.no_grad():
        a.up.normal_()
    want = base + 1.5 * (pre_o[0] @ a.down.T @ a.up.T)
    with audio_context(ids, valid, 99):
        actual = run_attn(attn, x, valid)[0]
    torch.testing.assert_close(actual, want)


@pytest.mark.parametrize("hide_key,expected", [(None, 79.0), (3, 64.0)])
def test_conditional_transport_is_stable_and_obeys_visibility(hide_key, expected):
    # At position 4, text values {3, 9} have equal logits; audio {70, 90}
    # has odds 1:3 but global audio probability rounds to zero (gap 256).
    # Masking key 3 in the decoder leaves audio value 70. Other queries:
    # padding, no past audio, audio, or padding. Row 1 has no audio at all.
    thinker = tiny_thinker()
    attn = thinker.model.layers[0].self_attn
    ids = torch.tensor([[0, 7, 99, 99, 8, 0, 99], [0, 7, 6, 5, 8, 0, 6]])
    valid = torch.tensor([[0, 1, 1, 1, 1, 0, 1]]).expand_as(ids)
    x = torch.zeros(2, 7, 12)
    x[:, :, 0] = torch.tensor([999., 3., 70., 90., 9., 999., -500.])
    x[:, :, 1] = torch.tensor([128., 128., -128., -128. + math.log(3), 128., 128., 500.])
    with torch.no_grad():
        for p in attn.parameters():
            p.zero_()
        attn.q_proj.bias[[0, 6]] = math.sqrt(6)
        attn.k_proj.weight[0, 1] = attn.v_proj.weight[0, 0] = 1
    install_adapter(thinker, last_n=1, rank=1, alpha=1)
    a = attn.ta_adapter
    with torch.no_grad(), audio_context(ids, valid, 99):
        a.down.zero_()
        a.down[0, 6] = a.up[0, 0] = 1
        ordinary = run_attn(attn, x, valid, identity_rope=True, hide_key=hide_key)[0]
        a.gate.fill_(1)
        actual = run_attn(attn, x, valid, identity_rope=True, hide_key=hide_key)[0]
    want = torch.zeros_like(actual)
    want[0, 4, 0] = expected
    torch.testing.assert_close(actual - ordinary, want, atol=1e-4, rtol=1e-5)


def test_cached_decode_matches_full_sequence_with_left_padded_prompt():
    thinker = tiny_thinker()
    attn = thinker.model.layers[0].self_attn
    install_adapter(thinker, last_n=1, rank=3, alpha=4.)
    with torch.no_grad():
        attn.ta_adapter.up.normal_()
        attn.ta_adapter.gate.fill_(1.)
    ids = torch.tensor([[0, 3, 99, 6, 7, 8], [1, 99, 99, 4, 5, 6]])
    valid = ids.ne(0)
    x = torch.randn(2, 6, 12)
    with audio_context(ids, valid, 99):
        full = run_attn(attn, x, valid)[0]
    # Generation sees prompt IDs only. The two tokens appended to DynamicCache
    # must be valid text; reuse the actual upstream cache across all three calls.
    cache = DynamicCache()
    with audio_context(ids[:, :4], valid[:, :4], 99):
        prefill = run_attn(attn, x[:, :4], valid[:, :4], cache=cache)[0]
        next_1 = run_attn(attn, x[:, 4:5], valid[:, :5], start=4, cache=cache)[0]
        next_2 = run_attn(attn, x[:, 5:6], valid, start=5, cache=cache)[0]
    # For a *padding query* with no visible keys, native eager attention
    # softmaxes equal finite minima to 1/K: it changes when total K changes.
    torch.testing.assert_close(torch.cat((prefill, next_1, next_2), dim=1)[valid], full[valid])


def test_bfloat16_base_fp32_master_gradients_and_context_cleanup():
    thinker = tiny_thinker(dtype=torch.bfloat16)
    attn = thinker.model.layers[0].self_attn
    install_adapter(thinker, last_n=1, rank=2, alpha=2)
    a = attn.ta_adapter
    assert all(p.dtype == torch.float32 for p in a.parameters())
    ids = torch.tensor([[0, 11, 99, 12]])
    x = torch.randn(1, 4, 12, dtype=torch.bfloat16, requires_grad=True)
    with torch.no_grad():
        a.up.normal_()
    with audio_context(ids, ids.ne(0), 99):
        out = run_attn(attn, x, ids.ne(0))[0]
        assert out.dtype == torch.bfloat16
        out[:, -1].float().sum().backward()
    assert all(p.grad.dtype == torch.float32 and p.grad.isfinite().all() and p.grad.abs().sum() > 0
               for p in a.parameters())
    assert x.grad.isfinite().all()
    with pytest.raises(RuntimeError, match="audio_context"):
        run_attn(attn, x, ids.ne(0))


def test_invalid_injection_is_not_a_partial_mutation():
    thinker = tiny_thinker()
    with pytest.raises(ValueError):
        install_adapter(thinker, last_n=2)
    assert all(p.requires_grad for p in thinker.parameters())
    thinker.model.layers[0].self_attn.config._attn_implementation = "sdpa"
    with pytest.raises(ValueError, match="eager"):
        install_adapter(thinker, last_n=1)
    assert all(p.requires_grad for p in thinker.parameters())
