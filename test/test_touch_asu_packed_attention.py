"""CPU arithmetic reference for the TouchASU FlashAttention transport branch."""
import importlib

import pytest
import torch
from transformers.models.qwen2.modeling_qwen2 import repeat_kv

from test_touch_asu_adapters import isolated_model
from west_adapters.ta_lora import _transport


@pytest.mark.parametrize("case", ["packed", "padded", "cached"])
@pytest.mark.parametrize("window", [None, 3])
def test_transport_and_gradient_match_full_logits(monkeypatch, tmp_path, case, window):
    with isolated_model(monkeypatch, tmp_path):
        adapter = importlib.import_module("west.models.touch_asu.adapter_touch_asu")
        torch.manual_seed(47)
        k = 14
        q = 2 if case == "cached" else k
        cache_position = torch.arange(k - q, k)
        positions = torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 0, 0, 0]]).expand(2, -1)
        audio = torch.tensor([[0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).bool()
        valid = torch.ones_like(audio)
        padding = None
        allowed = torch.ones(2, q, k, dtype=torch.bool)
        if case == "packed":
            segment = positions.eq(0).cumsum(-1)
            allowed &= segment[:, :, None].eq(segment[:, None, :])
        else:
            valid[0, :2] = False
            valid[1, -2:] = False
            padding = valid
            positions = None
        if window:
            allowed &= torch.arange(k) >= cache_position[:, None] - window
        mask = torch.zeros(allowed.shape).masked_fill_(~allowed, -torch.inf)[:, None]
        # The decode-time context covers the prompt. Cached generated tokens are text.
        prompt = k - 2 if case == "cached" else k
        layout = audio[:, :prompt], valid[:, :prompt]
        Q, K, V = [torch.randn(2, h, n, 4, requires_grad=True) for h, n in
                   [(4, q), (2, k), (2, k)]]
        scale = .5
        logits = (Q @ repeat_kv(K, 2).transpose(-1, -2)) * scale
        expected = _transport(logits, repeat_kv(V, 2), mask, cache_position, layout)
        actual = adapter._segmented_transport(Q, K, V, scale, layout, cache_position,
                                              positions, padding, window)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
        grad = torch.randn_like(actual)
        grads = torch.autograd.grad(actual, (Q, K, V), grad)
        reference_grads = torch.autograd.grad(expected, (Q, K, V), grad)
        for value, ref in zip(grads, reference_grads, strict=True):
            torch.testing.assert_close(value, ref, atol=2e-6, rtol=2e-6)


def test_long_pack_scores_stay_within_one_utterance(monkeypatch, tmp_path):
    with isolated_model(monkeypatch, tmp_path):
        adapter = importlib.import_module("west.models.touch_asu.adapter_touch_asu")
        seen = []

        def record_shape(logits, *args):
            seen.append(logits.shape[-2:])
            return _transport(logits, *args)

        monkeypatch.setattr(adapter, "_transport", record_shape)
        n, length = 8, 75
        positions = torch.arange(length).repeat(n)[None]
        layout = (positions.eq(1), torch.ones_like(positions, dtype=torch.bool))
        x = torch.randn(1, 2, n * length, 4)
        y = adapter._segmented_transport(x, x, x, .5, layout, None, positions, None)
        assert y.isfinite().all()
        assert max(q for q, _ in seen) == 32
        assert all(k == length for _, k in seen)


def _cpu_flash_reference(module, query, key, value, attention_mask, scaling,
                         dropout, position_ids=None, **kwargs):
    """Use torch eager for the FA2 kernel only, leaving HF's actual Qwen forward intact."""
    from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward
    q, k = query.size(2), key.size(2)
    slots = torch.arange(k - q, k)
    allowed = torch.arange(k) <= slots[:, None]
    allowed = allowed[None].expand(query.size(0), -1, -1)
    if attention_mask is not None:
        allowed = allowed & attention_mask[:, None, :].bool()
    elif q == k and position_ids is not None:
        segment = position_ids.eq(0).cumsum(-1)
        allowed = allowed & segment[:, :, None].eq(segment[:, None, :])
    additive = torch.zeros_like(allowed, dtype=query.dtype).masked_fill_(~allowed, torch.finfo(query.dtype).min)
    return eager_attention_forward(module, query, key, value, additive[:, None], scaling=scaling, dropout=dropout)


@pytest.mark.parametrize("real_cuda", [False, True], ids=["cpu-reference", "cuda"])
def test_flash_touchasu_packed_replay_and_decode(monkeypatch, tmp_path, real_cuda):
    import copy
    import os
    from transformers import AutoModelForCausalLM
    from test_touch_asu_adapters import speech_batch

    # Opt in explicitly on an idle GPU. The regular CPU suite never consumes GPU memory.
    if real_cuda and os.environ.get("WEST_TEST_FLASH_CUDA") != "1":
        pytest.skip("set WEST_TEST_FLASH_CUDA=1 to run native BF16 FlashAttention")
    with isolated_model(monkeypatch, tmp_path) as create:
        torch.manual_seed(713)
        adapter = importlib.import_module("west.models.touch_asu.adapter_touch_asu")
        dtype = torch.bfloat16 if real_cuda else torch.float32
        device = torch.device("cuda" if real_cuda else "cpu")
        teacher = create(dtype=dtype).to(device)
        config = copy.deepcopy(teacher.config)
        config.adapter_attention_backend = "flash_attention_2"
        if not real_cuda:
            original = AutoModelForCausalLM.from_pretrained

            def cpu_load(*args, **kwargs):
                assert kwargs.pop("attn_implementation") == "flash_attention_2"
                llm = original(*args, attn_implementation="eager", **kwargs)
                llm.config._attn_implementation = "flash_attention_2"
                return llm

            monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", cpu_load)
            monkeypatch.setattr(adapter, "flash_attention_forward", _cpu_flash_reference)
        student = type(teacher)._from_config(config).to(device)
        ordinary_config = copy.deepcopy(config)
        ordinary_config.lora_config["method"] = "o-lora"
        del ordinary_config.resolved_adapter_config
        ordinary = type(teacher)._from_config(ordinary_config).to(device)
        assert all(layer.self_attn.config._attn_implementation == "west_touch_asu_flash_adapter"
                   for layer in student.llm.model.layers)
        for layer in teacher.llm.model.layers:
            layer.self_attn.ta_adapter.up.data.fill_(.08)
            layer.self_attn.ta_adapter.gate.data.fill_(.6)
        student.load_state_dict(teacher.state_dict())
        student.gradient_checkpointing_enable()
        batch = {k: v.to(device) for k, v in speech_batch().items()}
        batch["audio_features"] = batch["audio_features"].to(dtype)
        batch["input_ids"] = torch.tensor([[1, 0, 0, 3, 2, 1, 0, 0, 4, 2, 0]], device=device)
        batch["position_ids"] = torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0]], device=device)
        batch["labels"] = batch["input_ids"].clone()
        batch["labels"][:, [0, 5, -1]] = -100
        batch["batch_idx"] = torch.tensor([0, 0], device=device)
        batch["audio_offsets"] = torch.tensor([1, 6], device=device)
        del batch["attention_mask"]
        ordinary.load_state_dict({k: v for k, v in student.state_dict().items()
                                  if not k.endswith("ta_adapter.gate")})
        for layer in student.llm.model.layers:
            layer.self_attn.ta_adapter.gate.data.zero_()
        ordinary.eval()
        student.eval()
        with torch.no_grad():
            torch.testing.assert_close(student(**batch, use_cache=False).logits,
                                       ordinary(**batch, use_cache=False).logits, atol=0, rtol=0)
        student.load_state_dict(teacher.state_dict())
        tolerances = dict(atol=0.007, rtol=.07) if real_cuda else dict(atol=2e-6, rtol=2e-5)
        for model in (teacher, student):
            model.train()
            first = model(**batch, use_cache=False)
            # Construct another graph with a different modal layout before backward.
            second = model(**dict(batch, has_audio=torch.tensor([1, 0], device=device)), use_cache=False)
            (first.loss + second.loss).backward()
        for a, b in zip(teacher.llm.model.layers, student.llm.model.layers, strict=True):
            ga, gb = a.self_attn.ta_adapter.gate.grad, b.self_attn.ta_adapter.gate.grad
            assert ga.isfinite().all() and gb.abs().sum() > 0
            torch.testing.assert_close(ga, gb, **tolerances)
        prompt = {k: v.to(device) for k, v in speech_batch().items()}
        prompt["audio_features"] = prompt["audio_features"].to(dtype)
        opts = dict(min_new_tokens=3, max_new_tokens=3, num_beams=2, do_sample=False)
        teacher.eval()
        student.eval()
        with torch.no_grad():
            actual = student.generate(**prompt, **opts)
            if real_cuda:
                assert actual.shape == (2, 3)
            else:
                assert torch.equal(actual, teacher.generate(**prompt, **opts))
