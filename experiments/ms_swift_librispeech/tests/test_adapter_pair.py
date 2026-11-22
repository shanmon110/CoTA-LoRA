"""Native PEFT versus integrated TA on actual miniature Qwen2.5-Omni attention."""
import copy
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen2_5OmniConfig, Qwen2_5OmniForConditionalGeneration

from experiments.ms_swift_librispeech.adapter_audit import audit_adapter, initialize_pair
from experiments.ms_swift_librispeech.swift_asr_ta_runtime import attach


def tiny_omni():
    config = Qwen2_5OmniConfig(
        thinker_config=dict(
            audio_config=dict(num_mel_bins=8, d_model=16, output_dim=16, encoder_layers=1,
                              encoder_attention_heads=2, encoder_ffn_dim=32, max_source_positions=16),
            vision_config=dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2,
                               out_hidden_size=16, fullatt_block_indexes=[0]),
            text_config=dict(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=5,
                             num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64,
                             rope_scaling={'rope_type': 'default', 'mrope_section': [1, 1, 2]}),
            audio_token_index=5, pad_token_id=0),
        enable_audio_output=False, attn_implementation='eager')
    outer = Qwen2_5OmniForConditionalGeneration(config).to(torch.bfloat16)
    outer.forward = outer.thinker.forward
    return outer


TARGETS = r'thinker\.model\.layers\.[1-4]\.self_attn\.o_proj'


def test_native_qv_only_adapter_uses_native_audit_but_ta_needs_o(tmp_path):
    from experiments.ms_swift_librispeech.plugin import _target_layout
    from experiments.ms_swift_librispeech.train import digest, model_options

    tiny = tiny_omni()
    tiny.config.to_json_file(tmp_path/'config.json')
    manifest = tmp_path/'snapshot.json'
    manifest.write_text(json.dumps({'revision':'mini', 'files':{'config.json':digest(tmp_path/'config.json')}}))
    cfg = dict(model=str(tmp_path), snapshot_manifest=str(manifest), model_revision='mini',
               adapter=dict(rank=2, alpha=4, last_n=1, target_modules=['v_proj','q_proj']))
    regex = model_options(cfg,'lora')['target_regex']
    names = [name for name,_ in tiny.named_modules() if re.fullmatch(regex,name)]
    assert names == ['thinker.model.layers.4.self_attn.'+key for key in ('q_proj','v_proj')]
    model = get_peft_model(tiny, LoraConfig(r=2,lora_alpha=4,target_modules=regex))
    assert initialize_pair(model,'lora',42,1,cfg['adapter']['target_modules'])['target_projections'] == ['q_proj','v_proj']
    with pytest.raises(ValueError,match='o_proj'):
        model_options(cfg,'ta-lora')
    with pytest.raises(ValueError,match='o_proj'):
        _target_layout(tiny,names)


def batch():
    ids = torch.tensor([[1, 5, 5, 6, 3]])
    return dict(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids, use_cache=False,
                position_ids=torch.arange(5).view(1, 1, 5).expand(3, 1, 5))


@pytest.mark.parametrize('amp', [False, True])
def test_paired_start_and_nonzero_weights_gate_off_matches_native_peft(amp):
    model = tiny_omni()
    ta = copy.deepcopy(model)
    baseline = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, target_modules=TARGETS))
    attach(ta, rank=2, alpha=4)
    ta._ta_lora_config = {}
    before = initialize_pair(baseline, 'lora', 25)
    after = initialize_pair(ta, 'ta_lora', 25)
    assert before['ab_init_sha256'] == after['ab_init_sha256']
    assert before['trainable_parameters'] + 4*2 == after['trainable_parameters']
    assert all('self_attn.' in name for name in before['trainable_names'] + after['trainable_names'])
    with torch.no_grad():
        for layer, mirror in zip(model.thinker.model.layers[-4:], ta.thinker.model.layers[-4:]):
            matrix = layer.self_attn.o_proj.lora_B.default.weight
            matrix.uniform_(-.2, .2)
            mirror.self_attn.ta_adapter.up.copy_(matrix)
    with torch.autocast('cpu', dtype=torch.bfloat16, enabled=amp):
        expected = baseline(**batch()).logits
        actual = ta(**batch()).logits
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.float().square().mean().backward()
    assert all(layer.self_attn.ta_adapter.gate.grad.abs().sum() > 0 for layer in ta.thinker.model.layers[-4:])
    ta.thinker.audio_tower.requires_grad_(True)
    with pytest.raises(ValueError, match='trainable'):
        audit_adapter(ta, 'ta_lora')


@pytest.mark.parametrize('version', [1, 2])
def test_ta_safetensor_roundtrip_preserves_nonzero_factors(tmp_path, version):
    from swift.tuner_plugin import tuners_map
    from swift.utils import import_external_file
    from safetensors.torch import save_file
    import json

    import_external_file(str(Path(__file__).resolve().parents[1] / 'plugin.py'))
    tuner = tuners_map['ta_lora']
    original = tiny_omni()
    restored = copy.deepcopy(original)
    if version == 2:
        tuner.prepare_model(SimpleNamespace(target_modules=TARGETS, lora_rank=2, lora_alpha=4), original)
        initialize_pair(original, 'ta_lora', 25)
    else:
        # Construct the actual format-v1 numerical path with nonzero, fixed weights.
        attach(original, rank=2, alpha=4, format_version=1)
    with torch.no_grad():
        for p in original.parameters():
            if p.requires_grad:
                p.uniform_(-.4,.4)
    if version == 2:
        tuner.save_pretrained(original, tmp_path)
    else:
        state = {n: p.detach() for n,p in original.named_parameters() if p.requires_grad}
        cfg = dict(format_version=1, tuner_type='ta_lora', rank=2, alpha=4, targets=[
            f'thinker.model.layers.{i}.self_attn.o_proj' for i in range(1,5)])
        save_file(state, str(tmp_path/'adapter_model.safetensors'))
        (tmp_path/'adapter_config.json').write_text(json.dumps(cfg))
    tuner.from_pretrained(restored, tmp_path)
    assert not any(param.requires_grad for param in restored.parameters())
    with torch.no_grad():
        torch.testing.assert_close(restored(**batch()).logits, original(**batch()).logits, atol=0, rtol=0)


def test_target_count_from_config_applies_to_restore(tmp_path):
    from experiments.ms_swift_librispeech.plugin import TALoRATuner
    m = tiny_omni()
    restored = copy.deepcopy(m)
    options = SimpleNamespace(target_modules=r'thinker\.model\.layers\.[34]\.self_attn\.o_proj',
                              lora_rank=2, lora_alpha=4)
    TALoRATuner.prepare_model(options, m)
    assert audit_adapter(m, 'ta_lora', last_n=2)['target_layers'] == [3,4]
    TALoRATuner.save_pretrained(m, tmp_path)
    TALoRATuner.from_pretrained(restored, tmp_path, is_trainable=True)
    assert audit_adapter(m, 'ta_lora', last_n=2) == audit_adapter(restored,'ta_lora',last_n=2)


def test_qkvo_all_layers_share_factors_and_reload_learned_gates(tmp_path):
    from experiments.ms_swift_librispeech.plugin import TALoRATuner

    projections = ('q_proj', 'k_proj', 'v_proj', 'o_proj')
    target = r'^thinker\.model\.layers\.[0-4]\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$'
    torch.manual_seed(37)
    base = tiny_omni()
    ta = copy.deepcopy(base)
    restored = copy.deepcopy(base)
    native = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, target_modules=target))
    TALoRATuner.prepare_model(SimpleNamespace(target_modules=target, lora_rank=2, lora_alpha=4), ta)
    ordinary = initialize_pair(native, 'lora', 42, 5, projections)
    paired = initialize_pair(ta, 'ta_lora', 42, 5, projections)
    assert ordinary['ab_init_sha256'] == paired['ab_init_sha256']
    assert ordinary['target_layers'] == paired['target_layers'] == list(range(5))
    assert (ordinary['trainable_parameters'], paired['trainable_parameters']) == (1120, 1130)
    with torch.no_grad():
        for left, right in zip(base.thinker.model.layers, ta.thinker.model.layers, strict=True):
            for name in projections:
                b = getattr(left.self_attn, name).lora_B.default.weight.uniform_(-0.2, 0.2)
                other = (right.self_attn.ta_adapter.up if name == 'o_proj'
                         else getattr(right.self_attn, name).lora_B.default.weight)
                other.copy_(b)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        logits = ta(**batch()).logits
        torch.testing.assert_close(logits, native(**batch()).logits, atol=0, rtol=0)
        logits.float().square().mean().backward()
    learned = [p for p in ta.parameters() if p.requires_grad]
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in learned)
    torch.optim.AdamW(learned, lr=0.05).step()
    TALoRATuner.save_pretrained(ta, tmp_path)
    TALoRATuner.from_pretrained(restored, tmp_path)
    assert not any(p.requires_grad for p in restored.parameters())
    with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16):
        trained = ta(**batch()).logits
        assert not torch.equal(trained, logits)
        torch.testing.assert_close(restored(**batch()).logits, trained, atol=0, rtol=0)


def test_gate_multiplier_preserves_zero_logits_scales_gradients_and_reloads(tmp_path):
    from experiments.ms_swift_librispeech.plugin import TALoRATuner
    from experiments.ms_swift_librispeech.swift_asr_ta_runtime import set_gate_multiplier

    torch.manual_seed(53)
    model = tiny_omni()
    restored = copy.deepcopy(model)
    TALoRATuner.prepare_model(SimpleNamespace(target_modules=TARGETS, lora_rank=2, lora_alpha=4), model)
    branches = [layer.self_attn.ta_adapter for layer in model.thinker.model.layers[-4:]]
    with torch.no_grad():
        for branch in branches:
            branch.up.uniform_(-.2, .2)
    zero_logits = model(**batch()).logits
    zero_logits.float().square().mean().backward()
    grads = [branch.gate.grad.clone() for branch in branches]
    assert all(g.abs().sum() > 0 for g in grads)
    model.zero_grad()
    set_gate_multiplier(model, 16.)
    scaled_zero_logits = model(**batch()).logits
    torch.testing.assert_close(scaled_zero_logits, zero_logits, rtol=0, atol=0)
    scaled_zero_logits.float().square().mean().backward()
    for branch, grad in zip(branches, grads, strict=True):
        torch.testing.assert_close(branch.gate.grad, 16 * grad, rtol=0, atol=0)
    with torch.no_grad():
        for branch in branches:
            branch.gate.fill_(.125)
        learned = model(**batch()).logits
        assert not torch.equal(learned, zero_logits)
    TALoRATuner.save_pretrained(model, tmp_path)
    TALoRATuner.from_pretrained(restored, tmp_path)
    with torch.no_grad():
        torch.testing.assert_close(restored(**batch()).logits, learned, atol=0, rtol=0)


def test_all_valid_transport_keeps_padding_causality_and_key_partition_guards():
    from experiments.ms_swift_librispeech.swift_asr_ta_runtime import _transport

    audio = torch.tensor([[False, False, True, True, False, False]])
    valid = torch.tensor([[False, True, True, True, True, False]])
    values = torch.tensor([1000., 10., 30., 70., 50., 900.]).view(1, 1, 6, 1)
    mask = torch.zeros(1, 1, 6, 6)
    mask[..., 4, 2] = -torch.inf
    args = (torch.zeros_like(mask), values, mask, torch.arange(6), (audio, valid))
    text = _transport(*args).flatten()
    all_valid = _transport(*args, transport_queries='all_valid').flatten()
    torch.testing.assert_close(text, torch.tensor([0., 0., 0., 0., 40., 0.]))
    torch.testing.assert_close(all_valid, torch.tensor([0., 0., 20., 40., 40., 0.]))
    cached = _transport(torch.zeros(1, 1, 1, 7), torch.cat([values, torch.tensor([[[[60.]]]])], dim=2),
                        None, torch.tensor([6]), (audio, valid), transport_queries='all_valid')
    torch.testing.assert_close(cached.flatten(), torch.tensor([10.]))
    no_text = _transport(*args[:-1], (valid, valid), transport_queries='all_valid')
    assert torch.equal(no_text, torch.zeros_like(no_text))


def test_query_scope_roundtrip_preserves_gate_multiplier_and_changes_audio_logits(tmp_path):
    from experiments.ms_swift_librispeech.plugin import TALoRATuner
    from experiments.ms_swift_librispeech.swift_asr_ta_runtime import set_gate_multiplier, set_transport_queries

    torch.manual_seed(99)
    model = tiny_omni()
    clean = copy.deepcopy(model)
    tuner = TALoRATuner()
    tuner.prepare_model(SimpleNamespace(target_modules=TARGETS, lora_rank=2, lora_alpha=4), model)
    set_gate_multiplier(model, 16.)
    with torch.no_grad():
        for layer in model.thinker.model.layers[-4:]:
            branch = layer.self_attn.ta_adapter
            branch.up.uniform_(-.2, .2)
            branch.gate.fill_(.125)
        text_only = model(**batch()).logits
        set_transport_queries(model, 'all_valid')
        all_valid = model(**batch()).logits
        assert not torch.equal(text_only[0, 1:3], all_valid[0, 1:3])
    tuner.save_pretrained(model, tmp_path)
    tuner.from_pretrained(clean, tmp_path)
    with torch.no_grad():
        torch.testing.assert_close(clean(**batch()).logits, all_valid, rtol=0, atol=0)
        set_transport_queries(clean, 'text')
        torch.testing.assert_close(clean(**batch()).logits, text_only, rtol=0, atol=0)


def test_independent_transport_up_learns_on_first_mixed_example_and_reloads(tmp_path):
    from experiments.ms_swift_librispeech.plugin import TALoRATuner
    from experiments.ms_swift_librispeech.swift_asr_ta_runtime import enable_transport_up

    torch.manual_seed(122)
    model = tiny_omni()
    clean = copy.deepcopy(model)
    TALoRATuner.prepare_model(SimpleNamespace(target_modules=TARGETS, lora_rank=2, lora_alpha=4), model)
    shared = initialize_pair(model, 'ta_lora', 42)
    with torch.no_grad():
        original = model(**batch()).logits
    enable_transport_up(model)
    audit = audit_adapter(model, 'ta_lora')
    assert audit['ab_sha256'] == shared['ab_init_sha256']
    assert audit['transport_up_parameters'] == audit['trainable_parameters'] - shared['trainable_parameters'] == 128
    logits = model(**batch()).logits
    torch.testing.assert_close(logits, original, atol=0, rtol=0)
    logits[0, -1].float().square().mean().backward()
    branches = [l.self_attn.ta_adapter for l in model.thinker.model.layers[-4:]]
    for b in branches:
        assert b.up.count_nonzero() == b.gate.count_nonzero() == b.gate.grad.count_nonzero() == 0
        assert torch.isfinite(b.transport_up.grad).all() and b.transport_up.grad.norm() > 0
        with torch.no_grad():
            b.transport_up.copy_(-b.transport_up.grad / b.transport_up.grad.norm())
    with torch.no_grad():
        learned = model(**batch()).logits
        assert not torch.equal(learned, original)
        torch.testing.assert_close(learned[:, :3], original[:, :3], atol=0, rtol=0)
    TALoRATuner.save_pretrained(model, tmp_path)
    TALoRATuner.from_pretrained(clean, tmp_path)
    assert not any(p.requires_grad for p in clean.parameters())
    with torch.no_grad():
        torch.testing.assert_close(clean(**batch()).logits, learned, rtol=0, atol=0)


def test_attention_overlap_transport_uses_causal_modality_mass_and_has_finite_gradients():
    from experiments.ms_swift_librispeech.swift_asr_ta_runtime import _transport

    # Four distinct heads: balanced, negligible audio, negligible text, asymmetric.
    logits = torch.tensor([0., -40., 40., 2.]).view(1, 4, 1, 1)
    logits = torch.cat([torch.zeros_like(logits), logits, torch.zeros_like(logits)], -1).requires_grad_()
    values = torch.tensor([1., 4., 9.]).view(1, 1, 3, 1).expand(1, 4, 3, 1).clone().requires_grad_()
    layout = (torch.tensor([[False, True, False]]), torch.tensor([[True, True, False]]))
    mask = torch.tensor([[[[0., 0., -torch.inf]]]])
    joint = (logits + mask).softmax(-1)
    args = logits, values, mask, torch.tensor([1]), layout, 'all_valid'
    result = _transport(*args, joint_probabilities=joint)
    expected = 12 * joint[..., 0] * joint[..., 1]
    torch.testing.assert_close(result.flatten(), expected.flatten(), rtol=1e-5, atol=1e-20)
    assert result.flatten()[0] == _transport(*args).flatten()[0] == 3.
    assert result.flatten()[1:3].abs().max() < 1e-14
    result.sum().backward()
    assert torch.isfinite(logits.grad).all() and torch.isfinite(values.grad).all()
    for audio, valid in ((torch.zeros(1,3).bool(), layout[1]), (layout[1], layout[1]),
                         (layout[0], torch.zeros(1,3).bool())):
        empty = _transport(logits, values, mask, torch.tensor([1]), (audio, valid),
                           'all_valid', joint_probabilities=joint)
        assert torch.isfinite(empty).all() and empty.count_nonzero() == 0


@pytest.mark.parametrize('independent', [False, True])
def test_attention_overlap_checkpoint_roundtrip_and_legacy_default(tmp_path, independent):
    from experiments.ms_swift_librispeech.plugin import TALoRATuner
    from experiments.ms_swift_librispeech.swift_asr_ta_runtime import enable_transport_up, set_transport_weighting
    import json

    torch.manual_seed(119)
    model = tiny_omni().eval()
    legacy_model, overlap_model = copy.deepcopy(model), copy.deepcopy(model)
    TALoRATuner.prepare_model(SimpleNamespace(target_modules=TARGETS, lora_rank=2, lora_alpha=4), model)
    if independent:
        enable_transport_up(model)
    with torch.no_grad():
        for layer in model.thinker.model.layers[-4:]:
            a = layer.self_attn.ta_adapter
            a.up.normal_(std=.4)
            a.gate.fill_(1.)
            if a.transport_up is not None:
                a.transport_up.normal_(std=.3)
    legacy, weighted = tmp_path/'legacy', tmp_path/'weighted'
    TALoRATuner.save_pretrained(model, legacy)
    assert 'transport_weighting' not in json.loads((legacy/'adapter_config.json').read_text())
    original = model(**batch()).logits
    TALoRATuner.from_pretrained(legacy_model, legacy)
    torch.testing.assert_close(legacy_model(**batch()).logits, original, atol=0, rtol=0)
    set_transport_weighting(model, 'attention_overlap')
    changed = model(**batch()).logits
    assert not torch.equal(changed, original)
    model.zero_grad()
    changed.float().square().sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
    TALoRATuner.save_pretrained(model, weighted)
    config = json.loads((weighted/'adapter_config.json').read_text())
    assert config['format_version'] == 7 and config['transport_weighting'] == 'attention_overlap'
    TALoRATuner.from_pretrained(overlap_model, weighted)
    torch.testing.assert_close(overlap_model(**batch()).logits, changed, atol=0, rtol=0)
    set_transport_weighting(overlap_model, 'uniform')
    torch.testing.assert_close(overlap_model(**batch()).logits, original, atol=0, rtol=0)


def test_attention_overlap_matches_conditional_means_for_padded_and_cached_queries():
    from experiments.ms_swift_librispeech.swift_asr_ta_runtime import _transport

    torch.manual_seed(905)
    # The two positions beyond the stored prompt are generated text.
    audio = torch.tensor([[0, 0, 1, 1, 0], [1, 1, 0, 0, 0]], dtype=torch.bool)
    valid = torch.tensor([[0, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool)
    key_audio = torch.nn.functional.pad(audio, (0, 2), value=False)
    key_valid = torch.nn.functional.pad(valid, (0, 2), value=True)
    position = torch.arange(7)
    logits = torch.randn(2, 3, 7, 7, requires_grad=True)
    value = torch.randn(2, 3, 7, 4, requires_grad=True)
    visible = key_valid[:, None, None, :] & (position[None, :] <= position[:, None])
    mask = torch.zeros_like(visible, dtype=logits.dtype).masked_fill(~visible, torch.finfo(logits.dtype).min)
    joint = (logits + mask).softmax(-1)
    old = _transport(logits, value, mask, position, (audio, valid), 'all_valid').reshape(2, 7, 3, 4)
    new = _transport(logits, value, mask, position, (audio, valid), 'all_valid', joint).reshape(2, 7, 3, 4)
    p_audio = (joint * (visible & key_audio[:, None, None, :])).sum(-1)
    p_text = (joint * (visible & ~key_audio[:, None, None, :])).sum(-1)
    reference = old * (4*p_audio*p_text).transpose(1, 2).unsqueeze(-1)
    torch.testing.assert_close(new, reference)
    for a,b in zip(torch.autograd.grad(new.sum(), (logits,value), retain_graph=True),
                   torch.autograd.grad(reference.sum(), (logits,value))):
        torch.testing.assert_close(a, b, atol=3e-6, rtol=1e-4)
    final = _transport(logits[:, :, -1:], value, mask[:, :, -1:], position[-1:], (audio,valid),
                       'all_valid', joint[:, :, -1:])
    torch.testing.assert_close(final, new[:, -1:].reshape(2,1,-1))


def test_evaluation_transport_ablation_zeros_only_transport_and_preserves_checkpoint(tmp_path):
    from experiments.ms_swift_aqa.transfer_eval import disable_transport_for_eval
    from experiments.ms_swift_librispeech.plugin import TALoRATuner
    from experiments.ms_swift_librispeech.swift_asr_ta_runtime import enable_transport_up
    from experiments.ms_swift_librispeech.train import digest

    torch.manual_seed(18)
    model = tiny_omni()
    loaded = copy.deepcopy(model)
    opts = SimpleNamespace(target_modules=TARGETS, lora_rank=2, lora_alpha=4)
    TALoRATuner.prepare_model(opts, model)
    enable_transport_up(model)
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.normal_(std=.3)
    TALoRATuner.save_pretrained(model, tmp_path)
    original_hash = digest(tmp_path/'adapter_model.safetensors')
    TALoRATuner.from_pretrained(loaded, tmp_path)
    loaded.eval()
    branches = [l.self_attn.ta_adapter for l in loaded.thinker.model.layers[-4:]]
    intact = loaded(**batch()).logits
    handles = [b.register_forward_pre_hook(lambda m, args: (args[0], torch.zeros_like(args[1]))) for b in branches]
    expected = loaded(**batch()).logits
    for h in handles:
        h.remove()
    before = {n: p.detach().clone() for n,p in loaded.named_parameters()}
    evidence = disable_transport_for_eval(loaded)
    assert evidence['gate_tensors'] == evidence['transport_up_tensors'] == 4
    assert evidence['tensors_zeroed'] == 8
    for n,p in loaded.named_parameters():
        if n.endswith(('ta_adapter.gate','ta_adapter.transport_up')):
            assert not p.count_nonzero()
        else:
            assert torch.equal(before[n], p)
    assert not torch.equal(intact, expected)
    torch.testing.assert_close(loaded(**batch()).logits, expected, atol=0, rtol=0)
    assert original_hash == digest(tmp_path/'adapter_model.safetensors')
