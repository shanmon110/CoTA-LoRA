"""Position-restricted adapter updates (checkpoint format 10).

Run in the project environment before training with `adapter.update_positions`:

    python -m pytest -q experiments/ms_swift_librispeech/tests/test_update_positions.py
"""
import copy
import json
from types import SimpleNamespace

import pytest
import torch

from test_adapter_pair import TARGETS, tiny_omni
from test_conditional_gate import inputs

# Token 5 is audio. The two rows place the audio span differently.
PROMPTS = [[1, 5, 5, 6, 3, 8, 2], [1, 6, 5, 5, 5, 9, 2]]
FIRST_AUDIO, LAST_AUDIO = [1, 2], [2, 4]


def build(scope=None, seed=7):
    from experiments.ms_swift_librispeech.plugin import TALoRATuner
    from experiments.ms_swift_librispeech.swift_asr_ta_runtime import set_update_positions

    torch.manual_seed(seed)
    base = tiny_omni().eval()
    base.requires_grad_(False)
    model = copy.deepcopy(base)
    TALoRATuner.prepare_model(SimpleNamespace(target_modules=TARGETS, lora_rank=2, lora_alpha=4), model)
    with torch.no_grad():
        for layer in model.thinker.model.layers[-4:]:
            layer.self_attn.ta_adapter.up.normal_(std=.4)
            layer.self_attn.ta_adapter.gate.fill_(.5)
    if scope is not None:
        set_update_positions(model, scope)
    return base, model.eval()


def test_update_mask_follows_the_prompt_layout_and_cache_positions():
    from experiments.ms_swift_librispeech.swift_asr_ta_runtime import _update_mask

    audio = torch.tensor([[0, 1, 1, 0, 0], [0, 0, 1, 1, 0]], dtype=torch.bool)
    full = _update_mask(audio, None, 5, 5, 'audio')
    assert full.shape == (2, 5, 1) and full[..., 0].tolist() == audio.tolist()
    after = _update_mask(audio, None, 5, 5, 'post_audio')[..., 0]
    assert after.tolist() == [[False, False, False, True, True], [False, False, False, False, True]]
    # Cached decoding: a slot beyond the stored prompt is generated text.
    step = torch.tensor([5])
    assert not _update_mask(audio, step, 1, 6, 'audio').any()
    assert _update_mask(audio, step, 1, 6, 'post_audio').all()
    # A suffix of the prompt, with and without explicit cache positions.
    assert _update_mask(audio, torch.tensor([3, 4]), 2, 5, 'audio')[..., 0].tolist() == [[False, False], [True, False]]
    assert _update_mask(audio, None, 2, 5, 'post_audio')[..., 0].tolist() == [[True, True], [False, True]]
    silent = torch.zeros(1, 4, dtype=torch.bool)
    assert not _update_mask(silent, None, 4, 4, 'audio').any()
    assert _update_mask(silent, None, 4, 4, 'post_audio').all()
    with pytest.raises(ValueError):
        _update_mask(audio, torch.tensor([0, 1]), 1, 5, 'audio')
    with pytest.raises(ValueError):
        _update_mask(audio, None, 5, 5, 'everywhere')


def test_all_positions_is_the_unrestricted_adapter_and_keeps_its_format():
    _, free = build()
    _, explicit = build('all')
    record = inputs(PROMPTS)
    torch.testing.assert_close(explicit(**record).logits, free(**record).logits, atol=0, rtol=0)
    assert 'update_positions' not in explicit._ta_lora_config
    assert explicit._ta_lora_config['format_version'] == free._ta_lora_config['format_version'] == 2


def test_post_audio_updates_leave_the_audio_span_and_everything_before_it_frozen():
    base, model = build('post_audio')
    record = inputs(PROMPTS)
    expected, actual = base(**record).logits, model(**record).logits
    for row, last in enumerate(LAST_AUDIO):
        torch.testing.assert_close(actual[row, :last + 1], expected[row, :last + 1], atol=0, rtol=0)
        assert not torch.equal(actual[row, last + 1:], expected[row, last + 1:])


def test_audio_updates_start_at_the_first_audio_token_and_reach_later_text_through_attention():
    base, model = build('audio')
    record = inputs(PROMPTS)
    expected, actual = base(**record).logits, model(**record).logits
    for row, first in enumerate(FIRST_AUDIO):
        torch.testing.assert_close(actual[row, :first], expected[row, :first], atol=0, rtol=0)
        assert not torch.equal(actual[row, -1], expected[row, -1])  # the answer position still changes


def test_without_audio_the_audio_scope_is_the_frozen_base_and_post_audio_is_the_whole_adapter():
    # A transcript-only or question-only prompt: nothing for the audio scope to update.
    silent = [[1, 6, 7, 6, 3, 8, 2], [1, 6, 9, 7, 8, 9, 2]]
    record = inputs(silent)
    base, hear = build('audio')
    expected = base(**record).logits
    torch.testing.assert_close(hear(**record).logits, expected, atol=0, rtol=0)
    _, after = build('post_audio')
    _, free = build()
    actual = after(**record).logits
    torch.testing.assert_close(actual, free(**record).logits, atol=0, rtol=0)
    assert torch.isfinite(actual.float()).all() and not torch.equal(actual, expected)


def test_audio_updates_in_the_last_layer_cannot_reach_a_text_answer():
    # The last adapted layer writes only at audio positions, and nothing after it reads them.
    # So an audio-side adapter cannot learn an answer mapping there; earlier layers can still help.
    _, model = build('audio')
    for param in model.parameters():
        if param.requires_grad:
            param.grad = None
    logits = model(**inputs(PROMPTS)).logits
    logits[:, -1].float().square().sum().backward()
    layers = model.thinker.model.layers
    assert not layers[-1].self_attn.ta_adapter.up.grad.count_nonzero()
    assert all(layer.self_attn.ta_adapter.up.grad.abs().sum() > 0 for layer in layers[-4:-1])

    _, everywhere = build('post_audio')
    everywhere(**inputs(PROMPTS)).logits[:, -1].float().square().sum().backward()
    assert all(layer.self_attn.ta_adapter.up.grad.abs().sum() > 0 for layer in everywhere.thinker.model.layers[-4:])


def test_restricted_adapter_matches_a_shorter_prompt_at_shared_positions():
    # Causality is unchanged: a prefix gives the same logits as the full sequence at its positions.
    _, model = build('audio')
    record = inputs(PROMPTS)
    full = model(**record).logits
    prefix = {key: (value[:, :, :5] if key == 'position_ids' else value[:, :5]) if isinstance(value, torch.Tensor) else value
              for key, value in record.items()}
    torch.testing.assert_close(model(**prefix).logits, full[:, :5], atol=.07, rtol=.025)


def test_zero_multiplier_skips_the_contrast_without_changing_outputs_or_gradients(monkeypatch):
    from experiments.ms_swift_librispeech import swift_asr_ta_runtime as runtime

    record = inputs(PROMPTS, 5)

    def run(model):
        for param in model.parameters():
            if param.requires_grad:
                param.grad = None
        output = model.train()(**record)
        output.loss.backward()
        layers = model.thinker.model.layers[-4:]
        return output.logits.detach(), [(l.self_attn.ta_adapter.down.grad.clone(), l.self_attn.ta_adapter.up.grad.clone()) for l in layers]

    # Reference: the contrast is computed and multiplied by a raw gate of zero.
    _, computed = build()
    with torch.no_grad():
        for layer in computed.thinker.model.layers[-4:]:
            layer.self_attn.ta_adapter.gate.zero_()
    expected, expected_grads = run(computed)

    _, skipped = build()                              # raw gate .5, switched off by the multiplier
    runtime.set_gate_multiplier(skipped, 0.)
    monkeypatch.setattr(runtime, '_transport', lambda *a, **k: pytest.fail('the contrast was computed'))
    actual, actual_grads = run(skipped)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for (down, up), (ref_down, ref_up) in zip(actual_grads, expected_grads):
        torch.testing.assert_close(down, ref_down, atol=0, rtol=0)
        torch.testing.assert_close(up, ref_up, atol=0, rtol=0)
    assert all(layer.self_attn.ta_adapter.gate.grad is None for layer in skipped.thinker.model.layers[-4:])

    monkeypatch.undo()
    runtime.enable_transport_up(skipped)              # an independent C still needs the contrast
    called = []
    original = runtime._transport
    monkeypatch.setattr(runtime, '_transport', lambda *a, **k: called.append(1) or original(*a, **k))
    skipped.eval()(**inputs(PROMPTS))
    assert len(called) == 4


def _text_positions(input_ids, image_grid_thw=None, video_grid_thw=None, attention_mask=None, *args, **kwargs):
    """Rotary positions for prompts without vision inputs; the toy config has no vision token ids."""
    pos = attention_mask.long().cumsum(-1) - 1
    pos = pos.masked_fill(attention_mask == 0, 1)
    return pos.unsqueeze(0).expand(3, -1, -1), pos.max(-1, keepdim=True)[0] + 1 - attention_mask.sum(-1, keepdim=True)


def decoder(scope):
    from experiments.ms_swift_librispeech.plugin import TALoRATuner
    from experiments.ms_swift_librispeech.swift_asr_ta_runtime import set_gate_multiplier, set_update_positions

    torch.manual_seed(7)
    model = tiny_omni().eval()
    model.requires_grad_(False)
    model.generate = model.thinker.generate          # ms-swift routes generate to the Thinker in the same way
    model.thinker.get_rope_index = _text_positions
    TALoRATuner.prepare_model(SimpleNamespace(target_modules=TARGETS, lora_rank=2, lora_alpha=4), model)
    with torch.no_grad():
        for layer in model.thinker.model.layers[-4:]:
            layer.self_attn.ta_adapter.up.normal_(std=.4)
    set_gate_multiplier(model, 0.)                   # the trainer's order: multiplier first, then positions
    set_update_positions(model, scope)
    return model.eval()


def test_cached_generation_adds_the_update_at_the_same_positions_as_a_full_forward():
    # A left-padded batch, as the evaluators build it. Generated tokens are text: the audio scope
    # never updates them and the post-audio scope always does, also when they arrive one at a time.
    ids = torch.tensor([[0, 0, 1, 5, 5, 6, 3], [1, 6, 5, 5, 5, 9, 2]])
    mask, new = ids.ne(0).long(), 4
    models = {scope: decoder(scope) for scope in ('all', 'audio', 'post_audio')}
    assert models['audio']._ta_lora_config['format_version'] == 10 and models['all']._ta_lora_config['format_version'] == 4
    for scope in ('all', 'audio', 'post_audio'):
        with torch.inference_mode():
            out = models[scope].generate(input_ids=ids, attention_mask=mask, max_new_tokens=new, do_sample=False,
                                         num_beams=1, use_cache=True, eos_token_id=None, pad_token_id=0,
                                         output_logits=True, return_dict_in_generate=True)
            steps = torch.stack(out.logits, 1).float()
            record = dict(input_ids=out.sequences[:, :-1], use_cache=False,
                          attention_mask=torch.cat([mask, torch.ones(2, new - 1, dtype=mask.dtype)], 1))
            full = {name: model(**record).logits[:, ids.shape[1] - 1:].float() for name, model in models.items()}
        assert steps.shape[:2] == (2, new)
        torch.testing.assert_close(steps, full[scope], atol=.02, rtol=0)
        for other in set(models) - {scope}:          # an update at the wrong positions would show at every step
            assert (steps - full[other]).abs().amax(-1).min() > .05


@pytest.mark.parametrize('scope', ['audio', 'post_audio'])
def test_format_10_roundtrip_and_rejected_edits(tmp_path, scope):
    from experiments.ms_swift_librispeech.plugin import TALoRATuner
    from experiments.ms_swift_librispeech.swift_asr_ta_runtime import set_gate_multiplier

    base, model = build(scope)
    set_gate_multiplier(model, 0.)   # the pilot arms: a plain low-rank update, no transport
    record = inputs(PROMPTS)
    expected = model(**record).logits
    TALoRATuner.save_pretrained(model, tmp_path / 'saved')
    cfg = json.loads((tmp_path / 'saved' / 'adapter_config.json').read_text())
    assert cfg['format_version'] == 10 and cfg['update_positions'] == scope and cfg['gate_multiplier'] == 0.
    assert 'transport_gate' not in cfg and 'transport_signal' not in cfg

    restored = TALoRATuner.from_pretrained(copy.deepcopy(base), tmp_path / 'saved').eval()
    torch.testing.assert_close(restored(**record).logits, expected, atol=0, rtol=0)
    assert all(layer.self_attn.ta_adapter.update_positions == scope for layer in restored.thinker.model.layers[-4:])

    def reload(**edit):
        folder = tmp_path / ('edit-' + '-'.join(f'{k}={v}' for k, v in edit.items()))
        folder.mkdir()
        (folder / 'adapter_model.safetensors').write_bytes((tmp_path / 'saved' / 'adapter_model.safetensors').read_bytes())
        changed = {k: v for k, v in dict(cfg, **edit).items() if v is not None}
        (folder / 'adapter_config.json').write_text(json.dumps(changed))
        return TALoRATuner.from_pretrained(copy.deepcopy(base), folder)

    for edit in (dict(update_positions='all'), dict(update_positions='video'), dict(update_positions=None),
                 dict(format_version=7), dict(transport_gate='static'), dict(transport_signal='contrast')):
        with pytest.raises(ValueError, match='Invalid TA-LoRA checkpoint format'):
            reload(**edit)


def test_position_restriction_guards():
    from experiments.ms_swift_librispeech.swift_asr_ta_runtime import (enable_transport_up, set_transport_gate,
                                                                       set_update_positions)

    _, model = build('audio')
    with pytest.raises(ValueError, match='Cannot change'):
        set_update_positions(model, 'post_audio')
    with pytest.raises(ValueError, match='cannot be changed'):
        set_update_positions(model, 'all')
    with pytest.raises(ValueError, match='Answer-query transport'):
        set_transport_gate(model, 'static')
    with pytest.raises(ValueError, match='must be one of'):
        set_update_positions(model, 'video')

    _, conditional = build()
    enable_transport_up(conditional)
    set_transport_gate(conditional, 'question_audio')
    with pytest.raises(ValueError, match='Position-restricted'):
        set_update_positions(conditional, 'audio')
    set_update_positions(conditional, 'all')
    assert conditional._ta_lora_config['format_version'] == 8


def test_earlier_formats_reject_a_stray_position_key(tmp_path):
    from experiments.ms_swift_librispeech.plugin import TALoRATuner

    base, model = build()
    TALoRATuner.save_pretrained(model, tmp_path)
    cfg = json.loads((tmp_path / 'adapter_config.json').read_text())
    assert cfg['format_version'] == 2
    TALoRATuner.from_pretrained(copy.deepcopy(base), tmp_path)
    (tmp_path / 'adapter_config.json').write_text(json.dumps(dict(cfg, update_positions='audio')))
    with pytest.raises(ValueError, match='Invalid TA-LoRA checkpoint format'):
        TALoRATuner.from_pretrained(copy.deepcopy(base), tmp_path)
