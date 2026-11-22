"""Answer-query controls, FP32 transport and single-factor ablations (checkpoint format 9).

Written on research/claude-review without a PyTorch runtime: these tests have NOT been executed by
their author. Run them in the project environment before relying on the new options:

    python -m pytest -q experiments/ms_swift_librispeech/tests/test_answer_controls.py
"""
import copy
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from test_adapter_pair import TARGETS, tiny_omni
from test_conditional_gate import inputs

PROMPTS = [[1, 5, 5, 6, 3, 8, 2], [1, 5, 5, 7, 3, 9, 2]]


def configure(model, mode='question_audio', *, independent=True, signal=None, precision=None):
    from experiments.ms_swift_librispeech.plugin import TALoRATuner
    from experiments.ms_swift_librispeech.swift_asr_ta_runtime import (
        enable_transport_up, set_transport_gate, set_transport_precision, set_transport_signal)

    TALoRATuner.prepare_model(SimpleNamespace(target_modules=TARGETS, lora_rank=2, lora_alpha=4), model)
    if independent:
        enable_transport_up(model)
    set_transport_gate(model, mode)
    if signal is not None:
        set_transport_signal(model, signal)
    if precision is not None:
        set_transport_precision(model, precision)


def randomize(model):
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad:
                param.uniform_(-1, 1)


def test_default_answer_query_configuration_stays_format_8():
    from experiments.ms_swift_librispeech.swift_asr_ta_runtime import set_transport_precision, set_transport_signal

    model = tiny_omni()
    configure(model)
    cfg = model._ta_lora_config
    set_transport_signal(model, 'contrast')
    set_transport_precision(model, 'autocast')
    assert cfg['format_version'] == 8
    assert 'transport_signal' not in cfg and 'transport_precision' not in cfg
    branch = model.thinker.model.layers[-1].self_attn.ta_adapter
    assert (branch.transport_signal, branch.transport_precision) == ('contrast', 'autocast')


def test_answer_options_need_answer_query_transport_and_known_values():
    from experiments.ms_swift_librispeech.plugin import TALoRATuner
    from experiments.ms_swift_librispeech.swift_asr_ta_runtime import set_transport_precision, set_transport_signal

    model = tiny_omni()
    TALoRATuner.prepare_model(SimpleNamespace(target_modules=TARGETS, lora_rank=2, lora_alpha=4), model)
    with pytest.raises(ValueError, match='answer-query'):
        set_transport_signal(model, 'ordinary')
    configured = tiny_omni()
    configure(configured)
    with pytest.raises(ValueError, match='transport_signal'):
        set_transport_signal(configured, 'video')
    with pytest.raises(ValueError, match='transport_precision'):
        set_transport_precision(configured, 'fp16')


@pytest.mark.parametrize('signal,precision,independent', [
    ('audio_mean', 'autocast', True),
    ('ordinary', 'autocast', True),
    ('ordinary_matched', 'autocast', True),
    ('contrast', 'fp32', True),
    ('contrast', 'autocast', False),
    ('ordinary', 'fp32', False),
])
def test_controls_are_zero_updates_at_init_and_reload_exactly(tmp_path, signal, precision, independent):
    from experiments.ms_swift_librispeech.plugin import TALoRATuner

    model = tiny_omni()
    baseline = copy.deepcopy(model)
    baseline.requires_grad_(False)
    restored = copy.deepcopy(model)
    configure(model, 'question_audio', independent=independent, signal=signal, precision=precision)
    record = inputs(PROMPTS, 5)
    torch.testing.assert_close(model(**record).logits, baseline(**record).logits, atol=0, rtol=0)
    model.eval()
    randomize(model)
    TALoRATuner.save_pretrained(model, tmp_path)
    metadata = json.loads((tmp_path / 'adapter_config.json').read_text())
    assert metadata['format_version'] == 9
    assert metadata['transport_signal'] == signal and metadata['transport_precision'] == precision
    assert metadata['independent_transport_up'] is independent
    TALoRATuner.from_pretrained(restored, tmp_path)
    branch = restored.thinker.model.layers[-1].self_attn.ta_adapter
    assert (branch.transport_signal, branch.transport_precision) == (signal, precision)
    assert (branch.transport_up is not None) is independent
    torch.testing.assert_close(model(**record).logits, restored(**record).logits, atol=0, rtol=0)


def test_format_9_metadata_is_rejected_when_inconsistent(tmp_path):
    from experiments.ms_swift_librispeech.plugin import TALoRATuner

    model = tiny_omni()
    configure(model, 'static', signal='ordinary')
    TALoRATuner.save_pretrained(model, tmp_path)
    path = tmp_path / 'adapter_config.json'
    good = json.loads(path.read_text())
    for edit in (dict(transport_signal='video'), dict(format_version=8), dict(transport_precision=None)):
        path.write_text(json.dumps({**good, **edit}))
        with pytest.raises(ValueError, match='Invalid TA-LoRA checkpoint format'):
            TALoRATuner.from_pretrained(tiny_omni(), tmp_path)


def test_control_signals_change_only_the_answer_query():
    from experiments.ms_swift_librispeech.swift_asr_ta_runtime import set_transport_signal

    torch.manual_seed(11)
    model = tiny_omni()
    configure(model, 'static')
    model.eval()
    randomize(model)
    record = inputs(PROMPTS, 5)
    with torch.no_grad():
        reference = model(**record).logits
        for signal in ('audio_mean', 'ordinary', 'ordinary_matched'):
            set_transport_signal(model, signal)
            logits = model(**record).logits
            # The transport enters at index 4 (first label - 1); causal prefixes cannot see it.
            torch.testing.assert_close(logits[:, :4], reference[:, :4], atol=0, rtol=0)
            assert not torch.equal(logits[:, 4], reference[:, 4])


def atoms(precision):
    from experiments.ms_swift_librispeech.swift_asr_ta_runtime import _SharedAtoms

    torch.manual_seed(7)
    branch = _SharedAtoms(nn.Linear(16, 16), 4, 8., 2)
    branch.transport_gate = 'question_audio'
    branch.transport_precision = precision
    branch.transport_up = nn.Parameter(torch.randn(16, 4))
    branch.gate_weight = nn.Parameter(torch.zeros(4, 4))
    with torch.no_grad():
        branch.up.normal_()
        branch.gate.fill_(1.)
    return branch


def answer_inputs():
    torch.manual_seed(3)
    return (torch.randn(2, 5, 16).to(torch.bfloat16), torch.randn(2, 16), torch.tensor([4, 3]), torch.randn(2, 16))


def test_bf16_autocast_erases_small_multipliers_in_format_8():
    # |R f| <= 1e-3 * sqrt(2), so |m - 1| < 7.1e-4 < 2**-9: every BF16 coefficient rounds back.
    ordinary, contrast, indices, question = answer_inputs()
    branch = atoms('autocast')
    with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16):
        neutral = branch.answer_update(ordinary, contrast, indices, question).float()
        branch.gate_weight.copy_(1e-3 * torch.eye(4))
        gated = branch.answer_update(ordinary, contrast, indices, question).float()
    assert torch.equal(neutral, gated)


def test_fp32_transport_keeps_small_multipliers_and_matches_inference_arithmetic():
    ordinary, contrast, indices, question = answer_inputs()
    branch = atoms('fp32')
    with torch.no_grad():
        with torch.autocast('cpu', dtype=torch.bfloat16):
            neutral = branch._fp32_transport(contrast, question, 'cpu')
            branch.gate_weight.copy_(1e-3 * torch.eye(4))
            gated = branch._fp32_transport(contrast, question, 'cpu')
        plain = branch._fp32_transport(contrast, question, 'cpu')
    assert gated.dtype == torch.float32
    assert not torch.equal(neutral, gated)
    assert torch.equal(gated, plain)


def test_fp32_and_format_8_transport_agree_without_autocast():
    ordinary, contrast, indices, question = answer_inputs()
    updates = []
    for precision in ('autocast', 'fp32'):
        branch = atoms(precision)
        with torch.no_grad():
            branch.gate_weight.copy_(.3 * torch.eye(4))
            updates.append(branch.answer_update(ordinary.float(), contrast, indices, question))
    torch.testing.assert_close(updates[0], updates[1], rtol=1e-6, atol=1e-6)


def test_shared_only_answer_transport_learns_through_the_ordinary_readout():
    model = tiny_omni()
    configure(model, 'static', independent=False)
    cfg = model._ta_lora_config
    assert cfg['format_version'] == 9 and cfg['independent_transport_up'] is False
    branches = [layer.self_attn.ta_adapter for layer in model.thinker.model.layers[-4:]]
    assert all(branch.transport_up is None for branch in branches)
    record = inputs(PROMPTS, 5)

    def readout_gradients(gate):  # config key adapter.gate_init
        with torch.no_grad():
            for branch in branches:
                branch.gate.fill_(gate)
        model.zero_grad()
        model(**record).loss.backward()
        return [branch.up.grad.clone() for branch in branches]

    # B is zero, so the forward pass is identical; only the transport term changes B's gradient.
    closed, open_ = readout_gradients(0.), readout_gradients(1.)
    assert all(g.abs().sum() > 0 for g in closed)
    assert not torch.equal(closed[-1], open_[-1])


def test_single_factor_eval_ablations(tmp_path):
    from experiments.ms_swift_aqa.transfer_eval import ablate_for_eval, disable_transport_for_eval
    from experiments.ms_swift_librispeech.plugin import TALoRATuner

    torch.manual_seed(5)
    model = tiny_omni()
    clean = copy.deepcopy(model)
    configure(model)
    model.eval()
    randomize(model)
    TALoRATuner.save_pretrained(model, tmp_path)
    record = inputs(PROMPTS)

    def load():
        loaded = copy.deepcopy(clean)
        TALoRATuner.from_pretrained(loaded, tmp_path)
        return loaded.eval()

    with pytest.raises(ValueError, match='Unknown transport ablation'):
        ablate_for_eval(load(), 'everything')

    # m == 1 exactly, so the conditional model must equal a static-gate model with the same factors.
    neutral = load()
    evidence = ablate_for_eval(neutral, 'neutral_gate')
    assert evidence['tensors_zeroed'] == 4 and evidence['scalars_zeroed'] == 16 and evidence['nonzero_before'] > 0
    static = copy.deepcopy(clean)
    configure(static, 'static')
    static.eval()
    source = dict(neutral.named_parameters())
    with torch.no_grad():
        for name, param in static.named_parameters():
            if '.ta_adapter.' in name:
                param.copy_(source[name])
        torch.testing.assert_close(neutral(**record).logits, static(**record).logits, atol=0, rtol=0)

    # Removing the shared and the independent readout one after the other removes all transport.
    stepwise, reference = load(), load()
    for mode, suffix in (('no_shared', 'ta_adapter.gate'), ('no_independent', 'ta_adapter.transport_up')):
        before = {name: param.detach().clone() for name, param in stepwise.named_parameters()}
        assert ablate_for_eval(stepwise, mode)['tensors_zeroed'] == 4
        for name, param in stepwise.named_parameters():
            if name.endswith(suffix):
                assert not param.count_nonzero()
            else:
                assert torch.equal(before[name], param)
    disable_transport_for_eval(reference)
    with torch.no_grad():
        torch.testing.assert_close(stepwise(**record).logits, reference(**record).logits, atol=0, rtol=0)

    static_loaded = copy.deepcopy(clean)
    configure(static_loaded, 'static')
    static_loaded.eval().requires_grad_(False)
    with pytest.raises(ValueError, match='gate_weight'):
        ablate_for_eval(static_loaded, 'neutral_gate')


def test_adapter_stats_groups_factors_and_tracks_gradient_direction(tmp_path):
    from experiments.ms_swift_librispeech.adapter_stats import AdapterStats, factor_of

    assert factor_of('thinker.model.layers.3.self_attn.ta_adapter.gate_weight') == 'gate_weight'
    assert factor_of('thinker.model.layers.3.self_attn.ta_adapter.gate') == 'gate'
    assert factor_of('thinker.model.layers.3.self_attn.ta_adapter.transport_up') == 'transport_up'
    assert factor_of('base_model.model.thinker.model.layers.3.self_attn.o_proj.lora_B.default.weight') == 'lora_B'
    model = nn.Module()
    model.ta_adapter = nn.Module()
    model.ta_adapter.register_parameter('gate', nn.Parameter(torch.ones(3)))
    model.ta_adapter.register_parameter('gate_weight', nn.Parameter(torch.eye(2)))
    callback = AdapterStats(tmp_path / 'stats.jsonl', 2)
    state = SimpleNamespace(global_step=0)
    for step, sign in enumerate([1., 1., -1., -1.]):
        for param in model.parameters():
            param.grad = sign * torch.ones_like(param)
        state.global_step = step
        callback.on_pre_optimizer_step(None, state, None, model=model)
    rows = [json.loads(line) for line in (tmp_path / 'stats.jsonl').read_text().splitlines()]
    assert [row['update'] for row in rows] == [2, 4]
    assert 'grad_cosine_to_previous' not in rows[0]['factors']['gate_weight']
    assert rows[1]['factors']['gate_weight']['grad_cosine_to_previous'] == pytest.approx(-1.)
    assert rows[0]['factors']['gate']['elements'] == 3
    assert rows[0]['factors']['gate']['grad_norm'] == pytest.approx(3 ** .5)
    with pytest.raises(ValueError, match='positive integer'):
        AdapterStats(tmp_path / 'other.jsonl', 0)
