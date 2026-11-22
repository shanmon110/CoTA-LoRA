"""Exercise answer-query TA on the native miniature Omni model."""
import copy
import json
from types import SimpleNamespace

import pytest
import torch

from test_adapter_pair import TARGETS, tiny_omni


def configure(model, mode):
    from experiments.ms_swift_librispeech.plugin import TALoRATuner
    from experiments.ms_swift_librispeech.swift_asr_ta_runtime import enable_transport_up, set_transport_gate

    TALoRATuner.prepare_model(SimpleNamespace(target_modules=TARGETS, lora_rank=2, lora_alpha=4), model)
    enable_transport_up(model)
    set_transport_gate(model, mode)


def inputs(ids, first_label=None):
    input_ids = torch.tensor(ids)
    labels = input_ids.clone()
    if first_label is not None:
        labels[:, :first_label] = -100
    kwargs = dict(input_ids=input_ids, attention_mask=input_ids.ne(0), use_cache=False,
                  position_ids=torch.arange(input_ids.shape[-1]).view(1, 1, -1).expand(3, *input_ids.shape))
    if first_label is not None:
        kwargs['labels'] = labels
    return kwargs


@pytest.mark.parametrize('mode', ['static', 'question', 'question_audio'])
def test_gate_is_zero_update_at_init_and_learns_at_answer_query(mode):
    model = tiny_omni()
    baseline = copy.deepcopy(model)
    baseline.requires_grad_(False)
    configure(model, mode)
    record = inputs([[1, 5, 5, 6, 3, 8, 2], [1, 5, 5, 7, 3, 9, 2]], 5)
    expected = baseline(**record).logits
    output = model(**record)
    torch.testing.assert_close(output.logits, expected, atol=0, rtol=0)
    output.loss.backward()
    for layer in model.thinker.model.layers[-4:]:
        branch = layer.self_attn.ta_adapter
        assert branch.transport_up.grad.abs().sum() > 0
        if branch.gate_weight is not None:
            assert not branch.gate_weight.grad.count_nonzero()


@pytest.mark.parametrize('mode', ['question', 'question_audio'])
def test_gate_roundtrip_and_training_prefix_equivalence(tmp_path, mode):
    from experiments.ms_swift_librispeech.adapter_audit import audit_adapter
    from experiments.ms_swift_librispeech.plugin import TALoRATuner
    from experiments.ms_swift_librispeech.swift_asr_ta_runtime import _answer_query

    model = tiny_omni()
    restored = copy.deepcopy(model)
    configure(model, mode)
    model.eval()
    assert audit_adapter(model, 'ta_lora')['transport_gate_parameters'] == 16
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad:
                param.uniform_(-1, 1)
    teacher = inputs([[1, 5, 5, 6, 3, 8, 2], [0, 1, 5, 5, 7, 9, 2]], 5)
    teacher['labels'][1, 5] = -100
    for row, stop in [(0, 5), (1, 6)]:
        single = {key: v[:, row:row + 1, :stop] if key == 'position_ids' else v[row:row + 1, :stop]
                  for key, v in teacher.items() if isinstance(v, torch.Tensor) and key != 'labels'}
        single['use_cache'] = False
        torch.testing.assert_close(model(**single).logits[:, -1], model(**teacher).logits[row:row+1, stop-1], atol=.07, rtol=.025)
    frame = _answer_query(teacher['input_ids'], teacher['attention_mask'], teacher['labels'],
                          audio_token_id=5, embed=model.thinker.get_input_embeddings())
    assert frame.positions.tolist() == [4, 5]
    expected_text = torch.stack([
        model.thinker.get_input_embeddings()(torch.tensor([6, 3])).float().mean(0),
        model.thinker.get_input_embeddings()(torch.tensor([7, 9])).float().mean(0),
    ])
    torch.testing.assert_close(frame.question, expected_text)
    TALoRATuner.save_pretrained(model, tmp_path)
    metadata = json.loads((tmp_path / 'adapter_config.json').read_text())
    assert metadata['format_version'] == 8
    assert metadata['transport_gate'] == mode
    TALoRATuner.from_pretrained(restored, tmp_path)
    assert not any(param.requires_grad for param in restored.parameters())
    torch.testing.assert_close(model(**teacher).logits, restored(**teacher).logits, atol=0, rtol=0)


def test_text_only_gate_ignores_audio_context():
    from experiments.ms_swift_librispeech.swift_asr_ta_runtime import _SharedAtoms

    settings = _SharedAtoms(torch.nn.Linear(4, 4), 2, 4, 2)
    settings.gate_weight = torch.nn.Parameter(torch.eye(2))
    question = torch.tensor([[.5, 1.]])
    audio_a = torch.tensor([[2., 0.]])
    audio_b = torch.tensor([[0., -2.]])
    settings.transport_gate = 'question'
    torch.testing.assert_close(settings._context_multiplier(question, audio_a), settings._context_multiplier(question, audio_b))
    settings.transport_gate = 'question_audio'
    assert not torch.equal(settings._context_multiplier(question, audio_a), settings._context_multiplier(question, audio_b))


def test_gate_optimizer_keeps_parameter_identity_decay_and_resume_state():
    from experiments.ms_swift_librispeech.train import scale_transport_gate_lr
    from torch import nn

    def create():
        model = nn.Module()
        model.register_parameter('unrelated', nn.Parameter(torch.ones(2)))
        model.ta_adapter = nn.Module()
        model.ta_adapter.register_parameter('gate_weight', nn.Parameter(torch.eye(2)))
        model.ta_adapter.register_parameter('gate', nn.Parameter(torch.ones(2)))
        params = dict(model.named_parameters())
        group_names = ['ta_adapter.gate_weight', 'unrelated']
        optimizer = torch.optim.AdamW([
            dict(params=[params[name] for name in group_names], param_names=group_names,
                 weight_decay=.2, lr=.001, eps=.01),
            dict(params=[params['ta_adapter.gate']], param_names=['ta_adapter.gate'], weight_decay=0.),
        ], lr=.002, betas=(.7, .9))
        return model, params, optimizer

    model, params, optimizer = create()
    assert scale_transport_gate_lr(optimizer, model, 16) is optimizer
    ordinary, dynamic, static = optimizer.param_groups
    assert ordinary['params'][0] is params['unrelated']
    assert ordinary['param_names'] == ['unrelated'] and ordinary['lr'] == .001
    assert dynamic['params'][0] is params['ta_adapter.gate_weight']
    assert dynamic['param_names'] == ['ta_adapter.gate_weight'] and dynamic['lr'] == .016
    assert static['params'][0] is params['ta_adapter.gate'] and static['lr'] == .002
    for key in ('weight_decay', 'eps', 'betas', 'amsgrad', 'maximize', 'foreach', 'fused'):
        assert dynamic[key] == ordinary[key]
    assert static['weight_decay'] == 0.
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 / (step + 1))
    for p in params.values():
        p.grad = torch.full_like(p, .4)
    optimizer.step()
    scheduler.step()
    saved = copy.deepcopy(optimizer.state_dict())
    more_model, more_params, resumed = create()
    scale_transport_gate_lr(resumed, more_model, 16)
    resumed_scheduler = torch.optim.lr_scheduler.LambdaLR(resumed, lambda step: 1 / (step + 1))
    resumed.load_state_dict(saved)
    resumed_scheduler.load_state_dict(scheduler.state_dict())
    assert resumed.param_groups[1]['lr'] == 16 * resumed.param_groups[0]['lr']
    assert resumed.param_groups[1]['params'][0] is more_params['ta_adapter.gate_weight']
    torch.testing.assert_close(resumed.state[more_params['ta_adapter.gate_weight']]['exp_avg'],
                               optimizer.state[params['ta_adapter.gate_weight']]['exp_avg'])
    resumed.step()
    resumed_scheduler.step()
    assert resumed.param_groups[1]['lr'] == 16 * resumed.param_groups[0]['lr']


@pytest.mark.parametrize('multiplier', [True, 0, -2, float('nan'), float('inf')])
def test_gate_optimizer_rejects_invalid_multiplier(multiplier):
    from experiments.ms_swift_librispeech.train import scale_transport_gate_lr

    model = torch.nn.Linear(3, 2)
    with pytest.raises(ValueError, match='finite positive'):
        scale_transport_gate_lr(torch.optim.AdamW(model.parameters()), model, multiplier)


def test_gate_optimizer_rejects_missing_gate_and_other_optimizers():
    from experiments.ms_swift_librispeech.train import scale_transport_gate_lr

    model = torch.nn.Linear(3, 2)
    with pytest.raises(ValueError, match='AdamW'):
        scale_transport_gate_lr(torch.optim.SGD(model.parameters()), model, 2)
    with pytest.raises(ValueError, match='trainable conditional gate'):
        scale_transport_gate_lr(torch.optim.AdamW(model.parameters()), model, 2)
