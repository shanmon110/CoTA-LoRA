"""Configured method selection, rank budgets and projection checkpoints."""

import copy

import pytest
import torch
from torch import nn

from examples.omni_talora.checkpoint import save_adapter


def fixture():
    model = nn.Module()
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([nn.Module(), nn.Module()])
    for layer in model.model.layers:
        attn = layer.self_attn = nn.Module()
        attn.o_proj = nn.Linear(16, 16)
        attn.q_proj = nn.Linear(16, 16)
        attn.v_proj = nn.Linear(16, 4)
    return model


def test_budget_uses_real_value_width_and_gate():
    from examples.omni_talora.adapters import qv_budget
    # 2 layers: TA r=8 costs 2*(32+1)*8=528; qv costs 104/rank.
    assert qv_budget(fixture(), 2, 8) == (5, 528, 520)


def test_new_checkpoint_cannot_include_unrelated_weights(tmp_path):
    from examples.omni_talora.adapters import install_from_config
    model = fixture()
    ranks = {f"model.layers.1.self_attn.{kind}_proj": 2 for kind in ("q", "v")}
    conf = dict(model_name_or_path="tiny", revision=None, task="asr", rank=2, alpha=4.,
                last_n=1, mode="lora", target_set="qv", layer_ranks=ranks)
    install_from_config(model, conf)
    model.model.layers[-1].self_attn.o_proj.weight.requires_grad_()
    with pytest.raises(ValueError, match="trainable"):
        save_adapter(model, tmp_path, conf)
    assert not list(tmp_path.iterdir())


def test_rank_map_cannot_substitute_an_earlier_layer():
    from examples.omni_talora.adapters import install_from_config
    model = fixture()
    conf = dict(rank=2, alpha=4., last_n=1, mode="dora", target_set="o",
                layer_ranks={"model.layers.0.self_attn.o_proj": 2})
    with pytest.raises(ValueError, match="layer_ranks"):
        install_from_config(model, conf)
    assert all(p.requires_grad for p in model.parameters())


@pytest.mark.parametrize("settings", [
    {"method": "unknown"}, {"method": "o-lora", "eva_rho": 3},
    {"method": "qv-lora", "target_modules": ["o_proj"]},
    {"method": "bora", "bora_blocks": False}, {"method": "eva", "eva_token_mask": "all"},
    {"method": "ta-lora", "r": True}, {"method": "dora", "lora_dropout": 0.1},
])
def test_tagged_config_rejects_unknown_or_inapplicable_settings(settings):
    from west_adapters import resolve_lora_config
    with pytest.raises(ValueError):
        resolve_lora_config(settings)


@pytest.fixture
def register_for_test(monkeypatch):
    from west_adapters import register_method, registry
    for name in ("_METHODS", "_CHECKPOINT_METHODS"):
        monkeypatch.setattr(registry, name, getattr(registry, name).copy())
    return register_method


def test_additional_projection_method_installs_and_reloads_without_trainer_changes(register_for_test, tmp_path):
    from examples.omni_talora.checkpoint import read_config, load_adapter
    from west_adapters import (AdapterMethod, ProjectionBranch, finalize_ranks, install_from_config,
                               resolve_layout, resolve_lora_config)

    class ScaledBranch(ProjectionBranch):
        def __init__(self, proj, rank, alpha, config):
            super().__init__(proj, rank, alpha, config)
            self.coefficient = config["coefficient"]

        def coefficients(self, x):
            return super().coefficients(x) * self.coefficient

    class ScaledMethod(AdapterMethod):
        name = mode = "scaled-test"
        options = {"coefficient": 1.0}
        branch = ScaledBranch

    registered = ScaledMethod()
    register_for_test(registered)
    with pytest.raises(ValueError, match="Duplicate"):
        register_for_test(ScaledMethod())
    options = resolve_lora_config(dict(method=registered.name, r=2, last_n=1, coefficient=.25))
    assert "coefficient" not in resolve_lora_config({"method": "o-lora"})
    model = fixture()
    clean = copy.deepcopy(model)
    method, layout, projections = resolve_layout(model, options)
    assert method is registered and layout["coefficient"] == .25
    finalize_ranks(layout, projections)
    layout.update(task="asr", model_name_or_path="tiny", revision=None)
    branch, = install_from_config(model, layout)
    with torch.no_grad():
        branch.down.fill_(.125)
        branch.up.fill_(.25)
    inputs = torch.ones(1, 16)
    logits = model.model.layers[-1].self_attn.o_proj(inputs)
    base = clean.model.layers[-1].self_attn.o_proj(inputs)
    torch.testing.assert_close(logits - base, torch.full_like(logits, 2.))
    save_adapter(model, tmp_path, layout)
    install_from_config(clean, read_config(tmp_path))
    load_adapter(clean, tmp_path)
    torch.testing.assert_close(logits, clean.model.layers[-1].self_attn.o_proj(inputs), rtol=0, atol=0)


def test_registered_attention_method_uses_host_installer_and_plugin_checkpoint_keys(register_for_test, tmp_path):
    from west_adapters import finalize_ranks, install_from_config, resolve_layout, resolve_lora_config
    from west_adapters.ta_lora import TransportAtoms, TransportLoRA
    from examples.omni_talora.adapters import install_from_config as install_omni
    from examples.omni_talora.checkpoint import load_adapter, read_config
    if __package__:
        from .test_omni_ta_attention import run_attn, tiny_thinker
    else:
        from test_omni_ta_attention import run_attn, tiny_thinker

    class CustomAtoms(TransportAtoms):
        def attention_delta(self, ordinary, logits, values, mask, cache_position, layout):
            assert logits.shape[-1] == values.shape[-2] == mask.shape[-1]
            assert cache_position.tolist() == [0, 1] and layout is None
            return self.gate.sum() * torch.ones_like(ordinary)

    class AdditionalTransport(TransportLoRA):
        name = mode = "transport-test"
        needs_audio_context = False
        atoms = CustomAtoms

    register_for_test(AdditionalTransport())
    model = tiny_thinker()
    clean = copy.deepcopy(model)
    options = resolve_lora_config(dict(method="transport-test", r=2, last_n=1))
    _, layout, projections = resolve_layout(model, options)
    finalize_ranks(layout, projections)
    layout.update(task="asr", model_name_or_path="tiny", revision=None)
    with pytest.raises(ValueError, match="host attention installer"):
        install_from_config(model, layout)
    module, = install_omni(model, layout)
    with torch.no_grad():
        module.ta_adapter.gate.fill_(.375)
        hidden = torch.randn(1, 2, 12)
        valid = torch.ones(1, 2)
        base = run_attn(clean.model.layers[0].self_attn, hidden, valid)[0]
        actual = run_attn(module, hidden, valid)[0]
        torch.testing.assert_close(actual, base + .75, rtol=0, atol=0)
    save_adapter(model, tmp_path, layout)
    restored, = install_omni(clean, read_config(tmp_path))
    load_adapter(clean, tmp_path)
    torch.testing.assert_close(restored.ta_adapter.gate, module.ta_adapter.gate, rtol=0, atol=0)
