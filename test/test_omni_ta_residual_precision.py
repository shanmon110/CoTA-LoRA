"""BF16 rounding at a half-ULP boundary in actual eager Thinker attention."""

import hashlib
import json
import math

import pytest
import torch
from safetensors.torch import save_file

from examples.omni_talora.adapters import install_from_config
from examples.omni_talora.attention import audio_context
from examples.omni_talora.checkpoint import load_adapter, read_config, save_adapter

if __package__:
    from .test_omni_ta_attention import run_attn, tiny_thinker
else:
    from test_omni_ta_attention import run_attn, tiny_thinker


POLICY = "fp32_add_then_cast"
PROJECTION = "model.layers.0.self_attn.o_proj"


def base_model():
    model = tiny_thinker(dtype=torch.bfloat16)
    attn = model.model.layers[0].self_attn
    with torch.no_grad():
        for param in attn.parameters():
            param.zero_()
        attn.v_proj.weight[0, 0] = attn.o_proj.weight[0, 0] = 1
    return model


def config_for(mode):
    config = dict(model_name_or_path="tiny", revision=None, task="asr",
                  rank=1, alpha=1., last_n=1, mode=mode)
    if mode == "eva":
        config.update(target_set="o", layer_ranks={PROJECTION: 1},
                      eva_samples=2, eva_seed=42, eva_rho=1.)
    return config


def state_for(mode, gate):
    prefix = PROJECTION + ".baseline_adapter." if mode == "eva" else "model.layers.0.self_attn.ta_adapter."
    down = torch.zeros(1, 12)
    up = torch.zeros(12, 1)
    down[0, 0], up[0, 0] = 1., .00391
    state = {prefix + "down": down, prefix + "up": up}
    if mode == "ta":
        state[prefix + "gate"] = torch.tensor([gate])
    return state


@torch.no_grad()
def output(model):
    # Zero Q/K yield a causal mean of values 1 and 3. At the text query
    # O=2 and transport=1-3=-2. Only the first output channel is active.
    ids = torch.tensor([[99, 5]])
    valid = torch.ones_like(ids)
    x = torch.zeros(1, 2, 12, dtype=torch.bfloat16)
    x[0, :, 0] = torch.tensor([1, 3], dtype=x.dtype)
    with audio_context(ids, valid, 99):
        out = run_attn(model.model.layers[0].self_attn, x, valid)[0]
    assert out.dtype == torch.bfloat16
    assert out[..., 1:].count_nonzero() == 0
    return out[0, :, 0]


@pytest.mark.parametrize(("mode", "expected"), [
    ("lora", [1., 2.]), ("ta", [1., 2.015625]), ("eva", [1.0078125, 2.015625]),
])
def test_fieldless_checkpoint_restores_historical_bf16_forward(mode, expected, tmp_path):
    # Write the old wire format independently of the new checkpoint writer.
    config = config_for(mode)
    digest = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    save_file(state_for(mode, gate=-1.), tmp_path / "adapter.safetensors",
              metadata={"format": "west-omni-talora-v1", "config_sha256": digest})
    (tmp_path / "adapter_config.json").write_text(json.dumps(config))
    model = base_model()
    install_from_config(model, read_config(tmp_path))
    load_adapter(model, tmp_path)
    assert output(model).tolist() == expected


@pytest.mark.parametrize("mode", ["ta", "lora", "eva"])
def test_new_zero_gate_ta_o_lora_and_projection_lora_match_at_bf16_boundary(mode, tmp_path):
    # EVA after installation is a projection LoRA. Identical A/B and a zero
    # TA gate should give the same result both before and after serialization.
    config = dict(config_for(mode), residual_precision=POLICY)
    model = base_model()
    install_from_config(model, config)
    with torch.no_grad():
        for name, value in state_for(mode, gate=0.).items():
            model.get_parameter(name).copy_(value)
    expected = torch.tensor([1.0078125, 2.015625], dtype=torch.bfloat16)
    assert torch.equal(output(model), expected)
    save_adapter(model, tmp_path, config)
    saved = read_config(tmp_path)
    assert saved["residual_precision"] == POLICY
    fresh = base_model()
    install_from_config(fresh, saved)
    load_adapter(fresh, tmp_path)
    assert torch.equal(output(fresh), expected)


@pytest.mark.parametrize("mode", ["lora", "toplora", "bora", "dora", "eva"])
def test_all_projection_hooks_round_after_adding_the_update(mode):
    model = base_model()
    attn = model.model.layers[0].self_attn
    config = dict(config_for(mode), residual_precision=POLICY, target_set="o",
                  layer_ranks={PROJECTION: 1})
    proj = attn.o_proj
    if mode == "lora":
        config.update(target_set="qv", layer_ranks={f"model.layers.0.self_attn.{kind}_proj": 1
                                                    for kind in "qv"})
        proj = attn.v_proj
    if mode == "bora":
        config["bora_blocks"] = 1
    install_from_config(model, config)
    branch = proj.baseline_adapter
    with torch.no_grad():
        branch.down.zero_()
        branch.down[0, 0] = 1.
        branch.up[0, 0] = .00391
        if mode == "toplora":
            branch.projector.zero_()  # exp(RMSNorm(0)) = 1
        elif mode == "bora":
            branch.sigma.fill_(1.)
            branch.up.div_(math.e)  # compensate for the single block's exp(1)
        elif mode == "dora":
            branch.magnitude[0] = 1. + .00391  # match the updated direction norm
        x = torch.zeros(1, proj.in_features, dtype=torch.bfloat16)
        x[0, 0] = 1.
        y = proj(x)
    assert y.dtype == torch.bfloat16
    assert y[0, 0].item() == 1.0078125


def test_refuses_unknown_or_mismatched_precision(tmp_path):
    model = base_model()
    config = config_for("lora")
    with pytest.raises(ValueError, match="residual_precision"):
        install_from_config(model, dict(config, residual_precision="bad"))
    install_from_config(model, dict(config, residual_precision=POLICY))
    with pytest.raises(ValueError, match="residual_precision"):
        save_adapter(model, tmp_path, config)
