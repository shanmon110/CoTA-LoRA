"""Compare the projection branches with independently assembled dense weights."""

import pytest
import torch
from torch import nn

from examples.omni_talora.projection import build_branch as build


MODES = ("toplora", "bora", "dora", "lora", "eva")


@pytest.mark.parametrize("mode", MODES)
def test_initial_output_is_native_and_adapters_own_no_frozen_weight(mode):
    torch.manual_seed(221)
    proj = nn.Linear(8, 6).to(torch.bfloat16).requires_grad_(False)
    x = torch.randn(3, 4, 8, dtype=torch.bfloat16)
    branch = build(proj, mode, rank=3, alpha=1.5)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        base = proj(x)
        actual = branch(proj, x, base)
        torch.testing.assert_close(actual, base, atol=0, rtol=0)
        actual.float().square().sum().backward()
    assert branch.scale == 0.5 and branch.mode == mode
    if mode == "bora":
        assert branch.blocks == 2
    assert tuple(branch.down.shape) == (3, 8) and tuple(branch.up.shape) == (6, 3)
    expected = {"down", "up"} | {
        "toplora": {"projector", "rms_weight"}, "bora": {"sigma"}, "dora": {"magnitude"}
    }.get(mode, set())
    assert set(branch.state_dict()) == expected
    assert all(p.dtype == torch.float32 for p in branch.parameters())
    assert branch.up.grad.isfinite().all() and branch.up.grad.abs().sum() > 0
    assert proj.weight.grad is None and proj.bias.grad is None


def dense_reference(mode, x, w, bias, p):
    # Double precision, matrix for every token, explicit diagonals/block matrix.
    # No use of the production implementation or its helpers.
    a, b, s = p["down"], p["up"], 1.5
    rows = []
    for token in x.reshape(-1, x.shape[-1]):
        if mode == "toplora":
            z = p["projector"] @ token
            z = z / (z.dot(z) / z.numel() + torch.finfo(torch.float32).eps).sqrt()
            delta = b @ torch.diag((p["rms_weight"] * z).exp()) @ a
        elif mode == "bora":
            raw = p["sigma"]
            diag = (raw / (raw.abs().sum() / raw.numel())).exp()
            delta = torch.cat([
                torch.cat([bi @ torch.diag(diag[i, j]) @ aj for j, aj in enumerate(a.chunk(2, 1))], 1)
                for i, bi in enumerate(b.chunk(2))
            ])
        else:
            delta = b @ a
        weight = w + s * delta
        if mode == "dora":
            norms = torch.stack([row.dot(row).sqrt() for row in weight]).detach()
            weight = torch.diag(p["magnitude"] / norms) @ weight
        rows.append(weight @ token + bias)
    return torch.stack(rows).reshape(*x.shape[:-1], w.shape[0])


@pytest.mark.parametrize("mode", MODES)
def test_outputs_input_and_parameter_gradients_match_dense_reference(mode):
    torch.manual_seed(123)
    proj = nn.Linear(6, 4).requires_grad_(False)
    x = torch.randn(2, 2, 6).requires_grad_()
    branch = build(proj, mode, rank=2, alpha=3)
    with torch.no_grad():
        for param in branch.parameters():
            param.uniform_(-0.7, 1)
    actual = branch(proj, x, proj(x))
    p = {name: v.detach().double().requires_grad_() for name, v in branch.named_parameters()}
    ref_x = x.detach().double().requires_grad_()
    expected = dense_reference(mode, ref_x, proj.weight.double(), proj.bias.double(), p)
    probe = torch.randn_like(actual)
    (actual * probe).sum().backward()
    (expected * probe).sum().backward()
    torch.testing.assert_close(actual.double(), expected, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(x.grad.double(), ref_x.grad, atol=2e-5, rtol=2e-5)
    for name, v in branch.named_parameters():
        torch.testing.assert_close(v.grad.double(), p[name].grad, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("mode", MODES)
def test_nonzero_branch_preserves_autocast_output_dtype_and_fp32_computation(mode):
    torch.manual_seed(701)
    proj = nn.Linear(6, 4, bias=False).to(torch.bfloat16).requires_grad_(False)
    x = torch.randn(7, 6, dtype=torch.bfloat16)
    branch = build(proj, mode, rank=2, alpha=3)
    with torch.no_grad():
        branch.up.uniform_(-1, 1)
    base = proj(x)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        low = branch(proj, x, base)
    high = branch(proj, x.float(), base.float())
    assert low.dtype == torch.bfloat16
    torch.testing.assert_close(low, high.bfloat16(), atol=0, rtol=0)


@pytest.mark.parametrize("mode,rank,alpha,blocks", [
    ("lora", 0, 1, 1), ("eva", 1, float("nan"), 1), ("dora", 1, -1, 1),
    ("bora", 1, 1, 0), ("bora", 1, 1, 3), ("bora", 1, 1, True),
    ("bora", 1, 1, 2.5), ("not-a-method", 2, 1, 1),
])
def test_invalid_configuration_rejected(mode, rank, alpha, blocks):
    with pytest.raises(ValueError):
        build(nn.Linear(8, 6), mode, rank, alpha, blocks)


def test_bora_all_zero_logits_recover_plain_lora_and_remain_finite():
    torch.manual_seed(36)
    proj = nn.Linear(4, 4, bias=False)
    x = torch.randn(4)
    branch = build(proj, "bora", rank=2, alpha=2)
    with torch.no_grad():
        branch.sigma.zero_()
        branch.up.uniform_()
    actual = branch(proj, x, proj(x))
    expected = (proj.weight + branch.up @ branch.down) @ x
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert branch.sigma.grad.isfinite().all()


def test_dora_zero_weight_row_and_nonzero_bias_are_safe():
    proj = nn.Linear(6, 4).requires_grad_(False)
    with torch.no_grad():
        proj.weight[0].zero_()
    x = torch.ones(6)
    branch = build(proj, "dora", rank=2, alpha=1)
    actual = branch(proj, x, proj(x))
    torch.testing.assert_close(actual, proj(x), rtol=0, atol=0)
    actual.sum().backward()
    assert branch.up.grad.isfinite().all()


def test_bora_reports_unrepresentable_gain_before_scoring():
    proj = nn.Linear(16, 16)
    branch = build(proj, "bora", rank=8, alpha=16, blocks=4)
    x = torch.ones(1, 16)
    with torch.no_grad():
        branch.sigma.fill_(1)
        torch.testing.assert_close(branch(proj, x, proj(x)), proj(x))
        branch.sigma.zero_()
        branch.sigma[0, 0, 0] = 1
        with pytest.raises(FloatingPointError, match="BoRA.*blocks=4.*rank=8"):
            branch(proj, x, proj(x))
