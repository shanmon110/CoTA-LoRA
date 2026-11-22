"""EVA calibration contracts on projections with hand-computable input spectra."""

import pytest
import torch
from torch import nn


class ProbeThinker(nn.Module):
    """Token IDs index known pre-o_proj activations; projection outputs are zero."""

    def __init__(self, tables):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList()
        self.rope_deltas = None
        for table in tables:
            layer = nn.Module()
            layer.register_buffer("table", table)
            layer.self_attn = nn.Module()
            layer.self_attn.o_proj = nn.Linear(table.size(1), table.size(1), bias=False)
            nn.init.zeros_(layer.self_attn.o_proj.weight)
            self.model.layers.append(layer)

    def forward(self, input_ids, attention_mask, use_cache):
        assert not self.training and not torch.is_grad_enabled() and not use_cache
        self.rope_deltas = input_ids
        for layer in self.model.layers:
            layer.self_attn.o_proj(layer.table[input_ids])


def calibrate(model, loader, *, rank=1, rho=2, last_n=1, **kwargs):
    from examples.omni_talora.eva import calibrate_eva
    return calibrate_eva(model, loader, last_n=last_n, rank=rank, rho=rho, device="cpu", **kwargs)


def supervised(ids):
    ids = torch.tensor([ids])
    return {"input_ids": ids, "labels": ids.clone(), "attention_mask": torch.ones_like(ids)}


def test_global_ratios_include_spectral_tail_and_can_allocate_zero_rank():
    # A has variance fractions .534, .466; B has .287, .238, .238, .238.
    # Raw eigenvalues allocate both ranks to B; renormalizing *only* the first
    # two eigenvalues of each layer allocates one to each. Correct result: A,A.
    def table(scales):
        diagonal = torch.diag(torch.tensor(scales))
        return torch.cat([diagonal, -diagonal]) + torch.tensor([0., 0., 0., 1000.])
    a, b = table([3., 2.8, 0., 0.]), table([11., 10., 10., 10.])
    model = ProbeThinker([torch.zeros_like(a), a, b])
    basis = calibrate(model, [supervised(range(8))], last_n=2)
    first, second = [f"model.layers.{i}.self_attn.o_proj" for i in (1, 2)]
    assert set(basis) == {first, second}
    torch.testing.assert_close(basis[first].abs(), torch.eye(4)[:2], rtol=1e-5, atol=1e-5)
    assert basis[second].shape == (0, 4)
    assert all(v.dtype == torch.float32 and v.device.type == "cpu" for v in basis.values())


@pytest.mark.parametrize("use_label_mask,axis", [(True, 0), (False, 1)])
def test_token_policy_never_includes_padding(use_label_mask, axis):
    # Same-position labeled tokens vary on x, prompt tokens on y, pad on z.
    model = ProbeThinker([torch.tensor([[1., 0., 40.], [-1., 0., 40.],
                                       [0., 50., 40.], [0., -50., 40.],
                                       [0., 0., 1000.], [0., 0., -1000.]])])
    batch = supervised(range(6))
    batch["labels"][0, 2:4] = -100
    batch["attention_mask"][0, 4:] = 0
    # Test that default is the SFT label mask, not all-nonpad.
    options = {} if use_label_mask else {"use_label_mask": False}
    [basis] = calibrate(model, [batch], rho=1, **options).values()
    torch.testing.assert_close(basis.abs(), torch.eye(3)[axis:axis + 1], rtol=1e-5, atol=1e-5)


def test_online_centering_handles_single_token_batches_and_restores_state():
    # Each batch has zero *within*-batch variance; all signal is between batches.
    model = ProbeThinker([torch.tensor([[1000., -3.], [1000., -1.], [1000., 1.], [1000., 3.]])])
    model.train()
    model.model.layers[0].eval()  # mixed training flags shouldn't be overwritten
    old_flags = [m.training for m in model.modules()]
    original_cache = model.rope_deltas = torch.tensor([-7])
    old_weights = {k: v.clone() for k, v in model.state_dict().items()}
    [basis] = calibrate(model, (supervised([i]) for i in range(4)), rho=1).values()
    torch.testing.assert_close(basis.abs(), torch.tensor([[0., 1.]]))
    assert [m.training for m in model.modules()] == old_flags
    assert model.rope_deltas is original_cache
    assert not model.model.layers[0].self_attn.o_proj._forward_pre_hooks
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, old_weights[k], rtol=0, atol=0)


def test_insufficient_supervision_does_not_silently_shrink_budget():
    model = ProbeThinker([torch.eye(4)])
    batch = supervised(range(4))
    batch["labels"][0, 2:] = -100
    with pytest.raises(ValueError, match=r"(?i)EVA.*3.*tokens.*2"):
        calibrate(model, [batch], rho=2)


def test_nonfinite_activation_removes_only_calibration_hooks_and_restores_mode():
    model = ProbeThinker([torch.tensor([[float("nan"), 1.], [0., 1.]])])
    projection = model.model.layers[0].self_attn.o_proj
    calls = []
    observer = projection.register_forward_pre_hook(lambda *_: calls.append(1))
    try:
        with pytest.raises(ValueError, match="nonfinite"):
            calibrate(model, [supervised(range(2))], rho=1)
        assert calls == [1]
        assert set(projection._forward_pre_hooks) == {observer.id}
        assert model.training and model.rope_deltas is None
    finally:
        observer.remove()


def test_constant_activations_report_unidentifiable_directions():
    model = ProbeThinker([torch.zeros(8, 4)])
    with pytest.raises(ValueError, match="variance"):
        calibrate(model, [supervised(range(8))])


def test_calibration_rejects_an_installed_projection_adapter_even_outside_targets():
    model = ProbeThinker([torch.eye(4), torch.eye(4)])
    model.model.layers[0].self_attn.o_proj.baseline_adapter = nn.Linear(4, 4)
    with pytest.raises(ValueError, match="unadapted"):
        calibrate(model, [supervised(range(4))], last_n=1)


def test_accumulated_ratios_are_against_all_training_variance_not_truncated_trace():
    from examples.omni_talora.eva import _IncrementalPCA
    # First block: x,y,z scatter = (18,8,2). With cap 2 the z basis is
    # discarded. The second block adds (2,0,0); total trace is 30, not 28.
    diag = torch.diag(torch.tensor([3., 2., 1.]))
    state = _IncrementalPCA(cap=2)
    state.update(torch.cat([diag, -diag]))
    state.update(torch.tensor([[1., 0., 0.], [-1., 0., 0.]]))
    assert state.ratios() == pytest.approx([20 / 30, 8 / 30], rel=1e-5)
