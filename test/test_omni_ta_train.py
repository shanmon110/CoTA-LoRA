"""Optimizer-level tests: compare unequal-length microbatches with one pooled batch."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from examples.omni_talora.train import train_updates, validate_output


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.tensor([1., 0., -0.3]))

    def forward(self, input_ids, attention_mask, labels, use_cache):
        assert use_cache is False
        targets = labels[:, 1:].flatten()
        return SimpleNamespace(loss=F.cross_entropy(self.logits.expand(targets.numel(), -1), targets))


def batch(targets):
    ids = torch.zeros((1, 1 + len(targets)), dtype=torch.long)
    return dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                labels=torch.tensor([[-100, *targets]]))


def test_weight_by_supervised_tokens_not_microbatch_count():
    one, split = Classifier(), Classifier()
    for m, items, acc in [(one, [batch([0, 2, 2, 1])], 1),
                          (split, [batch([0]), batch([2, 2, 1])], 2)]:
        opt = torch.optim.SGD(m.parameters(), lr=.2)
        train_updates(m, items, opt, device="cpu", audio_token_id=8,
                      max_steps=1, accumulation_steps=acc)
    torch.testing.assert_close(one.logits, split.logits)
    assert not torch.equal(split.logits, torch.tensor([1., 0., -.3]))


def test_cycles_finite_loader_and_rejects_no_supervision():
    m = Classifier()
    opt = torch.optim.SGD(m.parameters(), lr=.01)
    losses = train_updates(m, [batch([1])], opt, device="cpu", audio_token_id=8,
                           max_steps=3, accumulation_steps=2)
    assert len(losses) == 3
    with pytest.raises(ValueError, match="supervis"):
        train_updates(m, [batch([-100])], opt, device="cpu", audio_token_id=8,
                      max_steps=1, accumulation_steps=1)


def test_preserve_output(tmp_path):
    (tmp_path / "other_task.json").touch()
    with pytest.raises(FileExistsError):
        validate_output(tmp_path)


def test_epoch_tail_and_midpoint_observe_completed_updates():
    m, reference = Classifier(), Classifier()
    groups = [batch([0]), batch([1]), batch([2])]
    opt = torch.optim.SGD(m.parameters(), lr=.1)
    snapshots = {}
    train_updates(m, groups, opt, device="cpu", audio_token_id=8, max_steps=2,
                  accumulation_steps=2, one_epoch=True,
                  after_update=lambda step: snapshots.update({step: m.logits.detach().clone()}))
    ref_opt = torch.optim.SGD(reference.parameters(), lr=.1)
    for step, targets in enumerate(([0, 1], [2]), start=1):
        train_updates(reference, [batch(targets)], ref_opt, device="cpu", audio_token_id=8,
                      max_steps=1, accumulation_steps=1)
        torch.testing.assert_close(snapshots[step], reference.logits)
