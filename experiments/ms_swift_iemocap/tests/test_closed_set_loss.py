from math import log
from types import SimpleNamespace

import pytest
import torch

from experiments.ms_swift_iemocap import closed_set_loss


class Tokenizer:
    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return [{'A': 0, 'B': 2, 'C': 4, 'D': 6, '<|im_end|>': 1, '\n': 3}[text]]


def trainer():
    template = SimpleNamespace(tokenizer=Tokenizer(), packing=False, padding_free=False, sequence_parallel_size=1)
    args = SimpleNamespace(gradient_accumulation_steps=1, use_logits_to_keep=False, world_size=1)
    return SimpleNamespace(template=template, args=args)


def test_uses_preceding_logit_and_only_four_choices_with_equal_utterance_weight():
    tr = trainer()
    loss_fn = closed_set_loss.make_closed_set_loss(tr)
    # The first utterance has a padded tail. The second has a longer prompt.
    labels = torch.tensor([[-100, -100, 6, 1, 3, -100, -100], [-100, -100, -100, -100, 0, 1, 3]])
    logits = torch.zeros(2, 7, 8, dtype=torch.bfloat16)
    logits[:, :, 7] = 100  # Highest logit in the vocabulary is outside ABCD.
    logits[0, 1, 6] = 2
    logits[1, 3, 0] = -1
    logits.requires_grad_()
    loss = loss_fn(SimpleNamespace(logits=logits), labels, trainer=tr, num_items_in_batch=6, loss_scale=None)
    expected = (log(1 + 3 * torch.exp(torch.tensor(-2.)).item()) +
                log(1 + 3 * torch.exp(torch.tensor(1.)).item())) / 2
    assert loss.dtype == torch.float32 and loss.item() == pytest.approx(expected)
    loss.backward()
    support = logits.grad.ne(0).nonzero().tolist()
    assert support == [[row, pos, token] for row, pos in ((0, 1), (1, 3)) for token in (0, 2, 4, 6)]


@pytest.mark.parametrize('label', [
    [6, 1, 3, -100, -100, -100],  # no causal predecessor
    [-100, 6, -100, 1, 3, -100],  # noncontiguous target
    [-100, 7, 1, 3, -100, -100],  # fifth answer
    [-100, 2, 3, 1, -100, -100],  # wrong suffix
    [-100, -100, -100, -100, -100, -100],  # unlabeled row
    [-100, 2, 1, 3, 2, -100],  # multiple answers
])
def test_rejects_ambiguous_or_truncated_targets(label):
    tr = trainer()
    with pytest.raises(ValueError):
        closed_set_loss.make_closed_set_loss(tr)(SimpleNamespace(logits=torch.zeros(1, 6, 8)), torch.tensor([label]))


@pytest.mark.parametrize('key,value', [('gradient_accumulation_steps', 2), ('world_size', 2), ('use_logits_to_keep', True)])
def test_rejects_configs_with_other_denominators_or_trimmed_rows(key, value):
    tr = trainer()
    setattr(tr.args, key, value)
    with pytest.raises(ValueError):
        closed_set_loss.make_closed_set_loss(tr)
