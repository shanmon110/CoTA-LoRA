"""The closed-set scorer records the decoding settings with its metrics.

A comparison that puts two models side by side has to know that they were scored the same way, so
the generation settings travel with the metrics rather than being left in the caller's memory.
"""
import contextlib
import json

import pytest
import torch

from experiments.ms_swift_iemocap import evaluate

CHOICES = 'ABCD'


class Tokenizer:
    def encode(self, text, *, add_special_tokens):
        return [{'A': 0, 'B': 1, 'C': 2, 'D': 3}[text]]


class Template:
    """Returns the prompt plus one token, so that the generated prefix matches the prompt."""

    def __init__(self):
        self.tokenizer = Tokenizer()

    def generate_context(self):
        return contextlib.nullcontext()

    def set_mode(self, mode):
        assert mode == 'transformers'

    def encode(self, item):
        return item

    def data_collator(self, encoded):
        n = len(encoded)
        return dict(input_ids=torch.tensor([[1, 2]] * n), attention_mask=torch.tensor([[1, 1]] * n),
                    input_features=torch.zeros(n, 1), feature_attention_mask=torch.tensor([[1]] * n))

    def generate(self, model, **batch):
        size = batch['input_ids'].shape[0]
        return torch.cat([batch['input_ids'], torch.tensor([[0]]).repeat(size, 1)], dim=1)


class Model:
    device = 'cpu'


def rows(n=4):
    return [dict(id=f'u{i}', messages=[dict(role='user', content='q'), dict(role='assistant', content=CHOICES[i])],
                 audios=['x.wav']) for i in range(n)]


def test_the_metrics_carry_the_decoding_settings(tmp_path):
    manifest = tmp_path / 'manifest.jsonl'
    manifest.write_text('\n'.join(json.dumps(r) for r in rows()))
    predictions = tmp_path / 'predictions.jsonl'
    generation = dict(batch_size=2)

    result = evaluate.score(Model(), Template(), manifest, generation, predictions)
    assert result['generation'] == generation
    assert result['utterances'] == 4
    assert json.loads(predictions.read_text().splitlines()[0])['reference'] in CHOICES


def test_the_metrics_refuse_a_missing_class():
    with pytest.raises(ValueError, match='All four reference classes'):
        evaluate.metrics([('A', 'A'), ('B', 'B')])
