"""Exercise Transformer Trainer's real save/optimizer resume loop and WER-driven control."""
import json
from pathlib import Path

import pytest
import torch
from torch import nn
from transformers import Trainer, TrainingArguments, set_seed

from experiments.ms_swift_librispeech.train import Schedule, select


def test_tie_prefers_earlier_and_rejects_invalid_history():
    history = [{'step': i, 'eval_wer': w} for i, w in enumerate([.5, .25, .25, .4], 1)]
    assert select(history, 2) == (2, .25, True)
    assert select(history[:-1], 2) == (2, .25, False)
    with pytest.raises(ValueError):
        select(history + [{'step': 5, 'eval_wer': float('nan')}], 2)


def test_pair_accuracy_maximizes_both_correct_and_keeps_first_tie():
    history = [{'step': step, 'eval_pair_accuracy': value, 'eval_accuracy': 1 - value}
               for step, value in enumerate([.35, .6, .6, .4], 1)]
    assert select(history, 2, 'pair_accuracy') == (2, .6, True)
    with pytest.raises(ValueError):
        select(history + [{'step': 5, 'eval_pair_accuracy': 1.1}], 2, 'pair_accuracy')


class Regressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, input_ids, labels):
        logits = self.linear(input_ids)
        return {'loss': (labels - logits).square().mean(), 'logits': logits}


class MeasuredWer(Trainer):
    """A fixed dev trajectory separates checkpoint-selection logic from decoder quality."""
    def evaluate(self, **kwargs):
        metrics = {'eval_wer': {1: 1.0, 2: .5, 3: .5, 4: .75}[self.state.global_step]}
        self.log(metrics.copy())
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics


def test_early_stop_and_resume_best_checkpoint(tmp_path, monkeypatch):
    from transformers import trainer as hf_trainer
    from transformers.trainer_callback import DefaultFlowCallback, PrinterCallback

    # Swift replaces these module globals when imported, while this test exercises plain HF Trainer.
    monkeypatch.setattr(hf_trainer, 'DEFAULT_CALLBACKS', [DefaultFlowCallback])
    monkeypatch.setattr(hf_trainer, 'PrinterCallback', PrinterCallback)
    data = [dict(input_ids=torch.tensor([float(i)]), labels=torch.tensor([i / 3])) for i in range(8)]

    def run(output, stop=None, resume=None):
        set_seed(7)
        model = Regressor()
        args = TrainingArguments(output_dir=str(output), use_cpu=True, max_steps=6,
                                 per_device_train_batch_size=1, eval_strategy='steps', save_strategy='steps',
                                 eval_steps=1, save_steps=1, logging_steps=1, save_total_limit=4, seed=7,
                                 metric_for_best_model='wer', greater_is_better=False, report_to='none',
                                 disable_tqdm=True)
        trainer = MeasuredWer(model=model, args=args, train_dataset=data, eval_dataset=data,
                              callbacks=[Schedule(2, stop)])
        trainer.train(resume_from_checkpoint=resume)
        return trainer

    full = run(tmp_path / 'full')
    interrupted = run(tmp_path / 'restart', stop=3)
    assert interrupted.state.global_step == 3
    resumed = run(tmp_path / 'restart', resume=str(tmp_path / 'restart/checkpoint-3'))
    for t in (full, resumed):
        assert t.state.global_step == 4
        assert t.state.best_metric == .5
        best = Path(t.state.best_model_checkpoint)
        assert json.loads((best / 'trainer_state.json').read_text())['global_step'] == 2
        assert select(t.state.log_history, 2) == (2, .5, True)
    for a, b in zip(full.model.parameters(), resumed.model.parameters(), strict=True):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    for key in ('exp_avg', 'exp_avg_sq', 'step'):
        for left, right in zip(full.optimizer.state.values(), resumed.optimizer.state.values(), strict=True):
            torch.testing.assert_close(left[key], right[key], atol=0, rtol=0)


def test_trajectory_records_order_and_resumes_prefix(tmp_path):
    from types import SimpleNamespace
    from experiments.ms_swift_librispeech.train import InputTrajectory

    def batch(label):
        return {'input_ids': torch.tensor([[1, 2]]), 'labels': torch.tensor([[-100, label]]),
                'feature_attention_mask': torch.tensor([[1, 0]])}

    original = InputTrajectory()
    original.record(batch(2))
    cp = tmp_path / 'checkpoint-1'
    cp.mkdir()
    original.on_save(SimpleNamespace(output_dir=tmp_path), SimpleNamespace(global_step=1), None)
    resumed = InputTrajectory(cp)
    for i in range(9):
        original.record(batch(i))
        resumed.record(batch(i))
    assert len(resumed.batches) == 8 and original.batches == resumed.batches
    assert resumed.batches[0] != resumed.batches[1]
