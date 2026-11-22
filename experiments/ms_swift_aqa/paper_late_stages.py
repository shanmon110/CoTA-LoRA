"""Shared source-only preparation for paper algorithms with an auxiliary SFT stage."""
from contextlib import contextmanager
import time

import torch


@contextmanager
def stage_session(trainer):
    model = trainer.model
    prev = model.training
    model.train()
    trainer.template.register_post_encode_hook([model])
    device = next(model.parameters()).device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    measurements = {}
    try:
        yield measurements
    finally:
        trainer.template.remove_post_encode_hook()
        model.zero_grad(set_to_none=True)
        model.train(prev)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            measurements["stage_peak_allocated"] = torch.cuda.max_memory_allocated(device)
            measurements["stage_peak_reserved"] = torch.cuda.max_memory_reserved(device)
        measurements["stage_seconds"] = time.perf_counter() - start


def stage_loss(trainer, indices):
    batch = trainer.data_collator([trainer.train_dataset[i] for i in indices])
    inputs = trainer._prepare_inputs(batch)
    with trainer.template.forward_context(trainer.model, inputs), trainer.compute_loss_context_manager():
        loss = trainer.compute_loss(trainer.model, inputs)
    if not torch.isfinite(loss.detach()).item():
        raise FloatingPointError("Nonfinite paper preparation loss")
    return loss


def train_indices(trainer, count):
    length = len(trainer.train_dataset)
    if count <= 0 or count > length:
        raise ValueError(f"Requested {count} distinct examples from a training set of {length}")
    generator = torch.Generator().manual_seed(trainer.args.data_seed)
    return torch.randperm(length, generator=generator)[:count].tolist()
