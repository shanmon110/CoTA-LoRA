"""Replay two TA microbatches in a different autograd thread on CPU."""

import copy
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from threading import get_ident

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from examples.omni_talora import attention
if __package__:
    from .test_omni_ta_attention import run_attn, tiny_thinker
else:
    from test_omni_ta_attention import run_attn, tiny_thinker


@pytest.mark.parametrize("worker_has_layout", [False, True])
def test_checkpoint_replays_its_own_audio_layout_in_worker_and_restores_context(worker_has_layout):
    torch.manual_seed(88)
    eager = tiny_thinker()
    adapter = attention.install_adapter(eager, last_n=1, rank=2, alpha=4.)[0].ta_adapter
    with torch.no_grad():
        adapter.up.normal_()
        adapter.gate.fill_(.75)
    replay = copy.deepcopy(eager)
    main_thread = get_ident()
    calls = []
    layouts = (torch.tensor([[5, 99, 99, 6]]), torch.tensor([[99, 5, 6, 7]]))
    xs = [torch.randn(1, 4, 12) for _ in layouts]
    valid = torch.ones_like(layouts[0])

    def layer(x):
        calls.append(get_ident())
        return run_attn(replay.model.layers[0].self_attn, x, valid)[0]

    outputs = []
    for x, ids in zip(xs, layouts):
        with attention.audio_context(ids, valid, 99):
            baseline = run_attn(eager.model.layers[0].self_attn, x, valid)[0]
            baseline.square().sum().backward()
            out = checkpoint(layer, x, use_reentrant=False, context_fn=attention.checkpoint_audio_context)
            torch.testing.assert_close(out, baseline, rtol=0, atol=0)
            outputs.append(out)

    def backward():
        # A worker can have an unrelated layout. Each graph must restore
        # its own layout for replay and then leave this outer scope intact.
        outer_scope = attention.audio_context(torch.zeros_like(valid), valid, 99) if worker_has_layout else nullcontext()
        with outer_scope:
            outer = attention._layout.get()
            for out in outputs:
                out.square().sum().backward()
                assert attention._layout.get() is outer
        assert attention._layout.get() is None
        return get_ident()

    with ThreadPoolExecutor(max_workers=1) as pool:
        worker = pool.submit(backward).result()
    assert main_thread != worker and calls == [main_thread, main_thread, worker, worker]
    for name in ("up", "down", "gate"):
        reference = getattr(adapter, name).grad
        assert reference.abs().sum() > 0
        actual = getattr(replay.model.layers[0].self_attn.ta_adapter, name).grad
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
