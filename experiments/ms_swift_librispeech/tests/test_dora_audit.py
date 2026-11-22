"""Exercise the shared initialization with native PEFT's per-output DoRA magnitudes."""
import copy

import pytest
import torch
from peft import LoraConfig, get_peft_model
from torch import nn

from experiments.ms_swift_librispeech.adapter_audit import audit_adapter, initialize_pair


def test_dora_init_keeps_native_magnitudes_and_matches_all_lora_factors():
    base = nn.Module()
    base.thinker = nn.Module()
    base.thinker.model = nn.Module()
    layer = nn.Module()
    layer.self_attn = nn.ModuleDict({f'{name}_proj': nn.Linear(32, width, bias=False)
                                     for name, width in [('q', 32), ('k', 16), ('v', 16), ('o', 32)]})
    layer.mlp = nn.ModuleDict({'gate_proj': nn.Linear(32, 64, bias=False), 'up_proj': nn.Linear(32, 64, bias=False),
                               'down_proj': nn.Linear(64, 32, bias=False)})
    base.thinker.model.layers = nn.ModuleList([layer, copy.deepcopy(layer)])
    base.to(torch.bfloat16)
    targets = tuple(layer.self_attn.keys()) + tuple(layer.mlp.keys())
    args = dict(r=2, lora_alpha=4, target_modules=list(targets))
    lora = get_peft_model(copy.deepcopy(base), LoraConfig(**args))
    dora = get_peft_model(base, LoraConfig(**args, use_dora=True))
    mag = [p for n, p in dora.named_parameters() if 'lora_magnitude_vector' in n]
    native = [p.detach().clone() for p in mag]

    expected = initialize_pair(lora, 'lora', 42, 2, targets)
    actual = initialize_pair(dora, 'dora', 42, 2, targets)
    assert actual['ab_init_sha256'] == expected['ab_init_sha256']
    assert actual['trainable_parameters'] - expected['trainable_parameters'] == actual['magnitude_parameters'] == 512
    assert len(mag) == 14 and all(torch.equal(x, y) for x, y in zip(mag, native, strict=True))
    with torch.no_grad():
        mag[0].add_(0.1)
    changed = audit_adapter(dora, 'dora', 2, targets)
    assert changed['magnitude_sha256'] != actual['magnitude_sha256']
    assert changed['ab_sha256'] == actual['ab_init_sha256']
    mag[0].requires_grad_(False)
    with pytest.raises(ValueError, match='trainable'):
        audit_adapter(dora, 'dora', 2, targets)
    with pytest.raises(ValueError):
        audit_adapter(lora, 'dora', 2, targets)
