from pathlib import Path
from unittest.mock import patch
import hashlib
import importlib.util
import sys
import types

import pytest
import torch
from torch import nn
from peft import PeftModel

from experiments.ms_swift_aqa.papers28.known_target import factors, write_adapter, tensor_names
from experiments.ms_swift_aqa.papers28.expressive import FILES


def test_depth_one_matches_author_reference_and_svd_tail():
    g = torch.Generator().manual_seed(34)
    frozen, target = [torch.randn(7, 7, generator=g) for _ in range(2)]
    a, b, relative, total, retained = factors(frozen, target, 3, 6)
    # Direct expansion of upstream our_construction for depth=1 (the two inverses are I).
    delta = target - frozen
    _, _, v = torch.svd(delta)
    expected = (delta @ v[:, :3]) @ v.T[:3, :]
    torch.testing.assert_close(2 * b @ a, expected)
    torch.testing.assert_close((delta - 2 * b @ a).norm() / delta.norm(), torch.tensor(relative))
    assert relative == pytest.approx((1 - retained / total) ** .5, rel=1e-5)
    with pytest.raises(ValueError):
        factors(frozen[:5], target[:5], 3, 6)


def test_reference_file_matches_on_local_author_snapshot():
    path = Path('/root/autodl-tmp/r1-aqa/paper-envs/expressive/official/helper.py')
    if not path.is_file():
        pytest.skip('Author snapshot not installed')
    assert hashlib.sha256(path.read_bytes()).hexdigest() == FILES['helper.py']
    spec = importlib.util.spec_from_file_location('authors_construction', path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, wandb=types.ModuleType('wandb')):
        spec.loader.exec_module(module)
    torch.manual_seed(7204)
    for rank in (1, 4, 8):
        frozen, target = torch.randn(8,8), torch.randn(8,8)
        left, right = module.our_construction(target,[frozen],rank,False)
        a, b, *_ = factors(frozen,target,rank,2*rank)
        torch.testing.assert_close(2*b@a, left[0]@right[0].T, rtol=3e-5,atol=1e-5)


def test_native_clean_reload_reproduces_constructed_update(tmp_path):
    base = nn.Sequential(nn.Linear(6,6,bias=False),nn.Linear(6,6,bias=False))
    weight = base[0].weight.detach().clone()
    target = weight + torch.arange(36).reshape(6,6)/100
    a,b,*_ = factors(weight,target,2,4)
    # Generic PEFT model for CPU test. Actual Qwen adapters have task_type=CAUSAL_LM.
    write_adapter(tmp_path/'adapter', {'0':(a,b)}, 2,4,'tiny','test',task_type=None)
    model = PeftModel.from_pretrained(base,tmp_path/'adapter')
    x = torch.randn(5,6)
    expected = base[1](nn.functional.linear(x,weight + 2*b@a))
    torch.testing.assert_close(model(x),expected)
    assert not getattr(model.base_model.model[1],'lora_A',None)


def test_scope_requires_all_expected_matching_qo_and_no_other_layers():
    key = 'thinker.model.layers.0.self_attn.'
    keys = {key+x+'.weight' for x in ('q_proj','o_proj','k_proj','v_proj')}
    keys.add('thinker.audio_tower.layers.0.self_attn.q_proj.weight')
    assert set(tensor_names(keys,keys,1)) == {key+'q_proj.weight',key+'o_proj.weight'}
    with pytest.raises(ValueError):
        tensor_names(keys,keys-{key+'o_proj.weight'},1)


def test_publication_recovers_interrupted_directory_rename_and_rejects_changes(tmp_path):
    import json
    from experiments.ms_swift_aqa.papers28.known_target import published
    from experiments.ms_swift_aqa.tuner_suite import checkpoint_identity

    checkpoint=tmp_path/'checkpoint-constructed'
    a,b,*_=factors(torch.zeros(4,4),torch.eye(4),1,2)
    write_adapter(checkpoint,{'0':(a,b)},1,2,'tiny','test',task_type=None)
    selection={'step':32}
    expected=dict(teacher_selection=selection,protocol_sha256='abc',adapter_rank=1,adapter_alpha=2)
    record=dict(**expected,checkpoint_identity=checkpoint_identity(checkpoint,'lora'))
    pending=tmp_path/'.construction.pending.json'
    pending.write_text(json.dumps(record))
    assert published(tmp_path,expected)
    assert not pending.exists()
    assert json.loads((tmp_path/'construction.json').read_text())==record
    (checkpoint/'adapter_config.json').write_text((checkpoint/'adapter_config.json').read_text()+'\n')
    with pytest.raises(ValueError):
        published(tmp_path,expected)
