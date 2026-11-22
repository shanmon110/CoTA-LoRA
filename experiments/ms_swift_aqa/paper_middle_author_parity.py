"""Replay projections against pinned author source files with nonzero adapters on CPU."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import torch
from peft.tuners import _buffer_dict
from torch import nn
from torch.nn import functional as F

from .paper_middle import BUILDERS


SOURCES = {
    'aurora': ('aurora/commonsense_reasoning/peft/src/peft/tuners/ANL.py',
               '815e070d231c7febb01bcfa8699cf1b994f71d960bd276f61e7c19a2afb781ee'),
    'gralora': ('gralora/peft/src/peft/tuners/gralora/layer.py',
                '425202473cb6fb3951e0af90b55ee34304a73070c1fe55f6ddd8aa1244ef8b63'),
    'magnitude-loram': ('magnitude-loram/layer.py',
                        '13184af97958ab892fae466d11a7c7b2829f434e1975d7011b7df885d1c9d8d6'),
    'toplora': ('toplora/mypeft/toplora.py',
                'd283be853d497284c1f1cd205d2f6f25b4aced43e9093e9ef29de9fe9f4ea6fe'),
}


def replay(root, method):
    rel, checksum = SOURCES[method]
    path = root/rel
    if hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
        raise ValueError(f'Unexpected upstream revision of {path}')
    # The Magnitude-LoRAM author file uses package-relative PEFT imports.
    spec = importlib.util.spec_from_file_location('peft.tuners.lora.upstream_probe',path)
    author = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(author)
    torch.manual_seed(124)
    rank, alpha = 4, 8
    original = nn.Linear(12,8)
    adapted = BUILDERS[method](copy.deepcopy(original),rank,alpha,{})
    inp = torch.randn(2,5,12).requires_grad_()
    common = dict(base_layer=original,adapter_name='default',r=rank)
    if method == 'aurora':
        ref = author.ANL(rank,rank)
        with torch.no_grad():
            ref.base_weight.copy_(adapted.fixed_weight)
            ref.spline_weight.copy_(adapted.spline_weight)
            adapted.up.normal_()
        expected = original(inp) + 2*F.linear(ref(F.linear(inp,adapted.down)),adapted.up)
    elif method == 'gralora':
        # The author fork predates PEFT's move of BufferDict out of vera/.
        with patch.dict('sys.modules',{'peft.tuners.vera.buffer_dict':_buffer_dict}):
            ref = author.Linear(module_name='probe',gralora_alpha=alpha,**common)
        with torch.no_grad():
            adapted.up.normal_()
            ref.gralora_A['default'].copy_(adapted.down)
            ref.gralora_B['default'].copy_(adapted.up)
        expected = ref(inp)
    elif method == 'toplora':
        ref = author.TopLinear(lora_alpha=alpha,**common)
        with torch.no_grad():
            adapted.branch.up.normal_()
            ref.lora_A.default.weight.copy_(adapted.branch.down)
            ref.lora_B.default.weight.copy_(adapted.branch.up)
            ref.lora_lambda.default.weight.copy_(adapted.branch.projector.T)
            ref.lora_lambda.default.rms_norm.weight.copy_(adapted.branch.rms_weight)
        expected = ref(inp)
    else:
        ref = author.Linear(lora_alpha=alpha,init_lora_weights='loram',**common)
        torch.testing.assert_close(adapted.down,ref.lora_A.default.weight)
        torch.testing.assert_close(adapted.up,ref.lora_B.default.weight)
        torch.testing.assert_close(adapted.base_layer.weight,original.weight)
        # Leave the exactly-compensating initial point and exercise a trained update.
        with torch.no_grad():
            shift = torch.randn_like(adapted.up)
            adapted.up.add_(shift)
            ref.lora_B.default.weight.add_(shift)
        expected = ref(inp)
    actual = adapted(inp)
    expected_grad = torch.autograd.grad(expected.square().mean(),inp)[0]
    grad = torch.autograd.grad(actual.square().mean(),inp)[0]
    torch.testing.assert_close(actual,expected)
    torch.testing.assert_close(grad,expected_grad)
    return dict(author_sha256=checksum,output_max_abs=(actual-expected).abs().max().item(),
                input_grad_max_abs=(grad-expected_grad).abs().max().item())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources',type=Path,required=True)
    args = parser.parse_args()
    print(json.dumps({name:replay(args.sources,name) for name in SOURCES},indent=2))
