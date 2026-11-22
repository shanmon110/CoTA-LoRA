"""Load a hashed real mixed-bit decoder; unpack one linear weight on demand."""
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import load_file


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def unpack_cpu(packed, offsets, bits, scales, book):
    cols = torch.arange(scales.shape[1]*64, device=packed.device)
    b = bits.long()[:, None]
    where = offsets[:-1, None].long() + cols * b // 8
    switch = ((cols * b) % 8)
    codes = (packed[where].long() >> switch) & ((1 << b) - 1)
    return book.gather(1, codes) * scales[:, cols//64]


@lru_cache(maxsize=None)
def require_cuda_parity(device_index):
    """Validate the installed kernel before its first use on each process/device."""
    import triton
    from . import paper_middle_lowra_triton as kernel

    device = torch.device('cuda',device_index)
    width = 192  # partially filled Triton tile
    generator = torch.Generator().manual_seed(9)
    bits = torch.tensor([1,2,4]*4,dtype=torch.uint8)
    offsets = F.pad((bits.int()*width//8).cumsum(0,dtype=torch.int32),(1,0))
    packed = torch.randint(0,256,(int(offsets[-1]),),dtype=torch.uint8,generator=generator)
    scales = torch.rand((len(bits),width//64),generator=generator)
    book = torch.randn((len(bits),16),generator=generator)
    cpu = (packed,offsets,bits,scales,book)
    gpu = tuple(t.to(device) for t in cpu)
    dense = unpack_cpu(*cpu).to(device)
    for dtype in (torch.float32,torch.bfloat16):
        target = dense.to(dtype)
        class Probe:
            bias = None
            def dequant(self, dt):
                return kernel.unpack(*gpu,dt)
        # The first call can happen in inference_mode during eval. Establish a real
        # autograd path independently of the enclosing Trainer/Generation context.
        with torch.inference_mode(False), torch.enable_grad(), torch.autocast('cuda',enabled=False):
            probe = Probe()
            torch.testing.assert_close(probe.dequant(dtype),target,rtol=0,atol=0)
            x = (torch.arange(width,device=device).to(dtype)/128).reshape(1,1,-1).requires_grad_()
            reference_x = x.detach().clone().requires_grad_()
            actual = _LowBitLinear.apply(x,probe)
            expected = F.linear(reference_x,target.detach().clone())
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            torch.testing.assert_close(torch.autograd.grad(actual.sum(),x)[0],
                                       torch.autograd.grad(expected.sum(),reference_x)[0],rtol=0,atol=0)
    marker = dict(passed=True,device=torch.cuda.get_device_name(device_index),capability=torch.cuda.get_device_capability(device_index),
                  torch_version=str(torch.__version__),triton_version=triton.__version__,kernel_sha256=file_hash(kernel.__file__),
                  payload_bits=[1,2,4],dtypes=['float32','bfloat16'],exact_forward_and_input_gradient=True)
    print('LowRA CUDA parity '+json.dumps(marker,sort_keys=True),flush=True)
    return marker


class PackedProjection(nn.Module):
    def __init__(self, linear, values):
        super().__init__()
        self.in_features, self.out_features = linear.in_features, linear.out_features
        if (values['offsets'].shape != (linear.out_features+1,) or values['book'].shape != (linear.out_features,16)
                or values['scales'].shape != (linear.out_features,linear.in_features//64)
                or values['bits'].shape != (linear.out_features,) or values['packed'].dtype != torch.uint8):
            raise ValueError('Packed LowRA base does not match target linear')
        offsets, bits = values['offsets'], values['bits'].to(torch.long)
        if (not ((bits == 1) | (bits == 2) | (bits == 4)).all() or offsets[0] != 0
                or offsets[-1] != values['packed'].numel() or not torch.equal(offsets.diff(), bits*linear.in_features//8)):
            raise ValueError('Invalid LowRA row bit widths or packed offsets')
        for key in ('packed','offsets','bits','scales','book'):
            self.register_buffer(key, values[key].to(linear.weight.device))
        self.register_buffer('bias', None if linear.bias is None else linear.bias.detach())

    def dequant(self, dtype):
        if self.packed.is_cuda:
            from .paper_middle_lowra_triton import unpack
            require_cuda_parity(self.packed.device.index)
            return unpack(self.packed, self.offsets, self.bits, self.scales, self.book, dtype)
        return unpack_cpu(self.packed,self.offsets,self.bits,self.scales,self.book).to(dtype)

    def forward(self, x):
        return _LowBitLinear.apply(x, self)


class _LowBitLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, layer):
        ctx.layer = layer
        ctx.dtype = x.dtype
        weight = layer.dequant(x.dtype)
        bias = None if layer.bias is None else layer.bias.to(x.dtype)
        return F.linear(x, weight, bias)

    @staticmethod
    def backward(ctx, grad):
        # Re-create the frozen weight only for this matmul. Retaining every full-width
        # reconstructed weight from the forward pass would undermine low-bit training.
        weight = ctx.layer.dequant(grad.dtype)
        return torch.matmul(grad, weight).to(ctx.dtype), None


def bind_lowra(model, spec):
    root = Path(os.environ.get('SWIFT_LOWRA_BASE', spec['settings']['quantized_base']))
    indexfile = root/'index.json'
    identity = file_hash(indexfile)
    if spec['settings']['quantized_base_sha256'] != identity or spec.get('lowra_bundle_sha256',identity) != identity:
        raise ValueError('LowRA packed base index changed since training')
    spec['lowra_bundle_sha256'] = identity
    index = json.loads(indexfile.read_text())
    cfg, records = index['protocol'], index['projections']
    if (set(records) != set(spec['targets']) or cfg['rank'] != spec['rank'] or cfg['alpha'] != spec['alpha']
            or cfg['bpp'] != spec['settings']['bpp'] or cfg['base'] != spec['settings']['base_revision']):
        raise ValueError('LowRA base subset, snapshot, precision or rank differs from training configuration')
    for path in spec['targets']:
        item = records[path]
        basefile = root/item['file']
        if file_hash(basefile) != item['sha256']:
            raise ValueError(f'LowRA base shard checksum mismatch at {path}')
        values = load_file(str(basefile))
        wrapper = model.get_submodule(path)
        original = wrapper.base_layer
        wrapper.down = nn.Parameter(values['down'].to(original.weight.device))
        wrapper.up = nn.Parameter(values['up'].to(original.weight.device))
        wrapper.base_layer = PackedProjection(original, values)
