"""Mixed-row 1/2/4-bit unpack, per-channel codebook and 64-element absmax."""
import torch
import triton
import triton.language as tl


@triton.jit
def _unpack(q, offsets, bits, scales, book, out, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0,BLOCK)
    valid = col < WIDTH
    p = tl.load(bits + row).to(tl.int32)
    start = tl.load(offsets + row)
    pos = col * p
    value = tl.load(q + start + pos//8, mask=valid, other=0).to(tl.int32)
    code = (value >> (pos % 8)) & ((1 << p) - 1)
    center = tl.load(book + row*16 + code)
    scale = tl.load(scales + row*(WIDTH//64) + col//64, mask=valid, other=0)
    tl.store(out + row*WIDTH + col, center*scale, mask=valid)


def unpack(q, offsets, bits, scales, book, dtype):
    rows, width = len(bits), scales.shape[1]*64
    result = torch.empty(rows, width, device=q.device, dtype=dtype)
    _unpack[(rows, triton.cdiv(width, 256))](q,offsets,bits,scales,book,result,width,256)
    return result
