import numpy as np
import torch
import pytest
from torch import nn

from experiments.ms_swift_aqa.paper_middle_lowra import assign_bits, fit_books, pack_weight
from experiments.ms_swift_aqa.paper_middle_lowra_runtime import PackedProjection, bind_lowra, require_cuda_parity


def test_mixed_packed_bytes_error_and_allocation():
    rng = np.random.default_rng(17)
    matrix = rng.normal(size=(16,128)).astype('float32')
    matrix[3,:64] = 0
    threads = torch.get_num_threads()
    maps, edges, sse = fit_books(matrix)
    assert torch.get_num_threads() == threads
    widths = np.tile([1,2,4,1],4).astype('uint8')
    packed, offsets, scales, book, approx = pack_weight(matrix, widths, maps, edges)
    assert torch.get_num_threads() == threads
    assert packed.dtype == np.uint8 and packed.size*8 == widths.sum()*128
    for row, b in enumerate(widths.tolist()):
        for col in range(128):
            code = (int(packed[offsets[row] + col//(8//b)]) >> ((col%(8//b))*b)) & ((1<<b)-1)
            np.testing.assert_allclose(approx[row,col],book[row,code]*scales[row,col//64])
        measured = np.square(matrix[row]-approx[row]).sum()
        np.testing.assert_allclose(measured,sse[row,{1:0,2:1,4:2}[b]],rtol=2e-5)
    mixed = assign_bits(sse,1.75,4,13)
    assert set(mixed).issubset({1,2,4}) and 1 < np.mean(mixed) <= 1.75


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(),reason='requires CUDA'))])
def test_real_packed_forward_backward_and_cuda_kernel(device):
    if device == 'cuda':
        # Exactly the same fail-closed gate runs before real training, on every retry
        # and eval process. Exercise inference_mode, as in a clean evaluator reload.
        with torch.inference_mode():
            marker = require_cuda_parity(torch.cuda.current_device())
        assert marker['passed'] and marker['exact_forward_and_input_gradient']
        return
    generator = np.random.default_rng(9)
    matrix = generator.normal(size=(12,192)).astype('float32')
    maps, edges, _ = fit_books(matrix)
    bits = np.tile([1,2,4],4).astype('uint8')
    packed, offsets, scales, book, approx = pack_weight(matrix,bits,maps,edges)
    data = {k:torch.from_numpy(v) for k,v in dict(packed=packed,offsets=offsets,
                                                 scales=scales,book=book,bits=bits).items()}
    layer = PackedProjection(nn.Linear(192,12,bias=False).to(device), data)
    expected_weight = torch.tensor(approx,device=device)
    torch.testing.assert_close(expected_weight, layer.dequant(torch.float32), rtol=0,atol=0)
    torch.testing.assert_close(expected_weight.bfloat16(), layer.dequant(torch.bfloat16), rtol=0,atol=0)
    x = torch.randn(2,5,192,device=device,requires_grad=True)
    ref = x.detach().clone().requires_grad_(True)
    y = layer(x)
    expected = torch.nn.functional.linear(ref,expected_weight)
    torch.testing.assert_close(y,expected)
    y.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(x.grad,ref.grad)


def test_changed_packed_base_fails_before_loading_weights(tmp_path, monkeypatch):
    (tmp_path/'index.json').write_text('{}')
    monkeypatch.setenv('SWIFT_LOWRA_BASE',str(tmp_path))
    with pytest.raises(ValueError,match='changed since training'):
        bind_lowra(nn.Identity(),{'settings':{'quantized_base':'unused','quantized_base_sha256':'0'*64}})
