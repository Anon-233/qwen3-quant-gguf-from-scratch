import torch

from miniqwen.quant.int_quant import dequantize_symmetric, symmetric_quantize


def test_symmetric_quantization_correct():
    x = torch.tensor([-1.0, 0.0, 1.0])
    q, scale, qmin, qmax = symmetric_quantize(x, 8)
    y = dequantize_symmetric(q, scale)
    assert qmin == -128
    assert qmax == 127
    assert q.tolist() == [-127, 0, 127]
    assert torch.allclose(y, x, atol=1e-6)
