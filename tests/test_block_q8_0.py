import torch

from miniqwen.quant.block_q8_0 import dequantize_q8_0, quantize_q8_0


def test_q8_0_quant_dequant_shape_correct():
    x = torch.randn(5, 7)
    q = quantize_q8_0(x, block_size=32)
    y = dequantize_q8_0(q)
    assert y.shape == x.shape


def test_q8_0_error_reasonable():
    x = torch.randn(5, 7)
    q = quantize_q8_0(x, block_size=32)
    y = dequantize_q8_0(q)
    assert torch.mean((x - y) ** 2) < 1e-4
