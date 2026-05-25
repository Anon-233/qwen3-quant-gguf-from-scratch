import torch

from miniqwen.quant.block_q4_0 import (
    dequantize_q4_0,
    pack_int4_signed,
    quantize_q4_0,
    unpack_int4_signed,
)


def test_q4_0_pack_unpack_correct():
    q = torch.tensor([-8, -7, -1, 0, 1, 7], dtype=torch.int8)
    packed = pack_int4_signed(q)
    assert torch.equal(unpack_int4_signed(packed, q.numel()), q)


def test_q4_0_dequant_no_nan():
    x = torch.randn(3, 11)
    y = dequantize_q4_0(quantize_q4_0(x, block_size=32))
    assert y.shape == x.shape
    assert not torch.isnan(y).any()
