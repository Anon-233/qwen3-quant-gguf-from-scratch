import torch

from miniqwen.model.rope import apply_rope


def test_rope_shape_correct():
    x = torch.randn(2, 5, 3, 4)
    y = apply_rope(x, 0, 10000.0)
    assert y.shape == x.shape
