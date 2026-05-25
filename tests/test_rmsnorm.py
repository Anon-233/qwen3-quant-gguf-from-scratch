import torch

from miniqwen.model.norm import rms_norm


def test_rmsnorm_matches_reference():
    x = torch.randn(2, 3, 4)
    w = torch.randn(4)
    y = rms_norm(x, w, 1e-6)
    ref = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6) * w
    assert torch.allclose(y, ref, atol=1e-6)
