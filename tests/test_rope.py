import pytest
import torch

from miniqwen.model.rope import RoPECache, apply_rope


def test_rope_shape_correct():
    x = torch.randn(2, 5, 3, 4)
    y = apply_rope(x, 0, 10000.0)
    assert y.shape == x.shape


def test_rope_cache_reports_context_overflow():
    cache = RoPECache.precompute(head_dim=4, max_seq_len=4, theta=10000.0)
    with pytest.raises(ValueError, match="RoPE cache capacity exceeded"):
        cache.slice(seq_len=1, position_offset=4, device="cpu")
