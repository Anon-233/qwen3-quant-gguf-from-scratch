import torch


def rope_frequencies(
    head_dim: int, seq_len: int, theta: float, device: torch.device
) -> torch.Tensor:
    if head_dim % 2:
        raise ValueError("RoPE head_dim must be even")
    idx = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (theta ** (idx / head_dim))
    positions = torch.arange(seq_len, dtype=torch.float32, device=device)
    return torch.outer(positions, inv_freq)


def apply_rope(x: torch.Tensor, position_offset: int, theta: float) -> torch.Tensor:
    # x: [batch, seq, heads, head_dim]
    _bsz, seq_len, _heads, head_dim = x.shape
    freqs = rope_frequencies(head_dim, seq_len + position_offset, theta, x.device)[position_offset:]
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos()[None, :, None, :]
    sin = emb.sin()[None, :, None, :]
    x_float = x.to(torch.float32)
    first, second = x_float[..., : head_dim // 2], x_float[..., head_dim // 2 :]
    rotated = torch.cat((-second, first), dim=-1)
    return (x_float * cos + rotated * sin).to(x.dtype)
