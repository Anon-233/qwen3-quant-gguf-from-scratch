from __future__ import annotations

from dataclasses import dataclass

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


@dataclass(slots=True)
class RoPECache:
    """Pre-computed cos/sin tables for all positions up to max_seq_len."""

    cos: torch.Tensor  # [max_seq_len, head_dim]
    sin: torch.Tensor  # [max_seq_len, head_dim]

    @staticmethod
    def precompute(
        head_dim: int, max_seq_len: int, theta: float, device: torch.device | str = "cpu"
    ) -> RoPECache:
        freqs = rope_frequencies(head_dim, max_seq_len, theta, torch.device(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        return RoPECache(cos=emb.cos(), sin=emb.sin())

    def slice(
        self, seq_len: int, position_offset: int, device: torch.device | str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (cos, sin) for positions [offset, offset+seq_len)."""
        max_seq_len = int(self.cos.shape[0])
        requested = position_offset + seq_len
        if requested > max_seq_len:
            raise ValueError(
                "RoPE cache capacity exceeded: "
                f"position_offset={position_offset}, seq_len={seq_len}, "
                f"requested_total={requested}, max_seq_len={max_seq_len}. "
                "Increase max_position_embeddings or reduce the requested context length."
            )
        c = self.cos[position_offset : position_offset + seq_len].to(device)
        s = self.sin[position_offset : position_offset + seq_len].to(device)
        return c[None, :, None, :], s[None, :, None, :]


def apply_rope(
    x: torch.Tensor,
    position_offset: int,
    theta: float,
    rope_cache: RoPECache | None = None,
) -> torch.Tensor:
    # x: [batch, seq, heads, head_dim]
    _bsz, seq_len, _heads, head_dim = x.shape
    if rope_cache is not None:
        cos, sin = rope_cache.slice(seq_len, position_offset, x.device)
    else:
        freqs = rope_frequencies(
            head_dim, seq_len + position_offset, theta, x.device
        )[position_offset:]
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()[None, :, None, :]
        sin = emb.sin()[None, :, None, :]
    x_float = x.to(torch.float32)
    first, second = x_float[..., : head_dim // 2], x_float[..., head_dim // 2 :]
    rotated = torch.cat((-second, first), dim=-1)
    return (x_float * cos + rotated * sin).to(x.dtype)
