from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(slots=True)
class BlockQ8Tensor:
    q: torch.Tensor
    scales: torch.Tensor
    shape: tuple[int, ...]
    block_size: int = 32

    @property
    def padded_numel(self) -> int:
        return int(self.q.numel())


def _pad_flat(x: torch.Tensor, block_size: int) -> tuple[torch.Tensor, int]:
    flat = x.detach().to(torch.float32).flatten()
    pad = (-flat.numel()) % block_size
    if pad:
        flat = F.pad(flat, (0, pad))
    return flat, pad


def quantize_q8_0(x: torch.Tensor, block_size: int = 32) -> BlockQ8Tensor:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    flat, _ = _pad_flat(x, block_size)
    blocks = flat.view(-1, block_size)
    max_abs = blocks.abs().amax(dim=1)
    scales = torch.where(max_abs == 0, torch.ones_like(max_abs), max_abs / 127.0)
    q = torch.clamp(torch.round(blocks / scales[:, None]), -128, 127).to(torch.int8)
    return BlockQ8Tensor(
        q=q.flatten().cpu(),
        scales=scales.to(torch.float32).cpu(),
        shape=tuple(x.shape),
        block_size=block_size,
    )


def dequantize_q8_0(t: BlockQ8Tensor) -> torch.Tensor:
    q = t.q.to(torch.float32).view(-1, t.block_size)
    x = q * t.scales.to(torch.float32)[:, None]
    return x.flatten()[: int(torch.tensor(t.shape).prod().item())].view(t.shape)
