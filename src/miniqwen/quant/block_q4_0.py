from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(slots=True)
class BlockQ4Tensor:
    packed: torch.Tensor
    scales: torch.Tensor
    shape: tuple[int, ...]
    block_size: int = 32


def pack_int4_signed(q: torch.Tensor) -> torch.Tensor:
    q = torch.clamp(q.to(torch.int16), -8, 7)
    nibbles = (q & 0x0F).to(torch.uint8).flatten()
    if nibbles.numel() % 2:
        nibbles = F.pad(nibbles, (0, 1))
    low = nibbles[0::2]
    high = nibbles[1::2] << 4
    return low | high


def unpack_int4_signed(packed: torch.Tensor, count: int) -> torch.Tensor:
    packed = packed.to(torch.uint8).flatten()
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    out = torch.empty(packed.numel() * 2, dtype=torch.int16)
    out[0::2] = low.to(torch.int16)
    out[1::2] = high.to(torch.int16)
    out = out[:count]
    out = torch.where(out >= 8, out - 16, out)
    return out.to(torch.int8)


def quantize_q4_0(x: torch.Tensor, block_size: int = 32) -> BlockQ4Tensor:
    if block_size <= 0 or block_size % 2:
        raise ValueError("block_size must be a positive even integer")
    flat = x.detach().to(torch.float32).flatten()
    pad = (-flat.numel()) % block_size
    if pad:
        flat = F.pad(flat, (0, pad))
    blocks = flat.view(-1, block_size)
    max_abs = blocks.abs().amax(dim=1)
    scales = torch.where(max_abs == 0, torch.ones_like(max_abs), max_abs / 7.0)
    q = torch.clamp(torch.round(blocks / scales[:, None]), -8, 7).to(torch.int8)
    return BlockQ4Tensor(
        packed=pack_int4_signed(q.flatten()).cpu(),
        scales=scales.to(torch.float32).cpu(),
        shape=tuple(x.shape),
        block_size=block_size,
    )


def dequantize_q4_0(t: BlockQ4Tensor) -> torch.Tensor:
    count = t.scales.numel() * t.block_size
    q = unpack_int4_signed(t.packed, count).to(torch.float32).view(-1, t.block_size)
    x = q * t.scales.to(torch.float32)[:, None]
    return x.flatten()[: int(torch.tensor(t.shape).prod().item())].view(t.shape)
