from __future__ import annotations

import torch


def symmetric_quantize(
    x: torch.Tensor, num_bits: int = 8
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    if num_bits < 2:
        raise ValueError("num_bits must be >= 2")
    qmax = 2 ** (num_bits - 1) - 1
    qmin = -(2 ** (num_bits - 1))
    max_abs = x.detach().abs().max()
    scale = torch.where(max_abs == 0, torch.ones_like(max_abs), max_abs / qmax)
    q = torch.clamp(torch.round(x / scale), qmin, qmax).to(torch.int32)
    return q, scale, qmin, qmax


def dequantize_symmetric(q: torch.Tensor, scale: torch.Tensor | float) -> torch.Tensor:
    return q.to(torch.float32) * torch.as_tensor(scale, dtype=torch.float32, device=q.device)


def affine_quantize(
    x: torch.Tensor, num_bits: int = 8
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    qmin = 0
    qmax = 2**num_bits - 1
    xmin = x.detach().min()
    xmax = x.detach().max()
    scale = torch.where(xmax == xmin, torch.ones_like(xmax), (xmax - xmin) / (qmax - qmin))
    zero_point = torch.clamp(torch.round(qmin - xmin / scale), qmin, qmax)
    q = torch.clamp(torch.round(x / scale + zero_point), qmin, qmax).to(torch.int32)
    return q, scale, zero_point, qmin, qmax


def dequantize_affine(
    q: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor
) -> torch.Tensor:
    return (q.to(torch.float32) - zero_point.to(torch.float32)) * scale.to(torch.float32)
