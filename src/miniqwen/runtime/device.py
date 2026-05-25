from __future__ import annotations

import torch


def resolve_device(device: str | torch.device = "cpu") -> torch.device:
    if str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    return dev


def resolve_compute_dtype(
    compute_dtype: str | torch.dtype | None,
    device: torch.device,
) -> torch.dtype:
    if isinstance(compute_dtype, torch.dtype):
        return compute_dtype
    if compute_dtype in {None, "auto"}:
        return torch.float16 if device.type == "cuda" else torch.float32
    table = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if compute_dtype not in table:
        raise ValueError(f"Unsupported compute dtype: {compute_dtype}")
    return table[compute_dtype]
