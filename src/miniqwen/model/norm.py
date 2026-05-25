import torch


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    variance = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    y = x.to(torch.float32) * torch.rsqrt(variance + eps)
    return (y * weight.to(torch.float32)).to(x.dtype)
