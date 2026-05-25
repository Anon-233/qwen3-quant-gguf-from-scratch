import torch
import torch.nn.functional as F


def mse(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.mean((x.to(torch.float32) - y.to(torch.float32)) ** 2)


def cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(x.flatten().to(torch.float32), y.flatten().to(torch.float32), dim=0)


def sqnr_db(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    signal = torch.mean(x.to(torch.float32) ** 2)
    noise = mse(x, y)
    return 10 * torch.log10(signal / torch.clamp(noise, min=1e-12))


def topk_overlap(a: torch.Tensor, b: torch.Tensor, k: int = 10) -> float:
    ia = set(torch.topk(a.flatten(), k).indices.tolist())
    ib = set(torch.topk(b.flatten(), k).indices.tolist())
    return len(ia & ib) / k
