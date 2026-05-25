import torch

from miniqwen.quant.error_metrics import cosine_similarity, topk_overlap


def compare_logits(a: torch.Tensor, b: torch.Tensor, k: int = 10) -> dict:
    return {
        "max_abs": float((a - b).abs().max()),
        "mse": float(torch.mean((a - b).float() ** 2)),
        "cosine": float(cosine_similarity(a, b)),
        "topk_overlap": float(topk_overlap(a, b, k=k)),
    }
