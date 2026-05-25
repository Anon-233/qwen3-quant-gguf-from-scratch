import torch


def embedding(input_ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return weight[input_ids]
