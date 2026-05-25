import torch
import torch.nn.functional as F


def swiglu_mlp(
    x: torch.Tensor, w_gate: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor
) -> torch.Tensor:
    gate = x @ w_gate.to(x.dtype).T
    up = x @ w_up.to(x.dtype).T
    return (F.silu(gate) * up) @ w_down.to(x.dtype).T
