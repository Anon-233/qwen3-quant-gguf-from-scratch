from pathlib import Path

import torch


def load_safetensors_file(path: str | Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    return load_file(str(path), device="cpu")
