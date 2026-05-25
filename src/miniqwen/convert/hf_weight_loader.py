from __future__ import annotations

from pathlib import Path

import torch


def _resolve_model_dir(model_name_or_path: str) -> Path:
    path = Path(model_name_or_path)
    if path.exists():
        return path
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            model_name_or_path,
            allow_patterns=["*.safetensors", "*.json", "tokenizer*"],
        )
    )


def list_safetensors(model_name_or_path: str) -> list[Path]:
    model_dir = _resolve_model_dir(model_name_or_path)
    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors files found under {model_dir}")
    return files


def load_hf_state_dict(model_name_or_path: str) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    state: dict[str, torch.Tensor] = {}
    for file in list_safetensors(model_name_or_path):
        state.update(load_file(str(file), device="cpu"))
    return state


def tensor_summary(state_dict: dict[str, torch.Tensor]) -> list[dict]:
    return [
        {"name": name, "shape": tuple(t.shape), "dtype": str(t.dtype), "numel": int(t.numel())}
        for name, t in sorted(state_dict.items())
    ]
