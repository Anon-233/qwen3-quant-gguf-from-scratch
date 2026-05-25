import torch


def torch_dtype_from_string(name: str) -> torch.dtype:
    table = {
        "float16": torch.float16,
        "torch.float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "torch.bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "torch.float32": torch.float32,
    }
    if name not in table:
        raise ValueError(f"Unsupported dtype string: {name}")
    return table[name]
