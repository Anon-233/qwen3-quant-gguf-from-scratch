import torch

from miniqwen.gguf.reader import GGUFReader


class TensorStore:
    def __init__(self, reader: GGUFReader):
        self.reader = reader

    def to_state_dict(
        self,
        device: torch.device | str = "cpu",
        storage_dtype: torch.dtype | None = None,
    ) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for name in self.reader.tensors:
            tensor = self.reader.get_tensor(name, dequantize=True)
            if storage_dtype is not None and torch.is_floating_point(tensor):
                tensor = tensor.to(dtype=storage_dtype)
            out[name] = tensor.to(device=device)
        return out
