from __future__ import annotations

from miniqwen.config import Qwen3Config
from miniqwen.gguf.reader import GGUFReader
from miniqwen.runtime.device import resolve_compute_dtype, resolve_device
from miniqwen.runtime.executor import RuntimeExecutor
from miniqwen.runtime.tensor_store import TensorStore


def load_gguf_runtime(
    path: str,
    device: str = "auto",
    compute_dtype: str | None = "auto",
) -> RuntimeExecutor:
    reader = GGUFReader(path)
    config = Qwen3Config.from_metadata(reader.metadata)
    resolved_device = resolve_device(device)
    resolved_dtype = resolve_compute_dtype(compute_dtype, resolved_device)
    storage_dtype = resolved_dtype if resolved_device.type == "cuda" else None
    state = TensorStore(reader).to_state_dict(device=resolved_device, storage_dtype=storage_dtype)
    return RuntimeExecutor(config, state, device=resolved_device, compute_dtype=resolved_dtype)
