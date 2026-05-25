from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from miniqwen.gguf.constants import (
    DEFAULT_ALIGNMENT,
    DTYPE_Q4_0,
    DTYPE_Q8_0,
    MAGIC,
    NAME_TO_DTYPE,
    VERSION,
)
from miniqwen.quant.block_q4_0 import BlockQ4Tensor
from miniqwen.quant.block_q8_0 import BlockQ8Tensor


@dataclass(slots=True)
class _Record:
    name: str
    dtype: int
    shape: tuple[int, ...]
    raw: bytes
    quant: dict[str, Any]
    offset: int = 0


def _align(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


class GGUFWriter:
    def __init__(self, metadata: dict[str, Any] | None = None, alignment: int = DEFAULT_ALIGNMENT):
        self.metadata = metadata or {}
        self.alignment = alignment
        self.records: list[_Record] = []

    def add_tensor(self, name: str, tensor: torch.Tensor, dtype: str = "f16") -> None:
        if dtype not in NAME_TO_DTYPE:
            raise ValueError(f"Unsupported tensor dtype: {dtype}")
        if dtype == "f32":
            raw = tensor.detach().cpu().contiguous().to(torch.float32).numpy().tobytes()
        elif dtype == "f16":
            raw = tensor.detach().cpu().contiguous().to(torch.float16).numpy().tobytes()
        else:
            raise ValueError("Use add_quantized_tensor for q8_0/q4_0")
        self.records.append(_Record(name, NAME_TO_DTYPE[dtype], tuple(tensor.shape), raw, {}))

    def add_quantized_tensor(self, name: str, tensor: BlockQ8Tensor | BlockQ4Tensor) -> None:
        if isinstance(tensor, BlockQ8Tensor):
            raw = tensor.scales.numpy().astype("float32").tobytes() + tensor.q.numpy().tobytes()
            quant = {
                "block_size": tensor.block_size,
                "num_blocks": int(tensor.scales.numel()),
                "padded_numel": int(tensor.q.numel()),
            }
            dtype = DTYPE_Q8_0
            shape = tensor.shape
        elif isinstance(tensor, BlockQ4Tensor):
            raw = (
                tensor.scales.numpy().astype("float32").tobytes() + tensor.packed.numpy().tobytes()
            )
            quant = {
                "block_size": tensor.block_size,
                "num_blocks": int(tensor.scales.numel()),
                "padded_numel": int(tensor.scales.numel() * tensor.block_size),
                "packed_numel": int(tensor.packed.numel()),
            }
            dtype = DTYPE_Q4_0
            shape = tensor.shape
        else:
            raise TypeError(type(tensor))
        self.records.append(_Record(name, dtype, shape, raw, quant))

    def _directory_size(self) -> int:
        size = 0
        for rec in self.records:
            q = json.dumps(rec.quant, separators=(",", ":")).encode("utf-8")
            size += (
                4 + len(rec.name.encode("utf-8")) + 4 + 4 + 8 * len(rec.shape) + 8 + 8 + 4 + len(q)
            )
        return size

    def _directory_bytes(self) -> bytes:
        chunks: list[bytes] = []
        for rec in self.records:
            name = rec.name.encode("utf-8")
            q = json.dumps(rec.quant, separators=(",", ":")).encode("utf-8")
            chunks.append(struct.pack("<I", len(name)))
            chunks.append(name)
            chunks.append(struct.pack("<II", rec.dtype, len(rec.shape)))
            chunks.append(struct.pack("<" + "Q" * len(rec.shape), *rec.shape))
            chunks.append(struct.pack("<QQI", rec.offset, len(rec.raw), len(q)))
            chunks.append(q)
        return b"".join(chunks)

    def write(self, path: str | Path) -> None:
        meta = json.dumps(self.metadata, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        header_size = 4 + 4 + 4 + 8 + 8 + 8
        dir_size = self._directory_size()
        data_start = _align(header_size + len(meta) + dir_size, self.alignment)
        cursor = data_start
        for rec in self.records:
            rec.offset = cursor
            cursor = _align(cursor + len(rec.raw), self.alignment)
        directory = self._directory_bytes()
        header = struct.pack(
            "<4sIIQQQ", MAGIC, VERSION, self.alignment, len(meta), len(self.records), len(directory)
        )
        out = bytearray(header + meta + directory)
        out.extend(b"\x00" * (data_start - len(out)))
        for rec in self.records:
            if len(out) < rec.offset:
                out.extend(b"\x00" * (rec.offset - len(out)))
            out.extend(rec.raw)
            aligned = _align(len(out), self.alignment)
            out.extend(b"\x00" * (aligned - len(out)))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(bytes(out))
