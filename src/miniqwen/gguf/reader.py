from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import Any

import numpy as np
import torch

from miniqwen.gguf.constants import DTYPE_F16, DTYPE_F32, DTYPE_Q4_0, DTYPE_Q8_0, MAGIC
from miniqwen.gguf.tensor_info import TensorInfo
from miniqwen.quant.block_q4_0 import BlockQ4Tensor, dequantize_q4_0
from miniqwen.quant.block_q8_0 import BlockQ8Tensor, dequantize_q8_0


class GGUFReader:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._data = self.path.read_bytes()
        self.metadata: dict[str, Any]
        self.tensors: dict[str, TensorInfo]
        self._parse()

    def _parse(self) -> None:
        magic, _version, _alignment, meta_len, tensor_count, dir_len = struct.unpack_from(
            "<4sIIQQQ", self._data, 0
        )
        if magic != MAGIC:
            raise ValueError("Not a miniqwen teaching GGUF file")
        cursor = struct.calcsize("<4sIIQQQ")
        self.metadata = json.loads(self._data[cursor : cursor + meta_len].decode("utf-8"))
        cursor += meta_len
        dir_end = cursor + dir_len
        tensors: dict[str, TensorInfo] = {}
        for _ in range(tensor_count):
            name_len = struct.unpack_from("<I", self._data, cursor)[0]
            cursor += 4
            name = self._data[cursor : cursor + name_len].decode("utf-8")
            cursor += name_len
            dtype, ndim = struct.unpack_from("<II", self._data, cursor)
            cursor += 8
            shape = struct.unpack_from("<" + "Q" * ndim, self._data, cursor)
            cursor += 8 * ndim
            offset, nbytes, q_len = struct.unpack_from("<QQI", self._data, cursor)
            cursor += 20
            quant = json.loads(self._data[cursor : cursor + q_len].decode("utf-8") or "{}")
            cursor += q_len
            tensors[name] = TensorInfo(
                name, dtype, tuple(int(x) for x in shape), offset, nbytes, quant
            )
        if cursor != dir_end:
            raise ValueError("Corrupt tensor directory")
        self.tensors = tensors

    def raw(self, name: str) -> bytes:
        info = self.tensors[name]
        return self._data[info.offset : info.offset + info.nbytes]

    def get_tensor(self, name: str, dequantize: bool = True) -> torch.Tensor:
        info = self.tensors[name]
        raw = self.raw(name)
        if info.dtype == DTYPE_F32:
            return torch.from_numpy(np.frombuffer(raw, dtype=np.float32).copy()).view(info.shape)
        if info.dtype == DTYPE_F16:
            return torch.from_numpy(np.frombuffer(raw, dtype=np.float16).copy()).view(info.shape)
        if info.dtype == DTYPE_Q8_0:
            nblocks = int(info.quant["num_blocks"])
            scales = torch.from_numpy(np.frombuffer(raw[: nblocks * 4], dtype=np.float32).copy())
            q = torch.from_numpy(np.frombuffer(raw[nblocks * 4 :], dtype=np.int8).copy())
            obj = BlockQ8Tensor(
                q=q, scales=scales, shape=info.shape, block_size=info.quant["block_size"]
            )
            if not dequantize:
                return q
            return dequantize_q8_0(obj)
        if info.dtype == DTYPE_Q4_0:
            nblocks = int(info.quant["num_blocks"])
            packed_len = math.ceil(int(info.quant["padded_numel"]) / 2)
            scales = torch.from_numpy(np.frombuffer(raw[: nblocks * 4], dtype=np.float32).copy())
            packed = torch.from_numpy(
                np.frombuffer(raw[nblocks * 4 : nblocks * 4 + packed_len], dtype=np.uint8).copy()
            )
            obj = BlockQ4Tensor(
                packed=packed,
                scales=scales,
                shape=info.shape,
                block_size=info.quant["block_size"],
            )
            if not dequantize:
                return packed
            return dequantize_q4_0(obj)
        raise ValueError(f"Unsupported dtype id: {info.dtype}")
