from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class TensorInfo:
    name: str
    dtype: int
    shape: tuple[int, ...]
    offset: int
    nbytes: int
    quant: dict[str, Any]
