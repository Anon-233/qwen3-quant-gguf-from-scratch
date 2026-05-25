from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class KVCache:
    max_seq_len: int | None = None
    num_layers: int | None = None
    _keys: dict[int, torch.Tensor] = field(default_factory=dict)
    _values: dict[int, torch.Tensor] = field(default_factory=dict)
    _buf_k: torch.Tensor | None = field(default=None, init=False, repr=False)
    _buf_v: torch.Tensor | None = field(default=None, init=False, repr=False)
    _lengths: list[int] = field(default_factory=list, init=False)

    def _try_ensure_buffers(self, k: torch.Tensor, v: torch.Tensor) -> bool:
        """Allocate pre-allocated buffers on first call. Returns True if buffers are ready."""
        if self._buf_k is not None:
            return True
        if self.max_seq_len is None or self.num_layers is None:
            return False
        bsz, _, n_kv_heads, head_dim = k.shape
        self._buf_k = torch.zeros(
            self.num_layers, bsz, self.max_seq_len, n_kv_heads, head_dim,
            device=k.device, dtype=k.dtype,
        )
        self._buf_v = torch.zeros_like(self._buf_k)
        self._lengths = [0] * self.num_layers
        return True

    def append(
        self, layer_idx: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._try_ensure_buffers(k, v):
            buf_k = self._buf_k
            buf_v = self._buf_v
            assert buf_k is not None and buf_v is not None  # _try_ensure_buffers ensures this
            start = self._lengths[layer_idx]
            n = k.shape[1]
            buf_k[layer_idx, :, start : start + n] = k
            buf_v[layer_idx, :, start : start + n] = v
            self._lengths[layer_idx] = start + n
            return (
                buf_k[layer_idx, :, : start + n],
                buf_v[layer_idx, :, : start + n],
            )
        if layer_idx in self._keys:
            k = torch.cat([self._keys[layer_idx], k], dim=1)
            v = torch.cat([self._values[layer_idx], v], dim=1)
        self._keys[layer_idx] = k
        self._values[layer_idx] = v
        return k, v

    def length(self, layer_idx: int = 0) -> int:
        buf_k = self._buf_k
        if buf_k is not None:
            return self._lengths[layer_idx]
        if layer_idx not in self._keys:
            return 0
        return int(self._keys[layer_idx].shape[1])
