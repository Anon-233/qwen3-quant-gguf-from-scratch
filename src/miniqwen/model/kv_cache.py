from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class KVCache:
    keys: dict[int, torch.Tensor] = field(default_factory=dict)
    values: dict[int, torch.Tensor] = field(default_factory=dict)

    def append(
        self, layer_idx: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_idx in self.keys:
            k = torch.cat([self.keys[layer_idx], k], dim=1)
            v = torch.cat([self.values[layer_idx], v], dim=1)
        self.keys[layer_idx] = k
        self.values[layer_idx] = v
        return k, v

    def length(self, layer_idx: int = 0) -> int:
        if layer_idx not in self.keys:
            return 0
        return int(self.keys[layer_idx].shape[1])
