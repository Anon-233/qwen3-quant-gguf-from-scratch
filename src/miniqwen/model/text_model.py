from __future__ import annotations

import torch

from miniqwen.config import Qwen3Config
from miniqwen.model.attention import gqa_attention
from miniqwen.model.layers import embedding
from miniqwen.model.mlp import swiglu_mlp
from miniqwen.model.norm import rms_norm
from miniqwen.model.rope import RoPECache


class Qwen3TextModel:
    def __init__(
        self,
        config: Qwen3Config,
        state_dict: dict[str, torch.Tensor],
        device: torch.device | str | None = None,
        compute_dtype: torch.dtype | None = None,
    ):
        self.config = config
        self.state_dict = state_dict
        first = next(iter(state_dict.values()), None)
        self.device = (
            torch.device(device)
            if device is not None
            else (first.device if first is not None else torch.device("cpu"))
        )
        self.compute_dtype = compute_dtype or (
            torch.float16 if self.device.type == "cuda" else torch.float32
        )
        self._rope_cache = RoPECache.precompute(
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            device=self.device,
        )
        self._validate_minimum_tensors()

    def _tensor(self, name: str) -> torch.Tensor:
        if name not in self.state_dict:
            raise KeyError(f"Missing runtime tensor: {name}")
        return self.state_dict[name]

    def _validate_minimum_tensors(self) -> None:
        required = ["token_embd.weight", "output_norm.weight"]
        if not self.config.tie_word_embeddings:
            required.append("output.weight")
        for i in range(self.config.num_hidden_layers):
            required.extend(
                [
                    f"blk.{i}.attn_norm.weight",
                    f"blk.{i}.ffn_norm.weight",
                    f"blk.{i}.attn_q.weight",
                    f"blk.{i}.attn_k.weight",
                    f"blk.{i}.attn_v.weight",
                    f"blk.{i}.attn_o.weight",
                    f"blk.{i}.ffn_gate.weight",
                    f"blk.{i}.ffn_up.weight",
                    f"blk.{i}.ffn_down.weight",
                ]
            )
            if self.config.qk_layernorm:
                required.extend([f"blk.{i}.attn_q_norm.weight", f"blk.{i}.attn_k_norm.weight"])
        missing = [name for name in required if name not in self.state_dict]
        if missing:
            raise KeyError(f"Missing tensors for config-driven Qwen3 runtime: {missing[:8]}")

    def forward(self, input_ids: torch.Tensor, kv_cache=None) -> torch.Tensor:
        input_ids = input_ids.to(self.device)
        x = embedding(input_ids, self._tensor("token_embd.weight")).to(self.compute_dtype)
        for i in range(self.config.num_hidden_layers):
            h = rms_norm(x, self._tensor(f"blk.{i}.attn_norm.weight"), self.config.rms_norm_eps)
            x = x + gqa_attention(
                h,
                self._tensor(f"blk.{i}.attn_q.weight"),
                self._tensor(f"blk.{i}.attn_k.weight"),
                self._tensor(f"blk.{i}.attn_v.weight"),
                self._tensor(f"blk.{i}.attn_o.weight"),
                self.config,
                layer_idx=i,
                kv_cache=kv_cache,
                q_norm_weight=(
                    self._tensor(f"blk.{i}.attn_q_norm.weight")
                    if f"blk.{i}.attn_q_norm.weight" in self.state_dict
                    else None
                ),
                k_norm_weight=(
                    self._tensor(f"blk.{i}.attn_k_norm.weight")
                    if f"blk.{i}.attn_k_norm.weight" in self.state_dict
                    else None
                ),
                rope_cache=self._rope_cache,
            )
            h = rms_norm(x, self._tensor(f"blk.{i}.ffn_norm.weight"), self.config.rms_norm_eps)
            x = x + swiglu_mlp(
                h,
                self._tensor(f"blk.{i}.ffn_gate.weight"),
                self._tensor(f"blk.{i}.ffn_up.weight"),
                self._tensor(f"blk.{i}.ffn_down.weight"),
            )
        x = rms_norm(x, self._tensor("output_norm.weight"), self.config.rms_norm_eps)
        output_weight = (
            self._tensor("token_embd.weight")
            if self.config.tie_word_embeddings
            else self._tensor("output.weight")
        )
        return x @ output_weight.to(x.dtype).T


def make_tiny_state_dict(config: Qwen3Config, seed: int = 0) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    sd: dict[str, torch.Tensor] = {
        "token_embd.weight": torch.randn(config.vocab_size, config.hidden_size, generator=g) * 0.02,
        "output_norm.weight": torch.ones(config.hidden_size),
    }
    if not config.tie_word_embeddings:
        sd["output.weight"] = torch.randn(config.vocab_size, config.hidden_size, generator=g) * 0.02
    for i in range(config.num_hidden_layers):
        sd[f"blk.{i}.attn_norm.weight"] = torch.ones(config.hidden_size)
        sd[f"blk.{i}.ffn_norm.weight"] = torch.ones(config.hidden_size)
        sd[f"blk.{i}.attn_q.weight"] = (
            torch.randn(
                config.num_attention_heads * config.head_dim, config.hidden_size, generator=g
            )
            * 0.02
        )
        sd[f"blk.{i}.attn_k.weight"] = (
            torch.randn(
                config.num_key_value_heads * config.head_dim, config.hidden_size, generator=g
            )
            * 0.02
        )
        sd[f"blk.{i}.attn_v.weight"] = (
            torch.randn(
                config.num_key_value_heads * config.head_dim, config.hidden_size, generator=g
            )
            * 0.02
        )
        if config.qk_layernorm:
            sd[f"blk.{i}.attn_q_norm.weight"] = torch.ones(config.head_dim)
            sd[f"blk.{i}.attn_k_norm.weight"] = torch.ones(config.head_dim)
        sd[f"blk.{i}.attn_o.weight"] = (
            torch.randn(
                config.hidden_size, config.num_attention_heads * config.head_dim, generator=g
            )
            * 0.02
        )
        sd[f"blk.{i}.ffn_gate.weight"] = (
            torch.randn(config.intermediate_size, config.hidden_size, generator=g) * 0.02
        )
        sd[f"blk.{i}.ffn_up.weight"] = (
            torch.randn(config.intermediate_size, config.hidden_size, generator=g) * 0.02
        )
        sd[f"blk.{i}.ffn_down.weight"] = (
            torch.randn(config.hidden_size, config.intermediate_size, generator=g) * 0.02
        )
    return sd
