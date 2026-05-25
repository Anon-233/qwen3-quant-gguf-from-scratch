from __future__ import annotations

import re

from miniqwen.config import Qwen3Config


def hf_to_runtime_name(name: str) -> str | None:
    if name == "model.embed_tokens.weight":
        return "token_embd.weight"
    if name == "model.norm.weight":
        return "output_norm.weight"
    if name == "lm_head.weight":
        return "output.weight"
    m = re.match(r"model\.layers\.(\d+)\.(.+)", name)
    if not m:
        return None
    i, suffix = m.groups()
    table = {
        "input_layernorm.weight": f"blk.{i}.attn_norm.weight",
        "post_attention_layernorm.weight": f"blk.{i}.ffn_norm.weight",
        "self_attn.q_proj.weight": f"blk.{i}.attn_q.weight",
        "self_attn.k_proj.weight": f"blk.{i}.attn_k.weight",
        "self_attn.v_proj.weight": f"blk.{i}.attn_v.weight",
        "self_attn.o_proj.weight": f"blk.{i}.attn_o.weight",
        "self_attn.q_norm.weight": f"blk.{i}.attn_q_norm.weight",
        "self_attn.k_norm.weight": f"blk.{i}.attn_k_norm.weight",
        "mlp.gate_proj.weight": f"blk.{i}.ffn_gate.weight",
        "mlp.up_proj.weight": f"blk.{i}.ffn_up.weight",
        "mlp.down_proj.weight": f"blk.{i}.ffn_down.weight",
    }
    return table.get(suffix)


def expected_runtime_shape(name: str, config: Qwen3Config) -> tuple[int, ...] | None:
    if name == "token_embd.weight":
        return (config.vocab_size, config.hidden_size)
    if name == "output_norm.weight":
        return (config.hidden_size,)
    if name == "output.weight":
        return (config.vocab_size, config.hidden_size)
    m = re.match(r"blk\.(\d+)\.(.+)", name)
    if not m:
        return None
    layer = int(m.group(1))
    suffix = m.group(2)
    if layer >= config.num_hidden_layers:
        raise ValueError(f"Tensor {name} refers to layer outside config.num_hidden_layers")
    shapes = {
        "attn_norm.weight": (config.hidden_size,),
        "ffn_norm.weight": (config.hidden_size,),
        "attn_q.weight": (config.num_attention_heads * config.head_dim, config.hidden_size),
        "attn_k.weight": (config.num_key_value_heads * config.head_dim, config.hidden_size),
        "attn_v.weight": (config.num_key_value_heads * config.head_dim, config.hidden_size),
        "attn_o.weight": (config.hidden_size, config.num_attention_heads * config.head_dim),
        "attn_q_norm.weight": (config.head_dim,),
        "attn_k_norm.weight": (config.head_dim,),
        "ffn_gate.weight": (config.intermediate_size, config.hidden_size),
        "ffn_up.weight": (config.intermediate_size, config.hidden_size),
        "ffn_down.weight": (config.hidden_size, config.intermediate_size),
    }
    return shapes.get(suffix)


def map_state_dict(
    state_dict: dict,
    config: Qwen3Config,
    strict_shapes: bool = True,
) -> tuple[dict, list[str]]:
    mapped = {}
    skipped = []
    for hf_name, tensor in state_dict.items():
        rt_name = hf_to_runtime_name(hf_name)
        if rt_name is None:
            skipped.append(hf_name)
            continue
        expected = expected_runtime_shape(rt_name, config)
        if strict_shapes and expected is not None and tuple(tensor.shape) != expected:
            actual = tuple(tensor.shape)
            raise ValueError(
                f"Shape mismatch for {hf_name} -> {rt_name}: "
                f"got {actual}, expected {expected}"
            )
        mapped[rt_name] = tensor.detach().cpu()
    return mapped, skipped
