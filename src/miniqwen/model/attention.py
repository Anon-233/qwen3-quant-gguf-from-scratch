import math

import torch

from miniqwen.config import Qwen3Config
from miniqwen.model.norm import rms_norm
from miniqwen.model.rope import apply_rope


def repeat_kv(x: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return x
    return x.repeat_interleave(repeats, dim=2)


def gqa_attention(
    x: torch.Tensor,
    wq: torch.Tensor,
    wk: torch.Tensor,
    wv: torch.Tensor,
    wo: torch.Tensor,
    config: Qwen3Config,
    layer_idx: int = 0,
    kv_cache=None,
    q_norm_weight: torch.Tensor | None = None,
    k_norm_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    bsz, seq_len, _ = x.shape
    q = x @ wq.to(x.dtype).T
    k = x @ wk.to(x.dtype).T
    v = x @ wv.to(x.dtype).T
    q = q.view(bsz, seq_len, config.num_attention_heads, config.head_dim)
    k = k.view(bsz, seq_len, config.num_key_value_heads, config.head_dim)
    v = v.view(bsz, seq_len, config.num_key_value_heads, config.head_dim)
    if q_norm_weight is not None:
        q = rms_norm(q, q_norm_weight, config.rms_norm_eps)
    if k_norm_weight is not None:
        k = rms_norm(k, k_norm_weight, config.rms_norm_eps)
    position_offset = 0 if kv_cache is None else kv_cache.length(layer_idx)
    q = apply_rope(q, position_offset, config.rope_theta)
    k = apply_rope(k, position_offset, config.rope_theta)
    if kv_cache is not None:
        k, v = kv_cache.append(layer_idx, k, v)
    k = repeat_kv(k, config.q_per_kv_group)
    v = repeat_kv(v, config.q_per_kv_group)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    scores = q @ k.transpose(-2, -1) / math.sqrt(config.head_dim)
    key_len = scores.shape[-1]
    query_positions = torch.arange(position_offset, position_offset + seq_len, device=x.device)
    key_positions = torch.arange(key_len, device=x.device)
    mask = key_positions[None, :] > query_positions[:, None]
    scores = scores.masked_fill(mask[None, None, :, :], torch.finfo(scores.dtype).min)
    probs = torch.softmax(scores.to(torch.float32), dim=-1).to(x.dtype)
    out = probs @ v
    q_width = config.num_attention_heads * config.head_dim
    out = out.transpose(1, 2).contiguous().view(bsz, seq_len, q_width)
    return out @ wo.to(x.dtype).T
