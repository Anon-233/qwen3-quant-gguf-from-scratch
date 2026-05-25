import torch

from miniqwen.model.attention import gqa_attention
from miniqwen.model.text_model import make_tiny_state_dict


def test_attention_output_shape(tiny_config):
    state = make_tiny_state_dict(tiny_config)
    x = torch.randn(2, 4, tiny_config.hidden_size)
    y = gqa_attention(
        x,
        state["blk.0.attn_q.weight"],
        state["blk.0.attn_k.weight"],
        state["blk.0.attn_v.weight"],
        state["blk.0.attn_o.weight"],
        tiny_config,
    )
    assert y.shape == x.shape
