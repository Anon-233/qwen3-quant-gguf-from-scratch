import torch

from miniqwen.convert.tensor_mapper import map_state_dict


def test_tensor_mapping_toy_state_dict(tiny_config):
    state = {
        "model.embed_tokens.weight": torch.zeros(tiny_config.vocab_size, tiny_config.hidden_size),
        "model.norm.weight": torch.ones(tiny_config.hidden_size),
        "lm_head.weight": torch.zeros(tiny_config.vocab_size, tiny_config.hidden_size),
    }
    for i in range(tiny_config.num_hidden_layers):
        state[f"model.layers.{i}.input_layernorm.weight"] = torch.ones(tiny_config.hidden_size)
        state[f"model.layers.{i}.post_attention_layernorm.weight"] = torch.ones(
            tiny_config.hidden_size
        )
        state[f"model.layers.{i}.self_attn.q_proj.weight"] = torch.zeros(
            tiny_config.num_attention_heads * tiny_config.head_dim, tiny_config.hidden_size
        )
        state[f"model.layers.{i}.self_attn.k_proj.weight"] = torch.zeros(
            tiny_config.num_key_value_heads * tiny_config.head_dim, tiny_config.hidden_size
        )
        state[f"model.layers.{i}.self_attn.v_proj.weight"] = torch.zeros(
            tiny_config.num_key_value_heads * tiny_config.head_dim, tiny_config.hidden_size
        )
        state[f"model.layers.{i}.self_attn.o_proj.weight"] = torch.zeros(
            tiny_config.hidden_size, tiny_config.num_attention_heads * tiny_config.head_dim
        )
        state[f"model.layers.{i}.mlp.gate_proj.weight"] = torch.zeros(
            tiny_config.intermediate_size, tiny_config.hidden_size
        )
        state[f"model.layers.{i}.mlp.up_proj.weight"] = torch.zeros(
            tiny_config.intermediate_size, tiny_config.hidden_size
        )
        state[f"model.layers.{i}.mlp.down_proj.weight"] = torch.zeros(
            tiny_config.hidden_size, tiny_config.intermediate_size
        )
    mapped, skipped = map_state_dict(state, tiny_config)
    assert not skipped
    assert "blk.0.attn_q.weight" in mapped
