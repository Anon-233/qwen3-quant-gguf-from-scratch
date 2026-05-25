from miniqwen.config import Qwen3Config
from miniqwen.model.text_model import Qwen3TextModel, make_tiny_state_dict


def test_config_driven_shape_derivation(tiny_config):
    assert tiny_config.head_dim == tiny_config.hidden_size // tiny_config.num_attention_heads
    assert tiny_config.q_per_kv_group == 2


def test_runtime_adapts_when_layer_count_changes():
    cfg = Qwen3Config.from_dict(
        {
            "vocab_size": 16,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 3,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "max_position_embeddings": 16,
        }
    )
    model = Qwen3TextModel(cfg, make_tiny_state_dict(cfg))
    assert model.config.num_hidden_layers == 3


def test_attention_adapts_when_head_count_changes():
    cfg = Qwen3Config.from_dict(
        {
            "vocab_size": 16,
            "hidden_size": 12,
            "intermediate_size": 24,
            "num_hidden_layers": 1,
            "num_attention_heads": 3,
            "num_key_value_heads": 1,
            "max_position_embeddings": 16,
        }
    )
    state = make_tiny_state_dict(cfg)
    assert state["blk.0.attn_q.weight"].shape == (12, 12)
    assert state["blk.0.attn_k.weight"].shape == (4, 12)
