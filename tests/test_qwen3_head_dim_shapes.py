import torch

from miniqwen.config import Qwen3Config
from miniqwen.model.text_model import Qwen3TextModel, make_tiny_state_dict


def test_runtime_allows_head_dim_width_different_from_hidden_size():
    cfg = Qwen3Config.from_dict(
        {
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "max_position_embeddings": 64,
            "tie_word_embeddings": False,
            "qk_layernorm": True,
        }
    )
    model = Qwen3TextModel(cfg, make_tiny_state_dict(cfg))
    logits = model.forward(torch.tensor([[1, 2, 3]]))
    assert logits.shape == (1, 3, cfg.vocab_size)
