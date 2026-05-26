import pytest
import torch

from miniqwen.config import Qwen3Config
from miniqwen.generation import generate_tokens
from miniqwen.model.text_model import Qwen3TextModel, make_tiny_state_dict


def test_generation_loop_can_generate_token(tiny_config):
    model = Qwen3TextModel(tiny_config, make_tiny_state_dict(tiny_config))
    out = generate_tokens(model, torch.tensor([[1, 2]]), max_new_tokens=2, temperature=0)
    assert out.shape == (1, 4)


def test_generation_rejects_context_longer_than_max_position_embeddings():
    config = Qwen3Config.from_dict(
        {
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 4,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "tie_word_embeddings": False,
        }
    )
    model = Qwen3TextModel(config, make_tiny_state_dict(config))
    with pytest.raises(ValueError, match="max_position_embeddings=4"):
        generate_tokens(
            model,
            torch.tensor([[1, 2, 3, 4]]),
            max_new_tokens=1,
            temperature=0,
            use_cache=True,
        )
