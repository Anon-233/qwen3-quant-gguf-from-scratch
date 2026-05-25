import pytest

from miniqwen.config import Qwen3Config


@pytest.fixture
def tiny_config() -> Qwen3Config:
    return Qwen3Config.from_dict(
        {
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 64,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "tie_word_embeddings": False,
        }
    )
