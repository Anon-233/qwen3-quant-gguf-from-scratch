from __future__ import annotations

from miniqwen.config import Qwen3Config


def summarize_config(config: Qwen3Config) -> dict:
    return {
        "model_type": config.model_type,
        "architectures": list(config.architectures),
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "max_position_embeddings": config.max_position_embeddings,
        "rope_theta": config.rope_theta,
        "rms_norm_eps": config.rms_norm_eps,
        "tie_word_embeddings": config.tie_word_embeddings,
        "qk_layernorm": config.qk_layernorm,
    }
