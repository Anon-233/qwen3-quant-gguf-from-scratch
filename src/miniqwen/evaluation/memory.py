from miniqwen.config import Qwen3Config


def estimate_kv_cache_bytes(
    config: Qwen3Config, tokens: int, batch: int = 1, elem_bytes: int = 2
) -> int:
    return (
        2
        * config.num_hidden_layers
        * config.num_key_value_heads
        * config.head_dim
        * tokens
        * batch
        * elem_bytes
    )
