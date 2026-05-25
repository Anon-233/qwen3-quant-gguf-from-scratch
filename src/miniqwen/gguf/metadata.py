from miniqwen.config import Qwen3Config


def metadata_from_config(config: Qwen3Config, name: str = "qwen3") -> dict:
    return config.to_metadata(name=name)
