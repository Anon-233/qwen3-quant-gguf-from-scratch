from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def _required(data: dict[str, Any], key: str) -> Any:
    if key not in data or data[key] is None:
        raise ValueError(f"Missing required Qwen3 config field: {key}")
    return data[key]


@dataclass(slots=True)
class Qwen3Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool
    qk_layernorm: bool = False
    bos_token_id: int | None = None
    eos_token_id: int | list[int] | None = None
    pad_token_id: int | None = None
    torch_dtype: str = "float16"
    model_type: str = "qwen3"
    architectures: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Qwen3Config:
        hidden_size = int(_required(data, "hidden_size"))
        num_attention_heads = int(_required(data, "num_attention_heads"))
        if hidden_size % num_attention_heads != 0 and "head_dim" not in data:
            raise ValueError(
                "hidden_size must be divisible by num_attention_heads when head_dim is absent"
            )
        head_dim = int(data.get("head_dim") or hidden_size // num_attention_heads)
        num_key_value_heads = int(data.get("num_key_value_heads", num_attention_heads))
        if num_attention_heads % num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        return cls(
            vocab_size=int(_required(data, "vocab_size")),
            hidden_size=hidden_size,
            intermediate_size=int(_required(data, "intermediate_size")),
            num_hidden_layers=int(_required(data, "num_hidden_layers")),
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            max_position_embeddings=int(
                data.get("max_position_embeddings")
                or data.get("max_sequence_length")
                or data.get("seq_length")
                or _required(data, "max_position_embeddings")
            ),
            rms_norm_eps=float(data.get("rms_norm_eps", data.get("layer_norm_epsilon", 1e-6))),
            rope_theta=float(data.get("rope_theta", data.get("rope_freq_base", 1_000_000.0))),
            tie_word_embeddings=bool(data.get("tie_word_embeddings", False)),
            qk_layernorm=bool(data.get("qk_layernorm", data.get("use_qk_norm", False))),
            bos_token_id=data.get("bos_token_id"),
            eos_token_id=data.get("eos_token_id"),
            pad_token_id=data.get("pad_token_id"),
            torch_dtype=str(data.get("torch_dtype", "float16")),
            model_type=str(data.get("model_type", "qwen3")),
            architectures=tuple(data.get("architectures", ()) or ()),
        )

    @classmethod
    def from_json_file(cls, path: str | Path) -> Qwen3Config:
        with Path(path).open("r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_pretrained(cls, model_name_or_path: str) -> Qwen3Config:
        path = Path(model_name_or_path)
        if path.exists():
            cfg = path / "config.json" if path.is_dir() else path
            return cls.from_json_file(cfg)
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError("huggingface_hub is required to load remote configs") from exc
        cfg_path = hf_hub_download(model_name_or_path, "config.json")
        return cls.from_json_file(cfg_path)

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any]) -> Qwen3Config:
        data = {
            "vocab_size": metadata["model.vocab_size"],
            "hidden_size": metadata["model.embedding_length"],
            "intermediate_size": metadata["model.feed_forward_length"],
            "num_hidden_layers": metadata["model.block_count"],
            "num_attention_heads": metadata["model.attention.head_count"],
            "num_key_value_heads": metadata["model.attention.head_count_kv"],
            "head_dim": metadata.get("model.attention.head_dim"),
            "max_position_embeddings": metadata["model.context_length"],
            "rms_norm_eps": metadata["model.attention.layer_norm_rms_epsilon"],
            "rope_theta": metadata["model.rope.freq_base"],
            "tie_word_embeddings": metadata.get("model.tie_word_embeddings", False),
            "qk_layernorm": metadata.get("model.attention.qk_layernorm", False),
            "bos_token_id": metadata.get("tokenizer.bos_token_id"),
            "eos_token_id": metadata.get("tokenizer.eos_token_id"),
            "pad_token_id": metadata.get("tokenizer.pad_token_id"),
            "torch_dtype": metadata.get("model.torch_dtype", "float16"),
            "model_type": metadata.get("general.architecture", "qwen3"),
        }
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["architectures"] = list(self.architectures)
        return data

    def to_metadata(self, name: str = "qwen3") -> dict[str, Any]:
        return {
            "general.architecture": self.model_type,
            "general.name": name,
            "general.file_type": "teaching",
            "general.quantization_version": 1,
            "model.vocab_size": self.vocab_size,
            "model.context_length": self.max_position_embeddings,
            "model.embedding_length": self.hidden_size,
            "model.block_count": self.num_hidden_layers,
            "model.feed_forward_length": self.intermediate_size,
            "model.attention.head_count": self.num_attention_heads,
            "model.attention.head_count_kv": self.num_key_value_heads,
            "model.attention.head_dim": self.head_dim,
            "model.rope.freq_base": self.rope_theta,
            "model.attention.layer_norm_rms_epsilon": self.rms_norm_eps,
            "model.tie_word_embeddings": self.tie_word_embeddings,
            "model.attention.qk_layernorm": self.qk_layernorm,
            "model.torch_dtype": self.torch_dtype,
            "tokenizer.bos_token_id": self.bos_token_id,
            "tokenizer.eos_token_id": self.eos_token_id,
            "tokenizer.pad_token_id": self.pad_token_id,
        }

    @property
    def q_per_kv_group(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads
