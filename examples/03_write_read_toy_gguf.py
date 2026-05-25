from pathlib import Path

import torch

from miniqwen.config import Qwen3Config
from miniqwen.gguf.reader import GGUFReader
from miniqwen.gguf.writer import GGUFWriter

cfg = Qwen3Config.from_dict(
    {
        "vocab_size": 16,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "max_position_embeddings": 32,
    }
)
path = Path("outputs/toy.gguf")
writer = GGUFWriter(metadata=cfg.to_metadata("toy"))
writer.add_tensor("toy.weight", torch.arange(8, dtype=torch.float32).view(2, 4), dtype="f16")
writer.write(path)
reader = GGUFReader(path)
print(reader.metadata["model.embedding_length"])
print(reader.get_tensor("toy.weight"))
