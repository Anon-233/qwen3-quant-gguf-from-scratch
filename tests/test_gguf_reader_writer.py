import torch

from miniqwen.config import Qwen3Config
from miniqwen.gguf.reader import GGUFReader
from miniqwen.gguf.writer import GGUFWriter


def test_gguf_toy_round_trip(tmp_path, tiny_config):
    path = tmp_path / "toy.gguf"
    writer = GGUFWriter(metadata=tiny_config.to_metadata("toy"))
    writer.add_tensor("x", torch.arange(8).view(2, 4).float(), dtype="f16")
    writer.write(path)
    reader = GGUFReader(path)
    assert torch.allclose(reader.get_tensor("x").float(), torch.arange(8).view(2, 4).float())


def test_gguf_metadata_round_trip(tmp_path, tiny_config):
    path = tmp_path / "toy-meta.gguf"
    writer = GGUFWriter(metadata=tiny_config.to_metadata("toy"))
    writer.add_tensor("x", torch.arange(2).float(), dtype="f16")
    writer.write(path)
    reader = GGUFReader(path)
    assert reader.metadata["model.embedding_length"] == tiny_config.hidden_size
    restored = Qwen3Config.from_metadata(reader.metadata)
    assert restored.num_hidden_layers == tiny_config.num_hidden_layers
