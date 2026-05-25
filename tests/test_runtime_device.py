import pytest
import torch

from miniqwen.convert.hf_to_gguf import write_runtime_state_to_gguf
from miniqwen.model.text_model import make_tiny_state_dict
from miniqwen.runtime.loader import load_gguf_runtime


def test_runtime_loader_default_auto_device(tmp_path, tiny_config):
    path = tmp_path / "tiny-f16.gguf"
    write_runtime_state_to_gguf(make_tiny_state_dict(tiny_config), tiny_config, path, quant="f16")
    rt = load_gguf_runtime(str(path))
    logits = rt.forward(torch.tensor([[1, 2]]))
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert rt.device.type == expected
    assert logits.device.type == expected


def test_runtime_loader_explicit_cpu_device(tmp_path, tiny_config):
    path = tmp_path / "tiny-f16.gguf"
    write_runtime_state_to_gguf(make_tiny_state_dict(tiny_config), tiny_config, path, quant="f16")
    rt = load_gguf_runtime(str(path), device="cpu", compute_dtype="float32")
    logits = rt.forward(torch.tensor([[1, 2]]))
    assert rt.device.type == "cpu"
    assert logits.device.type == "cpu"
    assert rt.compute_dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_runtime_loader_cuda_device_smoke(tmp_path, tiny_config):
    path = tmp_path / "tiny-q8.gguf"
    write_runtime_state_to_gguf(make_tiny_state_dict(tiny_config), tiny_config, path, quant="q8_0")
    rt = load_gguf_runtime(str(path), device="cuda", compute_dtype="float16")
    logits = rt.forward(torch.tensor([[1, 2]]))
    assert rt.device.type == "cuda"
    assert logits.device.type == "cuda"
    assert logits.dtype == torch.float16
