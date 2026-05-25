import torch

from miniqwen.convert.hf_to_gguf import write_runtime_state_to_gguf
from miniqwen.generation import generate_tokens
from miniqwen.model.text_model import Qwen3TextModel, make_tiny_state_dict
from miniqwen.runtime.loader import load_gguf_runtime


def test_f16_tiny_gguf_runtime_close_to_reference(tmp_path, tiny_config):
    state = make_tiny_state_dict(tiny_config)
    path = tmp_path / "tiny-f16.gguf"
    write_runtime_state_to_gguf(state, tiny_config, path, quant="f16")
    ref = Qwen3TextModel(tiny_config, state)
    rt = load_gguf_runtime(str(path))
    ids = torch.tensor([[1, 2, 3]])
    ref_logits = ref.forward(ids).detach().cpu().float()
    rt_logits = rt.forward(ids).detach().cpu().float()
    assert torch.allclose(ref_logits, rt_logits, atol=3e-4, rtol=3e-3)


def test_q8_q4_tiny_runtime_generation_smoke(tmp_path, tiny_config):
    state = make_tiny_state_dict(tiny_config)
    for quant in ["q8_0", "q4_0"]:
        path = tmp_path / f"tiny-{quant}.gguf"
        write_runtime_state_to_gguf(state, tiny_config, path, quant=quant)
        rt = load_gguf_runtime(str(path))
        out = generate_tokens(rt, torch.tensor([[1, 2]]), max_new_tokens=1, temperature=0)
        assert out.shape == (1, 3)
        assert torch.isfinite(rt.forward(torch.tensor([[1, 2]]))).all()
