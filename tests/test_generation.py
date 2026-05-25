import torch

from miniqwen.generation import generate_tokens
from miniqwen.model.text_model import Qwen3TextModel, make_tiny_state_dict


def test_generation_loop_can_generate_token(tiny_config):
    model = Qwen3TextModel(tiny_config, make_tiny_state_dict(tiny_config))
    out = generate_tokens(model, torch.tensor([[1, 2]]), max_new_tokens=2, temperature=0)
    assert out.shape == (1, 4)
