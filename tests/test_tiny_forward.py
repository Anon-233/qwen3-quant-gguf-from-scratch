import torch

from miniqwen.model.text_model import Qwen3TextModel, make_tiny_state_dict


def test_tiny_text_model_forward_output_shape(tiny_config):
    model = Qwen3TextModel(tiny_config, make_tiny_state_dict(tiny_config))
    logits = model.forward(torch.tensor([[1, 2, 3]]))
    assert logits.shape == (1, 3, tiny_config.vocab_size)
