import torch

from miniqwen.config import Qwen3Config
from miniqwen.generation import generate_tokens
from miniqwen.model.text_model import Qwen3TextModel, make_tiny_state_dict

cfg = Qwen3Config.from_dict(
    {
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 64,
        "tie_word_embeddings": False,
    }
)
model = Qwen3TextModel(cfg, make_tiny_state_dict(cfg))
ids = torch.tensor([[1, 2, 3]])
out = generate_tokens(model, ids, max_new_tokens=4, temperature=0)
print(out.tolist())
