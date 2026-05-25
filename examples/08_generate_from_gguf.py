import torch

from miniqwen.generation import generate_tokens
from miniqwen.runtime.loader import load_gguf_runtime

model = load_gguf_runtime("outputs/qwen3-0.6b-q4_0.gguf")
ids = torch.tensor([[model.config.bos_token_id or 0]])
print(generate_tokens(model, ids, max_new_tokens=8, temperature=0).tolist())
