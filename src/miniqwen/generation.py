from __future__ import annotations

import torch

from miniqwen.model.kv_cache import KVCache
from miniqwen.model.sampling import sample_next_token


def generate_tokens(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int = 32,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    eos_token_id: int | list[int] | None = None,
    use_cache: bool = False,
) -> torch.Tensor:
    device = getattr(model, "device", input_ids.device)
    generated = input_ids.to(device).clone()
    eos = set(eos_token_id if isinstance(eos_token_id, list) else [eos_token_id]) - {None}
    config = getattr(model, "config", None)
    if config is not None:
        requested_len = generated.shape[1] + max_new_tokens
        if requested_len > config.max_position_embeddings:
            raise ValueError(
                "Requested generation exceeds model context length: "
                f"prompt_len={generated.shape[1]}, max_new_tokens={max_new_tokens}, "
                f"requested_total={requested_len}, "
                f"max_position_embeddings={config.max_position_embeddings}. "
                "Reduce the prompt length or max_new_tokens."
            )
    if use_cache and config is not None:
        cache = KVCache(
            max_seq_len=config.max_position_embeddings,
            num_layers=config.num_hidden_layers,
        )
    elif use_cache:
        cache = KVCache()
    else:
        cache = None
    for _ in range(max_new_tokens):
        context = (
            generated[:, -1:]
            if use_cache and generated.shape[1] > input_ids.shape[1]
            else generated
        )
        logits = model.forward(context, kv_cache=cache)
        next_token = sample_next_token(logits, temperature=temperature, top_k=top_k, top_p=top_p)
        generated = torch.cat([generated, next_token[:, None]], dim=1)
        if eos and all(int(t) in eos for t in next_token.tolist()):
            break
    return generated
