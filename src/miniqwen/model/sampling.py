import torch


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    logits = logits[:, -1, :].to(torch.float32)
    if temperature <= 0:
        return torch.argmax(logits, dim=-1)
    logits = logits / temperature
    if top_k is not None and top_k > 0:
        values, _ = torch.topk(logits, min(top_k, logits.shape[-1]), dim=-1)
        logits = logits.masked_fill(logits < values[:, [-1]], -torch.inf)
    probs = torch.softmax(logits, dim=-1)
    if top_p is not None and 0 < top_p < 1:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cum = torch.cumsum(sorted_probs, dim=-1)
        remove = cum > top_p
        remove[:, 0] = False
        sorted_probs = sorted_probs.masked_fill(remove, 0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        choice = torch.multinomial(sorted_probs, 1)
        return sorted_idx.gather(-1, choice).squeeze(-1)
    return torch.multinomial(probs, 1).squeeze(-1)
