from __future__ import annotations

import argparse
import gc

import torch

from miniqwen.evaluation.compare_logits import compare_logits
from miniqwen.generation import generate_tokens
from miniqwen.runtime.loader import load_gguf_runtime
from miniqwen.tokenizer_adapter import TokenizerAdapter


def _cleanup_device(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _logits_for(
    model_path: str,
    input_ids: torch.Tensor,
    device: str,
    compute_dtype: str,
) -> torch.Tensor:
    model = load_gguf_runtime(model_path, device=device, compute_dtype=compute_dtype)
    with torch.no_grad():
        logits = model.forward(input_ids)[:, -1, :].detach().cpu()
    del model
    gc.collect()
    _cleanup_device(device)
    return logits


def _generate_text(
    model_path: str,
    input_ids: torch.Tensor,
    tokenizer: TokenizerAdapter,
    max_new_tokens: int,
    device: str,
    compute_dtype: str,
) -> str:
    model = load_gguf_runtime(model_path, device=device, compute_dtype=compute_dtype)
    with torch.no_grad():
        out = generate_tokens(
            model,
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0,
            eos_token_id=model.config.eos_token_id,
            use_cache=True,
        )
    text = tokenizer.decode(out[0].detach().cpu().tolist())
    del model
    gc.collect()
    _cleanup_device(device)
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="Usually the f16 teaching GGUF.")
    parser.add_argument("--quantized", nargs="+", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--device", default="auto", help="cpu, cuda, cuda:0, or auto")
    parser.add_argument(
        "--compute_dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"]
    )
    args = parser.parse_args()

    tokenizer = TokenizerAdapter(args.tokenizer)
    input_ids = tokenizer.encode(args.prompt)
    baseline_logits = _logits_for(args.baseline, input_ids, args.device, args.compute_dtype)
    baseline_text = _generate_text(
        args.baseline, input_ids, tokenizer, args.max_new_tokens, args.device, args.compute_dtype
    )
    print({"baseline": args.baseline, "generation": baseline_text})
    for path in args.quantized:
        logits = _logits_for(path, input_ids, args.device, args.compute_dtype)
        metrics = compare_logits(baseline_logits, logits, k=args.top_k)
        text = _generate_text(
            path, input_ids, tokenizer, args.max_new_tokens, args.device, args.compute_dtype
        )
        print({"quantized": path, "metrics_vs_baseline": metrics, "generation": text})


if __name__ == "__main__":
    main()
