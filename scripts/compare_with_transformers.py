from __future__ import annotations

import argparse

import torch

from miniqwen.evaluation.compare_logits import compare_logits
from miniqwen.runtime.loader import load_gguf_runtime
from miniqwen.tokenizer_adapter import TokenizerAdapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_model", required=True)
    parser.add_argument("--gguf_model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max_tokens", type=int, default=8)
    parser.add_argument("--device", default="auto", help="cpu, cuda, cuda:0, or auto")
    parser.add_argument(
        "--compute_dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"]
    )
    args = parser.parse_args()
    from transformers import AutoModelForCausalLM

    tok = TokenizerAdapter(args.hf_model)
    ids = tok.encode(args.prompt)
    runtime = load_gguf_runtime(
        args.gguf_model, device=args.device, compute_dtype=args.compute_dtype
    )
    with torch.no_grad():
        gguf_logits = runtime.forward(ids)
        ref = AutoModelForCausalLM.from_pretrained(
            args.hf_model, torch_dtype=torch.float32, trust_remote_code=True
        ).eval()
        hf_logits = ref(ids).logits
    metrics = compare_logits(hf_logits[:, -1].cpu(), gguf_logits[:, -1].cpu(), k=args.max_tokens)
    print(metrics)


if __name__ == "__main__":
    main()
