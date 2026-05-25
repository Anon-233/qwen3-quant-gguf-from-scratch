from __future__ import annotations

import argparse

from miniqwen.generation import generate_tokens
from miniqwen.runtime.loader import load_gguf_runtime
from miniqwen.tokenizer_adapter import TokenizerAdapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--device", default="auto", help="cpu, cuda, cuda:0, or auto")
    parser.add_argument(
        "--compute_dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"]
    )
    args = parser.parse_args()
    runtime = load_gguf_runtime(args.model, device=args.device, compute_dtype=args.compute_dtype)
    tokenizer = TokenizerAdapter(args.tokenizer)
    input_ids = tokenizer.encode(args.prompt)
    output = generate_tokens(
        runtime,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_token_id=runtime.config.eos_token_id,
    )
    print(tokenizer.decode(output[0].detach().cpu().tolist()))


if __name__ == "__main__":
    main()
