from __future__ import annotations

import argparse

import torch

from miniqwen.evaluation.latency import Timer
from miniqwen.runtime.loader import load_gguf_runtime
from miniqwen.tokenizer_adapter import TokenizerAdapter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--prompt", default="北京是中国的")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--device", default="auto", help="cpu, cuda, cuda:0, or auto")
    parser.add_argument(
        "--compute_dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"]
    )
    args = parser.parse_args()
    runtime = load_gguf_runtime(args.model, device=args.device, compute_dtype=args.compute_dtype)
    ids = TokenizerAdapter(args.tokenizer).encode(args.prompt)
    if runtime.device.type == "cuda":
        torch.cuda.synchronize(runtime.device)
    with torch.no_grad(), Timer() as timer:
        for _ in range(args.steps):
            runtime.forward(ids)
        if runtime.device.type == "cuda":
            torch.cuda.synchronize(runtime.device)
    print(
        {
            "device": str(runtime.device),
            "compute_dtype": str(runtime.compute_dtype),
            "steps": args.steps,
            "seconds": timer.seconds,
            "forward_per_s": args.steps / timer.seconds,
        }
    )


if __name__ == "__main__":
    main()
