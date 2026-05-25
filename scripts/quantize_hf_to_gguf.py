from __future__ import annotations

import argparse

from miniqwen.convert.hf_to_gguf import convert_hf_to_gguf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--quant", choices=["f16", "q8_0", "q4_0"], required=True)
    parser.add_argument("--block_size", type=int, default=32)
    args = parser.parse_args()
    convert_hf_to_gguf(
        args.model_name_or_path, args.output, quant=args.quant, block_size=args.block_size
    )
    print(f"Wrote {args.quant} teaching GGUF to {args.output}")


if __name__ == "__main__":
    main()
