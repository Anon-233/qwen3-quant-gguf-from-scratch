from __future__ import annotations

import argparse
from collections import Counter

from miniqwen.config import Qwen3Config
from miniqwen.convert.hf_config_inspector import summarize_config
from miniqwen.convert.hf_weight_loader import load_hf_state_dict, tensor_summary
from miniqwen.convert.tensor_mapper import hf_to_runtime_name, map_state_dict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    args = parser.parse_args()
    config = Qwen3Config.from_pretrained(args.model_name_or_path)
    print("Config summary")
    for k, v in summarize_config(config).items():
        print(f"  {k}: {v}")
    state = load_hf_state_dict(args.model_name_or_path)
    mapped, skipped = map_state_dict(state, config, strict_shapes=True)
    print(f"\nTensor count: {len(state)}")
    print(f"Runtime text-path tensors: {len(mapped)}")
    print(f"Skipped tensors: {len(skipped)}")
    print(f"Tied embedding from config: {config.tie_word_embeddings}")
    dtypes = Counter(str(t.dtype) for t in state.values())
    print(f"Dtype distribution: {dict(dtypes)}")
    print(f"Parameter count: {sum(t.numel() for t in state.values()):,}")
    print("\nFirst tensors")
    for row in tensor_summary(state)[:40]:
        rt = hf_to_runtime_name(row["name"])
        print(f"  {row['name']} -> {rt} shape={row['shape']} dtype={row['dtype']}")


if __name__ == "__main__":
    main()
