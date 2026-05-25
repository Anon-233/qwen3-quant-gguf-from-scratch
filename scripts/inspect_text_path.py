from __future__ import annotations

import argparse

from miniqwen.config import Qwen3Config
from miniqwen.convert.hf_weight_loader import load_hf_state_dict
from miniqwen.convert.tensor_mapper import (
    expected_runtime_shape,
    hf_to_runtime_name,
    map_state_dict,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    args = parser.parse_args()
    config = Qwen3Config.from_pretrained(args.model_name_or_path)
    state = load_hf_state_dict(args.model_name_or_path)
    mapped, skipped = map_state_dict(state, config, strict_shapes=True)
    print("Text path tensors")
    for hf_name, tensor in sorted(state.items()):
        rt = hf_to_runtime_name(hf_name)
        if rt is None:
            continue
        expected = expected_runtime_shape(rt, config)
        ok = tuple(tensor.shape) == expected
        print(
            f"  {hf_name} -> {rt} shape={tuple(tensor.shape)} "
            f"expected={expected} ok={ok}"
        )
    print("\nSkipped tensors")
    for name in skipped[:80]:
        print(f"  {name}: not used by first-stage dense text runtime")
    print(f"\nMapped {len(mapped)} tensors from config-driven shapes.")


if __name__ == "__main__":
    main()
