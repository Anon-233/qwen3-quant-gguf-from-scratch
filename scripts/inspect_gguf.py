from __future__ import annotations

import argparse

from miniqwen.gguf.constants import DTYPE_NAMES
from miniqwen.gguf.reader import GGUFReader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    reader = GGUFReader(args.model)
    print("Metadata")
    for k, v in sorted(reader.metadata.items()):
        print(f"  {k}: {v}")
    print("\nTensors")
    for name, info in sorted(reader.tensors.items()):
        dtype_name = DTYPE_NAMES.get(info.dtype, info.dtype)
        print(
            f"  {name}: shape={info.shape} dtype={dtype_name} "
            f"bytes={info.nbytes}"
        )


if __name__ == "__main__":
    main()
