from __future__ import annotations

from pathlib import Path

import torch

from miniqwen.config import Qwen3Config
from miniqwen.convert.hf_weight_loader import load_hf_state_dict
from miniqwen.convert.tensor_mapper import map_state_dict
from miniqwen.gguf.writer import GGUFWriter
from miniqwen.quant.block_q4_0 import quantize_q4_0
from miniqwen.quant.block_q8_0 import quantize_q8_0


def write_runtime_state_to_gguf(
    state_dict: dict[str, torch.Tensor],
    config: Qwen3Config,
    output: str | Path,
    quant: str = "f16",
    block_size: int = 32,
    name: str = "qwen3",
) -> None:
    if quant not in {"f16", "q8_0", "q4_0"}:
        raise ValueError("quant must be one of: f16, q8_0, q4_0")
    metadata = config.to_metadata(name=name)
    metadata["general.file_type"] = quant
    metadata["tokenizer.hint"] = name
    writer = GGUFWriter(metadata=metadata)
    for tensor_name, tensor in sorted(state_dict.items()):
        if quant == "q8_0" and tensor.ndim >= 2:
            writer.add_quantized_tensor(tensor_name, quantize_q8_0(tensor, block_size=block_size))
        elif quant == "q4_0" and tensor.ndim >= 2:
            writer.add_quantized_tensor(tensor_name, quantize_q4_0(tensor, block_size=block_size))
        else:
            writer.add_tensor(tensor_name, tensor, dtype="f16")
    writer.write(output)


def convert_hf_to_gguf(
    model_name_or_path: str,
    output: str | Path,
    quant: str = "f16",
    block_size: int = 32,
) -> None:
    config = Qwen3Config.from_pretrained(model_name_or_path)
    hf_state = load_hf_state_dict(model_name_or_path)
    runtime_state, _skipped = map_state_dict(hf_state, config, strict_shapes=True)
    if any(name.endswith(".attn_q_norm.weight") for name in runtime_state):
        config.qk_layernorm = True
    write_runtime_state_to_gguf(
        runtime_state,
        config,
        output,
        quant=quant,
        block_size=block_size,
        name=model_name_or_path,
    )
