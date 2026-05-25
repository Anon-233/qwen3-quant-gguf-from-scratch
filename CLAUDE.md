# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Chinese-language textbook + minimal runnable implementation of LLM inference quantization. The pipeline converts `Qwen/Qwen3-0.6B` from Hugging Face weights into a teaching-grade GGUF format, then runs text generation through a custom Python runtime.

```
HF model -> quantization -> teaching GGUF -> custom runtime -> generation
```

This is an educational project, not a production inference framework. It is not llama.cpp/vLLM compatible.

## Build, test, and lint

```bash
# Install dependencies
uv sync

# Run all tests (defaults to tiny config, no model download needed)
uv run pytest

# Run a single test file
uv run pytest tests/test_block_q8_0.py

# Run a single test function
uv run pytest tests/test_block_q8_0.py::test_quantize_dequantize_roundtrip -v

# Lint
uv run ruff check src scripts examples tests
```

Make targets: `make sync`, `make test`, `make lint`.

## Architecture

Package: `miniqwen` under `src/`.

### Config (`config.py`)
`Qwen3Config` dataclass is the single source of truth for model structure. It can be constructed from:
- `from_dict()` — a raw dict
- `from_json_file()` — a HF `config.json`
- `from_pretrained()` — HF model name or local path (loads `config.json` from disk or hub)
- `from_metadata()` — GGUF metadata keys (e.g., `model.embedding_length`)

All tensor shapes and KV cache sizes are derived from these config fields at runtime. Never hardcode Qwen3-0.6B's numbers.

### GGUF format (`gguf/`)
A simplified, teaching-grade GGUF format (not full llama.cpp spec). Key files:
- `reader.py` — `GGUFReader`: parses header/metadata/tensor directory, reads f16/f32/q8_0/q4_0 tensors, dequantizes on load
- `writer.py` — `GGUFWriter`: writes metadata + aligned tensor data
- `constants.py` — magic bytes, dtype IDs, version
- `tensor_info.py` — `TensorInfo` namedtuple
- `mapping_qwen3.py` — HF tensor name to runtime tensor name mapping for Qwen3
- `metadata.py` — metadata helpers

The GGUF stores quantized tensors as: `[scales (f32)][quantized data (int8/uint8)]` preceded by scale/block metadata in the tensor directory.

### Quantization (`quant/`)
- `int_quant.py` — basic int8/int4 symmetric quantization math
- `block_q8_0.py` — block-wise q8_0: per-block symmetric scale, dequantizes to float32 for matmul
- `block_q4_0.py` — block-wise q4_0: 4-bit pairs packed into uint8, dequantizes to float32 for matmul
- `error_metrics.py` — MSE, MAE, SNR metrics for quality evaluation

Quantization is applied only to 2D+ tensors (weight matrices). Scalar/1D tensors (norm weights, biases) stay f16.

### Model (`model/`)
Pure Python/PyTorch Qwen3 decoder-only components, assembled in `text_model.py`:
- `Qwen3TextModel.forward()` executes the full forward pass: embed → [attn_norm → GQA attention → residual → ffn_norm → SwiGLU MLP → residual] × L → output_norm → lm_head
- `attention.py` — GQA with RoPE and optional QK layer norm
- `rope.py` — Rotary position embeddings
- `mlp.py` — SwiGLU (gate/up/down projection)
- `norm.py` — RMS normalization
- `kv_cache.py` — KVCache for autoregressive decoding
- `sampling.py` — temperature, top-k, top-p sampling

### Runtime (`runtime/`)
- `loader.py` — `load_gguf_runtime(path)` → `RuntimeExecutor`: reads GGUF, builds config from metadata, dequantizes tensors, creates executor
- `executor.py` — `RuntimeExecutor(Qwen3TextModel)`: thin wrapper, the model itself
- `tensor_store.py` — `TensorStore`: wraps `GGUFReader`, dequantizes all tensors to a state_dict on the target device
- `device.py` — `resolve_device()` / `resolve_compute_dtype()`: `"auto"` → CUDA if available, else CPU

### Convert (`convert/`)
- `hf_to_gguf.py` — top-level `convert_hf_to_gguf()`: loads HF config → loads HF state_dict → maps tensor names → optionally quantizes → writes GGUF
- `hf_weight_loader.py` — loads safetensors or pytorch_model.bin from HF
- `hf_config_inspector.py` — inspects and prints HF model structure
- `tensor_mapper.py` — maps HF tensor names (e.g., `model.layers.0.self_attn.q_proj.weight`) to runtime names (e.g., `blk.0.attn_q.weight`)

### Generation (`generation.py`)
`generate_tokens()` — autoregressive loop with optional KV cache. Calls `model.forward()` then `sample_next_token()` each step.

### Evaluation (`evaluation/`)
- `compare_logits.py` — logit-level comparison between custom runtime and Transformers
- `memory.py` — memory usage estimation
- `latency.py` — latency measurement

## Config-driven principle

All model dimensions come from `Qwen3Config`. The config itself is loaded from HF `config.json` or GGUF metadata at runtime. Tests use a `tiny_config` fixture (vocab=32, hidden=16, intermediate=32, layers=2, heads=4, kv_heads=2, head_dim=4, max_pos=64) to avoid downloading real weights.

## Testing

Tests live in `tests/` and use `conftest.py::tiny_config`. They verify shape correctness, quantization roundtrips, tensor mappings, reader/writer fidelity, and end-to-end generation with tiny random weights. No real model download is required for the default test suite.

## Scripts

Scripts in `scripts/` are CLI entry points:
- `inspect_hf_model.py` — print config, tensor names/shapes/dtypes from HF
- `quantize_hf_to_gguf.py` — full HF→GGUF conversion (f16/q8_0/q4_0)
- `run_gguf.py` — generate text from a GGUF file
- `inspect_gguf.py` — dump GGUF metadata and tensor directory
- `compare_with_transformers.py` — side-by-side logit comparison
- `compare_quantized_gguf.py` — compare quantized vs f16 GGUF outputs
- `evaluate_quant_suite.py` — multi-prompt evaluation suite with JSON/Markdown reports
- `benchmark_runtime.py` — latency and memory benchmarks

## Notebooks / book

Tutorial chapters in `book/` (Chinese, Markdown with LaTeX math). Evaluation result appendices in `book/appendices/`.

## Dependencies

Python >= 3.12, managed with `uv`. Key deps: torch, transformers, safetensors, huggingface-hub, numpy, tqdm. Dev deps: pytest, ruff.
