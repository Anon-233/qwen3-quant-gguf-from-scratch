.PHONY: sync test lint demo-quant demo-block demo-gguf demo-tiny inspect-qwen

sync:
	uv sync

test:
	uv run pytest

lint:
	uv run ruff check src scripts examples tests

demo-quant:
	uv run python examples/01_quantization_math_demo.py

demo-block:
	uv run python examples/02_block_q8_q4_demo.py

demo-gguf:
	uv run python examples/03_write_read_toy_gguf.py

demo-tiny:
	uv run python examples/04_run_tiny_text_model.py

inspect-qwen:
	uv run python scripts/inspect_hf_model.py --model_name_or_path Qwen/Qwen3-0.6B
