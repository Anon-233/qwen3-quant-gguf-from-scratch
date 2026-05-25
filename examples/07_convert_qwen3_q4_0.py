from miniqwen.convert.hf_to_gguf import convert_hf_to_gguf

convert_hf_to_gguf("Qwen/Qwen3-0.6B", "outputs/qwen3-0.6b-q4_0.gguf", quant="q4_0", block_size=32)
