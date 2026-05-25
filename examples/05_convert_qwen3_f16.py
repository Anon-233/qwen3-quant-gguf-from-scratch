from miniqwen.convert.hf_to_gguf import convert_hf_to_gguf

convert_hf_to_gguf("Qwen/Qwen3-0.6B", "outputs/qwen3-0.6b-f16.gguf", quant="f16")
