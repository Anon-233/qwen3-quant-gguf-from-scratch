# LLM Inference Quantization from Scratch

这是一套中文“教材 + 可运行最小工程”，主线是把 `Qwen/Qwen3-0.6B` 从 Hugging Face 权重转换为教学版 GGUF，再用自定义 Python runtime 执行文本生成。

本项目关注推理阶段量化，不是生产级推理框架，不是完整 GGUF 实现，也不是 llama.cpp、vLLM 或 TensorRT-LLM 的替代品。第一阶段只实现 Qwen3 dense decoder-only 文本生成路径。

```
HF model -> quantization -> teaching GGUF -> custom runtime -> generation
```

## 为什么选择 Qwen/Qwen3-0.6B

`Qwen/Qwen3-0.6B` 足够小，适合 CPU 上做教学实验；同时它包含现代 decoder-only LLM 的关键结构：RMSNorm、RoPE、GQA attention、SwiGLU MLP、token embedding 和 lm_head。代码可以针对 Qwen3 dense 的命名约定，但所有结构参数都必须来自 `config.json`。

## 安装

```bash
uv sync
```

兼容场景也提供 `requirements.txt`，但本仓库的主流程使用 uv。

## 快速开始

```bash
uv run pytest
uv run python examples/01_quantization_math_demo.py
uv run python examples/02_block_q8_q4_demo.py
uv run python examples/03_write_read_toy_gguf.py
uv run python examples/04_run_tiny_text_model.py
```

默认测试使用 tiny config，不下载完整 `Qwen/Qwen3-0.6B`。

## 检查 HF 模型

```bash
uv run python scripts/inspect_hf_model.py \
  --model_name_or_path Qwen/Qwen3-0.6B
```

输出包括 config summary、architecture、tensor names、tensor shapes、dtype distribution、tied embedding 状态、参数量估算，以及由 config 推导出的 runtime shape。

## 导出 GGUF

```bash
uv run python scripts/quantize_hf_to_gguf.py \
  --model_name_or_path Qwen/Qwen3-0.6B \
  --output ./outputs/qwen3-0.6b-f16.gguf \
  --quant f16

uv run python scripts/quantize_hf_to_gguf.py \
  --model_name_or_path Qwen/Qwen3-0.6B \
  --output ./outputs/qwen3-0.6b-q8_0.gguf \
  --quant q8_0 \
  --block_size 32

uv run python scripts/quantize_hf_to_gguf.py \
  --model_name_or_path Qwen/Qwen3-0.6B \
  --output ./outputs/qwen3-0.6b-q4_0.gguf \
  --quant q4_0 \
  --block_size 32
```

查看文件：

```bash
uv run python scripts/inspect_gguf.py --model ./outputs/qwen3-0.6b-q4_0.gguf
```

## 自定义 Runtime 生成

```bash
uv run python scripts/run_gguf.py \
  --model ./outputs/qwen3-0.6b-q4_0.gguf \
  --tokenizer Qwen/Qwen3-0.6B \
  --prompt "请用三句话解释大语言模型量化。" \
  --max_new_tokens 64 \
  --temperature 0.7 \
  --top_p 0.9 \
  --device auto \
  --compute_dtype auto
```

`--device auto` 会在 `torch.cuda.is_available()` 为真时选择 CUDA，否则回退 CPU。也可以显式使用 `--device cuda --compute_dtype float16`。

Python API 也默认使用同样策略：

```python
from miniqwen.runtime.loader import load_gguf_runtime

runtime = load_gguf_runtime("outputs/qwen3-0.6b-q8_0.gguf")
print(runtime.device)  # cuda if available, otherwise cpu
```

## 与 Transformers 对比

```bash
uv run python scripts/compare_with_transformers.py \
  --hf_model Qwen/Qwen3-0.6B \
  --gguf_model ./outputs/qwen3-0.6b-f16.gguf \
  --prompt "北京是中国的" \
  --max_tokens 8 \
  --device cuda \
  --compute_dtype float16
```

Transformers model 只作为 correctness reference；自定义 runtime 的 forward 不调用 Transformers model forward。

量化 GGUF 与 f16 GGUF 对比：

```bash
uv run python scripts/compare_quantized_gguf.py \
  --baseline ./outputs/qwen3-0.6b-f16.gguf \
  --quantized ./outputs/qwen3-0.6b-q8_0.gguf ./outputs/qwen3-0.6b-q4_0.gguf \
  --tokenizer ./models/Qwen3-0.6B \
  --prompt "北京是中国的" \
  --max_new_tokens 8 \
  --device cuda \
  --compute_dtype float16
```

更大的 10-prompt 评测套件：

```bash
uv run python scripts/evaluate_quant_suite.py \
  --prompts data/eval/qwen3_quant_prompts.jsonl \
  --baseline ./outputs/qwen3-0.6b-f16.gguf \
  --quantized ./outputs/qwen3-0.6b-q8_0.gguf ./outputs/qwen3-0.6b-q4_0.gguf \
  --tokenizer ./models/Qwen3-0.6B \
  --top_k 10 \
  --device cuda \
  --compute_dtype float16
```

消融式评测，同时比较 f16/q8/q4 GGUF 与 Transformers reference：

```bash
uv run python scripts/evaluate_quant_suite.py \
  --prompts data/eval/qwen3_quant_prompts.jsonl \
  --baseline ./outputs/qwen3-0.6b-f16.gguf \
  --quantized ./outputs/qwen3-0.6b-q8_0.gguf ./outputs/qwen3-0.6b-q4_0.gguf \
  --tokenizer ./models/Qwen3-0.6B \
  --transformers_reference ./models/Qwen3-0.6B \
  --top_k 10 \
  --device cuda \
  --compute_dtype float16 \
  --json_output reports/qwen3_quant_suite_ablation_cuda.json \
  --md_output reports/qwen3_quant_suite_ablation_cuda.md
```

该评测会生成 `reports/*.json` 和 `reports/*.md`。`reports/` 是生成物目录，不纳入版本管理；教程中的汇总结果见 `book/13_correctness_validation.md` 和 `book/appendices/f_qwen3_quant_eval_results.md`。

## 教学章节

教材在 `book/` 下：

1. `00_preface.md`
2. `01_qwen3_model_inspection.md`
3. `02_text_generation_path.md`
4. `03_inference_cost_model.md`
5. `04_quantization_math_basics.md`
6. `05_weight_only_quantization.md`
7. `06_block_quantization_q8_q4.md`
8. `07_quantization_error_analysis.md`
9. `08_gguf_format_minimal.md`
10. `09_hf_to_gguf_conversion.md`
11. `10_custom_runtime_design.md`
12. `11_forward_pass_from_scratch.md`
13. `12_token_generation_and_sampling.md`
14. `13_correctness_validation.md`
15. `14_benchmarking_memory_latency_quality.md`
16. `15_modern_inference_quantization_overview.md`
17. `16_extensions_gptq_awq_fp8_kv_cache.md`

公式使用 Markdown LaTeX：行内 `$...$`，独立公式 `$$...$$`。

## Config-driven Implementation

本项目禁止硬编码 Qwen3-0.6B 的结构参数。以下值全部来自 HF `config.json` 或 GGUF metadata：

- `vocab_size`
- `hidden_size`
- `intermediate_size`
- `num_hidden_layers`
- `num_attention_heads`
- `num_key_value_heads`
- `head_dim`
- `max_position_embeddings`
- `rms_norm_eps`
- `rope_theta`

例如 KV cache 估算使用：

$$
M_{\text{KV}} =
2 \times L \times H_{\text{kv}} \times D_{\text{head}}
\times T \times B \times B_{\text{elem}}
$$

其中 $L$、$H_{\text{kv}}$、$D_{\text{head}}$ 都来自 config 或由 config 推导。

## 支持矩阵

| 功能 | 状态 |
|---|---|
| Qwen3 dense 文本路径 | 支持 |
| f16 teaching GGUF | 支持 |
| q8_0 teaching GGUF | 支持 |
| q4_0 teaching GGUF | 支持 |
| 自定义 Python runtime | 支持 |
| Transformers tokenizer | 支持 |
| Transformers correctness reference | 支持 |
| 完整 llama.cpp GGUF 兼容 | 不保证 |
| MoE / 多模态 / 训练 | 不支持 |
| CUDA / Triton kernel | 不支持 |
| GPTQ / AWQ / FP8 实现 | 综述与扩展路线 |

q4_0 在本项目 PyTorch runtime 中不一定更快，因为当前路径是 dequant + matmul。真实加速依赖 kernel、runtime、硬件和 workload。

当前自定义 runtime 默认使用 `device="auto"`，有 CUDA 时优先 CUDA；显式传 `device="cpu"` 才会强制 CPU。注意这不是低比特 fused kernel：q8_0/q4_0 仍会反量化为浮点 tensor，再用 PyTorch matmul 执行。真正的 int8/int4 kernel、Triton/CUDA fused dequant + GEMM 仍属于扩展目标。
