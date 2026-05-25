# Appendix F Qwen3 Quant Evaluation Case Study

本附录把一次真实 `Qwen/Qwen3-0.6B` 本地评测整理成教材案例。它的目的不是给出通用排行榜，而是示范如何同时观察 runtime correctness、量化误差、Transformers reference 偏差和生成速度。

模型文件位于 `models/Qwen3-0.6B`，导出文件位于 `outputs/`。这些目录被 `.gitignore` 忽略，不进入源码版本管理。

## F.1 测试集

测试集位于 `data/eval/qwen3_quant_prompts.jsonl`，包含 10 个中文 prompt。它不是通用能力评测集，而是覆盖常见生成形态的 smoke/ablation suite。

| id | category | purpose |
|---|---|---|
| `zh_fact_001` | fact_completion | 常识续写 |
| `zh_explain_001` | technical_explanation | 技术解释 |
| `zh_math_001` | math_reasoning | 简单参数量计算 |
| `zh_code_001` | code_explanation | Python 代码解释 |
| `zh_summary_001` | summarization | 摘要 |
| `zh_compare_001` | comparison | 概念比较 |
| `zh_safety_001` | refusal_boundary | 学习误区 |
| `zh_translation_001` | translation | 翻译 |
| `zh_structured_001` | structured_output | JSON 风格输出 |
| `zh_sampling_001` | generation | 开放生成 |

这组 prompt 的设计原则是：覆盖不同 logits 分布和输出结构，但保持运行成本可控。

## F.2 评测命令

基础量化对比：

```bash
uv run python scripts/evaluate_quant_suite.py \
  --prompts data/eval/qwen3_quant_prompts.jsonl \
  --baseline outputs/qwen3-0.6b-f16.gguf \
  --quantized outputs/qwen3-0.6b-q8_0.gguf outputs/qwen3-0.6b-q4_0.gguf \
  --tokenizer models/Qwen3-0.6B \
  --top_k 10 \
  --device cuda \
  --compute_dtype float16
```

加入 Transformers reference 的消融对比：

```bash
uv run python scripts/evaluate_quant_suite.py \
  --prompts data/eval/qwen3_quant_prompts.jsonl \
  --baseline outputs/qwen3-0.6b-f16.gguf \
  --quantized outputs/qwen3-0.6b-q8_0.gguf outputs/qwen3-0.6b-q4_0.gguf \
  --tokenizer models/Qwen3-0.6B \
  --transformers_reference models/Qwen3-0.6B \
  --top_k 10 \
  --device cuda \
  --compute_dtype float16
```

## F.3 指标解释

对每个 prompt，评测会比较下一 token logits：

$$
\operatorname{MSE}(z, \hat{z})
= \frac{1}{V}\sum_{i=1}^{V}(z_i-\hat{z}_i)^2
$$

其中 $z$ 是 reference logits，$\hat{z}$ 是待测模型 logits，$V$ 是词表大小。

cosine similarity 衡量 logits 方向是否接近：

$$
\cos(z,\hat{z})
= \frac{z^\top \hat{z}}{\|z\|_2\|\hat{z}\|_2}
$$

top-k overlap 衡量最可能 token 集合是否一致：

$$
\operatorname{overlap@k}
= \frac{|TopK(z) \cap TopK(\hat{z})|}{k}
$$

next-token match 衡量 greedy 下一 token 是否相同。它比 cosine 更严格，但只观察一个 token。

## F.4 Aggregate Metrics: Quantized vs f16 GGUF

| model | avg cosine | avg MSE | avg top-k overlap | next-token match | avg tok/s |
|---|---:|---:|---:|---:|---:|
| q8_0 CUDA fp16 | 0.999826 | 0.003259 | 1.000 | 100.00% | 64.92 |
| q4_0 CUDA fp16 | 0.965181 | 0.700672 | 0.560 | 60.00% | 65.26 |

这张表回答的是：量化版本相对于本项目 f16 GGUF runtime 偏离多少。q8_0 基本保持 logits 排序；q4_0 文件更小，但 logits 偏差明显更大。

## F.5 Ablation Metrics: vs Transformers

| comparison | avg cosine | avg MSE | avg top-k overlap | next-token match |
|---|---:|---:|---:|---:|
| f16 GGUF vs Transformers | 0.999986 | 0.000071 | 0.980 | 100.00% |
| q8_0 GGUF vs Transformers | 0.999825 | 0.003274 | 0.980 | 100.00% |
| q4_0 GGUF vs Transformers | 0.965154 | 0.701718 | 0.560 | 60.00% |

这张表把误差拆成三层：

- `f16 GGUF vs Transformers`：主要衡量自定义 runtime 与 reference 实现的偏差。
- `q8_0/q4_0 vs f16 GGUF`：主要衡量量化引入的偏差。
- `q8_0/q4_0 vs Transformers`：衡量端到端偏差。

因此 baseline 不能只选一个。f16 GGUF 是 isolating quantization error 的好 reference；Transformers 是检查 runtime 实现风险的外部 reference。两者一起使用，才能区分“量化误差”和“runtime 写错”。

## F.6 速度与后端说明

当前 PyTorch CUDA 后端可以运行 f16/q8_0/q4_0 GGUF。CUDA 10-prompt suite 的平均 greedy generation 速度为：

| model | avg tok/s |
|---|---:|
| f16 teaching GGUF | 64.85 |
| q8_0 teaching GGUF | 64.92 |
| q4_0 teaching GGUF | 65.26 |

这说明 CUDA backend 能减少 CPU 等待时间并支持真实模型验证。它不说明 q4_0 具备生产级低比特加速，因为 q8_0/q4_0 当前路径仍然是：

```text
读取量化权重
  -> 反量化为 fp16 tensor
  -> PyTorch matmul
```

真正的低比特加速还需要 fused dequant + matmul kernel、lazy loading 或 layer-wise streaming、KV cache 和 attention kernel 优化。

## F.7 教材结论

这次案例支持三个教学结论：

1. f16 GGUF 与 Transformers 高度接近，说明自定义 runtime 的主路径基本正确。
2. q8_0 是第一阶段更稳健的量化版本，适合验证教学版 GGUF 和 runtime。
3. q4_0 能显著降低文件大小，但 logits 误差和 top-k 变化更明显，不能只凭“能生成文本”判断质量。
