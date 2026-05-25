# 09 HF to GGUF Conversion：从生态权重到教学 runtime 权重

## 本章目标

本章讲解 HF 权重到教学版 GGUF 的转换流程。读完后，读者应能说明 config、tokenizer、safetensors、tensor mapping、量化、metadata 写入和 inspect GGUF 之间的关系。

## 背景与问题

Hugging Face 模型目录面向通用生态：它包含 `config.json`、tokenizer 文件、safetensors 权重、generation config 和 README。自定义 runtime 需要的是另一种组织方式：一个可以直接恢复模型结构和 tensor store 的权重文件。

转换器的职责不是“改文件后缀”，而是把生态格式中的信息重新组织为 runtime 需要的格式。

## 数学定义

转换阶段最重要的数学对象是 shape contract。以 attention 为例：

$$
W_q \in \mathbb{R}^{(A D) \times H}
$$

$$
W_k,W_v \in \mathbb{R}^{(A_{\text{kv}}D) \times H}
$$

$$
W_o \in \mathbb{R}^{H \times (A D)}
$$

其中 $A$、$A_{\text{kv}}$、$D$、$H$ 都来自 config。转换器必须用这些公式校验 state_dict。

## 关键推导

转换流程可以写成：

$$
\text{HF config} \rightarrow C
$$

$$
\text{HF state\_dict} \rightarrow
\{(n_i, W_i)\}_{i=1}^{N}
$$

$$
\operatorname{map}(n_i) = r_i
$$

$$
\operatorname{shape}(W_i) \stackrel{?}{=}
\operatorname{expected\_shape}(r_i, C)
$$

只有当 shape 校验通过，才允许写入 GGUF。否则 runtime 可能在很深的 forward 里才崩溃，或者更糟糕：静默产生错误 logits。

## 对应到 Qwen3-0.6B

Qwen3 常见 HF 名称到 runtime 名称示例：

| HF name | runtime name |
|---|---|
| `model.embed_tokens.weight` | `token_embd.weight` |
| `model.layers.0.self_attn.q_proj.weight` | `blk.0.attn_q.weight` |
| `model.layers.0.self_attn.k_norm.weight` | `blk.0.attn_k_norm.weight` |
| `model.layers.0.mlp.gate_proj.weight` | `blk.0.ffn_gate.weight` |
| `model.norm.weight` | `output_norm.weight` |
| `lm_head.weight` | `output.weight` |

代码可以针对 Qwen3 命名约定，但不能假设 Qwen3-0.6B 的固定层数或维度。

## 最小代码实验

检查转换路径：

```bash
uv run python scripts/inspect_text_path.py \
  --model_name_or_path models/Qwen3-0.6B
```

导出 f16：

```bash
uv run python scripts/quantize_hf_to_gguf.py \
  --model_name_or_path models/Qwen3-0.6B \
  --output outputs/qwen3-0.6b-f16.gguf \
  --quant f16
```

导出 q8_0/q4_0 时添加 `--quant q8_0` 或 `--quant q4_0`。

## 常见误区

- 只映射 tensor 名称，不做 shape 校验。
- 用手写 metadata 代替 config。
- 忽略 `lm_head.weight` 与 embedding 的 tied 状态。
- 把 tokenizer 完整兼容问题混入第一阶段目标。

## 小结

转换器是 HF 生态与自定义 runtime 的边界。它必须严格、显式、可检查。

## 延伸阅读

参见 `src/miniqwen/convert/tensor_mapper.py`。
