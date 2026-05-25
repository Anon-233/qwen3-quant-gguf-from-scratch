# Appendix B GGUF Metadata Reference

本附录记录教学版 GGUF metadata 子集。它不是完整 llama.cpp GGUF schema，而是本项目 runtime 能恢复 Qwen3 dense 文本路径所需的最小字段集合。

## B.1 设计原则

metadata 的作用是让 runtime 不依赖外部 Python 常量恢复模型结构。也就是说，runtime 不应写：

```python
num_layers = 28
hidden_size = 1024
```

而应从 GGUF metadata 恢复：

```python
num_layers = metadata["model.block_count"]
hidden_size = metadata["model.embedding_length"]
```

这些字段在转换时来自 Hugging Face `config.json`，或由 config 明确推导。缺失字段应给出清晰错误，不能静默使用 Qwen3-0.6B 的固定值。

## B.2 Metadata 字段

| key | 含义 | 来源 |
|---|---|---|
| `general.architecture` | 模型架构标识，例如 `qwen3` | config `model_type` 或转换器设置 |
| `general.name` | 模型名称 | CLI 参数或 HF 路径 |
| `general.file_type` | 文件主量化类型 | `--quant` |
| `general.quantization_version` | 教学版量化格式版本 | writer 常量 |
| `model.context_length` | 最大上下文长度 | `max_position_embeddings` |
| `model.embedding_length` | hidden size | `hidden_size` |
| `model.block_count` | decoder layer 数 | `num_hidden_layers` |
| `model.feed_forward_length` | MLP 中间维度 | `intermediate_size` |
| `model.attention.head_count` | query head 数 | `num_attention_heads` |
| `model.attention.head_count_kv` | KV head 数 | `num_key_value_heads` |
| `model.rope.freq_base` | RoPE base | `rope_theta` 或明确 fallback |
| `model.attention.layer_norm_rms_epsilon` | RMSNorm epsilon | `rms_norm_eps` |
| `tokenizer.path` 或 tokenizer hint | tokenizer 加载提示 | CLI 参数 |

## B.3 从 metadata 恢复 shape

runtime 恢复配置后，应能推导：

$$
D_{\text{head}}
= \frac{H}{H_q}
$$

其中：

- $H = \texttt{model.embedding\_length}$。
- $H_q = \texttt{model.attention.head\_count}$。

KV cache 每层每 token 的 shape 为：

$$
[H_{\text{kv}}, D_{\text{head}}]
$$

其中 $H_{\text{kv}} = \texttt{model.attention.head\_count\_kv}$。

MLP 权重 shape 为：

$$
W_{\text{gate}}, W_{\text{up}} \in \mathbb{R}^{I \times H}
$$

$$
W_{\text{down}} \in \mathbb{R}^{H \times I}
$$

其中 $I = \texttt{model.feed\_forward\_length}$。

## B.4 与完整 GGUF 的差异

完整 llama.cpp GGUF 兼容还需要更多内容，包括 tokenizer tokens、scores、token type、chat template、special tokens、rope scaling、官方 tensor naming、官方 quantization layout 等。本项目第一阶段不保证导出的 GGUF 能被 llama.cpp 直接加载。

如果要追求兼容性，必须使用真实 llama.cpp 加载结果作为验收标准，而不是只检查本项目 reader 能读。
