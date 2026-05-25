# 08 GGUF Format Minimal：为教学 runtime 设计权重文件

## 本章目标

本章解释教学版 GGUF 子集的文件结构。读完后，读者应能说明 header、metadata、tensor directory、tensor data、alignment、dtype 和 quant type 在 runtime 中的作用。

## 背景与问题

模型权重文件不是简单的 tensor dump。runtime 需要知道：

- 这是什么架构；
- 有多少层；
- hidden size、head 数和 RoPE 参数是什么；
- 每个 tensor 的名称、shape、dtype 和数据偏移；
- 量化 tensor 的 block size、scale 和 payload 如何解释。

GGUF 的价值在于把 metadata 和 tensor data 放在同一个文件中，使 runtime 可以独立加载模型。

## 数学定义

alignment 规则：

$$
\operatorname{align}(o,A)
=
\left\lceil\frac{o}{A}\right\rceil A
$$

其中：

- $o$ 是原始 offset；
- $A$ 是 alignment 字节数；
- $\operatorname{align}(o,A)$ 是对齐后的 offset。

对齐能让 tensor data 起始位置满足 runtime 或硬件友好的边界。

## 关键推导

教学版 GGUF 的读取流程：

1. 读取 header，确认 magic 和 version；
2. 读取 metadata JSON；
3. 读取 tensor directory；
4. 根据 offset 读取 tensor data；
5. 对 q8_0/q4_0 执行 dequant；
6. 用 metadata 恢复 `Qwen3Config`；
7. 构造 runtime executor。

对于 q4_0 tensor，payload 不是直接浮点数组，而是：

$$
\text{raw} = [s_0, s_1, \dots, s_{B-1}] \; || \; \text{packed int4 bytes}
$$

## 对应到 Qwen3-0.6B

metadata 至少包含：

- `model.vocab_size`
- `model.embedding_length`
- `model.block_count`
- `model.feed_forward_length`
- `model.attention.head_count`
- `model.attention.head_count_kv`
- `model.attention.head_dim`
- `model.rope.freq_base`
- `model.attention.layer_norm_rms_epsilon`
- `model.tie_word_embeddings`

这些字段都来自 HF config 或 state_dict 检查结果。

## 最小代码实验

写入并读取 toy GGUF：

```bash
uv run python examples/03_write_read_toy_gguf.py
```

检查真实 q4_0 GGUF：

```bash
uv run python scripts/inspect_gguf.py \
  --model outputs/qwen3-0.6b-q4_0.gguf
```

## 常见误区

- 教学版 GGUF 不保证被 llama.cpp 加载。
- metadata 不能手写 Qwen3-0.6B 常量。
- tokenizer 第一版只保存 hint，不实现完整 GGUF tokenizer schema。
- dtype 名称相同不代表 layout 与官方实现完全一致。

## 小结

文件格式的目标是让 runtime 可以恢复结构和权重。教学版 GGUF 的价值在于透明，而不是完整兼容。

## 延伸阅读

参见 `09_hf_to_gguf_conversion.md` 和 `appendices/b_gguf_metadata_reference.md`。
