# 01 Qwen3 Model Inspection：先看懂模型，再谈量化

## 本章目标

本章训练读者养成一个系统习惯：在量化、转换或部署前，先检查模型配置和权重。读完本章后，读者应能：

- 从 Hugging Face `config.json` 构造内部配置；
- 列出 safetensors 权重名称、shape 和 dtype；
- 识别 Qwen3 dense decoder-only 的文本生成路径；
- 判断 tied embedding；
- 从 config 推导 attention、MLP、embedding 和 lm_head 的 shape；
- 生成 HF tensor name 到 runtime tensor name 的映射表。

## 背景与问题

很多量化 bug 不是数学公式错了，而是张量错位：把 Q projection 当成 K projection，把 MLP up/gate 搞反，把 head_dim 假设成 `hidden_size / num_heads`，或者忽略 Q/K norm。

因此，转换器的第一步不是量化，而是 inspection。inspection 的目标是回答：

1. 这个模型的结构参数是什么？
2. 哪些权重进入文本生成路径？
3. 每个权重的 shape 是否能由 config 推导？
4. state_dict 里是否存在 runtime 不支持的张量？

## 数学定义

令：

- $V$ 为 `vocab_size`；
- $H$ 为 `hidden_size`；
- $I$ 为 `intermediate_size`；
- $L$ 为 `num_hidden_layers`；
- $A$ 为 `num_attention_heads`；
- $A_{\text{kv}}$ 为 `num_key_value_heads`；
- $D$ 为 `head_dim`。

Qwen3 attention 的投影矩阵 shape 为：

$$
W_q \in \mathbb{R}^{(A D) \times H}
$$

$$
W_k, W_v \in \mathbb{R}^{(A_{\text{kv}} D) \times H}
$$

$$
W_o \in \mathbb{R}^{H \times (A D)}
$$

MLP 的 SwiGLU 权重 shape 为：

$$
W_g, W_u \in \mathbb{R}^{I \times H},
\qquad
W_d \in \mathbb{R}^{H \times I}
$$

embedding 与输出头：

$$
E \in \mathbb{R}^{V \times H},
\qquad
W_{\text{lm}} \in \mathbb{R}^{V \times H}
$$

如果 `tie_word_embeddings=True` 且实际权重相同，runtime 可以使用 $E$ 作为输出头。

## 关键推导

不要默认 $D = H / A$。有些 Qwen3 配置显式给出 `head_dim`，且 $A D$ 可能不等于 $H$。因此 attention 输出 reshape 应使用 $A D$，而不是 $H$：

$$
Q \in \mathbb{R}^{B \times T \times A \times D}
$$

$$
K,V \in \mathbb{R}^{B \times T \times A_{\text{kv}} \times D}
$$

GQA 通过重复 KV head 对齐 Q head：

$$
r = \frac{A}{A_{\text{kv}}}
$$

$$
K_{\text{repeat}}, V_{\text{repeat}}
\in \mathbb{R}^{B \times T \times A \times D}
$$

若 $A$ 不能被 $A_{\text{kv}}$ 整除，说明 config 与 GQA 假设不兼容，应直接报错。

## 对应到 Qwen3-0.6B

本项目实测 `Qwen/Qwen3-0.6B` 的关键配置为：

| 字段 | 值 |
|---|---:|
| `vocab_size` | 151936 |
| `hidden_size` | 1024 |
| `intermediate_size` | 3072 |
| `num_hidden_layers` | 28 |
| `num_attention_heads` | 16 |
| `num_key_value_heads` | 8 |
| `head_dim` | 128 |
| `max_position_embeddings` | 40960 |
| `rope_theta` | 1000000 |

注意这里 $A D = 16 \times 128 = 2048$，不等于 $H=1024$。这正是 config-driven 实现的必要性。

## 最小代码实验

检查本地或远程模型：

```bash
uv run python scripts/inspect_hf_model.py \
  --model_name_or_path models/Qwen3-0.6B
```

检查第一阶段 runtime 会使用哪些 tensor：

```bash
uv run python scripts/inspect_text_path.py \
  --model_name_or_path models/Qwen3-0.6B
```

输出中的每一行都应满足 `shape == expected`。

## 常见误区

- 根据模型名称猜测层数和 head 数。
- 忽略 `head_dim` 字段。
- 只检查 tensor 名称，不检查 shape。
- 认为 `tie_word_embeddings=True` 就一定可以不检查实际权重。
- 把 Qwen3 dense 和 MoE / multimodal 路径混在一起实现。

## 小结

inspection 是整个项目的地基。只有 config、state_dict 和 runtime shape 三者一致，后面的量化误差才有解释意义。

## 延伸阅读

参见 `appendices/a_tensor_name_mapping.md`，其中列出了本项目使用的 HF 到 runtime 名称映射。
