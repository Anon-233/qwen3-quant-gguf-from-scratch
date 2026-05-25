# Appendix A Tensor Name Mapping

本附录给出本项目第一阶段使用的 Qwen3 dense tensor mapping。mapping 可以针对 Qwen3 的 Hugging Face 命名约定，但不能把 `Qwen/Qwen3-0.6B` 的具体层数或维度写死。转换器必须先读取 `config.json`，再用实际 `state_dict` 校验 tensor 是否存在、shape 是否匹配。

## A.1 Mapping 原则

本项目把 Hugging Face tensor name 映射为 runtime 更容易读取的短名称：

```text
model.layers.{i}.self_attn.q_proj.weight
  -> blk.{i}.attn_q.weight
```

其中 `{i}` 的范围不是写死常量，而是：

$$
i \in [0, L-1]
$$

其中 $L = \texttt{num\_hidden\_layers}$，来自 Hugging Face config。

mapping 执行时应做三类检查：

- **存在性检查**：HF tensor name 必须在 `state_dict` 中。
- **shape 检查**：tensor shape 必须和 config 推导结果一致。
- **覆盖检查**：第一阶段 runtime 需要的 tensor 必须全部映射；跳过的 tensor 要说明原因。

## A.2 全局权重

| HF tensor | Runtime tensor | 期望 shape | shape 来源 |
|---|---|---|---|
| `model.embed_tokens.weight` | `token_embd.weight` | $[V, H]$ | `vocab_size`, `hidden_size` |
| `model.norm.weight` | `output_norm.weight` | $[H]$ | `hidden_size` |
| `lm_head.weight` | `output.weight` | $[V, H]$ | `vocab_size`, `hidden_size` |

其中：

- $V$ 是词表大小，来自 `vocab_size`。
- $H$ 是 hidden size，来自 `hidden_size`。

如果 `tie_word_embeddings=true`，并且实际权重中没有单独的 `lm_head.weight`，runtime 可以使用 `token_embd.weight` 作为输出投影。是否 tied 不应只相信 config；还要结合实际权重文件判断。

## A.3 Attention 权重

| HF tensor | Runtime tensor | 期望 shape | shape 来源 |
|---|---|---|---|
| `model.layers.{i}.self_attn.q_proj.weight` | `blk.{i}.attn_q.weight` | $[H_qD, H]$ | `num_attention_heads`, `head_dim`, `hidden_size` |
| `model.layers.{i}.self_attn.k_proj.weight` | `blk.{i}.attn_k.weight` | $[H_{kv}D, H]$ | `num_key_value_heads`, `head_dim`, `hidden_size` |
| `model.layers.{i}.self_attn.v_proj.weight` | `blk.{i}.attn_v.weight` | $[H_{kv}D, H]$ | `num_key_value_heads`, `head_dim`, `hidden_size` |
| `model.layers.{i}.self_attn.o_proj.weight` | `blk.{i}.attn_o.weight` | $[H, H_qD]$ | `hidden_size`, `num_attention_heads`, `head_dim` |
| `model.layers.{i}.self_attn.q_norm.weight` | `blk.{i}.attn_q_norm.weight` | $[D]$ 或实现约定 shape | 实际 state_dict + attention 实现 |
| `model.layers.{i}.self_attn.k_norm.weight` | `blk.{i}.attn_k_norm.weight` | $[D]$ 或实现约定 shape | 实际 state_dict + attention 实现 |

其中：

- $H_q$ 是 query head 数，即 `num_attention_heads`。
- $H_{kv}$ 是 key/value head 数，即 `num_key_value_heads`。
- $D$ 是 `head_dim`。
- 若 config 未显式给出 `head_dim`，可由 $D = H / H_q$ 推导，但必须检查能整除。

Qwen3 dense 使用 GQA，因此 $H_q$ 和 $H_{kv}$ 可以不同。runtime 中不能假设它们相等。

## A.4 MLP 权重

| HF tensor | Runtime tensor | 期望 shape | shape 来源 |
|---|---|---|---|
| `model.layers.{i}.mlp.gate_proj.weight` | `blk.{i}.ffn_gate.weight` | $[I, H]$ | `intermediate_size`, `hidden_size` |
| `model.layers.{i}.mlp.up_proj.weight` | `blk.{i}.ffn_up.weight` | $[I, H]$ | `intermediate_size`, `hidden_size` |
| `model.layers.{i}.mlp.down_proj.weight` | `blk.{i}.ffn_down.weight` | $[H, I]$ | `hidden_size`, `intermediate_size` |

其中 $I = \texttt{intermediate\_size}$。SwiGLU 的计算形式为：

$$
\operatorname{MLP}(x)
= W_{\text{down}}
\left(
\operatorname{SiLU}(W_{\text{gate}}x)
\odot W_{\text{up}}x
\right)
$$

因此 `gate_proj` 和 `up_proj` 的输出维度都应等于 `intermediate_size`。

## A.5 Layer Norm 权重

| HF tensor | Runtime tensor | 期望 shape | shape 来源 |
|---|---|---|---|
| `model.layers.{i}.input_layernorm.weight` | `blk.{i}.attn_norm.weight` | $[H]$ | `hidden_size` |
| `model.layers.{i}.post_attention_layernorm.weight` | `blk.{i}.ffn_norm.weight` | $[H]$ | `hidden_size` |

RMSNorm 的 epsilon 来自 `rms_norm_eps`，不应写死。

## A.6 Mapping 命令

查看真实模型的 mapping：

```bash
uv run python scripts/inspect_text_path.py \
  --model_name_or_path models/Qwen3-0.6B
```

该命令会列出进入第一阶段 runtime 的 tensor、runtime name、shape，以及 shape 是否与 config 推导一致。
