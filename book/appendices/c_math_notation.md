# Appendix C Math Notation

本附录汇总教材中的常用符号。所有与模型结构有关的符号都应由 Hugging Face config 或 GGUF metadata 推导，而不是由代码写死。

## C.1 模型结构

| 符号 | 含义 | 来源 |
|---|---|---|
| $B$ | batch size | runtime 输入 |
| $T$ | sequence length | tokenizer 输出或 decode 状态 |
| $V$ | vocab size | `vocab_size` |
| $H$ | hidden size | `hidden_size` |
| $I$ | MLP intermediate size | `intermediate_size` |
| $L$ | decoder layer count | `num_hidden_layers` |
| $H_q$ | query attention head count | `num_attention_heads` |
| $H_{\text{kv}}$ | key/value head count | `num_key_value_heads` |
| $D_{\text{head}}$ | head dimension | `head_dim` 或 $H/H_q$ |
| $T_{\max}$ | context length | `max_position_embeddings` |

## C.2 Tensor 与 forward

| 符号 | 含义 |
|---|---|
| $X$ | activation 或 calibration 输入 |
| $x_t$ | 第 $t$ 个 token 的 hidden state |
| $W$ | 浮点权重 |
| $\hat{W}$ | 反量化后的近似权重 |
| $Q,K,V$ | attention 中的 query、key、value |
| $K_{\text{cache}}, V_{\text{cache}}$ | KV cache |
| $z$ | logits |
| $p$ | sampling probability |

## C.3 量化

| 符号 | 含义 |
|---|---|
| $s$ | scale |
| $s_b$ | 第 $b$ 个 block 的 scale |
| $q$ | 量化后的整数值 |
| $q_{\min}, q_{\max}$ | 整数表示范围 |
| $\hat{x}$ | 反量化后的近似值 |
| $\Delta x$ | 量化误差，$\Delta x = x - \hat{x}$ |

基础 symmetric quantization 写作：

$$
q
= \operatorname{clip}
\left(
\operatorname{round}\left(\frac{x}{s}\right),
q_{\min},
q_{\max}
\right)
$$

$$
\hat{x}=sq
$$

## C.4 成本与评测

| 符号 | 含义 |
|---|---|
| $M_{\text{weight}}$ | 权重内存或存储 |
| $M_{\text{KV}}$ | KV cache 内存 |
| $B_{\text{param}}$ | 每个参数平均字节数 |
| $B_{\text{elem}}$ | cache 或 activation 每元素字节数 |
| $\operatorname{MSE}$ | 均方误差 |
| $\cos(x,\hat{x})$ | 余弦相似度 |
| $\operatorname{TTFT}$ | time to first token |
| $\operatorname{TPOT}$ | time per output token |
