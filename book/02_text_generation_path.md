# 02 Text Generation Path：从 token 到下一个 token

## 本章目标

本章描述最小文本生成路径。读完后，读者应能画出 Qwen3 dense decoder-only 的 forward 数据流，并说明每个模块的输入输出 shape 如何由 config 推导。

## 背景与问题

训练路径包含 loss、optimizer、gradient、checkpoint、distributed training 等大量组件。推理路径更窄：给定 token ids，执行 embedding、若干 decoder block、final norm、lm_head，得到 logits，再 sampling 选出下一个 token。

第一阶段 runtime 只实现这条路径：

$$
\text{tokenizer}
\rightarrow \text{embedding}
\rightarrow L \times \text{decoder block}
\rightarrow \text{final norm}
\rightarrow \text{lm head}
\rightarrow \text{sampling}
$$

## 数学定义

输入 token ids：

$$
X_{\text{id}} \in \mathbb{N}^{B \times T}
$$

embedding 矩阵：

$$
E \in \mathbb{R}^{V \times H}
$$

embedding lookup：

$$
X = E[X_{\text{id}}] \in \mathbb{R}^{B \times T \times H}
$$

第 $\ell$ 层 decoder block：

$$
U_\ell = X_\ell +
\operatorname{Attn}_\ell(\operatorname{RMSNorm}(X_\ell))
$$

$$
X_{\ell+1} = U_\ell +
\operatorname{MLP}_\ell(\operatorname{RMSNorm}(U_\ell))
$$

最终 logits：

$$
Z = \operatorname{RMSNorm}(X_L) W_{\text{lm}}^{\top}
\in \mathbb{R}^{B \times T \times V}
$$

其中 $L$、$H$、$V$ 均来自 config。

## 关键推导

生成时只需要最后一个位置的 logits：

$$
z_{\text{next}} = Z[:, -1, :]
$$

若使用 greedy decoding：

$$
\text{next\_id} = \arg\max_i z_{\text{next}, i}
$$

若使用 sampling，则先把 logits 转为概率分布，再采样。

## 对应到 Qwen3-0.6B

Qwen3-0.6B 的文本路径包含：

- tokenizer；
- `model.embed_tokens.weight`；
- `model.layers.{i}.input_layernorm.weight`；
- `q_proj/k_proj/v_proj/o_proj`；
- 可选 `q_norm/k_norm`；
- `mlp.gate_proj/up_proj/down_proj`；
- `model.norm.weight`；
- `lm_head.weight` 或 tied embedding。

runtime 内部会把这些名称映射为 `token_embd.weight`、`blk.{i}.attn_q.weight` 等更短的教学名称。

## 最小代码实验

运行 tiny forward：

```bash
uv run python examples/04_run_tiny_text_model.py
```

运行真实 GGUF 生成：

```bash
uv run python scripts/run_gguf.py \
  --model outputs/qwen3-0.6b-q8_0.gguf \
  --tokenizer models/Qwen3-0.6B \
  --prompt "北京是中国的" \
  --max_new_tokens 8 \
  --temperature 0 \
  --device auto
```

## 常见误区

- 把 tokenizer 当成模型内部权重的一部分。本项目第一阶段从 HF tokenizer 目录读取 tokenizer。
- 把 Transformers model forward 当成自定义 runtime。Transformers 只能作为 reference。
- 在 decode 阶段忘记 causal mask 或 RoPE position offset。

## 小结

文本生成路径是本项目 runtime 的执行图。后续所有量化、GGUF 和验证章节都围绕这条路径展开。

## 延伸阅读

参见 `11_forward_pass_from_scratch.md`，其中逐模块推导 forward。
