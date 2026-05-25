# 11 Forward Pass from Scratch：逐层实现 Qwen3 文本模型

## 本章目标

本章推导 Qwen3 dense decoder-only 的 forward。读完后，读者应能从输入 token ids 写出 embedding、RMSNorm、Q/K norm、RoPE、GQA attention、causal mask、SwiGLU MLP、residual 和 lm_head 的完整数据流。

## 背景与问题

自定义 runtime 最容易出错的地方不是矩阵乘法，而是 shape。尤其在 Qwen3-0.6B 中，`num_attention_heads * head_dim` 不等于 `hidden_size`。这要求实现必须忠实使用 config。

## 数学定义

RMSNorm：

$$
\operatorname{RMSNorm}(x)
=
\frac{x}
{\sqrt{\frac{1}{H}\sum_{j=1}^{H}x_j^2+\epsilon}}
\odot w
$$

Q/K/V projection：

$$
Q = XW_q^\top,\qquad
K = XW_k^\top,\qquad
V = XW_v^\top
$$

reshape：

$$
Q \in \mathbb{R}^{B \times T \times A \times D}
$$

$$
K,V \in \mathbb{R}^{B \times T \times A_{\text{kv}} \times D}
$$

attention：

$$
\operatorname{Attn}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{D}} + M
\right)V
$$

其中 $M$ 是 causal mask。

SwiGLU：

$$
\operatorname{MLP}(x)
=
\left(\operatorname{SiLU}(xW_g^\top)\odot xW_u^\top\right)
W_d^\top
$$

## 关键推导

GQA 中 Q head 数与 KV head 数不同。令：

$$
r = \frac{A}{A_{\text{kv}}}
$$

KV 需要按 head 维重复 $r$ 次，使其能与 Q 对齐。重复后：

$$
K_{\text{rep}},V_{\text{rep}}
\in
\mathbb{R}^{B \times T \times A \times D}
$$

Qwen/Llama 风格 RoPE 使用 half-split rotation：

$$
\operatorname{rotate}(x)
=
[-x_{D/2:D},\; x_{0:D/2}]
$$

$$
\operatorname{RoPE}(x)
=
x \odot \cos(\theta)
+
\operatorname{rotate}(x)\odot \sin(\theta)
$$

## 对应到 Qwen3-0.6B

Qwen3-0.6B 的 Q projection shape 为：

$$
(16 \times 128) \times 1024 = 2048 \times 1024
$$

O projection shape 为：

$$
1024 \times 2048
$$

这说明 attention 输出在投影前的最后一维是 $A D=2048$，不能写成 `hidden_size`。

## 最小代码实验

```bash
uv run pytest tests/test_attention_shapes.py tests/test_qwen3_head_dim_shapes.py
```

查看实现：`src/miniqwen/model/attention.py`、`rope.py`、`mlp.py`、`text_model.py`。

## 常见误区

- 忽略 Q/K norm。
- RoPE 使用错误的 rotate 布局。
- 把 $A D$ 当成 $H$。
- KV cache decode 时忘记 position offset。
- 输出头 tied embedding 未与 config 和权重共同校验。

## 小结

forward pass 是本项目最核心的工程章节。只有这里的 shape 和数学正确，量化误差才有意义。

## 延伸阅读

参见 `12_token_generation_and_sampling.md`。
