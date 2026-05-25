# 06 Block Quantization q8_0/q4_0：局部 scale 与低比特存储

## 本章目标

本章讲解本项目实现的 q8_0 和 q4_0。读完后，读者应能解释 block size、per-block scale、signed int4 范围、packing、metadata 开销和 block size 对误差的影响。

## 背景与问题

如果整张量只用一个 scale，少数 outlier 会迫使 scale 变大，导致大多数普通值落在很粗的整数格点上。block quantization 把张量切成多个 block，每个 block 独立选择 scale，从而降低局部误差。

## 数学定义

对 block $b$：

$$
s_b = \frac{\max_{i \in b}|x_i|}{q_{\max}}
$$

$$
q_i = \operatorname{clip}
\left(
\operatorname{round}\left(\frac{x_i}{s_b}\right),
q_{\min},
q_{\max}
\right)
$$

$$
\hat{x}_i = s_b q_i
$$

其中：

- $s_b$ 是 block $b$ 的 scale；
- $q_i$ 是 block 内第 $i$ 个整数值；
- $\hat{x}_i$ 是反量化近似。

## 关键推导

q8_0 使用 signed int8，整数范围为：

$$
q \in [-128, 127]
$$

scale 通常使用 $q_{\max}=127$。q4_0 使用 signed int4：

$$
q \in [-8, 7]
$$

由于 int4 不是常规 PyTorch dtype，本项目把两个 signed int4 打包到一个 byte：

$$
\text{byte} = (q_{\text{high}} \& 0x0F) \ll 4
\;|\;
(q_{\text{low}} \& 0x0F)
$$

每个 block 的存储开销为：

$$
M_b = M_{\text{payload}} + M_{\text{scale}}
$$

block size 越小，scale 越贴近局部数据，但 scale metadata 越多。

## 对应到 Qwen3-0.6B

导出 q8_0：

```bash
uv run python scripts/quantize_hf_to_gguf.py \
  --model_name_or_path models/Qwen3-0.6B \
  --output outputs/qwen3-0.6b-q8_0.gguf \
  --quant q8_0 \
  --block_size 32
```

导出 q4_0：

```bash
uv run python scripts/quantize_hf_to_gguf.py \
  --model_name_or_path models/Qwen3-0.6B \
  --output outputs/qwen3-0.6b-q4_0.gguf \
  --quant q4_0 \
  --block_size 32
```

## 最小代码实验

```bash
uv run python examples/02_block_q8_q4_demo.py
```

该 demo 会输出 q8 和 q4 的 MSE 与 cosine similarity。

## 常见误区

- int4 packing 只是存储格式，不是计算 kernel。
- signed int4 与 uint4 的 zero-point 语义不同。
- block size 越小不一定越好，因为 scale metadata 会增加。
- q4_0 不是 GGUF K-quants。

## 小结

q8_0/q4_0 是理解 GGUF 本地推理生态的入门量化格式。它们简单、直观，适合教学，但不是现代低比特量化的终点。

## 延伸阅读

参见 `src/miniqwen/quant/block_q8_0.py` 和 `src/miniqwen/quant/block_q4_0.py`。
