# 04 Quantization Math Basics：从实数到整数格点

## 本章目标

本章建立量化的基础数学。读完后，读者应能解释 symmetric quantization、affine quantization、scale、zero-point、rounding、clipping、saturation 和 dequantization。

## 背景与问题

神经网络权重通常以 fp16、bf16 或 fp32 存储。量化把连续浮点值映射到有限整数集合。这个过程不可避免地产生误差，核心问题是如何选择 scale 和整数范围，让误差足够小。

## 数学定义

对称量化：

$$
q = \operatorname{clip}
\left(
\operatorname{round}\left(\frac{x}{s}\right),
q_{\min},
q_{\max}
\right)
$$

反量化：

$$
\hat{x} = s \cdot q
$$

其中：

- $x$ 是原始浮点值；
- $s$ 是 scale；
- $q$ 是量化后的整数；
- $\hat{x}$ 是反量化后的近似浮点值；
- $q_{\min}$ 和 $q_{\max}$ 是整数范围。

affine quantization 增加 zero-point：

$$
q = \operatorname{clip}
\left(
\operatorname{round}\left(\frac{x}{s} + z\right),
q_{\min},
q_{\max}
\right)
$$

$$
\hat{x} = s(q - z)
$$

其中 $z$ 是 zero-point。

## 关键推导

对称量化通常选择：

$$
s = \frac{\max_i |x_i|}{q_{\max}}
$$

这样绝对值最大的元素刚好落到整数范围边界。若 $x_i$ 超出 scale 覆盖范围，clip 会造成 saturation error；若 $x_i$ 位于两个整数格点之间，round 会造成 rounding error。

量化误差为：

$$
e_i = x_i - \hat{x}_i
$$

误差来源包括：

1. scale 粗糙；
2. rounding；
3. clipping；
4. block 或 channel 内元素分布差异过大。

## 对应到 Qwen3-0.6B

本项目先在 toy tensor 上实现基础量化，再把同样思想用于 Qwen3 权重。真实转换中，二维权重使用 q8_0 或 q4_0 block quantization，一维 norm 权重保留 f16。

## 最小代码实验

```bash
uv run python examples/01_quantization_math_demo.py
```

对应代码在 `src/miniqwen/quant/int_quant.py`。

## 常见误区

- 认为量化就是简单类型转换，例如 `tensor.to(torch.int8)`。
- 忽略 scale 的存储开销。
- 混淆 fake quantization 与真实低比特存储。
- 混淆 dequantized matmul 与 integer kernel matmul。

## 小结

量化是把连续值投影到整数格点。理解 scale、round、clip 和 dequant，是理解所有后续低比特方法的基础。

## 延伸阅读

继续阅读 `05_weight_only_quantization.md`。
