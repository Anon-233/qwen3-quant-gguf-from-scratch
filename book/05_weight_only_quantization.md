# 05 Weight-only Quantization：为什么第一阶段只量化权重

## 本章目标

本章解释 weight-only quantization 的动机、数学形式和工程路径。读完后，读者应能说明 W8A16、W4A16、activation 保持浮点的原因，以及为什么文件变小不等于推理必然变快。

## 背景与问题

完整推理量化可能同时处理权重、activation 和 KV cache。第一阶段若同时实现这些内容，会掩盖主线：如何从 HF 权重导出量化文件，并在自定义 runtime 中完成 forward。

因此本项目先实现 weight-only：

- 权重 $W$ 存为 q8_0 或 q4_0；
- activation $x$ 仍为 fp16/fp32；
- runtime 读取权重后反量化；
- 使用普通 PyTorch matmul 计算。

## 数学定义

原始线性层：

$$
y = xW^\top
$$

其中：

- $x \in \mathbb{R}^{B \times T \times H}$；
- $W \in \mathbb{R}^{O \times H}$；
- $y \in \mathbb{R}^{B \times T \times O}$。

weight-only 量化：

$$
Q(W) \rightarrow \text{integer payload + scale}
$$

$$
\hat{W} = \operatorname{dequant}(Q(W))
$$

$$
\hat{y} = x\hat{W}^{\top}
$$

## 关键推导

输出误差为：

$$
\Delta y = y - \hat{y}
= xW^\top - x\hat{W}^{\top}
= x(W-\hat{W})^\top
$$

这说明权重误差不是孤立存在的。相同的 $W-\hat{W}$，遇到不同 activation $x$，会产生不同输出误差。

这也是 GPTQ、AWQ 等方法引入 calibration activation 的原因：它们不仅看权重本身，还看权重误差如何影响实际输入分布。

## 对应到 Qwen3-0.6B

本项目对以下二维权重做 q8_0/q4_0：

- embedding；
- Q/K/V/O projection；
- MLP gate/up/down；
- lm_head。

对以下一维权重保留 f16：

- RMSNorm；
- Q/K norm。

这样既能显著降低文件大小，又避免对 normalization 参数做不必要的低比特压缩。

## 最小代码实验

```bash
uv run python examples/02_block_q8_q4_demo.py
```

导出真实模型：

```bash
uv run python scripts/quantize_hf_to_gguf.py \
  --model_name_or_path models/Qwen3-0.6B \
  --output outputs/qwen3-0.6b-q8_0.gguf \
  --quant q8_0 \
  --block_size 32
```

## 常见误区

- W4A16 中 activation 不是 int4。
- q4_0 文件小，不代表当前 PyTorch runtime 一定更快。
- 反量化后 matmul 不等于真实 int4 GEMM。
- 只看权重 MSE，忽略 logits 和生成行为。

## 小结

weight-only 是最适合教学的第一步：它保留了 LLM 推理路径的主要结构，又让读者能清楚观察量化误差。

## 延伸阅读

参见 `06_block_quantization_q8_q4.md`。
