# 16 Extensions: GPTQ, AWQ, FP8, KV Cache, and Kernels

## 1. 本章目标

本章不是要求读者马上实现所有现代量化方法，而是给出一条不会破坏现有教学闭环的扩展路线。前面章节已经建立了：

```text
HF config + safetensors
  -> tensor mapping
  -> f16/q8_0/q4_0 teaching GGUF
  -> config-driven runtime
  -> logits and generation validation
```

任何扩展都应保留这条主线：结构参数来自 config，权重名称通过 mapping 校验，runtime 不调用 Transformers model forward，validation 能和 f16 GGUF 与 Transformers reference 对齐。

完成本章后，读者应当知道：

- 如何把 RTN q4_0 替换为 GPTQ 或 AWQ 风格的权重量化。
- SmoothQuant 为什么要在权重和 activation 之间迁移 scale。
- FP8 fake quant 如何作为真实 FP8 kernel 前的教学实验。
- KV cache quantization 改动 runtime 的哪些部分。
- 兼容 llama.cpp GGUF 和加入 Triton/CUDA kernel 分别需要补齐什么。

## 2. 背景与问题

本项目的 q8_0/q4_0 是“先把链路跑通”的选择。它有三个优点：

- 不需要 calibration data，容易复现。
- 数学形式简单，便于和代码逐行对应。
- 能清晰展示 GGUF、runtime 和 logits 误差之间的关系。

它也有明显局限：

- q4_0 的误差较大。
- activation 没有量化。
- KV cache 没有量化。
- Python/PyTorch 路径没有 fused low-bit kernel。

扩展项目时，最重要的不是一次性加入很多方法，而是保持接口边界清楚。建议把扩展分为四层：

1. quantization algorithm：如何产生量化权重或 scale。
2. file format：如何把新表示写入 GGUF 或其他格式。
3. runtime semantics：加载后如何恢复或消费该表示。
4. kernel backend：是否有真正低比特计算路径。

## 3. 数学定义

### 3.1 GPTQ 风格目标

GPTQ 类方法关注某一层在 calibration activation 上的输出误差，可抽象为：

$$
\min_{\hat{W}}
\|X(W-\hat{W})^\top\|_2^2
$$

其中：

- $W \in \mathbb{R}^{O \times I}$ 是原始权重。
- $\hat{W}$ 是低比特近似权重。
- $X \in \mathbb{R}^{N \times I}$ 是 calibration activation。
- $O$ 和 $I$ 来自具体 linear tensor shape，而不是硬编码模型常量。

与 RTN 的区别是，RTN 主要近似 $\|W-\hat{W}\|$，GPTQ 更接近近似 layer output 的误差。实现上通常还需要 Hessian 或二阶近似信息，本项目第一阶段不实现。

### 3.2 AWQ 风格 scale search

AWQ 的核心直觉是：并非所有 channel 对输出同等重要。可以通过 per-channel scale 保护重要通道。抽象写法是：

$$
XW^\top
= (XS^{-1})(SW^\top)
$$

其中：

- $S$ 是按通道定义的 scale 矩阵或向量。
- $XS^{-1}$ 调整 activation。
- $SW^\top$ 调整权重。

目标是在量化 $SW^\top$ 时降低重要通道的误差，同时保持整体输出接近原始输出。

### 3.3 SmoothQuant scale migration

SmoothQuant 用类似恒等变换把 activation outlier 的量化难度迁移到权重侧：

$$
XW
= (XS^{-1})(SW)
$$

其中 $S$ 的选择来自 activation 统计。这样做的目的是让 activation 更容易 INT8 量化，但会让权重侧承担相应 scale。

### 3.4 KV cache quantization

对第 $l$ 层、第 $t$ 个 token 的 key/value：

$$
K_{l,t}, V_{l,t}
\in \mathbb{R}^{H_{\text{kv}} \times D_{\text{head}}}
$$

量化后：

$$
\hat{K}_{l,t}
= \operatorname{dequant}(\operatorname{quant}(K_{l,t}))
$$

$$
\hat{V}_{l,t}
= \operatorname{dequant}(\operatorname{quant}(V_{l,t}))
$$

其中 $H_{\text{kv}}$ 和 $D_{\text{head}}$ 来自 config。KV cache 量化直接进入 attention：

$$
\operatorname{Attention}(Q, \hat{K}, \hat{V})
= \operatorname{softmax}
\left(
\frac{Q\hat{K}^{\top}}{\sqrt{D_{\text{head}}}}
\right)\hat{V}
$$

因此 KV 误差会随 decode step 被反复使用，不能只看单个 tensor 的 MSE。

## 4. 关键推导

从本项目出发，最稳妥的扩展顺序是：

1. 先加入 fake quant：仍用 PyTorch 浮点执行，但在权重或 activation 上插入 quant-dequant。
2. 再加入新权重格式：让 GGUF writer/reader 能保存新的 scale、zero-point 或 packing。
3. 再加入 runtime 路径：executor 能识别新 tensor type 并执行。
4. 最后加入 kernel：把 dequant + matmul 融合，减少显存带宽和中间 tensor。

不要先写 kernel。没有第 13 章的 correctness harness，kernel 即使跑得快，也很难判断是否正确。

## 5. 对应到 Qwen3-0.6B

Qwen3 dense decoder-only 的扩展点主要在这些权重：

- attention projections：`q_proj`、`k_proj`、`v_proj`、`o_proj`
- MLP projections：`gate_proj`、`up_proj`、`down_proj`
- embedding 和 lm_head：需要特别处理 tied embedding
- q/k norm 与 RMSNorm：通常保持浮点

实现扩展时，所有 shape 仍应来自 `Qwen3Config`：

- 层数循环使用 `num_hidden_layers`。
- attention head shape 使用 `num_attention_heads`、`num_key_value_heads`、`head_dim`。
- MLP shape 使用 `hidden_size` 和 `intermediate_size`。
- embedding 和 output head 使用 `vocab_size` 和 `hidden_size`。

扩展代码不应把 `Qwen/Qwen3-0.6B` 的具体层数、hidden size 或 head 数写进文件格式、runtime 或测试。

## 6. 扩展路线

### 6.1 GPTQ

GPTQ 扩展需要新增三部分：

- calibration capture：运行少量文本，保存每个 Linear 的输入 activation。
- layer-wise quantizer：按层或按 group 优化 $\hat{W}$。
- validation：比较 GPTQ GGUF、f16 GGUF 和 Transformers reference。

最小工程版本可以先只支持 MLP 和 attention projection 的 weight-only GPTQ，不处理 embedding、norm 或 KV cache。

### 6.2 AWQ

AWQ 扩展可以先实现 scale search 的教学版：

1. 收集每个 Linear 输入 activation 的 channel 统计。
2. 为权重列或输入通道选择 scale。
3. 量化缩放后的权重。
4. 在 runtime 中对 activation 做对应缩放，或把 scale 融合进相邻算子。

关键风险是 scale 的位置。scale 如果只写进文件但 runtime 不使用，就不会产生正确结果。

### 6.3 AutoRound

AutoRound 类方法调整 rounding decision。它可以在本项目中作为 q4_0 的替代 rounding policy：

$$
q_i
= \lfloor x_i / s \rfloor + r_i
$$

其中 $r_i \in \{0,1\}$ 表示向上还是向下取整。教学版可先用小层和少量 calibration 数据验证 rounding 对 logits 的影响。

### 6.4 SmoothQuant 与 W8A8

SmoothQuant 是进入 W8A8 的桥梁。扩展时需要新增：

- activation observer，用于统计 outlier。
- per-channel scale metadata。
- activation quant-dequant 或真实 INT8 activation kernel。
- 对 residual、norm、attention softmax 前后 dtype 的清晰约定。

教学版可以先实现 fake quant：

```text
float activation
  -> quantize to int8 grid
  -> dequantize back to float
  -> existing PyTorch matmul
```

这样可以先研究数值风险，再决定是否写 kernel。

### 6.5 FP8 fake quant

FP8 扩展首先要明确格式，例如 E4M3 或 E5M2。教学 fake quant 可模拟有限指数范围和尾数精度：

$$
\hat{x}
= \operatorname{FP8Dequant}(\operatorname{FP8Quant}(x))
$$

真实 FP8 serving 还依赖硬件和 backend。文档中必须明确 fake quant 不等于真实 FP8 kernel。

### 6.6 KV cache quantization

KV cache quantization 要改动 `kv_cache.py` 和 attention executor。最小路线：

1. 保持 prefill 计算为浮点。
2. 写入 cache 前量化 $K$ 和 $V$。
3. decode 读取 cache 时反量化。
4. 比较长 prompt 下的 logits 和生成稳定性。

如果进一步优化，需要 per-token、per-channel 或分组 scale，以及 fused attention kernel。

### 6.7 llama.cpp GGUF 兼容

当前 GGUF 是教学子集。要提高 llama.cpp 兼容性，需要对齐：

- 官方 metadata schema。
- tokenizer tokens、scores、token type、chat template。
- tensor naming。
- quantization layout 和 alignment。
- rope scaling、special token、模型架构字段。

这不是简单改文件后缀的问题。兼容性应通过真实 llama.cpp 加载测试验证。

### 6.8 Triton/CUDA kernel

kernel 扩展应先从单个算子开始，例如 q4_0 Linear：

```text
input fp16 tile
  x packed int4 weight tile
  + scale
  -> fused dequant + matmul
  -> fp16 output tile
```

验证顺序：

1. 单 tensor dequant 与 PyTorch reference 一致。
2. 单 Linear output 接近 reference。
3. 单 block forward 接近 reference。
4. 全模型 logits 接近 reference。
5. generation smoke test。

## 7. 最小代码实验

最小扩展实验建议从 fake quant 开始，而不是直接改 GGUF：

```python
import torch

def fake_int8_activation(x: torch.Tensor):
    qmax = 127
    scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
    q = torch.round(x / scale).clamp(-128, 127)
    return q * scale
```

把它插入 tiny model 的 MLP 输入处，比较 logits cosine 和 top-k overlap。如果 tiny test 都不稳定，就不应直接推进真实模型导出。

## 8. 常见误区

**误区一：扩展格式就等于扩展 runtime。**  
文件能写出新字段，只说明存储层完成。runtime 必须知道如何解释这些字段。

**误区二：fake quant 等于真实加速。**  
fake quant 用于研究数值误差，不提供低比特 kernel 的性能结论。

**误区三：只看单层误差。**  
单层误差合理，不代表生成路径稳定。最终仍要跑 logits、top-k 和 generation。

**误区四：为了某个模型写死 shape。**  
扩展方法更容易诱导硬编码，例如 group 数、head_dim、KV shape。所有这些都必须由 config 或 tensor shape 推导。

## 9. 小结

本项目的扩展原则是：先保持 correctness，再增加格式，再优化 runtime，最后追求 kernel 性能。GPTQ、AWQ、SmoothQuant、FP8 和 KV cache quantization 都可以沿着同一条 config-driven 主线加入；真正困难的部分不是公式，而是让算法、格式、runtime、kernel 和验证协议同时对齐。

## 10. 延伸阅读

- 第 13 章：建立扩展前必须保留的 validation harness。
- 第 15 章：现代推理量化技术矩阵。
- 附录 A：tensor name mapping。
- 附录 B：教学版 GGUF metadata。
- 附录 E：参考资料入口。扩展论文和官方文档链接应在实际实现前逐项核验。
