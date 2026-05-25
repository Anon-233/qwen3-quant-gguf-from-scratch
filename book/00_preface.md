# 00 Preface：为什么从推理量化开始

## 本章目标

本章回答三个问题：本项目要教什么、不教什么，以及为什么必须以 config-driven implementation 作为第一原则。读完本章后，读者应能解释本书的完整闭环：

$$
\text{Hugging Face 权重}
\rightarrow \text{教学量化}
\rightarrow \text{教学版 GGUF}
\rightarrow \text{自定义 Runtime}
\rightarrow \text{文本生成与验证}
$$

## 背景与问题

LLM 推理量化常被误解为“把 fp16 改成 int4”。这句话只描述了存储形式，没有说明模型结构、权重命名、文件格式、runtime 数据流、数值误差和硬件 kernel 之间的关系。

本项目选择 `Qwen/Qwen3-0.6B` 作为主线模型，因为它足够小，可以在教学环境中下载、转换和运行；同时它又包含现代 decoder-only LLM 的关键部件：RMSNorm、RoPE、GQA attention、SwiGLU MLP、tied embedding 和自回归 sampling。

本项目不是：

- llama.cpp 的替代品；
- vLLM 或 TensorRT-LLM 的替代品；
- 完整 GGUF 官方兼容实现；
- GPTQ、AWQ、FP8 或 KV cache 量化的生产实现；
- 训练框架或多模态框架。

本项目是：

- 一本解释推理量化链路的中文教材；
- 一个可以运行的最小工程；
- 一个把数学公式、权重格式和 runtime forward 对齐起来的实验平台。

## 数学定义

量化最直接影响的是权重存储。若模型共有 $N_{\text{param}}$ 个参数，每个参数占 $B_{\text{param}}$ 字节，则权重存储近似为：

$$
M_{\text{weight}} = N_{\text{param}} \times B_{\text{param}}
$$

其中：

- $M_{\text{weight}}$ 是权重占用的字节数；
- $N_{\text{param}}$ 是参数总量；
- $B_{\text{param}}$ 是单个参数的字节数。

从 fp16 到 int8，$B_{\text{param}}$ 约从 2 降到 1；从 fp16 到 int4，$B_{\text{param}}$ 约从 2 降到 0.5。但这只解释文件和权重内存，不解释端到端速度。

## 关键推导

推理速度由多个因素共同决定：

$$
t_{\text{decode}}
\approx
t_{\text{load weight}}
+ t_{\text{matmul}}
+ t_{\text{attention}}
+ t_{\text{sampling}}
+ t_{\text{runtime overhead}}
$$

权重量化主要降低 $t_{\text{load weight}}$ 和权重常驻内存。若 runtime 仍然执行

$$
\hat{W} = \operatorname{dequant}(Q(W)), \qquad y = x\hat{W}^{\top}
$$

则计算阶段仍是浮点矩阵乘法。真正的 int4/int8 加速需要 fused dequant + GEMM kernel，或者硬件原生支持的低精度矩阵指令。

## 对应到 Qwen3-0.6B

本项目可以针对 Qwen3 dense decoder-only 架构编写代码，但不能把 Qwen3-0.6B 的结构数值写死。以下字段必须从 `config.json` 读取或推导：

- `vocab_size`
- `hidden_size`
- `intermediate_size`
- `num_hidden_layers`
- `num_attention_heads`
- `num_key_value_heads`
- `head_dim`
- `max_position_embeddings`
- `rms_norm_eps`
- `rope_theta`
- `tie_word_embeddings`

原因很简单：模型名不是模型结构。Qwen3 系列中不同 dense 模型可能共享命名风格，但层数、head 数、KV head 数、head_dim 和 MLP 维度都可能不同。硬编码这些值会让转换器和 runtime 在迁移到其他小模型时产生静默错误。

## 最小代码实验

先运行默认 tiny 测试，确认本地环境可以执行：

```bash
uv run pytest
```

再运行最小量化 demo：

```bash
uv run python examples/01_quantization_math_demo.py
```

这两个命令不需要下载真实模型。

## 常见误区

- 文件更小不等于推理一定更快。
- 教学版 GGUF 不等于完整 llama.cpp GGUF。
- q4_0 能生成文本不等于质量与 f16 等价。
- Transformers 可以作为 reference，但不能替代自定义 runtime。

## 小结

本书的主线不是“堆功能”，而是把五件事连起来：模型结构、量化数学、权重格式、runtime forward 和正确性验证。

## 延伸阅读

继续阅读 `01_qwen3_model_inspection.md`。在理解模型结构之前，不应该开始写量化代码。
