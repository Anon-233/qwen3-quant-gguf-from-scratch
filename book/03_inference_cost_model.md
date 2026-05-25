# 03 Inference Cost Model：显存、带宽与延迟

## 本章目标

本章建立推理成本模型，解释为什么量化能降低权重存储，但不必然降低端到端延迟。读完后，读者应能区分权重显存、KV cache 显存、prefill、decode、TTFT 和 TPOT。

## 背景与问题

LLM 推理不是一次矩阵乘法。实际系统要经历 prompt prefill、逐 token decode、KV cache 读写、sampling、调度和数据搬运。量化只影响其中一部分。

因此，当有人说“q4 比 fp16 快 4 倍”时，应先问：

- 使用了什么 kernel？
- 权重是否以 int4 形式参与 GEMM？
- batch size 和上下文长度是多少？
- 测的是 TTFT、TPOT 还是平均 tokens/s？
- 是否包含 tokenizer 和采样开销？

## 数学定义

权重存储：

$$
M_{\text{weight}} = N_{\text{param}} \times B_{\text{param}}
$$

KV cache 存储：

$$
M_{\text{KV}} =
2 \times L \times H_{\text{kv}} \times D_{\text{head}}
\times T \times B \times B_{\text{elem}}
$$

其中：

- $L$ 是层数；
- $H_{\text{kv}}$ 是 KV head 数；
- $D_{\text{head}}$ 是每个 head 的维度；
- $T$ 是上下文长度；
- $B$ 是 batch size；
- $B_{\text{elem}}$ 是 KV cache 单元素字节数；
- 前面的 2 表示 K 和 V 两份缓存。

这些量中的 $L$、$H_{\text{kv}}$、$D_{\text{head}}$ 必须来自 config。

## 关键推导

prefill 阶段一次处理 prompt 中的全部 token。attention score 的形状近似为：

$$
\mathbb{R}^{B \times A \times T \times T}
$$

因此 prefill attention 的计算量随 $T^2$ 增长。

decode 阶段每次只处理一个新 token，但要读取历史 KV cache：

$$
\mathbb{R}^{B \times A \times 1 \times T}
$$

所以 decode 的单步成本随历史长度 $T$ 增长。

TTFT 与 TPOT：

$$
\text{TTFT} = t_{\text{prefill}} + t_{\text{first decode}}
$$

$$
\text{TPOT} =
\frac{t_{\text{decode total}}}{N_{\text{generated}}}
$$

## 对应到 Qwen3-0.6B

Qwen3-0.6B 的 $L=28$、$H_{\text{kv}}=8$、$D_{\text{head}}=128$。当 batch size 为 1、上下文长度为 4096、KV cache 使用 fp16 时：

$$
M_{\text{KV}}
= 2 \times 28 \times 8 \times 128 \times 4096 \times 1 \times 2
$$

这个数量级已经不可忽略。权重量化不能消除 KV cache 成本。

## 最小代码实验

使用代码估算 KV cache：

```python
from miniqwen.config import Qwen3Config
from miniqwen.evaluation.memory import estimate_kv_cache_bytes

config = Qwen3Config.from_pretrained("models/Qwen3-0.6B")
print(estimate_kv_cache_bytes(config, tokens=4096))
```

运行 CUDA benchmark：

```bash
uv run python scripts/benchmark_runtime.py \
  --model outputs/qwen3-0.6b-q8_0.gguf \
  --tokenizer models/Qwen3-0.6B \
  --device cuda \
  --compute_dtype float16
```

## 常见误区

- 用平均 tokens/s 同时解释 prefill 和 decode。
- 忽略 KV cache 显存。
- 把 Python 教学 runtime 的速度外推到 production runtime。
- 忽略 batch size、prompt length 和 decode length。

## 小结

量化首先是内存和带宽优化；速度收益取决于 kernel、runtime、硬件和 workload。

## 延伸阅读

参见 `14_benchmarking_memory_latency_quality.md`。
