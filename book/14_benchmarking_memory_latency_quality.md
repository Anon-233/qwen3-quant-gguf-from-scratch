# 14 Benchmarking Memory, Latency, and Quality

## 1. 本章目标

本章讨论如何给推理量化做可信的 benchmark。到这里，读者已经能够把 Hugging Face 权重转换为教学版 GGUF，并用自定义 runtime 生成文本。接下来的问题不是“能不能跑”，而是“跑出来的数字应当如何解释”。

完成本章后，读者应当能够：

- 区分文件大小、权重显存、运行时峰值显存和 KV cache 显存。
- 区分 prefill、decode、TTFT、TPOT 和平均 tokens/s。
- 解释为什么 q4_0 文件更小，不等价于 Python/PyTorch runtime 一定更快。
- 设计一个不会误导自己的最小 benchmark protocol。
- 把质量指标和速度指标放在同一张表中阅读。

本章在完整闭环中的位置是：把第 03 章的成本模型、第 07 章的误差指标和第 13 章的 correctness validation 合并成一套工程判断方法。

## 2. 背景与问题

量化 benchmark 经常被三类数字混淆：

第一类是**静态存储数字**，例如 `.gguf` 文件大小。它回答“模型放在磁盘上多大”。

第二类是**运行时内存数字**，例如 CUDA peak memory。它回答“执行时需要多少显存”。这个数字不仅包含权重，还包含临时 activation、KV cache、反量化缓存、PyTorch allocator 预留空间等。

第三类是**延迟和吞吐数字**，例如 TTFT、TPOT、tokens/s。它回答“用户等待多久”。这个数字受 kernel、batch、prompt length、decode length、sampling、I/O、warmup 和硬件状态共同影响。

因此，一张负责任的 benchmark 表必须同时说明：

- 硬件：CPU/GPU 型号、显存、驱动、CUDA 后端。
- dtype：权重存储 dtype、compute dtype、KV cache dtype。
- workload：prompt 长度、生成长度、batch size、是否 greedy。
- runtime：本项目 PyTorch runtime、Transformers、llama.cpp、vLLM 或 TensorRT-LLM。
- 是否使用真正低比特 kernel：例如 fused dequant + matmul，还是先反量化再 matmul。

本项目的 benchmark 目标不是证明它比生产 runtime 快，而是帮助读者把“量化后的存储收益、数值误差和实际执行路径”对应起来。

## 3. 数学定义

设一次生成产生 $N_{\text{gen}}$ 个 token，总 decode 耗时为 $t_{\text{decode}}$，平均 decode 吞吐为：

$$
\operatorname{tokens/s}
= \frac{N_{\text{gen}}}{t_{\text{decode}}}
$$

其中：

- $N_{\text{gen}}$ 是实际生成的新 token 数。
- $t_{\text{decode}}$ 是从第一个新 token 之后到生成结束的耗时；不同脚本可能把 prefill 也计入总时间，因此必须明确口径。

TTFT 表示 first token 前的等待时间：

$$
\operatorname{TTFT}
= t_{\text{prefill}} + t_{\text{first-decode}}
$$

其中：

- $t_{\text{prefill}}$ 是处理 prompt 的时间。
- $t_{\text{first-decode}}$ 是完成第一个 decode step 和 sampling 的时间。

TPOT 表示每个输出 token 的平均时间：

$$
\operatorname{TPOT}
= \frac{t_{\text{decode}}}{N_{\text{gen}}}
$$

如果用 $T_{\text{prompt}}$ 表示 prompt token 数，$T_{\text{gen}}$ 表示生成 token 数，decoder-only 模型的 prefill 需要一次性处理长度为 $T_{\text{prompt}}$ 的序列，而 decode 每步通常只输入最新 token，但会访问已有 KV cache。直观上：

$$
t_{\text{prefill}}
\approx f(T_{\text{prompt}}, L, H, H_{\text{kv}}, D_{\text{head}})
$$

$$
t_{\text{decode-step}}
\approx g(T_{\text{past}}, L, H, H_{\text{kv}}, D_{\text{head}})
$$

其中 $L$、$H$、$H_{\text{kv}}$、$D_{\text{head}}$ 均来自 config。这里的 $f$ 和 $g$ 不是闭式公式，而是提醒读者：benchmark 的时间和模型结构、上下文长度、kernel 实现同时相关。

## 4. 关键推导

权重量化首先影响静态权重存储：

$$
M_{\text{weight}}
= N_{\text{param}} \times B_{\text{param}}
$$

其中 $B_{\text{param}}$ 是每个参数平均占用字节数。f16 大约是 2 字节；q8_0 接近 1 字节再加 block scale；q4_0 接近 0.5 字节再加 block scale。

但推理延迟并不只由 $M_{\text{weight}}$ 决定。以本项目的 q4_0 路径为例，权重在文件中以 int4 packing 存储，执行矩阵乘法前会反量化：

$$
\hat{W} = \operatorname{dequant}(Q(W))
$$

$$
\hat{y} = x\hat{W}^{\top}
$$

如果 runtime 没有 fused int4 kernel，那么真实执行仍然是浮点 matmul。此时 q4_0 的主要收益是磁盘文件更小、加载源数据更小；是否更快取决于反量化开销、内存传输、PyTorch kernel 调度和缓存策略。

因此，benchmark 需要把“量化格式”拆成三层：

- 存储层：权重在 GGUF 中如何编码。
- 运行时层：加载后是否保持量化形式，还是反量化为浮点。
- kernel 层：矩阵乘法是否直接消费 int8/int4/FP8 数据。

只有三层同时支持低比特，才能把文件压缩比较稳定地转化为推理加速。

## 5. 对应到 Qwen3-0.6B

本项目已经支持 PyTorch CUDA 后端。当前 API 和 CLI 默认使用 `device="auto"`：当 CUDA 可用时优先使用 CUDA；也可以显式传入 `--device cuda --compute_dtype float16`。

需要强调的是，当前 q8_0/q4_0 CUDA 路径仍然是教学路径：

```text
GGUF quantized tensor
  -> dequantize to fp16/bf16/fp32 tensor
  -> torch matmul / attention / MLP
```

它能显著减少 CPU 等待时间，便于做真实模型验证；但它不是生产级 int8/int4 CUDA kernel benchmark。

在本机一次真实 `Qwen/Qwen3-0.6B` 教学导出中，文件大小如下：

| artifact | size |
|---|---:|
| HF local model directory | 1.5G |
| f16 teaching GGUF | 1.5G |
| q8_0 teaching GGUF | 807M |
| q4_0 teaching GGUF | 449M |

这张表只说明存储层结论：q8_0 和 q4_0 降低了权重文件大小。它不能单独说明 runtime latency。

同一套 CUDA 10-prompt greedy generation suite 的平均速度如下：

| model | avg tok/s |
|---|---:|
| f16 teaching GGUF | 64.85 |
| q8_0 teaching GGUF | 64.92 |
| q4_0 teaching GGUF | 65.26 |

这组结果说明当前 PyTorch CUDA backend 可以工作，并且三种格式在“反量化后浮点执行”的路径上速度接近。它不应被外推到 llama.cpp、vLLM、TensorRT-LLM 或自定义 fused kernel 场景。

质量侧的同一评测见附录 F。简要结论是：q8_0 与 f16 logits 非常接近；q4_0 仍能生成，但 top-k overlap 和 next-token match 明显下降。

## 6. 最小代码实验

运行单模型 benchmark：

```bash
uv run python scripts/benchmark_runtime.py \
  --model outputs/qwen3-0.6b-f16.gguf \
  --tokenizer models/Qwen3-0.6B \
  --device cuda \
  --compute_dtype float16
```

运行包含 Transformers reference 的消融评测：

```bash
uv run python scripts/evaluate_quant_suite.py \
  --prompts data/eval/qwen3_quant_prompts.jsonl \
  --baseline outputs/qwen3-0.6b-f16.gguf \
  --quantized outputs/qwen3-0.6b-q8_0.gguf outputs/qwen3-0.6b-q4_0.gguf \
  --tokenizer models/Qwen3-0.6B \
  --transformers_reference models/Qwen3-0.6B \
  --device cuda \
  --compute_dtype float16
```

建议读者记录四类结果：

- artifact size：磁盘文件大小。
- numeric quality：cosine、MSE、top-k overlap、next-token match。
- generation quality：固定 prompt 的实际输出。
- runtime speed：TTFT、TPOT、tokens/s。

## 7. 常见误区

**误区一：q4_0 文件更小，所以一定更快。**  
不一定。没有 fused low-bit kernel 时，q4_0 可能需要额外反量化，速度由整体执行路径决定。

**误区二：tokens/s 高就代表模型质量好。**  
速度指标和质量指标是两条轴。一个低质量量化版本可以很快，但不一定可用。

**误区三：只报告平均值。**  
平均 tokens/s 可能掩盖长 prompt、短 prompt、不同生成长度之间的差异。至少要说明测试集组成。

**误区四：把本项目 benchmark 当作生产性能结论。**  
本项目是教学 runtime。生产引擎会使用更复杂的内存管理、paged KV cache、kernel fusion、continuous batching 和硬件专用优化。

## 8. 小结

量化 benchmark 的核心不是制造一个漂亮数字，而是回答三个问题：

1. 存储是否下降？
2. logits 和生成质量是否仍然合理？
3. 当前 runtime 是否真的利用了低比特计算？

本项目的答案是：q8_0/q4_0 明显降低文件大小；q8_0 在当前评测中接近 f16；q4_0 更激进，质量风险更高；PyTorch CUDA 后端可用于快速验证，但还不是生产级低比特 kernel。

## 9. 延伸阅读

- 第 03 章：推理成本模型。
- 第 07 章：量化误差分析。
- 第 13 章：正确性验证。
- 附录 F：`Qwen/Qwen3-0.6B` 的本地量化评测案例。
