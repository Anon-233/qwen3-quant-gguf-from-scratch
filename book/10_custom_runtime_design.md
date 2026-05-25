# 10 Custom Runtime Design：一个最小推理引擎需要什么

## 本章目标

本章从系统角度拆解自定义 runtime。读完后，读者应能解释 GGUF reader、tensor store、device placement、executor、tokenizer adapter、generation loop 和 KV cache 的职责边界。

## 背景与问题

一个推理 runtime 至少要回答四个问题：

1. 模型结构从哪里来？
2. 权重 tensor 如何加载和解释？
3. forward 如何按 config 执行？
4. token 如何进入模型、logits 如何变回文本？

生产 runtime 还要处理 batching、paged KV cache、kernel fusion、调度、多 GPU 和并发。本项目不实现这些复杂系统，而是保留最小骨架。

## 数学定义

runtime 实现的函数可以写作：

$$
f_{\theta, C}:
\mathbb{N}^{B \times T}
\rightarrow
\mathbb{R}^{B \times T \times V}
$$

其中 $C$ 是 config，$\theta$ 是权重，输入是 token ids，输出是 logits。

## 关键推导

runtime 的核心不应该依赖模型名，而应该依赖 config：

$$
\text{for } \ell = 0,\dots,L-1
$$

其中 $L=\texttt{num\_hidden\_layers}$。attention reshape 使用：

$$
A,\quad A_{\text{kv}},\quad D
$$

而不是某个硬编码宽度。

device placement 是另一个独立维度。当前 API 默认：

```python
runtime = load_gguf_runtime("outputs/qwen3-0.6b-q8_0.gguf")
```

若 CUDA 可用，默认选择 CUDA；否则回退 CPU。显式调试时可以写：

```python
runtime = load_gguf_runtime(
    "outputs/qwen3-0.6b-q8_0.gguf",
    device="cpu",
    compute_dtype="float32",
)
```

## 对应到 Qwen3-0.6B

runtime 加载流程：

1. `GGUFReader` 读取 metadata 与 tensor table；
2. `Qwen3Config.from_metadata()` 恢复 config；
3. `TensorStore` 读取 tensor，q8/q4 先反量化；
4. `RuntimeExecutor` 按 config 执行 forward；
5. `TokenizerAdapter` 使用 HF tokenizer 编码/解码；
6. `generate_tokens()` 执行自回归循环。

需要强调：CUDA 后端是 PyTorch CUDA 后端，不是低比特 fused kernel。q8_0/q4_0 会反量化为 fp16 tensor，再执行普通 PyTorch matmul。

## 最小代码实验

```bash
uv run python scripts/run_gguf.py \
  --model outputs/qwen3-0.6b-q8_0.gguf \
  --tokenizer models/Qwen3-0.6B \
  --prompt "北京是中国的" \
  --max_new_tokens 8 \
  --temperature 0 \
  --device auto
```

使用 `nvidia-smi` 可以看到 CUDA 进程，前提是本机 CUDA 可用。

## 常见误区

- 认为 runtime 只是在读取权重。
- 认为 PyTorch CUDA 后端等价于 int4 CUDA kernel。
- 把 tokenizer、sampling 和 model forward 混成一个不可拆的函数。
- 在 runtime 中硬编码层数或 hidden size。

## 小结

一个最小 runtime 的价值在于暴露推理引擎的骨架。理解这个骨架后，读者才能理解 llama.cpp、vLLM 和 TensorRT-LLM 为什么复杂。

## 延伸阅读

参见 `11_forward_pass_from_scratch.md`。
