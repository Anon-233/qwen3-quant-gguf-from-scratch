# Appendix E References

本项目不伪造论文、链接或工具支持情况。扩展教材时，应优先引用官方文档、论文原文、arXiv 页面和项目仓库，并在运行环境中核验版本。

## E.1 必读文档类别

- Hugging Face Transformers：用于理解 config、tokenizer、`AutoModelForCausalLM` reference 和 safetensors 加载。
- safetensors：用于理解权重文件的安全读取和 tensor metadata。
- llama.cpp GGUF：用于理解官方 GGUF schema、metadata、tensor naming 和 quantization layout。
- PyTorch：用于理解 tensor dtype、CUDA device、`torch.no_grad()`、`torch.inference_mode()` 和 matmul 行为。
- vLLM：用于理解 serving runtime、paged attention、KV cache 管理和量化后端。
- NVIDIA TensorRT-LLM：用于理解 FP8/FP4、engine build、kernel 和硬件约束。
- PyTorch AO / TorchAO：用于理解 PyTorch 原生量化实验栈。

## E.2 论文与方法关键词

建议读者检索并核验以下关键词：

- GPTQ
- AWQ
- AutoRound
- SmoothQuant
- KIVI KV cache quantization
- QuaRot
- SpinQuant
- FP8 LLM inference
- GGUF K-quants
- compressed-tensors

这些方法在第 15 章和第 16 章中作为概念矩阵和扩展路线出现；本项目第一阶段没有实现它们。

## E.3 引用纪律

写技术教程时应避免三类问题：

- 只记住方法名，却没有确认论文目标、假设和适用范围。
- 引用框架支持情况时不说明版本。
- 把 fake quant、文件压缩和真实 kernel 加速混为一谈。

本项目实测结果见附录 F。实测结果只代表当时的硬件、依赖版本、模型文件和测试协议，不应外推为通用性能结论。
