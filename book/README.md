# 《LLM Inference Quantization from Scratch》教材导读

这套教材围绕一个完整闭环展开：

$$
\text{HF Qwen3} \rightarrow \text{量化} \rightarrow \text{教学版 GGUF}
\rightarrow \text{自定义 Runtime} \rightarrow \text{生成与验证}
$$

它不是论文综述，也不是 API 手册。读者应按顺序阅读章节，并在每章末尾运行对应实验。代码只服务于理解：为什么这些张量存在，为什么这些 shape 必须由 `config.json` 推导，为什么低比特文件不自动等于低延迟推理。

## 阅读路线

1. `00-03`：建立项目边界、模型结构和推理成本模型。
2. `04-07`：掌握量化数学、weight-only 路径、q8_0/q4_0 和误差指标。
3. `08-10`：理解教学版 GGUF、HF 转换器和 runtime 架构。
4. `11-13`：从零实现 forward、sampling，并验证数值正确性。
5. `14-16`：学习 benchmark 方法、现代量化版图和扩展路线。

## 统一约定

- 行内公式使用 `$...$`，独立公式使用 `$$...$$`。
- 所有模型结构参数来自 Hugging Face `config.json` 或 GGUF metadata。
- `Qwen/Qwen3-0.6B` 是贯穿案例，不是硬编码对象。
- q8_0/q4_0 是教学版 weight-only RTN 量化，不是生产级 fused kernel。
