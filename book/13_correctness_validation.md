# 13 Correctness Validation：怎样证明 runtime 和量化是可信的

## 本章目标

本章建立正确性验证方法。读完后，读者应能设计分层验证流程，区分 runtime 实现误差、量化误差和端到端偏差。

## 背景与问题

量化模型不追求 bit-exact，但必须数值合理。所谓“合理”不是一句主观判断，而是一组可重复检查：

1. tiny 单元测试通过；
2. f16 GGUF 与 Transformers 对齐；
3. q8/q4 与 f16 GGUF 的差异符合预期；
4. q8/q4 与 Transformers 的端到端偏差可解释；
5. 生成路径无 NaN/Inf，并能完成 smoke test。

## 数学定义

top-k overlap：

$$
\operatorname{overlap}_k =
\frac{
|\operatorname{TopK}(z) \cap \operatorname{TopK}(\hat{z})|
}{k}
$$

next-token match：

$$
\mathbb{1}
\left[
\arg\max_i z_i
=
\arg\max_i \hat{z}_i
\right]
$$

其中 $z$ 是 reference logits，$\hat{z}$ 是被测 logits。

## 关键推导

如果只比较 q4_0 与 Transformers，误差来源是混合的：

$$
\Delta_{\text{total}}
=
\Delta_{\text{runtime}}
+
\Delta_{\text{mapping}}
+
\Delta_{\text{dtype}}
+
\Delta_{\text{quant}}
$$

为了隔离量化误差，需要先证明：

$$
\text{f16 GGUF} \approx \text{Transformers}
$$

然后再比较：

$$
\text{q8/q4 GGUF} - \text{f16 GGUF}
\approx
\Delta_{\text{quant}}
$$

最后再报告：

$$
\text{q8/q4 GGUF} - \text{Transformers}
$$

作为端到端偏差。

## 对应到 Qwen3-0.6B

本项目使用 10 个中文 prompt 做验证，覆盖事实续写、技术解释、简单计算、代码解释、摘要、比较、翻译、结构化输出和开放生成。评测时使用 greedy decoding，避免随机 sampling 干扰。

核心消融结果：

| comparison | avg cosine | avg MSE | avg top-k overlap | next-token match |
|---|---:|---:|---:|---:|
| f16 GGUF vs Transformers | 0.999986 | 0.000071 | 0.980 | 100.00% |
| q8_0 GGUF vs Transformers | 0.999825 | 0.003274 | 0.980 | 100.00% |
| q4_0 GGUF vs Transformers | 0.965154 | 0.701718 | 0.560 | 60.00% |

相对 f16 GGUF 的纯量化视角：

| model | avg cosine | avg MSE | avg top-k overlap | next-token match |
|---|---:|---:|---:|---:|
| q8_0 CUDA fp16 | 0.999826 | 0.003259 | 1.000 | 100.00% |
| q4_0 CUDA fp16 | 0.965181 | 0.700672 | 0.560 | 60.00% |

这说明 f16 GGUF 已经足够接近 Transformers；q8_0 的量化误差很小；q4_0 仍能生成，但 logits 排序变化明显。

## 最小代码实验

默认测试：

```bash
uv run pytest
```

f16 runtime 对齐 Transformers：

```bash
uv run python scripts/compare_with_transformers.py \
  --hf_model models/Qwen3-0.6B \
  --gguf_model outputs/qwen3-0.6b-f16.gguf \
  --prompt "北京是中国的" \
  --device cuda \
  --compute_dtype float16
```

完整消融评测：

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

## 常见误区

- 只看短文本生成，不看 logits。
- 只用 Transformers 作为 baseline，不隔离量化误差。
- f16 GGUF 未对齐 Transformers，就开始解释 q4 误差。
- 把 q4_0 的 smoke test 通过理解为质量无损。

## 小结

正确性验证应分层：先验证 runtime，再验证量化，再报告端到端偏差。

## 延伸阅读

完整实验记录见 `appendices/f_qwen3_quant_eval_results.md`。
