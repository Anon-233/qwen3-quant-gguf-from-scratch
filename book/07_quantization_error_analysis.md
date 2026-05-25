# 07 Quantization Error Analysis：从权重误差到 logits 误差

## 本章目标

本章建立量化误差分析框架。读完后，读者应能使用 MSE、cosine similarity、SQNR、top-k overlap 和 generation smoke test 分析量化版本是否合理。

## 背景与问题

量化后权重变小，但模型行为是否保持，不能只看一个指标。权重 MSE 小，不代表 logits 排序不变；短文本能生成，也不代表质量等价。

因此验证需要分层：

1. 权重级误差；
2. 模块输出误差；
3. logits 分布误差；
4. top-k 排序变化；
5. 生成文本 smoke test；
6. 任务级评测。

## 数学定义

均方误差：

$$
\operatorname{MSE}(x,\hat{x})
=
\frac{1}{n}
\sum_{i=1}^{n}(x_i-\hat{x}_i)^2
$$

cosine similarity：

$$
\cos(x,\hat{x})
=
\frac{x^\top \hat{x}}
{\|x\|_2\|\hat{x}\|_2}
$$

SQNR：

$$
\operatorname{SQNR}_{dB}
=
10\log_{10}
\frac{\mathbb{E}[x^2]}
{\mathbb{E}[(x-\hat{x})^2]}
$$

top-k overlap：

$$
\operatorname{overlap}_k
=
\frac{
|\operatorname{TopK}(z) \cap \operatorname{TopK}(\hat{z})|
}{k}
$$

## 关键推导

weight-only 量化下：

$$
\Delta y = x(W-\hat{W})^\top
$$

若 $\Delta y$ 继续穿过后续 attention、MLP 和 residual block，最终会影响 logits：

$$
\Delta z = z - \hat{z}
$$

生成时真正敏感的是 logits 排序和采样概率，而不只是 logits 的绝对误差。两个 logits 向量可能 MSE 不大，但 top-1 token 改变；也可能 MSE 较大，但 top-k 集合基本保持。

## 对应到 Qwen3-0.6B

本项目的评测分两种 baseline：

- `Transformers`：验证 f16 GGUF runtime 的正确性；
- `f16 GGUF`：隔离 q8_0/q4_0 引入的量化误差。

这两层不能混淆。若 f16 GGUF 尚未对齐 Transformers，则 q8/q4 与 f16 的比较没有充分解释力。

## 最小代码实验

单 prompt 对比：

```bash
uv run python scripts/compare_quantized_gguf.py \
  --baseline outputs/qwen3-0.6b-f16.gguf \
  --quantized outputs/qwen3-0.6b-q8_0.gguf outputs/qwen3-0.6b-q4_0.gguf \
  --tokenizer models/Qwen3-0.6B \
  --prompt "北京是中国的" \
  --device cuda
```

批量消融评测见 `13_correctness_validation.md`。

## 常见误区

- 只看生成文本，不看 logits。
- 只看 MSE，不看 top-k。
- 把 q4 的 smoke test 通过解释为质量无损。
- 把 f16 GGUF baseline 当成天然正确，而不先对齐 Transformers。

## 小结

量化误差需要多视角分析。权重误差回答“压缩得多粗”，logits 误差回答“模型分布是否改变”，生成结果回答“行为是否仍然可用”。

## 延伸阅读

参见 `13_correctness_validation.md` 和 `appendices/f_qwen3_quant_eval_results.md`。
