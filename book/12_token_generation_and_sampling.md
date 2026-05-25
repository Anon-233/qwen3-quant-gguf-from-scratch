# 12 Token Generation and Sampling：logits 如何变成文字

## 本章目标

本章讲解自回归生成循环和 sampling。读完后，读者应能解释 greedy decoding、temperature、top-k、top-p、stop token 和 tokenizer 在生成中的作用。

## 背景与问题

模型 forward 只输出 logits。真正的文本生成还需要把 logits 转为 token id，再把 token id 拼接回上下文，重复执行直到达到停止条件。

## 数学定义

temperature softmax：

$$
p_i =
\frac{\exp(z_i/T)}
{\sum_j \exp(z_j/T)}
$$

其中 $z_i$ 是第 $i$ 个 token 的 logits，$T$ 是 temperature。

greedy decoding：

$$
\text{next} = \arg\max_i z_i
$$

top-p sampling 选择最小集合 $\mathcal{S}_p$，使：

$$
\sum_{i\in \mathcal{S}_p} p_i \ge p
$$

## 关键推导

自回归生成循环：

$$
x_{1:t}
\rightarrow
z_t
\rightarrow
x_{t+1}
\rightarrow
x_{1:t+1}
$$

若使用 KV cache，decode 阶段每一步只输入最新 token，但 attention 仍能访问历史 K/V。

## 对应到 Qwen3-0.6B

本项目使用 Transformers tokenizer。第一阶段不把完整 tokenizer schema 写入 GGUF。GGUF 只保存 tokenizer hint，runtime 由用户传入 tokenizer 路径。

## 最小代码实验

```bash
uv run python scripts/run_gguf.py \
  --model outputs/qwen3-0.6b-q8_0.gguf \
  --tokenizer models/Qwen3-0.6B \
  --prompt "北京是中国的" \
  --temperature 0 \
  --max_new_tokens 16
```

## 常见误区

- 把模型质量问题和 sampling 参数问题混为一谈。
- 比较量化结果时使用随机 sampling，导致不可重复。
- 忘记 stop token。
- 忘记把生成结果从 CUDA tensor 移回 CPU 后再 decode。

## 小结

sampling 是 logits 到文本的接口。正确性评测通常先使用 greedy decoding，以减少随机性。

## 延伸阅读

参见 `src/miniqwen/model/sampling.py`。
