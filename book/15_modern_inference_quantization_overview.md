# 15 Modern Inference Quantization Overview

## 1. 本章目标

前面章节只实现了 f16、q8_0 和 q4_0。这不是因为现代推理量化只需要这三种格式，而是因为它们最适合用来学习“浮点权重如何变成低比特权重，并被 runtime 执行”。

本章把常见推理量化技术放进同一张概念地图。读者应当能够回答：

- 某项技术量化的是权重、activation、KV cache，还是多个对象。
- 它是否需要 calibration data。
- 它是否依赖专用 kernel 或特定 serving runtime。
- 它更适合本地推理、离线压缩，还是线上 serving。
- 它和本项目 q8_0/q4_0 的关系是什么。

本章在完整闭环中的位置是：把教学实现放回现代推理系统的大图中，帮助读者知道下一步该扩展什么，而不是把 q4_0 当成低比特推理的终点。

## 2. 背景与问题

“量化”这个词在不同系统中可能指完全不同的东西：

- 在文件格式中，它可能指权重如何被压缩存储。
- 在算法论文中，它可能指如何选择 scale、rounding 或 channel rotation。
- 在 serving runtime 中，它可能指某个 kernel 是否真正消费 INT4、INT8、FP8。
- 在硬件文档中，它可能指矩阵乘法单元支持哪些 dtype。

如果把这些层次混在一起，就会出现常见误解：模型文件是 INT4，但执行时其实反量化成 fp16；算法论文报告了 W4A16，但你的 runtime 没有对应 kernel；某个框架支持 FP8，但只在特定 GPU 和特定算子上可用。

本章使用三个判断轴：

1. **对象轴**：量化权重、activation、KV cache，还是同时量化。
2. **算法轴**：是否只做 RTN，是否使用 calibration，是否优化 layer/channel/token 误差。
3. **系统轴**：是否需要文件格式、runtime、kernel 和硬件共同支持。

## 3. 数学定义

本项目 q8_0/q4_0 属于 RTN 风格的 weight-only quantization。它主要最小化权重自身的近似误差：

$$
\hat{W}
= \operatorname{dequant}(\operatorname{quant}(W))
$$

$$
\min_{\hat{W}} \|W - \hat{W}\|_2^2
$$

其中：

- $W \in \mathbb{R}^{O \times I}$ 是浮点权重。
- $\hat{W}$ 是反量化后的近似权重。
- $O$ 是输出维度，$I$ 是输入维度，它们由 config 和具体 tensor shape 推导。

校准型方法更关心权重误差传到 layer output 之后的影响：

$$
\min_{\hat{W}}
\|XW^\top - X\hat{W}^\top\|_2^2
$$

其中：

- $X \in \mathbb{R}^{N \times I}$ 是 calibration activation。
- $XW^\top$ 是原始 layer output。
- $X\hat{W}^\top$ 是量化后 layer output。

这解释了为什么 GPTQ、AWQ、AutoRound 等方法通常优于 naive RTN：它们不是只看每个权重值本身，而是看权重误差如何影响真实输入分布下的输出。

activation quantization 和 KV cache quantization 又多一层约束。它们不仅要压缩静态权重，还要处理运行时动态值：

$$
\hat{A}_t
= \operatorname{dequant}(\operatorname{quant}(A_t))
$$

其中 $A_t$ 可能是第 $t$ 个 token 的 activation、key cache 或 value cache。由于 $A_t$ 随输入变化，scale 的选择、更新频率和 outlier 处理会直接影响稳定性。

## 4. 关键推导

低比特推理可以拆成以下链条：

```text
algorithm chooses representation
  -> file stores representation
  -> runtime loads representation
  -> kernel consumes representation
  -> hardware executes representation
```

只要其中一环缺失，理论压缩率就不一定转化为真实速度。

以 INT4 weight-only 为例：

$$
W \rightarrow Q_4(W) \rightarrow \hat{W} \rightarrow x\hat{W}^{\top}
$$

如果 kernel 只接受 fp16，那么 $Q_4(W)$ 必须先反量化成 $\hat{W}$。此时 INT4 只影响存储和加载，不一定减少 matmul 的计算量。

如果 kernel 支持 fused dequant + matmul，则可以更接近：

$$
y
= \operatorname{MatmulWithDequant}(x, Q_4(W), s)
$$

其中 $s$ 是 scale metadata。此时权重读取带宽下降，kernel 在计算中即时反量化，低比特格式才真正进入性能路径。

## 5. 对应到 Qwen3-0.6B

本项目在 `Qwen/Qwen3-0.6B` 上实现的是：

- f16：作为教学 runtime 的高精度导出。
- q8_0：block-wise symmetric INT8 weight-only quantization。
- q4_0：block-wise symmetric signed INT4 weight-only quantization。

它们都保持 activation 为浮点，属于 W8A16/W4A16 风格的教学实现。模型结构仍然来自 `config.json`：层数、hidden size、head 数、KV head 数、MLP 中间维度、RoPE 参数都不写死。

未实现的技术不代表不重要。GPTQ、AWQ、SmoothQuant、FP8、KV cache quantization 和 K-quants 都是实际部署中常见方向；只是它们需要更复杂的 calibration、格式约定、runtime support 或 kernel support。

## 6. 现代方法矩阵

| 技术 | 量化对象 | 常见 bit-width | 是否需要 calibration | 是否需要专用 kernel | 适合场景 | 风险 | 本项目是否实现 |
|---|---|---|---|---|---|---|---|
| RTN | 权重 | INT8/INT4 | 否 | 最好有 | 教学、本地快速压缩 | 低比特误差较大 | 是，q8_0/q4_0 |
| q8_0 | 权重 | INT8 | 否 | 最好有 | 本地推理、格式学习 | 速度依赖 runtime | 是 |
| q4_0 | 权重 | INT4 | 否 | 是 | 内存受限本地推理 | logits 误差明显 | 是 |
| GGUF K-quants | 权重 | 2-6 bit 族 | 否或少量统计 | 是 | llama.cpp 生态 | layout 和 metadata 复杂 | 否 |
| GPTQ | 权重 | INT4/INT3 | 是 | 通常需要 | 本地推理、serving | 校准和实现复杂 | 否 |
| AWQ | 权重/scale | INT4 | 是 | 通常需要 | 本地推理、serving | activation 统计敏感 | 否 |
| AutoRound | 权重 rounding | INT4/INT8 | 是 | 通常需要 | 离线压缩工作流 | 压缩成本高于 RTN | 否 |
| SmoothQuant | 权重 + activation | INT8 | 是 | 是 | W8A8 serving | scale 迁移需谨慎 | 否 |
| INT8 W8A8 | 权重 + activation | INT8 | 通常需要 | 是 | 高吞吐 serving | outlier 处理困难 | 否 |
| FP8 W8A8 | 权重 + activation | FP8 | 通常需要 | 是 | 新 GPU serving | 强硬件依赖 | 否 |
| FP8 KV cache | KV cache | FP8 | 可选 | 是 | 长上下文 serving | attention 精度风险 | 否 |
| INT4 KV cache | KV cache | INT4 | 通常需要 | 是 | 极长上下文 | 累积误差明显 | 否 |
| KIVI-style KV quantization | KV cache | INT2/INT4 | 方法相关 | 是 | 长上下文压缩 | 实现复杂，策略依赖模型 | 否 |
| QuaRot-style rotation | 权重/activation | INT4/INT8 | 通常需要 | 是 | 低比特 serving | rotation 融合复杂 | 否 |
| SpinQuant-style learned rotation | 权重/activation/KV | INT4 等 | 是 | 是 | 高质量低比特 | 需要学习过程和部署融合 | 否 |
| vLLM quantization | runtime 权重/KV | 多种 | 方法相关 | 是 | 在线 serving | 依赖后端支持矩阵 | 否 |
| TensorRT-LLM FP8 / FP4 | 权重/activation | FP8/FP4 | 通常需要 | 是 | NVIDIA serving | 硬件和 engine 绑定强 | 否 |
| LLM Compressor compressed-tensors | 权重格式/压缩流程 | 多种 | 方法相关 | 后端相关 | 模型发布和压缩管线 | 兼容矩阵需逐项确认 | 否 |
| TorchAO | PyTorch 量化栈 | 多种 | 方法相关 | 后端相关 | PyTorch 原生实验 | API 与后端仍在演进 | 否 |

读这张表时，不要只看 bit-width。更重要的是判断：它有没有 calibration 成本；它有没有 runtime/kernel 支持；它的误差发生在权重、activation 还是 KV cache。

## 7. 最小代码实验

本项目可以用同一批 prompt 比较三条路径：

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

这个实验不是 GPTQ/AWQ benchmark，而是给后续方法提供 baseline：任何新增量化方法都应至少和 f16 GGUF、q8_0、q4_0、Transformers reference 做同一协议下的对比。

## 8. 常见误区

**误区一：bit 越低越先进。**  
低 bit-width 只是压缩强度，不代表质量、速度或可部署性一定更好。

**误区二：算法支持等于 runtime 支持。**  
一项量化算法可以生成 INT4 权重，但如果 runtime 没有对应 kernel，执行时仍可能回到浮点路径。

**误区三：校准数据可以随便选。**  
校准型方法优化的是 calibration distribution 下的误差。数据分布偏离目标场景时，指标可能失真。

**误区四：KV cache 量化和权重量化一样简单。**  
KV cache 是运行时动态状态，误差会影响后续所有 attention step。它通常需要额外的稳定性设计。

## 9. 小结

现代推理量化不是单一技术，而是算法、格式、runtime、kernel 和硬件的组合。本项目实现的是最适合从零学习的 RTN weight-only 路线；它足以解释 scale、block、GGUF、runtime 和 logits 误差之间的关系。后续章节讨论如何把这条主线扩展到 GPTQ、AWQ、FP8 和 KV cache quantization。

## 10. 延伸阅读

- 第 04-07 章：从基础量化到误差分析。
- 第 08-10 章：GGUF 子集和 runtime 设计。
- 第 16 章：扩展路线。
- 附录 E：参考资料入口。
