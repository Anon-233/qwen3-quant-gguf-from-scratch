# Appendix D Quantization Glossary

本附录给出教材中常用量化术语。术语解释偏向推理 runtime 视角，而不是训练量化或硬件电路视角。

| 术语 | 解释 |
|---|---|
| RTN | round-to-nearest。按 scale 把浮点值除到整数网格上，再四舍五入。q8_0/q4_0 的教学实现属于这一路线。 |
| symmetric quantization | 对称量化。整数范围围绕 0，通常没有 zero-point，适合权重量化教学。 |
| affine quantization | 仿射量化。使用 scale 和 zero-point，使浮点 0 对应某个整数值。activation INT8 常见。 |
| scale | 浮点值和整数格点之间的比例。scale 越大，覆盖范围越大；scale 越小，分辨率越高但更容易 clipping。 |
| zero-point | affine quantization 中的整数零点。symmetric quantization 通常不使用或固定为 0。 |
| clipping | 超出整数表示范围时截断到 $q_{\min}$ 或 $q_{\max}$。 |
| saturation | clipping 造成大量值堆积在边界，通常说明 scale 或 bit-width 不合适。 |
| dequantization | 把整数值和 scale 还原为近似浮点值，例如 $\hat{x}=sq$。 |
| block quantization | 把 tensor 切成 block，每个 block 独立计算 scale。q8_0/q4_0 使用这种方式。 |
| q8_0 | 每个 block 一个 scale，数据为 signed int8 的 weight-only 格式。 |
| q4_0 | 每个 block 一个 scale，数据为 signed int4 并做 packing 的 weight-only 格式。 |
| int4 packing | 把两个 4-bit 值放进一个 byte。读取时需要 unpack 并恢复符号。 |
| W8A16 | 权重 int8，activation float16/bfloat16。 |
| W4A16 | 权重 int4，activation float16/bfloat16。 |
| W8A8 | 权重和 activation 都为 int8 或类似 8-bit 表示。通常需要 calibration 和 kernel 支持。 |
| FP8 | 8-bit 浮点格式族，例如 E4M3、E5M2。真实性能依赖硬件和 runtime。 |
| fake quant | 在浮点 runtime 中模拟 quantize/dequantize，用于研究数值误差，不代表真实低比特加速。 |
| calibration | 使用样本文本或 activation 统计选择 scale、rounding、channel weight 或其他量化参数。 |
| GPTQ | 校准型 weight-only 量化方法，关注 layer output 误差，通常用于 INT4 权重量化。 |
| AWQ | 通过 activation 统计保护重要通道的 weight-only 量化路线。 |
| SmoothQuant | 在 activation 和 weight 之间迁移 scale，常用于 W8A8。 |
| KV cache quantization | 对 decode 阶段缓存的 key/value 进行量化，以降低长上下文显存。 |
| fused dequant + matmul | kernel 在矩阵乘法中直接消费量化权重和 scale，避免显式生成完整反量化权重。 |
| teaching GGUF | 本项目实现的 GGUF 子集，只保证本项目 runtime 可读，不保证 llama.cpp 兼容。 |
