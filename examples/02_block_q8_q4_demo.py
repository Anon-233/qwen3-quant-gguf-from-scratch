import torch

from miniqwen.quant.block_q4_0 import dequantize_q4_0, quantize_q4_0
from miniqwen.quant.block_q8_0 import dequantize_q8_0, quantize_q8_0
from miniqwen.quant.error_metrics import cosine_similarity, mse

x = torch.randn(3, 17)
q8 = quantize_q8_0(x, block_size=32)
q4 = quantize_q4_0(x, block_size=32)
x8 = dequantize_q8_0(q8)
x4 = dequantize_q4_0(q4)
print({"q8_mse": float(mse(x, x8)), "q8_cos": float(cosine_similarity(x, x8))})
print({"q4_mse": float(mse(x, x4)), "q4_cos": float(cosine_similarity(x, x4))})
