import torch

from miniqwen.quant.int_quant import dequantize_symmetric, symmetric_quantize

x = torch.tensor([-1.0, -0.2, 0.0, 0.3, 0.9])
q, scale, qmin, qmax = symmetric_quantize(x, num_bits=8)
x_hat = dequantize_symmetric(q, scale)
print({"x": x.tolist(), "q": q.tolist(), "scale": float(scale), "range": [qmin, qmax]})
print({"dequant": x_hat.tolist(), "max_abs_error": float((x - x_hat).abs().max())})
