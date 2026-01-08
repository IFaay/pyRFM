import torch

# params 替换为 10,12,14,16
params = torch.tensor([10, 12, 14, 16], dtype=torch.float64)

# 对应你的误差
errors = torch.tensor([
    5.2278e-04,
    3.0853e-04,
    2.6843e-04,
    1.2579e-04
], dtype=torch.float64)

# 逐段 order
for i in range(len(errors) - 1):
    p = torch.log(errors[i] / errors[i + 1]) / torch.log(params[i + 1] / params[i])
    print(f"params = {int(params[i])} -> {int(params[i + 1])}, order ≈ {p.item():.4f}")

# global
p_global = torch.log(errors[0] / errors[-1]) / torch.log(params[-1] / params[0])
print(f"\nGlobal order (overall): ≈ {p_global.item():.4f}")
