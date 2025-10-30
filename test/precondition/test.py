# -*- coding: utf-8 -*-
"""
Created on 2025/10/16

@author: Yifei Sun
"""
import torch


def func_f(x):
    return torch.cos(x[:, 0]) * torch.sin(x[:, 1])


torch.set_default_dtype(torch.float64)  # 默认双精度
# reproducibility
torch.manual_seed(0)

# 数据：随机生成输入 X
m, n_in, n_out = 1000, 2, 800
X = torch.rand(m, n_in) * 2 - 1  # [-1,1]^2 输入

# 参数
W = torch.rand(n_in, n_out, requires_grad=True)
b = torch.rand(n_out, requires_grad=True)

A = torch.tanh(X @ W + b)  # (m, n_out)
y = func_f(X)
w = torch.linalg.lstsq(A, y).solution

residual = (A @ w - y).norm()
print("{:.4e}".format(residual))

# 优化器
optimizer = torch.optim.Adam([W, b], lr=1, eps=torch.finfo(torch.float64).eps)

# 训练循环
for epoch in range(1):
    optimizer.zero_grad()

    # 前向
    A = torch.tanh(X @ W + b)  # (m, n_out)
    Q, R = torch.linalg.qr(A, mode="reduced")  # 可微分 QR

    diagR = torch.diag(R)
    abs_diag = torch.abs(diagR)

    # 防止 min 太小导致梯度爆炸
    loss = abs_diag.max() / (abs_diag.min())

    # if loss.item() < 1.0 / torch.finfo(loss.dtype).eps:
    #     print("Early stopping at epoch", epoch)
    #     break

    # 反向与优化
    loss.backward()
    optimizer.step()

    if epoch % 1 == 0:
        print(f"Epoch {epoch:03d} | loss = {loss.item():.3e}")
        A = torch.tanh(X @ W + b)  # (m, n_out)
        y = func_f(X)
        w = torch.linalg.lstsq(A, y).solution

        residual = (A @ w - y).norm()
        print("Error : {:.4e}".format(residual))

# 结果
print("\nFinal loss:", loss.item())
print("Final condition ratio:", (abs_diag.max() / abs_diag.min()).item())

A = torch.tanh(X @ W + b)  # (m, n_out)
y = func_f(X)
w = torch.linalg.lstsq(A, y).solution

residual = (A @ w - y).norm()
print("Error : {:.4e}".format(residual))
