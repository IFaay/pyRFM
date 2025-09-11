# -*- coding: utf-8 -*-
"""
Created on 6/14/25

@author: Yifei Sun
"""
"""
Problem 2: Laplace-Beltrami Operator on the Circle S¹ in natural domain

Consider u(θ) = 12 sin(3θ) as the analytical solution.

Geometry:
Alternatively, it can be defined implicitly by:
    φ(x, y) = x² + y² − 1

Δ_S¹ u = (1 − x²) ∂²u/∂x² + (1 − y²) ∂²u/∂y² − 2xy ∂²u/∂x∂y − x ∂u/∂x − y ∂u/∂y

Boundary Condition:
∫_S¹ u dx = 0

Basis Function Forms:
Natural domain: u(x, y) = ∑ φₘ(x, y), where φₘ(x, y) are random feature functions.

Solution Approach and Results:
Natural Domain:
Assume u(1, 0) = 0 initially, then adjust by the integral condition.
Sample and assemble the matrix over the circle S¹.
"""

import pyrfm
import torch
import numpy as np
import time
from matplotlib import pyplot as plt


def u(x):
    theta = torch.atan2(x[:, 1], x[:, 0])
    return 12 * torch.sin(3 * theta)


def f(x):
    theta = torch.atan2(x[:, 1], x[:, 0])
    return 12 * 9 * torch.sin(3 * theta)


if __name__ == "__main__":
    torch.set_default_device('cuda') if torch.cuda.is_available() else torch.set_default_device('cpu')
    start_time = time.time()
    domain = pyrfm.Circle2D(center=(0.0, 0.0), radius=1.0)
    model = pyrfm.RFMBase(dim=2, n_hidden=400, domain=domain, n_subdomains=1, pou=pyrfm.PsiB)
    x_in = domain.on_sample(1000)
    x_on = torch.tensor([[1.0, 0.0]])

    # Δ_S¹ u = (1 − x²) ∂²u/∂x² + (1 − y²) ∂²u/∂y² − 2xy ∂²u/∂x∂y − x ∂u/∂x − y ∂u/∂y
    A_in_xx = model.features_second_derivative(x_in, axis1=0, axis2=0).cat(dim=1)
    A_in_yy = model.features_second_derivative(x_in, axis1=1, axis2=1).cat(dim=1)
    A_in_xy = model.features_second_derivative(x_in, axis1=0, axis2=1).cat(dim=1)
    A_in_x = model.features_derivative(x_in, axis=0).cat(dim=1)
    A_in_y = model.features_derivative(x_in, axis=1).cat(dim=1)

    x = x_in[:, [0]]
    y = x_in[:, [1]]
    A_in = (1 - x ** 2) * A_in_xx + (1 - y ** 2) * A_in_yy - 2 * x * y * A_in_xy - x * A_in_x - y * A_in_y
    A_on = model.features(x_on).cat(dim=1)

    f_in = f(x_in).view(-1, 1)
    f_on = torch.tensor([[1.0]])  # assume u(1, 0) = 1 initially, then adjust by the integral condition

    A = pyrfm.concat_blocks([[-A_in], [A_on]])
    f = pyrfm.concat_blocks([[f_in], [f_on]])

    model.compute(A).solve(f)


    def gauss_integral(model, a=0.0, b=2 * np.pi, N=100):
        """
        用 Gauss-Legendre 方法计算 ∫_a^b model(x) dx 的近似值

        参数:
            model: 可接受 (N, 1) 输入并输出 (N, 1) 或 (N,) 的 PyTorch 模型
            a, b: 积分区间端点，默认是 [0, 2π]
            N: Gauss 点数（越大越精确）

        返回:
            积分常数：∫_a^b model(x) dx 的近似值（标量张量）
        """
        # NumPy: 在 [-1, 1] 上获取 Gauss 节点和权重
        xi, wi = np.polynomial.legendre.leggauss(N)

        # 映射到 [a, b]
        x_gauss_np = 0.5 * (b - a) * xi + 0.5 * (b + a)
        w_gauss_np = 0.5 * (b - a) * wi

        # 转为 torch 张量
        x_gauss = torch.tensor(x_gauss_np, dtype=torch.float64).unsqueeze(1)  # (N, 1)
        w_gauss = torch.tensor(w_gauss_np, dtype=torch.float64)  # (N,)

        # 评估模型
        # 转为自然域输入
        x_gauss = torch.cat([torch.cos(x_gauss), torch.sin(x_gauss)], dim=1)
        u_vals = model(x_gauss).squeeze()  # shape: (N,)

        # 加权求和得到积分
        integral = torch.sum(w_gauss * u_vals)

        return integral


    constant = gauss_integral(model, a=0.0, b=2 * np.pi, N=100) / (2 * np.pi)
    x_test = domain.on_sample(1000)
    u_test = u(x_test).view(-1, 1)
    u_pred = model(x_test).view(-1, 1)
    u_pred -= constant

    print((u_test - u_pred).norm() / u_test.norm())

    print('Time:', time.time() - start_time, ", with", torch.tensor(0.).device)

    # Plotting the results
    plt.figure(figsize=(8, 6))
    plt.scatter(x_test[:, 0].cpu(), x_test[:, 1].cpu(), c=u_pred.cpu().squeeze(), cmap='viridis', marker='o', s=10)
    plt.colorbar(label='Predicted u')
    plt.title('Predicted Solution on the Circle S¹')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.show()
