# -*- coding: utf-8 -*-
"""
Created on 6/13/25

@author: Yifei Sun
"""
"""
Problem 1: Laplace-Beltrami Operator on the Circle S¹ in parametric domain

Consider u(θ) = 12 sin(3θ) as the analytical solution.
- Δ_S¹ u = f
# where f(θ) = 12 * 9 * sin(3θ) is the corresponding source term.


Geometry:
The unit circle S¹ ⊂ ℝ² can be represented using polar angle θ ∈ [0, 2π) as:
    x(θ) = cos θ,
    y(θ) = sin θ.

Differential Operator Expression:
Δ_S¹ u = d²u / dθ²

Boundary Condition:
∫_S¹ u dx = 0

Basis Function Forms:
Parametric domain: u(θ) = ∑ φₘ(θ), where φₘ(θ) are random feature functions.

Solution Approach and Results:

Parametric Domain:
Assume zero mean and periodicity, e.g., u(0) = u(2π) = 0 (adjusted later by integral condition).
Sampling over (0, 2π), similar to solving a 1D Laplace equation.

"""

import pyrfm
import torch
import numpy as np
import time
from matplotlib import pyplot as plt


def u(theta):
    return 12 * torch.sin(3 * theta)


def f(theta):
    return 12 * 9 * torch.sin(3 * theta)


if __name__ == "__main__":
    torch.set_default_device('cuda') if torch.cuda.is_available() else torch.set_default_device('cpu')
    start_time = time.time()
    domain = pyrfm.Line1D(x1=0.0, x2=2 * torch.pi)
    model = pyrfm.RFMBase(dim=1, n_hidden=400, domain=domain, n_subdomains=1, pou=pyrfm.PsiB)
    x_in = domain.in_sample(1000, with_boundary=False)
    x_on = domain.on_sample(2)

    A_in_xx = model.features_second_derivative(x_in, axis1=0, axis2=0).cat(dim=1)
    A_on = model.features(x_on).cat(dim=1)

    A = pyrfm.concat_blocks([[-A_in_xx], [A_on]])

    f_in = f(x_in).view(-1, 1)
    # f_on = torch.zeros_like(x_on).view(-1, 1)  # g(x_on) is zero due to periodicity
    f_on = torch.ones_like(x_on).view(-1, 1)  # test for arbitrary setting

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
        u_vals = model(x_gauss).squeeze()  # shape: (N,)

        # 加权求和得到积分
        integral = torch.sum(w_gauss * u_vals)

        return integral


    # 计算积分常数
    constant = gauss_integral(model, a=0.0, b=2 * np.pi, N=100) / (2 * np.pi)

    x_test = domain.in_sample(100, with_boundary=True)[:40]
    u_test = u(x_test).view(-1, 1)
    u_pred = model(x_test)
    u_pred -= constant

    print((u_test - u_pred).norm() / u_test.norm())

    print('Time:', time.time() - start_time, ", with", torch.tensor(0.).device)

    # Plotting the results
    plt.figure(figsize=(10, 5))
    plt.plot(x_test.cpu().numpy(), u_test.cpu().numpy(), label='Analytical Solution', color='blue')
    plt.plot(x_test.cpu().numpy(), u_pred.cpu().detach().numpy(), label='Predicted Solution', linestyle='--',
             color='orange')
    plt.title('Laplace-Beltrami Operator on the Circle S¹')
    plt.xlabel('θ')
    plt.ylabel('u(θ)')
    plt.legend()
    plt.grid()
    plt.show()
