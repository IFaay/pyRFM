# -*- coding: utf-8 -*-
"""
Created on 6/13/25

@author: Yifei Sun
"""
"""
Problem 3: Laplace-Beltrami Operator on the Sphere S²

Consider using u(θ, φ) = sin²(θ) · cos(2φ) as the analytical solution.
Δ_{S²}(sin²θ · cos(2φ)) = -6 · sin²θ · cos(2φ)

Geometry:
The unit sphere S² ⊂ ℝ³ can be represented in spherical coordinates (θ, φ) as:
x(θ, φ) = sinθ · cosφ,
y(θ, φ) = sinθ · sinφ,
z(θ) = cosθ,

θ ∈ [0, π] (polar angle), φ ∈ [0, 2π] (azimuthal angle).


Differential Operator Expressions:

Δₛ² u = (1/sinθ) ∂/∂θ (sinθ ∂u/∂θ) + (1/sin²θ) ∂²u/∂φ²
      =  cotθ ∂u/∂θ + ∂²u/∂θ² + (1/sin²θ) ∂²u/∂φ²

Boundary Condition:

∫_{S²} u d𝑥 = 0

Basis Function Representation:

Parametric domain: u(θ, φ) = ∑ φₘ(θ, φ), where φₘ(θ, φ) are stochastic basis functions.

Solution Approach and Result:

Parametric Domain:
Under the zero-mean and periodic boundary conditions, assume:
u(θ, 0) = u(θ, 2π), u(π / 2, 0) = 1.0.

(After solving, adjust to satisfy the integral condition.)
"""

import pyrfm
import torch
import numpy as np
import time
from matplotlib import pyplot as plt


def u(theta, phi):
    return torch.sin(theta) ** 2 * torch.cos(2 * phi)


def f(theta, phi):
    return 6 * torch.sin(theta) ** 2 * torch.cos(2 * phi)


if __name__ == "__main__":
    torch.set_default_device('cuda') if torch.cuda.is_available() else torch.set_default_device('cpu')
    start_time = time.time()
    domain = pyrfm.Square2D(center=(torch.pi / 2.0, torch.pi), radius=(torch.pi / 2.0, torch.pi))
    model = pyrfm.RFMBase(dim=2, n_hidden=400, domain=domain, n_subdomains=1, pou=pyrfm.PsiB)
    x_in = domain.in_sample(1600, with_boundary=False)

    A_in_x = model.features_derivative(x_in, axis=0).cat(dim=1)  # ∂u/∂θ
    A_in_xx = model.features_second_derivative(x_in, axis1=0, axis2=0).cat(dim=1)  # ∂²u/∂θ²
    A_in_yy = model.features_second_derivative(x_in, axis1=1, axis2=1).cat(dim=1)  # ∂²u/∂φ²
    theta = x_in[:, 0].view(-1, 1)  # θ

    A_in = 1 / torch.tan(theta) * A_in_x + A_in_xx + 1 / torch.sin(theta) ** 2 * A_in_yy

    # periodic boundary conditions
    # u(θ, 0) = u(θ, 2π), θ ∈ [0, π]
    theta = torch.linspace(0, torch.pi, 400)
    x_on_0 = torch.stack((theta, torch.zeros_like(theta)), dim=1)  # φ = 0
    x_on_2pi = torch.stack((theta, 2 * np.pi * torch.ones_like(theta)), dim=1)  # φ = 2π

    A_on_0 = model.features(x_on_0).cat(dim=1)  # u(θ, 0)
    A_on_2pi = model.features(x_on_2pi).cat(dim=1)  # u(θ, 2π)
    A_on_periodic = A_on_0 - A_on_2pi

    # assume u(π / 2, 0) = 0, actually wrong
    x_on_try = torch.tensor([[torch.pi / 2, 0.0]], dtype=torch.float64)
    A_on_try = model.features(x_on_try).cat(dim=1)  # u(π / 2, 0)
    A = pyrfm.concat_blocks([[-A_in], [A_on_periodic], [A_on_try]])
    f = pyrfm.concat_blocks(
        [[f(x_in[:, 0], x_in[:, 1]).view(-1, 1)], [torch.zeros_like(A_on_periodic[:, 0]).view(-1, 1)],
         [torch.zeros_like(A_on_try[:, 0]).view(-1, 1)]])

    model.compute(A).solve(f)


    def sphere_integral(model, N_theta=64, N_phi=64):
        """
        使用 Gauss-Legendre 积分在单位球面上计算 ∫_{S²} model(θ, φ) dS ≈ ∫∫ u(θ, φ) sinθ dθ dφ

        参数:
            model: PyTorch 模型，输入 (N, 2) 张量 (θ, φ)，输出 shape (N, 1) 或 (N,)
            N_theta: θ 方向 Gauss-Legendre 点数，区间 [0, π]
            N_phi: φ 方向 Gauss-Legendre 点数，区间 [0, 2π]

        返回:
            近似积分值（标量张量）
        """
        # Gauss-Legendre 点和权重 for θ ∈ [0, π]
        xi_theta, wi_theta = np.polynomial.legendre.leggauss(N_theta)
        theta_vals = 0.5 * (np.pi) * (xi_theta + 1)
        w_theta_vals = 0.5 * np.pi * wi_theta

        # Gauss-Legendre 点和权重 for φ ∈ [0, 2π]
        xi_phi, wi_phi = np.polynomial.legendre.leggauss(N_phi)
        phi_vals = np.pi * (xi_phi + 1)
        w_phi_vals = np.pi * wi_phi

        # 网格化 θ 和 φ
        theta_grid, phi_grid = np.meshgrid(theta_vals, phi_vals, indexing='ij')  # shape: (N_theta, N_phi)
        w_theta_grid, w_phi_grid = np.meshgrid(w_theta_vals, w_phi_vals, indexing='ij')

        # 面积权重：sinθ × dθ × dφ
        sin_theta = np.sin(theta_grid)
        area_weights = sin_theta * w_theta_grid * w_phi_grid  # shape: (N_theta, N_phi)

        # 转为 PyTorch 张量
        theta_phi = torch.tensor(
            np.stack([theta_grid.ravel(), phi_grid.ravel()], axis=1),
            dtype=torch.float64
        )
        weights = torch.tensor(area_weights.ravel(), dtype=torch.float64)

        # 模型输出
        u_vals = model(theta_phi).squeeze()

        # 积分近似
        integral = torch.sum(weights * u_vals)

        return integral


    constant = sphere_integral(model, N_theta=64, N_phi=64) / (4 * np.pi)

    x_test = domain.in_sample(100, with_boundary=False)
    u_test = u(x_test[:, 0], x_test[:, 1]).view(-1, 1)
    u_pred = model(x_test)
    u_pred -= constant  # adjust to satisfy the integral condition

    print((u_test - u_pred).norm() / u_test.norm())

    print('Time:', time.time() - start_time, ", with", torch.tensor(0.).device)

    # 创建参数域网格
    N = 200
    theta_vals = torch.linspace(0, torch.pi, N)
    phi_vals = torch.linspace(0, 2 * torch.pi, N)
    theta_grid, phi_grid = torch.meshgrid(theta_vals, phi_vals, indexing='ij')

    # 生成对应的输入点 (θ, φ)
    theta_phi_grid = torch.stack([theta_grid.ravel(), phi_grid.ravel()], dim=1)

    # 预测 u(θ, φ)
    with torch.no_grad():
        u_vis = model(theta_phi_grid).view(N, N)
        u_vis -= constant  # 调整满足积分条件

    # 绘制图像
    plt.figure(figsize=(8, 4))
    plt.contourf(phi_grid.cpu(), theta_grid.cpu(), u_vis.cpu(), levels=100)
    plt.xlabel(r"$\phi$")
    plt.ylabel(r"$\theta$")
    plt.title(r"Approximate solution $u(\theta, \phi)$ on the parametric domain")
    plt.colorbar(label="u(θ, φ)")
    plt.tight_layout()
    plt.show()
