# -*- coding: utf-8 -*-
"""
Created on 6/13/25

@author: Yifei Sun
"""
"""
Problem 4: Laplace-Beltrami Operator on the Sphere S²

Consider using u(θ, φ) = sin²(θ) · cos(2φ) as the analytical solution.
Δ_{S²}(sin²θ · cos(2φ)) = -6 · sin²θ · cos(2φ)

Geometry:
φ(x, y, z) = x² + y² + z² - 1

Differential Operator Expressions:

Δ_{S²} u = (1 - x²) ∂²u/∂x² + (1 - y²) ∂²u/∂y² + (1 - z²) ∂²u/∂z²
           - 2xy ∂²u/∂x∂y - 2xz ∂²u/∂x∂z - 2yz ∂²u/∂y∂z
           - 2x ∂u/∂x - 2y ∂u/∂y - 2z ∂u/∂z

Boundary Condition:

∫_{S²} u d𝑥 = 0

Basis Function Representation:

Physical domain: u(x, y, z) = ∑ φₘ(x, y, z), where φₘ(x, y, z) are stochastic basis functions.


"""

import pyrfm
import torch
import numpy as np
import time
from matplotlib import pyplot as plt


def to_spherical(x):
    r = torch.norm(x, dim=-1)
    theta = torch.acos(x[..., 2] / r)  # z / r
    phi = torch.atan2(x[..., 1], x[..., 0])  # atan2(y, x)
    return theta, phi


def u_xyz(x):
    theta, phi = to_spherical(x)
    return (torch.sin(theta) ** 2 * torch.cos(2 * phi)).view(-1, 1)


def f_xyz(x):
    theta, phi = to_spherical(x)
    return (6 * torch.sin(theta) ** 2 * torch.cos(2 * phi)).view(-1, 1)


if __name__ == "__main__":
    torch.set_default_device('cuda') if torch.cuda.is_available() else torch.set_default_device('cpu')
    start_time = time.time()
    domain = pyrfm.Ball3D(center=(0.0, 0.0, 0.0), radius=1.0)
    model = pyrfm.RFMBase(dim=3, n_hidden=400, domain=domain, n_subdomains=1, pou=pyrfm.PsiB)
    x_in = domain.on_sample(1600)

    x_in_x = x_in[:, 0].view(-1, 1)
    x_in_y = x_in[:, 1].view(-1, 1)
    x_in_z = x_in[:, 2].view(-1, 1)
    A_in_x = model.features_derivative(x_in, axis=0).cat(dim=1)  # ∂u/∂x
    A_in_y = model.features_derivative(x_in, axis=1).cat(dim=1)  # ∂u/∂y
    A_in_z = model.features_derivative(x_in, axis=2).cat(dim=1)  # ∂u/∂z
    A_in_xx = model.features_second_derivative(x_in, axis1=0, axis2=0).cat(dim=1)  # ∂²u/∂x²
    A_in_yy = model.features_second_derivative(x_in, axis1=1, axis2=1).cat(dim=1)  # ∂²u/∂y²
    A_in_zz = model.features_second_derivative(x_in, axis1=2, axis2=2).cat(dim=1)  # ∂²u/∂z²
    A_in_xy = model.features_second_derivative(x_in, axis1=0, axis2=1).cat(dim=1)  # ∂²u/∂x∂y
    A_in_xz = model.features_second_derivative(x_in, axis1=0, axis2=2).cat(dim=1)  # ∂²u/∂x∂z
    A_in_yz = model.features_second_derivative(x_in, axis1=1, axis2=2).cat(dim=1)  # ∂²u/∂y∂z

    A_in = (1 - x_in_x ** 2) * A_in_xx + (1 - x_in_y ** 2) * A_in_yy + (1 - x_in_z ** 2) * A_in_zz \
           - 2 * x_in_x * x_in_y * A_in_xy - 2 * x_in_x * x_in_z * A_in_xz - 2 * x_in_y * x_in_z * A_in_yz \
           - 2 * x_in_x * A_in_x - 2 * x_in_y * A_in_y - 2 * x_in_z * A_in_z

    x_on = torch.tensor([[1.0, 0.0, 0.0]])
    A_on = model.features(x_on).cat(dim=1)  # u(1, 0, 0)

    A = pyrfm.concat_blocks([[-A_in], [A_on]])
    f = pyrfm.concat_blocks([[f_xyz(x_in).view(-1, 1)], [torch.ones_like(A_on[:, 0]).view(-1, 1)]])

    model.compute(A).solve(f)


    def sphere_integral(model, N_theta=64, N_phi=64):
        """
        在单位球面 S² 上计算 ∫_{S²} u(x, y, z) dS，适配 3D 模型输入 (x, y, z)。

        参数:
            model: 接收 (N, 3) 输入，返回 (N,) 或 (N, 1) 的 PyTorch 模型
            N_theta: θ 方向积分点数（θ ∈ [0, π]）
            N_phi: φ 方向积分点数（φ ∈ [0, 2π]）

        返回:
            单位球面上的积分值（标量）
        """
        import numpy as np
        import torch

        # Gauss-Legendre points and weights for θ ∈ [0, π]
        xi_theta, wi_theta = np.polynomial.legendre.leggauss(N_theta)
        theta_vals = 0.5 * np.pi * (xi_theta + 1)
        w_theta = 0.5 * np.pi * wi_theta

        # Gauss-Legendre points and weights for φ ∈ [0, 2π]
        xi_phi, wi_phi = np.polynomial.legendre.leggauss(N_phi)
        phi_vals = np.pi * (xi_phi + 1)
        w_phi = np.pi * wi_phi

        # 生成网格
        theta_grid, phi_grid = np.meshgrid(theta_vals, phi_vals, indexing='ij')  # shape (N_theta, N_phi)
        w_theta_grid, w_phi_grid = np.meshgrid(w_theta, w_phi, indexing='ij')  # shape (N_theta, N_phi)

        # 球面坐标 -> 直角坐标
        x = np.sin(theta_grid) * np.cos(phi_grid)
        y = np.sin(theta_grid) * np.sin(phi_grid)
        z = np.cos(theta_grid)

        # 面积权重：sinθ × dθ × dφ
        sin_theta = np.sin(theta_grid)
        area_weights = sin_theta * w_theta_grid * w_phi_grid  # shape: (N_theta, N_phi)

        # 构造输入张量
        xyz = np.stack([x, y, z], axis=-1).reshape(-1, 3)  # shape: (N_theta*N_phi, 3)
        xyz_tensor = torch.tensor(xyz, dtype=torch.float32)

        weights = torch.tensor(area_weights.ravel(), dtype=torch.float32)

        # 模型评估
        with torch.no_grad():
            u_vals = model(xyz_tensor).squeeze()
            if u_vals.ndim == 0:
                u_vals = u_vals.unsqueeze(0)

        # 加权积分
        integral = torch.sum(weights * u_vals)

        return integral


    constant = sphere_integral(model, N_theta=64, N_phi=64) / (4 * np.pi)
    print('Integral constant:', constant.item())

    x_test = domain.on_sample(100)
    u_test = u_xyz(x_test).view(-1, 1)  # analytical solution
    u_pred = model(x_test)
    u_pred -= constant  # adjust to satisfy the integral condition

    print((u_test - u_pred).norm() / u_test.norm())

    print('Time:', time.time() - start_time, ", with", torch.tensor(0.).device)

    from mpl_toolkits.mplot3d import Axes3D

    # 创建可视化网格点
    N_theta_vis = 100
    N_phi_vis = 200
    theta_vis = np.linspace(0, np.pi, N_theta_vis)
    phi_vis = np.linspace(0, 2 * np.pi, N_phi_vis)
    theta_grid_vis, phi_grid_vis = np.meshgrid(theta_vis, phi_vis, indexing='ij')

    # 球坐标 -> 笛卡尔坐标
    x_vis = np.sin(theta_grid_vis) * np.cos(phi_grid_vis)
    y_vis = np.sin(theta_grid_vis) * np.sin(phi_grid_vis)
    z_vis = np.cos(theta_grid_vis)

    # 拼接为点坐标
    xyz_vis = np.stack([x_vis, y_vis, z_vis], axis=-1).reshape(-1, 3)
    xyz_tensor_vis = torch.tensor(xyz_vis, dtype=torch.float32)

    # 计算预测值并做常数修正
    with torch.no_grad():
        u_vis_pred = model(xyz_tensor_vis).squeeze() - constant

    # reshape 回网格
    u_vis_pred = u_vis_pred.cpu().numpy().reshape(N_theta_vis, N_phi_vis)

    # 绘图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    # 显示在球面上的颜色映射
    ax.plot_surface(
        x_vis, y_vis, z_vis,
        facecolors=plt.cm.viridis((u_vis_pred - u_vis_pred.min()) / (u_vis_pred.max() - u_vis_pred.min())),
        rstride=1, cstride=1, antialiased=False, shade=False
    )
    ax.set_title("Predicted Solution on the Sphere", fontsize=14)
    ax.set_box_aspect([1, 1, 1])
    plt.show()
