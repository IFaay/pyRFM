# -*- coding: utf-8 -*-
"""
Created on 7/28/25

@author: Yifei Sun
"""
from typing import List
import pyrfm
import torch
import time


class Torus(pyrfm.ImplicitSurfaceBase):
    def __init__(self):
        super().__init__()

    def get_bounding_box(self) -> List[float]:
        return [-1.25, 1.25, -1.25, 1.25, -0.25, 0.25]

    def shape_func(self, p: torch.Tensor) -> torch.Tensor:
        # ψ(x, y, z) = (√(x² + y²) − 1)² + z² − 1 / 16
        return (((p[:, 0] ** 2 + p[:, 1] ** 2) ** 0.5 - 1.0) ** 2 + (p[:, 2] ** 2) - 1 / 16).unsqueeze(-1)


def func_u(p: torch.Tensor) -> torch.Tensor:
    """
    Example function u(x, y, z) = sin(x) * exp(cos(y - z))
    """
    return (torch.sin(p[:, 0]) * torch.exp(torch.cos(p[:, 1] - p[:, 2]))).unsqueeze(-1)


def func_rhs(p: torch.Tensor) -> torch.Tensor:
    """
    Laplace–Beltrami of u(x,y,z) on given implicit surface
    """
    return ((-(p[:, 0] ** 2 + p[:, 1] ** 2) ** (3 / 2) * (
            p[:, 0] * (torch.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2) - 1) * torch.cos(p[:, 0]) - p[:, 1] * (
            torch.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2) - 1) * torch.sin(p[:, 0]) * torch.sin(
        p[:, 1] - p[:, 2]) + p[:, 2] * torch.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2) * torch.sin(p[:, 0]) * torch.sin(
        p[:, 1] - p[:, 2])) * (p[:, 0] ** 2 * (p[:, 0] ** 2 + p[:, 1] ** 2) ** (7 / 2) * (
            p[:, 0] ** 2 * (torch.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2) - 1) ** 2 + p[:, 1] ** 2 * (
            torch.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2) - 1) ** 2 + p[:, 2] ** 2 * (
                    p[:, 0] ** 2 + p[:, 1] ** 2)) + p[:, 0] ** 2 * (p[:, 0] ** 2 + p[:, 1] ** 2) ** (3 / 2) * (
                                       torch.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2) - 1) ** 2 * (
                                       p[:, 0] ** 2 * (p[:, 0] ** 2 + p[:, 1] ** 2) ** (3 / 2) * (
                                       torch.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2) - 1) - p[:, 0] ** 2 * (
                                               p[:, 0] ** 2 + p[:, 1] ** 2) ** 2 + p[:, 1] ** 2 * (
                                               p[:, 0] ** 2 + p[:, 1] ** 2) ** (3 / 2) * (
                                               torch.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2) - 1) - p[:,
                                                                                              1] ** 2 * (
                                               p[:, 0] ** 2 + p[:, 1] ** 2) ** 2 + (
                                               1 - torch.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2)) * (
                                               p[:, 0] ** 2 + p[:, 1] ** 2) ** (5 / 2)) - p[:, 0] ** 2 * (
                                       p[:, 0] ** 2 + p[:, 1] ** 2) ** 3 * (
                                       torch.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2) - 1) * (
                                       p[:, 0] ** 2 * (torch.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2) - 1) ** 2 + p[:,
                                                                                                           1] ** 2 * (
                                               torch.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2) - 1) ** 2 + p[:,
                                                                                                   2] ** 2 * (
                                               p[:, 0] ** 2 + p[:, 1] ** 2)) + p[:, 1] ** 2 * (
                                       p[:, 0] ** 2 + p[:, 1] ** 2) ** (7 / 2) * (
                                       p[:, 0] ** 2 * (torch.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2) - 1) ** 2 + p[:,
                                                                                                           1] ** 2 * (
                                               torch.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2) - 1) ** 2 + p[:,
                                                                                                   2] ** 2 * (
                                               p[:, 0] ** 2 + p[:, 1] ** 2)) + p[:, 1] ** 2 * (
                                       p[:, 0] ** 2 + p[:, 1] ** 2) ** (3 / 2) * (
                                       torch.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2) - 1) ** 2 * (
                                       p[:, 0] ** 2 * (p[:, 0] ** 2 + p[:, 1] ** 2) ** (3 / 2) * (
                                       torch.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2) - 1) - p[:, 0] ** 2 * (
                                               p[:, 0] ** 2 + p[:, 1] ** 2) ** 2 + p[:, 1] ** 2 * (
                                               p[:, 0] ** 2 + p[:, 1] ** 2) ** (3 / 2) * (
                                               torch.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2) - 1) - p[:,
                                                                                              1] ** 2 * (
                                               p[:, 0] ** 2 + p[:, 1] ** 2) ** 2 + (
                                               1 - torch.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2)) * (
                                               p[:, 0] ** 2 + p[:, 1] ** 2) ** (5 / 2)) - p[:, 1] ** 2 * (
                                       p[:, 0] ** 2 + p[:, 1] ** 2) ** 3 * (
                                       torch.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2) - 1) * (
                                       p[:, 0] ** 2 * (torch.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2) - 1) ** 2 + p[:,
                                                                                                           1] ** 2 * (
                                               torch.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2) - 1) ** 2 + p[:,
                                                                                                   2] ** 2 * (
                                               p[:, 0] ** 2 + p[:, 1] ** 2)) - p[:, 2] ** 2 * (
                                       p[:, 0] ** 2 + p[:, 1] ** 2) ** (11 / 2) + (
                                       p[:, 0] ** 2 + p[:, 1] ** 2) ** (9 / 2) * (
                                       p[:, 0] ** 2 * (torch.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2) - 1) ** 2 + p[:,
                                                                                                           1] ** 2 * (
                                               torch.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2) - 1) ** 2 + p[:,
                                                                                                   2] ** 2 * (
                                               p[:, 0] ** 2 + p[:, 1] ** 2)) + 2 * (
                                       p[:, 0] ** 2 + p[:, 1] ** 2) ** 4 * (
                                       torch.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2) - 1) * (
                                       p[:, 0] ** 2 * (torch.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2) - 1) ** 2 + p[:,
                                                                                                           1] ** 2 * (
                                               torch.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2) - 1) ** 2 + p[:,
                                                                                                   2] ** 2 * (
                                               p[:, 0] ** 2 + p[:, 1] ** 2))) + (
                     -p[:, 0] ** 2 * (torch.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2) - 1) ** 2 - p[:, 1] ** 2 * (
                     torch.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2) - 1) ** 2 - p[:, 2] ** 2 * (
                             p[:, 0] ** 2 + p[:, 1] ** 2)) * (
                     -p[:, 0] * (p[:, 0] ** 2 + p[:, 1] ** 2) ** (11 / 2) * (
                     torch.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2) - 1) * (
                             p[:, 0] * (torch.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2) - 1) * torch.sin(p[:, 0]) + p[:,
                                                                                                            1] * (
                                     torch.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2) - 1) * torch.sin(
                         p[:, 1] - p[:, 2]) * torch.cos(p[:, 0]) - p[:, 2] * torch.sqrt(
                         p[:, 0] ** 2 + p[:, 1] ** 2) * torch.sin(p[:, 1] - p[:, 2]) * torch.cos(p[:, 0])) + p[
                                                                                                             :,
                                                                                                             1] * (
                             p[:, 0] ** 2 + p[:, 1] ** 2) ** (11 / 2) * (
                             torch.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2) - 1) * (
                             -p[:, 0] * (torch.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2) - 1) * torch.sin(
                         p[:, 1] - p[:, 2]) * torch.cos(p[:, 0]) + p[:, 1] * (
                                     torch.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2) - 1) * (
                                     torch.sin(p[:, 1] - p[:, 2]) ** 2 - torch.cos(
                                 p[:, 1] - p[:, 2])) * torch.sin(p[:, 0]) - p[:, 2] * torch.sqrt(
                         p[:, 0] ** 2 + p[:, 1] ** 2) * (torch.sin(p[:, 1] - p[:, 2]) ** 2 - torch.cos(
                         p[:, 1] - p[:, 2])) * torch.sin(p[:, 0])) + p[:, 2] * (
                             p[:, 0] ** 2 + p[:, 1] ** 2) ** 6 * (
                             p[:, 0] * (torch.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2) - 1) * torch.sin(
                         p[:, 1] - p[:, 2]) * torch.cos(p[:, 0]) - p[:, 1] * (
                                     torch.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2) - 1) * (
                                     torch.sin(p[:, 1] - p[:, 2]) ** 2 - torch.cos(
                                 p[:, 1] - p[:, 2])) * torch.sin(p[:, 0]) + p[:, 2] * torch.sqrt(
                         p[:, 0] ** 2 + p[:, 1] ** 2) * (torch.sin(p[:, 1] - p[:, 2]) ** 2 - torch.cos(
                         p[:, 1] - p[:, 2])) * torch.sin(p[:, 0])) - 2 * (p[:, 0] ** 2 + p[:, 1] ** 2) ** (
                             11 / 2) * (torch.sin(p[:, 1] - p[:, 2]) ** 2 - torch.cos(p[:, 1] - p[:, 2])) * (
                             p[:, 0] ** 2 * (torch.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2) - 1) ** 2 + p[:,
                                                                                                 1] ** 2 * (
                                     torch.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2) - 1) ** 2 + p[:, 2] ** 2 * (
                                     p[:, 0] ** 2 + p[:, 1] ** 2)) * torch.sin(p[:, 0]) + (
                             p[:, 0] ** 2 + p[:, 1] ** 2) ** (11 / 2) * (
                             p[:, 0] ** 2 * (torch.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2) - 1) ** 2 + p[:,
                                                                                                 1] ** 2 * (
                                     torch.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2) - 1) ** 2 + p[:, 2] ** 2 * (
                                     p[:, 0] ** 2 + p[:, 1] ** 2)) * torch.sin(p[:, 0]))) * torch.exp(
        torch.cos(p[:, 1] - p[:, 2])) / ((p[:, 0] ** 2 + p[:, 1] ** 2) ** (11 / 2) * (
            p[:, 0] ** 2 * (torch.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2) - 1) ** 2 + p[:, 1] ** 2 * (
            torch.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2) - 1) ** 2 + p[:, 2] ** 2 * (
                    p[:, 0] ** 2 + p[:, 1] ** 2)) ** 2)).unsqueeze(-1)


if __name__ == '__main__':
    torch.set_default_device('cuda') if torch.cuda.is_available() else torch.set_default_device('cpu')
    domain = Torus()
    model = pyrfm.RFMBase(dim=3, n_hidden=400, domain=domain, n_subdomains=2)

    # Step 1: 采样点 & 几何
    x_in = domain.in_sample(num_samples=6000)
    _, normal, mean_curvature = domain.sdf(x_in, with_normal=True, with_curvature=True)

    # -----------------------------------
    # 🔧 计时开始：组装矩阵
    t0 = time.time()
    # -----------------------------------

    A_xx = model.features_second_derivative(x_in, axis1=0, axis2=0).cat(dim=1)
    A_yy = model.features_second_derivative(x_in, axis1=1, axis2=1).cat(dim=1)
    A_zz = model.features_second_derivative(x_in, axis1=2, axis2=2).cat(dim=1)
    A_xy = model.features_second_derivative(x_in, axis1=0, axis2=1).cat(dim=1)
    A_xz = model.features_second_derivative(x_in, axis1=0, axis2=2).cat(dim=1)
    A_yz = model.features_second_derivative(x_in, axis1=1, axis2=2).cat(dim=1)

    A_lap = A_xx + A_yy + A_zz

    # 手动展开 einsum('ni,nijk,nj->nk') 等价于：
    A_nHn = (
            normal[:, 0:1] * (A_xx * normal[:, 0:1] + A_xy * normal[:, 1:2] + A_xz * normal[:, 2:3]) +
            normal[:, 1:2] * (A_xy * normal[:, 0:1] + A_yy * normal[:, 1:2] + A_yz * normal[:, 2:3]) +
            normal[:, 2:3] * (A_xz * normal[:, 0:1] + A_yz * normal[:, 1:2] + A_zz * normal[:, 2:3])
    )

    # 可选释放内存
    del A_xx, A_yy, A_zz, A_xy, A_xz, A_yz
    torch.cuda.empty_cache()

    A_in_x = model.features_derivative(x_in, axis=0).cat(dim=1)
    A_in_y = model.features_derivative(x_in, axis=1).cat(dim=1)
    A_in_z = model.features_derivative(x_in, axis=2).cat(dim=1)

    A_grad = torch.stack([A_in_x, A_in_y, A_in_z], dim=1)
    A_partial_n = (
            normal[:, 0:1] * A_in_x +
            normal[:, 1:2] * A_in_y +
            normal[:, 2:3] * A_in_z
    )
    del A_in_x, A_in_y, A_in_z
    torch.cuda.empty_cache()

    A_lap_beltrami = A_lap - 2 * mean_curvature * A_partial_n - A_nHn

    b_in = func_rhs(x_in)

    x_on = torch.tensor([1.0, 0.0, 1 / 4]).view(1, -1)
    A_on = model.features(x_on).cat(dim=1)
    b_on = func_u(x_on)

    A = pyrfm.concat_blocks([[A_lap_beltrami], [A_on]])
    b = pyrfm.concat_blocks([[b_in], [b_on]])

    del A_lap_beltrami, A_lap, A_partial_n, A_nHn, A_grad
    torch.cuda.empty_cache()

    # -----------------------------------
    # 🔧 计时结束：组装矩阵
    t1 = time.time()
    print(f'[Timer] Matrix assembly time: {t1 - t0:.3f} seconds')
    # -----------------------------------

    # -----------------------------------
    # 🧮 计时开始：求解系统
    t2 = time.time()
    # -----------------------------------

    model.compute(A).solve(b)

    del A
    torch.cuda.empty_cache()

    # -----------------------------------
    # 🧮 计时结束
    t3 = time.time()
    print(f'[Timer] Linear solve time: {t3 - t2:.3f} seconds')
    # -----------------------------------

    u_pred = model(x_in)
    u_exact = func_u(x_in)

    error = torch.norm(u_pred - u_exact) / torch.norm(u_exact)
    print(f'Error: {error.item():.4e}')

    # -----------------------------------
    # 🖼️ 计时开始：渲染
    t4 = time.time()
    # -----------------------------------

    visualizer = pyrfm.RFMVisualizer3D(model, view='iso', resolution=(800, 800))
    visualizer.plot()
    visualizer.show()

    # -----------------------------------
    # 🖼️ 计时结束
    t5 = time.time()
    print(f'[Timer] Rendering time: {t5 - t4:.3f} seconds')
    # -----------------------------------
