from typing import List

import pyrfm
import torch
import time


class Ellipsoid(pyrfm.ImplicitSurfaceBase):

    def __init__(self):
        super().__init__()
        self.a, self.b, self.c = 1.5, 1.0, 0.5

    def get_bounding_box(self) -> List[float]:
        return [-self.a, self.a, -self.b, self.b, -self.c, self.c]

    def shape_func(self, p: torch.Tensor) -> torch.Tensor:
        # Ellipsoid equation: (x/a)^2 + (y/b)^2 + (z/c)^2 = 1

        return ((p[:, 0] / self.a) ** 2 + (p[:, 1] / self.b) ** 2 + (p[:, 2] / self.c) ** 2 - 1).unsqueeze(-1)


def func_u(p: torch.Tensor) -> torch.Tensor:
    """
    Example function u(x, y, z) = sin(x) * exp(cos(y - z))
    """
    return (torch.sin(p[:, 0]) * torch.exp(torch.cos(p[:, 1] - p[:, 2]))).unsqueeze(-1)


def func_rhs(p: torch.Tensor) -> torch.Tensor:
    """
    Laplace–Beltrami of u(x,y,z) on given implicit surface
    """
    return ((4 * p[:, 0] * (16 * p[:, 0] ** 2 + 81 * p[:, 1] ** 2 + 1296 * p[:, 2] ** 2) * (
            4 * p[:, 0] * torch.sin(p[:, 0]) + 9 * p[:, 1] * torch.sin(p[:, 1] - p[:, 2]) * torch.cos(
        p[:, 0]) - 36 * p[:, 2] * torch.sin(p[:, 1] - p[:, 2]) * torch.cos(p[:, 0])) - 9 * p[:, 1] * (
                     16 * p[:, 0] ** 2 + 81 * p[:, 1] ** 2 + 1296 * p[:, 2] ** 2) * (
                     -4 * p[:, 0] * torch.sin(p[:, 1] - p[:, 2]) * torch.cos(p[:, 0]) + 9 * p[:, 1] * (
                     torch.sin(p[:, 1] - p[:, 2]) ** 2 - torch.cos(p[:, 1] - p[:, 2])) * torch.sin(
                 p[:, 0]) - 36 * p[:, 2] * (
                             torch.sin(p[:, 1] - p[:, 2]) ** 2 - torch.cos(p[:, 1] - p[:, 2])) * torch.sin(
                 p[:, 0])) - 36 * p[:, 2] * (16 * p[:, 0] ** 2 + 81 * p[:, 1] ** 2 + 1296 * p[:, 2] ** 2) * (
                     4 * p[:, 0] * torch.sin(p[:, 1] - p[:, 2]) * torch.cos(p[:, 0]) - 9 * p[:, 1] * (
                     torch.sin(p[:, 1] - p[:, 2]) ** 2 - torch.cos(p[:, 1] - p[:, 2])) * torch.sin(
                 p[:, 0]) + 36 * p[:, 2] * (
                             torch.sin(p[:, 1] - p[:, 2]) ** 2 - torch.cos(p[:, 1] - p[:, 2])) * torch.sin(
                 p[:, 0])) + 2 * (torch.sin(p[:, 1] - p[:, 2]) ** 2 - torch.cos(p[:, 1] - p[:, 2])) * (
                     16 * p[:, 0] ** 2 + 81 * p[:, 1] ** 2 + 1296 * p[:, 2] ** 2) ** 2 * torch.sin(p[:, 0]) + (
                     -720 * p[:, 0] ** 2 - 3240 * p[:, 1] ** 2 - 16848 * p[:, 2] ** 2) * (
                     4 * p[:, 0] * torch.cos(p[:, 0]) - 9 * p[:, 1] * torch.sin(p[:, 0]) * torch.sin(
                 p[:, 1] - p[:, 2]) + 36 * p[:, 2] * torch.sin(p[:, 0]) * torch.sin(p[:, 1] - p[:, 2])) - (
                     16 * p[:, 0] ** 2 + 81 * p[:, 1] ** 2 + 1296 * p[:, 2] ** 2) ** 2 * torch.sin(
        p[:, 0])) * torch.exp(torch.cos(p[:, 1] - p[:, 2])) / (
                    16 * p[:, 0] ** 2 + 81 * p[:, 1] ** 2 + 1296 * p[:, 2] ** 2) ** 2).unsqueeze(-1)


if __name__ == '__main__':
    torch.set_default_device('cuda') if torch.cuda.is_available() else torch.set_default_device('cpu')
    domain = Ellipsoid()
    model = pyrfm.RFMBase(dim=3, n_hidden=200, domain=domain, n_subdomains=2)

    # Step 1: 采样点 & 几何
    x_in = domain.in_sample(num_samples=10000)
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

    x_on = torch.tensor([1.5, 0.0, 0.0]).view(1, -1)
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

    visualizer = pyrfm.RFMVisualizer3D(model, view='iso', resolution=(800, 800), ref=func_u)
    visualizer.plot()
    visualizer.show()

    # -----------------------------------
    # 🖼️ 计时结束
    t5 = time.time()
    print(f'[Timer] Rendering time: {t5 - t4:.3f} seconds')
    # -----------------------------------

    # # plot x_in as scatter plot in 3D
    # from mpl_toolkits.mplot3d import Axes3D
    # import matplotlib.pyplot as plt
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x_in[:, 0].numpy(), x_in[:, 1].numpy(), x_in[:, 2].numpy(), s=1)
    # ax.set_xlabel('X axis')
    # ax.set_ylabel('Y axis')
    # ax.set_zlabel('Z axis')
    # plt.axis('equal')
    # plt.show()
    #
    # print(x_in)
