from typing import List

import pyrfm
import torch


class Heart(pyrfm.ImplicitSurfaceBase):
    """
    3D implicit heart surface.

    F(x, y, z) = (x^2 + 9/4 y^2 + z^2 - 1)^3
                 - x^2 z^3
                 - 9/80 y^2 z^3
    """

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale

    def get_bounding_box(self) -> List[float]:
        # Conservative bounding box for the heart
        s = self.scale
        return [-1.5 * s, 1.5 * s,
                -1.5 * s, 1.5 * s,
                -1.5 * s, 1.5 * s]

    def shape_func(self, p: torch.Tensor) -> torch.Tensor:
        """
        Implicit function F(p) = 0 defines the heart surface.
        """
        x = p[:, 0] / self.scale
        y = p[:, 1] / self.scale
        z = p[:, 2] / self.scale

        F = (
                (x ** 2 + (9.0 / 4.0) * y ** 2 + z ** 2 - 1.0) ** 3
                - x ** 2 * z ** 3
                - (9.0 / 80.0) * y ** 2 * z ** 3
        )

        return F.unsqueeze(-1)


def visualize_heart(
        x: torch.Tensor,
        normal: torch.Tensor,
        mean_curvature: torch.Tensor,
        max_points_scatter: int = 30000,
        max_points_quiver: int = 2000,
):
    """
    Simple matplotlib-based visualization:
    - 3D scatter of the surface colored by mean curvature
    - 3D quiver of normals (subsampled)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    # Move to CPU & numpy
    x_np = x.detach().cpu().numpy()
    n_np = normal.detach().cpu().numpy()
    mc_np = mean_curvature.detach().cpu().squeeze(-1).numpy()

    # 只保留有限值，避免 NaN/Inf 影响 colormap
    finite_mask = np.isfinite(mc_np)
    x_np = x_np[finite_mask]
    n_np = n_np[finite_mask]
    mc_np = mc_np[finite_mask]

    # 散点图：如果点太多，随机子采样
    if x_np.shape[0] > max_points_scatter:
        idx = np.random.choice(x_np.shape[0], max_points_scatter, replace=False)
        x_s = x_np[idx]
        mc_s = mc_np[idx]
    else:
        x_s = x_np
        mc_s = mc_np

    # 为避免极端值拉爆颜色范围，用分位数截断
    vmin = np.quantile(mc_s, 0.05)
    vmax = np.quantile(mc_s, 0.95)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(
        x_s[:, 0],
        x_s[:, 1],
        x_s[:, 2],
        c=mc_s,
        s=1,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        alpha=0.9,
    )
    cb = plt.colorbar(sc, ax=ax, shrink=0.6)
    cb.set_label("Mean curvature")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Implicit heart surface (colored by mean curvature)")

    # 保持三轴等比例
    max_range = (x_s.max(axis=0) - x_s.min(axis=0)).max() / 2.0
    mid_x = (x_s.max(axis=0)[0] + x_s.min(axis=0)[0]) * 0.5
    mid_y = (x_s.max(axis=0)[1] + x_s.min(axis=0)[1]) * 0.5
    mid_z = (x_s.max(axis=0)[2] + x_s.min(axis=0)[2]) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    plt.show()

    # 法向箭头可视化（再开一个图，子采样）
    if x_np.shape[0] > max_points_quiver:
        idx_q = np.random.choice(x_np.shape[0], max_points_quiver, replace=False)
        x_q = x_np[idx_q]
        n_q = n_np[idx_q]
    else:
        x_q = x_np
        n_q = n_np

    fig2 = plt.figure(figsize=(10, 8))
    ax2 = fig2.add_subplot(111, projection="3d")

    ax2.quiver(
        x_q[:, 0],
        x_q[:, 1],
        x_q[:, 2],
        n_q[:, 0],
        n_q[:, 1],
        n_q[:, 2],
        length=0.1,
        normalize=True,
        linewidths=0.3,
    )

    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("z")
    ax2.set_title("Normals on the implicit heart surface")

    max_range = (x_q.max(axis=0) - x_q.min(axis=0)).max() / 2.0
    mid_x = (x_q.max(axis=0)[0] + x_q.min(axis=0)[0]) * 0.5
    mid_y = (x_q.max(axis=0)[1] + x_q.min(axis=0)[1]) * 0.5
    mid_z = (x_q.max(axis=0)[2] + x_q.min(axis=0)[2]) * 0.5
    ax2.set_xlim(mid_x - max_range, mid_x + max_range)
    ax2.set_ylim(mid_y - max_range, mid_y + max_range)
    ax2.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # 设置默认设备
    if torch.cuda.is_available():
        torch.set_default_device('cuda')
    else:
        torch.set_default_device('cpu')

    domain = Heart(scale=1.0)

    # sample interior points (points on the surface)
    x = domain.in_sample(100000)

    # SDF + normal + mean curvature
    _, normal, mean_curvature = domain.sdf(
        x, with_normal=True, with_curvature=True
    )

    # save
    torch.save(
        (x, normal, mean_curvature),
        '../../data/heart_in.pth'
    )

    print(x.shape, normal.shape, mean_curvature.shape)

    # reload test
    x_loaded, normal_loaded, mean_curvature_loaded = torch.load(
        '../../data/heart_in.pth',
        map_location=torch.tensor(0.).device
    )

    # 可视化（用重载的数据也可以，这里直接用当前变量）
    visualize_heart(x, normal, mean_curvature)
