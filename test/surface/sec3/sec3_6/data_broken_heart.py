# -*- coding: utf-8 -*-
"""
Broken-heart surface:

- Sample an implicit heart surface via pyRFM.
- Compute normals and mean curvature.
- Define a crack-like folded surface in the xz-plane via a polyline x = x(z).
- Use this folded surface to split the heart into:
    x_trim_in, x_trim_out, x_trim_boundary

Naming:
- x               : full heart surface (all points)
- x_trim_in       : points on one side of the folded surface (kept side)
- x_trim_out      : points on the other side (clipped side)
- x_trim_boundary : intersection curve between folded surface and heart
"""

from typing import List

import numpy as np
import matplotlib.pyplot as plt
import pyrfm
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# =========================================================
# Heart implicit surface
# =========================================================

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
        # 当前代码假定 scale=1.0 使用；如需一般 scale 需对应修改解析求根部分
        x = p[:, 0] / self.scale
        y = p[:, 1] / self.scale
        z = p[:, 2] / self.scale

        F = (
                (x ** 2 + (9.0 / 4.0) * y ** 2 + z ** 2 - 1.0) ** 3
                - x ** 2 * z ** 3
                - (9.0 / 80.0) * y ** 2 * z ** 3
        )

        return F.unsqueeze(-1)


# =========================================================
# Crack polyline in xz-plane and clipping
# =========================================================

def make_heart_crack_polyline_xz(
        z_min: float,
        z_max: float,
        x_amp: float = 0.22,
):
    """
    Construct a crack-like polyline x = x(z) in the xz-plane.

    - z increases from bottom to top
    - x oscillates around 0 for 3–4 turns, mimicking a "crack"
    """
    # Normalized parameter t ∈ [0, 1], 5 key points from bottom to top
    t = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)

    # Base pattern before scaling by x_amp:
    # right -> left -> near center -> more left -> back to right
    pattern = np.array(
        [0.6, -0.3, 0.15, -0.5, 0.4],
        dtype=np.float64,
    )

    poly_z = z_min + t * (z_max - z_min)
    poly_x = x_amp * pattern

    return poly_z.astype(np.float64), poly_x.astype(np.float64)


def clip_by_polyline_xz(
        x_np: np.ndarray,
        poly_z: np.ndarray,
        poly_x: np.ndarray,
        keep: str = "right",
):
    """
    Use a polyline x = x_cut(z) in the xz-plane to split space in two,
    keep only one side.

    Parameters
    ----------
    x_np : (N, 3) ndarray
        Points to be classified.
    poly_z, poly_x : ndarrays
        Polyline vertices in xz-plane, representing x = x(z).
    keep : {"right", "left"}
        - "right": keep points with x >= x_cut(z)
        - "left" : keep points with x <= x_cut(z)

    Returns
    -------
    mask  : (N,) bool ndarray
        True for points that are kept.
    x_cut : (N,) float ndarray
        x_cut(z) for each point's z-coordinate.
    """
    x_coord = x_np[:, 0]
    z_coord = x_np[:, 2]

    # 1D linear interpolation on (poly_z, poly_x)
    x_cut = np.interp(z_coord, poly_z, poly_x,
                      left=poly_x[0], right=poly_x[-1])

    if keep == "right":
        mask = x_coord >= x_cut
    else:
        mask = x_coord <= x_cut

    return mask, x_cut


# =========================================================
# Intersection curve via analytic roots
# =========================================================

def compute_heart_crack_boundary_roots(
        poly_z: np.ndarray,
        poly_x: np.ndarray,
        z_min: float,
        z_max: float,
        n_z: int = 400,
        imag_tol: float = 1e-8,
):
    """
    使用解析求根方式计算折面与心形表面的交线 x_trim_boundary。

    对每个 z 采样点：
        1. 插值得到 x_cut(z)
        2. 将 (x, z) 带入心形方程 F(x, y, z) = 0
        3. 将 y^2 记为 s, 得到关于 s 的三次方程
        4. 求 s 的实根 (s >= 0), 然后取 y = ±sqrt(s)

    注意：这一推导基于 Heart.shape_func 中 scale = 1.0 的形式。
    """
    boundary_pts = []

    # 采样 z
    z_samples = np.linspace(z_min, z_max, n_z, dtype=np.float64)

    # 常数系数（来自展开）
    c3_const = 729.0 / 64.0  # (9/4)^3
    c2_coeff = 243.0 / 16.0  # 3*(9/4)^2
    c1_coeff = 27.0 / 4.0  # 3*(9/4)

    for z in z_samples:
        # 当前 z 的折线 x_cut
        x_cut = float(np.interp(z, poly_z, poly_x,
                                left=poly_x[0], right=poly_x[-1]))

        # B = x^2 + z^2 - 1.0
        B = x_cut * x_cut + z * z - 1.0

        # 三次方程 in s = y^2:
        # c3*s^3 + c2*s^2 + c1*s + c0 = 0
        c3 = c3_const
        c2 = c2_coeff * B
        c1 = c1_coeff * (B * B) - (9.0 / 80.0) * (z ** 3)
        c0 = B ** 3 - (x_cut ** 2) * (z ** 3)

        coeffs = np.array([c3, c2, c1, c0], dtype=np.float64)

        # 求 roots
        roots = np.roots(coeffs)

        for s in roots:
            # 丢掉虚部大的根
            if abs(s.imag) > imag_tol:
                continue
            s_real = float(s.real)
            # 只要 s >= 0
            if s_real < 0.0:
                continue

            y_abs = np.sqrt(s_real)
            if not np.isfinite(y_abs):
                continue

            # 两个对称分支 ±y
            boundary_pts.append([x_cut, y_abs, z])
            boundary_pts.append([x_cut, -y_abs, z])

    if len(boundary_pts) == 0:
        return np.zeros((0, 3), dtype=np.float64)

    boundary_pts = np.asarray(boundary_pts, dtype=np.float64)

    # 可选：去重（简单基于四舍五入）
    boundary_pts_rounded = np.round(boundary_pts, decimals=5)
    _, unique_idx = np.unique(boundary_pts_rounded, axis=0, return_index=True)
    boundary_pts = boundary_pts[sorted(unique_idx)]

    return boundary_pts


# =========================================================
# Visualization helpers
# =========================================================

def visualize_trim_sets(
        x_trim_in: torch.Tensor,
        x_trim_out: torch.Tensor,
        x_trim_boundary: torch.Tensor,
        max_in: int = 40000,
        max_out: int = 40000,
):
    """
    Visualize three point sets in 3D:

        - x_trim_in       : kept half (in)
        - x_trim_out      : clipped half (out)
        - x_trim_boundary : intersection curve between crack surface and heart

    max_in, max_out control subsampling to avoid too many points.
    """
    xin = x_trim_in.detach().cpu().numpy()
    xout = x_trim_out.detach().cpu().numpy()
    xbnd = x_trim_boundary.detach().cpu().numpy()

    # Subsample
    if xin.shape[0] > max_in:
        idx_in = np.random.choice(xin.shape[0], max_in, replace=False)
        xin = xin[idx_in]

    if xout.shape[0] > max_out:
        idx_out = np.random.choice(xout.shape[0], max_out, replace=False)
        xout = xout[idx_out]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # 如需一起看 in/out，可以取消下面两段注释
    # if xin.size > 0:
    #     ax.scatter(
    #         xin[:, 0], xin[:, 1], xin[:, 2],
    #         s=1, alpha=0.4, label="x_trim_in",
    #     )
    # if xout.size > 0:
    #     ax.scatter(
    #         xout[:, 0], xout[:, 1], xout[:, 2],
    #         s=1, alpha=0.2, label="x_trim_out",
    #     )

    # boundary (thicker markers)
    if xbnd.size > 0:
        ax.scatter(
            xbnd[:, 0],
            xbnd[:, 1],
            xbnd[:, 2],
            s=10,
            alpha=0.9,
            label="x_trim_boundary",
        )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Trimmed heart boundary curve")

    # Equal aspect ratio
    pts_list = []
    if xin.size > 0:
        pts_list.append(xin)
    if xout.size > 0:
        pts_list.append(xout)
    if xbnd.size > 0:
        pts_list.append(xbnd)

    if pts_list:
        all_pts = np.vstack(pts_list)
        max_range = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2.0
        mid_x = (all_pts[:, 0].max() + all_pts[:, 0].min()) * 0.5
        mid_y = (all_pts[:, 1].max() + all_pts[:, 1].min()) * 0.5
        mid_z = (all_pts[:, 2].max() + all_pts[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # 从右侧看（沿 +x 方向）
    ax.view_init(elev=0.0, azim=0.0)
    ax.legend()
    plt.tight_layout()
    plt.show()


def visualize_heart_half(
        x_trim_in: torch.Tensor,
        normal_trim_in: torch.Tensor,
        mean_curvature_trim_in: torch.Tensor,
        poly_z: np.ndarray,
        poly_x: np.ndarray,
        max_points_scatter: int = 30000,
        max_points_quiver: int = 2000,
):
    """
    Visualize the kept half of the heart (x_trim_in) with curvature and normals.

    Inputs are already trimmed:
        x_trim_in, normal_trim_in, mean_curvature_trim_in
    poly_z, poly_x are only used to draw the crack polyline in 3D.
    """
    # numpy conversion
    x_np = x_trim_in.detach().cpu().numpy()
    n_np = normal_trim_in.detach().cpu().numpy()
    mc_np = mean_curvature_trim_in.detach().cpu().squeeze(-1).numpy()

    # Keep only finite curvature
    finite_mask = np.isfinite(mc_np)
    x_np = x_np[finite_mask]
    n_np = n_np[finite_mask]
    mc_np = mc_np[finite_mask]

    if x_np.shape[0] == 0:
        print("Warning: no finite-curvature points to visualize.")
        return

    # Subsample for scatter
    if x_np.shape[0] > max_points_scatter:
        idx = np.random.choice(x_np.shape[0], max_points_scatter, replace=False)
        x_s = x_np[idx]
        mc_s = mc_np[idx]
    else:
        x_s = x_np
        mc_s = mc_np

    # Clamp color range
    vmin = np.quantile(mc_s, 0.05)
    vmax = np.quantile(mc_s, 0.95)

    # Use a representative y-plane for drawing the crack polyline
    y0 = float(np.mean(x_s[:, 1]))

    # -----------------------------
    # Figure 1: half heart colored by curvature + crack polyline
    # -----------------------------
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

    # crack polyline at y = y0
    ax.plot(
        poly_x,
        np.full_like(poly_x, y0),
        poly_z,
        linewidth=2.0,
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Broken heart half (colored by mean curvature)")

    # Equal aspect ratio
    max_range = (x_s.max(axis=0) - x_s.min(axis=0)).max() / 2.0
    mid_x = (x_s.max(axis=0)[0] + x_s.min(axis=0)[0]) * 0.5
    mid_y = (x_s.max(axis=0)[1] + x_s.min(axis=0)[1]) * 0.5
    mid_z = (x_s.max(axis=0)[2] + x_s.min(axis=0)[2]) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # 从右侧看
    ax.view_init(elev=0.0, azim=0.0)

    plt.tight_layout()
    plt.show()

    # -----------------------------
    # Figure 2: normals on the half heart
    # -----------------------------
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

    ax2.plot(
        poly_x,
        np.full_like(poly_x, y0),
        poly_z,
        linewidth=2.0,
    )

    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("z")
    ax2.set_title("Normals on broken-heart half (crack cut)")

    max_range = (x_q.max(axis=0) - x_q.min(axis=0)).max() / 2.0
    mid_x = (x_q.max(axis=0)[0] + x_q.min(axis=0)[0]) * 0.5
    mid_y = (x_q.max(axis=0)[1] + x_q.min(axis=0)[1]) * 0.5
    mid_z = (x_q.max(axis=0)[2] + x_q.min(axis=0)[2]) * 0.5
    ax2.set_xlim(mid_x - max_range, mid_x + max_range)
    ax2.set_ylim(mid_y - max_range, mid_y + max_range)
    ax2.set_zlim(mid_z - max_range, mid_z + max_range)

    ax2.view_init(elev=0.0, azim=0.0)

    plt.tight_layout()
    plt.show()


# =========================================================
# Entry
# =========================================================

if __name__ == '__main__':
    # Default device
    if torch.cuda.is_available():
        torch.set_default_device('cuda')
    else:
        torch.set_default_device('cpu')

    domain = Heart(scale=1.0)

    # ----------------------------------------
    # 1) Full sampling: x (heart_in.pth)
    # ----------------------------------------
    x = domain.in_sample(100000)  # (N, 3)

    # SDF + normal + mean curvature
    _, normal, mean_curvature = domain.sdf(
        x, with_normal=True, with_curvature=True
    )

    # Save full heart surface
    torch.save(
        (x, normal, mean_curvature),
        '../../data/heart_in.pth'
    )

    print("Full heart surface:", x.shape, normal.shape, mean_curvature.shape)

    # ----------------------------------------
    # 2) Construct crack polyline and trim sets
    # ----------------------------------------
    x_np = x.detach().cpu().numpy()

    # z-range slightly extended based on sampled points
    z_min = float(x_np[:, 2].min()) - 0.05
    z_max = float(x_np[:, 2].max()) + 0.05

    # Crack polyline in xz-plane
    poly_z, poly_x = make_heart_crack_polyline_xz(
        z_min=z_min,
        z_max=z_max,
        x_amp=0.22,
    )

    # Split by crack: keep "right" side for x_trim_in
    half_mask_np, x_cut_all = clip_by_polyline_xz(
        x_np, poly_z, poly_x, keep="right"
    )
    half_mask = torch.from_numpy(half_mask_np.astype(np.bool_)).to(x.device)

    # Split into in / out
    x_trim_in = x[half_mask]  # (N_in, 3)
    x_trim_out = x[~half_mask]  # (N_out, 3)
    normal_trim_in = normal[half_mask]
    normal_trim_out = normal[~half_mask]
    mean_curvature_trim_in = mean_curvature[half_mask]
    mean_curvature_trim_out = mean_curvature[~half_mask]

    # ----------------------------------------
    # 3) Intersection curve x_trim_boundary via analytic roots
    # ----------------------------------------
    boundary_np = compute_heart_crack_boundary_roots(
        poly_z=poly_z,
        poly_x=poly_x,
        z_min=z_min - 0.2,
        z_max=z_max + 0.2,
        n_z=5000,
        imag_tol=1e-8,
    )

    x_trim_boundary = torch.as_tensor(
        boundary_np,
        device=x.device,
        dtype=x.dtype,
    )

    # Save trim sets (optional)
    torch.save(
        (x_trim_in, x_trim_out, x_trim_boundary),
        '../../data/heart_trim_sets.pth'
    )

    print("Trimmed sets:")
    print("  x_trim_in      :", x_trim_in.shape)
    print("  x_trim_out     :", x_trim_out.shape)
    print("  x_trim_boundary:", x_trim_boundary.shape)

    # ----------------------------------------
    # 4) Visualizations
    # ----------------------------------------
    visualize_trim_sets(x_trim_in, x_trim_out, x_trim_boundary)

    # visualize_heart_half(
    #     x_trim_in,
    #     normal_trim_in,
    #     mean_curvature_trim_in,
    #     poly_z,
    #     poly_x,
    # )
