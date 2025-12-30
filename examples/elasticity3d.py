# -*- coding: utf-8 -*-
"""
Created on 2025/8/15
@author: Yifei Sun
"""
import torch
import pyrfm

# ---------------------------
# 材料参数
# ---------------------------
E = 3.0e7
nu = 0.3
lam = E * nu / ((1 + nu) * (1 - 2 * nu))
mu = E / (2 * (1 + nu))


# ---------------------------
# 可视化（von Mises）
# ---------------------------
class CustomVisulizer(pyrfm.RFMVisualizer3D):
    def __init__(self, model: pyrfm.RFMBase, t=0.0, resolution=(1920, 1080), component_idx=0, view='iso', ref=None):
        super().__init__(model, t, resolution, component_idx, view, ref)

    def compute_field_values(self, pts_hit, hits):
        pts_hit = pts_hit.reshape(-1, 3)

        disp_x = self.model.dForward(pts_hit, order=(1, 0, 0))
        ux, vx, wx = disp_x[:, [0]], disp_x[:, [1]], disp_x[:, [2]]
        disp_y = self.model.dForward(pts_hit, order=(0, 1, 0))
        uy, vy, wy = disp_y[:, [0]], disp_y[:, [1]], disp_y[:, [2]]
        disp_z = self.model.dForward(pts_hit, order=(0, 0, 1))
        uz, vz, wz = disp_z[:, [0]], disp_z[:, [1]], disp_z[:, [2]]

        # 小应变
        exx, eyy, ezz = ux, vy, wz
        exy = 0.5 * (uy + vx)
        exz = 0.5 * (uz + wx)
        eyz = 0.5 * (vz + wy)
        tr_e = exx + eyy + ezz

        # Cauchy 应力
        sxx = lam * tr_e + 2 * mu * exx
        syy = lam * tr_e + 2 * mu * eyy
        szz = lam * tr_e + 2 * mu * ezz
        sxy = 2 * mu * exy
        sxz = 2 * mu * exz
        syz = 2 * mu * eyz

        # von Mises
        vonmises = torch.sqrt(
            0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) + 3 * (sxy ** 2 + sxz ** 2 + syz ** 2)
        )
        return vonmises.detach().cpu().numpy()


# ---------------------------
# 工具函数（导数/牵引矩阵）
# ---------------------------
def second_derivs(model, x):
    """返回 u 组件基的二阶导张量集合，并复用为 v/w 的基"""
    uxx = model.features_second_derivative(x, axis1=0, axis2=0).cat(dim=1)
    uyy = model.features_second_derivative(x, axis1=1, axis2=1).cat(dim=1)
    uzz = model.features_second_derivative(x, axis1=2, axis2=2).cat(dim=1)
    uxy = model.features_second_derivative(x, axis1=0, axis2=1).cat(dim=1)
    uxz = model.features_second_derivative(x, axis1=0, axis2=2).cat(dim=1)
    uyz = model.features_second_derivative(x, axis1=1, axis2=2).cat(dim=1)
    # 共享基：v,w 与 u 共用
    return (uxx, uyy, uzz, uxy, uxz, uyz)


def first_derivs(model, pts):
    """返回 (ux,uy,uz) 并复用为 (vx,vy,vz),(wx,wy,wz)"""
    ux = model.features_derivative(pts, axis=0).cat(dim=1)
    uy = model.features_derivative(pts, axis=1).cat(dim=1)
    uz = model.features_derivative(pts, axis=2).cat(dim=1)
    return ux, uy, uz


def traction_blocks_from_derivs(ux, uy, uz, nx, ny, nz):
    """根据一阶导与法向，返回三行(对应 t_x, t_y, t_z)的 3×3 block 系数"""
    vx, vy, vz = ux, uy, uz
    wx, wy, wz = ux, uy, uz

    # t_x
    Tx_u = (lam + 2 * mu) * ux * nx + mu * uy * ny + mu * uz * nz
    Tx_v = lam * vy * nx + mu * vx * ny
    Tx_w = lam * wz * nx + mu * wx * nz

    # t_y
    Ty_u = mu * uy * nx + lam * ux * ny
    Ty_v = mu * vx * nx + (lam + 2 * mu) * vy * ny + mu * vz * nz
    Ty_w = lam * wz * ny + mu * wy * nz

    # t_z
    Tz_u = mu * uz * nx + lam * ux * nz
    Tz_v = lam * vy * nz + mu * vz * ny
    Tz_w = (lam + 2 * mu) * wz * nz + mu * wx * nx + mu * wy * ny

    return (Tx_u, Tx_v, Tx_w), (Ty_u, Ty_v, Ty_w), (Tz_u, Tz_v, Tz_w)


# ---------------------------
# 主流程
# ---------------------------
if __name__ == "__main__":
    torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- 几何 ---
    base = pyrfm.Square2D(center=[2.0, 1.5], half=[0.5, 0.5])

    # 这些圆是要在 base 内保留下来的部分
    region1 = pyrfm.Circle2D(center=[1.9575995206832886, 1.2456529587507248], radius=0.05024437326937914)
    region2 = pyrfm.Circle2D(center=[2.1249395608901978, 1.5534365177154540], radius=0.27430968359112740)
    region3 = pyrfm.Circle2D(center=[1.9882896542549133, 1.5191256999969482], radius=0.26661740615963936)

    # 这些圆是要从结果里挖掉的部分
    cut1 = pyrfm.Circle2D(center=[2.1955102682113647, 1.7093743383884430], radius=0.14129653573036194)
    cut2 = pyrfm.Circle2D(center=[2.0465531945228577, 1.0623437911272050], radius=0.15957007184624672)
    cut3 = pyrfm.Circle2D(center=[1.7482689023017883, 1.4980989694595337], radius=0.19152732565999030)
    cut4 = pyrfm.Circle2D(center=[1.9530203938484192, 1.6899727284908295], radius=0.09197942912578583)

    base2d = pyrfm.IntersectionGeometry(base, region1 + region2 + region3) - (cut1 + cut2 + cut3 + cut4)

    domain = pyrfm.ExtrudeBody(base2d=base2d, direction=(1.0, 0.0, 0.0))
    del base2d

    # 采样
    x_in = domain.in_sample(num_samples=1000)
    (top_pts, n_top), (bot_pts, n_bot), (pts_side, side_normals) = domain.on_sample(
        num_samples=2000, separate=True, with_normal=True
    )
    print(f"#in: {x_in.shape[0]}  |  #top/bot/side: {n_top.shape[0]}, {n_bot.shape[0]}, {pts_side.shape[0]}")

    # 模型
    model = pyrfm.RFMBase(dim=3, n_hidden=1000, domain=domain, n_subdomains=1)
    del domain

    # ---------------------------
    # (1) 内点方程 A1
    # ---------------------------
    uxx, uyy, uzz, uxy, uxz, uyz = second_derivs(model, x_in)
    vxx, vyy, vzz, vxy, vxz, vyz = uxx, uyy, uzz, uxy, uxz, uyz
    wxx, wyy, wzz, wxy, wxz, wyz = uxx, uyy, uzz, uxy, uxz, uyz

    A1 = pyrfm.concat_blocks([
        [(lam + 2 * mu) * uxx + mu * uyy + mu * uzz, (lam + mu) * vxy, (lam + mu) * wxz],
        [(lam + mu) * uxy, (lam + 2 * mu) * vyy + mu * vxx + mu * vzz, (lam + mu) * wyz],
        [(lam + mu) * uxz, (lam + mu) * vyz, (lam + 2 * mu) * wzz + mu * wxx + mu * wyy]
    ])
    b1 = torch.zeros(A1.shape[0], 1)

    # 清理二阶导中间引用
    del uxx, uyy, uzz, uxy, uxz, uyz, vxx, vyy, vzz, vxy, vxz, vyz, wxx, wyy, wzz, wxy, wyz, wxz

    # ---------------------------
    # (2) 侧面零牵引 A2
    # ---------------------------
    ux_s, uy_s, uz_s = first_derivs(model, pts_side)
    nx_s, ny_s, nz_s = side_normals[:, [0]], side_normals[:, [1]], side_normals[:, [2]]
    (Tx_u, Tx_v, Tx_w), (Ty_u, Ty_v, Ty_w), (Tz_u, Tz_v, Tz_w) = traction_blocks_from_derivs(
        ux_s, uy_s, uz_s, nx_s, ny_s, nz_s
    )
    A2 = pyrfm.concat_blocks([
        [Tx_u, Tx_v, Tx_w],
        [Ty_u, Ty_v, Ty_w],
        [Tz_u, Tz_v, Tz_w]
    ])
    b2 = torch.zeros(A2.shape[0], 1)

    del ux_s, uy_s, uz_s, nx_s, ny_s, nz_s, Tx_u, Tx_v, Tx_w, Ty_u, Ty_v, Ty_w, Tz_u, Tz_v, Tz_w
    del side_normals, pts_side

    # ---------------------------
    # (3) 顶面：v = -0.1；且 u,w 方向继续用牵引 (t_x=0, t_z=0)
    # ---------------------------
    # 位移块（仅 v）
    v_top = model.features(top_pts).cat(dim=1)

    # 顶面牵引块
    ux_t, uy_t, uz_t = first_derivs(model, top_pts)
    nx_t, ny_t, nz_t = n_top[:, [0]], n_top[:, [1]], n_top[:, [2]]
    (Tx_u, Tx_v, Tx_w), (_, _, _), (Tz_u, Tz_v, Tz_w) = traction_blocks_from_derivs(
        ux_t, uy_t, uz_t, nx_t, ny_t, nz_t
    )
    A3 = pyrfm.concat_blocks([
        [Tx_u, Tx_v, Tx_w],  # t_x = 0
        [torch.zeros_like(v_top), v_top, torch.zeros_like(v_top)],  # v   = -0.1
        [Tz_u, Tz_v, Tz_w]  # t_z = 0
    ])
    b3 = torch.cat([
        torch.zeros(top_pts.shape[0], 1),
        -0.1 * torch.ones(top_pts.shape[0], 1),
        torch.zeros(top_pts.shape[0], 1)
    ], dim=0)

    del ux_t, uy_t, uz_t, nx_t, ny_t, nz_t, Tx_u, Tx_v, Tx_w, Tz_u, Tz_v, Tz_w
    del n_top

    # ---------------------------
    # (4) 底面全固定 A4
    # ---------------------------
    u_bot = model.features(bot_pts).cat(dim=1)
    v_bot, w_bot = u_bot, u_bot
    A4 = pyrfm.concat_blocks([
        [u_bot, torch.zeros_like(v_bot), torch.zeros_like(w_bot)],
        [torch.zeros_like(u_bot), v_bot, torch.zeros_like(w_bot)],
        [torch.zeros_like(u_bot), torch.zeros_like(v_bot), w_bot]
    ])
    b4 = torch.zeros(A4.shape[0], 1)

    del bot_pts, n_bot  # 几何法向未再用

    # ---------------------------
    # 总装并求解
    # ---------------------------
    A = torch.cat([A1, A2, A3, A4], dim=0)
    b = torch.cat([b1, b2, b3, b4], dim=0)
    del A1, A2, A3, A4, b1, b2, b3, b4

    model.compute(A).solve(b)
    del A, b

    # 可选：CUDA 释放未引用显存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---------------------------
    # 可视化
    # ---------------------------
    vis = CustomVisulizer(model, view="iso")
    vis.plot()
    vis.show()
    vis.savefig("elasticity3d_vonmises.png", dpi=600)
