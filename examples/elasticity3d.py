# -*- coding: utf-8 -*-
"""
Created on 2025/8/15

@author: Yifei Sun
"""
import time

import pyrfm
import torch
import os

"""
A square (1.5, 2.5) × (1.0, 2.0) with three removed and four filled circular holes.

Removed
(1.9575995206832886, 1.2456529587507248) 0.05024437326937914
(2.1249395608901978, 1.5534365177154540) 0.27430968359112740
(1.9882896542549133, 1.5191256999969482) 0.26661740615963936
Filled
(2.1955102682113647, 1.7093743383884430) 0.14129653573036194
(2.0465531945228577, 1.0623437911272050) 0.15957007184624672
(1.7482689023017883, 1.4980989694595337) 0.19152732565999030
(1.9530203938484192, 1.6899727284908295) 0.09197942912578583
"""

E = 3.0e7  # Young's modulus
nu = 0.3  # Poisson's ratio
lam = E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame's first parameter
mu = E / (2 * (1 + nu))  # Lame's second parameter


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
        exx = ux
        eyy = vy
        ezz = wz
        exy = 0.5 * (uy + vx)
        exz = 0.5 * (uz + wx)
        eyz = 0.5 * (vz + wy)

        tr_e = exx + eyy + ezz

        # Cauchy应力（Hooke）
        sxx = lam * tr_e + 2 * mu * exx
        syy = lam * tr_e + 2 * mu * eyy
        szz = lam * tr_e + 2 * mu * ezz
        sxy = 2 * mu * exy  # = mu*(u_y + v_x)
        sxz = 2 * mu * exz  # = mu*(u_z + w_x)
        syz = 2 * mu * eyz  # = mu*(v_z + w_y)

        # von Mises
        vonmises = torch.sqrt(
            0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2) + 3 * (sxy ** 2 + sxz ** 2 + syz ** 2)
        )
        field_vals = vonmises.detach().cpu().numpy()
        return field_vals


if __name__ == "__main__":
    torch.set_default_device('cuda') if torch.cuda.is_available() else torch.set_default_device('cpu')
    base = pyrfm.Square2D(center=[2.0, 1.5], radius=[0.5, 0.5])
    hole1 = pyrfm.Circle2D(center=[1.9575995206832886, 1.2456529587507248], radius=0.05024437326937914)
    hole2 = pyrfm.Circle2D(center=[2.1249395608901978, 1.5534365177154540], radius=0.27430968359112740)
    hole3 = pyrfm.Circle2D(center=[1.9882896542549133, 1.5191256999969482], radius=0.26661740615963936)
    filled1 = pyrfm.Circle2D(center=[2.1955102682113647, 1.7093743383884430], radius=0.14129653573036194)
    filled2 = pyrfm.Circle2D(center=[2.0465531945228577, 1.0623437911272050], radius=0.15957007184624672)
    filled3 = pyrfm.Circle2D(center=[1.7482689023017883, 1.4980989694595337], radius=0.19152732565999030)
    filled4 = pyrfm.Circle2D(center=[1.9530203938484192, 1.6899727284908295], radius=0.09197942912578583)
    base2d = pyrfm.Square2D(center=[2.0, 1.5], radius=[0.49, 0.49]) - (
            base - hole1 - hole2 - hole3 + filled1 + filled2 + filled3 + filled4)

    domain = pyrfm.ExtrudeBody(base2d=base2d, direction=(1.0, 0.0, 0.0))

    x_in = domain.in_sample(num_samples=4000)
    (top_pts, n_top), (bot_pts, n_bot), (pts_side, side_normals) = domain.on_sample(num_samples=2000, separate=True,
                                                                                    with_normal=True)
    print(f"Number of input samples: {x_in.shape[0]}")
    print(f"Number of output samples: {n_top.shape[0]}, {n_bot.shape[0]}, {pts_side.shape[0]}")
    print("Shape of side normals:", side_normals.shape, pts_side.shape)

    model = pyrfm.RFMBase(dim=3, n_hidden=400, domain=domain, n_subdomains=1)

    uxx = model.features_second_derivative(x_in, axis1=0, axis2=0).cat(dim=1)
    uyy = model.features_second_derivative(x_in, axis1=1, axis2=1).cat(dim=1)
    uzz = model.features_second_derivative(x_in, axis1=2, axis2=2).cat(dim=1)
    uxy = model.features_second_derivative(x_in, axis1=0, axis2=1).cat(dim=1)
    uxz = model.features_second_derivative(x_in, axis1=0, axis2=2).cat(dim=1)
    uyz = model.features_second_derivative(x_in, axis1=1, axis2=2).cat(dim=1)
    # multiple components can share same basis
    vxx, vyy, vzz, vxy, vxz, vyz = uxx, uyy, uzz, uxy, uxz, uyz
    wxx, wyy, wzz, wxy, wxz, wyz = uxx, uyy, uzz, uxy, uxz, uyz

    """
    (λ+2μ) uxx + μ(uyy + uzz) + (λ+μ)(vxy)                + (λ+μ)(wxz) +                bx = 0
    (λ+μ)(uxy)                + (λ+2μ) vyy + μ(vxx + vzz) + (λ+μ)(wyz) +                by = 0
    (λ+μ)(uxz)                + (λ+μ)(vyz)                + (λ+2μ) wzz + μ(wxx + wyy) + bz = 0
    """

    A1 = pyrfm.concat_blocks([[(lam + 2 * mu) * uxx + mu * uyy + mu * uzz, (lam + mu) * vxy, (lam + mu) * wxz],
                              [(lam + mu) * uxy, (lam + 2 * mu) * vyy + mu * vxx + mu * vzz, (lam + mu) * wyz],
                              [(lam + mu) * uxz, (lam + mu) * vyz, (lam + 2 * mu) * wzz + mu * wxx + mu * wyy]])
    b1 = torch.zeros(A1.shape[0], 1)

    del uxx, uyy, uzz, uxy, uxz, uyz, vxx, vyy, vzz, vxy, vxz, vyz, wxx, wyy, wzz, wxy, wyz, wxz

    """
    t_x = σ_xx n_x + σ_xy n_y + σ_xz n_z
        = (λ+2μ) u_x*nx + μ u_y*ny + μ u_z*nz + λ v_y*nx + μ v_x*ny + λ w_z*nx + μ w_x*nz
    t_y = σ_yx n_x + σ_yy n_y + σ_yz n_z
        = μ u_y*nx + λ u_x*ny + μ v_x*nx + (λ+2μ) v_y*ny + μ v_z*nz + λ w_z*ny + μ w_y*nz
    t_z = σ_zx n_x + σ_zy n_y + σ_zz n_z
        = μ u_z*nx + λ u_x*nz + μ v_z*ny + λ v_y*nz + μ w_x*nx + μ w_y*ny + (λ+2μ) w_z*nz
    """

    ux = model.features_derivative(pts_side, axis=0).cat(dim=1)
    uy = model.features_derivative(pts_side, axis=1).cat(dim=1)
    uz = model.features_derivative(pts_side, axis=2).cat(dim=1)
    vx, vy, vz = ux, uy, uz
    wx, wy, wz = ux, uy, uz
    nx, ny, nz = side_normals[:, [0]], side_normals[:, [1]], side_normals[:, [2]]
    A2 = pyrfm.concat_blocks([
        # t_x
        [(lam + 2 * mu) * ux * nx + mu * uy * ny + mu * uz * nz,
         lam * vy * nx + mu * vx * ny,
         lam * wz * nx + mu * wx * nz],
        # t_y
        [mu * uy * nx + lam * ux * ny,
         mu * vx * nx + (lam + 2 * mu) * vy * ny + mu * vz * nz,
         lam * wz * ny + mu * wy * nz],
        # t_z
        [mu * uz * nx + lam * ux * nz,
         lam * vy * nz + mu * vz * ny,
         (lam + 2 * mu) * wz * nz + mu * wx * nx + mu * wy * ny]
    ])
    b2 = torch.zeros(A2.shape[0], 1)

    del ux, uy, uz, vx, vy, vz, wx, wy, wz, nx, ny, nz

    u_top = model.features(top_pts).cat(dim=1)
    v_top, w_top = u_top, u_top
    A3 = pyrfm.concat_blocks([[u_top, torch.zeros_like(v_top), torch.zeros_like(w_top)],
                              [torch.zeros_like(u_top), v_top, torch.zeros_like(w_top)],
                              [torch.zeros_like(u_top), torch.zeros_like(v_top), w_top]])
    b3 = torch.cat([torch.zeros(top_pts.shape[0], 1),
                    -0.1 * torch.ones(top_pts.shape[0], 1),
                    torch.zeros(top_pts.shape[0], 1)], dim=0)

    u_bot = model.features(bot_pts).cat(dim=1)
    v_bot, w_bot = u_bot, u_bot
    A4 = pyrfm.concat_blocks([[u_bot, torch.zeros_like(v_bot), torch.zeros_like(w_bot)],
                              [torch.zeros_like(u_bot), v_bot, torch.zeros_like(w_bot)],
                              [torch.zeros_like(u_bot), torch.zeros_like(v_bot), w_bot]])
    b4 = torch.zeros(A3.shape[0], 1)

    A = torch.cat([A1, A2, A3, A4], dim=0)
    b = torch.cat([b1, b2, b3, b4], dim=0)

    model.compute(A).solve(b)

    # visulizer = pyrfm.RFMVisualizer3D(model, view="iso", component_idx=2)
    visulizer = CustomVisulizer(model, view="iso", component_idx=2)
    visulizer.plot()
    visulizer.show()
