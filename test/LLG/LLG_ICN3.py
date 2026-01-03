# -*- coding: utf-8 -*-
"""
Created on 2025/12/30

@author: Yifei Sun
"""

import pyrfm
import torch
import os
import argparse
import sys
import time

"""
Consider the Landau-Lifshitz-Gilbert equation with homogeneous Neumann boundary condition:

In the 1D model, we choose 
𝑚⃗ₑ=(cos(𝑥̄) sin(t), sin(𝑥̄) sin(t), cos(t)) 
with 𝑥̄ = 𝑥²(1−𝑥)² as the exact solution over Ω=[0,1], which satisfies

𝑚⃗ₜ = −𝑚⃗ × 𝑚⃗ₓₓ − α 𝑚⃗ × (𝑚⃗ × 𝑚⃗ₓₓ) + 𝑔⃗

The forcing term turns out to be 
𝑔⃗ = 𝑚⃗ₑₜ + 𝑚⃗ₑ × 𝑚⃗ₑₓₓ + α 𝑚⃗ₑ × (𝑚⃗ₑ × 𝑚⃗ₑₓₓ), 
and 𝑚⃗ₑ satisfies the homogeneous Neumann boundary condition.

In the 3D model, we set the exact solution as

𝑚⃗ₑ=(cos(𝑥̄ 𝑦̄ 𝑧̄) sin(t), sin(𝑥̄ 𝑦̄ 𝑧̄) sin(t), cos(t))

over Ω=[0,1]³, which satisfies the homogeneous Neumann boundary condition 
and the following equation is valid:

𝑚ₜ = −𝑚 × Δ𝑚 − α 𝑚 × (𝑚 × Δ𝑚) + 𝑔

with 𝑥̄=𝑥²(1−𝑥)², 𝑦̄=𝑦²(1−𝑦)², 𝑧̄=𝑧²(1−𝑧)² and 

𝑔⃗ = 𝑚⃗ₑₜ + 𝑚⃗ₑ × Δ𝑚⃗ₑ + α 𝑚⃗ₑ × (𝑚⃗ₑ × Δ𝑚⃗ₑ)

ICN method is used to solve the equation.

∂rⁿ⁺¹_BC/∂cⁿ⁺¹ = ∂ₙΦ
∂rⁿ⁺¹_PDE/∂cⁿ⁺¹ 
= Φ/Δt + 1/2 ([uⁿ⁺¹ᐟ²]ₓΔΦ − [Δuⁿ⁺¹ᐟ²]ₓΦ) + α/2 (−[uⁿ⁺¹ᐟ²×Δuⁿ⁺¹ᐟ²]ₓΦ + [uⁿ⁺¹ᐟ²]ₓ²ΔΦ − [uⁿ⁺¹ᐟ²]ₓ[Δuⁿ⁺¹ᐟ²]ₓΦ)




"""


def func_m(xt, dim):
    if dim == 1:
        x = xt[:, [0]]
        t = xt[:, [-1]]
        xbar = x ** 2 * (1 - x) ** 2
        return torch.cat((torch.cos(xbar) * torch.sin(t),
                          torch.sin(xbar) * torch.sin(t),
                          torch.cos(t)), dim=1)
    elif dim == 3:
        x = xt[:, [0]]
        y = xt[:, [1]]
        z = xt[:, [2]]
        t = xt[:, [-1]]
        xbar = x ** 2 * (1 - x) ** 2
        ybar = y ** 2 * (1 - y) ** 2
        zbar = z ** 2 * (1 - z) ** 2
        return torch.cat((torch.cos(xbar * ybar * zbar) * torch.sin(t),
                          torch.sin(xbar * ybar * zbar) * torch.sin(t),
                          torch.cos(t)), dim=1)
    else:
        raise ValueError("The dimension should be 1 or 3.")


def func_g(xt, dim, alpha):
    if dim == 1:
        x = xt[:, [0]]
        t = xt[:, [-1]]
        xbar = x ** 2 * (1 - x) ** 2
        sin_xbar = torch.sin(xbar)
        cos_xbar = torch.cos(xbar)
        sin_t = torch.sin(t)
        common_term = 2 * x ** 2 * (x - 1) ** 2 * (2 * x - 1) ** 2
        additional_term = x ** 2 + 4 * x * (x - 1) + (x - 1) ** 2

        m = func_m(xt, dim)
        mxx = torch.cat([
            -2 * (common_term * cos_xbar + additional_term * sin_xbar) * sin_t,
            2 * (-common_term * sin_xbar + additional_term * cos_xbar) * sin_t,
            torch.zeros_like(x)
        ], dim=1)
        mt = torch.cat([torch.cos(xbar) * torch.cos(t),
                        torch.sin(xbar) * torch.cos(t),
                        -torch.sin(t)], dim=1)
        return mt + torch.cross(m, mxx, dim=1) + alpha * torch.cross(m, torch.cross(m, mxx, dim=1), dim=1)

    elif dim == 3:
        x = xt[:, [0]]
        y = xt[:, [1]]
        z = xt[:, [2]]
        t = xt[:, [-1]]
        xbar = x ** 2 * (1 - x) ** 2
        ybar = y ** 2 * (1 - y) ** 2
        zbar = z ** 2 * (1 - z) ** 2

        common_term = 2 * x ** 2 * y ** 2 * z ** 2 * (x - 1) ** 2 * (y - 1) ** 2 * (z - 1) ** 2
        additional_term_x = x ** 2 + 4 * x * (x - 1) + (x - 1) ** 2
        additional_term_y = y ** 2 + 4 * y * (y - 1) + (y - 1) ** 2
        additional_term_z = z ** 2 + 4 * z * (z - 1) + (z - 1) ** 2

        cos_term = torch.cos(xbar * ybar * zbar)
        sin_term = torch.sin(xbar * ybar * zbar)

        m = func_m(xt, dim)
        mt = torch.cat([cos_term * torch.cos(t),
                        sin_term * torch.cos(t),
                        -torch.sin(t)], dim=1)

        mxx = torch.cat([
            -2 * ybar * zbar * (
                    common_term * (2 * x - 1) ** 2 * cos_term + additional_term_x * sin_term) * torch.sin(t),
            2 * ybar * zbar * (
                    -common_term * (2 * x - 1) ** 2 * sin_term + additional_term_x * cos_term) * torch.sin(t),
            torch.zeros_like(x)
        ], dim=1)

        myy = torch.cat([
            -2 * xbar * zbar * (
                    common_term * (2 * y - 1) ** 2 * cos_term + additional_term_y * sin_term) * torch.sin(t),
            2 * xbar * zbar * (
                    -common_term * (2 * y - 1) ** 2 * sin_term + additional_term_y * cos_term) * torch.sin(t),
            torch.zeros_like(x)
        ], dim=1)

        mzz = torch.cat([
            -2 * xbar * ybar * (
                    common_term * (2 * z - 1) ** 2 * cos_term + additional_term_z * sin_term) * torch.sin(t),
            2 * xbar * ybar * (
                    -common_term * (2 * z - 1) ** 2 * sin_term + additional_term_z * cos_term) * torch.sin(t),
            torch.zeros_like(x)
        ], dim=1)

        mla = mxx + myy + mzz
        return mt + torch.cross(m, mla, dim=1) + alpha * torch.cross(m, torch.cross(m, mla, dim=1), dim=1)

    else:
        raise ValueError("The dimension should be 1 or 3.")


param_sets_groups = [
    # [
    # {"Nx": 1, "Nt": 1, "Qx": 10 ** 3, "Qt": 10, "Jn": 300, "Nb": 1, "type": "STC", "alpha": 0.00001, "T": 1e-2},
    #     # {"Nx": 2, "Nt": 2, "Qx": 20, "Qt": 20, "Jn": 100, "Nb": 2, "type": "STC", "alpha": 0.1, "T": 1.0},
    #     # {"Nx": 2, "Nt": 2, "Qx": 20, "Qt": 20, "Jn": 100, "Nb": 3, "type": "STC"},
    #     # {"Nx": 2, "Nt": 2, "Qx": 20, "Qt": 20, "Jn": 100, "Nb": 4, "type": "STC"},
    #     # {"Nx": 2, "Nt": 2, "Qx": 20, "Qt": 20, "Jn": 100, "Nb": 5, "type": "STC"}
    # ],
    [
        {"Nx": 1, "Nt": 1, "Qx": 20 ** 3, "Qt": 10, "Jn": 300, "Nb": 1, "type": "STC", "alpha": 0.00001, "T": 1.0},
        {"Nx": 1, "Nt": 1, "Qx": 20 ** 3, "Qt": 20, "Jn": 300, "Nb": 1, "type": "STC", "alpha": 0.00001, "T": 1.0},
        {"Nx": 1, "Nt": 1, "Qx": 20 ** 3, "Qt": 30, "Jn": 300, "Nb": 1, "type": "STC", "alpha": 0.00001, "T": 1.0},
        {"Nx": 1, "Nt": 1, "Qx": 20 ** 3, "Qt": 40, "Jn": 300, "Nb": 1, "type": "STC", "alpha": 0.00001, "T": 1.0}
    ],
    [
        {"Nx": 1, "Nt": 1, "Qx": 10 ** 3, "Qt": 100, "Jn": 100, "Nb": 1, "type": "STC", "alpha": 0.00001,
         "T": 1e-1},
        {"Nx": 1, "Nt": 1, "Qx": 12 ** 3, "Qt": 100, "Jn": int(100 * 1.2 ** 3), "Nb": 1, "type": "STC",
         "alpha": 0.00001,
         "T": 1e-1},
        {"Nx": 1, "Nt": 1, "Qx": 14 ** 3, "Qt": 100, "Jn": int(100 * 1.4 ** 3), "Nb": 1, "type": "STC",
         "alpha": 0.00001,
         "T": 1e-1},
        {"Nx": 1, "Nt": 1, "Qx": 16 ** 3, "Qt": 100, "Jn": int(100 * 1.6 ** 3), "Nb": 1, "type": "STC",
         "alpha": 0.00001,
         "T": 1e-1},
    ]
]

group_labels = [
    # "Convergence",
    "Convergence with respect to temporal resolution",
    "Convergence with respect to spatial resolution"
]


def run_rfm(args):
    t_end = args.T
    time_stamp = torch.linspace(0, t_end, args.Nb + 1)
    domain = pyrfm.Cube3D(center=[0.5, 0.5, 0.5], half=[0.5, 0.5, 0.5])

    x_in = domain.in_sample(args.Qx * args.Nx ** 3, with_boundary=False)
    x_test = domain.in_sample(args.Qx * args.Nx ** 3, with_boundary=True)
    x_on, x_on_normal = domain.on_sample(6 * int((args.Qx * args.Nx ** 3) ** (2 / 3)), with_normal=True)

    # print(x_on)

    t0 = 0.0
    dt = args.T / (args.Nt * args.Qt)
    n_steps = args.Nt * args.Qt

    def cross(a0, a1, a2, b0, b1, b2):
        return a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0

    model = pyrfm.RFMBase(dim=3, n_hidden=args.Jn, domain=domain)

    u_test = model.features(x_test).cat(dim=1)

    u_in = model.features(x_in).cat(dim=1)
    u_in_xx = model.features_second_derivative(x_in, axis1=0, axis2=0).cat(dim=1)
    u_in_yy = model.features_second_derivative(x_in, axis1=1, axis2=1).cat(dim=1)
    u_in_zz = model.features_second_derivative(x_in, axis1=2, axis2=2).cat(dim=1)
    u_in_xx += u_in_yy
    u_in_xx += u_in_zz

    u_on_x = model.features_derivative(x_on, axis=0).cat(dim=1)
    u_on_y = model.features_derivative(x_on, axis=1).cat(dim=1)
    u_on_z = model.features_derivative(x_on, axis=2).cat(dim=1)
    u_on_n = u_on_x * x_on_normal[:, [0]] + u_on_y * x_on_normal[:, [1]] + u_on_z * x_on_normal[:, [2]]

    v_test, w_test = u_test, u_test
    v_in, w_in = u_in, u_in
    v_in_xx, w_in_xx = u_in_xx, u_in_xx
    v_on_n, w_on_n = u_on_n, u_on_n

    # w_k = torch.zeros((args.Jn, 3))
    w_k_minus_1 = torch.zeros((args.Jn, 3))
    A_test = None

    # (2Φ/Δt + [ûⁿ⁺¹ᐟ²]×ΔΦ + α[ûⁿ⁺¹ᐟ²]×²ΔΦ) c = 2uⁿ/Δt − ûⁿ⁺¹ᐟ² × (Δuⁿ + 2f̂ⁿ⁺¹ᐟ²) − α ûⁿ⁺¹ᐟ² × (ûⁿ⁺¹ᐟ² × (Δuⁿ + 2f̂ⁿ⁺¹ᐟ²))
    for k in range(n_steps):
        if k == 0:
            A_test = pyrfm.concat_blocks([[u_test, torch.zeros_like(v_test), torch.zeros_like(w_test)],
                                          [torch.zeros_like(u_test), v_test, torch.zeros_like(w_test)],
                                          [torch.zeros_like(u_test), torch.zeros_like(v_test), w_test]])
            m0 = func_m(torch.cat([x_test, (t0 + k * dt) * torch.ones((x_test.shape[0], 1))], dim=1), dim=3)
            b = torch.cat([m0[:, [0]], m0[:, [1]], m0[:, [2]]], dim=0)
            model.compute(A_test.clone(), verbose=False).solve(b, verbose=False)
            m_pred = model(x_test)
            m_pred /= torch.linalg.norm(m_pred, dim=1, keepdim=True)
            m_exact = func_m(torch.cat([x_test, (t0 + k * dt) * torch.ones((x_test.shape[0], 1))], dim=1), dim=3)
            error = torch.norm(m_pred - m_exact) / torch.norm(m_exact)
            # print(f"Step {k}/{n_steps}, Time {t0 + k * dt:.4f}, Error: {error:.4e}")

            # plot m_pred[:, [0]], m_pred[:, [1]], m_pred[:, [2]]
            # import matplotlib.pyplot as plt
            #
            # x = x_test.cpu().numpy()
            #
            # plt.figure(figsize=(8, 6))
            #
            # # ---------------- RFM：空心 marker（上层） ----------------
            # plt.plot(
            #     x, m_pred[:, 0].cpu().numpy(),
            #     label='RFM m_x',
            #     linestyle='None',
            #     marker='o',
            #     markersize=4,
            #     markevery=10,
            #     markerfacecolor='none',
            #     markeredgecolor='r',
            #     markeredgewidth=1.0,
            #     zorder=3
            # )
            #
            # plt.plot(
            #     x, m_pred[:, 1].cpu().numpy(),
            #     label='RFM m_y',
            #     linestyle='None',
            #     marker='s',
            #     markersize=4,
            #     markevery=10,
            #     markerfacecolor='none',
            #     markeredgecolor='y',
            #     markeredgewidth=1.0,
            #     zorder=3
            # )
            #
            # plt.plot(
            #     x, m_pred[:, 2].cpu().numpy(),
            #     label='RFM m_z',
            #     linestyle='None',
            #     marker='^',
            #     markersize=4,
            #     markevery=10,
            #     markerfacecolor='none',
            #     markeredgecolor='g',
            #     markeredgewidth=1.0,
            #     zorder=3
            # )
            #
            # # ---------------- Exact：虚线（不同 dash pattern） ----------------
            # plt.plot(
            #     x, m_exact[:, 0].cpu().numpy(),
            #     label='Exact m_x',
            #     color='r',
            #     linewidth=1.4,
            #     alpha=0.6,
            #     linestyle=(0, (6, 2)),  # 长虚线
            #     zorder=1
            # )
            #
            # plt.plot(
            #     x, m_exact[:, 1].cpu().numpy(),
            #     label='Exact m_y',
            #     color='y',
            #     linewidth=1.4,
            #     alpha=0.6,
            #     linestyle=(0, (3, 2)),  # 中等虚线
            #     zorder=1
            # )
            #
            # plt.plot(
            #     x, m_exact[:, 2].cpu().numpy(),
            #     label='Exact m_z',
            #     color='g',
            #     linewidth=1.4,
            #     alpha=0.6,
            #     linestyle=(0, (1, 2)),  # 点虚线
            #     zorder=1
            # )
            #
            # plt.legend()
            # plt.tight_layout()
            # plt.show()
            w_k_minus_1 = model.W.clone()

        else:
            w_k_minus_1 = model.W.clone()
            xt_in = torch.cat([x_in, (t0 + (k - 0.5) * dt) * torch.ones((x_in.shape[0], 1))], dim=1)
            g_in = func_g(xt_in, dim=3, alpha=args.alpha)

            m_k_minus_1 = u_in @ w_k_minus_1
            m_k_xx_minus_1 = u_in_xx @ w_k_minus_1

            jac_u = pyrfm.concat_blocks([[u_in, torch.zeros_like(v_in), torch.zeros_like(w_in)]])
            jac_v = pyrfm.concat_blocks([[torch.zeros_like(u_in), v_in, torch.zeros_like(w_in)]])
            jac_w = pyrfm.concat_blocks([[torch.zeros_like(u_in), torch.zeros_like(v_in), w_in]])

            jac_uxx = pyrfm.concat_blocks([[u_in_xx, torch.zeros_like(v_in_xx), torch.zeros_like(w_in_xx)]])
            jac_vxx = pyrfm.concat_blocks([[torch.zeros_like(u_in_xx), v_in_xx, torch.zeros_like(w_in_xx)]])
            jac_wxx = pyrfm.concat_blocks([[torch.zeros_like(u_in_xx), torch.zeros_like(v_in_xx), w_in_xx]])

            jac_u_n = pyrfm.concat_blocks([[u_on_n, torch.zeros_like(v_on_n), torch.zeros_like(w_on_n)]])
            jac_v_n = pyrfm.concat_blocks([[torch.zeros_like(u_on_n), v_on_n, torch.zeros_like(w_on_n)]])
            jac_w_n = pyrfm.concat_blocks([[torch.zeros_like(u_on_n), torch.zeros_like(v_on_n), w_on_n]])

            def fcn(w):
                # 𝐫ⁿ⁺¹_PDE = Φ(cⁿ⁺¹ − cⁿ)/Δt + (Φ cⁿ⁺½) × (ΔΦ cⁿ⁺½) + α (Φ cⁿ⁺½) × ((Φ cⁿ⁺½) × (ΔΦ cⁿ⁺½)) − 𝐟ⁿ⁺½
                # 𝐫ⁿ⁺¹_BC = ∂ₙ 𝐮ⁿ⁺¹ = (∂ₙ Φ) cⁿ⁺¹
                w = w.reshape(3, -1).T  # reshape w to a torch tensor of shape (N, 3)
                r1 = u_in @ (w - w_k_minus_1) / dt
                m_k_minus_12 = u_in @ ((w_k_minus_1 + w) / 2)
                m_k_xx_minus_12 = u_in_xx @ ((w_k_minus_1 + w) / 2)
                r2 = torch.cat(cross(m_k_minus_12[:, [0]],
                                     m_k_minus_12[:, [1]],
                                     m_k_minus_12[:, [2]],
                                     m_k_xx_minus_12[:, [0]],
                                     m_k_xx_minus_12[:, [1]],
                                     m_k_xx_minus_12[:, [2]]
                                     ), dim=1)
                r3 = torch.cat(cross(m_k_minus_12[:, [0]],
                                     m_k_minus_12[:, [1]],
                                     m_k_minus_12[:, [2]],
                                     *cross(
                                         m_k_minus_12[:, [0]],
                                         m_k_minus_12[:, [1]],
                                         m_k_minus_12[:, [2]],
                                         m_k_xx_minus_12[:, [0]],
                                         m_k_xx_minus_12[:, [1]],
                                         m_k_xx_minus_12[:, [2]]
                                     )), dim=1)
                r_pde = r1 + r2 + args.alpha * r3 - g_in
                r_bc = u_on_n @ w
                return torch.cat([r_pde[:, [0]], r_pde[:, [1]], r_pde[:, [2]],
                                  r_bc[:, [0]], r_bc[:, [1]], r_bc[:, [2]]], dim=0)

            def jac(w):
                # ∂rⁿ⁺¹_PDE/∂cⁿ⁺¹ = Φ/Δt + 1/2 ([uⁿ⁺¹ᐟ²]ₓΔΦ − [Δuⁿ⁺¹ᐟ²]ₓΦ)
                #                      + α/2 (−[uⁿ⁺¹ᐟ²×Δuⁿ⁺¹ᐟ²]ₓΦ + [uⁿ⁺¹ᐟ²]ₓ²ΔΦ − [uⁿ⁺¹ᐟ²]ₓ[Δuⁿ⁺¹ᐟ²]ₓΦ)
                # ∂rⁿ⁺¹_BC/∂cⁿ⁺¹ = ∂ₙΦ
                w = w.reshape(3, -1).T  # reshape w to a torch tensor of shape (N, 3)
                m_k_minus_12 = u_in @ ((w_k_minus_1 + w) / 2)
                m_k_xx_minus_12 = u_in_xx @ ((w_k_minus_1 + w) / 2)
                jac1_1 = torch.cat([jac_u, jac_v, jac_w], dim=0)
                jac1_2 = torch.cat(cross(m_k_minus_12[:, [0]],
                                         m_k_minus_12[:, [1]],
                                         m_k_minus_12[:, [2]],
                                         jac_uxx, jac_vxx, jac_wxx), dim=0)
                jac1_3 = torch.cat(cross(m_k_xx_minus_12[:, [0]],
                                         m_k_xx_minus_12[:, [1]],
                                         m_k_xx_minus_12[:, [2]],
                                         jac_u, jac_v, jac_w
                                         ), dim=0)
                jac1_4 = torch.cat(cross(*cross(m_k_minus_12[:, [0]],
                                                m_k_minus_12[:, [1]],
                                                m_k_minus_12[:, [2]],
                                                m_k_xx_minus_12[:, [0]],
                                                m_k_xx_minus_12[:, [1]],
                                                m_k_xx_minus_12[:, [2]]),
                                         jac_u, jac_v, jac_w), dim=0)
                jac1_5 = torch.cat(cross(m_k_minus_12[:, [0]],
                                         m_k_minus_12[:, [1]],
                                         m_k_minus_12[:, [2]],
                                         *cross(m_k_minus_12[:, [0]],
                                                m_k_minus_12[:, [1]],
                                                m_k_minus_12[:, [2]],
                                                jac_uxx, jac_vxx, jac_wxx)), dim=0)
                jac1_6 = torch.cat(cross(m_k_minus_12[:, [0]],
                                         m_k_minus_12[:, [1]],
                                         m_k_minus_12[:, [2]],
                                         *cross(m_k_xx_minus_12[:, [0]],
                                                m_k_xx_minus_12[:, [1]],
                                                m_k_xx_minus_12[:, [2]],
                                                jac_u, jac_v, jac_w
                                                )), dim=0)
                jac1 = 1.0 / dt * jac1_1 + 0.5 * (jac1_2 - jac1_3) + 0.5 * args.alpha * (-jac1_4 + jac1_5 - jac1_6)
                jac2 = pyrfm.concat_blocks([[jac_u_n], [jac_v_n], [jac_w_n]])

                return pyrfm.concat_blocks([[jac1], [jac2]])

            tol = 1e-10
            x0 = torch.zeros((3 * u_in.shape[1], 1)) if k == 0 else model.W.T.reshape(-1, 1)
            result = pyrfm.nonlinear_least_square(fcn=fcn,
                                                  x0=x0,
                                                  jac=jac,
                                                  ftol=tol,
                                                  gtol=tol,
                                                  xtol=tol,
                                                  method='newton',
                                                  verbose=0)

            status = result[1]

            # if status == 0:
            #     print("The maximum number of function evaluations is exceeded.")
            # elif status == 1:
            #     print("gtol termination condition is satisfied.")
            # elif status == 2:
            #     print("ftol termination condition is satisfied.")
            # elif status == 3:
            #     print("xtol termination condition is satisfied.")
            # elif status == 4:
            #     print("Both ftol and xtol termination conditions are satisfied.")
            # else:
            #     print("Unknown status.")

            model.W = result[0].reshape(3, -1).T

            m_pred = model(x_test)
            # m_pred /= torch.linalg.norm(m_pred, dim=1, keepdim=True)
            # model.compute(A_test.clone()).solve(torch.cat([m_pred[:, [0]], m_pred[:, [1]], m_pred[:, [2]]], dim=0))

            # m_pred /= torch.linalg.norm(m_pred, dim=1, keepdim=True)
            m_exact = func_m(torch.cat([x_test, (t0 + k * dt) * torch.ones((x_test.shape[0], 1))], dim=1), dim=3)
            error = torch.norm(m_pred - m_exact) / torch.norm(m_exact)
            # print(f"Step {k}/{n_steps}, Time {t0 + k * dt:.4f}, Error: {error:.4e}")

            norm_check = torch.linalg.norm(m_pred, dim=1)
            # print(f"  |m_pred| min: {norm_check.min():.6f}, max: {norm_check.max():.6f}")

            # plot m_pred[:, [0]], m_pred[:, [1]], m_pred[:, [2]]
            # import matplotlib.pyplot as plt
            #
            # x = x_test.cpu().numpy()
            #
            # plt.figure(figsize=(8, 6))
            #
            # # ---------------- RFM：空心 marker（上层） ----------------
            # plt.plot(
            #     x, m_pred[:, 0].cpu().numpy(),
            #     label='RFM m_x',
            #     linestyle='None',
            #     marker='o',
            #     markersize=4,
            #     markevery=10,
            #     markerfacecolor='none',
            #     markeredgecolor='r',
            #     markeredgewidth=1.0,
            #     zorder=3
            # )
            #
            # plt.plot(
            #     x, m_pred[:, 1].cpu().numpy(),
            #     label='RFM m_y',
            #     linestyle='None',
            #     marker='s',
            #     markersize=4,
            #     markevery=10,
            #     markerfacecolor='none',
            #     markeredgecolor='y',
            #     markeredgewidth=1.0,
            #     zorder=3
            # )
            #
            # plt.plot(
            #     x, m_pred[:, 2].cpu().numpy(),
            #     label='RFM m_z',
            #     linestyle='None',
            #     marker='^',
            #     markersize=4,
            #     markevery=10,
            #     markerfacecolor='none',
            #     markeredgecolor='g',
            #     markeredgewidth=1.0,
            #     zorder=3
            # )
            #
            # # ---------------- Exact：虚线（不同 dash pattern） ----------------
            # plt.plot(
            #     x, m_exact[:, 0].cpu().numpy(),
            #     label='Exact m_x',
            #     color='r',
            #     linewidth=1.4,
            #     alpha=0.6,
            #     linestyle=(0, (6, 2)),  # 长虚线
            #     zorder=1
            # )
            #
            # plt.plot(
            #     x, m_exact[:, 1].cpu().numpy(),
            #     label='Exact m_y',
            #     color='y',
            #     linewidth=1.4,
            #     alpha=0.6,
            #     linestyle=(0, (3, 2)),  # 中等虚线
            #     zorder=1
            # )
            #
            # plt.plot(
            #     x, m_exact[:, 2].cpu().numpy(),
            #     label='Exact m_z',
            #     color='g',
            #     linewidth=1.4,
            #     alpha=0.6,
            #     linestyle=(0, (1, 2)),  # 点虚线
            #     zorder=1
            # )
            #
            # plt.legend()
            # plt.tight_layout()
            # plt.show()

    xt_test = torch.cat([x_test, (t0 + k * dt) * torch.ones((x_in.shape[0], 1))], dim=1)
    m_exact = func_m(xt_test, dim=3)
    m_pred = model(x_test)
    m_pred /= torch.linalg.norm(m_pred, dim=1, keepdim=True)

    error = torch.norm(m_pred - m_exact) / torch.norm(m_exact)
    print(f"Error: {error:.4e}")

    return error


if __name__ == '__main__':
    torch.set_default_device('cuda') if torch.cuda.is_available() else torch.set_default_device('cpu')
    parser = argparse.ArgumentParser(description="Solve the LLG equation using RFM")
    parser.add_argument("--Nx", type=int, required=True)
    parser.add_argument("--Nt", type=int, required=True)
    parser.add_argument("--Qx", type=int, required=True)
    parser.add_argument("--Qt", type=int, required=True)
    parser.add_argument("--Jn", type=int, required=True)
    parser.add_argument("--Nb", type=int, required=True)
    parser.add_argument("--type", type=str, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--T", type=float, required=True)

    if len(sys.argv) == 1:
        for group, label in zip(param_sets_groups, group_labels):
            print(f"\n\n{label}")
            errors = []
            params = []
            for param_set in group:
                args = argparse.Namespace(**param_set)
                print("\n" + "=" * 40)
                print(f"Simulation Started with Parameters:")
                print(
                    f"Nx = {args.Nx}, Nt = {args.Nt}, Qx = {args.Qx}, Qt = {args.Qt}, Jn = {args.Jn}, Nb = {args.Nb}, type = {args.type}")
                print(f"--------------------------")
                start_time = time.time()
                errors.append(run_rfm(args))
                if "temporal" in label.lower():
                    params.append(args.Qt * args.Nt)
                elif "spatial" in label.lower():
                    params.append(args.Qx * args.Nx)
                print(f"\nSimulation Results:")
                print(f"--------------------------")
                print(f"Elapsed Time: {time.time() - start_time:.2f} seconds")
                print("=" * 40)

            if params and errors:
                if "spatial" in label.lower():
                    params = [int(p ** (1 / 3)) for p in params]
                for i in range(len(errors) - 1):
                    p = torch.log(errors[i] / errors[i + 1]) / torch.log(
                        torch.tensor(params[i + 1] / params[i], dtype=errors[i].dtype)
                    )
                    print(
                        f"params = {params[i]:>3d} -> {params[i + 1]:>3d}, "
                        f"order ≈ {p.item():.4f}"
                    )

                p_global = torch.log(errors[0] / errors[-1]) / torch.log(
                    torch.tensor(params[-1] / params[0], dtype=errors[0].dtype)
                )

                print(f"\nGlobal order (overall): ≈ {p_global.item():.4f}")

    else:
        args = parser.parse_args()
        run_rfm(args)
