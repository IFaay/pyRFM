# -*- coding: utf-8 -*-
"""
Created on 2025/12/27

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

SICN method is used to solve the equation.

(2Φ/Δt + [ûⁿ⁺¹ᐟ²]×ΔΦ + α[ûⁿ⁺¹ᐟ²]×²ΔΦ) c = 2uⁿ/Δt − ûⁿ⁺¹ᐟ² × (Δuⁿ + 2f̂ⁿ⁺¹ᐟ²) − α ûⁿ⁺¹ᐟ² × (ûⁿ⁺¹ᐟ² × (Δuⁿ + 2f̂ⁿ⁺¹ᐟ²))

ûⁿ⁺¹ᐟ² = (3uⁿ − uⁿ⁻¹)/2, f̂ⁿ⁺¹ᐟ² = (3fⁿ − fⁿ⁻¹)/2


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
    [
        {"Nx": 2, "Nt": 2, "Qx": 200, "Qt": 20, "Jn": 100, "Nb": 1, "type": "STC", "alpha": 0.1, "T": 1.0},
        # {"Nx": 2, "Nt": 2, "Qx": 20, "Qt": 20, "Jn": 100, "Nb": 2, "type": "STC", "alpha": 0.1, "T": 1.0},
        # {"Nx": 2, "Nt": 2, "Qx": 20, "Qt": 20, "Jn": 100, "Nb": 3, "type": "STC"},
        # {"Nx": 2, "Nt": 2, "Qx": 20, "Qt": 20, "Jn": 100, "Nb": 4, "type": "STC"},
        # {"Nx": 2, "Nt": 2, "Qx": 20, "Qt": 20, "Jn": 100, "Nb": 5, "type": "STC"}
    ],
]

group_labels = ["Convergence",
                ]


def run_rfm(args):
    t_end = args.T
    time_stamp = torch.linspace(0, t_end, args.Nb + 1)
    domain = pyrfm.Line1D(x1=0.0, x2=1.0)

    x_in = domain.in_sample(args.Qx * args.Nx, with_boundary=False)
    x_on, x_on_normal = domain.on_sample(2, with_normal=True)

    print(x_on_normal)

    t0 = 0.0
    dt = 1e-2
    n_steps = round(args.T / dt)

    def cross(a0, a1, a2, b0, b1, b2):
        return a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0

    model = pyrfm.RFMBase(dim=1, n_hidden=args.Jn, domain=domain)

    u_in = model.features(x_in).cat(dim=1)
    u_in_xx = model.features_second_derivative(x_in, axis1=0, axis2=0).cat(dim=1)

    u_on_x = model.features_derivative(x_on, axis=0).cat(dim=1)
    u_on_n = u_on_x * x_on_normal[:, [0]]

    v_in, w_in = u_in, u_in
    v_in_xx, w_in_xx = u_in_xx, u_in_xx
    v_on_n, w_on_n = u_on_n, u_on_n

    w_k = torch.zeros((args.Jn, 3))
    w_k_minus_1 = torch.zeros((args.Jn, 3))
    w_k_minus_2 = torch.zeros((args.Jn, 3))

    # (2Φ/Δt + [ûⁿ⁺¹ᐟ²]×ΔΦ + α[ûⁿ⁺¹ᐟ²]×²ΔΦ) c = 2uⁿ/Δt − ûⁿ⁺¹ᐟ² × (Δuⁿ + 2f̂ⁿ⁺¹ᐟ²) − α ûⁿ⁺¹ᐟ² × (ûⁿ⁺¹ᐟ² × (Δuⁿ + 2f̂ⁿ⁺¹ᐟ²))
    for k in range(n_steps):
        if k == 0:
            A = pyrfm.concat_blocks([[u_in, torch.zeros_like(v_in), torch.zeros_like(w_in)],
                                     [torch.zeros_like(u_in), v_in, torch.zeros_like(w_in)],
                                     [torch.zeros_like(u_in), torch.zeros_like(v_in), w_in]])
            m0 = func_m(torch.cat([x_in, t0 * torch.ones((x_in.shape[0], 1))], dim=1), dim=1)
            b = torch.cat([m0[:, [0]], m0[:, [1]], m0[:, [2]]], dim=0)
            model.compute(A).solve(b)
            w_k = model.W.reshape(3, -1).T
            w_k_minus_1, w_k_minus_2 = w_k, w_k

        w_k = model.W.reshape(3, -1).T
        w_k_minus_1, w_k_minus_2 = w_k, w_k_minus_1
        xt_in = torch.cat([x_in, (t0 + (k + 0.5) * dt) * torch.ones((x_in.shape[0], 1))], dim=1)
        g_in = func_g(xt_in, dim=1, alpha=args.alpha)
        m_k = u_in @ w_k
        m_k_xx = u_in_xx @ w_k
        m_hat_k_plus_12 = u_in @ ((3 * w_k_minus_1 - w_k_minus_2) / 2)
        jac_u = pyrfm.concat_blocks([[u_in, torch.zeros_like(v_in), torch.zeros_like(w_in)]])
        jac_v = pyrfm.concat_blocks([[torch.zeros_like(u_in), v_in, torch.zeros_like(w_in)]])
        jac_w = pyrfm.concat_blocks([[torch.zeros_like(u_in), torch.zeros_like(v_in), w_in]])
        jac_uxx = pyrfm.concat_blocks([[u_in_xx, torch.zeros_like(v_in_xx), torch.zeros_like(w_in_xx)]])
        jac_vxx = pyrfm.concat_blocks([[torch.zeros_like(u_in_xx), v_in_xx, torch.zeros_like(w_in_xx)]])
        jac_wxx = pyrfm.concat_blocks([[torch.zeros_like(u_in_xx), torch.zeros_like(v_in_xx), w_in_xx]])
        jac_u_n = pyrfm.concat_blocks([[u_on_n, torch.zeros_like(v_on_n), torch.zeros_like(w_on_n)]])
        jac_v_n = pyrfm.concat_blocks([[torch.zeros_like(u_on_n), v_on_n, torch.zeros_like(w_on_n)]])
        jac_w_n = pyrfm.concat_blocks([[torch.zeros_like(u_on_n), torch.zeros_like(v_on_n), w_on_n]])
        term1 = (2 / dt) * torch.cat([jac_u, jac_v, jac_w], dim=0)
        term2 = torch.cat(cross(m_hat_k_plus_12[:, [0]],
                                m_hat_k_plus_12[:, [1]],
                                m_hat_k_plus_12[:, [2]],
                                jac_uxx, jac_vxx, jac_wxx), dim=0)
        term3 = args.alpha * torch.cat(cross(m_hat_k_plus_12[:, [0]],
                                             m_hat_k_plus_12[:, [1]],
                                             m_hat_k_plus_12[:, [2]],
                                             *cross(m_hat_k_plus_12[:, [0]],
                                                    m_hat_k_plus_12[:, [1]],
                                                    m_hat_k_plus_12[:, [2]],
                                                    jac_uxx, jac_vxx, jac_wxx)), dim=0)

        A = term1 + term2 + term3
        A = pyrfm.concat_blocks([[A], [jac_u_n], [jac_v_n], [jac_w_n]])
        m_x_m = cross(m_hat_k_plus_12[:, [0]],
                      m_hat_k_plus_12[:, [1]],
                      m_hat_k_plus_12[:, [2]],
                      (m_k_xx + 2 * g_in)[:, [0]],
                      (m_k_xx + 2 * g_in)[:, [1]],
                      (m_k_xx + 2 * g_in)[:, [2]])
        m_x_m_x_m = cross(m_hat_k_plus_12[:, [0]],
                          m_hat_k_plus_12[:, [1]],
                          m_hat_k_plus_12[:, [2]],
                          *m_x_m)
        b = (2 / dt) * m_k - torch.cat(m_x_m, dim=1) - args.alpha * torch.cat(m_x_m_x_m, dim=1)
        b = torch.cat([b[:, [0]], b[:, [1]], b[:, [2]],
                       torch.zeros((jac_u_n.shape[0], 1)),
                       torch.zeros((jac_v_n.shape[0], 1)),
                       torch.zeros((jac_w_n.shape[0], 1))], dim=0)
        print(A.shape, b.shape)

        model.compute(A).solve(b)


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
            for param_set in group:
                args = argparse.Namespace(**param_set)
                print("\n" + "=" * 40)
                print(f"Simulation Started with Parameters:")
                print(
                    f"Nx = {args.Nx}, Nt = {args.Nt}, Qx = {args.Qx}, Qt = {args.Qt}, Jn = {args.Jn}, Nb = {args.Nb}, type = {args.type}")
                print(f"--------------------------")
                start_time = time.time()
                run_rfm(args)
                print(f"\nSimulation Results:")
                print(f"--------------------------")
                print(f"Elapsed Time: {time.time() - start_time:.2f} seconds")
                print("=" * 40)
    else:
        args = parser.parse_args()
        run_rfm(args)
