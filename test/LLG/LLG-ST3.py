# -*- coding: utf-8 -*-
"""
Created on 2025/3/16

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
        {"Nx": 1, "Nt": 1, "Qx": 300, "Qt": 10, "Jn": 300, "Nb": 1, "type": "STC", "alpha": 0.1, "T": 1.0},
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
    domain = pyrfm.Cube3D(center=(0.5, 0.5, 0.5), half=(0.5, 0.5, 0.5))

    models = []
    for (t0, t1) in zip(time_stamp[:-1], time_stamp[1:]):
        models.append(pyrfm.STRFMBase(dim=3,
                                      n_hidden=args.Jn,
                                      domain=domain,
                                      time_interval=[t0, t1],
                                      n_spatial_subdomains=args.Nx,
                                      n_temporal_subdomains=args.Nt,
                                      st_type=args.type))

    x_in = domain.in_sample(args.Qx * args.Nx, with_boundary=False)
    x_on, x_on_normal = domain.on_sample(2, with_normal=True)

    for i, model in enumerate(models):
        t0 = torch.tensor(model.time_interval[0]).reshape(-1, 1)
        t = torch.linspace(*model.time_interval, (args.Qt * args.Nt) + 1)[1:].reshape(-1, 1)

        x_t0 = model.validate_and_prepare_xt(x=torch.cat([x_in, x_on], dim=0), t=t0)

        x_in_t = model.validate_and_prepare_xt(x=x_in, t=t)
        x_on_t = model.validate_and_prepare_xt(x=x_on, t=t)

        u_init = model.features(xt=x_t0).cat(dim=1)
        u_boundary_x = model.features_derivative(xt=x_on_t, axis=0).cat(dim=1)
        u_boundary_y = model.features_derivative(xt=x_on_t, axis=1).cat(dim=1)
        u_boundary_z = model.features_derivative(xt=x_on_t, axis=2).cat(dim=1)
        u_boundary_x = (u_boundary_x * x_on_normal[:, [0]] +
                        u_boundary_y * x_on_normal[:, [1]] +
                        u_boundary_z * x_on_normal[:, [2]])

        u_in = model.features(xt=x_in_t).cat(dim=1)
        u_in_t = model.features_derivative(xt=x_in_t, axis=3).cat(dim=1)
        u_in_xx = model.features_second_derivative(xt=x_in_t, axis1=0, axis2=0).cat(dim=1)
        u_in_yy = model.features_second_derivative(xt=x_in_t, axis1=1, axis2=1).cat(dim=1)
        u_in_zz = model.features_second_derivative(xt=x_in_t, axis1=2, axis2=2).cat(dim=1)
        u_in_xx += u_in_yy + u_in_zz

        v_init, w_init = u_init, u_init
        v_boundary_x, w_boundary_x = u_boundary_x, u_boundary_x
        v_in, w_in = u_in, u_in
        v_in_t, w_in_t = u_in_t, u_in_t
        v_in_xx, w_in_xx = u_in_xx, u_in_xx

        def cross(a0, a1, a2, b0, b1, b2):
            return a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0

        def fcn_with_g1(w, g1):
            w = w.reshape(3, -1).T  # reshape w to a torch tensor of shape (N, 3)
            m_init = u_init @ w
            m_boundary_x = u_boundary_x @ w
            g2 = torch.zeros(x_on_t.shape[0], 3)
            m = u_in @ w
            mt = u_in_t @ w
            mxx = u_in_xx @ w
            g3 = func_g(x_in_t, dim=3, alpha=args.alpha)

            return torch.cat([(m_init - g1).T.reshape(1, -1).T,
                              (m_boundary_x - g2).T.reshape(1, -1).T,
                              (mt + m.cross(mxx, dim=1) + args.alpha * m.cross(m.cross(mxx, dim=1),
                                                                               dim=1) - g3).T.reshape(1, -1).T], dim=0)

        if i == 0:
            def fcn(w):
                g1 = func_m(x_t0, dim=3)
                return fcn_with_g1(w, g1)
        else:
            def fcn(w):
                g1 = models[i - 1].forward(xt=x_t0)
                g1 /= torch.norm(g1, dim=1, keepdim=True)
                return fcn_with_g1(w, g1)

        def jac(w):
            jac1 = pyrfm.concat_blocks([[u_init, torch.zeros_like(v_init), torch.zeros_like(w_init)],
                                        [torch.zeros_like(u_init), v_init, torch.zeros_like(w_init)],
                                        [torch.zeros_like(u_init), torch.zeros_like(v_init), w_init]])
            jac2 = pyrfm.concat_blocks([[u_boundary_x, torch.zeros_like(v_boundary_x), torch.zeros_like(w_boundary_x)],
                                        [torch.zeros_like(u_boundary_x), v_boundary_x, torch.zeros_like(w_boundary_x)],
                                        [torch.zeros_like(u_boundary_x), torch.zeros_like(v_boundary_x), w_boundary_x]])

            """
            𝓙 = 𝚽ₜ + 𝚽 × Δ𝐦 + 𝐦 × Δ𝚽 
                        + α 𝚽 × (𝐦 × Δ𝐦) 
                        + α 𝐦 × (𝚽 × Δ𝐦 + 𝐦 × Δ𝚽)
            """
            w = w.reshape(3, -1).T  # reshape w to a torch tensor of shape (N, 3)
            m = u_in @ w
            mxx = u_in_xx @ w
            jac_u = pyrfm.concat_blocks([[u_in, torch.zeros_like(v_in), torch.zeros_like(w_in)]])
            jac_v = pyrfm.concat_blocks([[torch.zeros_like(u_in), v_in, torch.zeros_like(w_in)]])
            jac_w = pyrfm.concat_blocks([[torch.zeros_like(u_in), torch.zeros_like(v_in), w_in]])
            jac_uxx = pyrfm.concat_blocks([[u_in_xx, torch.zeros_like(v_in_xx), torch.zeros_like(w_in_xx)]])
            jac_vxx = pyrfm.concat_blocks([[torch.zeros_like(u_in_xx), v_in_xx, torch.zeros_like(w_in_xx)]])
            jac_wxx = pyrfm.concat_blocks([[torch.zeros_like(u_in_xx), torch.zeros_like(v_in_xx), w_in_xx]])
            jac3_1 = pyrfm.concat_blocks([[u_in_t, torch.zeros_like(v_in_t), torch.zeros_like(w_in_t)],
                                          [torch.zeros_like(u_in_t), v_in_t, torch.zeros_like(w_in_t)],
                                          [torch.zeros_like(u_in_t), torch.zeros_like(v_in_t), w_in_t]])
            jac3_2 = torch.cat(cross(jac_u, jac_v, jac_w, mxx[:, [0]], mxx[:, [1]], mxx[:, [2]]), dim=0)
            jac3_3 = torch.cat(cross(m[:, [0]], m[:, [1]], m[:, [2]], jac_uxx, jac_vxx, jac_wxx), dim=0)
            jac3_4 = torch.cat(cross(jac_u, jac_v, jac_w,
                                     *cross(m[:, [0]], m[:, [1]], m[:, [2]], mxx[:, [0]], mxx[:, [1]], mxx[:, [2]])),
                               dim=0)
            jac3_5 = torch.cat(cross(m[:, [0]], m[:, [1]], m[:, [2]],
                                     *cross(jac_u, jac_v, jac_w, mxx[:, [0]], mxx[:, [1]], mxx[:, [2]])),
                               dim=0)
            jac3_6 = torch.cat(cross(m[:, [0]], m[:, [1]], m[:, [2]],
                                     *cross(m[:, [0]], m[:, [1]], m[:, [2]], jac_uxx, jac_vxx, jac_wxx)), dim=0)
            jac3 = jac3_1 + jac3_2 + jac3_3 + args.alpha * (jac3_4 + jac3_5 + jac3_6)

            return torch.cat([jac1, jac2, jac3], dim=0)

        tol = 1e-8
        x0 = torch.zeros((3 * u_in.shape[1], 1)) if i == 0 else models[i - 1].W.T.reshape(-1, 1)
        result = pyrfm.nonlinear_least_square(fcn=fcn,
                                              x0=x0,
                                              jac=jac,
                                              ftol=tol,
                                              gtol=tol,
                                              xtol=tol,
                                              method='newton',
                                              verbose=1)

        status = result[1]

        if status == 0:
            print("The maximum number of function evaluations is exceeded.")
        elif status == 1:
            print("gtol termination condition is satisfied.")
        elif status == 2:
            print("ftol termination condition is satisfied.")
        elif status == 3:
            print("xtol termination condition is satisfied.")
        elif status == 4:
            print("Both ftol and xtol termination conditions are satisfied.")
        else:
            print("Unknown status.")

        model.W = result[0].reshape(3, -1).T

    xt_test = models[-1].validate_and_prepare_xt(x=x_in, t=torch.tensor([[t_end]]))
    m_pred = models[-1].forward(xt=xt_test)
    m_pred /= torch.linalg.norm(m_pred, dim=1, keepdim=True)
    m_exact = func_m(xt_test, dim=3)
    error = torch.norm(m_pred - m_exact) / torch.norm(m_exact)
    print(f"Error: {error:.4e}")


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
