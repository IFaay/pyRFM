# -*- coding: utf-8 -*-
"""
Created on 5/3/25

@author: Yifei Sun
"""
import pyrfm
import torch
import os
import argparse
import sys
import time

"""
Section 3.2.1 (Membrane Vibration over a Simple Geometry) 
Consider the following problem

⎧ ∂ₜₜ u(x, y, t) − α² Δu(x, y, t) = 0,        (x, y), t ∈ Ω × [0, T],
⎪ u(x, y, 0) = φ(x, y),                      (x, y) ∈ Ω,
⎪ ∂ₜ u(x, y, 0) = ψ(x, y),                   (x, y) ∈ Ω,
⎩ u(x, y, t) = 0,                            (x, y) ∈ ∂Ω × [0, T],

where Ω = [0, 5] × [0, 4], α = 1 and T = 10. The exact solution is chosen to be:

    uₑ(x, y, t) = sin(μx) · sin(νy) · (2 cos(λt) + sin(λt)),
    μ = 2π / (x₁ − x₀), ν = 2π / (y₁ − y₀), λ = √(μ² + ν²)

and φ(x, y) and ψ(x, y) are chosen accordingly.
"""


def func_u(xt):
    x = xt[:, [0]]
    y = xt[:, [1]]
    t = xt[:, [2]]
    mu = 2 * torch.pi / 5
    nu = 2 * torch.pi / 4
    lam = (mu ** 2 + nu ** 2) ** 0.5
    return (torch.sin(mu * x) * torch.sin(nu * y) *
            (2 * torch.cos(lam * t) + torch.sin(lam * t)))


def func_phi(xt):
    return func_u(xt)


def func_psi(xt):
    x = xt[:, [0]]
    y = xt[:, [1]]
    t = xt[:, [2]]
    mu = 2 * torch.pi / 5
    nu = 2 * torch.pi / 4
    lam = (mu ** 2 + nu ** 2) ** 0.5
    return (torch.sin(mu * x) * torch.sin(nu * y) * (-2 * lam * torch.sin(lam * t) + lam * torch.cos(lam * t)))


def run_rfm(args):
    time_stamp = torch.linspace(start=0, end=10, steps=args.Nb + 1)
    domain = pyrfm.Square2D(center=[2.5, 2], half=[2.5, 2])
    models = []
    for (t0, t1) in zip(time_stamp[:-1], time_stamp[1:]):
        models.append(pyrfm.STRFMBase(dim=2,
                                      n_hidden=args.Jn,
                                      domain=domain,
                                      time_interval=[t0, t1],
                                      n_spatial_subdomains=[args.Nx, args.Ny],
                                      n_temporal_subdomains=args.Nt,
                                      st_type=args.type)
                      )

    x_in = domain.in_sample([args.Qx * args.Nx, args.Qy * args.Ny], with_boundary=False)
    x_on = domain.on_sample([args.Qx * args.Nx, args.Qy * args.Ny])

    for i, model in enumerate(models):
        t0 = torch.tensor(model.time_interval[0]).reshape(-1, 1)
        t = torch.linspace(*model.time_interval, (args.Qt * args.Nt) + 1)[1:].reshape(-1, 1)

        x_t0 = model.validate_and_prepare_xt(x=torch.cat([x_in, x_on], dim=0),
                                             t=t0)
        x_in_t = model.validate_and_prepare_xt(x=x_in, t=t)
        x_on_t = model.validate_and_prepare_xt(x=x_on, t=t)

        A_init = model.features(xt=x_t0).cat(dim=1)
        A_init_1 = model.features_derivative(xt=x_t0, axis=2).cat(dim=1)
        A_boundary = model.features(xt=x_on_t).cat(dim=1)
        A_tt = model.features_second_derivative(xt=x_in_t, axis1=2, axis2=2).cat(dim=1)
        A_xx = model.features_second_derivative(xt=x_in_t, axis1=0, axis2=0).cat(dim=1)
        A_yy = model.features_second_derivative(xt=x_in_t, axis1=1, axis2=1).cat(dim=1)
        A_pde = A_tt - (A_xx + A_yy)

        if i == 0:
            f_init = func_phi(x_t0)
            f_init_1 = func_psi(x_t0)
        else:
            f_init = models[i - 1].forward(xt=x_t0)
            f_init_1 = models[i - 1].dForward(xt=x_t0, order=[0, 0, 1])

        f_boundary = torch.zeros(x_on_t.shape[0], 1)
        f_pde = torch.zeros(x_in_t.shape[0], 1)

        A = pyrfm.concat_blocks([[A_init], [A_init_1], [A_boundary], [A_pde]])
        f = pyrfm.concat_blocks([[f_init], [f_init_1], [f_boundary], [f_pde]])

        print(A.shape, f.shape)
        model.compute(A).solve(f)

    x_test = domain.in_sample(1000, with_boundary=True)
    u_pred = []
    u_truth = []
    for i, (t0, t1) in enumerate(zip(time_stamp[:-1], time_stamp[1:])):
        t_test = torch.linspace(t0, t1, 100).reshape(-1, 1)
        x_t = models[i].validate_and_prepare_xt(x=x_test, t=t_test)
        u_pred.append(models[i].forward(xt=x_t))
        u_truth.append(func_u(x_t))

    u_pred = torch.cat(u_pred, dim=0)
    u_truth = torch.cat(u_truth, dim=0)
    print("Relative Error: ", torch.linalg.norm(u_pred - u_truth) / torch.linalg.norm(u_truth))


param_sets_groups = [
    # Convergence w.r.t Nt (Figure 10a)
    [
        {"Nx": 2, "Ny": 2, "Nt": 1, "Qx": 30, "Qy": 30, "Qt": 30, "Jn": 400, "Nb": 1, "type": "STC"},
        {"Nx": 2, "Ny": 2, "Nt": 2, "Qx": 30, "Qy": 30, "Qt": 30, "Jn": 400, "Nb": 1, "type": "STC"},
        {"Nx": 2, "Ny": 2, "Nt": 3, "Qx": 30, "Qy": 30, "Qt": 30, "Jn": 400, "Nb": 1, "type": "STC"},
        {"Nx": 2, "Ny": 2, "Nt": 4, "Qx": 30, "Qy": 30, "Qt": 30, "Jn": 400, "Nb": 1, "type": "STC"},
        {"Nx": 2, "Ny": 2, "Nt": 5, "Qx": 30, "Qy": 30, "Qt": 30, "Jn": 400, "Nb": 1, "type": "STC"},
        {"Nx": 2, "Ny": 2, "Nt": 6, "Qx": 30, "Qy": 30, "Qt": 30, "Jn": 400, "Nb": 1, "type": "STC"}
    ],
    # Convergence w.r.t Jn (Figure 10b)
    [
        {"Nx": 2, "Ny": 2, "Nt": 2, "Qx": 30, "Qy": 30, "Qt": 30, "Jn": 100, "Nb": 1, "type": "STC"},
        {"Nx": 2, "Ny": 2, "Nt": 2, "Qx": 30, "Qy": 30, "Qt": 30, "Jn": 150, "Nb": 1, "type": "STC"},
        {"Nx": 2, "Ny": 2, "Nt": 2, "Qx": 30, "Qy": 30, "Qt": 30, "Jn": 200, "Nb": 1, "type": "STC"},
        {"Nx": 2, "Ny": 2, "Nt": 2, "Qx": 30, "Qy": 30, "Qt": 30, "Jn": 250, "Nb": 1, "type": "STC"},
        {"Nx": 2, "Ny": 2, "Nt": 2, "Qx": 30, "Qy": 30, "Qt": 30, "Jn": 300, "Nb": 1, "type": "STC"},
        {"Nx": 2, "Ny": 2, "Nt": 2, "Qx": 30, "Qy": 30, "Qt": 30, "Jn": 350, "Nb": 1, "type": "STC"},
        {"Nx": 2, "Ny": 2, "Nt": 2, "Qx": 30, "Qy": 30, "Qt": 30, "Jn": 400, "Nb": 1, "type": "STC"}
    ],
    # Convergence w.r.t Q (Figure 10c)
    [
        {"Nx": 5, "Ny": 5, "Nt": 5, "Qx": 10, "Qy": 10, "Qt": 10, "Jn": 400, "Nb": 5, "type": "STC"},
        {"Nx": 5, "Ny": 5, "Nt": 5, "Qx": 15, "Qy": 15, "Qt": 15, "Jn": 400, "Nb": 5, "type": "STC"},
        {"Nx": 5, "Ny": 5, "Nt": 5, "Qx": 20, "Qy": 20, "Qt": 20, "Jn": 400, "Nb": 5, "type": "STC"},
        {"Nx": 5, "Ny": 5, "Nt": 5, "Qx": 25, "Qy": 25, "Qt": 25, "Jn": 400, "Nb": 5, "type": "STC"},
        {"Nx": 5, "Ny": 5, "Nt": 5, "Qx": 30, "Qy": 30, "Qt": 30, "Jn": 400, "Nb": 5, "type": "STC"}
    ],
    # STC vs SoV (Figure 10d)
    [
        {"Nx": 2, "Ny": 2, "Nt": 1, "Qx": 30, "Qy": 30, "Qt": 30, "Jn": 400, "Nb": 1, "type": "STC"},
        {"Nx": 2, "Ny": 2, "Nt": 2, "Qx": 30, "Qy": 30, "Qt": 30, "Jn": 400, "Nb": 1, "type": "STC"},
        {"Nx": 2, "Ny": 2, "Nt": 3, "Qx": 30, "Qy": 30, "Qt": 30, "Jn": 400, "Nb": 1, "type": "STC"},
        {"Nx": 2, "Ny": 2, "Nt": 4, "Qx": 30, "Qy": 30, "Qt": 30, "Jn": 400, "Nb": 1, "type": "STC"},
        {"Nx": 2, "Ny": 2, "Nt": 5, "Qx": 30, "Qy": 30, "Qt": 30, "Jn": 400, "Nb": 1, "type": "STC"},
        {"Nx": 2, "Ny": 2, "Nt": 6, "Qx": 30, "Qy": 30, "Qt": 30, "Jn": 400, "Nb": 1, "type": "STC"},
        {"Nx": 2, "Ny": 2, "Nt": 6, "Qx": 30, "Qy": 30, "Qt": 30, "Jn": 400, "Nb": 1, "type": "SOV"}
    ]
]

group_labels = [
    "Convergence w.r.t Nt (10a)",
    "Convergence w.r.t Jn (10b)",
    "Convergence w.r.t Q (10c)",
    "Comparison STC vs SoV (10d)"
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RFM for Time-dependent PDEs")
    parser.add_argument("--Nx", type=int, default=2, help="Number of spatial subdomains in x direction")
    parser.add_argument("--Ny", type=int, default=2, help="Number of spatial subdomains in y direction")
    parser.add_argument("--Nt", type=int, default=1, help="Number of temporal subdomains")
    parser.add_argument("--Qx", type=int, default=30, help="Number of quadrature points in x direction")
    parser.add_argument("--Qy", type=int, default=30, help="Number of quadrature points in y direction")
    parser.add_argument("--Qt", type=int, default=30, help="Number of quadrature points in t direction")
    parser.add_argument("--Jn", type=int, default=400, help="Number of neurons in each hidden layer")
    parser.add_argument("--Nb", type=int, default=1, help="Number of basis functions")
    parser.add_argument("--type", type=str, default="STC", choices=["STC", "SOV"], help="Type of RFM")

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
