# -*- coding: utf-8 -*-
"""
Created on 4/25/25

@author: Yifei Sun

"""
import pyrfm
import torch
import os
import argparse
import sys
import time

"""
Section 3.1.4 (Schrödinger Equation)
Consider the following problem:

    i∂ₜψ(x, t) + 0.5Δψ(x, t) = 0,                  for (x, t) ∈ [x₀, x₁] × [0, T],
    ψ(x, 0) = g(x),                                for x ∈ [x₀, x₁],
    ψ(x₀, t) = ψ(x₁, t),                           for t ∈ [0, T],
    ∂ₓψ(x₀, t) = ∂ₓψ(x₁, t),                       for t ∈ [0, T].

where x₀ = 0, x₁ = 5, and T = 10. The exact solution is chosen to be:

    ψ(x, t) = exp(−iω²t / 2) · (2 · cos(ωx) + sin(ωx)),

where ω = 2π / (x₁ − x₀), and g(x) is chosen accordingly.
"""


def func_psi(xt):
    x = xt[:, :-1]
    t = xt[:, -1:]
    return torch.exp(-1j * (2 * torch.pi / 5) ** 2 * t / 2) * (
            2 * torch.cos((2 * torch.pi / 5) * x) + torch.sin((2 * torch.pi / 5) * x))


def func_g(xt):
    return func_psi(xt)


def run_rfm(args):
    time_stamp = torch.linspace(start=0, end=10, steps=args.Nt + 1)
    domain = pyrfm.Line1D(x1=0, x2=5)
    models = []
    for (t0, t1) in zip(time_stamp[:-1], time_stamp[1:]):
        models.append(pyrfm.STRFMBase(dim=1,
                                      n_hidden=args.Jn,
                                      domain=domain,
                                      time_interval=[t0, t1],
                                      n_spatial_subdomains=args.Nx,
                                      n_temporal_subdomains=args.Nt,
                                      st_type=args.type
                                      ))

    x_in = domain.in_sample(args.Qx * args.Nx, with_boundary=False)
    x_on = domain.on_sample(2)

    for i, model in enumerate(models):
        t0 = torch.tensor(model.time_interval[0]).reshape(-1, 1)
        x0 = x_on[0].reshape(-1, 1)
        x1 = x_on[1].reshape(-1, 1)
        t = torch.linspace(*model.time_interval, (args.Qt * args.Nt) + 1)[1:].reshape(-1, 1)

        x_in_t = model.validate_and_prepare_xt(x=x_in, t=t)
        x_t0 = model.validate_and_prepare_xt(x=torch.cat([x_in, x_on], dim=0),
                                             t=t0)
        x0_t = model.validate_and_prepare_xt(x=x0, t=t)
        x1_t = model.validate_and_prepare_xt(x=x1, t=t)

        A_init = model.features(xt=x_t0).cat(dim=1)
        A_t = model.features_derivative(xt=x_in_t, axis=1).cat(dim=1)
        A_xx = model.features_second_derivative(xt=x_in_t, axis1=0, axis2=0).cat(dim=1)
        A_pde = 1j * A_t + 0.5 * A_xx

        A_boundary = model.features(xt=x0_t).cat(dim=1) - model.features(xt=x1_t).cat(dim=1)
        A_boundary_1 = model.features_derivative(xt=x0_t, axis=0).cat(dim=1) - model.features_derivative(xt=x1_t,
                                                                                                         axis=0).cat(
            dim=1)

        if i == 0:
            f_init = func_g(x_t0)
        else:
            f_init = models[i - 1].forward(xt=x_t0)

        f_pde = torch.zeros(x_in_t.shape[0], 1)
        f_boundary = torch.zeros(x0_t.shape[0], 1)
        f_boundary_1 = torch.zeros(x0_t.shape[0], 1)

        A = pyrfm.concat_blocks([[A_init], [A_boundary], [A_boundary_1], [A_pde]])
        f = pyrfm.concat_blocks([[f_init], [f_boundary], [f_boundary_1], [f_pde]])

        print(A.shape, f.shape)
        model.dtype = torch.complex128 if model.dtype == torch.float64 else torch.complex64
        model.compute(A).solve(f)

    x_test = domain.in_sample(1000, with_boundary=True)
    u_pred = []
    u_truth = []
    for i, (t0, t1) in enumerate(zip(time_stamp[:-1], time_stamp[1:])):
        t_test = torch.linspace(t0, t1, 100).reshape(-1, 1)
        x_t = models[i].validate_and_prepare_xt(x=x_test, t=t_test)
        u_pred.append(models[i].forward(xt=x_t))
        u_truth.append(func_psi(x_t))

    u_pred = torch.cat(u_pred, dim=0)
    u_truth = torch.cat(u_truth, dim=0)
    print("Relative Error: ", torch.linalg.norm(u_pred - u_truth) / torch.linalg.norm(u_truth))


param_sets_groups = [
    # Convergence w.r.t Nt
    [
        {"Nx": 5, "Nt": 1, "Qx": 30, "Qt": 30, "Jn": 300, "Nb": 1, "type": "STC"},
        {"Nx": 5, "Nt": 2, "Qx": 30, "Qt": 30, "Jn": 300, "Nb": 1, "type": "STC"},
        {"Nx": 5, "Nt": 3, "Qx": 30, "Qt": 30, "Jn": 300, "Nb": 1, "type": "STC"}
    ],
    # Convergence w.r.t Nb
    [
        {"Nx": 5, "Nt": 1, "Qx": 30, "Qt": 30, "Jn": 300, "Nb": 1, "type": "STC"},
        {"Nx": 5, "Nt": 1, "Qx": 30, "Qt": 30, "Jn": 300, "Nb": 2, "type": "STC"},
        {"Nx": 5, "Nt": 1, "Qx": 30, "Qt": 30, "Jn": 300, "Nb": 3, "type": "STC"}
    ],
    # Convergence w.r.t Jn
    [
        {"Nx": 5, "Nt": 1, "Qx": 30, "Qt": 30, "Jn": 100, "Nb": 3, "type": "STC"},
        {"Nx": 5, "Nt": 1, "Qx": 30, "Qt": 30, "Jn": 150, "Nb": 3, "type": "STC"},
        {"Nx": 5, "Nt": 1, "Qx": 30, "Qt": 30, "Jn": 200, "Nb": 3, "type": "STC"},
        {"Nx": 5, "Nt": 1, "Qx": 30, "Qt": 30, "Jn": 250, "Nb": 3, "type": "STC"},
        {"Nx": 5, "Nt": 1, "Qx": 30, "Qt": 30, "Jn": 300, "Nb": 3, "type": "STC"}
    ],
    # Convergence w.r.t Qx/Qt
    [
        {"Nx": 5, "Nt": 1, "Qx": 10, "Qt": 10, "Jn": 300, "Nb": 3, "type": "STC"},
        {"Nx": 5, "Nt": 1, "Qx": 15, "Qt": 15, "Jn": 300, "Nb": 3, "type": "STC"},
        {"Nx": 5, "Nt": 1, "Qx": 20, "Qt": 20, "Jn": 300, "Nb": 3, "type": "STC"},
        {"Nx": 5, "Nt": 1, "Qx": 25, "Qt": 25, "Jn": 300, "Nb": 3, "type": "STC"},
        {"Nx": 5, "Nt": 1, "Qx": 30, "Qt": 30, "Jn": 300, "Nb": 3, "type": "STC"}
    ],
    # Comparison between STC and SoV
    [
        {"Nx": 5, "Nt": 3, "Qx": 30, "Qt": 30, "Jn": 300, "Nb": 1, "type": "STC"},
        {"Nx": 5, "Nt": 3, "Qx": 30, "Qt": 30, "Jn": 300, "Nb": 1, "type": "SOV"}
    ]
]

group_labels = [
    "Convergence w.r.t Nt",
    "Convergence w.r.t Nb",
    "Convergence w.r.t Jn",
    "Convergence w.r.t Qx/Qt",
    "Comparison between STC and SoV"
]

if __name__ == '__main__':
    # torch.set_default_device('cuda') if torch.cuda.is_available() else torch.set_default_device('cpu')
    parser = argparse.ArgumentParser(description="Solve the heat equation using RFM")
    parser.add_argument("--Nx", type=int, required=True)
    parser.add_argument("--Nt", type=int, required=True)
    parser.add_argument("--Qx", type=int, required=True)
    parser.add_argument("--Qt", type=int, required=True)
    parser.add_argument("--Jn", type=int, required=True)
    parser.add_argument("--Nb", type=int, required=True)
    parser.add_argument("--type", type=str, required=True)

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
