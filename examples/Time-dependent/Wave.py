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
Section 3.1.3 (Wave Equation) 
Consider the following problem:

    ∂ₜ²u(x, t) − α²∂ₓ²u(x, t) = 0,      for (x, t) ∈ [x₀, x₁] × [0, T],
    u(x₀, t) = u(x₁, t) = 0,             for t ∈ [0, T],
    u(x, 0) = g₁(x),                     for x ∈ [x₀, x₁],
    ∂ₜu(x, 0) = g₂(x),                   for x ∈ [x₀, x₁].

where x₀ = 0, x₁ = 6π, α = 1, and T = 10. The exact solution is chosen to be:

    uₑ(x, t) = cos((απ / l) · t) · sin((π / l) · x)
             + [cos((2απ / l) · t) + (l / 2απ) · sin((2απ / l) · t)] · sin((2π / l) · x),

where l = x₁ − x₀.

Initial conditions g₁(x) and g₂(x) are chosen accordingly.
"""


def func_u(xt, alpha=1, l=6 * torch.pi):
    x = xt[:, :-1]
    t = xt[:, -1:]
    return (torch.cos((alpha * torch.pi / l) * t) * torch.sin((torch.pi / l) * x)
            + (torch.cos((2 * alpha * torch.pi / l) * t) + (l / (2 * alpha * torch.pi)) * torch.sin(
                (2 * alpha * torch.pi / l) * t)) * torch.sin((2 * torch.pi / l) * x))


def func_f(xt):
    return torch.zeros(xt.shape[0], 1)


def func_h(xt):
    t = xt[:, -1:]
    return torch.zeros(t.shape[0], 1)


def func_g1(xt, alpha=1, l=6 * torch.pi):
    return func_u(xt, alpha=alpha, l=l)


def func_g2(xt, alpha=1, l=6 * torch.pi):
    x = xt[:, :-1]
    t = xt[:, -1:]
    return (-(alpha * torch.pi / l) * torch.sin((alpha * torch.pi / l) * t) * torch.sin((torch.pi / l) * x)
            + (-(2 * alpha * torch.pi / l) * torch.sin((2 * alpha * torch.pi / l) * t) + (
                    l / (2 * alpha * torch.pi)) * (2 * alpha * torch.pi / l) * torch.cos(
                (2 * alpha * torch.pi / l) * t)) * torch.sin((2 * torch.pi / l) * x))


param_sets_groups = [
    # Convergence w.r.t Nt
    [
        {"Nx": 5, "Nt": 1, "Qx": 30, "Qt": 30, "Jn": 300, "Nb": 1, "type": "STC"},
        {"Nx": 5, "Nt": 2, "Qx": 30, "Qt": 30, "Jn": 300, "Nb": 1, "type": "STC"},
        {"Nx": 5, "Nt": 3, "Qx": 30, "Qt": 30, "Jn": 300, "Nb": 1, "type": "STC"},
        {"Nx": 5, "Nt": 4, "Qx": 30, "Qt": 30, "Jn": 300, "Nb": 1, "type": "STC"},
        {"Nx": 5, "Nt": 5, "Qx": 30, "Qt": 30, "Jn": 300, "Nb": 1, "type": "STC"}
    ],
    # Convergence w.r.t Nb
    [
        {"Nx": 5, "Nt": 1, "Qx": 30, "Qt": 30, "Jn": 300, "Nb": 1, "type": "STC"},
        {"Nx": 5, "Nt": 1, "Qx": 30, "Qt": 30, "Jn": 300, "Nb": 2, "type": "STC"},
        {"Nx": 5, "Nt": 1, "Qx": 30, "Qt": 30, "Jn": 300, "Nb": 3, "type": "STC"},
        {"Nx": 5, "Nt": 1, "Qx": 30, "Qt": 30, "Jn": 300, "Nb": 4, "type": "STC"},
        {"Nx": 5, "Nt": 1, "Qx": 30, "Qt": 30, "Jn": 300, "Nb": 5, "type": "STC"}
    ],
    # Convergence w.r.t Jn
    [
        {"Nx": 5, "Nt": 1, "Qx": 30, "Qt": 30, "Jn": 100, "Nb": 5, "type": "STC"},
        {"Nx": 5, "Nt": 1, "Qx": 30, "Qt": 30, "Jn": 150, "Nb": 5, "type": "STC"},
        {"Nx": 5, "Nt": 1, "Qx": 30, "Qt": 30, "Jn": 200, "Nb": 5, "type": "STC"},
        {"Nx": 5, "Nt": 1, "Qx": 30, "Qt": 30, "Jn": 250, "Nb": 5, "type": "STC"},
        {"Nx": 5, "Nt": 1, "Qx": 30, "Qt": 30, "Jn": 300, "Nb": 5, "type": "STC"}
    ],
    # Convergence w.r.t Qx/Qt
    [
        {"Nx": 5, "Nt": 1, "Qx": 10, "Qt": 10, "Jn": 300, "Nb": 5, "type": "STC"},
        {"Nx": 5, "Nt": 1, "Qx": 15, "Qt": 15, "Jn": 300, "Nb": 5, "type": "STC"},
        {"Nx": 5, "Nt": 1, "Qx": 20, "Qt": 20, "Jn": 300, "Nb": 5, "type": "STC"},
        {"Nx": 5, "Nt": 1, "Qx": 25, "Qt": 25, "Jn": 300, "Nb": 5, "type": "STC"},
        {"Nx": 5, "Nt": 1, "Qx": 30, "Qt": 30, "Jn": 300, "Nb": 5, "type": "STC"}
    ]
]

group_labels = [
    "Convergence w.r.t Nt",
    "Convergence w.r.t Nb",
    "Convergence w.r.t Jn",
    "Convergence w.r.t Qx/Qt"
]


def run_rfm(args):
    time_stamp = torch.linspace(start=0.0, end=10.0, steps=args.Nb + 1)
    domain = pyrfm.Line1D(x1=0.0, x2=6 * torch.pi)
    models = []
    for (t0, t1) in zip(time_stamp[:-1], time_stamp[1:]):
        models.append(pyrfm.STRFMBase(dim=1,
                                      n_hidden=args.Jn,
                                      domain=domain,
                                      time_interval=[t0, t1],
                                      n_spatial_subdomains=args.Nx,
                                      n_temporal_subdomains=args.Nt,
                                      st_type=args.type))

    x_in = domain.in_sample(args.Qx * args.Nx, with_boundary=False)
    x_on = domain.on_sample(2)

    for i, model in enumerate(models):
        t0 = torch.tensor(model.time_interval[0]).reshape(-1, 1)
        t = torch.linspace(*model.time_interval, (args.Qt * args.Nt) + 1)[1:].reshape(-1, 1)

        x_t0 = model.validate_and_prepare_xt(x=torch.cat([x_in, x_on], dim=0),
                                             t=t0)
        x_in_t = model.validate_and_prepare_xt(x=x_in, t=t)
        x_on_t = model.validate_and_prepare_xt(x=x_on, t=t)

        A_init = model.features(xt=x_t0).cat(dim=1)
        A_init_1 = model.features_derivative(xt=x_t0, axis=1).cat(dim=1)
        A_boundary = model.features(xt=x_on_t).cat(dim=1)
        A_tt = model.features_second_derivative(xt=x_in_t, axis1=1, axis2=1).cat(dim=1)
        A_xx = model.features_second_derivative(xt=x_in_t, axis1=0, axis2=0).cat(dim=1)
        A_pde = A_tt - A_xx

        if i == 0:
            f_init = func_g1(x_t0)
            f_init_1 = func_g2(x_t0)
        else:
            f_init = models[i - 1].forward(xt=x_t0)
            f_init_1 = models[i - 1].dForward(xt=x_t0, axis=[0, 1])

        f_boundary = func_h(x_on_t)
        f_pde = func_f(x_in_t)

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


if __name__ == "__main__":
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
