# -*- coding: utf-8 -*-
"""
Created on 2025/2/16

@author: Yifei Sun

"""
from xml.etree.ElementPath import xpath_tokenizer

import pyrfm
import torch
import os
import argparse
import sys
import time

"""
Section 3.1.1 (Heat equation)
Consider the following problem:

    ∂ₜ u(x, t) - α² ∂ₓ² u(x, t) = 0,  x ∈ [x₀, x₁],  t ∈ [0, T],
    u(x₀, t) = g₁(t),                 t ∈ [0, T],
    u(x₁, t) = g₂(t),                 t ∈ [0, T],
    u(x, 0) = h(x),                   x ∈ [x₀, x₁],

where α = π / 2, x₀ = 0, x₁ = 12 and T = 10. The exact solution is chosen to be
uₑ(x, t) = 2 sin(α x) e^(-t)

"""


def func_u(xt):
    x = xt[:, :-1]
    t = xt[:, -1:]
    return 2 * torch.sin(torch.tensor(torch.pi / 2) * x) * torch.exp(-t)


def func_f(xt):
    x = xt[:, :-1]
    return torch.zeros(x.shape[0], 1)


def func_g(xt):
    return torch.zeros(xt.shape[0], 1)


def func_h(xt):
    x = xt[:, :-1]
    return 2 * torch.sin(torch.tensor(torch.pi / 2) * x)


# default setting: Nx = 2, Nt = 5, Qx = 20, Qt = 20, Jn = 400 and Nb = 1.
# In Figure 2(a), we set Nb = 1, Nx = 2, Jn = 400, Qx = Qt = 20 and Nt = 1, · · · , 5
# In Figure 2(b), we set Nx = 2, Nt = 1, Jn = 400, Qx = Qt = 20 and Nb = 1, · · · , 5
# In Figure 2(c), we set Nb = 5, Nx = 2, Nt = 1, Qx = Qt = 20 and Jn = 50, 100, 200, 300, 400
# In Figure 2(d), we set Nb = 5, Nx = 2, Nt = 1, Jn = 400 and Qx = Qt = 5, 10, 15, 20, 25
# In Figure 2(e), we compare STC and SoV with default setting
# In Figure 2(f), we compare the block time-marching strategy and the STRFM (Nb = 5, Nt = 1 and Nb = 1, Nt = 5)


param_sets_groups = [
    [
        {"Nx": 2, "Nt": 5, "Qx": 20, "Qt": 20, "Jn": 400, "Nb": 1, "type": "STC"},
        {"Nx": 2, "Nt": 5, "Qx": 20, "Qt": 20, "Jn": 400, "Nb": 2, "type": "STC"},
        {"Nx": 2, "Nt": 5, "Qx": 20, "Qt": 20, "Jn": 400, "Nb": 3, "type": "STC"},
        {"Nx": 2, "Nt": 5, "Qx": 20, "Qt": 20, "Jn": 400, "Nb": 4, "type": "STC"},
        {"Nx": 2, "Nt": 5, "Qx": 20, "Qt": 20, "Jn": 400, "Nb": 5, "type": "STC"}
    ],
    [
        {"Nx": 2, "Nt": 1, "Qx": 20, "Qt": 20, "Jn": 400, "Nb": 1, "type": "STC"},
        {"Nx": 2, "Nt": 1, "Qx": 20, "Qt": 20, "Jn": 400, "Nb": 2, "type": "STC"},
        {"Nx": 2, "Nt": 1, "Qx": 20, "Qt": 20, "Jn": 400, "Nb": 3, "type": "STC"},
        {"Nx": 2, "Nt": 1, "Qx": 20, "Qt": 20, "Jn": 400, "Nb": 4, "type": "STC"},
        {"Nx": 2, "Nt": 1, "Qx": 20, "Qt": 20, "Jn": 400, "Nb": 5, "type": "STC"}
    ],
    [
        {"Nx": 2, "Nt": 1, "Qx": 20, "Qt": 20, "Jn": 50, "Nb": 5, "type": "STC"},
        {"Nx": 2, "Nt": 1, "Qx": 20, "Qt": 20, "Jn": 100, "Nb": 5, "type": "STC"},
        {"Nx": 2, "Nt": 1, "Qx": 20, "Qt": 20, "Jn": 200, "Nb": 5, "type": "STC"},
        {"Nx": 2, "Nt": 1, "Qx": 20, "Qt": 20, "Jn": 300, "Nb": 5, "type": "STC"},
        {"Nx": 2, "Nt": 1, "Qx": 20, "Qt": 20, "Jn": 400, "Nb": 5, "type": "STC"}
    ],
    [
        {"Nx": 2, "Nt": 1, "Qx": 5, "Qt": 5, "Jn": 400, "Nb": 5, "type": "STC"},
        {"Nx": 2, "Nt": 1, "Qx": 10, "Qt": 10, "Jn": 400, "Nb": 5, "type": "STC"},
        {"Nx": 2, "Nt": 1, "Qx": 15, "Qt": 15, "Jn": 400, "Nb": 5, "type": "STC"},
        {"Nx": 2, "Nt": 1, "Qx": 20, "Qt": 20, "Jn": 400, "Nb": 5, "type": "STC"},
        {"Nx": 2, "Nt": 1, "Qx": 25, "Qt": 25, "Jn": 400, "Nb": 5, "type": "STC"}
    ],
    [
        {"Nx": 2, "Nt": 5, "Qx": 20, "Qt": 20, "Jn": 400, "Nb": 1, "type": "STC"},
        {"Nx": 2, "Nt": 5, "Qx": 20, "Qt": 20, "Jn": 400, "Nb": 1, "type": "SOV"}
    ],
    [
        {"Nx": 2, "Nt": 1, "Qx": 20, "Qt": 20, "Jn": 400, "Nb": 5, "type": "STC"},
        {"Nx": 2, "Nt": 5, "Qx": 20, "Qt": 20, "Jn": 400, "Nb": 1, "type": "STC"}
    ]
]

group_labels = ["Convergence w.r.t Nt",
                "Convergence w.r.t Nb",
                "Convergence w.r.t Jn",
                "Convergence w.r.t Q",
                "Comparison between STC and SoV",
                "Comparison between block time-marching and STRFM"]


def run_rfm(args):
    # model = pyrfm.STRFMBase(1, 50, [0, 1], [0, 10], 5, 3, 1,
    #                         st_type="SoV")

    time_stamp = torch.linspace(start=0, end=10, steps=args.Nb + 1)
    domain = pyrfm.Line1D(x1=0.0, x2=12.0)
    models = []
    for (t0, t1) in zip(time_stamp[:-1], time_stamp[1:]):
        models.append(pyrfm.STRFMBase(dim=1,
                                      n_hidden=args.Jn,
                                      domain=domain,
                                      time_interval=[t0, t1],
                                      n_spatial_subdomains=args.Nx,
                                      n_temporal_subdomains=args.Nt,
                                      st_type="SOV"))

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
        A_boundary = model.features(xt=x_on_t).cat(dim=1)
        A_t = model.features_derivative(xt=x_in_t, axis=1).cat(dim=1)
        A_xx = model.features_second_derivative(xt=x_in_t, axis1=0, axis2=0).cat(dim=1)
        A_pde = A_t - torch.tensor(torch.pi / 2) ** (-2) * A_xx

        if i == 0:
            f_init = func_h(x_t0)
        else:
            f_init = models[i - 1].forward(xt=x_t0)

        f_boundary = func_g(x_on_t)
        f_pde = func_f(x_in_t)

        A = pyrfm.concat_blocks([[A_init], [A_boundary], [A_pde]])
        f = pyrfm.concat_blocks([[f_init], [f_boundary], [f_pde]])

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
                print(f"Nx = {args.Nx}, Nt = {args.Nt}, Qx = {args.Qx}, Qt = {args.Qt}, Jn = {args.Jn}, Nb = {args.Nb}, type = {args.type}")
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
