# -*- coding: utf-8 -*-
"""
Created on 4/25/25

@author: Yifei Sun

"""
from pyparsing import empty

import pyrfm
import torch
import os
import argparse
import sys
import time

"""
Section 3.1.2 (Heat Equation with Nonsmooth Initial Condition)
Consider the heat equation

    ∂ₜ u(x, t) - α² ∂ₓ² u(x, t) = 0,  x ∈ [x₀, x₁],  t ∈ [0, T],
    u(x₀, t) = g₁(t),                 t ∈ [0, T],
    u(x₁, t) = g₂(t),                 t ∈ [0, T],
    u(x, 0) = h(x),                   x ∈ [x₀, x₁],

where α = π / 2 and T = 10 with x₀ = 0, x₁ = 8 and the nonsmooth initial condition as follows:

    h(x) = 2𝟙_{0 ≤ x < 4} · sin(πx / 2) + 2𝟙_{4 ≤ x ≤ 8} · sin(πx)

It is easy to check that h(x) only belongs to C([0, 8]).

Nx = 2, Nt = 5, Qx = 30, Qt = 50, Jn = 250, and Nb = 1, to obtain a reference solution.
"""


def func_u(xt):
    # not analytic solution
    pass


def func_f(xt):
    x = xt[:, :-1]
    return torch.zeros(x.shape[0], 1)


def func_g(xt):
    return torch.zeros(xt.shape[0], 1)


def func_h(xt):
    x = xt[:, :-1]
    return 2 * torch.sin(torch.tensor(torch.pi / 2) * x) * (x < 4).double() + \
        2 * torch.sin(torch.tensor(torch.pi) * x) * (x >= 4).double()


def run_rfm(args, reference=False):
    time_stamp = torch.linspace(start=0, end=10, steps=args.Nb + 1)
    domain = pyrfm.Line1D(x1=0.0, x2=8.0)
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

    if reference:
        global reference_model
        reference_model = ReferenceModel(models)

    else:
        if reference_model is None:
            raise ValueError("Reference model is not provided.")

        x_test = domain.in_sample(1000, with_boundary=True)
        u_pred = []
        u_truth = []
        for i, (t0, t1) in enumerate(zip(time_stamp[:-1], time_stamp[1:])):
            t_test = torch.linspace(t0, t1, 100).reshape(-1, 1)
            x_t = models[i].validate_and_prepare_xt(x=x_test, t=t_test)
            u_pred.append(models[i].forward(xt=x_t))
            u_truth.append(reference_model(x_t))

        u_pred = torch.cat(u_pred, dim=0)
        u_truth = torch.cat(u_truth, dim=0)
        print("Relative Error: ", torch.linalg.norm(u_pred - u_truth) / torch.linalg.norm(u_truth))


class ReferenceModel:
    def __init__(self, models):
        self.models = models

    def __call__(self, xt):
        t = xt[:, -1]  # Extract the time component from xt
        results = []
        indices = []  # To store the original indices of xt
        processed = torch.zeros(t.shape[0], dtype=torch.bool)  # Track processed indices

        for model in self.models:
            t0, t1 = model.time_interval
            mask = (t >= t0) & (t <= t1) & ~processed  # Ensure no overlap with already processed indices
            if mask.any():
                selected_xt = xt[mask]
                results.append(model.forward(xt=selected_xt))
                indices.append(torch.nonzero(mask, as_tuple=True)[0])  # Store the indices
                processed |= mask  # Mark these indices as processed

        if results:
            results = torch.cat(results, dim=0)
            indices = torch.cat(indices, dim=0)
            _, sorted_indices = torch.sort(indices)  # Sort by the original indices
            results = results[sorted_indices]  # Reorder results to match xt's order

        return results if results.numel() > 0 else torch.empty(0, xt.shape[1])


reference_model = None
reference_args = {
    "Nx": 2,
    "Nt": 5,
    "Qx": 30,
    "Qt": 50,
    "Jn": 250,
    "Nb": 1,
    'type': "STC", }

param_sets_groups = [
    [
        {"Nx": 2, "Nt": 1, "Qx": 30, "Qt": 50, "Jn": 250, "Nb": 1, "type": "STC"},
        {"Nx": 2, "Nt": 2, "Qx": 30, "Qt": 50, "Jn": 250, "Nb": 1, "type": "STC"},
        {"Nx": 2, "Nt": 3, "Qx": 30, "Qt": 50, "Jn": 250, "Nb": 1, "type": "STC"},
        {"Nx": 2, "Nt": 4, "Qx": 30, "Qt": 50, "Jn": 250, "Nb": 1, "type": "STC"},
    ],
    [
        {"Nx": 2, "Nt": 5, "Qx": 30, "Qt": 50, "Jn": 50, "Nb": 1, "type": "STC"},
        {"Nx": 2, "Nt": 5, "Qx": 30, "Qt": 50, "Jn": 100, "Nb": 1, "type": "STC"},
        {"Nx": 2, "Nt": 5, "Qx": 30, "Qt": 50, "Jn": 150, "Nb": 1, "type": "STC"},
        {"Nx": 2, "Nt": 5, "Qx": 30, "Qt": 50, "Jn": 200, "Nb": 1, "type": "STC"},
    ],
    [
        {"Nx": 2, "Nt": 5, "Qx": 30, "Qt": 25, "Jn": 250, "Nb": 1, "type": "STC"},
        {"Nx": 2, "Nt": 5, "Qx": 30, "Qt": 30, "Jn": 250, "Nb": 1, "type": "STC"},
        {"Nx": 2, "Nt": 5, "Qx": 30, "Qt": 35, "Jn": 250, "Nb": 1, "type": "STC"},
        {"Nx": 2, "Nt": 5, "Qx": 30, "Qt": 40, "Jn": 250, "Nb": 1, "type": "STC"},
        {"Nx": 2, "Nt": 5, "Qx": 30, "Qt": 45, "Jn": 250, "Nb": 1, "type": "STC"},
    ]

]

group_labels = ["Convergence w.r.t Nt",
                "Convergence w.r.t Jn",
                "Convergence w.r.t Q", ]

if __name__ == "__main__":
    # # plot the initial condition
    # import matplotlib.pyplot as plt
    # import numpy as np
    #
    # x = torch.linspace(0, 8, 100).reshape(-1, 1)
    # y = func_h(torch.cat([x, torch.zeros_like(x)], dim=1))
    # x = x.detach().numpy()
    # y = y.detach().numpy()
    #
    # plt.plot(x, y)
    # plt.title("Initial Condition")
    # plt.xlabel("x")
    # plt.ylabel("u(x, 0)")
    # plt.grid()
    # plt.show()
    # torch.set_default_device('cuda') if torch.cuda.is_available() else torch.set_default_device('cpu')
    run_rfm(args=argparse.Namespace(**reference_args), reference=True)

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
