# -*- coding: utf-8 -*-
"""
Created on 2025/10/28

@author: Yifei Sun
"""
import pyrfm
import torch
import os
import argparse
import sys
import time
from typing import List

from pyrfm import RFMBase

from matplotlib import pyplot as plt

"""
Consider the following problem:

    ∂ₜ u(x, t) = 1 / α² * ∂ₓ² u(x, t),  x ∈ [x₀, x₁],  t ∈ [0, T],
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


def func_F(model: RFMBase, x, t):
    alpha = torch.pi / 2.0
    return alpha ** (-2) * model.dForward(x, order=[2])


if __name__ == "__main__":
    torch.set_default_device('cuda') if torch.cuda.is_available() else torch.set_default_device('cpu')
    time1 = time.time()
    domain = pyrfm.Line1D(x1=0.0, x2=12.0)
    model = pyrfm.RFMBase(dim=1, domain=domain, n_subdomains=4, n_hidden=20)

    x_in = domain.in_sample(400, with_boundary=False)
    x_on = domain.on_sample(2)

    c = torch.tensor([0.0, 1.0])
    a = torch.tensor([
        [0.0, 0.0],
        [1.0, 0.0],
    ])
    b = torch.tensor([0.5, 0.5])

    dt = 1e-3
    t_k = 0
    T = 10.0

    x_all = torch.cat([x_in, x_on], dim=0)
    xt0 = torch.cat([x_all, torch.zeros((x_all.shape[0], 1))], dim=1)
    model.compute(model.features(x_all).cat(dim=1), damp=1e-14).solve(func_h(xt0))

    # for damp in [0.0, 1e-16, 1e-14, 1e-12, 1e-10, 1e-8]:
    #     features = model.features(x_all).cat(dim=1)
    #     model.compute(features, damp=damp).solve(func_h(xt0)) if damp > 0 else model.compute(features).solve(
    #         func_h(xt0))
    #     errors = []
    #     for i in range(10000):
    #         model.solve(model(x_all), verbose=False)
    #         errors.append(torch.norm(model.forward(x_all) - func_h(xt0)) / torch.norm(func_h(xt0)))
    #
    #     plt.plot([err.cpu() for err in errors], label="damp={:.0e}".format(damp))
    # plt.yscale("log")
    # plt.xlabel("Iteration")
    # plt.ylabel("Relative Error")
    # plt.ylim(None, 1e8)
    # plt.legend()
    # plt.title("Convergence of RFM with Different Damping Factors")
    # plt.grid()
    # plt.savefig("convergence.png", dpi=600)
    # plt.show()

    # print("error = {:.3e}".format(torch.norm(model.forward(x_all) - func_h(xt0)) / torch.norm(func_h(xt0))))
    # for i in range(100000):
    #     model.solve(model(x_all), verbose=False)
    #     t_k += dt
    # print("error = {:.3e}".format(torch.norm(model.forward(x_all) - func_h(xt0)) / torch.norm(func_h(xt0))))
    # t_k = 0.0

    # exit(0)

    features_all = model.features(x_all).cat(dim=1)

    d_model = model.clone()
    d_model.compute(d_model.features(x_in).cat(dim=1), damp=1e-14)

    alpha = torch.pi / 2.0
    F_feature = alpha ** (-2) * model.features_second_derivative(x_in, axis1=0, axis2=0).cat(dim=1)

    while t_k <= T * (1 - torch.finfo(float).eps):
        Ds: List[torch.Tensor] = []
        p = 0
        t_p = t_k + c[p] * dt
        # Dp = func_F(model, x_in, t_p)
        Dp = F_feature @ model.W
        Ds.append(Dp.clone())

        p = 1
        t_p = t_k + c[p] * dt
        d_model.solve(Ds[0], verbose=False)
        # Dp = func_F(model, x_in, t_p) + dt * a[p, 0] * func_F(d_model, x_in, t_p)
        Dp = F_feature @ (model.W + dt * a[p, 0] * d_model.W)
        Ds.append(Dp.clone())

        u = model.forward(x_in) + dt * (b[0] * Ds[0] + b[1] * Ds[1])
        u = torch.cat([u, func_g(torch.cat([x_on, (t_k + dt) * torch.ones((x_on.shape[0], 1))], dim=1))], dim=0)
        model.solve(u, verbose=False)

        t_k += dt

    xtk = torch.cat([x_all, t_k * torch.ones((x_all.shape[0], 1))], dim=1)
    print(
        "error at t = {:.4f}: {:.3e}".format(t_k, torch.norm(features_all @ model.W - func_u(xtk), 2) / torch.norm(
            func_u(xtk), 2)))

    time2 = time.time()
    print("Total time: {:.2f} seconds".format(time2 - time1))
