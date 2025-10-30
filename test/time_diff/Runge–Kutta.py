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


def func_F(model: pyrfm.RFMBase, x, t):
    alpha = torch.pi / 2.0
    return alpha ** (-2) * model.dForward(x, order=[2])


if __name__ == "__main__":
    torch.set_default_device('cuda') if torch.cuda.is_available() else torch.set_default_device('cpu')
    time1 = time.time()
    domain = pyrfm.Line1D(x1=0.0, x2=12.0)
    model = pyrfm.RFMBase(dim=1, domain=domain, n_subdomains=2, n_hidden=20)

    x_in = domain.in_sample(200, with_boundary=False)
    x_on = domain.on_sample(2)

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
    #
    # exit(0)

    # c = torch.tensor([0.0, 1.0])
    # a = torch.tensor([
    #     [0.0, 0.0],
    #     [1.0, 0.0],
    # ])
    # b = torch.tensor([0.5, 0.5])

    dt = 1e-2
    t_k = 0
    T = 1.0

    x_all = torch.cat([x_in, x_on], dim=0)
    features = model.features(x_in).cat(dim=1)
    features_all = model.features(x_all).cat(dim=1)

    alpha = torch.pi / 2.0
    F_feature = alpha ** (-2) * model.features_second_derivative(x_in, axis1=0, axis2=0).cat(dim=1)


    def func(model, x, t):
        return F_feature @ model.W


    U0 = func_h(torch.cat([x_all, torch.zeros((x_all.shape[0], 1))], dim=1))
    model.compute(features_all, damp=1e-14).solve(U0)

    # provide a model to fast approximate at x_in
    d_model = model.clone()
    d_model.compute(features, damp=1e-14)

    n_steps = round(T / dt)
    u0 = U0[:x_in.shape[0]]
    rk = pyrfm.RK23()
    for k in range(n_steps):
        K, u = rk.rk_step(func, x_in, t_k, dt, u0, d_model)
        # concat boundary condition
        U = torch.cat([u, func_g(torch.cat([x_on, (t_k + dt) * torch.ones((x_on.shape[0], 1))], dim=1))], dim=0)
        model.solve(U, verbose=False)
        d_model.W = model.W.clone()
        u0 = d_model(x_in)
        t_k += dt

    xtk = torch.cat([x_all, t_k * torch.ones((x_all.shape[0], 1))], dim=1)
    print(
        "error at t = {:.4f}: {:.3e}".format(t_k, (model(x_all) - func_u(xtk)).norm(2) / func_u(xtk).norm(2)))

    time2 = time.time()
    print("Total time: {:.2f} seconds".format(time2 - time1))

    exit(0)

    # n_steps = round(T / dt)
    #
    # for i in range(n_steps):
    #     def func_F(model, x, t):
    #         return F_feature @ model.W
    #     K = [model(x_all)[:x_in.shape[0]]]
    #
    #     p = 0
    #
    #     d_model.solve(K[0], verbose=False)
    #     K.append(func_F(d_model, x_in, t_k))
    #
    #     p = 1
    #     d_model.solve(K[0] + dt * a[p, 0] * K[1], verbose=False)
    #     K.append(func_F(d_model, x_in, t_k + c[p] * dt))
    #
    #     u = K[0] + dt * (b[0] * K[1] + b[1] * K[2])
    #
    #     u = torch.cat([u, func_g(torch.cat([x_on, (t_k + dt) * torch.ones((x_on.shape[0], 1))], dim=1))], dim=0)
    #     model.solve(u, verbose=False)
    #     t_k += dt

    # Ds: List[torch.Tensor] = []
    # p = 0
    # t_p = t_k + c[p] * dt
    # # Dp = func_F(model, x_in, t_p)
    # Dp = F_feature @ model.W
    # Ds.append(Dp.clone())
    #
    # p = 1
    # t_p = t_k + c[p] * dt
    # d_model.solve(Ds[0], verbose=False)
    # # Dp = func_F(model, x_in, t_p) + dt * a[p, 0] * func_F(d_model, x_in, t_p)
    # Dp = F_feature @ (model.W + dt * a[p, 0] * d_model.W)
    # Ds.append(Dp.clone())
    #
    # u = model.forward(x_in) + dt * (b[0] * Ds[0] + b[1] * Ds[1])
    # u = torch.cat([u, func_g(torch.cat([x_on, (t_k + dt) * torch.ones((x_on.shape[0], 1))], dim=1))], dim=0)
    # model.solve(u, verbose=False)
    #
    # t_k += dt

    xtk = torch.cat([x_all, t_k * torch.ones((x_all.shape[0], 1))], dim=1)
    print(
        "error at t = {:.4f}: {:.3e}".format(t_k, torch.norm(features_all @ model.W - func_u(xtk), 2) / torch.norm(
            func_u(xtk), 2)))

    time2 = time.time()
    print("Total time: {:.2f} seconds".format(time2 - time1))
