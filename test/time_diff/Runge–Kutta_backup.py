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


def func_F(model: RFMBase, xt):
    x = xt[:, :-1]
    alpha = torch.pi / 2.0
    return alpha ** (-2) * model.dForward(x, order=[2]) + func_f(xt)


if __name__ == "__main__":
    domain = pyrfm.Line1D(x1=0.0, x2=12.0)
    model = pyrfm.RFMBase(dim=1, domain=domain, n_subdomains=4, n_hidden=15)

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
    T = 1.0

    x_all = torch.cat([x_in, x_on], dim=0)
    xt0 = torch.cat([x_all, torch.zeros((x_all.shape[0], 1))], dim=1)
    features = model.features(x_all).cat(dim=1)
    model.compute(features, damp=1e-10).solve(func_h(xt0))
    print("error = {:.3e}".format(torch.norm(model.forward(x_all) - func_h(xt0))))
    #
    # while t_k < T:
    #     model.solve(model(x_all))
    #     t_k += dt
    #
    # print("error = {:.3e}".format(torch.norm(model.forward(x_all) - func_h(xt0))))

    u_model = model.clone()
    features = u_model.features(x_in).cat(dim=1)
    u_model.compute(features, damp=1e-10)
    while t_k <= T:
        Ds: List[torch.Tensor] = []
        p = 0
        t_p = t_k + c[p] * dt
        xt_p = torch.cat([x_in, t_p * torch.ones((x_in.shape[0], 1))], dim=1)
        Dp = func_F(model, xt_p)
        Ds.append(Dp.clone())

        p = 1
        t_p = t_k + c[p] * dt
        xt_p = torch.cat([x_in, t_p * torch.ones((x_in.shape[0], 1))], dim=1)
        term = Ds[0]
        # term = u_model.forward(x_in) + dt * a[p, 0] * Ds[0]
        # term = torch.cat([term, func_g(torch.cat([x_on, t_p * torch.ones((x_on.shape[0], 1))], dim=1))], dim=0)
        u_model.solve(term)
        Dp = func_F(model, xt_p) + dt * a[p, 0] * func_F(u_model, xt_p)
        Ds.append(Dp.clone())

        u = model.forward(x_in) + dt * (b[0] * Ds[0] + b[1] * Ds[1])
        u = torch.cat([u, func_g(torch.cat([x_on, (t_k + dt) * torch.ones((x_on.shape[0], 1))], dim=1))], dim=0)
        model.solve(u)

        t_k += dt

        xtk = torch.cat([x_all, t_k * torch.ones((x_all.shape[0], 1))], dim=1)
        print("error at t = {:.2f}: {:.3e}".format(t_k, torch.norm(model.forward(x_all) - func_u(xtk), 2) / torch.norm(
            func_u(xtk), 2)))
