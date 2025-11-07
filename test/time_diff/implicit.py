# -*- coding: utf-8 -*-
"""
Created on 2025/10/30

@author: Yifei Sun
"""
from pyexpat import features

import pyrfm
import torch
import os
import argparse
import sys
import time
from typing import List

"""
Consider the following 2D heat equation problem:

    ∂ₜ u(x, y, t) = α (∂ₓ² u(x, y, t) + ∂ᵧ² u(x, y, t)),   (x, y) ∈ (0, 1)²,  t ∈ [0, T],

with homogeneous Dirichlet boundary conditions

    u(x, y, t) = 0,   for (x, y) ∈ ∂Ω = {x=0, x=1 or y=0, y=1},   t ∈ [0, T],

and initial condition

    u(x, y, 0) = sin(πx) sin(πy),   (x, y) ∈ (0, 1)².

where α = 1/(2π²).

The exact solution is chosen to be

    uₑ(x, y, t) = exp(-t) sin(πx) sin(πy),

which satisfies the heat equation and vanishes on the boundary.
Here T = 1 is used in numerical experiments.

BDF1 (Backward Euler) scheme is used for time discretization:

 (uⁿ⁺¹ − uⁿ)/Δt = (1 / 2α²) * Δ uⁿ⁺¹,

which leads to the linear system

    uⁿ⁺¹ − (1 / 2α²) *  Δt Δ uⁿ⁺¹ = uⁿ.

BDF2 scheme (second-order backward differentiation)

(3uⁿ⁺¹ − 4uⁿ + uⁿ⁻¹) / (2Δt) = (1 / 2α²) * Δ uⁿ⁺¹,

equivalently,

    (3/2) uⁿ⁺¹ − (1 / 2α²) *  ΔtΔ uⁿ⁺¹ = (4uⁿ − uⁿ⁻¹) / 2,

"""


def func_u(xt):
    x = xt[:, :-1]
    t = xt[:, -1:]
    return torch.exp(-t) * torch.sin(torch.pi * x[:, 0:1]) * torch.sin(torch.pi * x[:, 1:2])


def func_g(xt):
    return torch.zeros(xt.shape[0], 1)


if __name__ == "__main__":
    torch.set_default_device('cuda') if torch.cuda.is_available() else torch.set_default_device('cpu')
    time1 = time.time()

    domain = pyrfm.Square2D(center=[0.5, 0.5], half=[0.5, 0.5])
    model = pyrfm.RFMBase(dim=2, domain=domain, n_subdomains=2, n_hidden=100)

    x_in = domain.in_sample(2000)
    x_on = domain.on_sample(200)

    # featrues = model.features(x_in).cat(dim=1)
    # # print the singular values of the feature matrix
    # U, S, Vh = torch.linalg.svd(featrues, full_matrices=False)
    # # print("Singular values of the feature matrix:", S)
    # import matplotlib.pyplot as plt
    #
    # plt.plot(S.cpu().numpy())
    # plt.yscale('log')
    # plt.title('Singular values of the feature matrix')
    # plt.xlabel('Index')
    # plt.ylabel('Singular value (log scale)')
    # plt.grid()
    # plt.show()

    x_all = torch.cat([x_in, x_on], dim=0)
    x_all_t0 = torch.cat([x_all, torch.zeros(x_all.shape[0], 1)], dim=1)

    dt = 1e-5
    t_k = 0
    T = 1.0

    n_steps = round(T / dt)

    u0 = func_u(x_all_t0)

    # features = model.features(x_in).cat(dim=1)
    # features_laplace = (model.features_second_derivative(x_in, axis1=0, axis2=0).cat(dim=1)
    #                     + model.features_second_derivative(x_in, axis1=1, axis2=1).cat(dim=1))
    #
    # D = model.compute(features).solve(features_laplace)

    feature_in = model.features(x_in).cat(dim=1) - 1 / (2 * torch.pi ** 2) * dt * (
            model.features_second_derivative(x_in, axis1=0, axis2=0)
            + model.features_second_derivative(x_in, axis1=1, axis2=1)).cat(dim=1)
    feature_on = model.features(x_on).cat(dim=1)

    model.compute(torch.cat([feature_in, feature_on], dim=0), damp=1e-14)

    model.solve(u0)

    for step in range(n_steps):
        u_in = model(x_in)
        u_exact = func_u(torch.cat([x_in, t_k * torch.ones(x_in.shape[0], 1)], dim=1))
        error = torch.norm(u_in - u_exact) / torch.norm(u_exact)
        print(f"t={t_k:.3e}, Relative L2 error : {error.item():.3e}")

        u = torch.cat([u_in, func_g(x_on)], dim=0)
        model.solve(u, verbose=False)

        t_k += dt

    # u_pred = model(x_all)
    # u_exact = func_u(torch.cat([x_all, T * torch.ones(x_all.shape[0], 1)], dim=1))
    # error = torch.norm(u_pred - u_exact) / torch.norm(u_exact)
    # print(f"Relative L2 error at T={T}: {error.item():.3e}")
