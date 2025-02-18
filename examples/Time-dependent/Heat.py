# -*- coding: utf-8 -*-
"""
Created on 2025/2/16

@author: Yifei Sun

"""

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


def func_u(x, t):
    return 2 * torch.sin(torch.tensor(torch.pi / 2) * x) * torch.exp(-t)


def func_f(x, t):
    return torch.zeros(x.shape[0], 1)


def func_g1(t):
    return torch.zeros(t.shape[0], 1)


def func_g2(t):
    return 2 * torch.sin(torch.tensor(torch.pi / 2 * 12)) * torch.exp(-t)


def func_h(x):
    return 2 * torch.sin(torch.tensor(torch.pi / 2) * x)


param_sets = [
    {"Nx": 2, "Nt": 2, "Qx": 20, "Qt": 20, "Jn": 50, "Nb": 1},
    {"Nx": 2, "Nt": 2, "Qx": 20, "Qt": 20, "Jn": 100, "Nb": 1},
    {"Nx": 2, "Nt": 2, "Qx": 20, "Qt": 20, "Jn": 200, "Nb": 1},
    {"Nx": 2, "Nt": 2, "Qx": 20, "Qt": 20, "Jn": 400, "Nb": 1},
]


def run_rfm(args):
    model = pyrfm.STRFMBase(1, 50, [0, 1], [0, 10], 5, 3, 1,
                            st_type="SoV")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve the heat equation using RFM")
    parser.add_argument("--Nx", type=int, required=True)
    parser.add_argument("--Nt", type=int, required=True)
    parser.add_argument("--Qx", type=int, required=True)
    parser.add_argument("--Qt", type=int, required=True)
    parser.add_argument("--Jn", type=int, required=True)
    parser.add_argument("--Nb", type=int, required=True)

    if len(sys.argv) == 1:
        for param_set in param_sets:
            args = argparse.Namespace(**param_set)
            run_rfm(args)
    else:
        args = parser.parse_args()
        run_rfm(args)
