# -*- coding: utf-8 -*-
"""
Created on 2026/1/12

@author: Xurong Chi
"""
import time
import argparse

from fontTools.ttLib.reorderGlyphs import reorderGlyphs

import numpy as np
import pyrfm
import torch
import os
import sys

import warnings

# 全局忽略所有警告
warnings.filterwarnings('ignore')
"""
Consider Stokes problem defined by the system:

    -νΔu(x) + ∇p(x) = f_s(x)     for x in Ω_S,
    ∇·u(x) = 0                   for x in Ω_S,
    u(x) = U(x) = 0               for x on ∂Ω_S.

And Darcy problem defined by the system:

    u(x) = -K·∇p(x)              for x in Ω_D,
    ∇·u(x) = f_d(x)              for x in Ω_D,
    u(x)·n(x) = 0                 for x on ∂Ω_D.

In this problem, the pressure p is only determined up to a constant. 
To avoid difficulties, we fix the value of p at the left-bottom corner.

Ω_D is the square (0, 1) × (0, 1), Ω_S is the square (0, 1) × (1, 2)..
"""

K11 = 0.505 * 1e-4
K12 = -0.495 * 1e-4
K21 = -0.495 * 1e-4
K22 = 0.505 * 1e-4
nu = 1e-6


def global_compute_solve(model, A: torch.Tensor, b: torch.Tensor, check_condition=False):
    """
    Compute the QR decomposition of matrix A.

    :param A: Input matrix.


    Solve the linear system Ax = b using the QR decomposition.

    :param b: Right-hand side tensor.
    :param check_condition: Whether to check the condition number of A, and switch to SVD if necessary.
    """

    A = A.to(dtype=model.dtype, device=model.device)
    A_norm = torch.linalg.norm(A, ord=2, dim=1, keepdim=True)
    A /= A_norm
    print("Decomposing the problem size of A: ", A.shape, "with solver QR")

    A, tau = torch.geqrf(A)

    b = b.view(-1, 1).to(dtype=model.dtype, device=model.device)
    if A.shape[0] != b.shape[0]:
        raise ValueError("Input dimension mismatch.")
    b /= A_norm

    y = torch.ormqr(A, tau, b, transpose=True)[:A.shape[1]]
    W = torch.linalg.solve_triangular(A[:A.shape[1], :], y, upper=True)
    b_ = torch.ormqr(A, tau, torch.matmul(torch.triu(A), W), transpose=False)
    residual = torch.norm(b_ - b) / torch.norm(b)

    print(f"Least Square Relative residual: {residual:.4e}")
    return (W)


def func_s_u(x):
    u = -x[:, [1]] * (x[:, [1]] - 0.3)
    return u


def func_s_v(x):
    v = 0 * x[:, [0]]
    return v


def func_s_f(x):
    f1 = x[:, [0]] * 0
    f2 = x[:, [1]] * 0
    return f1, f2


def func_d_f(x):
    f = 0 * x[:, [1]]
    return f


def run_rfm(args):
    print("\n" + "=" * 40)
    print(f"Simulation Started with Parameters:")
    print(f"Q = {args.Q}, M = {args.M}")
    print(f"--------------------------")

    torch.set_default_device('cuda') if torch.cuda.is_available() else torch.set_default_device('cpu')

    ''' Stokes Problem '''
    start_time = time.time()
    domain_s = pyrfm.Square2D(center=(1.3, 0.15), radius=(1.3, 0.15)) - pyrfm.Square2D(center=(0.5, 0.1),
                                                                                       radius=(0.1, 0.1))

    model_s = pyrfm.RFMBase(dim=2, n_hidden=args.M, domain=domain_s, n_subdomains=[4, 1])
    x_in_s = domain_s.in_sample(args.Q, with_boundary=False)

    A_s = model_s.features(x_in_s).cat(dim=1)
    b_s = x_in_s[:, 0:1] * x_in_s[:, 1:2]

    model_s.compute(A_s).solve(b_s)

    visualizer = pyrfm.RFMVisualizer2D(model_s, resolution=(800, 800))
    visualizer.plot()
    visualizer.show()


param_sets = [{"Q": 400, "M": 100}, ]
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--Q", type=int, required=True)
    parser.add_argument("--M", type=int, required=True)

    if len(sys.argv) == 1:
        for param_set in param_sets:
            args = argparse.Namespace(**param_set)
            run_rfm(args)
    else:
        args = parser.parse_args()
        run_rfm(args)
