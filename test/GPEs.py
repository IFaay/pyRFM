# -*- encoding: utf-8 -*-
'''
@File    :   GPEs.py
@Time    :   2025/08/13 22:30:14
@Author  :   zyliang
@Version :   latest
@Contact :   liangzhangyong1994@gmail.com
'''

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.append(parent_dir)

import pyrfm
import torch
import os
import argparse
import sys
import time
import scipy.io

"""
Non-linear Schrödinger Equation with potential in Unbounded Domain
Consider the following problem:

    i∂ₜψ(x, t) + 0.5Δψ(x, t) - V1(x)ψ(x, t) - beta1|ψ(x, t)|**2ψ(x, t) = 0,     for (x, t) ∈ [x₀, x₁] × [0, T],
    ψ(x, 0) = ψ0(x),                                                           for x ∈ [x₀, x₁],

where x₀ = -5, x₁ = 5, T = 10, V1(x)=0.5*gamma_z**2*x**2, beta1=10. The init solution is chosen to be:
    ψ0(x) = π**(-0.25)exp(-x**2/2).
"""


def func_psi(xt):
    data = scipy.io.loadmat(f'gpe.mat')
    psiN = torch.tensor(data['psiN'].T, dtype=torch.complex64)

    return psiN.reshape(-1, 1)


def func_psi0(xt):
    x = xt[:, :-1]
    return torch.pi ** (-1 / 4) * torch.exp(-0.5 * x ** 2)


def run_rfm(args):
    time_stamp = torch.linspace(start=0, end=1, steps=args.Nt + 1)
    domain = pyrfm.Line1D(x1=-6, x2=6)
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

    gamma_z = 2
    beta1 = 10
    for i, model in enumerate(models):
        model.dtype = torch.complex128 if model.dtype == torch.float64 else torch.complex64
        t0 = torch.tensor(model.time_interval[0]).reshape(-1, 1)
        t = torch.linspace(*model.time_interval, (args.Qt * args.Nt) + 1)[1:].reshape(-1, 1)

        x_in_t = model.validate_and_prepare_xt(x=x_in, t=t)
        x_t0 = model.validate_and_prepare_xt(x=torch.cat([x_in, x_on], dim=0), t=t0)

        A_init = model.features(xt=x_t0).cat(dim=1)
        A = model.features(xt=x_in_t).cat(dim=1)
        A_t = model.features_derivative(xt=x_in_t, axis=1).cat(dim=1)
        A_xx = model.features_second_derivative(xt=x_in_t, axis1=0, axis2=0).cat(dim=1)
        V1 = 0.5 * gamma_z ** 2 * x_in_t[:, 0:1] ** 2
        # A_pde = 1j * A_t + 0.5 * A_xx - V1 * A

        if i == 0:
            f_init = func_psi0(x_t0)
        else:
            f_init = models[i - 1].forward(xt=x_t0)

        f_pde = torch.zeros(x_in_t.shape[0], 1)

        def fcn(w):
            w = w.to(A.dtype)
            u = A @ w
            u_t = A_t @ w
            u_xx = A_xx @ w
            u_pde = -1j * u_t - 0.5 * u_xx + V1 * u + beta1 * torch.abs(u) ** 2 * u
            u_ic = A_init @ w
            return torch.cat([u_pde - f_pde, u_ic - f_init], dim=0)

        def jac(w):
            w = w.to(A.dtype)
            jac_pde = -1j * A_t - 0.5 * A_xx + V1 * A + beta1 * (
                    A * torch.abs(A @ w) ** 2 + (A @ w) * torch.conj(A @ w) * A)
            jac_ic = A_init
            return torch.cat([jac_pde, jac_ic], dim=0)

        # tol = torch.finfo(torch.float64).eps
        tol = 1e-10
        result = pyrfm.nonlinear_least_square(fcn=fcn,
                                              x0=torch.zeros((A.shape[1], 1)),
                                              jac=jac,
                                              ftol=tol,
                                              gtol=tol,
                                              xtol=tol,
                                              method='newton',
                                              verbose=2)
        model.W = result[0]

    x_test = torch.linspace(-6, 6, 257).reshape(-1, 1)
    u_pred = []
    u_truth = []
    for i, (t0, t1) in enumerate(zip(time_stamp[:-1], time_stamp[1:])):
        t_test = torch.linspace(t0, t1, 1001).reshape(-1, 1)
        x_t = models[i].validate_and_prepare_xt(x=x_test, t=t_test)
        u_pred.append(models[i].forward(xt=x_t))
        u_truth.append(func_psi(x_t))

    u_pred = torch.cat(u_pred, dim=0)
    u_truth = torch.cat(u_truth, dim=0)
    print("Relative Error: ", torch.linalg.norm(u_pred - u_truth) / torch.linalg.norm(u_truth))


param_sets_groups = [
    # Convergence w.r.t Nt
    [
        {"Nx": 5, "Nt": 1, "Qx": 30, "Qt": 30, "Jn": 30, "type": "STC"},
        # {"Nx": 5, "Nt": 2, "Qx": 30, "Qt": 30, "Jn": 300, "type": "STC"},
        # {"Nx": 5, "Nt": 3, "Qx": 30, "Qt": 30, "Jn": 300, "type": "STC"}
    ],
]

group_labels = [
    "Convergence w.r.t Nt",
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
                    f"Nx = {args.Nx}, Nt = {args.Nt}, Qx = {args.Qx}, Qt = {args.Qt}, Jn = {args.Jn}, type = {args.type}")
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
