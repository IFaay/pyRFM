# -*- coding: utf-8 -*-
"""
Created on 2025/2/13

@author: Yifei Sun

"""

import pyrfm
import torch
import os
import argparse
import sys
import time

"""
Example 3.1 (Helmholtz equation)

Consider the one-dimensional Helmholtz equation with Dirichlet boundary condition over Ω = [0, 8]:

    d²u(x)/dx² - λu(x) = f(x),   for x ∈ Ω

with Dirichlet boundary conditions:

    u(0) = c₁,   u(8) = c₂

Once an explicit form of u is given, c₁, c₂, and f can be computed.
"""

Lambda = -4

def func_u(x):
    return torch.sin(3 * torch.pi * x[:, [0]] + 3 * torch.pi / 20) * torch.cos(
        2 * torch.pi * x[:, [0]] + torch.pi / 10) + 2

def func_f(x):
    x.requires_grad = True
    u = func_u(x)
    ux = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    uxx = torch.autograd.grad(ux, x, grad_outputs=torch.ones_like(ux), create_graph=False)[0]

    return uxx - Lambda * u

def func_g(x):
    return func_u(x)

param_sets = [
    {"Q": 200, "M": 200, "psi": "A"},
    {"Q": 400, "M": 400, "psi": "A"},
    {"Q": 800, "M": 800, "psi": "A"},
    {"Q": 1600, "M": 1600, "psi": "A"},
    {"Q": 200, "M": 200, "psi": "B"},
    {"Q": 400, "M": 400, "psi": "B"},
    {"Q": 800, "M": 800, "psi": "B"},
    {"Q": 1600, "M": 1600, "psi": "B"},
]


def run_rfm(args):
    print("\n" + "=" * 40)
    print(f"Simulation Started with Parameters:")
    print(f"Q = {args.Q}, M = {args.M}, Psi Type = {args.psi}")
    print(f"--------------------------")

    torch.set_default_device('cuda') if torch.cuda.is_available() else torch.set_default_device('cpu')
    start_time = time.time()
    domain = pyrfm.Line1D(x1=0.0, x2=8.0)
    if args.psi == "A":
        model = pyrfm.RFMBase(dim=1, n_hidden=50, domain=domain, n_subdomains=args.M / 50, pou=pyrfm.PsiA)
        x_in = domain.in_sample(args.Q, with_boundary=False)
        x_on = domain.on_sample(2)
        A_in = model.features(x_in).cat(dim=1)
        A_in_xx = model.features_second_derivative(x_in, axis1=0, axis2=0).cat(dim=1)
        A_on = model.features(x_on).cat(dim=1)
        A_interface = model.add_c_condition((args.M / 50 - 1)).cat(dim=1)

        A = pyrfm.concat_blocks([[A_in_xx - Lambda * A_in], [A_on], [A_interface]])

        f_in = func_f(x_in).view(-1, 1)
        f_on = func_g(x_on).view(-1, 1)

        f = pyrfm.concat_blocks([[f_in], [f_on], [torch.zeros(A_interface.shape[0], 1)]])
        model.compute(A).solve(f)

        x_test = domain.in_sample(40, with_boundary=True)
        u_test = func_u(x_test)
        u_pred = model(x_test)
        error = (u_test - u_pred).norm() / u_test.norm()

    elif args.psi == "B":
        model = pyrfm.RFMBase(dim=1, n_hidden=50, domain=domain, n_subdomains=args.M // 50,
                              pou=pyrfm.PsiB)
        x_in = domain.in_sample(args.Q, with_boundary=False)
        x_on = domain.on_sample(2)
        A_in = model.features(x_in).cat(dim=1)
        A_in_xx = model.features_second_derivative(x_in, axis1=0, axis2=0).cat(dim=1)
        A_on = model.features(x_on).cat(dim=1)
        A = pyrfm.concat_blocks([[A_in_xx - Lambda * A_in], [A_on]])
        f_in = func_f(x_in).view(-1, 1)
        f_on = func_g(x_on).view(-1, 1)
        f = pyrfm.concat_blocks([[f_in], [f_on]])
        model.compute(A).solve(f, check_condition=True)

        x_test = domain.in_sample(40, with_boundary=True)
        u_test = func_u(x_test)
        u_pred = model(x_test)
        error = (u_test - u_pred).norm() / u_test.norm()

    else:
        raise ValueError("Invalid psi")

    print(f"\nSimulation Results:")
    print(f"--------------------------")
    print(f"Problem size: N = {A.shape[0]}, M = {A.shape[1]}, Partial of Unity type = {args.psi}")
    print(f"Relative Error: {error:.4e}")
    print(f"Elapsed Time: {time.time() - start_time:.2f} seconds")
    print("=" * 40)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--Q", type=int, required=True)
    parser.add_argument("--M", type=int, required=True)
    parser.add_argument("--psi", type=str, required=True)

    if len(sys.argv) == 1:
        for param_set in param_sets:
            args = argparse.Namespace(**param_set)
            run_rfm(args)
    else:
        args = parser.parse_args()
        run_rfm(args)


