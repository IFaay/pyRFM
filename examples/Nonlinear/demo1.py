# -*- coding: utf-8 -*-
"""
Created on 2025/2/23

@author: Yifei Sun
"""

import pyrfm
import torch
import os
import argparse
import sys
import time

from scipy.optimize import least_squares
import numpy as np

"""
Consider the nonlinear equation with Dirichlet boundary condition over Ω = [0,1] × [0,1]:

     - Δu(x, y) + u²(x, y)  = f(x, y),    (x, y) ∈ Ω

with boundary conditions:

    u(x, 0) = g1(x),   u(x, 1) = g2(x)
    u(0, y) = h1(y),   u(1, y) = h2(y)

Once an explicit form of u is given, the functions g1, g2, h1, h2, and f can be computed.

"""


def func_u(x):
    # x is a torch tensor of shape (N, 2)
    return -0.5 * (1.5 * torch.cos(torch.pi * x[:, [0]] + 2 * torch.pi / 5) +
                   2 * torch.cos(2 * torch.pi * x[:, [0]] - torch.pi / 5)) * \
        (1.5 * torch.cos(torch.pi * x[:, [1]] + 2 * torch.pi / 5) +
         2 * torch.cos(2 * torch.pi * x[:, [1]] - torch.pi / 5)) - \
        0.5 * (1.5 * torch.cos(2 * torch.pi * x[:, [0]] + 2 * torch.pi / 5) +
               2 * torch.cos(4 * torch.pi * x[:, [0]] - torch.pi / 5)) * \
        (1.5 * torch.cos(2 * torch.pi * x[:, [1]] + 2 * torch.pi / 5) +
         2 * torch.cos(4 * torch.pi * x[:, [1]] - torch.pi / 5))


def func_f(x):
    u = func_u(x)
    return -(-0.5 * (-1.5 * torch.pi ** 2 * torch.cos(torch.pi * x[:, [0]] + 2 * torch.pi / 5) -
                     2 * (2 * torch.pi) ** 2 * torch.cos(2 * torch.pi * x[:, [0]] - torch.pi / 5)) * \
             (1.5 * torch.cos(torch.pi * x[:, [1]] + 2 * torch.pi / 5) +
              2 * torch.cos(2 * torch.pi * x[:, [1]] - torch.pi / 5)) - \
             0.5 * (1.5 * torch.cos(torch.pi * x[:, [0]] + 2 * torch.pi / 5) +
                    2 * torch.cos(2 * torch.pi * x[:, [0]] - torch.pi / 5)) * \
             (-1.5 * torch.pi ** 2 * torch.cos(torch.pi * x[:, [1]] + 2 * torch.pi / 5) -
              2 * (2 * torch.pi) ** 2 * torch.cos(2 * torch.pi * x[:, [1]] - torch.pi / 5)) - \
             0.5 * (-1.5 * (2 * torch.pi) ** 2 * torch.cos(2 * torch.pi * x[:, [0]] + 2 * torch.pi / 5) -
                    2 * (4 * torch.pi) ** 2 * torch.cos(4 * torch.pi * x[:, [0]] - torch.pi / 5)) * \
             (1.5 * torch.cos(2 * torch.pi * x[:, [1]] + 2 * torch.pi / 5) +
              2 * torch.cos(4 * torch.pi * x[:, [1]] - torch.pi / 5)) - \
             0.5 * (1.5 * torch.cos(2 * torch.pi * x[:, [0]] + 2 * torch.pi / 5) +
                    2 * torch.cos(4 * torch.pi * x[:, [0]] - torch.pi / 5)) * \
             (-1.5 * (2 * torch.pi) ** 2 * torch.cos(2 * torch.pi * x[:, [1]] + 2 * torch.pi / 5) -
              2 * (4 * torch.pi) ** 2 * torch.cos(4 * torch.pi * x[:, [1]] - torch.pi / 5))) + u ** 2


def func_g(x):
    return func_u(x)


if __name__ == '__main__':
    torch.set_default_device('cuda') if torch.cuda.is_available() else torch.set_default_device('cpu')

    start_time = time.time()
    domain = pyrfm.Square2D(center=[0.5, 0.5], radius=[0.5, 0.5])
    model = pyrfm.RFMBase(dim=2, n_hidden=200, domain=domain, n_subdomains=2, pou=pyrfm.PsiB)

    x_in = domain.in_sample(10000, with_boundary=False)

    x_on = domain.on_sample(2000)

    A_in = model.features(x_in).cat(dim=1)
    A_in_xx = model.features_second_derivative(x_in, axis1=0, axis2=0).cat(dim=1)
    A_in_yy = model.features_second_derivative(x_in, axis1=1, axis2=1).cat(dim=1)

    A_on = model.features(x_on).cat(dim=1)

    f_in = func_f(x_in)
    f_on = func_g(x_on)


    def fcn(w):
        u = A_in @ w
        u_xx = A_in_xx @ w
        u_yy = A_in_yy @ w
        u_on = A_on @ w
        return torch.cat([(-u_xx - u_yy + u ** 2) - f_in, u_on - f_on])


    def jac(w):
        return torch.cat([-A_in_xx - A_in_yy + 2 * (A_in @ w) * A_in, A_on], dim=0)


    # tol = torch.finfo(torch.float64).eps
    tol = 1e-8
    result = pyrfm.nonlinear_least_square(fcn=fcn,
                                          x0=torch.zeros((A_in.shape[1], 1)),
                                          jac=jac,
                                          ftol=tol,
                                          gtol=tol,
                                          xtol=tol,
                                          method='newton')

    status = result[1]

    if status == 0:
        print("The maximum number of function evaluations is exceeded.")
    elif status == 1:
        print("gtol termination condition is satisfied.")
    elif status == 2:
        print("ftol termination condition is satisfied.")
    elif status == 3:
        print("xtol termination condition is satisfied.")
    elif status == 4:
        print("Both ftol and xtol termination conditions are satisfied.")
    else:
        print("Unknown status.")

    model.W = result[0]

    x_test = domain.in_sample(2000, with_boundary=True)
    u_test = func_u(x_test).view(-1, 1)
    u_pred = model(x_test)

    print((u_test - u_pred).norm() / u_test.norm())

    print('Time:', time.time() - start_time, ", with", torch.tensor(0.).device)
