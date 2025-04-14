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
Consider the burgers equation with Dirichlet boundary condition over (x, t) ∈ [-1, 1] × [0, 1]:

    du(x)/dt + u(x) * du(x)/dx - v * d²u(x)/dx² = 0,   (x, y) ∈ Ω    x ∈ [-1, 1], t ∈ [0, 1]

with boundary conditions:

    u(0, x) = -sin(pi*x),   
    u(t, -1) = u(t, 1) = 0

"""

v = 0.1

if __name__ == '__main__':
    torch.set_default_device('cuda') if torch.cuda.is_available() else torch.set_default_device('cpu')
    start_time = time.time()
    domain = pyrfm.Square2D(center=[0.5, 0.0], radius=[0.5, 1.0])
    model = pyrfm.RFMBase(dim=2, n_hidden=200, domain=domain, n_subdomains=4, pou=pyrfm.PsiB)

    x_in = domain.in_sample(10000, with_boundary=False)

    left_points = 1000
    up_points = 1000
    down_points = 1000

    x = torch.zeros(left_points)
    y = torch.rand(left_points) * 2 - 1
    x_left = torch.column_stack((x, y))
    x_left_tensor = x_left

    x = torch.rand(up_points)
    y = torch.ones(up_points)
    x_up = torch.column_stack((x, y))
    x_up_tensor = x_up

    x = torch.rand(up_points)
    y = - torch.ones(up_points)
    x_down = torch.column_stack((x, y))
    x_down_tensor = x_down

    A_in = model.features(x_in).cat(dim=1)
    A_in_t = model.features_derivative(x_in, axis=0).cat(dim=1)
    A_in_x = model.features_derivative(x_in, axis=1).cat(dim=1)
    A_in_xx = model.features_second_derivative(x_in, axis1=1, axis2=1).cat(dim=1)

    A_left = model.features(x_left_tensor).cat(dim=1)
    A_up = model.features(x_up_tensor).cat(dim=1)
    A_down = model.features(x_down_tensor).cat(dim=1)

    f_in = 0
    f_left = - torch.sin(torch.pi * x_left_tensor[:, -1:])
    f_up = 0
    f_down = 0


    def fcn(w):
        u = A_in @ w
        u_t = A_in_t @ w
        u_x = A_in_x @ w
        u_xx = A_in_xx @ w
        u_left = A_left @ w
        u_up = A_up @ w
        u_down = A_down @ w
        return torch.cat([u_t + u * u_x - v * u_xx, u_left - f_left, u_up - f_up, u_down - f_down])


    def jac(w):
        return torch.cat([A_in_t + (A_in @ w) * A_in_x + (A_in_x @ w) * A_in - v * A_in_xx, A_left, A_up, A_down],
                         dim=0)


    # tol = torch.finfo(torch.float64).eps
    tol = 1e-6
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

    data = np.load("burgers_data_in.npy")
    x_test = data[:, :2]
    x_test_tensor = torch.from_numpy(x_test).float()
    u_test = data[:, -1]
    u_test_tensor = torch.from_numpy(u_test).float().view(-1, 1)
    u_pred = model(x_test_tensor)

    print((u_test - u_pred).norm() / u_test.norm())

    print('Time:', time.time() - start_time, ", with", torch.tensor(0.).device)
