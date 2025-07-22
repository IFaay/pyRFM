# -*- coding: utf-8 -*-
"""
Created on 2025/2/25

@author: Yifei Sun
"""
import matplotlib.pyplot as plt

import pyrfm
import torch
import os
import argparse
import sys
import time
import math

from pyrfm import RFMVisualizer2D

"""
Example 3.3 (Elasticity problem):
The two-dimensional elasticity problem with complex geometry:

-div(σ(u(x))) = B(x),  x ∈ Ω
σ(u(x)) · n = N(x),  x ∈ Γ_N
u(x) · n = U(x),  x ∈ Γ_D

where σ: ℝ² → ℝ² is the stress tensor induced by the displacement field u: Ω → ℝ², 
B is the body force over Ω, 
N is the surface force on Γ_N, 
U is the displacement on Γ_D, and ∂Ω = Γ_N ∪ Γ_D.
"""

"""
Here Ω is defined as a square (0,8) × (0,8) with 40 holes of radius between 0.3 and 0.6 inside. 

# coordinate (x, y) and radius r
coordinates = [
    [4.56, 3.7], [6.53, 3.33], [6.63, 1.81], [4.75, 5.65], [3.36, 5.8],
    [1.01, 4.92], [4.25, 0.68], [6.11, 4.44], [1.55, 6.97], [2.6, 2.94],
    [6.55, 6.25], [7.25, 5.7], [0.93, 6.18], [3.19, 2.11], [5.03, 4.64],
    [6.28, 0.71], [4.98, 6.95], [2.61, 4.52], [4.13, 2.18], [1.46, 2.14],
    [0.77, 1.75], [2.1, 5.75], [0.91, 3.45], [5.91, 1.93], [7.3, 3.59],
    [2.48, 0.98], [1.48, 0.94], [3.52, 7.25], [2.5, 6.43], [5.8, 2.86],
    [5.4, 1.31], [3.27, 3.88], [7.33, 2.78], [5.59, 6.26], [7.34, 0.65],
    [3.86, 4.91], [0.7, 6.91], [6.57, 5.21], [1.38, 2.83], [7.26, 2.12]
]
radii = [
    0.34, 0.33, 0.35, 0.49, 0.59, 0.54, 0.58, 0.54, 0.45, 0.59, 0.59, 0.3,
    0.36, 0.35, 0.55, 0.43, 0.44, 0.42, 0.48, 0.35, 0.35, 0.33, 0.37, 0.37,
    0.44, 0.33, 0.6, 0.5, 0.44, 0.53, 0.35, 0.45, 0.34, 0.31, 0.46, 0.43,
    0.39, 0.33, 0.31, 0.32
]

Note that there is a cluster of holes that are nearly touching, as shown in the inset. 

u = (1/10) y ((x + 10) sin(y) + (y + 5) cos(x))
v = (1/60) y ((30 + 5 x sin(5 x)) (4 + e^(-5 y)) - 100)

Note:
u_x =  y*(-(y + 5)*sin(x) + sin(y))/10
u_y =  y*((x + 10)*cos(y) + cos(x))/10 + (x + 10)*sin(y)/10 + (y + 5)*cos(x)/10
v_x =  y*(4 + exp(-5*y))*(25*x*cos(5*x) + 5*sin(5*x))/60
v_y =  -y*(5*x*sin(5*x) + 30)*exp(-5*y)/12 + (4 + exp(-5*y))*(5*x*sin(5*x) + 30)/60 - 5/3

u_xx =  -y*(y + 5)*cos(x)/10
u_xy =  (-y*(sin(x) - cos(y)) - (y + 5)*sin(x) + sin(y))/10
u_yy =  (-y*(x + 10)*sin(y) + 2*(x + 10)*cos(y) + 2*cos(x))/10
v_xx =  -5*y*(4 + exp(-5*y))*(5*x*sin(5*x) - 2*cos(5*x))/12
v_xy =  (5*x*cos(5*x) + sin(5*x))*(-5*y*exp(-5*y) + 4 + exp(-5*y))/12
v_yy =  5*(5*y - 2)*(x*sin(5*x) + 6)*exp(-5*y)/12

Dirichlet boundary condition is applied on the lower boundary y = 0 
and Neumann boundary condition is applied on the other boundaries and 
the holes inside. The material constants are:
- The Young's modulus E = 3 × 10⁷ Pa
- Poisson ratio v = 0.3
"""


def func_u(x, args):
    u = (1 / 10) * x[:, [1]] * ((x[:, [0]] + 10) * torch.sin(x[:, [1]]) + (x[:, [1]] + 5) * torch.cos(x[:, [0]]))
    v = (1 / 60) * x[:, [1]] * ((30 + 5 * x[:, [0]] * torch.sin(5 * x[:, [0]])) * (4 + torch.exp(-5 * x[:, [1]])) - 100)
    return u, v


def func_b(x, args):
    """
    u_xx =  -y*(y + 5)*cos(x)/10
    u_xy =  (-y*(sin(x) - cos(y)) - (y + 5)*sin(x) + sin(y))/10
    u_yy =  (-y*(x + 10)*sin(y) + 2*(x + 10)*cos(y) + 2*cos(x))/10
    v_xx =  -5*y*(4 + exp(-5*y))*(5*x*sin(5*x) - 2*cos(5*x))/12
    v_xy =  (5*x*cos(5*x) + sin(5*x))*(-5*y*exp(-5*y) + 4 + exp(-5*y))/12
    v_yy =  5*(5*y - 2)*(x*sin(5*x) + 6)*exp(-5*y)/12
    """
    uxx = -x[:, [1]] * (x[:, [1]] + 5) * torch.cos(x[:, [0]]) / 10
    uxy = (-x[:, [1]] * (torch.sin(x[:, [0]]) - torch.cos(x[:, [1]])) - (x[:, [1]] + 5) * torch.sin(
        x[:, [0]]) + torch.sin(x[:, [1]])) / 10
    uyy = (-x[:, [1]] * (x[:, [0]] + 10) * torch.sin(x[:, [1]]) + 2 * (x[:, [0]] + 10) * torch.cos(
        x[:, [1]]) + 2 * torch.cos(x[:, [0]])) / 10
    vxx = -5 * x[:, [1]] * (4 + torch.exp(-5 * x[:, [1]])) * (
            5 * x[:, [0]] * torch.sin(5 * x[:, [0]]) - 2 * torch.cos(5 * x[:, [0]])) / 12
    vxy = (5 * x[:, [0]] * torch.cos(5 * x[:, [0]]) + torch.sin(5 * x[:, [0]])) * (
            -5 * x[:, [1]] * torch.exp(-5 * x[:, [1]]) + 4 + torch.exp(-5 * x[:, [1]])) / 12
    vyy = 5 * (5 * x[:, [1]] - 2) * (x[:, [0]] * torch.sin(5 * x[:, [0]]) + 6) * torch.exp(-5 * x[:, [1]]) / 12

    pa = args.E / (1 - args.nu ** 2)  # Material constant
    pb = (1 - args.nu) / 2  # Auxiliary constant
    pc = (1 + args.nu) / 2  # Auxiliary constant

    return (-pa * (uxx + pb * uyy)) + (-pa * pc * vxy), (-pa * pc * uxy) + (-pa * (vyy + pb * vxx))


def func_n(x, args, n):
    """
    u_x =  y*(-(y + 5)*sin(x) + sin(y))/10
    u_y =  y*((x + 10)*cos(y) + cos(x))/10 + (x + 10)*sin(y)/10 + (y + 5)*cos(x)/10
    v_x =  y*(4 + exp(-5*y))*(25*x*cos(5*x) + 5*sin(5*x))/60
    v_y =  -y*(5*x*sin(5*x) + 30)*exp(-5*y)/12 + (4 + exp(-5*y))*(5*x*sin(5*x) + 30)/60 - 5/3
    """
    ux = x[:, [1]] * (-(x[:, [1]] + 5) * torch.sin(x[:, [0]]) + torch.sin(x[:, [1]])) / 10
    uy = x[:, [1]] * ((x[:, [0]] + 10) * torch.cos(x[:, [1]]) + torch.cos(x[:, [0]])) / 10 + (
            x[:, [0]] + 10) * torch.sin(x[:, [1]]) / 10 + (x[:, [1]] + 5) * torch.cos(x[:, [0]]) / 10
    vx = x[:, [1]] * (4 + torch.exp(-5 * x[:, [1]])) * (25 * x[:, [0]] * torch.cos(5 * x[:, [0]]) + 5 * torch.sin(
        5 * x[:, [0]])) / 60
    vy = -x[:, [1]] * (5 * x[:, [0]] * torch.sin(5 * x[:, [0]]) + 30) * torch.exp(-5 * x[:, [1]]) / 12 + (
            4 + torch.exp(-5 * x[:, [1]])) * (5 * x[:, [0]] * torch.sin(5 * x[:, [0]]) + 30) / 60 - 5 / 3

    pa = args.E / (1 - args.nu ** 2)  # Material constant
    pb = (1 - args.nu) / 2  # Auxiliary constant
    pc = (1 + args.nu) / 2  # Auxiliary constant

    sigma = [[pa * ux, pa * pb * uy + pa * pc * vx], [pa * pb * vx + pa * pc * uy, pa * vy]]

    return sigma[0][0] * n[:, [0]] + sigma[0][1] * n[:, [1]], sigma[1][0] * n[:, [0]] + sigma[1][1] * n[:, [1]]


def run_rfm(args):
    coordinates = [
        [4.56, 3.7], [6.53, 3.33], [6.63, 1.81], [4.75, 5.65], [3.36, 5.8],
        [1.01, 4.92], [4.25, 0.68], [6.11, 4.44], [1.55, 6.97], [2.6, 2.94],
        [6.55, 6.25], [7.25, 5.7], [0.93, 6.18], [3.19, 2.11], [5.03, 4.64],
        [6.28, 0.71], [4.98, 6.95], [2.61, 4.52], [4.13, 2.18], [1.46, 2.14],
        [0.77, 1.75], [2.1, 5.75], [0.91, 3.45], [5.91, 1.93], [7.3, 3.59],
        [2.48, 0.98], [1.48, 0.94], [3.52, 7.25], [2.5, 6.43], [5.8, 2.86],
        [5.4, 1.31], [3.27, 3.88], [7.33, 2.78], [5.59, 6.26], [7.34, 0.65],
        [3.86, 4.91], [0.7, 6.91], [6.57, 5.21], [1.38, 2.83], [7.26, 2.12]
    ]
    radii = [
        0.34, 0.33, 0.35, 0.49, 0.59, 0.54, 0.58, 0.54, 0.45, 0.59, 0.59, 0.3,
        0.36, 0.35, 0.55, 0.43, 0.44, 0.42, 0.48, 0.35, 0.35, 0.33, 0.37, 0.37,
        0.44, 0.33, 0.6, 0.5, 0.44, 0.53, 0.35, 0.45, 0.34, 0.31, 0.46, 0.43,
        0.39, 0.33, 0.31, 0.32
    ]

    square = pyrfm.Square2D([4, 4], [4, 4])
    circles = sum([pyrfm.Circle2D([x, y], r) for (x, y), r in zip(coordinates, radii)], pyrfm.EmptyGeometry())
    domain = square - circles

    x_in = domain.in_sample(args.Q, with_boundary=False)
    b1, b1n, b2, b2n, b3, b3n, b4, b4n = square.on_sample(math.isqrt(args.Q), with_normal=True, separate=True)
    b_circle, b_circle_n = circles.on_sample(math.isqrt(args.Q), with_normal=True)

    # shuffle the points
    x_in = x_in[torch.randperm(x_in.shape[0])]
    indices = torch.randperm(b1.shape[0])
    b1, b1n = b1[indices], b1n[indices]
    indices = torch.randperm(b2.shape[0])
    b2, b2n = b2[indices], b2n[indices]
    indices = torch.randperm(b3.shape[0])
    b3, b3n = b3[indices], b3n[indices]
    indices = torch.randperm(b4.shape[0])
    b4, b4n = b4[indices], b4n[indices]
    indices = torch.randperm(b_circle.shape[0])
    b_circle, b_circle_n = b_circle[indices], b_circle_n[indices]

    print(
        f"Number of points inside: {x_in.shape[0]} and "
        f"on boundary: {b1.shape[0] + b2.shape[0] + b3.shape[0] + b4.shape[0] + b_circle.shape[0]}")

    model = pyrfm.RFMBase(dim=2, n_hidden=args.M, domain=domain, n_subdomains=2)

    pa = args.E / (1 - args.nu ** 2)  # Material constant
    pb = (1 - args.nu) / 2  # Auxiliary constant
    pc = (1 + args.nu) / 2  # Auxiliary constant

    n_points = []
    errors = []
    counter = 0
    solver = pyrfm.BatchQR(args.M * 2 * 2 * 2, 1)
    n_batch = args.Q // (model.n_hidden * model.submodels.numel()) + 1
    for i, (
            x_in_batch, b1_batch, b1n_batch, b2_batch, b2n_batch, b3_batch, b3n_batch, b4_batch, b4n_batch,
            b_circle_batch,
            b_circle_n_batch) in enumerate(zip(torch.chunk(x_in, n_batch),
                                               torch.chunk(b1, n_batch),
                                               torch.chunk(b1n, n_batch),
                                               torch.chunk(b2, n_batch),
                                               torch.chunk(b2n, n_batch),
                                               torch.chunk(b3, n_batch),
                                               torch.chunk(b3n, n_batch),
                                               torch.chunk(b4, n_batch),
                                               torch.chunk(b4n, n_batch),
                                               torch.chunk(b_circle, n_batch),
                                               torch.chunk(b_circle_n, n_batch))):
        uxx = model.features_second_derivative(x_in_batch, axis1=0, axis2=0).cat(dim=1)
        uyy = model.features_second_derivative(x_in_batch, axis1=1, axis2=1).cat(dim=1)
        uxy = model.features_second_derivative(x_in_batch, axis1=0, axis2=1).cat(dim=1)
        # multiple components can share same basis
        vxx = uxx
        vyy = uyy
        vxy = uxy

        A_in = pyrfm.concat_blocks([[-pa * (uxx + pb * uyy), -pa * pc * vxy],
                                    [-pa * pc * uxy, -pa * (vyy + pb * vxx)]])

        bx, by = func_b(x_in_batch, args)
        f_in = pyrfm.concat_blocks([[bx], [by]])

        u = model.features(b1_batch).cat(dim=1)
        v = u
        A_b1 = pyrfm.concat_blocks([[u, torch.zeros_like(u)], [torch.zeros_like(v), v]])
        f_b1 = torch.zeros(2 * b1_batch.shape[0], 1)

        b_others = torch.cat([b2_batch, b3_batch, b4_batch, b_circle_batch], dim=0)
        u = model.features(b_others).cat(dim=1)
        v = u
        A_others = pyrfm.concat_blocks([[u, torch.zeros_like(u)], [torch.zeros_like(v), v]])
        f_others = torch.cat(func_u(b_others, args), dim=0)

        # bn_others = torch.cat([b2n_batch, b3n_batch, b4n_batch, b_circle_n_batch], dim=0)
        # nx = bn_others[:, [0]]
        # ny = bn_others[:, [1]]
        # ux = model.features_derivative(b_others, axis=0).cat(dim=1)
        # uy = model.features_derivative(b_others, axis=1).cat(dim=1)
        # vx = ux
        # vy = uy
        #
        # A_others = pyrfm.concat_blocks([[pa * (ux * nx + pb * uy * ny), pa * pc * vx * ny],
        #                                 [pa * pc * uy * nx, pa * (vy * ny + pb * vx * nx)]])
        # f_others = torch.cat(func_n(b_others, args, bn_others), dim=0)

        A = pyrfm.concat_blocks([[A_in], [A_b1], [A_others]])
        f = torch.cat([f_in, f_b1, f_others], dim=0)

        print("Batch", i, "A.shape", A.shape, "f.shape", f.shape)
        solver.add_rows(A, f)
        counter += A.shape[0] // 2
        if i % 2 == 0 and i > 0:
            model.W = solver.get_solution().view(2, -1).T
            x_test = domain.in_sample(400, with_boundary=True)

            uv = model(x_test)
            uv_true = torch.cat(func_u(x_test, args), dim=1)
            n_points.append(counter)
            errors.append(((uv - uv_true).norm() / uv_true.norm()).item())
            print('Error:', errors[-1])

    # plt.plot(n_points, errors)
    # plt.yscale("log")
    # plt.xlabel("Number of points")
    # plt.ylabel("Relative error")
    # plt.show()

    visualizer = RFMVisualizer2D(model, resolution=(800, 800))
    visualizer.plot()
    visualizer.show()


param_sets = [
    {"Q": 16000, "M": 200, "E": 3e7, "nu": 0.3},
]

if __name__ == "__main__":
    # torch.set_default_device('cuda') if torch.cuda.is_available() else torch.set_default_device('cpu')
    parser = argparse.ArgumentParser(description="Elasticity problem")
    parser.add_argument("--Q", type=int, required=True)
    parser.add_argument("--M", type=int, required=True)
    parser.add_argument("--E", type=float, default=3e7)
    parser.add_argument("--nu", type=float, default=0.3)

    # if len(sys.argv) == 1:
    #     for param_set in param_sets:
    #         args = argparse.Namespace(**param_set)
    #         run_rfm(args)
    # else:
    #     args = parser.parse_args()
    #     run_rfm(args)

    args = argparse.Namespace(**param_sets[0])
    run_rfm(args)
