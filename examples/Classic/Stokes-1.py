# -*- coding: utf-8 -*-
"""
Created on 2025/7/23

@author: Yifei Sun
"""
import time
import argparse

from fontTools.ttLib.reorderGlyphs import reorderGlyphs

import pyrfm
import torch
import os
import sys

"""
Consider Stokes flow defined by the system:

    -Δu(x) + ∇p(x) = f(x)        for x in Ω,
    ∇·u(x) = 0                  for x in Ω,
    u(x) = U(x)                 for x on ∂Ω.

In this problem, the pressure p is only determined up to a constant. 
To avoid difficulties, we fix the value of p at the left-bottom corner.

Ω is the square (0, 1) × (0, 1) with three holes centered at (0.5, 0.2), (0.2, 0.8), (0.8, 0.8) of radius 0.1.

The exact displacement field for the Stokes flow is given by
⎧ u = x + x² - 2xy + x³ - 3xy² + x²y
⎨ v = -y - 2xy + y² - 3x²y + y³ - xy²
⎩ p = xy + x + y + x³y² - ⁴⁄₃

From the exact solution, we can derive the source term f as follows:
⎧ f₁(x, y) = 3x²y² - y - 1
⎩ f₂(x, y) = 2x³y + 3x - 1

∇·u(x, y) = 0
"""


def func_u(x):
    u = x[:, [0]] + x[:, [0]] ** 2 - 2 * x[:, [0]] * x[:, [1]] + x[:, [0]] ** 3 - 3 * x[:, [0]] * x[:, [1]] ** 2 + x[
        :, [0]] ** 2 * x[:, [1]]
    return u


def func_v(x):
    v = -x[:, [1]] - 2 * x[:, [0]] * x[:, [1]] + x[:, [1]] ** 2 - 3 * x[:, [0]] ** 2 * x[:, [1]] + x[:, [1]] ** 3 - x[
        :, [0]] * x[:, [1]] ** 2
    return v


def func_p(x):
    p = x[:, [0]] * x[:, [1]] + x[:, [0]] + x[:, [1]] + x[:, [0]] ** 3 * x[:, [1]] ** 2 - 4 / 3
    return p


def func_f(x):
    f1 = 3 * x[:, [0]] ** 2 * x[:, [1]] ** 2 - x[:, [1]] - 1
    f2 = 2 * x[:, [0]] ** 3 * x[:, [1]] + 3 * x[:, [0]] - 1
    return f1, f2


def run_rfm(args):
    print("\n" + "=" * 40)
    print(f"Simulation Started with Parameters:")
    print(f"Q = {args.Q}, M = {args.M}")
    print(f"--------------------------")

    torch.set_default_device('cuda') if torch.cuda.is_available() else torch.set_default_device('cpu')
    start_time = time.time()

    domain = pyrfm.Square2D(center=(0.5, 0.5), half=(0.5, 0.5)) - pyrfm.Circle2D(center=(0.5, 0.2),
                                                                                 radius=0.1) - pyrfm.Circle2D(
        center=(0.2, 0.8), radius=0.1) - pyrfm.Circle2D(center=(0.8, 0.8), radius=0.1)

    model = pyrfm.RFMBase(dim=2, n_hidden=args.M, domain=domain, n_subdomains=1)
    x_in = domain.in_sample(args.Q, with_boundary=False)
    x_on = domain.on_sample(400)
    x_corner = torch.tensor([[0.0, 0.0]])

    u_in_x = model.features_derivative(x_in, axis=0).cat(dim=1)
    u_in_y = model.features_derivative(x_in, axis=1).cat(dim=1)
    u_in_xx = model.features_second_derivative(x_in, axis1=0, axis2=0).cat(dim=1)
    u_in_yy = model.features_second_derivative(x_in, axis1=1, axis2=1).cat(dim=1)

    u_on = model.features(x_on).cat(dim=1)
    u_corner = model.features(x_corner).cat(dim=1)

    # A1 = pyrfm.concat_blocks([[-(u_in_xx + u_in_yy), torch.zeros_like(u_in_xx), u_in_x],
    #                           [torch.zeros_like(u_in_xx), -(u_in_xx + u_in_yy), u_in_y]])
    # b1 = torch.cat([func_f(x_in)], dim=0)
    # A2 = pyrfm.concat_blocks(
    #     [[u_on, torch.zeros_like(u_on), torch.zeros_like(u_on)], [torch.zeros_like(u_on), u_on, torch.zeros_like(u_on)],
    #      [torch.zeros_like(u_on), torch.zeros_like(u_on), u_on]])
    # b2 = func_u(x_on)

    A1 = pyrfm.concat_blocks([[-(u_in_xx + u_in_yy), torch.zeros_like(u_in_xx), u_in_x],
                              [torch.zeros_like(u_in_xx), -(u_in_xx + u_in_yy), u_in_y],
                              [u_in_x, u_in_y, torch.zeros_like(u_in_x)]])
    f1_in, f2_in = func_f(x_in)
    b1 = torch.cat([f1_in, f2_in, torch.zeros_like(f1_in)], dim=0)
    A2 = pyrfm.concat_blocks([[u_on, torch.zeros_like(u_on), torch.zeros_like(u_on)],
                              [torch.zeros_like(u_on), u_on, torch.zeros_like(u_on)]])
    u_on, v_on = func_u(x_on), func_v(x_on)
    b2 = torch.cat([u_on, v_on], dim=0)
    A3 = pyrfm.concat_blocks([[torch.zeros_like(u_corner), torch.zeros_like(u_corner), u_corner]])
    b3 = torch.tensor([[-4 / 3]])

    A = torch.cat([A1, A2, A3], dim=0)
    b = torch.cat([b1, b2, b3], dim=0)
    # A = torch.cat([A1, A2], dim=0)
    # b = torch.cat([b1, b2], dim=0)
    model.compute(A).solve(b)

    visualizer = pyrfm.RFMVisualizer2D(model, component_idx=2)
    visualizer.plot()
    visualizer.show()
    # visualizer.savefig('stokes_flow_p.png', dpi=600)

    uvp = model.forward(x_in)
    u, v, p = uvp[:, [0]], uvp[:, [1]], uvp[:, [2]]
    u_exact, v_exact = func_u(x_in), func_v(x_in)
    p_exact = func_p(x_in)

    error = torch.linalg.norm(u - u_exact) / torch.linalg.norm(u_exact)

    print(f"\nSimulation Results:")
    print(f"--------------------------")
    print(f"Problem size: N = {A.shape[0]}, M = {A.shape[1]}")
    print(f"Relative Error: {error:.4e}")
    print(f"Elapsed Time: {time.time() - start_time:.2f} seconds")
    print("=" * 40)


param_sets = [{"Q": 400, "M": 400}, {"Q": 800, "M": 400}, ]
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
