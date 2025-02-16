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
Example 3.2 (Poisson equation):
Consider the Poisson equation with Dirichlet boundary condition over Ω = [0,1] × [0,1]:

    Δu(x, y) = f(x, y),    (x, y) ∈ Ω

with boundary conditions:

    u(x, 0) = g1(x),   u(x, 1) = g2(x)
    u(0, y) = h1(y),   u(1, y) = h2(y)

Once an explicit form of u is given, the functions g1, g2, h1, h2, and f can be computed.
"""

# mixed-frequency problem
def u(x):
    return -0.5 * (1.5 * torch.cos(torch.pi * x[:, [0]] + 2 * torch.pi / 5) +
                   2 * torch.cos(2 * torch.pi * x[:, [0]] - torch.pi / 5)) * \
        (1.5 * torch.cos(torch.pi * x[:, [1]] + 2 * torch.pi / 5) +
         2 * torch.cos(2 * torch.pi * x[:, [1]] - torch.pi / 5)) - \
        0.5 * (1.5 * torch.cos(2 * torch.pi * x[:, [0]] + 2 * torch.pi / 5) +
               2 * torch.cos(4 * torch.pi * x[:, [0]] - torch.pi / 5)) * \
        (1.5 * torch.cos(2 * torch.pi * x[:, [1]] + 2 * torch.pi / 5) +
         2 * torch.cos(4 * torch.pi * x[:, [1]] - torch.pi / 5))


# -(uxx + uyy) = f
def f(x):
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
              2 * (4 * torch.pi) ** 2 * torch.cos(4 * torch.pi * x[:, [1]] - torch.pi / 5)))


def g(x):
    return u(x)


def run_rfm(x):
    pass


param_sets = [
    {"Q": 200, "M": 160, "psi": "A"},
    {"Q": 400, "M": 320, "psi": "A"},
    {"Q": 800, "M": 640, "psi": "A"},
    {"Q": 1600, "M": 1280, "psi": "A"},
    {"Q": 200, "M": 160, "psi": "B"},
    {"Q": 400, "M": 320, "psi": "B"},
    {"Q": 800, "M": 640, "psi": "B"},
    {"Q": 1600, "M": 1280, "psi": "B"},
]


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