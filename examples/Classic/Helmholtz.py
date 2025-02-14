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


def u(x):
    return torch.sin(3 * torch.pi * x[:, [0]] + 3 * torch.pi / 20) * torch.cos(
        2 * torch.pi * x[:, [0]] + torch.pi / 10) + 2


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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--Q", type=int, required=True)
    parser.add_argument("--M", type=int, required=True)
    parser.add_argument("--psi", type=str, required=True)

    # args = parser.parse_args()
    for param_set in param_sets:
        args = argparse.Namespace(**param_set)
        print(args.Q, args.M, args.psi)

        torch.set_default_device('cuda') if torch.cuda.is_available() else torch.set_default_device('cpu')
        start_time = time.time()
        domain = pyrfm.Line1D(x1=0.0, x2=8.0)
        
    M = 100
    Mp = M / 50
