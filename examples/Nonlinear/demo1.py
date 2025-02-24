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

"""
Consider the nonlinear equation with Dirichlet boundary condition over Ω = [0,1] × [0,1]:

     - Δu(x, y) + u²(x, y)  = f(x, y),    (x, y) ∈ Ω

with boundary conditions:

    u(x, 0) = g1(x),   u(x, 1) = g2(x)
    u(0, y) = h1(y),   u(1, y) = h2(y)

Once an explicit form of u is given, the functions g1, g2, h1, h2, and f can be computed.

"""


def func_u(x):
    """ True solution u(x,y) = sin(πx) * sin(πy) with homogeneous Dirichlet BC """
    # x is a torch tensor of shape (N, 2)
    x_coord = x[:, [0]]
    y_coord = x[:, [1]]
    return torch.sin(torch.pi * x_coord) * torch.sin(torch.pi * y_coord)


def func_f(x):
    """ Source term f(x,y) = 2π² sin(πx)sin(πy) + [sin(πx)sin(πy)]^2 """
    u = func_u(x)
    return 2 * (torch.pi ** 2) * u + u ** 2


def func_g(x):
    return func_u(x)
