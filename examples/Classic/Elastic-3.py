# -*- coding: utf-8 -*-
"""
Created on 2025/2/25

@author: Yifei Sun
"""

import pyrfm
import torch
import os
import argparse
import sys
import time
import math

"""
Example 3.3 (Elasticity problem):
The two-dimensional elasticity problem:

-div(σ(u(x))) = B(x),  x ∈ Ω
σ(u(x)) · n = N(x),  x ∈ Γ_N
u(x) · n = U(x),  x ∈ Γ_D

where σ: ℝ² → ℝ² is the stress tensor induced by the displacement field u: Ω → ℝ², 
B is the body force over Ω, 
N is the surface force on Γ_N, 
U is the displacement on Γ_D, and ∂Ω = Γ_N ∪ Γ_D.
"""

"""
The domain for the first example is given by a square
𝑆₁ = [−1, 1] × [−0.5, 0.5]
jointed by a semi-disk centered at (1.0, 0.0) with radius 0.5,
𝑆₂ = {(x, y) | (x − 1)² + y² ≤ 0.25, y ≥ 0}

with two disks centered at (1.2, 0.0) and (−0.5, 0.0)
with radius 0.2 removed:
𝑆₃ = {(x, y) | (x − 1.2)² + y² ≤ 0.04} ∪ {(x, y) | (x + 0.5)² + y² ≤ 0.04}

u = (1/10) y ((x + 10) sin(y) + (y + 5) cos(x))
v = (1/60) y ((30 + 5 x sin(5 x)) (4 + e^(-5 y)) - 100)

Dirichlet boundary condition is applied on the lower boundary y = 0 
and Neumann boundary condition is applied on the other boundaries and 
the holes inside. The material constants are:
- The Young's modulus E = 3 × 10⁷ Pa
- Poisson ratio v = 0.3
"""


def func_u(x, args):
    pass


def func_f(x, args):
    pass


def run_rfm(args):
    domain = pyrfm.Square2D(center=[0.0, 0.0], half=[1.0, 0.5]) + pyrfm.Circle2D(center=[1.0, 0.0], radius=0.5) \
             - (pyrfm.Circle2D(center=[1.2, 0.0], radius=0.2) + pyrfm.Circle2D(center=[-0.5, 0.0], radius=0.2))

    x_in = domain.in_sample(args.Q, with_boundary=False)
    x_on = domain.on_sample(args.Q)
    # plot scatter points
    from matplotlib import pyplot as plt

    plt.scatter(x_on[:, 0], x_on[:, 1], c='b', label='on', s=0.5)
    plt.scatter(x_in[:, 0], x_in[:, 1], c='r', label='in', s=0.5)
    plt.axis('equal')
    plt.show()


param_sets = [
    {"Q": 4000, "M": 200},
]

if __name__ == "__main__":
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
