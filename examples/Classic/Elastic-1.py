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
The two-dimensional elasticity problem we consider here is of the following form

-div(σ(u(x))) = B(x),  x ∈ Ω
σ(u(x)) · n = N(x),  x ∈ Γ_N
u(x) · n = U(x),  x ∈ Γ_D

where σ: ℝ² → ℝ² is the stress tensor induced by the displacement field u: Ω → ℝ², 
B is the body force over Ω, 
N is the surface force on Γ_N, 
U is the displacement on Γ_D, and ∂Ω = Γ_N ∪ Γ_D.
"""

"""
u = - (P y)/(6 E I) [(6 L - 3 x) x + (2 + ν)(y² - D²/4)]
v = (P)/(6 E I) [3 ν y² (L - x) + (4 + 5 ν) (D² x)/4 + (3 L - x) x²],

where I = D³/12. Homogeneous Dirichlet boundary condition is applied on the left boundary x = 0, 
and Homogeneous Neumann boundary condition is applied on the other boundaries. 
The material parameters are as follows: 
    the Young's modulus E = 3 × 10⁷ Pa, 
    Poisson ratio ν = 0.3. 
    We choose D = 10, L = 10, and the shear force is P = 1000 Pa.
"""


def func_u(x, args):
    pass


def func_f(x, args):
    pass


def run_rfm(args):
    pass


param_sets = []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Elasticity problem")
    parser.add_argument("--Q", type=int, required=True)
    parser.add_argument("--M", type=int, required=True)
    parser.add_argument("--activation", type=str, required=True)
    parser.add_argument("--Rm", type=float, required=True)
    parser.add_argument("--D", type=float, default=10)
    parser.add_argument("--L", type=float, default=10)
    parser.add_argument("--P", type=float, default=1000)
    parser.add_argument("--E", type=float, default=3e7)
    parser.add_argument("--nu", type=float, default=0.3)
    parser.add_argument("--I", type=float, default=1000.0 / 12)

    if len(sys.argv) == 1:
        for param_set in param_sets:
            args = argparse.Namespace(**param_set)
            run_rfm(args)
    else:
        args = parser.parse_args()
        run_rfm(args)
