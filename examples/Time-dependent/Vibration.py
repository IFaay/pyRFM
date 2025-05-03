# -*- coding: utf-8 -*-
"""
Created on 5/3/25

@author: Yifei Sun
"""
import pyrfm
import torch
import os
import argparse
import sys
import time

"""
Section 3.2.1 (Membrane Vibration over a Simple Geometry) 
Consider the following problem

⎧ ∂ₜₜ u(x, y, t) − α² Δu(x, y, t) = 0,        (x, y), t ∈ Ω × [0, T],
⎪ u(x, y, 0) = φ(x, y),                      (x, y) ∈ Ω,
⎪ ∂ₜ u(x, y, 0) = ψ(x, y),                   (x, y) ∈ Ω,
⎩ u(x, y, t) = 0,                            (x, y) ∈ ∂Ω × [0, T],

where Ω = [0, 5] × [0, 4], α = 1 and T = 10. The exact solution is chosen to be:

    uₑ(x, y, t) = sin(μx) · sin(νy) · (2 cos(λt) + sin(λt)),
    μ = 2π / (x₁ − x₀), ν = 2π / (y₁ − y₀), λ = √(μ² + ν²)

and φ(x, y) and ψ(x, y) are chosen accordingly.
"""


def func_u(xt):
    x = xt[:, [0]]
    y = xt[:, [1]]
    t = xt[:, [2]]
    mu = 2 * torch.pi / 5
    nu = 2 * torch.pi / 4
    lam = (mu ** 2 + nu ** 2) ** 0.5
    return (torch.sin(mu * x) * torch.sin(nu * y) *
            (2 * torch.cos(lam * t) + torch.sin(lam * t)))


def func_phi(xt):
    return func_u(xt)


def func_psi(xt):
    x = xt[:, [0]]
    y = xt[:, [1]]
    t = xt[:, [2]]
    mu = 2 * torch.pi / 5
    nu = 2 * torch.pi / 4
    lam = (mu ** 2 + nu ** 2) ** 0.5
    return (torch.sin(mu * x) * torch.sin(nu * y) * (-2 * lam * torch.sin(lam * t) + lam * torch.cos(lam * t)))


def run_rfm(args):
    time_stamp = torch.linspace(start=0, end=10, steps=args.Nt + 1)
    domain = pyrfm.Square2D(center=[2.5, 2], radius=[2.5, 2])
    models = []
    for (t0, t1) in zip(time_stamp[:-1], time_stamp[1:]):
        models.append(pyrfm.STRFMBase(dim=2,
                                      n_hidden=args.Jn,
                                      domain=domain,
                                      time_interval=[t0, t1],
                                      n_spatial_subdomains=[args.Nx, args.Ny],
                                      n_temporal_subdomains=args.Nt,
                                      st_type=args.type)
                      )

    x_in = domain.in_sample([args.Qx * args.Nx, args.Qy * args.Ny], with_boundary=False)
    x_on = domain.on_sample(4 * args.Nx * args.Ny * args.Qx * args.Qy)


param_sets_groups = [
    # Convergence w.r.t Nt (Figure 10a)
    [
        {"Nx": 2, "Ny": 2, "Nt": 1, "Qx": 30, "Qy": 30, "Qt": 30, "Jn": 400, "Nb": 1, "type": "STC"},
        {"Nx": 2, "Ny": 2, "Nt": 2, "Qx": 30, "Qy": 30, "Qt": 30, "Jn": 400, "Nb": 1, "type": "STC"},
        {"Nx": 2, "Ny": 2, "Nt": 3, "Qx": 30, "Qy": 30, "Qt": 30, "Jn": 400, "Nb": 1, "type": "STC"},
        {"Nx": 2, "Ny": 2, "Nt": 4, "Qx": 30, "Qy": 30, "Qt": 30, "Jn": 400, "Nb": 1, "type": "STC"},
        {"Nx": 2, "Ny": 2, "Nt": 5, "Qx": 30, "Qy": 30, "Qt": 30, "Jn": 400, "Nb": 1, "type": "STC"},
        {"Nx": 2, "Ny": 2, "Nt": 6, "Qx": 30, "Qy": 30, "Qt": 30, "Jn": 400, "Nb": 1, "type": "STC"}
    ],
    # Convergence w.r.t Jn (Figure 10b)
    [
        {"Nx": 2, "Ny": 2, "Nt": 2, "Qx": 30, "Qy": 30, "Qt": 30, "Jn": 100, "Nb": 1, "type": "STC"},
        {"Nx": 2, "Ny": 2, "Nt": 2, "Qx": 30, "Qy": 30, "Qt": 30, "Jn": 150, "Nb": 1, "type": "STC"},
        {"Nx": 2, "Ny": 2, "Nt": 2, "Qx": 30, "Qy": 30, "Qt": 30, "Jn": 200, "Nb": 1, "type": "STC"},
        {"Nx": 2, "Ny": 2, "Nt": 2, "Qx": 30, "Qy": 30, "Qt": 30, "Jn": 250, "Nb": 1, "type": "STC"},
        {"Nx": 2, "Ny": 2, "Nt": 2, "Qx": 30, "Qy": 30, "Qt": 30, "Jn": 300, "Nb": 1, "type": "STC"},
        {"Nx": 2, "Ny": 2, "Nt": 2, "Qx": 30, "Qy": 30, "Qt": 30, "Jn": 350, "Nb": 1, "type": "STC"},
        {"Nx": 2, "Ny": 2, "Nt": 2, "Qx": 30, "Qy": 30, "Qt": 30, "Jn": 400, "Nb": 1, "type": "STC"}
    ],
    # Convergence w.r.t Q (Figure 10c)
    [
        {"Nx": 5, "Ny": 5, "Nt": 5, "Qx": 10, "Qy": 10, "Qt": 10, "Jn": 400, "Nb": 5, "type": "STC"},
        {"Nx": 5, "Ny": 5, "Nt": 5, "Qx": 15, "Qy": 15, "Qt": 15, "Jn": 400, "Nb": 5, "type": "STC"},
        {"Nx": 5, "Ny": 5, "Nt": 5, "Qx": 20, "Qy": 20, "Qt": 20, "Jn": 400, "Nb": 5, "type": "STC"},
        {"Nx": 5, "Ny": 5, "Nt": 5, "Qx": 25, "Qy": 25, "Qt": 25, "Jn": 400, "Nb": 5, "type": "STC"},
        {"Nx": 5, "Ny": 5, "Nt": 5, "Qx": 30, "Qy": 30, "Qt": 30, "Jn": 400, "Nb": 5, "type": "STC"}
    ],
    # STC vs SoV (Figure 10d)
    [
        {"Nx": 2, "Ny": 2, "Nt": 1, "Qx": 30, "Qy": 30, "Qt": 30, "Jn": 400, "Nb": 1, "type": "STC"},
        {"Nx": 2, "Ny": 2, "Nt": 2, "Qx": 30, "Qy": 30, "Qt": 30, "Jn": 400, "Nb": 1, "type": "STC"},
        {"Nx": 2, "Ny": 2, "Nt": 3, "Qx": 30, "Qy": 30, "Qt": 30, "Jn": 400, "Nb": 1, "type": "STC"},
        {"Nx": 2, "Ny": 2, "Nt": 4, "Qx": 30, "Qy": 30, "Qt": 30, "Jn": 400, "Nb": 1, "type": "STC"},
        {"Nx": 2, "Ny": 2, "Nt": 5, "Qx": 30, "Qy": 30, "Qt": 30, "Jn": 400, "Nb": 1, "type": "STC"},
        {"Nx": 2, "Ny": 2, "Nt": 6, "Qx": 30, "Qy": 30, "Qt": 30, "Jn": 400, "Nb": 1, "type": "STC"},
        {"Nx": 2, "Ny": 2, "Nt": 6, "Qx": 30, "Qy": 30, "Qt": 30, "Jn": 400, "Nb": 1, "type": "SOV"}
    ]
]

group_labels = [
    "Convergence w.r.t Nt (10a)",
    "Convergence w.r.t Jn (10b)",
    "Convergence w.r.t Q (10c)",
    "Comparison STC vs SoV (10d)"
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RFM for Time-dependent PDEs")
    parser.add_argument("--Nx", type=int, default=2, help="Number of spatial subdomains in x direction")
    parser.add_argument("--Ny", type=int, default=2, help="Number of spatial subdomains in y direction")
    parser.add_argument("--Nt", type=int, default=1, help="Number of temporal subdomains")
    parser.add_argument("--Qx", type=int, default=30, help="Number of quadrature points in x direction")
    parser.add_argument("--Qy", type=int, default=30, help="Number of quadrature points in y direction")
    parser.add_argument("--Qt", type=int, default=30, help="Number of quadrature points in t direction")
    parser.add_argument("--Jn", type=int, default=400, help="Number of neurons in each hidden layer")
    parser.add_argument("--Nb", type=int, default=1, help="Number of basis functions")
    parser.add_argument("--type", type=str, default="STC", choices=["STC", "SOV"], help="Type of RFM")
    args = parser.parse_args()

    run_rfm(args)
