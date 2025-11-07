# -*- coding: utf-8 -*-
"""
Created on 2025/10/8

@author: Yifei Sun
"""

"""
Train a time-dependent random feature model for laplace-beltrami operator

A classical problem in surface PDEs is the surface diffusion equation (𝓛 = Δₛ):

    ∂ₜ u(x, t) = Δₛ u(x, t) + f(x, t),   x ∈ Γ,  t ∈ (0, T].

To evaluate accuracy, we consider the exact solution:
    u(x, y, z, t) = sin(x + sin(t)) · exp(cos(y − z)),
from which the initial condition and source term f can be derived.

The Crank–Nicolson (CN) scheme is:
    (uⁿ⁺¹ − uⁿ)/Δt = ½ [ 𝓛(uⁿ) + 𝓛(uⁿ⁺¹) ] + ½ [ fⁿ + fⁿ⁺¹ ],

where Δt is the time step size, uⁿ = u(x, nΔt), fⁿ = f(x, nΔt).

Rearranging gives the system for uⁿ⁺¹:
    (I − Δt/2 · 𝓛) uⁿ⁺¹ = (I + Δt/2 · 𝓛) uⁿ + Δt/2 (fⁿ + fⁿ⁺¹).

"""

from collections import defaultdict
import re
import torch
import numpy as np
import pyrfm
from typing import Union, Tuple, List, Optional, Dict

from pathlib import Path
import matplotlib

# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.colors import TwoSlopeNorm
import sys
import time


def func_u(p: torch.Tensor) -> torch.Tensor:
    """
    Example function u(x, y, z) = 0
    """
    return torch.zeros(p.shape[0], 1)


def func_f(p: torch.Tensor) -> torch.Tensor:
    """
    f(𝐱) = exp(−‖𝐱 − 𝐱₀‖²)
    """
    return torch.exp(-(p ** 2).sum(dim=1, keepdim=True))


if __name__ == '__main__':
    torch.set_default_device('cuda') if torch.cuda.is_available() else torch.set_default_device('cpu')
    device = torch.tensor(0.0).device
    dtype = torch.tensor(0.0).dtype

    domain = pyrfm.Cube3D(center=(0.0, 0.0, 0.0), radius=(1, 1, 1))
    x_in = domain.in_sample(4000)
    x_on = domain.on_sample(1000)

    model = pyrfm.RFMBase(dim=3, n_hidden=1000, domain=domain, n_subdomains=1, rf=pyrfm.RFTanH)

    dts = [1e-2, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4]

    for dt in dts:
        print(dt)
        import sys
        import io

        backup = sys.stdout
        sys.stdout = io.StringIO()

        t0 = time.time()
        t = 0.0

        u0 = func_u(x_in)
        A = model.features(x_in).cat(dim=1)
        b = u0
        model.compute(A).solve(b)

        A_lap = (model.features_second_derivative(x_in, axis1=0, axis2=0)
                 + model.features_second_derivative(x_in, axis1=1, axis2=1)
                 + model.features_second_derivative(x_in, axis1=2, axis2=2)).cat(dim=1)

        A_rhs = model.features(x_in).cat(dim=1) + 0.5 * dt * A_lap
        A_on = model.features(x_on).cat(dim=1)

        A = torch.cat([model.features(x_in).cat(dim=1) - 0.5 * dt * A_lap, A_on], dim=0)
        model.compute(A)

        while t < 1.0:
            t += dt

            b = A_rhs @ model.W + dt * func_f(x_in)
            b = torch.cat([b, torch.zeros(A_on.shape[0], 1)], dim=0)
            model.solve(b)

        sys.stdout = backup

        torch.save(model(x_in), f'heat_dt{dt:.0e}.pt')

    u_reference = torch.load(f'heat_dt{dts[-1]:.0e}.pt')

    for dt in dts[:-1]:
        u = torch.load(f'heat_dt{dt:.0e}.pt')
        error = torch.norm(u - u_reference) / torch.norm(u_reference)
        print(f'dt={dt:.0e}, rel error={error:.2e}')
