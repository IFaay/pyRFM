# -*- coding: utf-8 -*-
"""
Created on 2024/12/17

@author: Yifei Sun
"""
import time

import pyrfm
import torch

# Global parameters
L = 10  # Length of the beam
D = 10  # Depth of the beam

x_l = 0  # Left boundary
x_r = L  # Right boundary
y_d = -D / 2  # Bottom boundary
y_u = D / 2  # Top boundary

E = 3.0e7  # Young's modulus
nu = 0.3  # Poisson's ratio
P = 1.0e3  # Load
I = D ** 3 / 12  # Moment of inertia

a = E / (1 - nu ** 2)  # Material constant
b = (1 - nu) / 2  # Auxiliary constant
c = (1 + nu) / 2  # Auxiliary constant
# lam = E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame's first parameter
lam = E * nu / (1 - nu ** 2)  # plane stress condition
mu = E / (2 * (1 + nu))  # Lame's second parameter
epsilon = 1.0e-24  # Small value for numerical stability


# Displacement function u
def fu(x):
    return -P * x[:, [1]] / (6 * E * I) * (
            (6 * L - 3 * x[:, [0]]) * x[:, [0]] + (2 + nu) * (x[:, [1]] ** 2 - D ** 2 / 4)
    )


# Displacement function v
def fv(x):
    return P / (6 * E * I) * (
            3 * nu * x[:, [1]] ** 2 * (L - x[:, [0]]) +
            (4 + 5 * nu) * D ** 2 * x[:, [0]] / 4 +
            (3 * L - x[:, [0]]) * x[:, [0]] ** 2
    )


# Body force in x-direction (bx)
def bx(x):
    return torch.zeros_like(x[:, [0]])


# Body force in y-direction (by)
def by(x):
    return torch.zeros_like(x[:, [0]])


# Surface force in x-direction (px)
def px(x, n):
    pass


# Surface force in y-direction (py)
def py(x, n):
    pass


# Stress sigma_x
def stress_sigma_x(x):
    pass


# Stress sigma_y
def stress_sigma_y(x):
    pass


# Stress tau_xy
def stress_tau_xy(x):
    pass


if __name__ == '__main__':
    torch.set_default_device('cuda') if torch.cuda.is_available() else torch.set_default_device('cpu')
    domain = pyrfm.Square2D(center=[L / 2, 0], half=[L / 2, D / 2])
    model = pyrfm.RFMBase(dim=2, n_hidden=400, domain=domain, n_subdomains=2)

    x_in = domain.in_sample(6000, with_boundary=False)

    uxx = model.features_second_derivative(x_in, axis1=0, axis2=0).cat(dim=1)
    uyy = model.features_second_derivative(x_in, axis1=1, axis2=1).cat(dim=1)
    uxy = model.features_second_derivative(x_in, axis1=0, axis2=1).cat(dim=1)
    # multiple components can share same basis
    vxx = uxx
    vyy = uyy
    vxy = uxy

    # A_in = pyrfm.concat_blocks([[-a * (uxx + b * uyy), -a * c * vxy],
    #                             [-a * c * uxy, -a * (vyy + b * vxx)]])

    A_in = pyrfm.concat_blocks([
        [(lam + 2 * mu) * uxx + mu * uyy, (lam + mu) * vxy],
        [(lam + mu) * uxy, mu * vxx + (lam + 2 * mu) * vyy]
    ])

    x_on = domain.on_sample(400)
    u = model.features(x_on).cat(dim=1)
    v = u

    A = pyrfm.concat_blocks([[A_in],
                             [u, torch.zeros_like(u)],
                             [torch.zeros_like(v), v]])
    b = pyrfm.concat_blocks([[bx(x_in)],
                             [by(x_in)],
                             [fu(x_on)],
                             [fv(x_on)]])

    model.compute(A).solve(b)

    x_test = domain.in_sample(40, with_boundary=True)
    uv = model(x_test)
    uv_true = torch.cat([fu(x_test), fv(x_test)], dim=1)

    print('Error:', (uv - uv_true).norm() / uv_true.norm())
