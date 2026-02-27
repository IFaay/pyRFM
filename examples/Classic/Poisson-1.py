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
import math

from pyrfm import Tensor

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
def func_u(x, frequency="mixed"):
    if frequency == "low":
        AA = 1.0
        BB = 0.0
    elif frequency == "high":
        AA = 0.0
        BB = 1.0
    elif frequency == "mixed":
        AA = 0.5
        BB = 0.5
    else:
        raise ValueError("Invalid frequency")
    return -AA * (1.5 * torch.cos(torch.pi * x[:, [0]] + 2 * torch.pi / 5) +
                  2 * torch.cos(2 * torch.pi * x[:, [0]] - torch.pi / 5)) * \
        (1.5 * torch.cos(torch.pi * x[:, [1]] + 2 * torch.pi / 5) +
         2 * torch.cos(2 * torch.pi * x[:, [1]] - torch.pi / 5)) - \
        BB * (1.5 * torch.cos(2 * torch.pi * x[:, [0]] + 2 * torch.pi / 5) +
              2 * torch.cos(4 * torch.pi * x[:, [0]] - torch.pi / 5)) * \
        (1.5 * torch.cos(2 * torch.pi * x[:, [1]] + 2 * torch.pi / 5) +
         2 * torch.cos(4 * torch.pi * x[:, [1]] - torch.pi / 5))


# -(uxx + uyy) = f
def func_f(x, frequency="mixed"):
    if frequency == "low":
        AA = 1.0
        BB = 0.0
    elif frequency == "high":
        AA = 0.0
        BB = 1.0
    elif frequency == "mixed":
        AA = 0.5
        BB = 0.5
    else:
        raise ValueError("Invalid frequency")
    return -(-AA * (-1.5 * torch.pi ** 2 * torch.cos(torch.pi * x[:, [0]] + 2 * torch.pi / 5) -
                    2 * (2 * torch.pi) ** 2 * torch.cos(2 * torch.pi * x[:, [0]] - torch.pi / 5)) * \
             (1.5 * torch.cos(torch.pi * x[:, [1]] + 2 * torch.pi / 5) +
              2 * torch.cos(2 * torch.pi * x[:, [1]] - torch.pi / 5)) - \
             AA * (1.5 * torch.cos(torch.pi * x[:, [0]] + 2 * torch.pi / 5) +
                   2 * torch.cos(2 * torch.pi * x[:, [0]] - torch.pi / 5)) * \
             (-1.5 * torch.pi ** 2 * torch.cos(torch.pi * x[:, [1]] + 2 * torch.pi / 5) -
              2 * (2 * torch.pi) ** 2 * torch.cos(2 * torch.pi * x[:, [1]] - torch.pi / 5)) - \
             BB * (-1.5 * (2 * torch.pi) ** 2 * torch.cos(2 * torch.pi * x[:, [0]] + 2 * torch.pi / 5) -
                   2 * (4 * torch.pi) ** 2 * torch.cos(4 * torch.pi * x[:, [0]] - torch.pi / 5)) * \
             (1.5 * torch.cos(2 * torch.pi * x[:, [1]] + 2 * torch.pi / 5) +
              2 * torch.cos(4 * torch.pi * x[:, [1]] - torch.pi / 5)) - \
             BB * (1.5 * torch.cos(2 * torch.pi * x[:, [0]] + 2 * torch.pi / 5) +
                   2 * torch.cos(4 * torch.pi * x[:, [0]] - torch.pi / 5)) * \
             (-1.5 * (2 * torch.pi) ** 2 * torch.cos(2 * torch.pi * x[:, [1]] + 2 * torch.pi / 5) -
              2 * (4 * torch.pi) ** 2 * torch.cos(4 * torch.pi * x[:, [1]] - torch.pi / 5)))


def func_g(x, frequency="mixed"):
    return func_u(x, frequency)


class MultiScaleRFM(pyrfm.RFMBase):
    def __init__(self, dim, n_hidden, domain, n_subdomains, pou=pyrfm.PsiB,
                 rf=pyrfm.RFTanH, *args, **kwargs):
        super().__init__(dim=dim,
                         n_hidden=n_hidden,
                         domain=domain,
                         n_subdomains=n_subdomains,
                         pou=pou,
                         rf=rf,
                         *args, **kwargs)
        bounding_box = self.domain.get_bounding_box()
        center = torch.tensor([(bounding_box[2 * i + 1] + bounding_box[2 * i]) / 2.0 for i in range(dim)],
                              dtype=self.dtype, device=self.device)
        radius = torch.tensor([(bounding_box[2 * i + 1] - bounding_box[2 * i]) / 2.0 for i in range(dim)],
                              dtype=self.dtype, device=self.device)
        self.global_model = rf(dim=dim, center=center, radius=radius, n_hidden=n_hidden)

    def forward(self, x):
        """
        Forward pass of the model.

        :param x: Input tensor.
        :return: Output tensor after forward pass.
        """
        from typing import List
        if self.W is None:
            raise ValueError("Weights have not been computed yet.")
        elif isinstance(self.W, Tensor):
            self.W = self.W.cat(dim=1)
        elif isinstance(self.W, List) and isinstance(self.W[0], torch.Tensor):
            self.W = torch.cat(self.W, dim=1)

        return torch.matmul(self.features(x), self.W)

    def solve(self, b: torch.Tensor, check_condition=False):
        """
        Solve the linear system Ax = b using the QR decomposition.

        :param b: Right-hand side tensor.
        :param check_condition: Whether to check the condition number of A, and switch to SVD if necessary.
        """
        b = b.view(-1, 1).to(dtype=self.dtype, device=self.device)
        if self.A.shape[0] != b.shape[0]:
            raise ValueError("Input dimension mismatch.")
        b /= self.A_norm

        y = torch.ormqr(self.A, self.tau, b, transpose=True)[:self.A.shape[1]]
        self.W = torch.linalg.solve_triangular(self.A[:self.A.shape[1], :], y, upper=True)
        b_ = torch.ormqr(self.A, self.tau, torch.matmul(torch.triu(self.A), self.W), transpose=False)
        residual = torch.norm(b_ - b) / torch.norm(b)

        if check_condition and torch.linalg.cond(self.A_backup) > 1.0 / torch.finfo(self.dtype).eps:
            print(f"The condition number exceeds 1/eps; switching to SVD.")
            self.W = torch.linalg.lstsq(self.A_backup, b.cpu(), driver='gelsd')[0].to(dtype=self.dtype,
                                                                                      device=self.device)
            residual = torch.norm(
                torch.matmul(self.A_backup.to(dtype=self.dtype, device=self.device), self.W) - b) / torch.norm(b)

        print(f"Least Square Relative residual: {residual:.4e}")

        # all submodels and one global model
        if self.W.numel() % ((self.submodels.numel() + 1) * self.n_hidden) == 0:
            n_out = int(self.W.numel() / (self.submodels.numel() * self.n_hidden))
            self.W = self.W.view(n_out, -1).T
        else:
            raise ValueError("The output weight mismatch.")

    def features(self, x: torch.Tensor, use_sparse: bool = False) -> Tensor:
        return torch.cat([super().features(x, use_sparse).cat(dim=1), self.global_model.forward(x)], dim=1)

    def features_derivative(self, x: torch.Tensor, axis: int, use_sparse: bool = False) -> Tensor:
        return torch.cat(
            [super().features_derivative(x, axis, use_sparse).cat(dim=1), self.global_model.first_derivative(x, axis)],
            dim=1)

    def features_second_derivative(self, x: torch.Tensor, axis1: int, axis2: int, use_sparse: bool = False) -> Tensor:
        return torch.cat([super().features_second_derivative(x, axis1, axis2, use_sparse).cat(dim=1),
                          self.global_model.second_derivative(x, axis1, axis2)], dim=1)


def run_rfm(args):
    print("\n" + "=" * 40)
    print(f"Simulation Started with Parameters:")
    print(f"Q = {args.Q}, M = {args.M}, {args.frequency} frequency, multiscale = {args.multiscale}")
    print(f"--------------------------")

    torch.set_default_device('cuda') if torch.cuda.is_available() else torch.set_default_device('cpu')
    start_time = time.time()
    domain = pyrfm.Square2D([0.5, 0.5], [0.5, 0.5])

    if args.multiscale:
        model = MultiScaleRFM(dim=2, n_hidden=300, domain=domain, n_subdomains=math.isqrt(args.M // 300),
                              pou=pyrfm.PsiB)
        x_in = domain.in_sample(args.Q, with_boundary=False)
        x_on = domain.on_sample(int(0.2 * args.Q))

        A_in_xx = model.features_second_derivative(x_in, axis1=0, axis2=0)
        A_in_yy = model.features_second_derivative(x_in, axis1=1, axis2=1)
        A_on = model.features(x_on)

        A = pyrfm.concat_blocks([[-(A_in_xx + A_in_yy)], [A_on]])

        f_in = func_f(x_in, frequency=args.frequency).view(-1, 1)
        f_on = func_g(x_on, frequency=args.frequency).view(-1, 1)

        f = pyrfm.concat_blocks([[f_in], [f_on]])
        model.compute(A).solve(f)

        x_test = domain.in_sample(400, with_boundary=True)
        u_test = func_u(x_test, frequency=args.frequency)
        u_pred = model(x_test)

        error = (u_test - u_pred).norm() / u_test.norm()

    else:
        model = pyrfm.RFMBase(dim=2, n_hidden=300, domain=domain, n_subdomains=math.isqrt(args.M // 300),
                              pou=pyrfm.PsiB)
        x_in = domain.in_sample(args.Q, with_boundary=False)
        x_on = domain.on_sample(int(0.2 * args.Q))

        A_in_xx = model.features_second_derivative(x_in, axis1=0, axis2=0).cat(dim=1)
        A_in_yy = model.features_second_derivative(x_in, axis1=1, axis2=1).cat(dim=1)
        A_on = model.features(x_on).cat(dim=1)

        A = pyrfm.concat_blocks([[-(A_in_xx + A_in_yy)], [A_on]])

        f_in = func_f(x_in, frequency=args.frequency).view(-1, 1)
        f_on = func_g(x_on, frequency=args.frequency).view(-1, 1)

        f = pyrfm.concat_blocks([[f_in], [f_on]])
        model.compute(A).solve(f)

        x_test = domain.in_sample(400, with_boundary=True)
        u_test = func_u(x_test, frequency=args.frequency)
        u_pred = model(x_test)

        error = (u_test - u_pred).norm() / u_test.norm()

    print(f"\nSimulation Results:")
    print(f"--------------------------")
    print(f"Problem size: N = {A.shape[0]}, M = {A.shape[1]}")
    print(f"Relative Error: {error:.4e}")
    print(f"Elapsed Time: {time.time() - start_time:.2f} seconds")
    print("=" * 40)


param_sets = [
    {"Q": 1600, "M": 1200, "frequency": "low", "multiscale": False},
    {"Q": 3600, "M": 2700, "frequency": "low", "multiscale": False},
    {"Q": 6400, "M": 4800, "frequency": "low", "multiscale": False},
    {"Q": 1600, "M": 1200, "frequency": "high", "multiscale": False},
    {"Q": 3600, "M": 2700, "frequency": "high", "multiscale": False},
    {"Q": 6400, "M": 4800, "frequency": "high", "multiscale": False},
    {"Q": 1600, "M": 1200, "frequency": "mixed", "multiscale": False},
    {"Q": 3600, "M": 2700, "frequency": "mixed", "multiscale": False},
    {"Q": 6400, "M": 4800, "frequency": "mixed", "multiscale": False},
    {"Q": 1600, "M": 1200, "frequency": "low", "multiscale": True},
    {"Q": 3600, "M": 2700, "frequency": "low", "multiscale": True},
    {"Q": 6400, "M": 4800, "frequency": "low", "multiscale": True},
    {"Q": 1600, "M": 1200, "frequency": "high", "multiscale": True},
    {"Q": 3600, "M": 2700, "frequency": "high", "multiscale": True},
    {"Q": 6400, "M": 4800, "frequency": "high", "multiscale": True},
    {"Q": 1600, "M": 1200, "frequency": "mixed", "multiscale": True},
    {"Q": 3600, "M": 2700, "frequency": "mixed", "multiscale": True},
    {"Q": 6400, "M": 4800, "frequency": "mixed", "multiscale": True},
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--Q", type=int, required=True)
    parser.add_argument("--M", type=int, required=True)
    parser.add_argument("--frequency", type=str, required=True)
    parser.add_argument("--multiscale", type=bool, required=True)

    if len(sys.argv) == 1:
        for param_set in param_sets:
            args = argparse.Namespace(**param_set)
            run_rfm(args)
    else:
        args = parser.parse_args()
        run_rfm(args)
