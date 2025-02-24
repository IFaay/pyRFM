# -*- coding: utf-8 -*-
"""
Created on 2025/2/20

@author: Yifei Sun
"""
import torch

from .utils import *
from typing import Callable


def nonlinear_least_square(fcn: Callable[[torch.Tensor], torch.Tensor],
                           x0: torch.Tensor,
                           jac: Callable[[torch.Tensor], torch.Tensor],
                           outer_iter: Optional[int] = None,
                           inner_iter: Optional[int] = None,
                           ftol: float = 1e-08,
                           xtol: float = 1e-08,
                           gtol: float = 1e-08):
    """
    Solves a nonlinear least squares problem.

    Args:
        fcn (Callable[[torch.Tensor], torch.Tensor]): The function to minimize.
        x0 (torch.Tensor): Initial guess for the variables.
        jac (Callable[[torch.Tensor], torch.Tensor]): Function to compute the Jacobian matrix.
        outer_iter (int, optional): Maximum number of outer iterations. Defaults to None.
        inner_iter (int, optional): Maximum number of inner iterations. Defaults to None.
        ftol (float, optional): Tolerance for the function value. Defaults to 1e-08.
        xtol (float, optional): Tolerance for the solution. Defaults to 1e-08.
        gtol (float, optional): Tolerance for the gradient. Defaults to 1e-08.
    """
    F_vec = fcn(x0)
    F_jac = jac(x0)
    F_norm = torch.linalg.norm(F_jac, ord=2, dim=1, keep_dim=True)
    F_vec, F_jac = F_vec / F_norm, F_jac / F_norm

    p = torch.lstsq(F_jac, -F_vec)[0]
    x = x0 + p

    while True:
        if torch.norm(p) < xtol:
            break

        if torch.norm(F_vec) < ftol:
            break

        # if torch.norm(F_jac @ p) < gtol:
        #     break

        F_vec = fcn(x)
        F_jac = jac(x)
        F_norm = torch.linalg.norm(F_jac, ord=2, dim=1, keep_dim=True)
        F_vec, F_jac = F_vec / F_norm, F_jac / F_norm

        p = torch.lstsq(F_jac, -F_vec)[0]

        def phi(step_size):
            return torch.linalg.norm(fcn(x + step_size * p))

        if inner_iter is not None:
            alpha = line_search(phi, 0.0, 1.0, max_iter=inner_iter)
        else:
            alpha = line_search(phi, 0.0, (1.0 + 5 ** 0.5) / 2)

        x = x + alpha * p

        if outer_iter is not None:
            if outer_iter == 0:
                break
            outer_iter -= 1


def line_search(fn: Callable[[float], float], a, b, max_iter=10):
    ratio = (1.0 + 5 ** 0.5) / 2
    c = b - (b - a) / ratio
    d = a + (b - a) / ratio
    for _ in range(max_iter):
        if fn(c) < fn(d):
            b = d
        else:
            a = c

        c = b - (b - a) / ratio
        d = a + (b - a) / ratio
    return (a + b) / 2


class GivensRotation:
    def __init__(self, a, b, i, k):
        a, b = float(a), float(b)
        self.i = i
        self.k = k
        if b == 0.0:
            self.c = torch.tensor(1.0)
            self.s = torch.tensor(0.0)
        else:
            if abs(b) > abs(a):
                tau = a / b
                self.s = torch.tensor(1.0 / (1 + tau ** 2) ** 0.5)
                self.c = torch.tensor(self.s * tau)
            else:
                tau = b / a
                self.c = torch.tensor(1.0 / (1 + tau ** 2) ** 0.5)
                self.s = torch.tensor(self.c * tau)

    def apply(self, other):
        if isinstance(other, torch.Tensor):
            row_i = other[self.i].clone()
            row_k = other[self.k].clone()
            other[self.i] = row_i * self.c + row_k * self.s
            other[self.k] = -row_i * self.s + row_k * self.c


def givens(a, b):
    a, b = float(a), float(b)
    if b == 0.0:
        c = 1.0
        s = 0.0
    else:
        if abs(b) > abs(a):
            tau = a / b
            s = 1.0 / (1 + tau ** 2) ** 0.5
            c = s * tau
        else:
            tau = b / a
            c = 1.0 / (1 + tau ** 2) ** 0.5
            s = c * tau
    return torch.tensor(c), torch.tensor(s)


class BatchQR:
    """
    Class to perform Batch QR decomposition for solving linear systems.

    Attributes:
        m (int): Number of columns.
        n_rhs (int): Number of right-hand sides.
        R (torch.Tensor): Upper triangular matrix from QR decomposition.
        Qtb (torch.Tensor): Product of Q^T and b.
        solution (torch.Tensor or None): Solution of the linear system.
    """

    def __init__(self, m, n_rhs):
        """
        Initializes the BatchQR object with the given dimensions.

        Args:
            m (int): Number of columns.
            n_rhs (int): Number of right-hand sides.
        """
        self.m = m  # number of cols
        self.n_rhs = n_rhs  # number of right hand sides
        self.R = torch.zeros((0, m))
        self.Qtb = torch.zeros((0, n_rhs))
        self.solution = None

    def add_rows(self, A_new: torch.Tensor, b_new: torch.Tensor):
        """
        Adds new rows to the QR decomposition.

        Args:
            A_new (torch.Tensor): New rows to add to the matrix A.
            b_new (torch.Tensor): New rows to add to the vector b.
        """
        A_new_norm = torch.linalg.norm(A_new, ord=2, dim=1, keepdim=True)
        A_new, b_new = A_new / A_new_norm, b_new / A_new_norm

        if not self.R.shape[0] == self.m:
            self.R = torch.cat([self.R, A_new], dim=0)
            self.Qtb = torch.cat([self.Qtb, b_new], dim=0)

            if self.R.shape[0] >= self.m:
                self.R, tau = torch.geqrf(self.R)
                self.Qtb = torch.ormqr(self.R, tau, self.Qtb, transpose=True)
                res = self.Qtb[self.m:]
                self.Qtb = self.Qtb[:self.m]
                self.R = torch.triu(self.R)[:self.m]

                logger.info("Residual: {}".format(torch.norm(res) / torch.norm(torch.ones_like(res))))
                return

            logger.warn("More conditions are needed to determine the solution.")
            return

        else:
            print(self.R.shape, A_new.shape)

            self.R = torch.cat([self.R, A_new], dim=0)
            self.Qtb = torch.cat([self.Qtb, b_new], dim=0)

            self.R, tau = torch.geqrf(self.R)
            self.Qtb = torch.ormqr(self.R, tau, self.Qtb, transpose=True)
            res = self.Qtb[self.m:]
            self.Qtb = self.Qtb[:self.m]
            self.R = torch.triu(self.R)[:self.m]

            logger.info("Residual: {}".format(torch.norm(res) / torch.norm(torch.ones_like(res))))

    def get_solution(self):
        """
        Solves the linear system using the QR decomposition.

        Returns:
            torch.Tensor: Solution of the linear system.
        """
        self.solution = torch.linalg.solve_triangular(self.R, self.Qtb, upper=True)
        return self.solution
