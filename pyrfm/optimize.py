# -*- coding: utf-8 -*-
"""
Created on 2025/2/20

@author: Yifei Sun
"""
import torch

from .utils import *


def nonlinear_least_square(fun, x0, jac, ftol=1e-08, xtol=1e-08, gtol=1e-08):
    pass


class GivensRotation:
    def __init__(self, a, b, i, k):
        a, b = float(a), float(b)
        self.i = i
        self.k = k
        if b == 0.0:
            self.c = 1.0
            self.s = 0.0
        else:
            if abs(b) > abs(a):
                tau = a / b
                self.s = 1.0 / (1 + tau ** 2) ** 0.5
                self.c = self.s * tau
            else:
                tau = b / a
                self.c = 1.0 / (1 + tau ** 2) ** 0.5
                self.s = self.c * tau

    def __matmul__(self, other):
        if isinstance(other, torch.Tensor):
            result = other.clone()
            row_i = result[self.i]
            row_k = result[self.k]
            result[self.i] = self.c * row_i + self.s * row_k
            result[self.k] = -self.s * row_i + self.c * row_k
            return result


class GivensQR:
    def __init__(self, m, n_rhs):
        self.m = m  # number of cols
        self.n_rhs = n_rhs  # number of right hand sides
        self.R = torch.zeros((0, m))
        self.Qtb = torch.zeros((0, n_rhs))
        self.solution = None

    def add_rows(self, A_new: torch.Tensor, b_new: torch.Tensor):

        if not self.R.shape[0] == self.m:
            if self.R.shape[0] + A_new.shape[0] < self.m:
                self.R = torch.cat([self.R, A_new], dim=0)
                self.Qtb = torch.cat([self.Qtb, b_new], dim=0)
                logger.warn("More conditions are needed to determine the solution.")
                return

            elif self.R.shape[0] + A_new.shape[0] == self.m:
                self.R = torch.cat([self.R, A_new], dim=0)
                self.Qtb = torch.cat([self.Qtb, b_new], dim=0)
                self.R, tau = torch.geqrf(self.R)
                self.Qtb = torch.ormqr(self.R, tau, self.Qtb, transpose=True)[:self.R.shape[1]]
                self.R = torch.triu(self.R)
                logger.warn("More conditions are needed to determine the solution.")
                return
            else:
                self.R = torch.cat([self.R, A_new[:self.m - self.R.shape[0]]], dim=0)
                self.Qtb = torch.cat([self.Qtb, b_new[:self.m - self.R.shape[0]]], dim=0)
                self.R, tau = torch.geqrf(self.R)
                self.Qtb = torch.ormqr(self.R, tau, self.Qtb, transpose=True)[:self.R.shape[1]]
                self.R = torch.triu(self.R)
                A_new = A_new[self.m:]
                b_new = b_new[self.m:]

        self.R = torch.cat([self.R, A_new], dim=0)
        self.Qtb = torch.cat([self.Qtb, b_new], dim=0)

        for i in range(self.m, self.R.shape[0]):
            for k in range(self.m):
                givens = GivensRotation(self.R[k, k], self.R[i, k], k, i)
                self.R = givens @ self.R
                self.Qtb = givens @ self.Qtb

        self.R = torch.triu(self.R[:self.m])
        self.Qtb = self.Qtb[:self.m]

        self.solution = torch.linalg.solve_triangular(self.R, self.Qtb, upper=True)
        logger.info("Relative Residual: {}".format(torch.norm(A_new @ self.solution - b_new) / torch.norm(b_new)))
