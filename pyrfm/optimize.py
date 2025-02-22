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
    def __init__(self, m, n_rhs):
        self.m = m  # number of cols
        self.n_rhs = n_rhs  # number of right hand sides
        self.R = torch.zeros((0, m))
        self.Qtb = torch.zeros((0, n_rhs))
        self.solution = None

    def add_rows(self, A_new: torch.Tensor, b_new: torch.Tensor):
        A_new_norm = torch.linalg.norm(A_new, ord=2, dim=1, keepdim=True)
        A_new, b_new = A_new / A_new_norm, b_new / A_new_norm

        # self.R = torch.cat([self.R, A_new], dim=0)
        # self.Qtb = torch.cat([self.Qtb, b_new], dim=0)
        #
        # if self.R.shape[0] >= self.m:
        #     self.R, tau = torch.geqrf(self.R)
        #     self.Qtb = torch.ormqr(self.R, tau, self.Qtb, transpose=True)[:self.m]
        #     self.R = torch.triu(self.R)[:self.m]
        #
        #     self.solution = torch.linalg.solve_triangular(self.R, self.Qtb, upper=True)
        #     logger.info("Relative Residual: {}".format(torch.norm(A_new @ self.solution - b_new)))
        #
        # else:
        #     logger.warn("More conditions are needed to determine the solution.")
        #     return


        if not self.R.shape[0] == self.m:
            self.R = torch.cat([self.R, A_new], dim=0)
            self.Qtb = torch.cat([self.Qtb, b_new], dim=0)

            if self.R.shape[0] >=  self.m:
                self.R, tau = torch.geqrf(self.R)
                self.Qtb = torch.ormqr(self.R, tau, self.Qtb, transpose=True)[:self.m]
                self.R = torch.triu(self.R)[:self.m]

                self.solution = torch.linalg.solve_triangular(self.R, self.Qtb, upper=True)

                logger.info("Relative Residual: {}".format(torch.norm(A_new @ self.solution - b_new) / torch.norm(b_new)))
                return

            logger.warn("More conditions are needed to determine the solution.")
            return

        else:
            print(self.R.shape, A_new.shape)

            self.R = torch.cat([self.R, A_new], dim=0)
            self.Qtb = torch.cat([self.Qtb, b_new], dim=0)


            # for i in range(self.m, self.R.shape[0]):
            #     # c = torch.zeros(self.m)
            #     # s = torch.zeros(self.m)
            #     # print(self.R[i][self.R[i] != 0])
            #     for k in range(self.m):
            #
            #         if self.R[i, k] == 0:
            #             continue
            #         # c[k], s[k] = givens(self.R[k, k], self.R[i, k])
            #
            #
            #         c, s = givens(self.R[k, k], self.R[i, k])
            #         temp1 = c * self.R[k, k:] + s * self.R[i, k:]
            #         self.R[i, k:] = -s * self.R[k, k:] + c * self.R[i, k:]
            #         self.R[k, k:] = temp1
            #
            #         temp2 = c * self.Qtb[k] + s * self.Qtb[i]
            #         self.Qtb[i] = -s * self.Qtb[k] + c * self.Qtb[i]
            #         self.Qtb[k] = temp2
            #         # givens = GivensRotation(self.R[k, k], self.R[i, k], k, i)
            #         # givens.apply(self.R)
            #         # givens.apply(self.Qtb)
            #         # self.R = givens @ self.R
            #         # self.Qtb = givens @ self.Qtb
            #     # print(self.R[i][self.R[i] != 0])

            self.R, tau = torch.geqrf(self.R)
            self.Qtb = torch.ormqr(self.R, tau, self.Qtb, transpose=True)[:self.m]
            self.R = torch.triu(self.R)[:self.m]

            # self.R = torch.triu(self.R[:self.m])
            # self.Qtb = self.Qtb[:self.m]

            self.solution = torch.linalg.solve_triangular(self.R, self.Qtb, upper=True)

            logger.info("Relative Residual: {}".format(torch.norm(A_new @ self.solution - b_new) / torch.norm(b_new)))
