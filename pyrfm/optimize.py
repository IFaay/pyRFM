# -*- coding: utf-8 -*-
"""
Created on 2025/2/20

@author: Yifei Sun
"""
import torch

from .utils import *
import numpy as np
import torch
from scipy.optimize import least_squares
from typing import Callable


def nonlinear_least_square(fcn: Callable[[torch.Tensor], torch.Tensor],
                           x0: torch.Tensor,
                           jac: Callable[[torch.Tensor], torch.Tensor],
                           method: str = None,
                           maxfev: int = None,
                           ftol: float = 1e-08,
                           xtol: float = 1e-08,
                           gtol: float = 1e-08,
                           verbose: int = 0,
                           ):
    """
    Solves a nonlinear least squares problem using different optimization methods.

    Args:
        fcn (Callable[[torch.Tensor], torch.Tensor]): The function to minimize.
        x0 (torch.Tensor): Initial guess for the variables.
        jac (Callable[[torch.Tensor], torch.Tensor]): Function to compute the Jacobian matrix.
        method (str, optional): Optimization method ('trf', 'lm', 'dogbox', 'newton'). Defaults to None.
        maxfev (int, optional): Maximum function evaluations before termination.
        ftol (float, optional): Tolerance for function value change. Defaults to 1e-08.
        xtol (float, optional): Tolerance for parameter updates. Defaults to 1e-08.
        gtol (float, optional): Tolerance for gradient norm. Defaults to 1e-08.
        verbose (int, optional): Verbosity level.
            0: No output.
            1: Output at each iteration.
            2: Output at each function evaluation. Defaults to 0.

    Returns:
        torch.Tensor: The optimized solution.
        int: Status of optimization termination:
            - 0: Maximum function evaluations exceeded.
            - 1: Gradient norm condition met.
            - 2: Function value tolerance condition met.
            - 3: Parameter update tolerance condition met.
            - 4: Both function value and parameter update tolerance met.
    """
    if method in ['trf', 'lm', 'dogbox']:
        dtype, device = x0.dtype, x0.device

        def fcn_numpy(w):
            w = torch.tensor(w, dtype=dtype, device=device).reshape(-1, 1)
            return fcn(w).cpu().numpy().flatten()

        def jac_numpy(w):
            w = torch.tensor(w, dtype=dtype, device=device).reshape(-1, 1)
            return jac(w).cpu().numpy()

        # Output problem size once (n_vars, n_residuals, jac_shape)
        if verbose >= 1:
            _x_probe = x0.detach()
            _f_probe = fcn(_x_probe)
            print(
                f"[NLS size] n_vars={_x_probe.numel()}, n_residuals={_f_probe.numel()}, jac_shape=({_f_probe.numel()}, {_x_probe.numel()})")

        result = least_squares(fun=fcn_numpy, x0=x0.cpu().numpy().flatten(), jac=jac_numpy,
                               method=method, tr_solver='exact',
                               ftol=ftol, xtol=xtol, gtol=gtol, verbose=verbose)
        return torch.tensor(result.x, dtype=dtype, device=device).reshape(-1, 1), result.status

    elif method == "newton":
        def print_header():
            print(f"{'Iter':>6} {'Cost':>12} {'||grad||_inf':>15} {'Step norm':>12}")

        def print_iteration(k, cost, grad_norm, step_norm):
            print(f"{k:6d} {cost:12.4e} {grad_norm:15.4e} {step_norm:12.4e}")

        x = x0.clone()
        k = 0
        alpha = 1.0

        if maxfev is None:
            maxfev = 100 * x0.numel()

        # Output problem size once (n_vars, n_residuals, jac_shape)
        _printed_size = False

        while True:
            F_vec = fcn(x)
            F_jac = jac(x)

            if (not _printed_size) and (verbose >= 1):
                print(f"[NLS size] n_vars={x.numel()}, n_residuals={F_vec.numel()}, jac_shape={tuple(F_jac.shape)}")
                _printed_size = True

            cost = 0.5 * torch.sum(F_vec ** 2).item()
            grad = F_jac.T @ F_vec
            grad_norm = torch.linalg.norm(grad, float('inf')).item()

            scale_inv = torch.linalg.norm(F_jac, dim=0, keepdim=True)
            scale_inv[scale_inv == 0] = 1.0

            solver = torch.linalg.lstsq(F_jac / scale_inv, -F_vec, driver='gels')
            p = solver.solution / scale_inv.T
            step_norm = torch.linalg.norm(alpha * p).item()

            if verbose >= 1:
                if k == 0:
                    print_header()
                print_iteration(k, cost, grad_norm, step_norm)
            if verbose == 2:
                print(f"  Function evaluation {k}: ||F(x)|| = {torch.linalg.norm(F_vec).item():.4e}")

            # Termination checks
            if grad_norm < gtol:
                status = 1
                break
            if torch.linalg.norm(F_vec) < ftol:
                status = 2
                break
            if step_norm < xtol * (xtol + torch.linalg.norm(x).item()):
                status = 3
                break
            if maxfev <= 0:
                status = 0
                break

            def phi(step_size):
                return torch.linalg.norm(fcn(x + step_size * p))

            alpha, maxfev = line_search(phi, 0.0, 1.0, maxfev, ftol)
            x = x + alpha * p

            k += 1
            maxfev -= 1

        if verbose >= 1:
            print(f"Terminated with status {status}")
            if status == 0:
                print("Maximum function evaluations exceeded.")
            elif status == 1:
                print("Gradient norm condition met.")
                print("Gradient norm: ", grad_norm, " < gtol: ", gtol)
            elif status == 2:
                print("Function value tolerance condition met.")
                print("Function value: ", torch.linalg.norm(F_vec), " < ftol: ", ftol)
            elif status == 3:
                print("Parameter update tolerance condition met.")
                print("Step norm: ", step_norm, " < relative_xtol: ", xtol * (xtol + torch.linalg.norm(x).item()))

        return x, status


def line_search(fn: Callable[[float], float], a, b, maxfev, ftol=1e-8):
    """
    Performs a golden-section line search to find the step size that minimizes the function.

    Args:
        fn (Callable[[float], float]): Function to minimize.
        a (float): Lower bound of the search interval.
        b (float): Upper bound of the search interval.
        maxfev (int): Maximum function evaluations.
        ftol (float, optional): Function tolerance for convergence. Defaults to 1e-08.

    Returns:
        float: Optimal step size.
        int: Updated maxfev count.
    """
    f0, f1 = fn(0.0), fn(1.0)
    maxfev -= 2
    ratio = (1.0 + 5 ** 0.5) / 2
    c, d = b - (b - a) / ratio, a + (b - a) / ratio
    fnc, fnd = fn(c), fn(d)

    while maxfev > 0:
        maxfev -= 2

        if fnc < fnd:
            if fnc < f1:
                a, b = a, d
                c, d = b - (b - a) / ratio, a + (b - a) / ratio
                fnc, fnd = fn(c), fn(d)
            else:
                a, b = c, 1.0
                c, d = b - (b - a) / ratio, a + (b - a) / ratio
                fnc, fnd = fn(c), fn(d)
        else:
            if fnd < f0:
                a, b = c, b
                c, d = b - (b - a) / ratio, a + (b - a) / ratio
                fnc, fnd = fn(c), fn(d)
            else:
                a, b = 0.0, d
                c, d = b - (b - a) / ratio, a + (b - a) / ratio
                fnc, fnd = fn(c), fn(d)

        if abs(fnc - fnd) < 0.5 * ftol * (abs(fnc) + abs(fnd)):
            if a == 0.0:
                return 0.0, maxfev
            return (a + b) / 2, maxfev
    return (a + b) / 2, maxfev


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

                logger.info("Residual: {}".format(torch.linalg.norm(res) / torch.linalg.norm(torch.ones_like(res))))
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

            logger.info("Residual: {}".format(torch.linalg.norm(res) / torch.linalg.norm(torch.ones_like(res))))

    def get_solution(self):
        """
        Solves the linear system using the QR decomposition.

        Returns:
            torch.Tensor: Solution of the linear system.
        """
        self.solution = torch.linalg.solve_triangular(self.R, self.Qtb, upper=True)
        return self.solution

# def lmpar(n: int, r: torch.Tensor,
#           Diag: torch.Tensor,
#           Qtb: torch.Tensor,
#           Delta: torch.Tensor,
#           Par: float,
#           x: torch.Tensor,
#           Sdiag: torch.Tensor,
#           Wa1: torch.Tensor,
#           Wa2: torch.Tensor):
#     """
#     Determine parameter par such that the least squares solution x satisfies:   A*x ≈ b,  sqrt(par)*D*x ≈ 0
#     with the Euclidean norm of D*x (dxnorm) meeting:
#     When par=0: (dxnorm - delta) ≤ 0.1*delta
#     When par>0: |dxnorm - delta| ≤ 0.1*delta
#
#     Args:
#         n:
#         r:
#         Diag:
#         Qtb:
#         Delta:
#         Par:
#         x:
#         Sdiag:
#         Wa1:
#         Wa2:
#
#     Returns:
#
#     """
#     p1 = 0.1
#     dwarf = torch.finfo(torch.tensor(0.0).dtype).tiny  # Smallest positive value
#
#     nsing = n
#     Wa1 = torch.zeros((n, 1))
#     for j in range(n):
#         Wa1[j] = Qtb[j]
#         if r[j, j] == 0 and nsing == n:
#             nsing = j
#         if nsing < n:
#             Wa1[j] = 0.0
#     x = torch.linalg.solve_triangular(torch.triu(r)[:n], Wa1, upper=True)
#
#     iter = 0
#     Wa2 = Diag * x
#     dxnorm = torch.linalg.norm(Wa2)
#     fp = dxnorm - Delta
#
#     if fp <= p1 * Delta:
#         if iter == 0:
#             Par = 0.0
#
#     else:
#         parl = 0.0
#         if nsing >= n:
#             Wa1 = Diag * (Wa2 / dxnorm)
#             for j in range(n):
#                 sum = 0.0
#                 jm1 = j - 1
#                 if jm1 >= 0:
#                     for i in range(jm1):
#                         sum += r[i, j] * Wa1[i]
#                 Wa1[j] = (Wa1[j] - sum) / r[j, j]
#             temp = torch.linalg.norm(Wa1)
#             parl = ((fp / Delta) / temp) / temp
#
#     for j in range(n):
#         sum = 0.0
#         for i in range(j):
#             sum += r[i, j] * Qtb[i]
#
#         Wa1[j] = sum / Diag[j]
#
#     gnorm = torch.linalg.norm(Wa1)
#     paru = gnorm / Delta
#     if paru == 0.0:
#         paru = dwarf / min(Delta, p1)
#
#     Par = max(Par, parl)
#     Par = min(Par, paru)
#     if Par == 0.0:
#         Par = gnorm / dxnorm
#
#     while True:
#         iter += 1
#
#         if Par == 0.0:
#             Par = max(dwarf, 0.001 * paru)
#         Wa1 = Par ** 0.5 * Diag
#         r, tau = torch.geqrf(torch.cat([r, torch.diagflat(Wa1)], dim=0))
#         x = torch.linalg.solve_triangular(r[:n, :n], Qtb[:n], upper=True)
#         Wa2 = Diag * x
#         dxnorm = torch.linalg.norm(Wa2)
#         temp = fp
#         fp = dxnorm - Delta
#
#         if torch.abs(fp) <= p1 * Delta or parl == 0.0 and fp <= temp and temp < 0.0 or iter == 10:
#             if iter == 0:
#                 Par = 0.0
#             return r, Diag, Par, x, Sdiag, Wa1, Wa2
#         else:
#             for j in range(n):
#                 Wa1[j] = Diag[j] * (Wa2[j] / dxnorm)
#
#             Wa1 = torch.linalg.solve_triangular(r[:n, :n], Wa1, upper=True)
#             temp = torch.linalg.norm(Wa1)
#             parc = ((fp / Delta) / temp) / temp
#             if fp > 0:
#                 parl = max(parl, Par)
#             if fp < 0:
#                 paru = min(paru, Par)
#             Par = max(parl, Par + parc)
#
#
# def nonlinear_least_square_2(fcn: Callable[[torch.Tensor], torch.Tensor],
#                              x0: torch.Tensor,
#                              jac: Callable[[torch.Tensor], torch.Tensor],
#                              method: str = None,
#                              maxfev: int = None,
#                              outer_iter: Optional[int] = None,
#                              inner_iter: Optional[int] = None,
#                              ftol: float = 1e-08,
#                              xtol: float = 1e-08,
#                              gtol: float = 1e-08):
#     # Bad performance here. Need to be fixed.
#     dpmpar = [torch.finfo(torch.tensor(0.0).dtype).eps,
#               torch.finfo(torch.tensor(0.0).dtype).tiny,
#               torch.finfo(torch.tensor(0.0).dtype).max]
#     epsmch = dpmpar[0]
#     # Levenberg-Marquardt algorithm
#     Factor = 100.
#     Info = 0
#     iflag = 0
#     Nfev = 0
#     Njev = 0
#
#     # evaluate the function at the starting point and calculate its norm.
#     iflag = 1
#     Fvec = fcn(x0)
#     Fjac = jac(x0)
#     m, n = Fjac.shape
#     if maxfev is None:
#         maxfev = 100 * n
#     Nfev = 1
#     if iflag < 0:
#         return
#     fnorm = torch.linalg.norm(Fvec, ord=2)
#
#     Wa1: torch.Tensor = torch.zeros(
#         (n, 1))  # an output array of length n which contains the diagonal elements of r.
#     Wa2: torch.Tensor = torch.zeros(
#         (n,
#          1))  # an output array of length n which contains the norms of the corresponding columns of the input matrix a.
#     Wa3: torch.Tensor = torch.zeros(
#         (n, 1))  # a work array of length n. if pivot is false, then wa can coincide with rdiag.
#     Diag: torch.Tensor = torch.zeros((n, 1))  # an array of length n, diag is internally set.
#     x = x0
#
#     # initialize levenberg-marquardt parameter and iteration counter.
#     par = 0.0
#     iter = 1
#
#     while True:
#         # calculate the jacobian matrix.
#         iflag = 2
#         Fvec = fcn(x)
#         Fjac = jac(x)
#         Njev += 1
#         if iflag < 0:
#             return x, Info
#
#         # compute the qr factorization of the jacobian.
#
#         Fjac, tau = torch.geqrf(Fjac)
#         Wa1 = torch.linalg.diagonal(Fjac).reshape(-1, 1)
#         Wa2 = torch.linalg.norm(Fjac, ord=2, dim=0).reshape(-1, 1)
#         Wa3 = Wa1
#
#         if iter == 1:
#             Diag = Wa2
#             Diag[Diag == 0] = 1.0
#             Wa3 = Diag * x
#             xnorm = torch.linalg.norm(Wa3, ord=2)
#             delta = Factor * xnorm
#             if delta == 0:
#                 delta = Factor
#
#         Qtf = Wa4 = torch.ormqr(Fjac, tau, Fvec, transpose=True)
#
#         # compute the norm of the scaled gradient.
#         gnorm = 0.0
#         if fnorm != 0:
#             for j in range(n):
#                 l = j
#                 if Wa2[l] != 0:
#                     sum = 0.0
#                     for i in range(j):
#                         sum += Fjac[i, j] * Qtf[i] / fnorm
#                     gnorm = max(gnorm, abs(sum / Wa2[l]))
#
#         # test for convergence of the gradient norm.
#         if gnorm <= gtol:
#             Info = 4
#         if Info != 0:
#             return x, Info
#
#         Diag = torch.max(Diag, Wa2)
#
#         while True:
#             Fjac, Diag, par, Wa1, Wa2, Wa3, Wa4 = lmpar(n, Fjac, Diag, Qtf, delta, par, Wa1, Wa2, Wa3, Wa4)
#
#             Wa1 = -Wa1
#             Wa2 = x + Wa1
#             Wa3 = Diag * Wa1
#
#             pnorm = torch.linalg.norm(Wa3)
#
#             if iter == 1:
#                 delta = min(delta, pnorm)
#
#             # evaluate the function at x + p and calculate its norm.
#
#             iflag = 1
#
#             Wa4 = fcn(Wa2)
#             print("Evaluated function at x + p", Wa4.norm())
#             Fjac = jac(Wa2)
#             Nfev += 1
#
#             if iflag < 0:
#                 return x, Info
#             fnorm1 = torch.linalg.norm(Wa4)
#
#             # compute the scaled actual reduction.
#             actred = -1.0
#             if 0.1 * fnorm1 < fnorm:
#                 actred = 1.0 - (fnorm1 / fnorm) ** 2
#
#             # compute the scaled predicted reduction and the scaled directional derivative.
#
#             for j in range(n):
#                 Wa3[j] = 0.0
#                 temp = Wa1[j]
#                 for i in range(j):
#                     Wa3[i] += Fjac[i, j] * temp
#             temp1 = torch.linalg.norm(Wa3) / fnorm
#             temp2 = (par ** 0.5 * pnorm) / fnorm
#             prered = temp1 ** 2 + temp2 ** 2 / 0.5
#             dirder = -(temp1 ** 2 + temp2 ** 2)
#
#             # compute the ratio of the actual to the predicted reduction.
#             ratio = 0.0
#             if prered != 0:
#                 ratio = actred / prered
#
#             # update the step bound.
#
#             if ratio <= 0.25:
#                 if actred >= 0:
#                     temp = 0.5
#                 else:
#                     temp = 0.5 * dirder / (dirder + 0.5 * actred)
#                 if 0.1 * fnorm1 >= fnorm or temp < 0.1:
#                     temp = 0.1
#                 delta = temp * min(delta, pnorm / 0.1)
#                 par /= temp
#             elif par == 0 or ratio >= 0.75:
#                 delta = pnorm / 0.5
#                 par *= 0.5
#
#             # test for successful iteration.
#
#             if ratio >= 0.0001:
#                 # successful iteration. update x, fvec, and their norms.
#                 x = Wa2
#                 Wa2 = Diag * x
#                 Fvec = Wa4
#                 xnorm = torch.linalg.norm(Wa2)
#                 fnorm = fnorm1
#                 iter += 1
#
#             # tests for convergence.
#             if torch.abs(actred) <= ftol and prered <= ftol and 0.5 * ratio <= 1:
#                 Info = 1
#             if delta <= xtol * xnorm:
#                 Info = 2
#             if torch.abs(actred) <= ftol and prered <= ftol and 0.5 * ratio <= 1 and Info == 2:
#                 Info = 3
#             if Info != 0:
#                 return x, Info
#
#             # tests for termination and stringent tolerances.
#             if Nfev >= maxfev:
#                 Info = 5
#             if abs(actred) <= epsmch and prered <= epsmch and 0.5 * ratio <= 1:
#                 Info = 6
#             if delta <= epsmch * xnorm:
#                 Info = 7
#             if gnorm <= epsmch:
#                 Info = 8
#             if Info != 0:
#                 return x, Info
#
#             if ratio >= 0.0001:
#                 break
#
#         # rescale if necessary.
#         Diag = torch.max(Diag, torch.linalg.norm(Fjac, ord=2, dim=0).reshape(-1, 1))
#
#         # form (q transpose)*fvec and store the first n components in qtf.
#         Wa4 = Fvec.clone()
#         for j in range(n):
#             if Fjac[j, j] != 0:
#                 sum = 0.0
#                 for i in range(j, m):
#                     sum += Fjac[i, j] * Wa4[i]
#                 temp = -sum / Fjac[j, j]
#                 for i in range(j, m):
#                     Wa4[i] += Fjac[i, j] * temp
