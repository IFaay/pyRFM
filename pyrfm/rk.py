# -*- coding: utf-8 -*-
"""
Runge-Kutta time integration methods for PDE solvers based on Random Feature Methods (RFM).

This module provides explicit embedded Runge-Kutta schemes (RK23 and RK45)
implemented in PyTorch, with coefficients taken from classical literature.
These integrators are designed to work with RFM-based PDE solvers where
the spatial problem is solved via a regression model (RFMBase), while time
discretization is done explicitly using Runge-Kutta.

Created on 2025/10/30

@author: Yifei Sun
"""

from .utils import *
from .core import *
import numpy as np
import torch
from typing import Callable, List


class RungeKutta:
    """
    Base class for explicit Runge-Kutta schemes.

    Each RK method specifies the Butcher tableau coefficients:
        C — stage time fractions (vector)
        A — stage coefficients (lower triangular matrix for explicit RK)
        B — final combination coefficients for the time step
        E — error estimation coefficients (embedded method)
        P — dense output interpolation coefficients (for continuous solution)

    Attributes
    ----------
    C : torch.Tensor
        Stage nodes c_i (time fractions).
    A : torch.Tensor
        Runge-Kutta coefficient matrix (lower triangular for explicit RK).
    B : torch.Tensor
        Output weights for advancing the solution.
    E : torch.Tensor
        Weights for embedded error estimator.
    P : torch.Tensor
        Coefficients for dense output polynomial interpolation.
    order : int
        Order of accuracy of the main scheme.
    error_estimator_order : int
        Order of the embedded (error estimator) scheme.
    n_stages : int
        Number of RK stages.
    """

    C: torch.Tensor = NotImplemented
    A: torch.Tensor = NotImplemented
    B: torch.Tensor = NotImplemented
    E: torch.Tensor = NotImplemented
    P: torch.Tensor = NotImplemented
    order: int = NotImplemented
    error_estimator_order: int = NotImplemented
    n_stages: int = NotImplemented

    def rk_step(self, func: Callable, x: torch.Tensor, t: float, dt: float,
                u0: torch.Tensor, rk_model):
        """
        Perform a single explicit Runge-Kutta time step.

        Parameters
        ----------
        func : Callable
            Right-hand side function f(model, x, t) returning time derivative du/dt.
            This function should evaluate the PDE residual or learned dynamics.
        x : torch.Tensor
            Input spatial points (batch_size × dim).
        t : float
            Current time.
        dt : float
            Time step size.
        u0 : torch.Tensor
            Current solution values at x.
        rk_model : RFMBase
            The RFM model that solves the spatial system. It must implement `.solve()`.

        Returns
        -------
        K : List[torch.Tensor]
            List of stage derivatives k_i.
        u : torch.Tensor
            Updated solution after one RK step.

        Notes
        -----
        * u0 is the solution at time t.
        * Intermediate solves enforce boundary conditions at each stage.
        * Uses explicit RK: k_i = f(u_i, t_i) where u_i is stage solution.
        """

        if not isinstance(rk_model, RFMBase):
            raise TypeError("rk_model must be an instance of RFMBase.")
        if x.shape[0] != u0.shape[0]:
            raise ValueError("The batch size of x and u0 must be the same.")

        K: List[torch.Tensor] = []

        for s in range(self.n_stages):
            t_s = t + self.C[s] * dt
            u = u0 + dt * sum(self.A[s, j] * K[j] for j in range(s))

            # Enforce PDE operator/BC via RFM solve at each stage
            rk_model.solve(u, verbose=False)

            # Evaluate RHS at current stage
            K.append(func(rk_model, x, t_s))

        u = u0 + dt * sum(self.B[j] * K[j] for j in range(self.n_stages))
        return K, u


class RK23(RungeKutta):
    """
    Bogacki–Shampine explicit Runge-Kutta 3(2) method.

    A low-storage embedded RK method providing a third-order accurate step
    with a second-order embedded error estimator, suitable for adaptive
    time stepping in moderately stiff problems.

    References
    ----------
    .. [1] P. Bogacki, L.F. Shampine,
           "A 3(2) Pair of Runge-Kutta Formulas",
           Appl. Math. Lett., 2(4), 321–325, 1989.
    """

    order = 3
    error_estimator_order = 2
    n_stages = 3
    C = np.array([0, 1 / 2, 3 / 4])
    A = np.array([
        [0, 0, 0],
        [1 / 2, 0, 0],
        [0, 3 / 4, 0]
    ])
    B = np.array([2 / 9, 1 / 3, 4 / 9])
    E = np.array([5 / 72, -1 / 12, -1 / 9, 1 / 8])
    P = np.array([[1, -4 / 3, 5 / 9],
                  [0, 1, -2 / 3],
                  [0, 4 / 3, -8 / 9],
                  [0, -1, 1]])


class RK45(RungeKutta):
    """
    Dormand–Prince explicit Runge-Kutta 5(4) method (a.k.a. DOPRI5).

    Widely used embedded RK scheme providing 5th-order accurate steps and
    4th-order error estimation. This is the classical method used in
    MATLAB `ode45` and SciPy `solve_ivp(method="RK45")`.

    References
    ----------
    .. [1] J.R. Dormand, P.J. Prince,
           "A family of embedded Runge-Kutta formulae",
           J. Comput. Appl. Math., 6(1), 19–26, 1980.
    .. [2] L.W. Shampine,
           "Some Practical Runge-Kutta Formulas",
           Math. Comput., 46(173), 135–150, 1986.
    """

    order = 5
    error_estimator_order = 4
    n_stages = 6
    C = np.array([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1])
    A = np.array([
        [0, 0, 0, 0, 0],
        [1 / 5, 0, 0, 0, 0],
        [3 / 40, 9 / 40, 0, 0, 0],
        [44 / 45, -56 / 15, 32 / 9, 0, 0],
        [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0],
        [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656]
    ])
    B = np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84])
    E = np.array([-71 / 57600, 0, 71 / 16695, -71 / 1920, 17253 / 339200, -22 / 525,
                  1 / 40])
    # Corresponds to the optimum value of c_6 from [2]_.
    P = np.array([
        [1, -8048581381 / 2820520608, 8663915743 / 2820520608,
         -12715105075 / 11282082432],
        [0, 0, 0, 0],
        [0, 131558114200 / 32700410799, -68118460800 / 10900136933,
         87487479700 / 32700410799],
        [0, -1754552775 / 470086768, 14199869525 / 1410260304,
         -10690763975 / 1880347072],
        [0, 127303824393 / 49829197408, -318862633887 / 49829197408,
         701980252875 / 199316789632],
        [0, -282668133 / 205662961, 2019193451 / 616988883, -1453857185 / 822651844],
        [0, 40617522 / 29380423, -110615467 / 29380423, 69997945 / 29380423]])
