# -*- coding: utf-8 -*-
"""
Implicit-domain quadrature utilities for pyRFM.

This module provides:
- 1D Gauss-Legendre rules on [0, 1]
- Interval bounds for sign/monotonicity analysis
- High-order quadrature generation for implicit domains in hyperrectangles
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import product as cart_product
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

from .utils import *

from mpmath import mp, mpf

MAX_SUBDIV_LEVEL = 16
MAX_ROOTFIND_LEVEL = 40


class GaussQuad:
    """
    Gauss-Legendre quadrature on the interval [0, 1].

    Mirrors the algoim::GaussQuad C++ struct:
        x(p, i)  -> position of node i for the p-point rule
        w(p, i)  -> weight   of node i for the p-point rule

    Parameters
    ----------
    dps : int
        Decimal places of precision used internally by mpmath.
        Default 20 matches algoim's ~20-digit tables.
        algoim header comment: "Around 20 digits of accuracy".
    p_max : int
        Maximum supported p (number of nodes).  Default 100 matches algoim.
    cache : bool
        If True (default), computed rules are cached; subsequent calls for the
        same p are O(1) lookups.
    """

    P_MAX_DEFAULT = 100

    def __init__(self, dps: int = 20, p_max: int = P_MAX_DEFAULT, cache: bool = True):
        if dps < 1:
            raise ValueError("dps must be >= 1")
        if p_max < 1:
            raise ValueError("p_max must be >= 1")

        self._dps = dps
        self._p_max = p_max
        self._use_cache = cache
        self._cache: dict = {}  # p -> (nodes_sorted, weights_sorted)

    # ------------------------------------------------------------------
    # Public API matching algoim
    # ------------------------------------------------------------------

    @property
    def p_max(self) -> int:
        """Maximum supported number of quadrature nodes."""
        return self._p_max

    @property
    def dps(self) -> int:
        """Decimal places of precision."""
        return self._dps

    def x(self, p: int, i: int) -> mpf:
        """
        Node i (0-indexed) of the p-point Gauss-Legendre rule on [0, 1].

        Equivalent to algoim::GaussQuad::x(p, i).
        """
        self._validate(p, i)
        nodes, _ = self._get_rule(p)
        return nodes[i]

    def w(self, p: int, i: int) -> mpf:
        """
        Weight i (0-indexed) of the p-point Gauss-Legendre rule on [0, 1].

        Equivalent to algoim::GaussQuad::w(p, i).
        """
        self._validate(p, i)
        _, weights = self._get_rule(p)
        return weights[i]

    def rule(
            self, p: int, as_float: bool = True
    ) -> Tuple[List, List]:
        """
        Return (nodes, weights) for the p-point rule on [0, 1].

        Parameters
        ----------
        p : int
            Number of quadrature points.
        as_float : bool
            If True (default) return plain Python floats (float64).
            If False return mpmath mpf objects at full precision.

        Returns
        -------
        nodes : list of length p, sorted ascending
        weights : list of length p (corresponding to sorted nodes)
        """
        if not (1 <= p <= self._p_max):
            raise ValueError(f"p must be in [1, {self._p_max}], got {p}")
        nodes, weights = self._get_rule(p)
        if as_float:
            return [float(x) for x in nodes], [float(w) for w in weights]
        return list(nodes), list(weights)

    def integrate(self, f, p: int) -> mpf:
        """
        Numerically integrate f(x) over [0, 1] using the p-point rule.

        Parameters
        ----------
        f : callable
            Function f(x) accepting an mpf argument.
        p : int
            Number of quadrature points.
        """
        if not (1 <= p <= self._p_max):
            raise ValueError(f"p must be in [1, {self._p_max}], got {p}")
        nodes, weights = self._get_rule(p)
        with mp.workdps(self._dps):
            return sum(w * f(x) for x, w in zip(nodes, weights))

    def precompute_all(self) -> None:
        """Pre-fill cache for all p in [1, p_max]. Useful for warm-up."""
        for p in range(1, self._p_max + 1):
            self._get_rule(p)

    def clear_cache(self) -> None:
        """Clear the internal rule cache."""
        self._cache.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate(self, p: int, i: int) -> None:
        if not (1 <= p <= self._p_max):
            raise ValueError(f"p must be in [1, {self._p_max}], got {p}")
        if not (0 <= i < p):
            raise ValueError(f"i must be in [0, {p - 1}] for p={p}, got {i}")

    def _get_rule(self, p: int) -> Tuple[list, list]:
        """Return cached or freshly computed (nodes, weights) for p."""
        if self._use_cache and p in self._cache:
            return self._cache[p]

        nodes, weights = self._compute(p)

        if self._use_cache:
            self._cache[p] = (nodes, weights)
        return nodes, weights

    def _compute(self, p: int) -> Tuple[list, list]:
        """
        Compute p-point Gauss-Legendre nodes and weights on [0, 1].

        mpmath's gauss_quadrature with qtype='legendre01' gives the rule
        directly on [0, 1] with W(x) = 1, matching algoim's convention.
        """
        # Use a few extra digits internally to guard against rounding
        extra = 5
        with mp.workdps(self._dps + extra):
            X, W = mp.gauss_quadrature(p, 'legendre01')

        # Sort by node position (ascending), keep node/weight paired
        paired = sorted(zip(X, W), key=lambda xw: xw[0])
        nodes = [mpf(x) for x, _ in paired]
        weights = [mpf(w) for _, w in paired]
        return nodes, weights

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (f"GaussQuad(dps={self._dps}, p_max={self._p_max}, "
                f"cache={self._use_cache}, cached_rules={len(self._cache)})")


class Interval:
    """
    Interval estimate for a scalar function f over a hyperrectangle.

    Parameters
    ----------
    alpha : (B,) or scalar  – f at the centre
    beta  : (B, N)          – ∇f at the centre
    eps   : (B,) or scalar  – remainder bound
    delta : (N,)            – half-widths of the hyperrectangle
    """

    def __init__(
            self,
            alpha: Tensor,
            beta: Tensor,
            eps: Tensor,
            delta: Tensor,
    ):
        self.alpha = alpha  # (B,)
        self.beta = beta  # (B, N)
        self.eps = eps  # (B,)
        self.delta = delta  # (N,)

    # ------------------------------------------------------------------
    # Factory: build from any callable f and a batch of centre points
    # ------------------------------------------------------------------

    @classmethod
    def from_fn(
            cls,
            fn: Callable[[Tensor], Tensor],
            centres: Tensor,
            delta: Tensor,
            eps_order: int = 2,
    ) -> "Interval":
        """
        Evaluate f over hyperrectangles centred at *centres*.

        Parameters
        ----------
        fn       : callable (B, N) → (B,) or (B, 1)
                   Any torch function, e.g. your geometry's .sdf()
        centres  : (B, N)  centre points of the hyperrectangles
        delta    : (N,)    half-widths (same for every cell)
        eps_order: 1 → eps = 0  (linear, fast, less tight)
                   2 → eps estimated from Hessian diagonal (default)

        Returns
        -------
        Interval with batch size B
        """
        delta = torch.as_tensor(delta, dtype=centres.dtype, device=centres.device)
        B, N = centres.shape

        # ---- alpha: forward pass ----
        x = centres.clone().requires_grad_(True)
        out = fn(x)  # (B,) or (B,1)
        out = out.view(B)  # → (B,)
        alpha = out.detach()

        # ---- beta: Jacobian row by row via grad ----
        # sum trick: grad of sum_b out_b w.r.t. x gives (∂out_b/∂x_b) for each b
        # because outputs are independent across batch dimension.
        beta = torch.zeros(B, N, dtype=centres.dtype, device=centres.device)
        for b in range(B):
            if x.grad is not None:
                x.grad.zero_()
            out[b].backward(retain_graph=(b < B - 1))
            beta[b] = x.grad[b].detach()

        # ---- eps: second-order remainder via Hessian diagonal ----
        if eps_order >= 2:
            eps = cls._hessian_eps(fn, centres, delta)
        else:
            eps = torch.zeros(B, dtype=centres.dtype, device=centres.device)

        # cls._assert_local_smoothness(fn, centres, delta, alpha, beta, eps)
        return cls(alpha, beta, eps, delta)

    @classmethod
    def from_fn_vmap(
            cls,
            fn: Callable[[Tensor], Tensor],
            centres: Tensor,
            delta: Tensor,
            eps_order: int = 2,
    ) -> "Interval":
        """
        Same as from_fn but uses torch.func (functorch) for vectorised
        Jacobian / Hessian — much faster for large batches.

        Requires PyTorch >= 2.0.
        """
        from torch.func import jacrev, hessian, vmap

        delta = torch.as_tensor(delta, dtype=centres.dtype, device=centres.device)
        B, N = centres.shape

        # scalar version of fn for a single point (N,) → scalar
        def fn_single(x):  # x: (N,)
            return fn(x.unsqueeze(0)).squeeze()

        # ---- alpha ----
        alpha = vmap(fn_single)(centres).detach()  # (B,)

        # ---- beta: gradient for each point ----
        grad_fn = jacrev(fn_single)
        beta = vmap(grad_fn)(centres).detach()  # (B, N)

        # ---- eps: Hessian diagonal ----
        if eps_order >= 2:
            hess_fn = hessian(fn_single)

            def hess_eps_single(x):
                H = hess_fn(x)  # (N, N)
                diag = H.diagonal().abs()  # (N,)
                return 0.5 * (diag * delta.pow(2)).sum()

            eps = vmap(hess_eps_single)(centres).detach()  # (B,)
        else:
            eps = torch.zeros(B, dtype=centres.dtype, device=centres.device)

        # cls._assert_local_smoothness(fn, centres, delta, alpha, beta, eps)
        return cls(alpha, beta, eps, delta)

    # ------------------------------------------------------------------
    # Internal: Hessian-diagonal eps estimate
    # ------------------------------------------------------------------

    @staticmethod
    def _hessian_eps(
            fn: Callable[[Tensor], Tensor],
            centres: Tensor,
            delta: Tensor,
    ) -> Tensor:
        """
        eps_b = 0.5 * Σ_i |H_ii(x_b)| * delta_i^2

        Uses forward-over-backward (double autograd) to get diagonal of H.
        """
        B, N = centres.shape
        eps = torch.zeros(B, dtype=centres.dtype, device=centres.device)

        for i in range(N):
            x = centres.clone().requires_grad_(True)
            out = fn(x).view(B)

            grad_i = torch.autograd.grad(
                out.sum(), x, create_graph=True
            )[0][:, i]  # (B,)

            hess_ii = torch.autograd.grad(
                grad_i.sum(), x, retain_graph=False
            )[0][:, i].detach()  # (B,)

            eps += 0.5 * hess_ii.abs() * delta[i].pow(2)

        return eps

    @staticmethod
    def _assert_local_smoothness(
            fn: Callable[[Tensor], Tensor],
            centres: Tensor,
            delta: Tensor,
            alpha: Tensor,
            beta: Tensor,
            eps: Tensor,
    ) -> None:
        """
        Reject non-smooth SDFs only near the zero level set by checking
        local Taylor consistency on cell corners.
        """
        B, N = centres.shape
        K = 1 << N

        signs = torch.empty(K, N, dtype=centres.dtype, device=centres.device)
        for mask in range(K):
            for d in range(N):
                signs[mask, d] = -1.0 if ((mask >> d) & 1) == 0 else 1.0

        corners = centres.unsqueeze(1) + signs.unsqueeze(0) * delta.view(1, 1, N)
        flat = corners.reshape(-1, N)
        try:
            vals = fn(flat).reshape(B, K)
        except Exception:
            vals = []
            for i in range(B * K):
                vals.append(fn(flat[i:i + 1]).reshape(-1)[0])
            vals = torch.stack(vals, dim=0).reshape(B, K)
        dev_est = eps + (beta.abs() * delta).sum(-1)
        # Only care about cells potentially interacting with phi=0.
        near_interface = alpha.abs() <= (2.0 * dev_est + 32.0 * torch.finfo(alpha.dtype).eps)
        if not near_interface.any():
            return

        dev_actual = (vals - alpha.unsqueeze(1)).abs().max(dim=1).values

        tol = 1.5 * dev_est + 64.0 * torch.finfo(alpha.dtype).eps * (1.0 + alpha.abs())
        bad = near_interface & (dev_actual > tol)
        if bad.any():
            raise RuntimeError(
                "Detected non-smooth or non-differentiable SDF near quadrature cell center. "
                "This quadrature implementation requires smooth level-set functions."
            )

    def max_deviation(self) -> Tensor:
        """Upper bound on |f(x) - alpha|.  Shape: (B,)"""
        return self.eps + (self.beta.abs() * self.delta).sum(-1)

    def sign(self) -> Tensor:
        """
        +1 : f > 0 everywhere in the cell
        -1 : f < 0 everywhere
         0 : sign cannot be guaranteed (surface may cross)
        Shape: (B,) int
        """
        tol = (1.0 - 10.0 * torch.finfo(self.alpha.dtype).eps) * self.max_deviation()
        s = torch.zeros(len(self.alpha), dtype=torch.int32, device=self.alpha.device)
        s[self.alpha > tol] = 1
        s[self.alpha < -tol] = -1
        return s

    def uniform_sign(self) -> Tensor:
        """Boolean (B,): True where f has guaranteed uniform sign."""
        return self.sign() != 0

    def bounds(self):
        """Returns (lower, upper) tensors of shape (B,)."""
        d = self.max_deviation()
        return self.alpha - d, self.alpha + d

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        lo, hi = self.bounds()
        return (f"Interval(B={len(self.alpha)}, N={self.beta.shape[1]},\n"
                f"  alpha ={self.alpha},\n"
                f"  beta  ={self.beta},\n"
                f"  eps   ={self.eps},\n"
                f"  bounds=[{lo}, {hi}])")


# ──────────────────────────────────────────────────────────────────────
#  Helper types
# ──────────────────────────────────────────────────────────────────────

@dataclass
class QuadratureNode:
    """Single quadrature node: position x ∈ ℝᴺ and scalar weight w."""
    x: Tensor  # (N,)
    w: float


@dataclass
class QuadratureRule:
    """
    Collection of weighted quadrature nodes.

    Mirrors algoim::QuadratureRule<N>.
    """
    nodes: List[QuadratureNode] = field(default_factory=list)

    # -- used by ImplicitIntegral as the "functor F" --
    def eval_integrand(self, x: Tensor, w: float) -> None:
        """Record a new quadrature node (callback from ImplicitIntegral)."""
        self.nodes.append(QuadratureNode(x.clone().detach(), float(w)))

    # -- user-facing helpers --
    def positions(self) -> Tensor:
        """(K, N) tensor of node positions."""
        if not self.nodes:
            return torch.empty(0)
        return torch.stack([nd.x for nd in self.nodes])

    def weights(self) -> Tensor:
        """(K,) tensor of weights."""
        if not self.nodes:
            return torch.empty(0)
        return torch.tensor([nd.w for nd in self.nodes],
                            dtype=self.nodes[0].x.dtype)

    def integrate(self, f: Callable[[Tensor], Tensor]) -> Tensor:
        """
        Evaluate ∫ f(x) dΩ  using the stored quadrature rule.

        Parameters
        ----------
        f : callable  (N,) → scalar  or  (K, N) → (K,)

        Returns
        -------
        scalar Tensor
        """
        if not self.nodes:
            return torch.tensor(0.0)
        pts = self.positions()  # (K, N)
        wts = self.weights()  # (K,)
        vals = f(pts)  # (K,) or scalar
        return (vals * wts).sum()

    def sum_weights(self) -> float:
        """Sum of weights ≈ measure of the integration domain."""
        return sum(nd.w for nd in self.nodes)

    def __len__(self):
        return len(self.nodes)

    def __repr__(self):
        return f"QuadratureRule(n_nodes={len(self.nodes)}, sum_w={self.sum_weights():.6e})"


@dataclass
class _HyperRectangle:
    """
    Axis-aligned hyperrectangle [xmin, xmax] ⊂ ℝᴺ.

    Mirrors algoim::HyperRectangle<real,N>.
    """
    xmin: Tensor  # (N,)
    xmax: Tensor  # (N,)

    def extent(self, dim: int) -> float:
        return float(self.xmax[dim] - self.xmin[dim])

    def midpoint_dim(self, dim: int) -> float:
        return float((self.xmin[dim] + self.xmax[dim]) * 0.5)

    def midpoint(self) -> Tensor:
        return (self.xmin + self.xmax) * 0.5

    def side(self, s: int, dim: int) -> float:
        """s=0 → xmin[dim],  s=1 → xmax[dim]."""
        return float(self.xmin[dim] if s == 0 else self.xmax[dim])

    def split(self, dim: int) -> Tuple["_HyperRectangle", "_HyperRectangle"]:
        mid = self.midpoint_dim(dim)
        lo = _HyperRectangle(self.xmin.clone(), self.xmax.clone())
        hi = _HyperRectangle(self.xmin.clone(), self.xmax.clone())
        lo.xmax[dim] = mid
        hi.xmin[dim] = mid
        return lo, hi

    @property
    def N(self) -> int:
        return len(self.xmin)


def _as_hyperrectangle(
        xrange: Union["_HyperRectangle", Sequence[float], Tensor],
) -> _HyperRectangle:
    """
    Convert supported bounding-box inputs to ``_HyperRectangle``.

    Supported inputs:
    - ``_HyperRectangle`` (returned as-is)
    - flat sequence/tensor of length ``2N``:
      ``[x_min, x_max, y_min, y_max, ...]``.
    """
    if isinstance(xrange, _HyperRectangle):
        return xrange

    if isinstance(xrange, torch.Tensor):
        flat = xrange.reshape(-1)
        if flat.numel() < 2 or flat.numel() % 2 != 0:
            raise ValueError(
                "Bounding box tensor must have even length 2N, "
                f"got shape {tuple(xrange.shape)}."
            )
        xmin = flat[0::2].clone()
        xmax = flat[1::2].clone()
        return _HyperRectangle(xmin=xmin, xmax=xmax)

    if isinstance(xrange, Sequence):
        vals = list(xrange)
        if len(vals) < 2 or len(vals) % 2 != 0:
            raise ValueError(
                "Bounding box sequence must have even length 2N, "
                f"got length {len(vals)}."
            )
        flat = torch.tensor(vals, dtype=torch.float64)
        xmin = flat[0::2].clone()
        xmax = flat[1::2].clone()
        return _HyperRectangle(xmin=xmin, xmax=xmax)

    raise TypeError(
        "xrange must be _HyperRectangle, Tensor, or sequence of length 2N "
        "formatted as [x_min, x_max, y_min, y_max, ...]."
    )


# ──────────────────────────────────────────────────────────────────────
#  PsiCode  –  encodes side / sign information for restricted level sets
# ──────────────────────────────────────────────────────────────────────

class _PsiCode:
    """
    Mirrors algoim::PsiCode<N>.

    Stores which *side* of the hyperrectangle a restricted function lives on
    for each frozen dimension, plus a sign constraint (-1, 0, +1).
    """
    __slots__ = ("sides", "sgn")

    def __init__(self, sides: List[int], sgn: int):
        self.sides = list(sides)  # length N, entry 0/1 per dim
        self.sgn = sgn  # -1, 0, or +1

    @classmethod
    def from_zero(cls, N: int, sgn: int) -> "_PsiCode":
        return cls([0] * N, sgn)

    def restrict(self, dim: int, side: int, sgn: int) -> "_PsiCode":
        """Return a new PsiCode with dimension *dim* pinned to *side*."""
        new_sides = list(self.sides)
        new_sides[dim] = side
        return _PsiCode(new_sides, sgn)


# ──────────────────────────────────────────────────────────────────────
#  Newton–bisection root finder (1-D, scalar)
# ──────────────────────────────────────────────────────────────────────

def _newton_bisection(
        f_val: Callable[[float], float],
        f_grad: Callable[[float], float],
        x0: float,
        x1: float,
        tol: float = 1e-14,
        maxsteps: int = 1024,
) -> Optional[float]:
    """
    Find a root of a monotone scalar function on [x0, x1] using
    Newton's method safeguarded by bisection.

    Returns the root or None if the interval does not bracket a sign change.
    """
    f0, f1 = f_val(x0), f_val(x1)
    if (f0 > 0 and f1 > 0) or (f0 < 0 and f1 < 0):
        return None
    if f0 == 0.0:
        return x0
    if f1 == 0.0:
        return x1
    # ensure x0 -> negative, x1 -> positive
    if f1 < 0:
        x0, x1 = x1, x0

    x = (x0 + x1) * 0.5
    fx = f_val(x)
    fpx = f_grad(x)
    dx = x1 - x0
    for _ in range(maxsteps):
        # try Newton
        if (fpx * (x - x0) - fx) * (fpx * (x - x1) - fx) < 0.0 and abs(fx) < abs(dx * fpx) * 0.5:
            dx = -fx / fpx
            x_old = x
            x += dx
            if x_old == x:
                return x
        else:
            dx = (x1 - x0) * 0.5
            x = x0 + dx
            if x == x0:
                return x
        if abs(dx) < tol:
            return x
        fx = f_val(x)
        fpx = f_grad(x)
        if fx == 0.0:
            return x
        if fx < 0:
            x0 = x
        else:
            x1 = x
    return (x0 + x1) * 0.5


# ──────────────────────────────────────────────────────────────────────
#  Multi-dimensional root finding (restricted to one axis)
# ──────────────────────────────────────────────────────────────────────

def _root_find_1d(
        phi: "_ImplicitFunction",
        x: Tensor,
        dim: int,
        xmin: float,
        xmax: float,
        is_monotone: bool = False,
        level: int = 0,
) -> List[float]:
    """
    Find all roots of phi along axis *dim* in [xmin, xmax], with all other
    components of x held fixed.

    Mirrors algoim::detail::rootFind.
    """
    N = x.shape[0]

    def f_val(t):
        xc = x.clone()
        xc[dim] = t
        return float(phi(xc))

    def f_grad(t):
        xc = x.clone()
        xc[dim] = t
        return float(phi.grad(xc)[dim])

    if is_monotone or level >= MAX_ROOTFIND_LEVEL:
        tol = 1e2 * torch.finfo(x.dtype).eps * abs(xmax - xmin)
        r = _newton_bisection(f_val, f_grad, xmin, xmax, tol=tol)
        return [r] if r is not None else []

    # Use interval arithmetic to check if there's a root and if monotone
    centre = (xmin + xmax) * 0.5
    half = (xmax - xmin) * 0.5
    delta = torch.zeros(N, dtype=x.dtype)
    delta[dim] = half
    centres_t = x.clone().unsqueeze(0)
    centres_t[0, dim] = centre

    iv = Interval.from_fn(
        lambda p: phi(p.squeeze(0)).unsqueeze(0),
        centres_t, delta, eps_order=2,
    )
    if iv.uniform_sign().item():
        # Safeguard for non-smooth-at-center but root-existing intervals.
        s0 = math.copysign(1.0, f_val(xmin)) if f_val(xmin) != 0.0 else 0.0
        s1 = math.copysign(1.0, f_val(xmax)) if f_val(xmax) != 0.0 else 0.0
        sm = math.copysign(1.0, f_val(centre)) if f_val(centre) != 0.0 else 0.0
        if s0 == s1 == sm and s0 != 0.0:
            return []

    # check gradient monotonicity
    iv_grad = Interval.from_fn(
        lambda p: phi.grad(p.squeeze(0))[dim].unsqueeze(0),
        centres_t, delta, eps_order=2,
    )
    if iv_grad.uniform_sign().item():
        tol = 1e2 * torch.finfo(x.dtype).eps * abs(xmax - xmin)
        r = _newton_bisection(f_val, f_grad, xmin, xmax, tol=tol)
        return [r] if r is not None else []

    # inconclusive → bisect
    xmid = (xmin + xmax) * 0.5
    roots_lo = _root_find_1d(phi, x, dim, xmin, xmid, False, level + 1)
    roots_hi = _root_find_1d(phi, x, dim, xmid, xmax, False, level + 1)
    return roots_lo + roots_hi


# ──────────────────────────────────────────────────────────────────────
#  Determine signs for restricted level set functions
# ──────────────────────────────────────────────────────────────────────

def _determine_signs(positive_above: bool, sign: int, surface: bool):
    """Return (bottom_sign, top_sign).  Mirrors algoim::detail::determineSigns."""
    if surface:
        return (-1 if positive_above else 1,
                1 if positive_above else -1)
    else:
        if sign == 1:
            return (0 if positive_above else 1,
                    1 if positive_above else 0)
        elif sign == -1:
            return (-1 if positive_above else 0,
                    0 if positive_above else -1)
        else:
            return 0, 0


def _assert_zero_levelset_smooth(
        phi: "_ImplicitFunction",
        xrange: "_HyperRectangle",
        samples_per_dim: int = 7,
) -> None:
    """
    Coarse pre-check for smoothness on/near phi=0.
    Rejects obvious kinks/corners by testing gradient jumps across small offsets.
    """
    N = xrange.N
    dtype = xrange.xmin.dtype

    coords_per_dim = []
    for d in range(N):
        xmin = float(xrange.xmin[d])
        xmax = float(xrange.xmax[d])
        coords_per_dim.append(torch.linspace(xmin, xmax, samples_per_dim).tolist())

    from itertools import product as cart_product
    points = []
    for tup in cart_product(*coords_per_dim):
        points.append(torch.tensor(tup, dtype=dtype))

    # Near-zero candidates only; points far from interface are irrelevant.
    min_extent = min(xrange.extent(d) for d in range(N))
    val_tol = 0.3 * min_extent
    candidates = []
    for p in points:
        v = float(phi(p))
        if abs(v) <= val_tol:
            candidates.append(p)

    if not candidates:
        return

    for p in candidates:
        for d in range(N):
            h = 1e-4 * max(xrange.extent(d), 1.0)
            pp = p.clone()
            pm = p.clone()
            pp[d] = min(float(xrange.xmax[d]), float(pp[d] + h))
            pm[d] = max(float(xrange.xmin[d]), float(pm[d] - h))
            gp = phi.grad(pp)
            gm = phi.grad(pm)
            jump = float(torch.norm(gp - gm))
            if jump > 0.5:
                raise RuntimeError(
                    "Detected non-smooth or non-differentiable SDF on/near phi=0. "
                    "Current algo assumes a smooth level set."
                )


# ──────────────────────────────────────────────────────────────────────
#  ImplicitFunction protocol
# ──────────────────────────────────────────────────────────────────────

class _ImplicitFunction:
    """
    Base class / protocol for level-set functions used by quadGen.

    Subclass and implement:
        __call__(self, x: Tensor) -> Tensor   # x: (N,) → scalar
        grad(self, x: Tensor) -> Tensor        # x: (N,) → (N,)

    Both must work with plain floats (Tensor scalars).
    """

    def __call__(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    def grad(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class _AutoGradImplicit(_ImplicitFunction):
    """
    Wraps an arbitrary differentiable callable into an ImplicitFunction,
    computing gradients via torch.autograd.

    Parameters
    ----------
    fn : callable  (N,) → scalar Tensor   (must be torch-differentiable)
    """

    def __init__(self, fn: Callable[[Tensor], Tensor]):
        self._fn = fn

    def __call__(self, x: Tensor) -> Tensor:
        return self._fn(x)

    def grad(self, x: Tensor) -> Tensor:
        xr = x if x.requires_grad else x.clone().requires_grad_(True)
        try:
            val = self._fn(xr)
            if not isinstance(val, torch.Tensor):
                raise TypeError("Implicit function must return a torch.Tensor")
            if val.numel() == 1:
                g = torch.autograd.grad(val, xr, create_graph=True)[0]
            else:
                g = torch.autograd.grad(val.reshape(-1).sum(), xr, create_graph=True)[0]
            return g
        except Exception as e:
            raise RuntimeError(
                "Detected non-smooth or non-differentiable SDF in quadrature pipeline. "
                "Current algo assumes smooth level-set functions."
            ) from e


# ──────────────────────────────────────────────────────────────────────
#  Core recursive engine: ImplicitIntegral
# ──────────────────────────────────────────────────────────────────────

class _ImplicitIntegral:
    """
    M-dimensional integral of an N-dimensional function restricted to
    implicitly defined domains.

    Mirrors algoim::ImplicitIntegral<M,N,Phi,F,S>.

    This is the main recursive engine. Users should not instantiate it
    directly; use :func:`quad_gen` instead.
    """

    def __init__(
            self,
            phi: _ImplicitFunction,
            recorder: QuadratureRule,
            free: List[bool],
            psi: List[_PsiCode],
            psi_count: int,
            xrange: _HyperRectangle,
            p: int,
            surface: bool,
            gauss: GaussQuad,
            level: int = 0,
    ):
        self.phi = phi
        self.recorder = recorder
        self.free = list(free)
        self.psi = [_PsiCode(ps.sides[:], ps.sgn) for ps in psi[:psi_count]]
        self.psi_count = psi_count
        self.xrange = xrange
        self.p = p
        self.surface = surface
        self.gauss = gauss
        self.e0 = -1

        N = xrange.N
        M = sum(free)
        dtype = xrange.xmin.dtype

        # ---- M == 1 base case ----
        if M == 1:
            # Match algoim: choose the only free dim as e0, then run the
            # 1-D segment integration constrained by psi/root finding.
            for dim in range(N):
                if self.free[dim]:
                    self.e0 = dim
                    break
            x0 = torch.zeros(N, dtype=dtype)
            self._eval_integrand(x0, 1.0)
            return

        # ---- Prune ----
        if not self._prune(M, N, dtype):
            return  # domain empty

        # no constraints => integrate full box
        if self.psi_count == 0:
            if not surface:
                self._tensor_product_integral(M, N, dtype)
            return

        # ---- Choose height-function direction e0 ----
        self.e0 = self._choose_direction(N, dtype)

        # if not found, subdivide
        if self.e0 == -1:
            if level < MAX_SUBDIV_LEVEL:
                self._subdivide(M, N, phi, recorder, free, self.psi,
                                self.psi_count, xrange, p, surface, gauss, level)
            else:
                self._midpoint_fallback(M, N, phi, recorder, free, xrange, surface)
            return

        # ---- Build restricted psi codes ----
        new_psi: List[_PsiCode] = []
        for i in range(self.psi_count):
            g_alpha, g_sign = self._grad_interval_along(self.e0, self.psi[i], N, dtype)
            direction_ok = (g_sign != 0)

            if not direction_ok:
                if level < MAX_SUBDIV_LEVEL:
                    self._subdivide(M, N, phi, recorder, free, self.psi,
                                    self.psi_count, xrange, p, surface, gauss, level)
                    return
                else:
                    self._midpoint_fallback(M, N, phi, recorder, free, xrange, surface)
                    return

            positive_above = (g_alpha > 0)
            bot_sign, top_sign = _determine_signs(positive_above, self.psi[i].sgn, surface)
            new_psi.append(self.psi[i].restrict(self.e0, 0, bot_sign))
            new_psi.append(self.psi[i].restrict(self.e0, 1, top_sign))

        # ---- Dimension reduction ----
        new_free = list(free)
        new_free[self.e0] = False

        # Wrap low-dim integrand to trigger this level's _eval_integrand
        class _EvalWrapper:
            __slots__ = ("parent",)

            def __init__(self, parent): self.parent = parent

            def eval_integrand(self, x: torch.Tensor, w: float):
                self.parent._eval_integrand(x, w)

        _ImplicitIntegral(
            phi, _EvalWrapper(self), new_free, new_psi, len(new_psi),
            xrange, p, False, gauss, level=level + 1,
        )

    # ────────────── prune ──────────────

    def _prune(self, M, N, dtype) -> bool:
        """Remove psi functions whose interface is provably absent. Returns False if domain is empty."""
        i = 0
        while i < self.psi_count:
            xpt = self.xrange.midpoint()
            delta = torch.zeros(N, dtype=dtype)
            for dim in range(N):
                if self.free[dim]:
                    delta[dim] = self.xrange.extent(dim) * 0.5
                else:
                    xpt[dim] = self.xrange.side(self.psi[i].sides[dim], dim)

            # relies on Interval.max_deviation() already having Lipschitz safety (per your patch)
            iv = Interval.from_fn_vmap(self.phi, xpt.unsqueeze(0), delta, eps_order=2)

            if iv.uniform_sign().item():
                if self._cell_has_mixed_sample_sign(i, N, dtype):
                    i += 1
                    continue
                alpha_val = float(iv.alpha)
                sgn = self.psi[i].sgn
                if (alpha_val >= 0 and sgn >= 0) or (alpha_val <= 0 and sgn <= 0):
                    # consistent → prune
                    self.psi_count -= 1
                    self.psi[i] = self.psi[self.psi_count]
                else:
                    return False
            else:
                i += 1
        return True

    def _cell_has_mixed_sample_sign(self, psi_idx: int, N: int, dtype) -> bool:
        """
        Cheap consistency check: evaluate phi at midpoint and free-dim corners.
        If signs are mixed, this cell cannot be pruned as uniformly signed.
        """
        x0 = self.xrange.midpoint().to(dtype)
        free_dims = []
        for d in range(N):
            if self.free[d]:
                free_dims.append(d)
            else:
                x0[d] = self.xrange.side(self.psi[psi_idx].sides[d], d)

        signs = set()
        s_mid = float(self.phi(x0))
        if s_mid > 0:
            signs.add(1)
        elif s_mid < 0:
            signs.add(-1)

        k = len(free_dims)
        for mask in range(1 << k):
            x = x0.clone()
            for j, d in enumerate(free_dims):
                x[d] = self.xrange.side(1 if ((mask >> j) & 1) else 0, d)
            v = float(self.phi(x))
            if v > 0:
                signs.add(1)
            elif v < 0:
                signs.add(-1)
            if len(signs) >= 2:
                return True
        return False

    # ────────────── tensor product integral ──────────────

    def _tensor_product_integral(self, M, N, dtype):
        """Standard Gauss quadrature over the full M-dim sub-rectangle (free dims only)."""
        free_dims = [d for d in range(N) if self.free[d]]
        assert len(free_dims) == M

        nodes_1d, weights_1d = self.gauss.rule(self.p, as_float=True)

        from itertools import product as cart_product
        indices = [list(range(self.p))] * M

        for idx in cart_product(*indices):
            x = torch.zeros(N, dtype=dtype)
            w = 1.0
            for k_local, dim in enumerate(free_dims):
                x[dim] = self.xrange.xmin[dim] + self.xrange.extent(dim) * nodes_1d[idx[k_local]]
                w *= self.xrange.extent(dim) * weights_1d[idx[k_local]]
            self.recorder.eval_integrand(x, w)

    # ────────────── direction selection ──────────────

    def _choose_direction(self, N, dtype) -> int:
        """Choose height-function direction using monotonicity test on grad interval."""
        best_dim = -1
        best_qty = 0.0
        for dim in range(N):
            if not self.free[dim]:
                continue
            ga, gs = self._grad_interval_along(dim, self.psi[0], N, dtype)
            if gs != 0:
                qty = abs(ga) * self.xrange.extent(dim)
                if qty > best_qty:
                    best_qty = qty
                    best_dim = dim
        return best_dim

    # ────────────── grad interval (FIXED) ──────────────

    def _phi_grad_batch(self, X: torch.Tensor) -> torch.Tensor:
        """
        Evaluate phi.grad on a batch safely.

        Supports grad(x) where x is (N,) -> (N,) OR (B,N) -> (B,N).
        """
        g = self.phi.grad(X)
        if isinstance(g, torch.Tensor):
            if g.dim() == 1:
                # single point -> (1,N)
                return g.unsqueeze(0)
            return g
        raise TypeError("phi.grad must return a torch.Tensor")

    def _grad_interval_along(self, dim: int, psi_code: _PsiCode, N: int, dtype):
        """
        Return (alpha_of_grad_dim, sign_of_grad_interval).

        We avoid Interval.from_fn on grad (which would require 2nd derivatives / autograd graph).
        Instead we bound grad component by sampling phi.grad at all corners of the current cell.
        This is robust for non-smooth phi like ||x||-1.
        """
        x0 = self.xrange.midpoint().to(dtype)
        delta = torch.zeros(N, dtype=dtype)

        for d in range(N):
            if self.free[d]:
                delta[d] = self.xrange.extent(d) * 0.5
            else:
                x0[d] = self.xrange.side(psi_code.sides[d], d)

        # build 2^k corners over free dims only
        free_dims = [d for d in range(N) if self.free[d]]
        k = len(free_dims)

        corners = []
        for mask in range(1 << k):
            x = x0.clone()
            for j, d in enumerate(free_dims):
                s = -1.0 if ((mask >> j) & 1) == 0 else 1.0
                x[d] = x0[d] + s * delta[d]
            corners.append(x)
        X = torch.stack(corners, dim=0)  # (2^k, N)

        G = self._phi_grad_batch(X)  # (2^k, N)
        g = G[:, dim]

        gmin = float(g.min().item())
        gmax = float(g.max().item())
        alpha = 0.5 * (gmin + gmax)

        # monotonic if grad component is bounded away from 0
        tol = 32.0 * float(torch.finfo(dtype).eps) * (abs(gmin) + abs(gmax) + 1.0)
        if gmin > tol:
            sgn = +1
        elif gmax < -tol:
            sgn = -1
        else:
            sgn = 0

        return alpha, sgn

    # ────────────── subdivide ──────────────

    def _subdivide(self, M, N, phi, recorder, free, psi, psi_count,
                   xrange, p, surface, gauss, level):
        """Split along the biggest free extent and recurse."""
        best_dim = -1
        best_ext = 0.0
        for dim in range(N):
            if free[dim]:
                ext = xrange.extent(dim)
                if ext > best_ext:
                    best_ext = ext
                    best_dim = dim
        assert best_dim >= 0
        lo, hi = xrange.split(best_dim)
        _ImplicitIntegral(phi, recorder, free, psi, psi_count,
                          lo, p, surface, gauss, level + 1)
        _ImplicitIntegral(phi, recorder, free, psi, psi_count,
                          hi, p, surface, gauss, level + 1)

    def _midpoint_fallback(self, M, N, phi, recorder, free, xrange, surface):
        """
        Match algoim deep-recursion fallback:
        evaluate sign consistency at box midpoint and, if consistent,
        approximate by midpoint with full free-measure weight (volume only).
        """
        xpt = xrange.midpoint()
        ok = True
        for j in range(self.psi_count):
            for dim in range(N):
                if not free[dim]:
                    xpt[dim] = xrange.side(self.psi[j].sides[dim], dim)
            val = float(phi(xpt))
            if not ((val >= 0 and self.psi[j].sgn >= 0) or
                    (val <= 0 and self.psi[j].sgn <= 0)):
                ok = False
                break

        if ok and not surface:
            measure = 1.0
            for dim in range(N):
                if free[dim]:
                    measure *= xrange.extent(dim)
            recorder.eval_integrand(xpt, measure)

    # ────────────── eval_integrand (called from M-1 level) ──────────────

    def _eval_integrand(self, x: Tensor, w: float):
        """
        Given x valid in all free dims except e0, root-find along e0
        and apply Gauss quadrature on each valid segment.
        """
        N = self.xrange.N
        dtype = self.xrange.xmin.dtype
        e0 = self.e0
        xmin_e0 = float(self.xrange.xmin[e0])
        xmax_e0 = float(self.xrange.xmax[e0])

        if self.surface:
            roots = _root_find_1d(self.phi, x, e0, xmin_e0, xmax_e0, True)
            for r in roots:
                xr = x.clone()
                xr[e0] = r
                g = self.phi.grad(xr)
                g_norm = float(torch.norm(g))
                g_e0 = abs(float(g[e0]))
                if g_e0 > 0:
                    self.recorder.eval_integrand(xr, g_norm / g_e0 * w)
            return

        roots = [xmin_e0]
        m_current = sum(self.free)
        monotone_flag = (m_current > 1)
        for i in range(self.psi_count):
            xr = x.clone()
            for dim in range(N):
                if not self.free[dim]:
                    xr[dim] = self.xrange.side(self.psi[i].sides[dim], dim)
            found = _root_find_1d(
                self.phi, xr, e0, xmin_e0, xmax_e0, is_monotone=monotone_flag
            )
            roots.extend(found)
        roots.sort()
        roots.append(xmax_e0)

        tol = 10.0 * torch.finfo(dtype).eps * (xmax_e0 - xmin_e0)
        nodes_1d, weights_1d = self.gauss.rule(self.p, as_float=True)

        for i in range(len(roots) - 1):
            if roots[i + 1] - roots[i] < tol:
                continue

            seg_mid = (roots[i] + roots[i + 1]) * 0.5
            ok = True
            for j in range(self.psi_count):
                xc = x.clone()
                xc[e0] = seg_mid
                for dim in range(N):
                    if not self.free[dim]:
                        xc[dim] = self.xrange.side(self.psi[j].sides[dim], dim)
                val = float(self.phi(xc))
                if not ((val > 0 and self.psi[j].sgn >= 0) or
                        (val < 0 and self.psi[j].sgn <= 0) or
                        self.psi[j].sgn == 0):
                    ok = False
                    break
            if not ok:
                continue

            seg_len = roots[i + 1] - roots[i]
            for j in range(self.p):
                xq = x.clone()
                xq[e0] = roots[i] + seg_len * nodes_1d[j]
                gw = seg_len * weights_1d[j]
                self.recorder.eval_integrand(xq, w * gw)


# ──────────────────────────────────────────────────────────────────────
#  Public API: quad_gen
# ──────────────────────────────────────────────────────────────────────

def quad_gen(
        phi: Union[_ImplicitFunction, Callable[[Tensor], Tensor]],
        xrange: Union[_HyperRectangle, Sequence[float], Tensor],
        dim: int = -1,
        side: int = 0,
        qo: int = 4,
        gauss: Optional[GaussQuad] = None,
) -> QuadratureRule:
    """
    Generate a high-order quadrature rule for an implicitly defined domain.

    This is the main user-facing function, mirroring ``algoim::quadGen``.

    Parameters
    ----------
    phi : ImplicitFunction or callable (N,) → scalar
        Level-set function. If a plain callable is given it is wrapped with
        :class:`AutoGradImplicit` (requires the function to be torch-differentiable).
    xrange : _HyperRectangle or sequence/tensor of length 2N
        Bounding box in ℝᴺ. Flat format:
        ``[x_min, x_max, y_min, y_max, ...]``.
    dim : int
        * ``dim < 0``  → volume quadrature on {phi < 0} ∩ xrange
        * ``dim == N``  → surface quadrature on {phi = 0} ∩ xrange
        * ``0 <= dim < N`` → face quadrature on {phi < 0} restricted to
          one face of the hyperrectangle (specified by *side*).
    side : int (0 or 1)
        Only used when 0 <= dim < N.  0 = "left" face, 1 = "right" face.
    qo : int
        Order of the underlying 1-D Gaussian quadrature (number of points).
    gauss : GaussQuad, optional
        Pre-constructed GaussQuad instance. One is created if not provided.

    Returns
    -------
    QuadratureRule
        Collection of (position, weight) nodes.

    """
    xrange = _as_hyperrectangle(xrange)
    N = xrange.N

    # Wrap plain callable
    if not isinstance(phi, _ImplicitFunction):
        phi = _AutoGradImplicit(phi)

    # _assert_zero_levelset_smooth(phi, xrange)

    if gauss is None:
        gauss = GaussQuad(dps=20, p_max=max(qo, 10))

    q = QuadratureRule()
    free = [True] * N

    if 0 <= dim < N:
        # Face integral
        assert side in (0, 1)
        sides_init = [0] * N
        sides_init[dim] = side
        psi0 = _PsiCode(sides_init, -1)
        free[dim] = False
        _ImplicitIntegral(phi, q, free, [psi0], 1, xrange, qo, False, gauss)
        # fill in the frozen coordinate
        for nd in q.nodes:
            nd.x[dim] = xrange.side(side, dim)

    elif dim == N:
        # Surface integral
        psi0 = _PsiCode.from_zero(N, -1)
        _ImplicitIntegral(phi, q, free, [psi0], 1, xrange, qo, True, gauss)

    else:
        # Volume integral
        psi0 = _PsiCode.from_zero(N, -1)
        _ImplicitIntegral(phi, q, free, [psi0], 1, xrange, qo, False, gauss)

    return q


# ──────────────────────────────────────────────────────────────────────
#  Geometry-aware boolean volume quadrature (detailed path)
# ──────────────────────────────────────────────────────────────────────

def _geometry_types():
    """Lazy import geometry classes to avoid heavy import/cycles."""
    from .geometry import (
        GeometryBase,
        UnionGeometry,
        IntersectionGeometry,
        ComplementGeometry,
    )
    return GeometryBase, UnionGeometry, IntersectionGeometry, ComplementGeometry


def _is_boolean_composite_geometry(geom: Any) -> bool:
    """Return True if geometry tree contains boolean composite operators."""
    _, UnionGeometry, IntersectionGeometry, ComplementGeometry = _geometry_types()
    if isinstance(geom, UnionGeometry) or isinstance(geom, IntersectionGeometry) or isinstance(geom,
                                                                                               ComplementGeometry):
        return True
    if hasattr(geom, "geomA") and hasattr(geom, "geomB"):
        return _is_boolean_composite_geometry(geom.geomA) or _is_boolean_composite_geometry(geom.geomB)
    if hasattr(geom, "geom"):
        return _is_boolean_composite_geometry(geom.geom)
    return False


def _collect_geometry_atoms(geom) -> Dict[int, Any]:
    """Collect primitive geometry leaves from a composite geometry tree."""
    _, UnionGeometry, IntersectionGeometry, ComplementGeometry = _geometry_types()
    out: Dict[int, Any] = {}

    def rec(g):
        if isinstance(g, UnionGeometry) or isinstance(g, IntersectionGeometry):
            rec(g.geomA)
            rec(g.geomB)
            return
        if isinstance(g, ComplementGeometry):
            rec(g.geom)
            return
        out[id(g)] = g

    rec(geom)
    return out


def _geometry_sdf_callable(geom) -> Callable[[Tensor], Tensor]:
    """Adapt GeometryBase.sdf to accept both (N,) and (B,N) tensors."""

    def wrapped(x: Tensor) -> Tensor:
        if x.dim() == 1:
            return geom.sdf(x.view(1, -1)).reshape(-1)[0]
        return geom.sdf(x).reshape(-1)

    return wrapped


def _eval_geometry_inside(geom, inside_atom: Callable[[int], bool]) -> bool:
    """Evaluate inside/outside with short-circuit boolean logic."""
    _, UnionGeometry, IntersectionGeometry, ComplementGeometry = _geometry_types()
    if isinstance(geom, UnionGeometry):
        return _eval_geometry_inside(geom.geomA, inside_atom) or _eval_geometry_inside(geom.geomB, inside_atom)
    if isinstance(geom, IntersectionGeometry):
        return _eval_geometry_inside(geom.geomA, inside_atom) and _eval_geometry_inside(geom.geomB, inside_atom)
    if isinstance(geom, ComplementGeometry):
        return not _eval_geometry_inside(geom.geom, inside_atom)
    return inside_atom(id(geom))


def _unique_sorted(values: List[float], tol: float) -> List[float]:
    if not values:
        return []
    vals = sorted(values)
    out = [vals[0]]
    for v in vals[1:]:
        if abs(v - out[-1]) > tol:
            out.append(v)
    return out


def _bisection_root_scalar(
        f: Callable[[float], float],
        a: float,
        b: float,
        fa: float,
        fb: float,
        tol: float,
        max_iter: int = 64,
) -> Optional[float]:
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa * fb > 0.0:
        return None
    lo, hi = a, b
    flo = fa
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if fm == 0.0 or abs(hi - lo) <= tol:
            return mid
        if flo * fm <= 0.0:
            hi = mid
        else:
            lo, flo = mid, fm
    return 0.5 * (lo + hi)


def _scan_roots_on_line(
        f: Callable[[float], float],
        a: float,
        b: float,
        num_samples: int,
        value_tol: float,
        merge_tol: float,
        bisect_tol: float,
) -> List[float]:
    if b <= a:
        return []
    num_samples = max(3, int(num_samples))
    ts = [a + (b - a) * (i / (num_samples - 1)) for i in range(num_samples)]
    fs = [f(t) for t in ts]
    roots: List[float] = []
    for i in range(len(ts) - 1):
        t0, t1 = ts[i], ts[i + 1]
        f0, f1 = fs[i], fs[i + 1]
        if abs(f0) <= value_tol:
            roots.append(t0)
        if abs(f1) <= value_tol:
            roots.append(t1)
        if f0 * f1 < 0.0:
            r = _bisection_root_scalar(f, t0, t1, f0, f1, tol=bisect_tol)
            if r is not None:
                roots.append(r)
    return _unique_sorted(roots, tol=merge_tol)


def _line_hits_atom_bbox(
        bbox: Sequence[float],
        x_base: Tensor,
        free_dims: Sequence[int],
        sweep_dim: int,
        line_a: float,
        line_b: float,
        tol: float,
) -> bool:
    for d in free_dims:
        lo = float(bbox[2 * d])
        hi = float(bbox[2 * d + 1])
        xv = float(x_base[d])
        if xv < lo - tol or xv > hi + tol:
            return False
    lo_s = float(bbox[2 * sweep_dim])
    hi_s = float(bbox[2 * sweep_dim + 1])
    return not (line_b < lo_s - tol or line_a > hi_s + tol)


def quad_gen_geometry_boolean(
        geometry,
        xrange: Optional[Union[_HyperRectangle, Sequence[float], Tensor]] = None,
        qo: int = 4,
        sweep_dim: Optional[int] = None,
        gauss: Optional[GaussQuad] = None,
        value_tol: float = 1e-12,
        adapt_base: bool = True,
        base_tol: float = 1e-6,
        max_base_subdiv: int = 6,
        root_scan_samples: int = 33,
) -> QuadratureRule:
    """
    Detailed boolean volume quadrature directly from GeometryBase composites.
    """
    GeometryBase, _, _, _ = _geometry_types()
    if not isinstance(geometry, GeometryBase):
        raise TypeError("geometry must be an instance of GeometryBase.")

    if xrange is None:
        box = _as_hyperrectangle(geometry.get_bounding_box())
    else:
        box = _as_hyperrectangle(xrange)
    N = box.N
    if N < 1:
        raise ValueError("Dimension must be >= 1.")

    atoms = _collect_geometry_atoms(geometry)
    if not atoms:
        return QuadratureRule()

    atom_impl: Dict[int, _ImplicitFunction] = {}
    atom_bbox: Dict[int, Sequence[float]] = {}
    for pid, g in atoms.items():
        atom_impl[pid] = _AutoGradImplicit(_geometry_sdf_callable(g))
        atom_bbox[pid] = g.get_bounding_box()

    if gauss is None:
        gauss = GaussQuad(dps=20, p_max=max(qo, 20))
    nodes_1d, weights_1d = gauss.rule(qo, as_float=True)

    if sweep_dim is None:
        sweep_dim = max(range(N), key=lambda d: box.extent(d))
    if sweep_dim < 0 or sweep_dim >= N:
        raise ValueError(f"sweep_dim must be in [0, {N - 1}], got {sweep_dim}")

    free_dims = [d for d in range(N) if d != sweep_dim]
    qrule = QuadratureRule()
    dtype = box.xmin.dtype
    eps_len = 10.0 * float(torch.finfo(dtype).eps) * max(1.0, box.extent(sweep_dim))
    root_scan_samples = max(3, int(root_scan_samples))

    def _roots_scan_fallback(phi_i: _ImplicitFunction, x_base: Tensor, a: float, b: float) -> List[float]:
        def fval(t: float) -> float:
            xt = x_base.clone()
            xt[sweep_dim] = t
            return float(phi_i(xt))

        return _scan_roots_on_line(
            f=fval,
            a=a,
            b=b,
            num_samples=root_scan_samples,
            value_tol=value_tol,
            merge_tol=eps_len,
            bisect_tol=eps_len,
        )

    def _line_segments(x_base: Tensor) -> List[Tuple[float, float]]:
        roots: List[float] = []
        a = float(box.xmin[sweep_dim])
        b = float(box.xmax[sweep_dim])

        candidate_pids = [
            pid for pid in atoms.keys()
            if _line_hits_atom_bbox(atom_bbox[pid], x_base, free_dims, sweep_dim, a, b, tol=1e-12)
        ]

        for pid in candidate_pids:
            phi_i = atom_impl[pid]
            found = _root_find_1d(phi_i, x_base, sweep_dim, a, b, is_monotone=False)
            roots.extend([r for r in found if r is not None])
            roots.extend(_roots_scan_fallback(phi_i, x_base, a, b))

        split = _unique_sorted([a, *roots, b], tol=eps_len)
        if len(split) < 2:
            return []

        segs: List[Tuple[float, float]] = []
        for k in range(len(split) - 1):
            lo = split[k]
            hi = split[k + 1]
            if (hi - lo) <= eps_len:
                continue
            mid = 0.5 * (lo + hi)
            x_mid = x_base.clone()
            x_mid[sweep_dim] = mid

            sign_cache: Dict[int, bool] = {}

            def inside_atom(pid: int) -> bool:
                if pid not in sign_cache:
                    sign_cache[pid] = (float(atom_impl[pid](x_mid)) < -value_tol)
                return sign_cache[pid]

            if _eval_geometry_inside(geometry, inside_atom):
                segs.append((lo, hi))
        return segs

    def _accumulate_base_cell(base_xmin: Tensor, base_xmax: Tensor, record_nodes: bool) -> float:
        cell_indices = [list(range(qo))] * len(free_dims)
        total = 0.0
        for idx in cart_product(*cell_indices) if cell_indices else [()]:
            x_base = torch.zeros(N, dtype=dtype, device=box.xmin.device)
            w_base = 1.0
            for j, d in enumerate(free_dims):
                lo = float(base_xmin[j])
                hi = float(base_xmax[j])
                x_base[d] = lo + (hi - lo) * nodes_1d[idx[j]]
                w_base *= (hi - lo) * weights_1d[idx[j]]

            segs = _line_segments(x_base)
            for lo_t, hi_t in segs:
                seg_len = hi_t - lo_t
                if seg_len <= eps_len:
                    continue
                for j in range(qo):
                    xq = x_base.clone()
                    xq[sweep_dim] = lo_t + seg_len * nodes_1d[j]
                    wq = w_base * seg_len * weights_1d[j]
                    total += wq
                    if record_nodes:
                        qrule.eval_integrand(xq, wq)
        return total

    def _split_base_cell(base_xmin: Tensor, base_xmax: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        ext = base_xmax - base_xmin
        split_j = int(torch.argmax(ext).item())
        mid = 0.5 * (base_xmin[split_j] + base_xmax[split_j])
        lo_min = base_xmin.clone()
        lo_max = base_xmax.clone()
        hi_min = base_xmin.clone()
        hi_max = base_xmax.clone()
        lo_max[split_j] = mid
        hi_min[split_j] = mid
        return lo_min, lo_max, hi_min, hi_max

    def _cell_measure(base_xmin: Tensor, base_xmax: Tensor) -> float:
        if len(free_dims) == 0:
            return 1.0
        return float((base_xmax - base_xmin).prod().item())

    def _recurse_base(base_xmin: Tensor, base_xmax: Tensor, level: int) -> None:
        coarse = _accumulate_base_cell(base_xmin, base_xmax, record_nodes=False)
        if (not adapt_base) or level >= max_base_subdiv or len(free_dims) == 0:
            _accumulate_base_cell(base_xmin, base_xmax, record_nodes=True)
            return
        lo_min, lo_max, hi_min, hi_max = _split_base_cell(base_xmin, base_xmax)
        ref_lo = _accumulate_base_cell(lo_min, lo_max, record_nodes=False)
        ref_hi = _accumulate_base_cell(hi_min, hi_max, record_nodes=False)
        refined = ref_lo + ref_hi
        err = abs(refined - coarse)
        tol_local = base_tol * _cell_measure(base_xmin, base_xmax)
        if err <= tol_local:
            _accumulate_base_cell(lo_min, lo_max, record_nodes=True)
            _accumulate_base_cell(hi_min, hi_max, record_nodes=True)
            return
        _recurse_base(lo_min, lo_max, level + 1)
        _recurse_base(hi_min, hi_max, level + 1)

    base_xmin = torch.tensor([float(box.xmin[d]) for d in free_dims], dtype=dtype, device=box.xmin.device)
    base_xmax = torch.tensor([float(box.xmax[d]) for d in free_dims], dtype=dtype, device=box.xmin.device)
    _recurse_base(base_xmin, base_xmax, 0)
    return qrule


# ──────────────────────────────────────────────────────────────────────
#  Convenience: ImplicitQuadrature class wrapping quad_gen
# ──────────────────────────────────────────────────────────────────────

class ImplicitQuadrature:
    """
    High-level interface for quadrature on geometry-defined implicit domains.

    Usage::

        # Preferred unified interface: geometry object
        geom = Circle2D(center=[0.0, 0.0], radius=1.0)
        iq = ImplicitQuadrature(geom, qo=5)

        area  = iq.volume()                       # ≈ π
        perim = iq.surface()                       # ≈ 2π
        val   = iq.integrate_volume(lambda x: x[:, 0]**2 + x[:, 1]**2)
    """

    def __init__(
            self,
            geometry: Any,
            xrange: Optional[Union[_HyperRectangle, Sequence[float], Tensor]] = None,
            qo: int = 4,
            sweep_dim: Optional[int] = None,
            adapt_base: bool = True,
            base_tol: float = 1e-6,
            max_base_subdiv: int = 6,
            root_scan_samples: int = 33,
    ):
        GeometryBase, _, _, _ = _geometry_types()
        if not isinstance(geometry, GeometryBase):
            raise TypeError(
                "ImplicitQuadrature now expects a GeometryBase object as input. "
                "Use a geometry primitive/composite from pyrfm.geometry."
            )

        self.geometry = geometry
        self._use_boolean_detailed = _is_boolean_composite_geometry(self.geometry)

        if xrange is None:
            self.xrange = _as_hyperrectangle(self.geometry.get_bounding_box())
        else:
            self.xrange = _as_hyperrectangle(xrange)
        self.phi_geom = _AutoGradImplicit(_geometry_sdf_callable(self.geometry))

        self.qo = qo
        self.sweep_dim = sweep_dim
        self.adapt_base = adapt_base
        self.base_tol = base_tol
        self.max_base_subdiv = max_base_subdiv
        self.root_scan_samples = root_scan_samples
        self.gauss = GaussQuad(dps=20, p_max=max(qo, 20))
        self._cache = {}

    def volume_rule(self) -> QuadratureRule:
        """Quadrature rule for {phi < 0} ∩ xrange."""
        if "vol" not in self._cache:
            if self._use_boolean_detailed:
                self._cache["vol"] = quad_gen_geometry_boolean(
                    geometry=self.geometry,
                    xrange=self.xrange,
                    qo=self.qo,
                    sweep_dim=self.sweep_dim,
                    gauss=self.gauss,
                    adapt_base=self.adapt_base,
                    base_tol=self.base_tol,
                    max_base_subdiv=self.max_base_subdiv,
                    root_scan_samples=self.root_scan_samples,
                )
            else:
                self._cache["vol"] = quad_gen(self.phi_geom, self.xrange, dim=-1,
                                              qo=self.qo, gauss=self.gauss)
        # print(self._cache["vol"].positions(), self._cache["vol"].weights())
        return self._cache["vol"]

    def surface_rule(self) -> QuadratureRule:
        """Quadrature rule for {phi = 0} ∩ xrange."""
        if "surf" not in self._cache:
            N = self.xrange.N
            self._cache["surf"] = quad_gen(self.phi_geom, self.xrange, dim=N,
                                           qo=self.qo, gauss=self.gauss)
        # print(self._cache["surf"].positions(), self._cache["surf"].weights())
        return self._cache["surf"]

    def face_rule(self, dim: int, side: int) -> QuadratureRule:
        """Quadrature rule on {phi < 0} ∩ face(dim, side)."""
        key = ("face", dim, side)
        if key not in self._cache:
            self._cache[key] = quad_gen(self.phi_geom, self.xrange, dim=dim,
                                        side=side, qo=self.qo, gauss=self.gauss)
        return self._cache[key]

    # ---- shortcuts ----

    def volume(self) -> float:
        """Measure of {phi < 0} ∩ xrange."""
        return self.volume_rule().sum_weights()

    def surface_area(self) -> float:
        """Measure of {phi = 0} ∩ xrange."""
        return self.surface_rule().sum_weights()

    def integrate_volume(self, f: Callable) -> Tensor:
        """∫_{phi<0} f(x) dΩ"""
        return self.volume_rule().integrate(f)

    def integrate_surface(self, f: Callable) -> Tensor:
        """∫_{phi=0} f(x) dS"""
        return self.surface_rule().integrate(f)

    def clear_cache(self):
        self._cache.clear()

# ──────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────
