# -*- coding: utf-8 -*-
"""
Created on 2024/12/15

@author: Yifei Sun
"""
from typing import Any

import torch
from torch import Tensor

from .utils import *


# SDF Reference : https://iquilezles.org/articles/distfunctions2d/ , https://iquilezles.org/articles/distfunctions/

class State(Enum):
    """
    Enum class for the state of a point with respect to a geometry.

    Attributes:
    ----------
    isIn : int
        Represents that the point is inside the geometry.
    isOut : int
        Represents that the point is outside the geometry.
    isOn : int
        Represents that the point is on the boundary of the geometry.
    isUnknown : int
        Represents an undefined or indeterminate state of the point.
    """
    isIn = 0
    isOut = 1
    isOn = 2
    isUnknown = 3


class GeometryBase(ABC):
    """
    Abstract base class for geometric objects.

    Attributes:
    ----------
    dim : int
        The dimension of the geometry.
    intrinsic_dim : int
        The intrinsic dimension of the geometry.
    boundary : list
        The boundary of the geometry.
    """

    def __init__(self, dim: Optional[int] = None, intrinsic_dim: Optional[int] = None, seed: int = 100):
        """
        Initialize the GeometryBase object.

        Args:
        ----
        dim : int, optional
            The dimension of the geometry.
        intrinsic_dim : int, optional
            The intrinsic dimension of the geometry.
        """
        self.dim = dim if dim is not None else 0
        self.dtype = torch.tensor(0.).dtype
        self.device = torch.tensor(0.).device
        self.intrinsic_dim = intrinsic_dim if intrinsic_dim is not None else dim
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(seed)
        self.boundary: List = []

    def __eq__(self, other):
        """
        Check if two geometries are equal.

        Args:
        ----
        other : GeometryBase
            Another geometry object.

        Returns:
        -------
        bool
            True if the geometries are equal, False otherwise.
        """
        if not isinstance(other, self.__class__):
            return False

        if self.dim != other.dim or self.intrinsic_dim != other.intrinsic_dim:
            return False

        if len(self.boundary) != len(other.boundary):
            return False
        else:
            if Counter(self.boundary) != Counter(other.boundary):
                return False

    @abstractmethod
    def sdf(self, p: torch.Tensor):
        """
        Compute the signed distance of a point to the geometry.

        Args:
        ----
        p : torch.Tensor
            A tensor of points.

        Returns:
        -------
        torch.Tensor
            A tensor of signed distances.
        """
        pass

    # def glsl_sdf(self) -> str:
    #     """
    #     Return a GLSL expression (string) that evaluates the signed distance
    #     at a coordinate variable named `p` (float for 1‑D, vec2 for 2‑D,
    #     vec3 for 3‑D) which must be in scope inside the GLSL shader.
    #     The expression must be syntactically valid GLSL and reference only
    #     constants and the variable `p`.
    #     """
    #     pass

    @abstractmethod
    def get_bounding_box(self) -> List[float]:
        """
        Get the bounding box of the geometry.

        Returns:
        -------
        list
            For 2D: [x_min, x_max, y_min, y_max]
            For 3D: [x_min, x_max, y_min, y_max, z_min, z_max]
        """
        pass

    @abstractmethod
    def in_sample(self, num_samples: int, with_boundary: bool = False) -> torch.Tensor:
        """
        Generate samples within the geometry.

        Args:
        ----
        num_samples : int
            The number of samples to generate.
        with_boundary : bool, optional
            Whether to include boundary points in the samples.

        Returns:
        -------
        torch.Tensor
            A tensor of points sampled from the geometry.
        """
        pass

    @abstractmethod
    def on_sample(self, num_samples: int, with_normal: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Generate samples on the boundary of the geometry.

        Args:
        ----
        num_samples : int
            The number of samples to generate.
        with_normal : bool, optional
            Whether to include normal vectors.

        Returns:
        -------
        torch.Tensor or tuple
            A tensor of points sampled from the boundary of the geometry or a tuple of tensors of points and normal vectors.
        """
        pass

    def __and__(self, other: 'GeometryBase') -> 'GeometryBase':
        """
        Compute the intersection of two geometries.

        Args:
        ----
        other : GeometryBase
            Another geometry object.

        Returns:
        -------
        IntersectionGeometry
            The intersection of the two geometries.
        """
        return IntersectionGeometry(self, other)

    def __or__(self, other: 'GeometryBase') -> 'GeometryBase':
        """
        Compute the union of two geometries.

        Args:
        ----
        other : GeometryBase
            Another geometry object.

        Returns:
        -------
        UnionGeometry
            The union of the two geometries.
        """
        return UnionGeometry(self, other)

    def __invert__(self) -> 'GeometryBase':
        """
        Compute the complement of the geometry.

        Returns:
        -------
        ComplementGeometry
            The complement of the geometry.
        """
        return ComplementGeometry(self)

    def __add__(self, other: 'GeometryBase') -> 'GeometryBase':
        if isinstance(other, EmptyGeometry):
            return self
        return UnionGeometry(self, other)

    def __sub__(self, other: 'GeometryBase') -> 'GeometryBase':
        if isinstance(other, EmptyGeometry):
            return self
        return IntersectionGeometry(self, ComplementGeometry(other))

    def __radd__(self, other: 'GeometryBase') -> 'GeometryBase':
        """
        To support sum() function.
        """
        return self.__add__(other)


class EmptyGeometry(GeometryBase):
    def __init__(self):
        super().__init__(dim=0, intrinsic_dim=0)
        self.boundary = []

    def sdf(self, p: torch.Tensor):
        """
        For empty geometry, the signed distance to the geometry is always infinity.
        """
        return torch.full_like(p, float('inf'))

    # # GLSL: empty space has effectively infinite distance
    # def glsl_sdf(self) -> str:
    #     return "1e20"

    """
    A class to represent the empty geometry.
    """

    def get_bounding_box(self) -> List[float]:
        """
        The bounding box for empty geometry is an empty list.
        """
        return []

    def in_sample(self, num_samples: int, with_boundary: bool = False) -> torch.Tensor:
        """
        There are no samples for the empty geometry.
        """
        return torch.empty((num_samples, 0))  # No points can be sampled

    def on_sample(self, num_samples: int, with_normal: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        There are no boundary samples for the empty geometry.
        """
        return torch.empty((num_samples, 0))  # No boundary points

    def __eq__(self, other):
        """
        Empty geometry is equal to another empty geometry.
        """
        return isinstance(other, EmptyGeometry)

    def __add__(self, other: 'GeometryBase') -> 'GeometryBase':
        """
        Union with empty geometry is the other geometry.
        """
        return other

    def __or__(self, other: 'GeometryBase') -> 'GeometryBase':
        """
        Union with empty geometry is the other geometry.
        """
        return other

    def __invert__(self) -> 'GeometryBase':
        """
        The complement of an empty geometry is the entire space.
        """
        return ComplementGeometry(self)


class UnionGeometry(GeometryBase):
    def __init__(self, geomA: GeometryBase, geomB: GeometryBase):
        super().__init__()
        self.geomA = geomA
        self.geomB = geomB
        self.dim = geomA.dim
        self.intrinsic_dim = geomA.intrinsic_dim
        self.boundary = [*geomA.boundary, *geomB.boundary]

    def sdf(self, p: torch.Tensor):
        return torch.min(self.geomA.sdf(p), self.geomB.sdf(p))

    # # GLSL expression for the union: min(dA,dB)
    # def glsl_sdf(self) -> str:
    #     return f"min({self.geomA.glsl_sdf()}, {self.geomB.glsl_sdf()})"

    def get_bounding_box(self):
        boxA = self.geomA.get_bounding_box()
        boxB = self.geomB.get_bounding_box()
        return [min(boxA[i], boxB[i]) if i % 2 == 0 else max(boxA[i], boxB[i]) for i in range(2 * self.dim)]

    def in_sample(self, num_samples: int, with_boundary: bool = False):
        boxA = self.geomA.get_bounding_box()
        boxB = self.geomB.get_bounding_box()

        VA, VB = 1.0, 1.0
        dim = len(boxA) // 2
        for i in range(dim):
            VA *= max(0.0, boxA[2 * i + 1] - boxA[2 * i])
            VB *= max(0.0, boxB[2 * i + 1] - boxB[2 * i])

        # sampling ratio
        r = min(2.0, max(0.5, VA / (VB + 1e-12)))

        # allocate
        NA = max(5, int(num_samples * r / (1 + r)))
        NB = max(5, num_samples - NA)

        # --- 采样 ---
        a = self.geomA.in_sample(NA, with_boundary)
        b = self.geomB.in_sample(NB, with_boundary)
        samples = torch.cat([a, b], dim=0)

        # --- 过滤 union ---
        mask = (self.sdf(samples) <= 0).squeeze() if with_boundary else (self.sdf(samples) < 0).squeeze()
        filtered = samples[mask]

        # fallback：确保不会返回空
        if filtered.shape[0] == 0:
            return samples

        return filtered

    def on_sample(self, num_samples: int, with_normal: bool = False):
        boxA = self.geomA.get_bounding_box()
        boxB = self.geomB.get_bounding_box()

        VA, VB = 1.0, 1.0
        dim = len(boxA) // 2
        for i in range(dim):
            VA *= max(0.0, boxA[2 * i + 1] - boxA[2 * i])
            VB *= max(0.0, boxB[2 * i + 1] - boxB[2 * i])

        # sampling ratio
        r = min(2.0, max(0.5, VA / (VB + 1e-12)))

        # allocate
        NA = max(5, int(num_samples * r / (1 + r)))
        NB = max(5, num_samples - NA)

        # --- 采样 ---
        if with_normal:
            a, an = self.geomA.on_sample(NA, True)
            b, bn = self.geomB.on_sample(NB, True)

            samples = torch.cat([a, b], dim=0)
            normals = torch.cat([an, bn], dim=0)

            mask = torch.isclose(self.sdf(samples), torch.tensor(0., device=samples.device))

            if mask.sum() == 0:
                return samples, normals
            return samples[mask.flatten()], normals[mask.flatten()]

        else:
            a = self.geomA.on_sample(NA, False)
            b = self.geomB.on_sample(NB, False)
            samples = torch.cat([a, b], dim=0)

            mask = torch.isclose(self.sdf(samples), torch.tensor(0., device=samples.device))
            if mask.sum() == 0:
                return samples
            return samples[mask.flatten()]


class IntersectionGeometry(GeometryBase):
    def __init__(self, geomA: GeometryBase, geomB: GeometryBase):
        super().__init__()
        if geomA.dim != geomB.dim:
            raise ValueError("The dimensions of the two geometries must be equal.")
        elif geomA.intrinsic_dim != geomB.intrinsic_dim:
            raise ValueError("The intrinsic dimensions of the two geometries must be equal.")
        self.geomA = geomA
        self.geomB = geomB
        self.dim = geomA.dim
        self.intrinsic_dim = geomA.intrinsic_dim
        self.boundary = [*geomA.boundary, *geomB.boundary]

    def sdf(self, p: torch.Tensor):
        return torch.max(self.geomA.sdf(p), self.geomB.sdf(p))

    # # GLSL expression for the intersection: max(dA,dB)
    # def glsl_sdf(self) -> str:
    #     return f"max({self.geomA.glsl_sdf()}, {self.geomB.glsl_sdf()})"

    def get_bounding_box(self):
        boxA = self.geomA.get_bounding_box()
        boxB = self.geomB.get_bounding_box()
        return [max(boxA[i], boxB[i]) if i % 2 == 0 else min(boxA[i], boxB[i]) for i in range(2 * self.dim)]

    def in_sample(self, num_samples: int, with_boundary: bool = False) -> torch.Tensor:
        samples = torch.cat(
            [self.geomA.in_sample(num_samples, with_boundary), self.geomB.in_sample(num_samples, with_boundary)], dim=0)
        if with_boundary:
            return samples[(self.sdf(samples) <= 0).squeeze()]

        return samples[(self.sdf(samples) < 0).squeeze()]

    def on_sample(self, num_samples: int, with_normal: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        if with_normal:
            a, an = self.geomA.on_sample(num_samples, with_normal=True)
            b, bn = self.geomB.on_sample(num_samples, with_normal=True)
            samples = torch.cat([a, b], dim=0)
            normals = torch.cat([an, bn], dim=0)
            return samples[torch.isclose(self.sdf(samples), torch.tensor(0.)).squeeze()], normals[
                torch.isclose(self.sdf(samples), torch.tensor(0.)).squeeze()]

        samples = torch.cat(
            [self.geomA.on_sample(num_samples, with_normal), self.geomB.on_sample(num_samples, with_normal)], dim=0)
        return samples[torch.isclose(self.sdf(samples), torch.tensor(0.)).squeeze()]


class ComplementGeometry(GeometryBase):
    def __init__(self, geom: GeometryBase):
        super().__init__()
        self.geom = geom
        self.dim = geom.dim
        self.intrinsic_dim = geom.intrinsic_dim
        self.boundary = [*geom.boundary]

    def sdf(self, p: torch.Tensor):
        return -self.geom.sdf(p)

    # # GLSL expression for the complement: -d
    # def glsl_sdf(self) -> str:
    #     return f"-({self.geom.glsl_sdf()})"

    def get_bounding_box(self) -> List[float]:
        bounding_box_geom = self.geom.get_bounding_box()
        return [float('-inf') if i % 2 == 0 else float('inf') for d in range(self.dim) for i in range(2)]

    def in_sample(self, num_samples: int, with_boundary: bool = False) -> torch.Tensor:
        return self.geom.in_sample(num_samples, with_boundary)

    def on_sample(self, num_samples: int, with_normal: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        return self.geom.on_sample(num_samples, with_normal)


class ExtrudeBody(GeometryBase):
    """
    ExtrudeBody — turn a 2-D geometry into a 3-D solid by extruding it
    along an arbitrary direction vector.

        direction  d  (len = |d|)
        unit dir   d̂ = d / |d|
        half-thick h  = |d| / 2

        SDF:  max( d₂(q), |dot(p,d̂)| – h )
        q = ( dot(p,u), dot(p,v) )   with  u,v,d̂ orthonormal.

    Parameters
    ----------
    base2d : GeometryBase
        Any 2-D geometry that already implements `glsl_sdf`.
    direction : (3,) sequence / torch.Tensor
        Direction *and* length of the extrusion (e.g. (0,0,2) ⇒ thickness 2).
    """

    # ------------------------------------------------------------------ #
    # construction helpers
    # ------------------------------------------------------------------ #
    def _orthonormal(self, n: torch.Tensor) -> torch.Tensor:
        """Return a unit vector orthogonal to n (robust for all n)."""
        ex = torch.tensor([1., 0., 0.], dtype=n.dtype, device=n.device)
        ey = torch.tensor([0., 1., 0.], dtype=n.dtype, device=n.device)
        v = torch.linalg.cross(n, ex)
        if torch.norm(v) < 1e-7:
            v = torch.linalg.cross(n, ey)
        return v / torch.norm(v)

    # ------------------------------------------------------------------ #
    # ctor
    # ------------------------------------------------------------------ #
    def __init__(self, base2d: GeometryBase, direction: Union[torch.Tensor, list, tuple] = (0.0, 0.0, 1.0), ):
        super().__init__(dim=3, intrinsic_dim=3)
        if base2d.dim != 2:
            raise ValueError("base2d must be 2-D")
        self.base = base2d

        d = torch.tensor(direction, dtype=self.dtype)
        L = torch.norm(d)
        if L < 1e-8:
            raise ValueError("direction vector must be non-zero")
        self.d = d / L  # unit direction
        self.len = L.item()  # total thickness
        self.h = self.len * 0.5  # half thickness

        self.u = self._orthonormal(self.d)
        self.v = torch.linalg.cross(self.d, self.u)

    # ------------------------------------------------------------------ #
    # SDF (Torch)
    # ------------------------------------------------------------------ #
    def sdf(self, p: torch.Tensor):
        proj_u = torch.matmul(p, self.u)  # (N,)
        proj_v = torch.matmul(p, self.v)
        q = torch.stack([proj_u, proj_v], dim=1)  # (N,2)

        d2 = self.base.sdf(q)  # (N,1) or (N,)
        dz = torch.abs(torch.matmul(p, self.d)) - self.h
        return torch.max(d2, dz.unsqueeze(1))

    # ------------------------------------------------------------------ #
    # Axis-aligned bounding box (tight)
    # ------------------------------------------------------------------ #
    def get_bounding_box(self) -> List[float]:
        # Obtain 2-D bbox in (u,v) space
        bx_min, bx_max, by_min, by_max = self.base.get_bounding_box()
        corners_2d = torch.tensor([[bx_min, by_min], [bx_min, by_max], [bx_max, by_min], [bx_max, by_max], ],
                                  dtype=torch.float64, )

        pts = []
        for s in (-self.h, self.h):
            for x, y in corners_2d:
                pts.append(x * self.u + y * self.v + s * self.d)
        pts = torch.stack(pts, dim=0)  # (8,3)

        xyz_min = pts.min(dim=0).values
        xyz_max = pts.max(dim=0).values
        x_min, y_min, z_min = xyz_min.tolist()
        x_max, y_max, z_max = xyz_max.tolist()
        return [x_min, x_max, y_min, y_max, z_min, z_max]

    # ------------------------------------------------------------------ #
    # interior sampling
    # ------------------------------------------------------------------ #
    def in_sample(self, num_samples: int, with_boundary: bool = False) -> torch.Tensor:
        """
        Uniform volume sampling:
          * pick (u,v) inside base2d
          * pick z uniformly in [-h, h]
        """
        # Number of base2d samples
        pts2d = self.base.in_sample(num_samples, with_boundary=False)
        # if base2d returns fewer than requested, repeat
        if pts2d.shape[0] < num_samples:
            reps = (num_samples + pts2d.shape[0] - 1) // pts2d.shape[0]
            pts2d = pts2d.repeat(reps, 1)[:num_samples]

        z = torch.rand(pts2d.shape[0], 1, generator=self.gen) * self.len - self.h  # (-h, h)

        # map to 3-D
        xyz = pts2d[:, 0:1] * self.u + pts2d[:, 1:2] * self.v + z * self.d
        return xyz

    # ------------------------------------------------------------------ #
    # boundary sampling
    # ------------------------------------------------------------------ #
    def on_sample(self, num_samples: int, with_normal: bool = False, separate: bool = False) -> Any:
        """
        * 2/3 样本在两个盖子（顶/底整块面域）
        * 1/3 样本在侧壁（由 base2d 的边界沿 d 拉伸）
        """
        n_cap = num_samples // 3  # 顶+底共用的2D采样数（复制到两层 → 2*n_cap）
        n_side = num_samples - 2 * n_cap  # 剩余给侧壁

        # ---- caps: 用 2D 面域采样 ----
        cap2d = self.base.in_sample(n_cap, with_boundary=True)
        # 若底层实现返回不足，重复补齐
        if cap2d.shape[0] < n_cap:
            reps = (n_cap + cap2d.shape[0] - 1) // cap2d.shape[0]
            cap2d = cap2d.repeat(reps, 1)[:n_cap]

        top_pts = cap2d[:, 0:1] * self.u + cap2d[:, 1:2] * self.v + self.h * self.d
        bot_pts = cap2d[:, 0:1] * self.u + cap2d[:, 1:2] * self.v + -self.h * self.d
        pts_cap = torch.cat([top_pts, bot_pts], dim=0)

        if with_normal:
            n_top = self.d.expand_as(top_pts)  # 顶盖法向 = +d
            n_bot = (-self.d).expand_as(bot_pts)  # 底盖法向 = -d
            normals_cap = torch.cat([n_top, n_bot], dim=0)

        # ---- side walls: 用 2D 边界采样 ----
        if with_normal:
            edge2d, edge_n2d = self.base.on_sample(n_side, with_normal=True)
        else:
            edge2d = self.base.on_sample(n_side, with_normal=False)

        m_side = edge2d.shape[0]  # 实际侧壁2D边界采样数
        z_side = (torch.rand(m_side, 1, device=edge2d.device, dtype=edge2d.dtype,
                             generator=self.gen) * self.len) - self.h
        pts_side = edge2d[:, 0:1] * self.u + edge2d[:, 1:2] * self.v + z_side * self.d

        if with_normal:
            # 侧壁法向 = 由2D边界法向投影到(u,v)平面后归一化（与d正交）
            side_norm_vec = edge_n2d[:, 0:1] * self.u + edge_n2d[:, 1:2] * self.v
            side_normals = side_norm_vec / torch.norm(side_norm_vec, dim=1, keepdim=True)

        # ---- merge & return ----
        if separate:
            if with_normal:
                return (top_pts, n_top), (bot_pts, n_bot), (pts_side, side_normals)
            else:
                return top_pts, bot_pts, pts_side
        else:
            if with_normal:
                points = torch.cat([pts_cap, pts_side], dim=0)
                normals = torch.cat([normals_cap, side_normals], dim=0)
                return points, normals
            else:
                return torch.cat([pts_cap, pts_side], dim=0)

    # # ------------------------------------------------------------------ #
    # # GLSL SDF expression
    # # ------------------------------------------------------------------ #
    # def glsl_sdf(self) -> str:
    #     dx, dy, dz = [f"{x:.6f}" for x in self.d.tolist()]
    #     ux, uy, uz = [f"{x:.6f}" for x in self.u.tolist()]
    #     vx, vy, vz = [f"{x:.6f}" for x in self.v.tolist()]
    #     h = f"{self.h:.6f}"
    #
    #     # project vec3 p → vec2 q
    #     proj = (f"vec2(dot(p, vec3({ux},{uy},{uz})), "
    #             f"dot(p, vec3({vx},{vy},{vz})))")
    #     base_expr = self.base.glsl_sdf().replace("p", proj)
    #
    #     return f"max({base_expr}, abs(dot(p, vec3({dx},{dy},{dz}))) - {h})"


class ImplicitFunctionBase(GeometryBase):
    @abstractmethod
    def shape_func(self, p: torch.Tensor) -> torch.Tensor:
        pass

    # ---------- 工具：定位可用于 dForward 的模型 ----------
    def _get_model_for_dforward(self):
        # 你也可以在子类里重写这个方法，返回真正带 dForward 的对象
        cand = getattr(self, "model", None)
        return cand if (cand is not None and hasattr(cand, "dForward")) else None

    def _has_dforward(self) -> bool:
        return self._get_model_for_dforward() is not None

    @torch.no_grad()
    def _eval_grad_dforward(self, p: torch.Tensor) -> torch.Tensor:
        model = self._get_model_for_dforward()
        nx = model.dForward(p, (1, 0, 0)).squeeze(-1)
        ny = model.dForward(p, (0, 1, 0)).squeeze(-1)
        nz = model.dForward(p, (0, 0, 1)).squeeze(-1)
        return torch.stack([nx, ny, nz], dim=-1)  # (N,3)

    def _eval_grad(self, p: torch.Tensor) -> torch.Tensor:
        if self._has_dforward():
            return self._eval_grad_dforward(p)
        # 回退到 autograd
        p_req = p.detach().clone().requires_grad_(True)
        f = self.shape_func(p_req)
        if f.ndim == 2 and f.size(-1) == 1:
            f = f.squeeze(-1)
        g = torch.autograd.grad(f, p_req, grad_outputs=torch.ones_like(f),
                                create_graph=False, retain_graph=False)[0]
        return g

    @torch.no_grad()
    def _eval_hessian_dforward(self, p: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回 (H, lap)：
          H: (N,3,3)  Hessian
          lap: (N,)   Δf = trace(H)
        需要 dForward 支持二阶多重指标。
        """
        model = self._get_model_for_dforward()
        f_xx = model.dForward(p, (2, 0, 0)).squeeze(-1)
        f_yy = model.dForward(p, (0, 2, 0)).squeeze(-1)
        f_zz = model.dForward(p, (0, 0, 2)).squeeze(-1)
        f_xy = model.dForward(p, (1, 1, 0)).squeeze(-1)
        f_xz = model.dForward(p, (1, 0, 1)).squeeze(-1)
        f_yz = model.dForward(p, (0, 1, 1)).squeeze(-1)
        # 组装对称 Hessian
        H = torch.zeros(p.shape[0], 3, 3, device=p.device, dtype=p.dtype)
        H[:, 0, 0] = f_xx
        H[:, 1, 1] = f_yy
        H[:, 2, 2] = f_zz
        H[:, 0, 1] = H[:, 1, 0] = f_xy
        H[:, 0, 2] = H[:, 2, 0] = f_xz
        H[:, 1, 2] = H[:, 2, 1] = f_yz
        lap = f_xx + f_yy + f_zz
        return H, lap

    def sdf(self, p: torch.Tensor, with_normal=False, with_curvature=False) -> Union[
        torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        计算 规范化 SDF (= f/||∇f||)，可选返回单位法向与(按约定)平均曲率 H = 1/2 ∇·n。
        - 优先 dForward；若不可用则回退 autograd。
        - 不假设 ||∇f|| ≈ 1。
        """
        eps = torch.finfo(p.dtype).eps
        # 评估 f
        f = self.shape_func(p)
        if f.ndim == 2 and f.size(-1) == 1:
            f = f.squeeze(-1)

        sdf = f.unsqueeze(-1)
        if not (with_normal or with_curvature):
            return sdf.detach()

        # 评估 ∇f
        g = self._eval_grad(p)
        gnorm = g.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        n = g / gnorm
        if with_normal and (not with_curvature):
            return sdf.detach(), n.detach()

        # ---- with_curvature: 计算 div(n) ----
        if self._has_dforward():
            # 用闭式公式: div(n) = (||g||^2 * tr(H) - g^T H g) / ||g||^3
            H, lap = self._eval_hessian_dforward(p)
            # g^T H g
            g_col = g.unsqueeze(-1)  # (N,3,1)
            Hg = torch.matmul(H, g_col)  # (N,3,1)
            gT_H_g = torch.matmul(g_col.transpose(1, 2), Hg).squeeze(-1).squeeze(-1)  # (N,)
            g2 = (gnorm.squeeze(-1) ** 2)  # (N,)
            div_n = (g2 * lap - gT_H_g) / (gnorm.squeeze(-1) ** 3 + eps)  # (N,)
            mean_curv = 0.5 * div_n
            return sdf.detach(), n.detach(), mean_curv.detach().unsqueeze(-1)
        else:
            # 回退：autograd 直接对 n 的各分量求散度
            p_req = p.detach().clone().requires_grad_(True)
            # 重新评估以建立计算图
            f2 = self.shape_func(p_req)
            if f2.ndim == 2 and f2.size(-1) == 1:
                f2 = f2.squeeze(-1)
            g2 = torch.autograd.grad(f2, p_req, grad_outputs=torch.ones_like(f2),
                                     create_graph=True, retain_graph=True)[0]
            g2n = g2.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            n2 = g2 / g2n
            div = 0.0
            for i in range(3):
                di = torch.autograd.grad(n2[:, i], p_req,
                                         grad_outputs=torch.ones_like(n2[:, i]),
                                         create_graph=False, retain_graph=True)[0][:, i]
                div = div + di
            mean_curv = 0.5 * div
            # 注意：sdf/n 我们用上一段的（不带图），只把 mean_curv 取当前图的值
            return sdf.detach(), n.detach(), mean_curv.detach().unsqueeze()


class ImplicitSurfaceBase(ImplicitFunctionBase):

    def __init__(self):
        super().__init__(dim=3, intrinsic_dim=2)

    @abstractmethod
    def shape_func(self, p: torch.Tensor) -> torch.Tensor:
        pass

    # === 新：基于 Marching Cubes 的 in_sample ===
    @torch.no_grad()
    def in_sample(self, num_samples: int, with_boundary: bool = False) -> torch.Tensor:
        """
        Marching Cubes 取初值 + 法向牛顿式投影到 φ=0.
        - 优先 skimage.measure.marching_cubes。无则退化为体素零集近似。
        - 不假设 ||∇f||≈1；投影步长为 f/||∇f||。
        - 对大网格/大批量自动分块，避免 OOM。
        """
        if num_samples <= 0:
            return torch.empty(0, self.dim, dtype=self.dtype, device=self.device)

        # 1) 估计网格分辨率（与包围盒体积、目标点数相关）
        (x_min, x_max, y_min, y_max, z_min, z_max) = self.get_bounding_box()
        # 扩大一点 bbox，避免边界截断
        margin = 0.1 * max((x_max - x_min), (y_max - y_min), (z_max - z_min))
        x_min -= margin
        x_max += margin
        y_min -= margin
        y_max += margin
        z_min -= margin
        z_max += margin
        box = torch.tensor([x_min, x_max, y_min, y_max, z_min, z_max],
                           dtype=self.dtype, device=self.device)
        # 目标：三角面片数量 ~ O(num_samples)。经验上把每维分辨率取 ~ c * num_samples^(1/3)
        c = 2.0  # 稍微密一点，便于后续抽样
        n_per_axis = int(max(128, min(320, round(c * (num_samples ** (1 / 3))))))
        nx = ny = nz = n_per_axis

        # 2) 评估 SDF 到规则网格（分块）
        def _linspace(a, b, n):
            # 与 dtype/device 保持一致
            return torch.linspace(a, b, steps=n, device=self.device, dtype=self.dtype)

        xs = _linspace(x_min, x_max, nx)
        ys = _linspace(y_min, y_max, ny)
        zs = _linspace(z_min, z_max, nz)

        # 分块评估避免一次性 nx*ny*nz
        def _eval_grid():
            sdf_grid = torch.empty((nx, ny, nz), dtype=self.dtype, device=self.device)
            # 以 z 为外层块，减小峰值内存
            bz = max(8, min(nz, 64))
            for z0 in range(0, nz, bz):
                z1 = min(nz, z0 + bz)
                Z = zs[z0:z1]
                # 网格 -> 扁平化点
                X, Y, Zm = torch.meshgrid(xs, ys, Z, indexing="ij")
                pts = torch.stack([X, Y, Zm], dim=-1).reshape(-1, 3)
                # 批量 eval
                f = self.shape_func(pts)
                if f.ndim == 2 and f.size(-1) == 1:
                    f = f.squeeze(-1)
                f = f.reshape(nx, ny, z1 - z0)
                sdf_grid[:, :, z0:z1] = f
                del X, Y, Zm, pts, f
            return sdf_grid

        sdf_grid = _eval_grid()

        try:
            from skimage import measure as _sk_measure
            _HAS_SKIMAGE = True
        except Exception:
            _HAS_SKIMAGE = False

        # 3) Marching Cubes / 退化路径：取到三角网格或体素近似点集
        # if _HAS_SKIMAGE:
        #     # skimage 需要 CPU+float32/64 numpy
        #     sdf_np = sdf_grid.detach().to("cpu").numpy()
        #     # spacing 与 origin：把网格索引坐标映射到真实坐标
        #     dx = float((x_max - x_min) / max(nx - 1, 1))
        #     dy = float((y_max - y_min) / max(ny - 1, 1))
        #     dz = float((z_max - z_min) / max(nz - 1, 1))
        #     verts, faces, _, _ = _sk_measure.marching_cubes(
        #         volume=sdf_np, level=0.0, spacing=(dx, dy, dz)
        #     )
        #     # 平移到真实原点
        #     verts[:, 0] += float(x_min)
        #     verts[:, 1] += float(y_min)
        #     verts[:, 2] += float(z_min)
        #
        #     if len(faces) == 0 or len(verts) == 0:
        #         # 回退：用原有拒绝采样+投影
        #         return self._fallback_rejection_projection(num_samples)
        #
        #     # 保证是 C 连续并消除负步长
        #     import numpy as np
        #     verts = np.ascontiguousarray(verts)  # float64/32 都行
        #     faces = np.ascontiguousarray(faces, dtype=np.int64)
        #
        #     # 先在 CPU from_numpy，再搬到目标 device
        #     verts_t = torch.from_numpy(verts).to(device=self.device, dtype=self.dtype)
        #     faces_t = torch.from_numpy(faces).to(device=self.device)
        #     # 在三角面上按面积概率，采样三角形质心/重心点
        #     init_pts = self._sample_on_tri_mesh(verts_t, faces_t, k=max(num_samples * 2, 1024))
        # else:
        #     # 无 skimage：取零截面体素的“面中心/边中心近似”，凑一个初值云
        init_pts = self._fallback_voxel_zerocross(xs, ys, zs, sdf_grid, k=max(num_samples * 2, 512))

        # 4) 法向牛顿式投影到 φ=0
        proj = self._project_to_surface(init_pts, max_iter=60)

        # 5) 清洗：去 NaN、去离群（按 |φ| 阈）、去重复（四舍五入到网格步长）
        f_proj = self.shape_func(proj).squeeze(-1) if proj.ndim == 2 else self.shape_func(proj)
        tol = torch.finfo(self.dtype).eps * 100
        mask = torch.isfinite(f_proj) & (f_proj.abs() < tol)
        proj = proj[mask]

        if proj.numel() == 0:
            return self._fallback_rejection_projection(num_samples)

        # # 去重：按分辨率四舍五入
        # res = max((x_max - x_min) / nx, (y_max - y_min) / ny, (z_max - z_min) / nz)
        # q = torch.round(proj / res)
        # # unique 需要把 3 列拼成一列键
        # keys = q[:, 0] * 73856093 + q[:, 1] * 19349663 + q[:, 2] * 83492791  # hash-ish
        # _, uniq_idx = torch.unique(keys.to(torch.int64), return_index=True)
        # proj = proj[uniq_idx]

        # 裁切数量
        if proj.shape[0] >= num_samples:
            # 打乱顺序，取前 num_samples 个
            idx = torch.randperm(proj.shape[0], device=proj.device, generator=self.gen)
            proj = proj[idx]
            return proj[:num_samples].contiguous()
        else:
            # 若不足，补一部分“拒绝+投影”以达到数量
            need = num_samples - proj.shape[0]
            extra = self._fallback_rejection_projection(need)
            return torch.cat([proj, extra], dim=0)[:num_samples].contiguous()

    # === 工具：法向牛顿式投影 ===
    @torch.no_grad()
    def _project_to_surface(self, points: torch.Tensor, max_iter: int = 10) -> torch.Tensor:
        pts = points.clone()
        # grad_eps = torch.tensor(1e-12, dtype=self.dtype, device=self.device)
        # 分块，避免爆显存
        B = max(2048, min(32768, points.shape[0]))
        for _ in range(max_iter):
            for i in range(0, pts.shape[0], B):
                sl = slice(i, i + B)
                p = pts[sl]
                f = self.shape_func(p.clone())
                if f.ndim == 2 and f.size(-1) == 1:
                    f = f.squeeze(-1)
                g = self._eval_grad(p.clone())
                gnorm = g.norm(dim=1, keepdim=True)
                n_hat = g / gnorm
                pts[sl] = p - f.unsqueeze(-1) * n_hat  # p_{k+1} = p_k - f/||∇f|| n̂
                if f.abs().max() < torch.finfo(self.dtype).eps:
                    break
        return pts

    # === 工具：在三角网格上按面积采样重心点 ===
    @torch.no_grad()
    def _sample_on_tri_mesh(self, V: torch.Tensor, F: torch.Tensor, k: int) -> torch.Tensor:
        v0 = V[F[:, 0]]
        v1 = V[F[:, 1]]
        v2 = V[F[:, 2]]
        # 三角形面积
        areas = torch.linalg.norm(torch.cross(v1 - v0, v2 - v0, dim=1), dim=1) * 0.5
        probs = (areas / (areas.sum() + 1e-16)).clamp_min(torch.finfo(self.dtype).tiny)
        # 多项式抽样
        idx = torch.multinomial(probs, num_samples=k, replacement=True)
        v0s, v1s, v2s = v0[idx], v1[idx], v2[idx]
        # 重心采样（u,v ~ U(0,1), u+v<=1）
        u = torch.rand(k, 1, device=self.device, dtype=self.dtype, generator=self.gen)
        v = torch.rand(k, 1, device=self.device, dtype=self.dtype, generator=self.gen)
        mask = (u + v > 1.0)
        u[mask] = 1.0 - u[mask]
        v[mask] = 1.0 - v[mask]
        w = 1.0 - u - v
        pts = u * v0s + v * v1s + w * v2s
        return pts

    # === 工具：无 skimage 时的体素零截面近似 ===
    @torch.no_grad()
    def _fallback_voxel_zerocross(self, xs, ys, zs, sdf_grid, k: int) -> torch.Tensor:
        # 简单地找出相邻体素符号变化的边/面中心作为初值
        # 这里用面中心（六个方向），够鲁棒且实现简洁
        nx, ny, nz = sdf_grid.shape
        pts = []

        def _center(i, j, k0):
            return torch.stack([xs[i], ys[j], zs[k0]])

        # x-相邻
        sign_x = torch.signbit(sdf_grid[1:, :, :]) != torch.signbit(sdf_grid[:-1, :, :])
        ix, iy, iz = torch.where(sign_x)
        if ix.numel():
            cx = (xs[ix] + xs[ix + 1]) / 2
            yy = ys[iy]
            zz = zs[iz]
            pts.append(torch.stack([cx, yy, zz], dim=1))
        # y-相邻
        sign_y = torch.signbit(sdf_grid[:, 1:, :]) != torch.signbit(sdf_grid[:, :-1, :])
        ix, iy, iz = torch.where(sign_y)
        if iy.numel():
            xx = xs[ix]
            cy = (ys[iy] + ys[iy + 1]) / 2
            zz = zs[iz]
            pts.append(torch.stack([xx, cy, zz], dim=1))
        # z-相邻
        sign_z = torch.signbit(sdf_grid[:, :, 1:]) != torch.signbit(sdf_grid[:, :, :-1])
        ix, iy, iz = torch.where(sign_z)
        if iz.numel():
            xx = xs[ix]
            yy = ys[iy]
            cz = (zs[iz] + zs[iz + 1]) / 2
            pts.append(torch.stack([xx, yy, cz], dim=1))

        if len(pts) == 0:
            # 没截到：退回拒绝采样+投影
            return self._fallback_rejection_projection(k)

        P = torch.cat(pts, dim=0).to(self.device, self.dtype)
        # # 如果过多，随机下采样到 3k（给投影留余量）
        if P.shape[0] > 3 * k:
            sel = torch.randperm(P.shape[0], device=self.device)[:3 * k]
            P = P[sel]
        return P

    # === 工具：老版“拒绝采样 + 投影”兜底 ===
    @torch.no_grad()
    def _fallback_rejection_projection(self, num_samples: int) -> torch.Tensor:
        """
        拒绝采样 + 迭代投影到 φ=0。
        导数优先用 dForward，不可用则回退 autograd。
        不假设 ||∇f||≈1：步长为 (f/||∇f||) 沿单位法向。
        """
        print("Back to rejection_projection")
        x_min, x_max, y_min, y_max, z_min, z_max = self.get_bounding_box()
        volume = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
        resolution = (volume / max(num_samples, 1)) ** (1 / self.dim)
        eps_band = 1 * resolution

        collected = []
        max_iter = 10
        oversample = int(num_samples * 1.5)
        grad_eps = 1e-12

        while sum(c.shape[0] for c in collected) < num_samples:
            rand = torch.rand(oversample, self.dim, device=self.device, generator=self.gen, dtype=self.dtype)
            p = torch.empty(oversample, self.dim, dtype=self.dtype, device=self.device)
            p[:, 0] = (x_min - 4 * eps_band) + rand[:, 0] * ((x_max + 4 * eps_band) - (x_min - 4 * eps_band))
            p[:, 1] = (y_min - 4 * eps_band) + rand[:, 1] * ((y_max + 4 * eps_band) - (y_min - 4 * eps_band))
            p[:, 2] = (z_min - 4 * eps_band) + rand[:, 2] * ((z_max + 4 * eps_band) - (z_min - 4 * eps_band))

            # φ、∇φ
            f = self.shape_func(p)
            if f.ndim == 2 and f.size(-1) == 1:
                f = f.squeeze(-1)
            g = self._eval_grad(p)  # dForward or autograd
            gnorm = g.norm(dim=1, keepdim=True).clamp_min(grad_eps)
            n_hat = g / gnorm
            sdf = f.unsqueeze(-1)

            near_mask = (sdf.abs() < eps_band).squeeze(-1)
            near_points = p[near_mask]
            near_normals = n_hat[near_mask]
            near_sdf = sdf[near_mask].squeeze(-1)

            for _ in range(max_iter):
                if near_points.shape[0] == 0:
                    break
                # p_{k+1} = p_k - (f/||∇f||) * n̂
                near_points = near_points - near_sdf.unsqueeze(-1) * near_normals

                # 重新评估
                f_proj = self.shape_func(near_points)
                if f_proj.ndim == 2 and f_proj.size(-1) == 1:
                    f_proj = f_proj.squeeze(-1)
                g_proj = self._eval_grad(near_points)
                gnorm_proj = g_proj.norm(dim=1, keepdim=True).clamp_min(grad_eps)
                near_normals = g_proj / gnorm_proj
                near_sdf = (f_proj.unsqueeze(-1)).squeeze(-1)

                # 收敛阈值考虑尺度
                if near_sdf.abs().max().item() < torch.finfo(self.dtype).eps * resolution:
                    break

            mask = ~torch.isnan(near_sdf).squeeze() & (
                    near_sdf.abs() < torch.finfo(self.dtype).eps * resolution).squeeze()
            collected.append(near_points[mask].detach())

        return torch.cat(collected, dim=0)[:num_samples]

    def on_sample(self, num_samples: int, with_normal: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        empty = torch.empty((0, self.dim), dtype=self.dtype, device=self.device)
        if with_normal:
            return empty, empty
        return empty


class Point1D(GeometryBase):
    """
    Class representing a 1D point.

    Attributes:
    ----------
    x : torch.float64
        The x-coordinate of the point.
    """

    def __init__(self, x: torch.float64):
        """
        Initialize the Point1D object.

        Args:
        ----
        x : torch.float64
            The x-coordinate of the point.
        """
        super().__init__(dim=1, intrinsic_dim=0)
        self.x = x

    def sdf(self, p: torch.Tensor):
        """
        Compute the signed distance of a point to the point.

        Args:
        ----
        p : torch.Tensor
            A tensor of points.

        Returns:
        -------
        torch.Tensor
            A tensor of signed distances.
        """
        return torch.abs(p - self.x)

    # def glsl_sdf(self) -> str:
    #     return f"abs(p - {float(self.x)})"

    def get_bounding_box(self):
        """
        Get the bounding box of the point.

        Returns:
        -------
        list
            The bounding box of the point.
        """
        return [self.x, self.x]

    def __eq__(self, other):
        """
        Check if two points are equal.

        Args:
        ----
        other : Point1D
            Another point object.

        Returns:
        -------
        bool
            True if the points are equal, False otherwise.
        """
        if not isinstance(other, Point1D):
            return False

        return self.x == other.x

    def in_sample(self, num_samples: int, with_boundary: bool = False) -> torch.Tensor:
        """
        Generate samples within the point.

        Args:
        ----
        num_samples : int
            The number of samples to generate.
        with_boundary : bool, optional
            Whether to include boundary points in the samples.

        Returns:
        -------
        torch.Tensor
            A tensor of points sampled from the point.
        """
        return torch.tensor([[self.x]] * num_samples)

    def on_sample(self, num_samples: int, with_normal: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Generate samples on the boundary of the point.

        Args:
        ----
        num_samples : int
            The number of samples to generate.
        with_normal : bool, optional
            Whether to include normal vectors.

        Returns:
        -------
        torch.Tensor or tuple
            A tensor of points sampled from the boundary of the point or a tuple of tensors of points and normal vectors.
        """
        if with_normal:
            raise NotImplementedError("Normal vectors are not available for 1D points.")
        return torch.tensor([[self.x]] * num_samples)


class Point2D(GeometryBase):
    """
    Class representing a 2D point.

    Attributes:
    ----------
    x : torch.float64
        The x-coordinate of the point.
    y : torch.float64
        The y-coordinate of the point.
    """

    def __init__(self, x: torch.float64, y: torch.float64):
        """
        Initialize the Point2D object.

        Args:
        ----
        x : torch.float64
            The x-coordinate of the point.
        y : torch.float64
            The y-coordinate of the point.
        """
        super().__init__(dim=2, intrinsic_dim=0)
        self.x = x
        self.y = y

    def sdf(self, p: torch.Tensor):
        """
        Compute the signed distance of a point to the point.

        Args:
        ----
        p : torch.Tensor
            A tensor of points.

        Returns:
        -------
        torch.Tensor
            A tensor of signed distances.
        """
        return torch.norm(p - torch.tensor([self.x, self.y]), dim=1)

    # def glsl_sdf(self) -> str:
    #     return f"length(p - vec2({float(self.x)}, {float(self.y)}))"

    def get_bounding_box(self):
        """
        Get the bounding box of the point.

        Returns:
        -------
        list
            The bounding box of the point.
        """
        return [self.x, self.x, self.y, self.y]

    def __eq__(self, other):
        """
        Check if two points are equal.

        Args:
        ----
        other : Point2D
            Another point object.

        Returns:
        -------
        bool
            True if the points are equal, False otherwise.
        """
        if not isinstance(other, Point2D):
            return False

        return self.x == other.x and self.y == other.y

    def in_sample(self, num_samples: int, with_boundary: bool = False) -> torch.Tensor:
        """
        Generate samples within the point.

        Args:
        ----
        num_samples : int
            The number of samples to generate.
        with_boundary : bool, optional
            Whether to include boundary points in the samples.

        Returns:
        -------
        torch.Tensor
            A tensor of points sampled from the point.
        """
        return torch.tensor([[self.x, self.y]] * num_samples)

    def on_sample(self, num_samples: int, with_normal: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Generate samples on the boundary of the point.

        Args:
        ----
        num_samples : int
            The number of samples to generate.
        with_normal : bool, optional
            Whether to include normal vectors.

        Returns:
        -------
        torch.Tensor or tuple
            A tensor of points sampled from the boundary of the point or a tuple of tensors of points and normal vectors.
        """
        if with_normal:
            raise NotImplementedError("Normal vectors are not available for 2D points.")
        return torch.tensor([[self.x, self.y]] * num_samples)


class Point3D(GeometryBase):
    """
    Class representing a 3D point.

    Attributes:
    ----------
    x : torch.float64
        The x-coordinate of the point.
    y : torch.float64
        The y-coordinate of the point.
    z : torch.float64
        The z-coordinate of the point.
    """

    def __init__(self, x: torch.float64, y: torch.float64, z: torch.float64):
        """
        Initialize the Point3D object.

        Args:
        ----
        x : torch.float64
            The x-coordinate of the point.
        y : torch.float64
            The y-coordinate of the point.
        z : torch.float64
            The z-coordinate of the point.
        """
        super().__init__(dim=3, intrinsic_dim=0)
        self.x = x
        self.y = y
        self.z = z

    def sdf(self, p: torch.Tensor):
        """
        Compute the signed distance of a point to the point.

        Args:
        ----
        p : torch.Tensor
            A tensor of points.

        Returns:
        -------
        torch.Tensor
            A tensor of signed distances.
        """
        return torch.norm(p - torch.tensor([self.x, self.y, self.z]), dim=1)

    # def glsl_sdf(self) -> str:
    #     return f"length(p - vec3({float(self.x)}, {float(self.y)}, {float(self.z)}))"

    def get_bounding_box(self):
        """
        Get the bounding box of the point.

        Returns:
        -------
        list
            The bounding box of the point.
        """
        return [self.x, self.x, self.y, self.y, self.z, self.z]

    def __eq__(self, other):
        """
        Check if two points are equal.

        Args:
        ----
        other : Point3D
            Another point object.

        Returns:
        -------
        bool
            True if the points are equal, False otherwise.
        """
        if not isinstance(other, Point3D):
            return False

        return self.x == other.x and self.y == other.y and self.z == other.z

    def in_sample(self, num_samples: int, with_boundary: bool = False) -> torch.Tensor:
        """
        Generate samples within the point.

        Args:
        ----
        num_samples : int
            The number of samples to generate.
        with_boundary : bool, optional
            Whether to include boundary points in the samples.

        Returns:
        -------
        torch.Tensor
            A tensor of points sampled from the point.
        """
        return torch.tensor([[self.x, self.y, self.z]] * num_samples)

    def on_sample(self, num_samples: int, with_normal: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Generate samples on the boundary of the point.

        Args:
        ----
        num_samples : int
            The number of samples to generate.
        with_normal : bool, optional
            Whether to include normal vectors.

        Returns:
        -------
        torch.Tensor or tuple
            A tensor of points sampled from the boundary of the point or a tuple of tensors of points and normal vectors.
        """
        if with_normal:
            raise NotImplementedError("Normal vectors are not available for 3D points.")
        return torch.tensor([[self.x, self.y, self.z]] * num_samples)


class Line1D(GeometryBase):
    """
    Class representing a 1D line segment.

    Attributes:
    ----------
    x1 : torch.float64
        The x-coordinate of the first endpoint.
    x2 : torch.float64
        The x-coordinate of the second endpoint.
    boundary : list
        The boundary points of the line segment.
    """

    def __init__(self, x1: torch.float64, x2: torch.float64):
        """
        Initialize the Line1D object.

        Args:
        ----
        x1 : torch.float64
            The x-coordinate of the first endpoint.
        x2 : torch.float64
            The x-coordinate of the second endpoint.
        """
        super().__init__(dim=1, intrinsic_dim=1)
        self.x1 = x1
        self.x2 = x2
        self.boundary = [Point1D(x1), Point1D(x2)]

    def sdf(self, p: torch.Tensor):
        """
        Compute the signed distance of a point to the line segment.

        Args:
        ----
        p : torch.Tensor
            A tensor of points.

        Returns:
        -------
        torch.Tensor
            A tensor of signed distances.
        """

        return torch.abs(p - (self.x1 + self.x2) / 2) - abs(self.x2 - self.x1) / 2

    # def glsl_sdf(self) -> str:
    #     mid = (float(self.x1) + float(self.x2)) * 0.5
    #     half = abs(float(self.x2) - float(self.x1)) * 0.5
    #     return f"abs(p - {mid}) - {half}"

    def get_bounding_box(self):
        """
        Get the bounding box of the line segment.

        Returns:
        -------
        list
            The bounding box of the line segment.
        """
        return [self.x1, self.x2] if self.x1 < self.x2 else [self.x2, self.x1]

    def in_sample(self, num_samples: int, with_boundary: bool = False, with_random: bool = False) -> torch.Tensor:
        """
        Generate samples within the line segment.

        Args:
        ----
        num_samples : int
            The number of samples to generate.
        with_boundary : bool, optional
            Whether to include boundary points in the samples.

        Returns:
        -------
        torch.Tensor
            A tensor of points sampled from the line segment.
        """
        if with_boundary:
            return torch.linspace(self.x1, self.x2, num_samples).reshape(-1, 1)
        else:
            return torch.linspace(self.x1, self.x2, num_samples + 2)[1:-1].reshape(-1, 1)

    def on_sample(self, num_samples: int, with_normal: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Generate samples on the boundary of the line segment.

        Args:
        ----
        num_samples : int
            The number of samples to generate.
        with_normal : bool, optional
            Whether to include normal vectors.

        Returns:
        -------
        torch.Tensor or tuple
            A tensor of points sampled from the boundary of the line segment or a tuple of tensors of points and normal vectors.
        """

        a = self.boundary[0].in_sample(num_samples // 2, with_boundary=True)
        b = self.boundary[1].in_sample(num_samples // 2, with_boundary=True)
        if with_normal:
            return torch.cat([a, b], dim=0), torch.cat(
                [torch.tensor([[(self.x2 - self.x1) / abs(self.x2 - self.x1)]] * (num_samples // 2)),
                 torch.tensor([[(self.x1 - self.x2) / abs(self.x1 - self.x2)]] * (num_samples // 2))], dim=0)
        else:
            return torch.cat([a, b], dim=0)


class Line2D(GeometryBase):
    def __init__(self, x1: torch.float64, y1: torch.float64, x2: torch.float64, y2: torch.float64):
        super().__init__(dim=2, intrinsic_dim=1)
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.boundary = [Point2D(x1, y1), Point2D(x2, y2)]

    def sdf(self, p: torch.Tensor):
        a = torch.tensor([self.x1, self.y1])
        b = torch.tensor([self.x2, self.y2])
        ap = p - a
        ab = b - a
        t = torch.clamp(torch.dot(ap, ab) / torch.dot(ab, ab), 0, 1)
        return torch.norm(ap - t * ab)

    # def glsl_sdf(self) -> str:
    #     return (f"sdSegment(p, vec2({float(self.x1)}, {float(self.y1)}), "
    #             f"vec2({float(self.x2)}, {float(self.y2)}))")

    def get_bounding_box(self):
        x_min = min(self.x1, self.x2)
        x_max = max(self.x1, self.x2)
        y_min = min(self.y1, self.y2)
        y_max = max(self.y1, self.y2)
        return [x_min, x_max, y_min, y_max]

    def in_sample(self, num_samples: int, with_boundary: bool = False) -> torch.Tensor:
        if with_boundary:
            x = torch.linspace(self.x1, self.x2, num_samples).reshape(-1, 1)
            y = torch.linspace(self.y1, self.y2, num_samples).reshape(-1, 1)
            return torch.cat([x, y], dim=1)
        else:
            x = torch.linspace(self.x1, self.x2, num_samples + 2)[1:-1].reshape(-1, 1)
            y = torch.linspace(self.y1, self.y2, num_samples + 2)[1:-1].reshape(-1, 1)
            return torch.cat([x, y], dim=1)

    def on_sample(self, num_samples: int, with_normal: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        a = self.boundary[0].in_sample(num_samples // 2, with_boundary=True)
        b = self.boundary[1].in_sample(num_samples // 2, with_boundary=True)
        if with_normal:
            return torch.cat([a, b], dim=0), torch.cat([torch.tensor(
                [[(self.x2 - self.x1) / abs(self.x2 - self.x1), (self.y2 - self.y1) / abs(self.y2 - self.y1)]] * (
                        num_samples // 2)), torch.tensor(
                [[(self.x1 - self.x2) / abs(self.x1 - self.x2), (self.y1 - self.y2) / abs(self.y1 - self.y2)]] * (
                        num_samples // 2))], dim=0)
        else:
            return torch.cat([a, b], dim=0)


class Line3D(GeometryBase):
    def __init__(self, x1: torch.float64, y1: torch.float64, z1: torch.float64, x2: torch.float64, y2: torch.float64,
                 z2: torch.float64):
        super().__init__(dim=3, intrinsic_dim=1)
        self.x1 = x1
        self.y1 = y1
        self.z1 = z1
        self.x2 = x2
        self.y2 = y2
        self.z2 = z2
        self.boundary = [Point3D(x1, y1, z1), Point3D(x2, y2, z2)]

    def sdf(self, p: torch.Tensor):
        a = torch.tensor([self.x1, self.y1, self.z1])
        b = torch.tensor([self.x2, self.y2, self.z2])
        ap = p - a
        ab = b - a
        t = torch.clamp(torch.dot(ap, ab) / torch.dot(ab, ab), 0, 1)
        return torch.norm(ap - t * ab)

    # def glsl_sdf(self) -> str:
    #     raise NotImplementedError("Line3D.glsl_sdf not yet implemented")

    def get_bounding_box(self):
        x_min = min(self.x1, self.x2)
        x_max = max(self.x1, self.x2)
        y_min = min(self.y1, self.y2)
        y_max = max(self.y1, self.y2)
        z_min = min(self.z1, self.z2)
        z_max = max(self.z1, self.z2)
        return [x_min.item(), x_max.item(), y_min.item(), y_max.item(), z_min.item(), z_max.item()]

    def in_sample(self, num_samples: int, with_boundary: bool = False) -> torch.Tensor:
        if with_boundary:
            x = torch.linspace(self.x1, self.x2, num_samples).reshape(-1, 1)
            y = torch.linspace(self.y1, self.y2, num_samples).reshape(-1, 1)
            z = torch.linspace(self.z1, self.z2, num_samples).reshape(-1, 1)
            return torch.cat([x, y, z], dim=1)
        else:
            x = torch.linspace(self.x1, self.x2, num_samples + 2)[1:-1].reshape(-1, 1)
            y = torch.linspace(self.y1, self.y2, num_samples + 2)[1:-1].reshape(-1, 1)
            z = torch.linspace(self.z1, self.z2, num_samples + 2)[1:-1].reshape(-1, 1)
            return torch.cat([x, y, z], dim=1)

    def on_sample(self, num_samples: int, with_normal: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        a = self.boundary[0].in_sample(num_samples // 2, with_boundary=True)
        b = self.boundary[1].in_sample(num_samples // 2, with_boundary=True)
        if with_normal:
            return torch.cat([a, b], dim=0), torch.cat([torch.tensor(
                [[(self.x2 - self.x1) / abs(self.x2 - self.x1), (self.y2 - self.y1) / abs(self.y2 - self.y1),
                  (self.z2 - self.z1) / abs(self.z2 - self.z1)]] * (num_samples // 2)), torch.tensor(
                [[(self.x1 - self.x2) / abs(self.x1 - self.x2), (self.y1 - self.y2) / abs(self.y1 - self.y2),
                  (self.z1 - self.z2) / abs(self.z1 - self.z2)]] * (num_samples // 2))], dim=0)
        else:
            return torch.cat([a, b], dim=0)


class Square2D(GeometryBase):
    def __init__(
            self,
            center: Union[torch.Tensor, List, Tuple],
            half: Union[torch.Tensor, List, Tuple] = None,
            radius: Union[torch.Tensor, List, Tuple] = None,
    ):
        super().__init__(dim=2, intrinsic_dim=2)

        # backward compatibility
        if half is None and radius is None:
            raise ValueError("You must provide `half` (preferred) or `radius` (deprecated)")

        if radius is not None:
            import warnings
            warnings.warn(
                "`radius` is deprecated and will be removed in future versions. Use `half` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            half = radius

        self.center = torch.tensor(center).view(1, -1)
        self.half = torch.tensor(half).view(1, -1)
        self.radius = self.half  # alias for backward compatibility

        # boundary lines
        self.boundary = [
            Line2D(self.center[0, 0] - self.half[0, 0], self.center[0, 1] - self.half[0, 1],
                   self.center[0, 0] + self.half[0, 0], self.center[0, 1] - self.half[0, 1]),
            Line2D(self.center[0, 0] + self.half[0, 0], self.center[0, 1] - self.half[0, 1],
                   self.center[0, 0] + self.half[0, 0], self.center[0, 1] + self.half[0, 1]),
            Line2D(self.center[0, 0] + self.half[0, 0], self.center[0, 1] + self.half[0, 1],
                   self.center[0, 0] - self.half[0, 0], self.center[0, 1] + self.half[0, 1]),
            Line2D(self.center[0, 0] - self.half[0, 0], self.center[0, 1] + self.half[0, 1],
                   self.center[0, 0] - self.half[0, 0], self.center[0, 1] - self.half[0, 1]),
        ]

    def sdf(self, p: torch.Tensor):
        d = torch.abs(p - self.center) - self.half
        return torch.norm(torch.clamp(d, min=0.0), dim=1, keepdim=True) + torch.clamp(
            torch.max(d, dim=1, keepdim=True).values, max=0.0
        )

    def get_bounding_box(self):
        x_min = self.center[0, 0] - self.half[0, 0]
        x_max = self.center[0, 0] + self.half[0, 0]
        y_min = self.center[0, 1] - self.half[0, 1]
        y_max = self.center[0, 1] + self.half[0, 1]
        return [x_min.item(), x_max.item(), y_min.item(), y_max.item()]

    def in_sample(self, num_samples: Union[int, List[int], Tuple[int, int]],
                  with_boundary: bool = False) -> torch.Tensor:
        if isinstance(num_samples, int):
            num_x = num_y = int(num_samples ** 0.5)
        elif isinstance(num_samples, (list, tuple)) and len(num_samples) == 2:
            num_x, num_y = int(num_samples[0]), int(num_samples[1])
        else:
            raise ValueError("num_samples must be an int or a list/tuple of two integers.")

        x_min, x_max = self.center[0, 0] - self.half[0, 0], self.center[0, 0] + self.half[0, 0]
        y_min, y_max = self.center[0, 1] - self.half[0, 1], self.center[0, 1] + self.half[0, 1]

        if with_boundary:
            x = torch.linspace(x_min, x_max, num_x)
            y = torch.linspace(y_min, y_max, num_y)
        else:
            x = torch.linspace(x_min, x_max, num_x + 2)[1:-1]
            y = torch.linspace(y_min, y_max, num_y + 2)[1:-1]

        X, Y = torch.meshgrid(x, y, indexing='ij')
        return torch.cat([X.reshape(-1, 1), Y.reshape(-1, 1)], dim=1)

    def on_sample(self, num_samples: Union[int, List[int], Tuple], with_normal: bool = False, separate: bool = False):
        if isinstance(num_samples, int):
            nums = [num_samples // 4] * 4
        elif isinstance(num_samples, (list, tuple)) and len(num_samples) == 2:
            nums = list(map(int, [num_samples[0], num_samples[1], num_samples[0], num_samples[1]]))
        elif isinstance(num_samples, (list, tuple)) and len(num_samples) == 4:
            nums = list(map(int, num_samples))
        else:
            raise ValueError("num_samples must be an int or a list/tuple of four integers.")

        a = self.boundary[0].in_sample(nums[0], with_boundary=True)
        b = self.boundary[1].in_sample(nums[1], with_boundary=True)
        c = self.boundary[2].in_sample(nums[2], with_boundary=True)
        d = self.boundary[3].in_sample(nums[3], with_boundary=True)

        if not separate:
            if with_normal:
                normals = torch.cat([torch.tensor([[0.0, -1.0]] * nums[0]),
                                     torch.tensor([[1.0, 0.0]] * nums[1]),
                                     torch.tensor([[0.0, 1.0]] * nums[2]),
                                     torch.tensor([[-1.0, 0.0]] * nums[3])], dim=0)
                return torch.cat([a, b, c, d], dim=0), normals
            else:
                return torch.cat([a, b, c, d], dim=0)
        else:
            if with_normal:
                return ((a, torch.tensor([[0.0, -1.0]] * nums[0])),
                        (b, torch.tensor([[1.0, 0.0]] * nums[1])),
                        (c, torch.tensor([[0.0, 1.0]] * nums[2])),
                        (d, torch.tensor([[-1.0, 0.0]] * nums[3])))
            else:
                return a, b, c, d


class Square3D(GeometryBase):
    def __init__(self, center: Union[torch.Tensor, List, Tuple], radius: Union[torch.Tensor, List, Tuple]):
        super().__init__(dim=3, intrinsic_dim=2)
        self.center = torch.tensor(center).view(1, -1) if isinstance(center, (list, tuple)) else center.view(1, -1)
        self.radius = torch.tensor(radius).view(1, -1) if isinstance(radius, (list, tuple)) else radius.view(1, -1)

        for i in range(3):
            if self.radius[0, i] == 0.0:
                j, k = (i + 1) % 3, (i + 2) % 3

                p1 = self.center.clone().squeeze()
                p1[j] -= self.radius[0, j]
                p1[k] -= self.radius[0, k]

                p2 = p1.clone()
                p2[j] += 2 * self.radius[0, j]

                p3 = p2.clone()
                p3[k] += 2 * self.radius[0, k]

                p4 = p3.clone()
                p4[j] -= 2 * self.radius[0, j]

                # 使用顶点定义四条边
                self.boundary = [Line3D(*p1, *p2), Line3D(*p2, *p3), Line3D(*p3, *p4), Line3D(*p4, *p1), ]
                break

    def sdf(self, p: torch.Tensor):
        d = torch.abs(p - self.center) - self.radius
        return torch.norm(torch.clamp(d, min=0.0), dim=1, keepdim=True) + torch.clamp(
            torch.max(d, dim=1, keepdim=True).values, max=0.0)

        # def glsl_sdf(self) -> str:
        #     """
        #     Return a GLSL expression that computes the signed distance from `p`
        #     (a vec3 in shader scope) to this axis‑aligned cube.
        #     """
        #     cx, cy, cz = map(float, self.center.squeeze())
        #     rx, ry, rz = map(float, self.radius.squeeze())
        #     return ("length(max(abs(p - vec3({cx},{cy},{cz})) - vec3({rx},{ry},{rz}), 0.0))"
        #             "+ min(max(max(abs(p.x-{cx})-{rx}, abs(p.y-{cy})-{ry}), abs(p.z-{cz})-{rz}), 0.0)").format(cx=cx, cy=cy,
        # cz = cz, rx = rx,
        # ry = ry, rz = rz)

    def get_bounding_box(self):
        x_min = self.center[0, 0] - self.radius[0, 0]
        x_max = self.center[0, 0] + self.radius[0, 0]
        y_min = self.center[0, 1] - self.radius[0, 1]
        y_max = self.center[0, 1] + self.radius[0, 1]
        z_min = self.center[0, 2] - self.radius[0, 2]
        z_max = self.center[0, 2] + self.radius[0, 2]
        return [x_min.item(), x_max.item(), y_min.item(), y_max.item(), z_min.item(), z_max.item()]

    def in_sample(self, num_samples: int, with_boundary: bool = False) -> torch.Tensor:
        """
        Uniform sampling on the interior of the square face (2D manifold in R^3).
        """
        n = int(num_samples ** 0.5)

        # 找到法向方向 i，以及面内方向 j,k
        for i in range(3):
            if self.radius[0, i] == 0.0:
                j, k = (i + 1) % 3, (i + 2) % 3
                break
        else:
            raise ValueError("Square3D requires exactly one zero radius.")

        if with_boundary:
            tj = torch.linspace(-self.radius[0, j], self.radius[0, j], n)
            tk = torch.linspace(-self.radius[0, k], self.radius[0, k], n)
        else:
            tj = torch.linspace(-self.radius[0, j], self.radius[0, j], n + 2)[1:-1]
            tk = torch.linspace(-self.radius[0, k], self.radius[0, k], n + 2)[1:-1]

        TJ, TK = torch.meshgrid(tj, tk, indexing="ij")

        pts = torch.zeros((TJ.numel(), 3), dtype=self.center.dtype)
        pts[:, i] = self.center[0, i]
        pts[:, j] = self.center[0, j] + TJ.reshape(-1)
        pts[:, k] = self.center[0, k] + TK.reshape(-1)

        return pts

    def on_sample(self, num_samples: int, with_normal: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        a = self.boundary[0].in_sample(num_samples // 4, with_boundary=True)
        b = self.boundary[1].in_sample(num_samples // 4, with_boundary=True)
        c = self.boundary[2].in_sample(num_samples // 4, with_boundary=True)
        d = self.boundary[3].in_sample(num_samples // 4, with_boundary=True)
        if with_normal:
            for i in range(3):
                if self.radius[0, i] == 0.0:
                    j, k = (i + 1) % 3, (i + 2) % 3
                    an = torch.tensor([[0.0, 0.0, 0.0]] * (num_samples // 4))
                    bn = torch.tensor([[0.0, 0.0, 0.0]] * (num_samples // 4))
                    cn = torch.tensor([[0.0, 0.0, 0.0]] * (num_samples // 4))
                    dn = torch.tensor([[0.0, 0.0, 0.0]] * (num_samples // 4))
                    an[:, k] = -1.0
                    bn[:, j] = 1.0
                    cn[:, k] = 1.0
                    dn[:, j] = -1.0
                    return torch.cat([a, b, c, d], dim=0), torch.cat([an, bn, cn, dn], dim=0)
        else:
            return torch.cat([a, b, c, d], dim=0)


class Cube3D(GeometryBase):
    def __init__(self, center: Union[torch.Tensor, List, Tuple], half: Union[torch.Tensor, List, Tuple],
                 radius: Union[torch.Tensor, List, Tuple] = None):
        super().__init__(dim=3, intrinsic_dim=3)
        # backward compatibility
        if half is None and radius is None:
            raise ValueError("You must provide `half` (preferred) or `radius` (deprecated)")

        if radius is not None:
            import warnings
            warnings.warn(
                "`radius` is deprecated and will be removed in future versions. Use `half` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            half = radius
        self.center = torch.tensor(center).view(1, -1).to(dtype=self.dtype)
        self.half = torch.tensor(half).view(1, -1).to(dtype=self.dtype)
        offsets = [[self.half[0, 0], 0.0, 0.0], [-self.half[0, 0], 0.0, 0.0], [0.0, self.half[0, 1], 0.0],
                   [0.0, -self.half[0, 1], 0.0], [0.0, 0.0, self.half[0, 2]], [0.0, 0.0, -self.half[0, 2]]]
        self.boundary = [Square3D(self.center + torch.tensor(offset),
                                  torch.tensor([self.half[0, i] if offset[i] == 0.0 else 0.0 for i in range(3)])) for
                         offset in offsets]

    def sdf(self, p: torch.Tensor):
        d = torch.abs(p - self.center) - self.half
        return torch.norm(torch.clamp(d, min=0.0), dim=1, keepdim=True) + torch.clamp(
            torch.max(d, dim=1, keepdim=True).values, max=0.0)

    def get_bounding_box(self):
        x_min = self.center[0, 0] - self.half[0, 0]
        x_max = self.center[0, 0] + self.half[0, 0]
        y_min = self.center[0, 1] - self.half[0, 1]
        y_max = self.center[0, 1] + self.half[0, 1]
        z_min = self.center[0, 2] - self.half[0, 2]
        z_max = self.center[0, 2] + self.half[0, 2]
        return [x_min.item(), x_max.item(), y_min.item(), y_max.item(), z_min.item(), z_max.item()]

    def in_sample(self, num_samples: Union[int, List[int], Tuple[int, int, int]],
                  with_boundary: bool = False) -> torch.Tensor:
        if isinstance(num_samples, int):
            num_x = num_y = num_z = int(round(num_samples ** (1 / 3)))
        elif isinstance(num_samples, (list, tuple)) and len(num_samples) == 3:
            num_x, num_y, num_z = map(int, num_samples)
        else:
            raise ValueError("num_samples must be an int or a list/tuple of three integers.")

        x_min, x_max = self.center[0, 0] - self.half[0, 0], self.center[0, 0] + self.half[0, 0]
        y_min, y_max = self.center[0, 1] - self.half[0, 1], self.center[0, 1] + self.half[0, 1]
        z_min, z_max = self.center[0, 2] - self.half[0, 2], self.center[0, 2] + self.half[0, 2]

        if with_boundary:
            x = torch.linspace(x_min, x_max, num_x)
            y = torch.linspace(y_min, y_max, num_y)
            z = torch.linspace(z_min, z_max, num_z)
        else:
            x = torch.linspace(x_min, x_max, num_x + 2)[1:-1]
            y = torch.linspace(y_min, y_max, num_y + 2)[1:-1]
            z = torch.linspace(z_min, z_max, num_z + 2)[1:-1]

        X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
        return torch.cat([X.reshape(-1, 1), Y.reshape(-1, 1), Z.reshape(-1, 1)], dim=1)

    def on_sample(
            self,
            num_samples: int,
            with_normal: bool = False,
            separate: bool = False,
    ):
        """
        Sample points on the boundary of a three-dimensional cube.

        This method generates sample points on the six planar boundary faces
        of a 3D cube (±x, ±y, ±z). Each face is sampled independently via the
        corresponding `Square3D` object stored in `self.boundary`.

        The function supports both unified and face-wise sampling outputs,
        controlled by the `separate` flag, and optionally provides outward
        unit normal vectors associated with each sampled point.

        Parameters
        ----------
        num_samples : int
            Global sampling budget for the cube boundary.
            The budget is evenly distributed across the six faces as

                n_face = max(1, num_samples // 6).

            The actual number of returned sample points may differ from
            `num_samples`, depending on the sampling strategy implemented
            in `Square3D.in_sample`.

        with_normal : bool, optional
            Whether to return outward unit normal vectors for the sampled
            boundary points.

            If True, a normal vector is associated with each sampled point.
            Normals are constant over each face and aligned with the Cartesian
            coordinate axes.

            Default is False.

        separate : bool, optional
            Whether to preserve the face-wise separation of boundary samples.

            - If False (default):
              Sampled points from all faces are concatenated into a single
              tensor (and normals are concatenated accordingly).

            - If True:
              Sampled points are returned face by face, preserving the
              boundary partition. This is particularly useful when different
              boundary conditions are applied on different faces.

        Returns
        -------
        points : torch.Tensor
            Returned when `separate=False` and `with_normal=False`.

            A tensor of shape (N, 3) containing all sampled boundary points,
            where N is the total number of sampled points across all faces.

        (points, normals) : tuple of torch.Tensor
            Returned when `separate=False` and `with_normal=True`.

            - points : torch.Tensor of shape (N, 3)
              All sampled boundary points.
            - normals : torch.Tensor of shape (N, 3)
              Corresponding outward unit normal vectors.

        points_per_face : tuple of torch.Tensor
            Returned when `separate=True` and `with_normal=False`.

            A tuple of length 6. Each element is a tensor of shape (Ni, 3)
            containing the sampled points on one boundary face.

        face_data : tuple of (points, normals)
            Returned when `separate=True` and `with_normal=True`.

            A tuple of length 6. Each element is a pair

                (points_i, normals_i),

            where both tensors have shape (Ni, 3) and correspond to one
            boundary face.

        Notes
        -----
        - The ordering of boundary faces follows the ordering of
          `self.boundary`:

              0 : +x face
              1 : -x face
              2 : +y face
              3 : -y face
              4 : +z face
              5 : -z face

        - This interface is designed to be consistent with lower-dimensional
          geometries (e.g., 2D squares) and supports boundary-aware numerical
          methods such as collocation-based solvers, PINNs, and random
          feature methods.

        - When `separate=True`, the returned structure preserves geometric
          information at the face level and should be preferred when
          implementing mixed or face-dependent boundary conditions.

        Examples
        --------
        Unified boundary sampling:

        >>> points = cube.on_sample(600)

        Unified sampling with outward normals:

        >>> points, normals = cube.on_sample(600, with_normal=True)

        Face-wise sampling (no normals):

        >>> faces = cube.on_sample(600, separate=True)
        >>> for face_points in faces:
        ...     process(face_points)

        Face-wise sampling with normals (recommended for mixed BCs):

        >>> faces = cube.on_sample(600, with_normal=True, separate=True)
        >>> for face_points, face_normals in faces:
        ...     apply_boundary_condition(face_points, face_normals)
        """

        samples = []
        normals = []

        n_face = max(1, num_samples // 6)

        for i, square in enumerate(self.boundary):
            # 每个面独立采样
            s = square.in_sample(n_face, with_boundary=True)
            samples.append(s)

            if with_normal:
                n = torch.zeros(
                    (s.shape[0], 3),
                    dtype=s.dtype,
                    device=s.device
                )
                axis = i // 2  # 0:x, 1:y, 2:z
                sign = 1.0 if (i % 2 == 0) else -1.0
                n[:, axis] = sign
                normals.append(n)

        # -------------------------
        # 对齐 reference 中的 separate 语义
        # -------------------------
        if not separate:
            if with_normal:
                return (
                    torch.cat(samples, dim=0),
                    torch.cat(normals, dim=0)
                )
            else:
                return torch.cat(samples, dim=0)

        else:
            if with_normal:
                return tuple(
                    (samples[i], normals[i])
                    for i in range(len(samples))
                )
            else:
                return tuple(samples)


class CircleArc2D(GeometryBase):
    def __init__(self, center: Union[torch.Tensor, List, Tuple], radius: torch.float64):
        super().__init__(dim=2, intrinsic_dim=1)
        self.center = torch.tensor(center).view(1, -1) if not isinstance(center, torch.Tensor) else center
        self.radius = radius
        self.boundary = [Point2D(self.center[0, 0] + self.radius, self.center[0, 1])]

    def sdf(self, p: torch.Tensor):
        d = torch.norm(p - self.center, dim=1, keepdim=True) - self.radius
        return torch.abs(d)

    # def glsl_sdf(self) -> str:
    #     raise NotImplementedError("CircleArc2D.glsl_sdf not yet implemented")

    def get_bounding_box(self):
        x_min = self.center[0, 0] - self.radius
        x_max = self.center[0, 0] + self.radius
        y_min = self.center[0, 1] - self.radius
        y_max = self.center[0, 1] + self.radius
        return [x_min.item(), x_max.item(), y_min.item(), y_max.item()]

    def in_sample(self, num_samples: int, with_boundary: bool = False) -> torch.Tensor:
        if with_boundary:
            theta = torch.linspace(0.0, 2 * torch.pi, num_samples).reshape(-1, 1)
        else:
            theta = torch.linspace(0.0, 2 * torch.pi, num_samples + 2)[1:-1].reshape(-1, 1)
        x = self.center[0, 0] + self.radius * torch.cos(theta)
        y = self.center[0, 1] + self.radius * torch.sin(theta)
        return torch.cat([x, y], dim=1)

    def on_sample(self, num_samples: int, with_normal: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        raise NotImplementedError


class Circle2D(GeometryBase):
    def __init__(self, center: Union[torch.Tensor, List, Tuple], radius: torch.float64):
        super().__init__(dim=2, intrinsic_dim=2)
        self.center = torch.tensor(center).view(1, -1) if not isinstance(center, torch.Tensor) else center
        self.radius = radius
        self.boundary = [CircleArc2D(center, radius)]

    def sdf(self, p: torch.Tensor):
        return torch.norm(p - self.center, dim=1, keepdim=True) - self.radius

    # def glsl_sdf(self) -> str:
    #     cx, cy = map(float, self.center.squeeze())
    #     r = float(self.radius)
    #     return f"length(p - vec2({cx}, {cy})) - {r}"

    def get_bounding_box(self):
        x_min = self.center[0, 0] - self.radius
        x_max = self.center[0, 0] + self.radius
        y_min = self.center[0, 1] - self.radius
        y_max = self.center[0, 1] + self.radius
        return [x_min.item(), x_max.item(), y_min.item(), y_max.item()]

    def in_sample(self, num_samples: int, with_boundary: bool = False) -> torch.Tensor:
        num_samples = int(num_samples ** 0.5)
        if with_boundary:
            r = torch.linspace(0.0, self.radius, num_samples)[1:]  # 不包含0
        else:
            r = torch.linspace(0.0, self.radius, num_samples + 1)[1:-1]  # 不包含0和半径

        theta = torch.linspace(0.0, 2 * torch.pi, num_samples + 1)[:-1]
        R, T = torch.meshgrid(r, theta, indexing='ij')
        x = self.center[0, 0] + R * torch.cos(T)
        y = self.center[0, 1] + R * torch.sin(T)

        # 先加原点，再加采样点
        x = torch.cat([self.center[0, 0].view(1, 1), x.reshape(-1, 1)], dim=0)
        y = torch.cat([self.center[0, 1].view(1, 1), y.reshape(-1, 1)], dim=0)
        return torch.cat([x, y], dim=1)

    def on_sample(self, num_samples: int, with_normal: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        theta = torch.linspace(0.0, 2 * torch.pi, num_samples + 1)[:-1].reshape(-1, 1)
        x = self.center[0, 0] + self.radius * torch.cos(theta)
        y = self.center[0, 1] + self.radius * torch.sin(theta)
        a = torch.cat([x, y], dim=1)
        an = (a - self.center) / self.radius
        if with_normal:
            return a, an
        else:
            return a


class Sphere3D(GeometryBase):
    def __init__(self, center: Union[torch.Tensor, List, Tuple], radius: Union[torch.Tensor, float]):
        super().__init__(dim=3, intrinsic_dim=2)
        self.center = torch.tensor(center, dtype=torch.float64).view(1, 3) if not isinstance(center,
                                                                                             torch.Tensor) else center.view(
            1, 3)
        self.radius = torch.tensor(radius, dtype=torch.float64) if not isinstance(radius, torch.Tensor) else radius
        self.boundary = [Circle2D(self.center, self.radius)]

    def sdf(self, p: torch.Tensor):
        return torch.abs(torch.norm(p - self.center.to(p.device), dim=1, keepdim=True) - self.radius.to(p.device))

    # def glsl_sdf(self) -> str:
    #     cx, cy, cz = map(float, self.center.squeeze())
    #     r = float(self.radius)
    #     return f"length(p - vec3({cx}, {cy}, {cz})) - {r}"

    def get_bounding_box(self):
        r = self.radius.item()
        x_min = self.center[0, 0] - r
        x_max = self.center[0, 0] + r
        y_min = self.center[0, 1] - r
        y_max = self.center[0, 1] + r
        z_min = self.center[0, 2] - r
        z_max = self.center[0, 2] + r
        return [x_min.item(), x_max.item(), y_min.item(), y_max.item(), z_min.item(), z_max.item()]

    def in_sample(self, num_samples: int, with_boundary: bool = False) -> torch.Tensor:
        device = self.center.device
        num_samples = int(num_samples ** 0.5)

        theta = torch.linspace(0.0, 2 * torch.pi, num_samples, device=device)  # 1D
        phi = torch.linspace(0.0, torch.pi, num_samples, device=device)  # 1D
        T, P = torch.meshgrid(theta, phi, indexing='ij')  # 2D tensors

        R = self.radius.to(device)  # scalar tensor

        x = self.center[0, 0] + R * torch.sin(P) * torch.cos(T)
        y = self.center[0, 1] + R * torch.sin(P) * torch.sin(T)
        z = self.center[0, 2] + R * torch.cos(P)

        return torch.stack([x, y, z], dim=-1).reshape(-1, 3)

    def on_sample(self, num_samples: int, with_normal: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        empty = torch.empty((0, self.dim), dtype=self.dtype, device=self.device)
        if with_normal:
            return empty, empty
        return empty


class Ball3D(GeometryBase):
    def __init__(self, center: Union[torch.Tensor, List, Tuple], radius: float):
        super().__init__(dim=3, intrinsic_dim=3)
        self.center = torch.tensor(center, dtype=torch.float64).view(1, 3) if not isinstance(center,
                                                                                             torch.Tensor) else center.view(
            1, 3)
        self.radius = torch.tensor(radius, dtype=torch.float64) if not isinstance(radius, torch.Tensor) else radius
        self.boundary = [Sphere3D(self.center, self.radius)]

    def sdf(self, p: torch.Tensor):
        return torch.norm(p - self.center.to(p.device), dim=1, keepdim=True) - self.radius.to(p.device)

    # def glsl_sdf(self) -> str:
    #     cx, cy, cz = map(float, self.center.squeeze())
    #     r = float(self.radius)
    #     return f"length(p - vec3({cx}, {cy}, {cz})) - {r}"

    def get_bounding_box(self):
        r = self.radius.item()
        x_min = self.center[0, 0] - r
        x_max = self.center[0, 0] + r
        y_min = self.center[0, 1] - r
        y_max = self.center[0, 1] + r
        z_min = self.center[0, 2] - r
        z_max = self.center[0, 2] + r
        return [x_min.item(), x_max.item(), y_min.item(), y_max.item(), z_min.item(), z_max.item()]

    def in_sample(self, num_samples: int, with_boundary: bool = False) -> torch.Tensor:
        device = self.center.device
        num_samples = int(num_samples ** (1 / 3))

        r = torch.linspace(0.0, 1.0, num_samples, device=device)
        if not with_boundary:
            r = r[:-1]
        r = r * self.radius.to(device)

        theta = torch.linspace(0.0, 2 * torch.pi, num_samples, device=device)
        phi = torch.linspace(0.0, torch.pi, num_samples, device=device)

        R, T, P = torch.meshgrid(r, theta, phi, indexing='ij')

        x = self.center[0, 0] + R * torch.sin(P) * torch.cos(T)
        y = self.center[0, 1] + R * torch.sin(P) * torch.sin(T)
        z = self.center[0, 2] + R * torch.cos(P)

        return torch.stack([x, y, z], dim=-1).reshape(-1, 3)

    def on_sample(self, num_samples: int, with_normal: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        device = self.center.device
        num_samples = int(num_samples ** (1 / 2))
        theta = torch.linspace(0.0, 2 * torch.pi, num_samples, device=device)
        phi = torch.linspace(0.0, torch.pi, num_samples, device=device)

        T, P = torch.meshgrid(theta, phi, indexing='ij')

        R = self.radius.to(device)

        x = self.center[0, 0] + R * torch.sin(P) * torch.cos(T)
        y = self.center[0, 1] + R * torch.sin(P) * torch.sin(T)
        z = self.center[0, 2] + R * torch.cos(P)

        a = torch.stack([x, y, z], dim=-1).reshape(-1, 3)
        an = (a - self.center.to(device)) / self.radius.to(device)

        return (a, an) if with_normal else a


class Polygon2D(GeometryBase):
    # def glsl_sdf(self) -> str:
    #     raise NotImplementedError("Polygon2D.glsl_sdf not yet implemented")

    """
    Polygon class inheriting from GeometryBase.

    Attributes:
    ----------
    vertices : torch.Tensor
        A tensor of shape (N, 2) representing the vertices of the polygon.
    """

    def __init__(self, vertices: torch.Tensor):
        """
        Initialize the Polygon object.

        Args:
        ----
        vertices : torch.Tensor
            A tensor of shape (N, 2) representing the vertices of the polygon.
        """
        super().__init__(dim=2, intrinsic_dim=2)
        if vertices.ndim != 2 or vertices.shape[1] != 2:
            raise ValueError("Vertices must be a tensor of shape (N, 2).")
        self.vertices = vertices
        for i in range(vertices.shape[0]):
            self.boundary.append(Line2D(vertices[i, 0], vertices[i, 1], vertices[(i + 1) % vertices.shape[0], 0],
                                        vertices[(i + 1) % vertices.shape[0], 1]))

    def sdf(self, points: torch.Tensor) -> torch.Tensor:
        """
        Compute the signed distance function for the polygon.

        Args:
        ----
        points : torch.Tensor
            A tensor of shape (M, 2) representing the points to evaluate.

        Returns:
        -------
        torch.Tensor
            A tensor of shape (M,) containing the signed distances.
        """
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("Points must be a tensor of shape (M, 2).")

        num_points = points.shape[0]
        num_vertices = self.vertices.shape[0]

        dists = torch.full((num_points,), float('inf'), dtype=self.dtype, device=self.device)
        signs = torch.ones((num_points,), dtype=self.dtype, device=self.device)

        for i in range(num_vertices):
            v_start = self.vertices[i]
            v_end = self.vertices[(i + 1) % num_vertices]

            edge = v_end - v_start
            to_point = points - v_start

            t = torch.clamp((to_point @ edge) / (edge @ edge), 0.0, 1.0)
            closest_point = v_start + t[:, None] * edge
            dist_to_edge = torch.norm(points - closest_point, dim=1)

            dists = torch.min(dists, dist_to_edge)

            cross = edge[0] * to_point[:, 1] - edge[1] * to_point[:, 0]
            is_below = (points[:, 1] >= v_start[1]) & (points[:, 1] < v_end[1])
            is_above = (points[:, 1] < v_start[1]) & (points[:, 1] >= v_end[1])

            signs *= torch.where(is_below & (cross > 0) | is_above & (cross < 0), -1.0, 1.0)

        return signs * dists

    def get_bounding_box(self):
        """
        Get the bounding box of the polygon.

        Returns:
        -------
        List[float]
            A list of the form [x_min, x_max, y_min, y_max].
        """
        x_min = self.vertices[:, 0].min().item()
        x_max = self.vertices[:, 0].max().item()
        y_min = self.vertices[:, 1].min().item()
        y_max = self.vertices[:, 1].max().item()
        return [x_min, x_max, y_min, y_max]

    def in_sample(self, num_samples: int, with_boundary: bool = False) -> torch.Tensor:
        num_samples = int(num_samples ** (1 / 2))
        x_min, x_max, y_min, y_max = self.get_bounding_box()
        x = torch.linspace(x_min, x_max, num_samples)[1:-1]
        y = torch.linspace(y_min, y_max, num_samples)[1:-1]
        X, Y = torch.meshgrid(x, y, indexing='ij')
        interior = torch.cat([X.reshape(-1, 1), Y.reshape(-1, 1)], dim=1)
        interior = interior[self.sdf(interior) < 0]
        if with_boundary:
            return torch.cat([interior, self.on_sample(len(self.boundary) * num_samples, with_normal=False)], dim=0)
        return interior

    def on_sample(self, num_samples: int, with_normal=False) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        a = torch.cat(
            [boundary.in_sample(num_samples // len(self.boundary), with_boundary=True) for boundary in self.boundary],
            dim=0)

        if with_normal:
            normals = []
            for i in range(self.vertices.shape[0]):
                p1 = self.vertices[[i], :]
                p2 = self.vertices[[(i + 1) % self.vertices.shape[0]], :]
                normal = torch.tensor([[p1[0, 1] - p2[0, 1], p1[0, 0] - p2[0, 0]]])
                normal /= torch.norm(normal, dim=1, keepdim=True)
                normals.append(normal.repeat(num_samples // len(self.boundary), 1))
            return a, torch.cat(normals, dim=0)

        return a


class Polygon3D(GeometryBase):
    # def glsl_sdf(self) -> str:
    #     raise NotImplementedError("Polygon3D.glsl_sdf not yet implemented")

    def __init__(self, vertices: torch.Tensor):
        super().__init__(dim=3, intrinsic_dim=2)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("Vertices must be a tensor of shape (N, 3).")
        self.vertices = vertices
        self.boundary = [
            Line3D(vertices[i, 0], vertices[i, 1], vertices[i, 2], vertices[(i + 1) % vertices.shape[0], 0],
                   vertices[(i + 1) % vertices.shape[0], 1], vertices[(i + 1) % vertices.shape[0], 2]) for i in
            range(vertices.shape[0])]

    def sdf(self, points: torch.Tensor) -> torch.Tensor:
        # Not implemented here
        raise NotImplementedError

    def get_bounding_box(self):
        x_min = self.vertices[:, 0].min().item()
        x_max = self.vertices[:, 0].max().item()
        y_min = self.vertices[:, 1].min().item()
        y_max = self.vertices[:, 1].max().item()
        z_min = self.vertices[:, 2].min().item()
        z_max = self.vertices[:, 2].max().item()
        return [x_min, x_max, y_min, y_max, z_min, z_max]

    def in_sample(self, num_samples: int, with_boundary: bool = False) -> torch.Tensor:
        """
        Sample points inside the 3D polygon by:
        1. Building a local orthonormal frame (e1, e2, n) for the plane.
        2. Projecting all vertices to the (e1, e2) 2D coordinate system.
        3. Using a Polygon2D to sample points in 2D.
        4. Mapping the 2D samples back to 3D using the local frame.
        """

        # 1. Check the vertex count
        if self.vertices.shape[0] < 3:
            raise ValueError("Polygon3D must have at least 3 vertices to form a plane.")

        # 2. Compute the plane normal from the first three vertices (assuming no degeneracy)
        v0 = self.vertices[0]
        v1 = self.vertices[1]
        v2 = self.vertices[2]
        n = torch.linalg.cross(v1 - v0, v2 - v0)  # normal = (v1-v0) x (v2-v0)
        if torch.allclose(n, torch.zeros_like(n)):
            raise ValueError("The given vertices are degenerate (normal is zero).")

        # Normalize the normal vector
        n = n / torch.norm(n)

        # 3. Build a local orthonormal frame {e1, e2, n}
        #    We want e1 and e2 to lie in the plane, both perpendicular to n.
        e1 = self._find_orthonormal_vector(n)
        e2 = torch.linalg.cross(n, e1)

        # 4. Project all polygon vertices onto (e1, e2) plane
        #    We choose v0 as "plane origin" in 3D, so each vertex v_i maps to:
        #        ( (v_i - v0) dot e1,  (v_i - v0) dot e2 )
        proj_2d_vertices = []
        for vi in self.vertices:
            vi_local = vi - v0
            u = torch.dot(vi_local, e1)
            v = torch.dot(vi_local, e2)
            proj_2d_vertices.append([u, v])
        proj_2d_vertices = torch.tensor(proj_2d_vertices, dtype=self.vertices.dtype, device=self.vertices.device)

        print(proj_2d_vertices)
        # 5. Create a 2D polygon for sampling
        poly2d = Polygon2D(proj_2d_vertices)

        # 6. Perform 2D sampling
        samples_2d = poly2d.in_sample(num_samples, with_boundary=with_boundary)
        # samples_2d.shape -> (M, 2)

        # 7. Map the 2D samples back to 3D using the local frame
        #    If a 2D sample is (u_s, v_s), its corresponding 3D position is:
        #        v0 + u_s * e1 + v_s * e2
        samples_3d = []
        for (u_s, v_s) in samples_2d:
            pt_3d = v0 + u_s * e1 + v_s * e2
            samples_3d.append(pt_3d)
        samples_3d = torch.stack(samples_3d, dim=0)  # shape: (M, 3)

        return samples_3d

    def on_sample(self, num_samples: int, with_normal: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        num_samples = num_samples // len(self.boundary)
        if with_normal:
            raise NotImplementedError

        return torch.cat([boundary.in_sample(num_samples, with_boundary=True) for boundary in self.boundary], dim=0)

    @staticmethod
    def _find_orthonormal_vector(n: torch.Tensor) -> torch.Tensor:
        """
        Find one vector e1 that is perpendicular to n.
        Then e1 is normalized to be a unit vector.

        A common approach:
        - If abs(n.x) < 0.9, try e1 = cross(n, ex) where ex = (1, 0, 0).
        - Otherwise, cross with ey = (0, 1, 0), etc.
        """

        # Try crossing with the X-axis if possible
        ex = torch.tensor([1.0, 0.0, 0.0], device=n.device, dtype=n.dtype)
        ey = torch.tensor([0.0, 1.0, 0.0], device=n.device, dtype=n.dtype)

        # Check if cross(n, ex) is large enough
        c1 = torch.linalg.cross(n, ex)
        if torch.norm(c1) > 1e-7:
            e1 = c1 / torch.norm(c1)
            return e1

        # Otherwise use ey
        c2 = torch.linalg.cross(n, ey)
        if torch.norm(c2) > 1e-7:
            e1 = c2 / torch.norm(c2)
            return e1

        # Fallback: n might be (0, 0, ±1). Then crossing with ex or ey is 0.
        # So let's cross with ez = (0, 0, 1)
        ez = torch.tensor([0.0, 0.0, 1.0], device=n.device, dtype=n.dtype)
        c3 = torch.linalg.cross(n, ez)
        e1 = c3 / torch.norm(c3)
        return e1


class HyperCube(GeometryBase):
    def __init__(self, dim: int, center: Optional[torch.Tensor] = None, radius: Optional[torch.Tensor] = None):
        super().__init__(dim=dim, intrinsic_dim=dim)
        if center is None:
            self.center = torch.zeros(1, dim)
        elif isinstance(center, (list, tuple)):
            self.center = torch.tensor(center).view(1, -1)
        else:
            self.center = center.view(1, -1)

        if radius is None:
            self.radius = torch.ones(1, dim)
        elif isinstance(radius, (list, tuple)):
            self.radius = torch.tensor(radius).view(1, -1)
        else:
            self.radius = radius.view(1, -1)

    def sdf(self, p: torch.Tensor) -> torch.Tensor:
        d = torch.abs(p - self.center) - self.radius
        return torch.norm(torch.clamp(d, min=0.0), dim=1, keepdim=True) + torch.clamp(
            torch.max(d, dim=1, keepdim=True).values, max=0.0)

    def get_bounding_box(self) -> List[float]:
        bounding_box = []
        for i in range(self.dim):
            bounding_box.append((self.center[0, i] - self.radius[0, i]).item())
            bounding_box.append((self.center[0, i] + self.radius[0, i]).item())
        return bounding_box

    def in_sample(self, num_samples: int, with_boundary: bool = False) -> torch.Tensor:
        x_in = torch.rand((num_samples, self.dim), dtype=self.dtype, device=self.device, generator=self.gen)
        return x_in * 2 * self.radius - self.radius + self.center

    def on_sample(self, num_samples: int, with_normal: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        bounding_box = self.get_bounding_box()
        x_on = []
        if not with_normal:
            x_ = self.in_sample(num_samples // (2 * self.dim), with_boundary=True)
            for i in range(self.dim):
                for j in range(2):
                    x = x_.clone()
                    x[:, i] = bounding_box[2 * i + j]
                    x_on.append(x)

        return torch.cat(x_on, dim=0)

    # def glsl_sdf(self) -> str:
    #     raise NotImplementedError("HyperCube.glsl_sdf not yet implemented")


class GmshAdaptor(GeometryBase):
    """
    轻量版 Gmsh 适配器（仅保留 in_sample / on_sample / get_bounding_box / sdf）
    - 坐标一般 3D；拓扑可为 2D 或 3D
    - 2D：将多段边界连接为闭环/多环整体；sdf 为到边界折线的有符号距离（内负外正）
    - 3D：若网格为封闭体，sdf 为对外表面三角网的有符号距离（体内负体外正）；开放曲面返回无符号距离
    """

    # ========= 构造 =========
    def __init__(self, msh_path: str):
        import meshio

        self.mesh = meshio.read(msh_path)
        self.coord_dim = int(self.mesh.points.shape[1])
        self.topo_dim = self._infer_topo_dim_from_cells(self.mesh)

        # 继承基类（保持 dim / intrinsic_dim 与 topo_dim 一致）
        super().__init__(dim=self.topo_dim, intrinsic_dim=self.topo_dim)

        # 顶点坐标（numpy, float64）
        self.points: np.ndarray = self.mesh.points.astype(np.float64)

        # 规范 cells 映射（去重、排序）
        self._cells: Dict[str, np.ndarray] = self._cells_dict(self.mesh)

        # —— 边界原语缓存（仅内部使用）——
        self._boundary_edges: Optional[np.ndarray] = None  # (E,2) 仅 topo_dim==2
        self._boundary_tris: Optional[np.ndarray] = None  # (F,3) 仅 topo_dim==3
        self._is_closed_3d: bool = False

        # —— 边界/内点索引（用于 in_sample/on_sample）——
        self.boundary_vertex_mask: np.ndarray = self._find_boundary_vertices_from_cells(self.mesh, self.topo_dim)
        self.boundary_vertex_idx: np.ndarray = np.nonzero(self.boundary_vertex_mask)[0]
        all_idx = np.arange(self.points.shape[0])
        self.interior_vertex_idx: np.ndarray = all_idx[~self.boundary_vertex_mask]

        # —— 2D / 3D 边界原语构建 ——
        if self.topo_dim == 2:
            self._boundary_edges = self._build_boundary_edges_2d(self._cells)
        elif self.topo_dim == 3:
            self._boundary_tris, self._is_closed_3d = self._build_boundary_tris_3d(self._cells, self.points)

        # —— 有序边界顶点序列（多连通分量串接）——
        self._ordered_boundary_idx: np.ndarray = self._order_boundary_vertices(self._boundary_edges) \
            if self._boundary_edges is not None else self.boundary_vertex_idx

        # —— 法向（可选）——
        self.boundary_normals: Optional[np.ndarray] = None
        if self.topo_dim == 3 and self.coord_dim == 3 and self._boundary_tris is not None:
            self.boundary_normals = self._compute_vertex_normals_3d(self.points, self._boundary_tris)
        elif self.topo_dim == 2:
            pts2 = self.points[:, :2]
            if self._boundary_edges is not None:
                self.boundary_normals = self._compute_vertex_normals_2d(pts2, self._boundary_edges)
                # 按闭环（外边界/孔洞）统一方向
                self._flip_normals_2d_by_loops(pts2, self.boundary_normals, self._boundary_edges)

        # —— torch 视图（与 GeometryBase 对齐 dtype/device）——
        self.points_torch = self._ensure_tensor(self.points)  # (N, D)
        self._boundary_points_torch = self.points_torch[self._to_tensor_idx(self._ordered_boundary_idx)]
        self._interior_points_torch = self.points_torch[self._to_tensor_idx(self.interior_vertex_idx)]
        self.boundary_normals_torch = None
        if self.boundary_normals is not None:
            try:
                if self._boundary_edges is not None and self._ordered_boundary_idx.size > 0:
                    bn = self.boundary_normals[self._ordered_boundary_idx]
                else:
                    bn = self.boundary_normals[self.boundary_vertex_idx]
                self.boundary_normals_torch = self._ensure_tensor(bn)
            except Exception:
                pass

        # —— 若拓扑 2D 而坐标为 3D：公开点/法向统一投影到前两维 ——
        if self.topo_dim == 2 and self.coord_dim == 3:
            self._boundary_points_torch = self._boundary_points_torch[:, :2]
            self._interior_points_torch = self._interior_points_torch[:, :2]
            if self.boundary_normals_torch is not None and self.boundary_normals_torch.shape[1] == 3:
                self.boundary_normals_torch = self.boundary_normals_torch[:, :2]

    # ========= 对外 API =========
    def in_sample(self, num_samples: int = None, with_boundary: bool = False) -> torch.Tensor:
        """返回内点；with_boundary=True 时附加边界点（忽略 num_samples）。"""
        if with_boundary:
            if self._interior_points_torch.numel() == 0:
                return self._boundary_points_torch
            return torch.vstack([self._interior_points_torch, self._boundary_points_torch])
        return self._interior_points_torch

    def on_sample(self, num_samples: int = None, with_normal: bool = False) -> Union[
        torch.Tensor, Tuple[torch.Tensor, ...]]:
        """返回边界点；with_normal=True 且可得法向时同时返回法向。忽略 num_samples。"""
        if not with_normal or self.boundary_normals_torch is None:
            return self._boundary_points_torch
        return self._boundary_points_torch, self.boundary_normals_torch

    def get_bounding_box(self) -> List[float]:
        """
        返回 [xmin, xmax, ymin, ymax, zmin, zmax]
        - topo_dim == 2：固定返回 2D 盒（zmin=zmax=0）
        - 其他：按坐标维度推断，若仅 2D 坐标，同样 z=0
        """
        if self.topo_dim == 2:
            pts2 = self.points[:, :2]
            xmin, ymin = pts2.min(axis=0)
            xmax, ymax = pts2.max(axis=0)
            return [float(xmin), float(xmax), float(ymin), float(ymax), 0.0, 0.0]

        pts = self.points
        if pts.shape[1] == 2:
            xmin, ymin = pts.min(axis=0)
            xmax, ymax = pts.max(axis=0)
            return [float(xmin), float(xmax), float(ymin), float(ymax), 0.0, 0.0]
        else:
            xmin, ymin, zmin = pts.min(axis=0)
            xmax, ymax, zmax = pts.max(axis=0)
            return [float(xmin), float(xmax), float(ymin), float(ymax), float(zmin), float(zmax)]

    def sdf(self, p: Union[np.ndarray, torch.Tensor], batch_size: int = 32768) -> torch.Tensor:
        """
        有符号距离：
        - topo_dim==2：到边界折线的距离，域内为负
        - topo_dim==3：若为闭合体，到外表面三角网的距离，体内为负；否则返回**无符号距离**
        输入 p 可为 numpy 或 torch，形状 (N,2) 或 (N,3)
        """
        P = self._ensure_tensor(p)  # (N,D) on self.device/self.dtype
        if P.ndim != 2:
            raise ValueError("p must be (N, D)")

        # 统一升至 3D 计算
        if P.shape[1] == 2:
            P3 = torch.cat([P, torch.zeros((P.shape[0], 1), dtype=P.dtype, device=P.device)], dim=1)
        elif P.shape[1] == 3:
            P3 = P
        else:
            raise ValueError("The last dimension of p must be 2 or 3.")

        du = []
        inside = []
        while True:
            try:
                for i in range(0, P3.shape[0], batch_size):
                    p_batch = P3[i:i + batch_size]
                    du.append(self._unsigned_dist_to_boundary(p_batch))
                    inside.append(self._inside_mask(p_batch))
                du = torch.cat(du, dim=0)
                inside = torch.cat(inside, dim=0)

                # du = self._unsigned_dist_to_boundary(P3)  # (N,)
                # inside = self._inside_mask(P3)  # (N,)
                # 仅当能定义内外时给符号
                if self.topo_dim == 3 and not self._is_closed_3d:
                    return du  # 开放曲面：无符号
                return torch.where(inside, -du, du)
            except RuntimeError as e:
                if any(keyword in str(e).lower() for keyword in
                       ["out of memory", "can't allocate", "not enough memory", "std::bad_alloc"]) and batch_size > 1:
                    batch_size //= 2
                    outputs = []  # Clear outputs to retry
                    torch.cuda.empty_cache()  # Clear GPU memory
                else:
                    raise e

    # ========= 内部工具 =========
    # —— 基础：类型/设备/索引 ——
    def _ensure_tensor(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(device=self.device, dtype=self.dtype)
        return torch.as_tensor(x, device=self.device, dtype=self.dtype)

    def _to_tensor_idx(self, idx: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(idx, dtype=torch.long, device=self.device)

    # —— cells 规范化 ——
    @staticmethod
    def _cells_dict(mesh) -> Dict[str, np.ndarray]:
        d: Dict[str, List[np.ndarray]] = {}
        for block in mesh.cells:
            d.setdefault(block.type, []).append(block.data)
        out: Dict[str, np.ndarray] = {}
        for t, lst in d.items():
            arr = np.vstack(lst) if len(lst) > 1 else lst[0]
            if t in ("line", "triangle", "quad", "polygon"):
                arr = np.unique(np.sort(arr, axis=1), axis=0)  # 无向去重
            out[t] = arr
        return out

    @staticmethod
    def _infer_topo_dim_from_cells(mesh) -> int:
        has3 = has2 = has1 = has0 = False
        for block in mesh.cells:
            ct = block.type.lower()
            if ct.startswith(("tetra", "hexahedron", "wedge", "pyramid", "polyhedron")):
                has3 = True
            elif ct.startswith(("triangle", "quad", "polygon")):
                has2 = True
            elif ct.startswith(("line", "edge")) or ct in ("line",):
                has1 = True
            elif ct in ("vertex", "point"):
                has0 = True
        if has3:
            return 3
        elif has2:
            return 2
        elif has1:
            return 1
        elif has0:
            return 0
        return 0

    # —— 边界顶点检测 ——
    def _find_boundary_vertices_from_cells(self, mesh, topo_dim: int) -> np.ndarray:
        n_pts = mesh.points.shape[0]
        mask = np.zeros(n_pts, dtype=bool)
        cd = self._cells_dict(mesh)

        if topo_dim == 3:
            tri, _ = self._build_boundary_tris_3d(cd, mesh.points)
            if tri is not None:
                mask[np.unique(tri)] = True
            return mask

        if topo_dim == 2:
            edges = self._build_boundary_edges_2d(cd)
            if edges is not None and edges.size > 0:
                mask[np.unique(edges)] = True
            return mask

        if topo_dim == 1:
            line = cd.get('line')
            if line is not None:
                deg = np.zeros(n_pts, dtype=int)
                for a, b in line:
                    deg[a] += 1
                    deg[b] += 1
                mask[deg == 1] = True
            return mask

        return mask

    # —— 2D 边界构建 ——
    @staticmethod
    def _build_boundary_edges_2d(cells: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        """优先使用 line；否则从 triangle 提取边界边（仅出现一次的边）。"""
        if 'line' in cells and cells['line'].size > 0:
            return cells['line']
        tri = cells.get('triangle')
        if tri is None or tri.size == 0:
            return None
        edges = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
        uniq, cnt = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
        return uniq[cnt == 1]

    # —— 3D 外表面构建 ——
    @staticmethod
    def _build_boundary_tris_3d(cells: Dict[str, np.ndarray], points: np.ndarray) -> Tuple[Optional[np.ndarray], bool]:
        tri = cells.get('triangle')
        if tri is None or tri.size == 0:
            tets = cells.get('tetra')
            if tets is None or tets.size == 0:
                return None, False
            faces = np.vstack([
                tets[:, [0, 1, 2]],
                tets[:, [0, 1, 3]],
                tets[:, [0, 2, 3]],
                tets[:, [1, 2, 3]],
            ])
            uniq, cnt = np.unique(np.sort(faces, axis=1), axis=0, return_counts=True)
            tri = uniq[cnt == 1]
        # 粗略闭合性检测：所有边恰为两三角共享
        edges = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
        _, e_cnt = np.unique(np.sort(edges, axis=1), axis=0, return_counts=True)
        is_closed = bool(np.all(e_cnt == 2))
        return tri, is_closed

    # —— 将多段边界拆分为环/链并排序 ——
    @staticmethod
    def _order_boundary_loops(boundary_edges: Optional[np.ndarray]) -> List[np.ndarray]:
        """
        将 (E,2) 的无向边界边，按连通分量拆成若干条有序顶点序列：
        - 闭环：首尾相接，序列不重复起点；
        - 开链：首尾为度=1顶点。
        返回：每个分量一个 1D int ndarray。
        """
        if boundary_edges is None or boundary_edges.size == 0:
            return []

        from collections import defaultdict, deque
        adj = defaultdict(list)
        for a, b in boundary_edges:
            a = int(a)
            b = int(b)
            adj[a].append(b)
            adj[b].append(a)

        visited_v = set()
        loops: List[np.ndarray] = []

        # 遍历每个连通分量
        all_vs = set(adj.keys())
        processed = set()

        def component_vertices(v0: int) -> List[int]:
            comp = []
            q = deque([v0])
            processed.add(v0)
            while q:
                v = q.popleft()
                comp.append(v)
                for w in adj[v]:
                    if w not in processed:
                        processed.add(w)
                        q.append(w)
            return comp

        def walk(start: int) -> np.ndarray:
            seq = [start]
            prev = None
            cur = start
            seen = {start}
            while True:
                nbrs = adj[cur]
                nxts = [x for x in nbrs if x != prev]
                if not nxts:
                    break
                nxt = nxts[0]
                if nxt in seen:
                    # 回到起点 -> 闭环
                    if nxt == seq[0]:
                        break
                    else:
                        break
                seq.append(nxt)
                seen.add(nxt)
                prev, cur = cur, nxt
            return np.asarray(seq, dtype=int)

        while all_vs - set().union(*[set(s) for s in loops]) - set().union(processed):
            # 找到未处理的顶点
            remaining = list(all_vs - set().union(processed))
            if not remaining:
                break
            comp = component_vertices(remaining[0])
            deg1 = [v for v in comp if len(adj[v]) == 1]
            used_in_comp = set()
            if len(deg1) >= 1:
                for ep in deg1:
                    if ep in used_in_comp:
                        continue
                    seq = walk(ep)
                    loops.append(seq)
                    used_in_comp.update(seq.tolist())
            else:
                # 纯闭环
                seq = walk(comp[0])
                loops.append(seq)

        return loops

    @staticmethod
    def _order_boundary_vertices(boundary_edges: Optional[np.ndarray]) -> np.ndarray:
        """
        兼容旧接口：将所有环/链串接为一个长序列（不重复起点）。
        """
        loops = GmshAdaptor._order_boundary_loops(boundary_edges)
        if not loops:
            return np.array([], dtype=int)
        return np.concatenate(loops, axis=0)

    # —— 法向（3D/2D） ——
    @staticmethod
    def _compute_vertex_normals_3d(points: np.ndarray, faces: np.ndarray) -> Optional[np.ndarray]:
        if points.shape[1] != 3 or faces is None or faces.size == 0:
            return None
        normals = np.zeros((points.shape[0], 3), dtype=np.float64)
        v1 = points[faces[:, 1]] - points[faces[:, 0]]
        v2 = points[faces[:, 2]] - points[faces[:, 0]]
        fn = np.cross(v1, v2)  # 面法向 * 面积因子
        for i in range(3):
            np.add.at(normals, faces[:, i], fn)
        lens = np.linalg.norm(normals, axis=1, keepdims=True)
        lens[lens == 0.0] = 1.0
        return normals / lens

    @staticmethod
    def _compute_vertex_normals_2d(points2: np.ndarray, boundary_edges: np.ndarray) -> Optional[np.ndarray]:
        """
        对每个环/链，按有序边切向的垂线做长度加权平均，得到顶点法向。
        闭环与开链都支持；方向一致性在 _flip_normals_2d_by_loops 中处理。
        """
        if points2 is None or points2.shape[1] != 2 or boundary_edges is None or boundary_edges.size == 0:
            return None

        loops = GmshAdaptor._order_boundary_loops(boundary_edges)
        if not loops:
            return None

        normals = np.zeros((points2.shape[0], 2), dtype=np.float64)
        edge_set = set(tuple(sorted(e)) for e in boundary_edges)

        def add_edge_normal(i, j, v_idx):
            t = points2[j] - points2[i]
            L = np.linalg.norm(t)
            if L < 1e-12:
                return
            n = np.array([t[1], -t[0]], dtype=np.float64)  # 右手法向
            n /= (np.linalg.norm(n) + 1e-30)
            normals[v_idx] += n * L

        for seq in loops:
            seq = [int(x) for x in seq.tolist()]
            m = len(seq)
            if m < 2:
                continue
            # 判闭环：首尾是否有边
            closed = tuple(sorted((seq[0], seq[-1]))) in edge_set

            for k in range(m - 1):
                i, j = seq[k], seq[k + 1]
                # 将法向加到两个端点（长度加权）
                add_edge_normal(i, j, i)
                add_edge_normal(i, j, j)
            if closed:
                i, j = seq[-1], seq[0]
                add_edge_normal(i, j, i)
                add_edge_normal(i, j, j)

        lens = np.linalg.norm(normals, axis=1)
        nz = lens > 1e-12
        normals[nz] /= lens[nz][:, None]
        return normals

    @staticmethod
    def _try_flip_2d_normals_outward(points2: np.ndarray, normals2: np.ndarray) -> None:
        if normals2 is None:
            return
        try:
            centroid = points2.mean(axis=0)
            dots = np.einsum('ij,ij->i', normals2, points2 - centroid)
            flip = dots < 0
            normals2[flip] *= -1.0
        except Exception:
            pass

    def _flip_normals_2d_by_loops(self, points2: np.ndarray, normals2: np.ndarray, boundary_edges: np.ndarray) -> None:
        """
        将 2D 顶点法向按闭环方向统一：
        - 以“绝对面积最大”的闭环为外边界，其法向保持外向；
        - 其余闭环视作孔洞，法向整体翻转（指向孔内）。
        开链不处理（保持原样）。
        """
        if normals2 is None:
            return
        loops = self._order_boundary_loops(boundary_edges)
        if not loops:
            return

        edge_set = set(tuple(sorted(e)) for e in boundary_edges)

        def loop_signed_area(seq: np.ndarray) -> float:
            idx = seq.astype(int)
            if tuple(sorted((int(seq[0]), int(seq[-1])))) not in edge_set:
                return 0.0  # 非闭环
            xy = points2[idx]
            x, y = xy[:, 0], xy[:, 1]
            x2 = np.r_[x, x[0]]
            y2 = np.r_[y, y[0]]
            return 0.5 * float(np.sum(x2[:-1] * y2[1:] - x2[1:] * y2[:-1]))

        areas = [loop_signed_area(np.asarray(s, dtype=int)) for s in loops]
        abs_areas = [abs(a) for a in areas]
        if all(a == 0.0 for a in abs_areas):
            # 没有闭环或无法判定；回退到质心启发式
            self._try_flip_2d_normals_outward(points2, normals2)
            return

        outer_id = int(np.argmax(abs_areas))
        outer_idx = np.asarray(loops[outer_id], dtype=int)
        c_outer = points2[outer_idx].mean(axis=0)

        # 外边界：让法向远离外边界质心
        dots_outer = np.einsum("ij,ij->i", normals2[outer_idx], points2[outer_idx] - c_outer)
        flip_outer = dots_outer < 0
        normals2[outer_idx[flip_outer]] *= -1.0

        # 孔洞：整体翻转（与外边界相反）
        for k, seq in enumerate(loops):
            if k == outer_id:
                continue
            idx = np.asarray(seq, dtype=int)
            # 仅对闭环操作
            if tuple(sorted((int(seq[0]), int(seq[-1])))) not in edge_set:
                continue
            normals2[idx] *= -1.0

    # —— 距离计算 ——
    def _unsigned_dist_to_boundary(self, P3: torch.Tensor) -> torch.Tensor:
        """到边界的**无符号**距离。"""
        if self.topo_dim == 2:
            # 用边界线段集
            edges = self._boundary_edges
            if edges is None or edges.size == 0:
                # 退化：到边界顶点集合
                V = self._ensure_tensor(self.points[:, :2])
                P2 = P3[:, :2]
                d2 = ((P2[:, None, :] - V[None, :, :]) ** 2).sum(dim=2)
                return torch.sqrt(d2.min(dim=1).values)
            V2 = self._ensure_tensor(self.points[:, :2])  # (V,2)
            A = V2[edges[:, 0]]
            B = V2[edges[:, 1]]
            # 升维到 z=0 用统一段距函数
            zeroA = torch.zeros((A.shape[0], 1), dtype=A.dtype, device=A.device)
            A3 = torch.cat([A, zeroA], dim=1)
            B3 = torch.cat([B, zeroA], dim=1)
            P3z = torch.cat([P3[:, :2], torch.zeros((P3.shape[0], 1), dtype=P3.dtype, device=P3.device)], dim=1)
            return self._pointset_to_segments_distance(P3z, A3, B3)

        # topo_dim == 3
        if self._boundary_tris is not None:
            tri = self._boundary_tris
            A = self._ensure_tensor(self.points[tri[:, 0]])
            B = self._ensure_tensor(self.points[tri[:, 1]])
            C = self._ensure_tensor(self.points[tri[:, 2]])
            if A.shape[1] == 2:
                z = torch.zeros((A.shape[0], 1), dtype=A.dtype, device=A.device)
                A = torch.cat([A, z], 1)
                B = torch.cat([B, z], 1)
                C = torch.cat([C, z], 1)
            return self._pointset_to_triangles_distance(P3, A, B, C)

        # 兜底：到所有顶点
        V = self._ensure_tensor(self.points)
        if V.shape[1] == 2:
            V = torch.cat([V, torch.zeros((V.shape[0], 1), dtype=V.dtype, device=V.device)], dim=1)
        d2 = ((P3[:, None, :] - V[None, :, :]) ** 2).sum(dim=2)
        return torch.sqrt(d2.min(dim=1).values)

    @staticmethod
    def _pointset_to_segments_distance(P: torch.Tensor, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        AB = B - A  # (M,3)
        AP = P[:, None, :] - A[None, :, :]  # (N,M,3)
        AB_len2 = (AB * AB).sum(dim=1).clamp_min(1e-30)  # (M,)
        t = (AP * AB[None, :, :]).sum(dim=2) / AB_len2[None, :]  # (N,M)
        t = torch.clamp(t, 0.0, 1.0)
        closest = A[None, :, :] + t[:, :, None] * AB[None, :, :]
        d2 = ((P[:, None, :] - closest) ** 2).sum(dim=2)
        return torch.sqrt(d2.min(dim=1).values)

    @staticmethod
    def _pointset_to_triangles_distance(P: torch.Tensor, A: torch.Tensor, B: torch.Tensor,
                                        C: torch.Tensor) -> torch.Tensor:
        # Ericson: 投影 + 内外判定 + 边距
        PA = P[:, None, :] - A[None, :, :]
        PB = P[:, None, :] - B[None, :, :]
        PC = P[:, None, :] - C[None, :, :]
        AB = B[None, :, :] - A[None, :, :]
        AC = C[None, :, :] - A[None, :, :]
        N = torch.cross(AB, AC, dim=2)
        N_len2 = (N * N).sum(dim=2).clamp_min(1e-30)
        dist_plane = ((PA * N).sum(dim=2)) / torch.sqrt(N_len2)  # (N,M) 符号距离大小
        proj = P[:, None, :] - ((PA * N).sum(dim=2) / N_len2)[:, :, None] * N
        C1 = torch.cross(AB, proj - A[None, :, :], dim=2)
        C2 = torch.cross(C[None, :, :] - B[None, :, :], proj - B[None, :, :], dim=2)
        C3 = torch.cross(A[None, :, :] - C[None, :, :], proj - C[None, :, :], dim=2)
        inside = ((C1 * N).sum(dim=2) >= 0) & ((C2 * N).sum(dim=2) >= 0) & ((C3 * N).sum(dim=2) >= 0)
        d_inside = torch.abs(dist_plane)
        d_edge_ab = GmshAdaptor._pointset_to_segments_distance(P, A, B)
        d_edge_bc = GmshAdaptor._pointset_to_segments_distance(P, B, C)
        d_edge_ca = GmshAdaptor._pointset_to_segments_distance(P, C, A)
        d_outside = torch.min(torch.min(d_edge_ab, d_edge_bc), d_edge_ca)
        d_full = torch.where(inside, d_inside, d_outside[:, None])
        return d_full.min(dim=1).values

    @staticmethod
    def _point_in_polygon_evenodd(P2: torch.Tensor,
                                  edges: torch.Tensor,
                                  V2: torch.Tensor,
                                  eps: float = 1e-12) -> torch.Tensor:
        """
        P2: (N,2), edges: (M,2) int tensor forming (possibly multiple) closed loops,
        V2: (V,2). 返回 bool 内点掩码（True 为内部），边上点也视为内部。
        采用：先“点在边上”检测；后 even-odd 射线计数（半开区间处理 & 数值缓冲）。
        """

        # ----- 端点坐标（保留原始副本，避免 where 连带修改） -----
        A0 = V2[edges[:, 0]]  # (M,2)
        B0 = V2[edges[:, 1]]  # (M,2)

        # 剔除零长边（可能来自错误拼接/顶点合并前的重复）
        good = (torch.linalg.norm(B0 - A0, dim=1) > eps)
        if not torch.all(good):
            A0 = A0[good]
            B0 = B0[good]
            if A0.shape[0] == 0:
                return torch.zeros(P2.shape[0], dtype=torch.bool, device=P2.device)

        # ----- 边上点检测（先于射线法；边上视为内部） -----
        # 距离点到线段：投影截断
        E = B0 - A0  # (M,2)
        EE = (E * E).sum(dim=1).clamp_min(eps)  # (M,)
        AP = P2[:, None, :] - A0[None, :, :]  # (N,M,2)
        t = (AP * E[None, :, :]).sum(dim=2) / EE[None, :]  # (N,M)
        t = t.clamp(0.0, 1.0)
        closest = A0[None, :, :] + t[:, :, None] * E[None, :, :]  # (N,M,2)
        dist2 = ((P2[:, None, :] - closest) ** 2).sum(dim=2)  # (N,M)
        on_edge = (dist2 <= (10.0 * eps) ** 2)  # 放大一点容差更稳
        on_any_edge = on_edge.any(dim=1)  # (N,)

        # ----- 射线法（even-odd），处理水平边与顶点双计数 -----
        # 交换使 A.y <= B.y ：注意使用原始 A0/B0 的拷贝
        swap = (A0[:, 1] > B0[:, 1]).unsqueeze(1)  # (M,1)
        A = torch.where(swap, B0, A0)  # (M,2)
        B = torch.where(swap, A0, B0)  # (M,2)

        # 半开区间：y ∈ [Ay, By) 避免上端点重复计数；再给一点 eps 缓冲
        py = P2[:, 1].unsqueeze(1)  # (N,1)
        Ay = A[:, 1].unsqueeze(0)  # (1,M)
        By = B[:, 1].unsqueeze(0)  # (1,M)
        cond_y = (py >= Ay - eps) & (py < By - eps)  # (N,M)

        # 计算交点 x 坐标 xi
        denom = (B[:, 1] - A[:, 1]).clamp_min(eps)  # (M,)
        denom = denom.unsqueeze(0)  # (1,M)
        Ax = A[:, 0].unsqueeze(0)  # (1,M)
        Bx = B[:, 0].unsqueeze(0)  # (1,M)
        xi = Ax + (py - Ay) * (Bx - Ax) / denom  # (N,M)

        # 交点在点的右侧（含少许容差）
        px = P2[:, 0].unsqueeze(1)  # (N,1)
        cross_right = xi > (px - eps)  # (N,M)

        hits = (cond_y & cross_right).sum(dim=1)  # (N,)
        inside_by_parity = (hits % 2 == 1)

        # 边上点直接归为内部
        return inside_by_parity | on_any_edge

    @staticmethod
    def _ray_intersect_count_px(P: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        # 固定射线方向 (1,0,0)
        dir = torch.tensor([1.0, 0.0, 0.0], dtype=P.dtype, device=P.device).view(1, 1, 3)
        O = P.unsqueeze(1)  # (N,1,3)
        A = A.unsqueeze(0)
        B = B.unsqueeze(0)
        C = C.unsqueeze(0)

        eps = 1e-12
        e1 = B - A
        e2 = C - A
        h = torch.cross(dir, e2, dim=2)
        a = (e1 * h).sum(dim=2)  # (N,M)
        mask = torch.abs(a) > eps

        f = torch.zeros_like(a)
        f[mask] = 1.0 / a[mask]
        s = O - A
        u = f * (s * h).sum(dim=2)
        mask = mask & (u >= 0.0) & (u <= 1.0)

        q = torch.cross(s, e1, dim=2)
        v = f * (dir * q).sum(dim=2)
        mask = mask & (v >= 0.0) & (u + v <= 1.0)

        t = f * (e2 * q).sum(dim=2)
        hit = mask & (t > eps)
        return hit.sum(dim=1)  # (N,)

    def _inside_mask(self, P3: torch.Tensor) -> torch.Tensor:
        if self.topo_dim == 2 and self._boundary_edges is not None and self._boundary_edges.size > 0:
            V2 = self._ensure_tensor(self.points[:, :2])
            E = torch.as_tensor(self._boundary_edges, dtype=torch.long, device=P3.device)
            return self._point_in_polygon_evenodd(P3[:, :2], E, V2)

        if self.topo_dim == 3 and self._boundary_tris is not None and self._is_closed_3d:
            tri = self._boundary_tris
            A = self._ensure_tensor(self.points[tri[:, 0]])
            B = self._ensure_tensor(self.points[tri[:, 1]])
            C = self._ensure_tensor(self.points[tri[:, 2]])
            if A.shape[1] == 2:
                z = torch.zeros((A.shape[0], 1), dtype=A.dtype, device=A.device)
                A = torch.cat([A, z], 1)
                B = torch.cat([B, z], 1)
                C = torch.cat([C, z], 1)
            hits = self._ray_intersect_count_px(P3, A, B, C)
            return (hits % 2 == 1)

        # 其他情形（开放曲面/未知）：无法定义“内外”
        return torch.zeros(P3.shape[0], dtype=torch.bool, device=P3.device)
