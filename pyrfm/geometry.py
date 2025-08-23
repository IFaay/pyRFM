# -*- coding: utf-8 -*-
"""
Created on 2024/12/15

@author: Yifei Sun
"""
import torch

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

    def glsl_sdf(self) -> str:
        """
        Return a GLSL expression (string) that evaluates the signed distance
        at a coordinate variable named `p` (float for 1‑D, vec2 for 2‑D,
        vec3 for 3‑D) which must be in scope inside the GLSL shader.
        The expression must be syntactically valid GLSL and reference only
        constants and the variable `p`.
        """
        pass

    @abstractmethod
    def get_bounding_box(self) -> List[float]:
        """
        Get the bounding box of the geometry.

        Returns:
        -------
        list
            For 2D: [x_min, x_max, y_min, y_max];
            For 3D: [x_min, x_max, y_min, y_max, z_min, z_max];
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

    # GLSL: empty space has effectively infinite distance
    def glsl_sdf(self) -> str:
        return "1e20"

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

    # GLSL expression for the union: min(dA,dB)
    def glsl_sdf(self) -> str:
        return f"min({self.geomA.glsl_sdf()}, {self.geomB.glsl_sdf()})"

    def get_bounding_box(self):
        boxA = self.geomA.get_bounding_box()
        boxB = self.geomB.get_bounding_box()
        return [min(boxA[i], boxB[i]) if i % 2 == 0 else max(boxA[i], boxB[i]) for i in range(2 * self.dim)]

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

    # GLSL expression for the intersection: max(dA,dB)
    def glsl_sdf(self) -> str:
        return f"max({self.geomA.glsl_sdf()}, {self.geomB.glsl_sdf()})"

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

    # GLSL expression for the complement: -d
    def glsl_sdf(self) -> str:
        return f"-({self.geom.glsl_sdf()})"

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
    def on_sample(self, num_samples: int, with_normal: bool = False, separate: bool = False) -> Union[
        torch.Tensor, Tuple[torch.Tensor, ...]]:
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

    # ------------------------------------------------------------------ #
    # GLSL SDF expression
    # ------------------------------------------------------------------ #
    def glsl_sdf(self) -> str:
        dx, dy, dz = [f"{x:.6f}" for x in self.d.tolist()]
        ux, uy, uz = [f"{x:.6f}" for x in self.u.tolist()]
        vx, vy, vz = [f"{x:.6f}" for x in self.v.tolist()]
        h = f"{self.h:.6f}"

        # project vec3 p → vec2 q
        proj = (f"vec2(dot(p, vec3({ux},{uy},{uz})), "
                f"dot(p, vec3({vx},{vy},{vz})))")
        base_expr = self.base.glsl_sdf().replace("p", proj)

        return f"max({base_expr}, abs(dot(p, vec3({dx},{dy},{dz}))) - {h})"


class ImplicitFunctionBase(GeometryBase):
    @abstractmethod
    def shape_func(self, p: torch.Tensor) -> torch.Tensor:
        pass

    def sdf(self, p: torch.Tensor, with_normal=False, with_curvature=False) -> Union[
        torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Memory-efficient computation of the normalized signed distance function (SDF),
        with optional normal and mean curvature computation.

        Args:
            p (torch.Tensor): Input point cloud of shape (N, 3)
            with_normal (bool): If True, also return normal vectors.
            with_curvature (bool): If True, also return mean curvature.

        Returns:
            Union of tensors depending on flags:
                - sdf
                - (sdf, normal)
                - (sdf, normal, mean_curvature)
        """
        p = p.detach().requires_grad_(True)  # Detach to avoid tracking history
        f = self.shape_func(p)

        # Compute gradient (∇f)
        grad = torch.autograd.grad(outputs=f, inputs=p, grad_outputs=torch.ones_like(f), create_graph=with_curvature,
                                   # Need graph for second-order derivative
                                   retain_graph=with_curvature, only_inputs=True)[0]

        grad_norm = torch.norm(grad, dim=-1, keepdim=True)
        sdf = f / grad_norm
        normal = grad / grad_norm

        if not (with_normal or with_curvature):
            return sdf.detach()
        elif with_normal and (not with_curvature):
            return sdf.detach(), normal.detach()
        else:
            divergence = 0.0
            for i in range(p.shape[-1]):  # Loop over x, y, z
                dni = torch.autograd.grad(outputs=normal[:, i], inputs=p, grad_outputs=torch.ones_like(normal[:, i]),
                                          create_graph=False, retain_graph=True, only_inputs=True)[0][:, [i]]
                divergence += dni

            mean_curvature = 0.5 * divergence  # H = ½ ∇·n
            return sdf.detach(), normal.detach(), mean_curvature.detach()


class ImplicitSurfaceBase(ImplicitFunctionBase):
    def __init__(self):
        super().__init__(dim=3, intrinsic_dim=2)

    @abstractmethod
    def shape_func(self, p: torch.Tensor) -> torch.Tensor:
        pass

    def in_sample(self, num_samples: int, with_boundary: bool = False) -> torch.Tensor:
        """
        Sample near-surface points by rejection and iterative projection.

        Args:
            num_samples (int): Number of samples to generate.
            with_boundary (bool): Ignored for implicit surfaces.

        Returns:
            torch.Tensor: Sampled points projected onto the surface.
        """
        x_min, x_max, y_min, y_max, z_min, z_max = self.get_bounding_box()
        volume = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
        resolution = (volume / num_samples) ** (1 / self.dim)
        eps = 2 * resolution

        collected = []
        max_iter = 10
        oversample = int(num_samples * 1.5)

        while sum(c.shape[0] for c in collected) < num_samples:
            rand = torch.rand(oversample, self.dim, device=self.device, generator=self.gen)
            p = torch.empty(oversample, self.dim, dtype=self.dtype, device=self.device)
            p[:, 0] = (x_min - eps) + rand[:, 0] * ((x_max + eps) - (x_min - eps))
            p[:, 1] = (y_min - eps) + rand[:, 1] * ((y_max + eps) - (y_min - eps))
            p[:, 2] = (z_min - eps) + rand[:, 2] * ((z_max + eps) - (z_min - eps))
            p.requires_grad_(True)
            f = self.shape_func(p)
            grad = torch.autograd.grad(f, p, torch.ones_like(f), create_graph=False)[0]
            grad_norm = grad.norm(dim=1, keepdim=True)
            normal = grad / grad_norm
            sdf = f / grad_norm
            near_mask = (sdf.abs() < eps).squeeze()
            near_points = p[near_mask]
            near_normals = normal[near_mask]
            near_sdf = sdf[near_mask]

            for _ in range(max_iter):
                if near_points.shape[0] == 0:
                    break
                near_points = near_points - near_sdf * near_normals
                near_points.requires_grad_(True)
                f_proj = self.shape_func(near_points)
                grad_proj = torch.autograd.grad(f_proj, near_points, torch.ones_like(f_proj), create_graph=False)[0]
                grad_norm_proj = grad_proj.norm(dim=1, keepdim=True)
                near_normals = grad_proj / grad_norm_proj
                near_sdf = f_proj / grad_norm_proj
                if near_sdf.abs().max().item() < torch.finfo(self.dtype).eps * resolution:
                    break

            collected.append(near_points.detach())

        return torch.cat(collected, dim=0)[:num_samples]

    def on_sample(self, num_samples: int, with_normal: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Implicit surfaces do not explicitly provide boundary samples.
        This method returns an empty tensor compatible with Boolean ops.

        Args:
            num_samples (int): Number of samples (ignored).
            with_normal (bool): Whether to include normals.

        Returns:
            torch.Tensor or Tuple[torch.Tensor, torch.Tensor]: Empty tensor(s).
        """
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

    def glsl_sdf(self) -> str:
        return f"abs(p - {float(self.x)})"

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

    def glsl_sdf(self) -> str:
        return f"length(p - vec2({float(self.x)}, {float(self.y)}))"

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

    def glsl_sdf(self) -> str:
        return f"length(p - vec3({float(self.x)}, {float(self.y)}, {float(self.z)}))"

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

    def glsl_sdf(self) -> str:
        mid = (float(self.x1) + float(self.x2)) * 0.5
        half = abs(float(self.x2) - float(self.x1)) * 0.5
        return f"abs(p - {mid}) - {half}"

    def get_bounding_box(self):
        """
        Get the bounding box of the line segment.

        Returns:
        -------
        list
            The bounding box of the line segment.
        """
        return [self.x1, self.x2] if self.x1 < self.x2 else [self.x2, self.x1]

    def in_sample(self, num_samples: int, with_boundary: bool = False) -> torch.Tensor:
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

    def glsl_sdf(self) -> str:
        return (f"sdSegment(p, vec2({float(self.x1)}, {float(self.y1)}), "
                f"vec2({float(self.x2)}, {float(self.y2)}))")

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

    def glsl_sdf(self) -> str:
        raise NotImplementedError("Line3D.glsl_sdf not yet implemented")

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
    def __init__(self, center: Union[torch.Tensor, List, Tuple], radius: Union[torch.Tensor, List, Tuple]):
        super().__init__(dim=2, intrinsic_dim=2)
        self.center = torch.tensor(center).view(1, -1)
        self.radius = torch.tensor(radius).view(1, -1)
        # Define the boundary of the square (bottom(y_min), right(x_max), top(y_max), left(x_min))
        self.boundary = [Line2D(self.center[0, 0] - self.radius[0, 0], self.center[0, 1] - self.radius[0, 1],
                                self.center[0, 0] + self.radius[0, 0], self.center[0, 1] - self.radius[0, 1]),
                         Line2D(self.center[0, 0] + self.radius[0, 0], self.center[0, 1] - self.radius[0, 1],
                                self.center[0, 0] + self.radius[0, 0], self.center[0, 1] + self.radius[0, 1]),
                         Line2D(self.center[0, 0] + self.radius[0, 0], self.center[0, 1] + self.radius[0, 1],
                                self.center[0, 0] - self.radius[0, 0], self.center[0, 1] + self.radius[0, 1]),
                         Line2D(self.center[0, 0] - self.radius[0, 0], self.center[0, 1] + self.radius[0, 1],
                                self.center[0, 0] - self.radius[0, 0], self.center[0, 1] - self.radius[0, 1])]

    def sdf(self, p: torch.Tensor):
        d = torch.abs(p - self.center) - self.radius
        return torch.norm(torch.clamp(d, min=0.0), dim=1, keepdim=True) + torch.clamp(
            torch.max(d, dim=1, keepdim=True).values, max=0.0)

    def glsl_sdf(self) -> str:
        cx, cy = map(float, self.center.squeeze())
        rx, ry = map(float, self.radius.squeeze())
        return ("length(max(abs(p - vec2({cx},{cy})) - vec2({rx},{ry}), 0.0))"
                "+ min(max(abs(p.x-{cx})-{rx}, abs(p.y-{cy})-{ry}), 0.0)").format(cx=cx, cy=cy, rx=rx, ry=ry)

    def get_bounding_box(self):
        x_min = self.center[0, 0] - self.radius[0, 0]
        x_max = self.center[0, 0] + self.radius[0, 0]
        y_min = self.center[0, 1] - self.radius[0, 1]
        y_max = self.center[0, 1] + self.radius[0, 1]
        return [x_min.item(), x_max.item(), y_min.item(), y_max.item()]

    def in_sample(self, num_samples: Union[int, List[int], Tuple[int, int]],
                  with_boundary: bool = False) -> torch.Tensor:
        if isinstance(num_samples, int):
            num_x = num_y = int(num_samples ** 0.5)
        elif isinstance(num_samples, (list, tuple)) and len(num_samples) == 2:
            num_x, num_y = int(num_samples[0]), int(num_samples[1])
        else:
            raise ValueError("num_samples must be an int or a list/tuple of two integers.")

        x_min, x_max = self.center[0, 0] - self.radius[0, 0], self.center[0, 0] + self.radius[0, 0]
        y_min, y_max = self.center[0, 1] - self.radius[0, 1], self.center[0, 1] + self.radius[0, 1]

        if with_boundary:
            x = torch.linspace(x_min, x_max, num_x)
            y = torch.linspace(y_min, y_max, num_y)
        else:
            x = torch.linspace(x_min, x_max, num_x + 2)[1:-1]
            y = torch.linspace(y_min, y_max, num_y + 2)[1:-1]

        X, Y = torch.meshgrid(x, y, indexing='ij')
        return torch.cat([X.reshape(-1, 1), Y.reshape(-1, 1)], dim=1)

    def on_sample(self, num_samples: Union[int, List[int], Tuple], with_normal: bool = False, separate: bool = False) -> \
            Union[torch.Tensor, Tuple[torch.Tensor, ...]]:

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
                normals = torch.cat([torch.tensor([[0.0, -1.0]] * nums[0]),  # bottom
                                     torch.tensor([[1.0, 0.0]] * nums[1]),  # right
                                     torch.tensor([[0.0, 1.0]] * nums[2]),  # top
                                     torch.tensor([[-1.0, 0.0]] * nums[3])  # left
                                     ], dim=0)
                return torch.cat([a, b, c, d], dim=0), normals
            else:
                return torch.cat([a, b, c, d], dim=0)
        else:
            if with_normal:
                return ((a, torch.tensor([[0.0, -1.0]] * nums[0])), (b, torch.tensor([[1.0, 0.0]] * nums[1])),
                        (c, torch.tensor([[0.0, 1.0]] * nums[2])), (d, torch.tensor([[-1.0, 0.0]] * nums[3])))
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

    def glsl_sdf(self) -> str:
        """
        Return a GLSL expression that computes the signed distance from `p`
        (a vec3 in shader scope) to this axis‑aligned cube.
        """
        cx, cy, cz = map(float, self.center.squeeze())
        rx, ry, rz = map(float, self.radius.squeeze())
        return ("length(max(abs(p - vec3({cx},{cy},{cz})) - vec3({rx},{ry},{rz}), 0.0))"
                "+ min(max(max(abs(p.x-{cx})-{rx}, abs(p.y-{cy})-{ry}), abs(p.z-{cz})-{rz}), 0.0)").format(cx=cx, cy=cy,
                                                                                                           cz=cz, rx=rx,
                                                                                                           ry=ry, rz=rz)

    def get_bounding_box(self):
        x_min = self.center[0, 0] - self.radius[0, 0]
        x_max = self.center[0, 0] + self.radius[0, 0]
        y_min = self.center[0, 1] - self.radius[0, 1]
        y_max = self.center[0, 1] + self.radius[0, 1]
        z_min = self.center[0, 2] - self.radius[0, 2]
        z_max = self.center[0, 2] + self.radius[0, 2]
        return [x_min.item(), x_max.item(), y_min.item(), y_max.item(), z_min.item(), z_max.item()]

    def in_sample(self, num_samples: int, with_boundary: bool = False) -> torch.Tensor:
        # FIXME: wrong use with meshgrid
        num_samples = int(num_samples ** (1 / 2))
        if with_boundary:
            x = torch.linspace(self.center[0, 0] - self.radius[0, 0], self.center[0, 0] + self.radius[0, 0],
                               num_samples)
            y = torch.linspace(self.center[0, 1] - self.radius[0, 1], self.center[0, 1] + self.radius[0, 1],
                               num_samples)
            z = torch.linspace(self.center[0, 2] - self.radius[0, 2], self.center[0, 2] + self.radius[0, 2],
                               num_samples)
            X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
            return torch.cat([X.reshape(-1, 1), Y.reshape(-1, 1), Z.reshape(-1, 1)], dim=1)
        else:
            x = torch.linspace(self.center[0, 0] - self.radius[0, 0], self.center[0, 0] + self.radius[0, 0],
                               num_samples + 2)[1:-1]
            y = torch.linspace(self.center[0, 1] - self.radius[0, 1], self.center[0, 1] + self.radius[0, 1],
                               num_samples + 2)[1:-1]
            z = torch.linspace(self.center[0, 2] - self.radius[0, 2], self.center[0, 2] + self.radius[0, 2],
                               num_samples + 2)[1:-1]
            X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
            return torch.cat([X.reshape(-1, 1), Y.reshape(-1, 1), Z.reshape(-1, 1)], dim=1)

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
    def __init__(self, center: Union[torch.Tensor, List, Tuple], radius: Union[torch.Tensor, List, Tuple]):
        super().__init__(dim=3, intrinsic_dim=3)
        self.center = torch.tensor(center).view(1, -1).to(dtype=self.dtype)
        self.radius = torch.tensor(radius).view(1, -1).to(dtype=self.dtype)
        offsets = [[self.radius[0, 0], 0.0, 0.0], [-self.radius[0, 0], 0.0, 0.0], [0.0, self.radius[0, 1], 0.0],
                   [0.0, -self.radius[0, 1], 0.0], [0.0, 0.0, self.radius[0, 2]], [0.0, 0.0, -self.radius[0, 2]]]
        self.boundary = [Square3D(self.center + torch.tensor(offset),
                                  torch.tensor([self.radius[0, i] if offset[i] == 0.0 else 0.0 for i in range(3)])) for
                         offset in offsets]

    def sdf(self, p: torch.Tensor):
        d = torch.abs(p - self.center) - self.radius
        return torch.norm(torch.clamp(d, min=0.0), dim=1, keepdim=True) + torch.clamp(
            torch.max(d, dim=1, keepdim=True).values, max=0.0)

    def get_bounding_box(self):
        x_min = self.center[0, 0] - self.radius[0, 0]
        x_max = self.center[0, 0] + self.radius[0, 0]
        y_min = self.center[0, 1] - self.radius[0, 1]
        y_max = self.center[0, 1] + self.radius[0, 1]
        z_min = self.center[0, 2] - self.radius[0, 2]
        z_max = self.center[0, 2] + self.radius[0, 2]
        return [x_min.item(), x_max.item(), y_min.item(), y_max.item(), z_min.item(), z_max.item()]

    def in_sample(self, num_samples: Union[int, List[int], Tuple[int, int, int]],
                  with_boundary: bool = False) -> torch.Tensor:
        if isinstance(num_samples, int):
            num_x = num_y = num_z = int(round(num_samples ** (1 / 3)))
        elif isinstance(num_samples, (list, tuple)) and len(num_samples) == 3:
            num_x, num_y, num_z = map(int, num_samples)
        else:
            raise ValueError("num_samples must be an int or a list/tuple of three integers.")

        x_min, x_max = self.center[0, 0] - self.radius[0, 0], self.center[0, 0] + self.radius[0, 0]
        y_min, y_max = self.center[0, 1] - self.radius[0, 1], self.center[0, 1] + self.radius[0, 1]
        z_min, z_max = self.center[0, 2] - self.radius[0, 2], self.center[0, 2] + self.radius[0, 2]

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

    def on_sample(self, num_samples: int, with_normal: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        samples = []
        for square in self.boundary:
            samples.append(square.in_sample(num_samples // 6, with_boundary=True))
        if with_normal:
            normals = []
            for i in range(6):
                normal = torch.zeros((num_samples // 6, 3))
                normal[:, i // 2] = 1.0 if i % 2 == 0 else -1.0
                normals.append(normal)
            return torch.cat(samples, dim=0), torch.cat(normals, dim=0)
        else:
            return torch.cat(samples, dim=0)


class CircleArc2D(GeometryBase):
    def __init__(self, center: Union[torch.Tensor, List, Tuple], radius: torch.float64):
        super().__init__(dim=2, intrinsic_dim=1)
        self.center = torch.tensor(center).view(1, -1) if not isinstance(center, torch.Tensor) else center
        self.radius = radius
        self.boundary = [Point2D(self.center[0, 0] + self.radius, self.center[0, 1])]

    def sdf(self, p: torch.Tensor):
        d = torch.norm(p - self.center, dim=1, keepdim=True) - self.radius
        return torch.abs(d)

    def glsl_sdf(self) -> str:
        raise NotImplementedError("CircleArc2D.glsl_sdf not yet implemented")

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

    def glsl_sdf(self) -> str:
        cx, cy = map(float, self.center.squeeze())
        r = float(self.radius)
        return f"length(p - vec2({cx}, {cy})) - {r}"

    def get_bounding_box(self):
        x_min = self.center[0, 0] - self.radius
        x_max = self.center[0, 0] + self.radius
        y_min = self.center[0, 1] - self.radius
        y_max = self.center[0, 1] + self.radius
        return [x_min.item(), x_max.item(), y_min.item(), y_max.item()]

    def in_sample(self, num_samples: int, with_boundary: bool = False) -> torch.Tensor:
        num_samples = int(num_samples ** (1 / 2))
        if with_boundary:
            r = torch.linspace(0.0, self.radius, num_samples)[1:]
        else:
            r = torch.linspace(0.0, self.radius, num_samples + 1)[1:-1]

        theta = torch.linspace(0.0, 2 * torch.pi, num_samples + 1)[:-1]
        R, T = torch.meshgrid(r, theta, indexing='ij')
        x = self.center[0, 0] + R * torch.cos(T)
        y = self.center[0, 1] + R * torch.sin(T)
        x = torch.cat([self.center[0, 0].view(1, 1), x.reshape(-1, 1)], dim=0)
        y = torch.cat([self.center[0, 1].view(1, 1), y.reshape(-1, 1)], dim=0)
        return torch.cat([x.reshape(-1, 1), y.reshape(-1, 1)], dim=1)

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

    def glsl_sdf(self) -> str:
        cx, cy, cz = map(float, self.center.squeeze())
        r = float(self.radius)
        return f"length(p - vec3({cx}, {cy}, {cz})) - {r}"

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

    def glsl_sdf(self) -> str:
        cx, cy, cz = map(float, self.center.squeeze())
        r = float(self.radius)
        return f"length(p - vec3({cx}, {cy}, {cz})) - {r}"

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
    def glsl_sdf(self) -> str:
        raise NotImplementedError("Polygon2D.glsl_sdf not yet implemented")

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
    def glsl_sdf(self) -> str:
        raise NotImplementedError("Polygon3D.glsl_sdf not yet implemented")

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

    def glsl_sdf(self) -> str:
        raise NotImplementedError("HyperCube.glsl_sdf not yet implemented")
