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

    This class defines the common interface for all geometry primitives and
    composite geometries (e.g., union, intersection, complement).

    Attributes:
        dim (int): Ambient dimension of the geometry (e.g., 2 or 3).
        intrinsic_dim (int): Intrinsic/topological dimension of the geometry.
        boundary (list): List of boundary components.
        device (torch.device): Device on which tensors are allocated.
        dtype (torch.dtype): Default tensor dtype.
        gen (torch.Generator): Random number generator used for sampling.
    """

    def __init__(
            self,
            dim: Optional[int] = None,
            intrinsic_dim: Optional[int] = None,
            seed: int = 100,
    ):
        """
        Initialize a GeometryBase instance.

        Args:
            dim (int, optional): Ambient dimension of the geometry.
                Defaults to 0 if not provided.
            intrinsic_dim (int, optional): Intrinsic dimension of the geometry.
                Defaults to `dim` if not provided.
            seed (int): Random seed used to initialize the internal
                torch.Generator.
        """
        self.dim = dim if dim is not None else 0
        self.dtype = torch.tensor(0.0).dtype
        self.device = torch.tensor(0.0).device

        self.intrinsic_dim = intrinsic_dim if intrinsic_dim is not None else dim

        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(seed)

        self.boundary: List = []

    def __eq__(self, other):
        """
        Check structural equality between two geometry objects.

        Two geometries are considered equal if:
        - They are instances of the same class
        - They have the same ambient and intrinsic dimensions
        - Their boundary components match (up to ordering)

        Args:
            other (GeometryBase): Another geometry object.

        Returns:
            bool: True if the two geometries are equivalent, False otherwise.
        """
        if not isinstance(other, self.__class__):
            return False

        if self.dim != other.dim or self.intrinsic_dim != other.intrinsic_dim:
            return False

        if len(self.boundary) != len(other.boundary):
            return False

        if Counter(self.boundary) != Counter(other.boundary):
            return False

        return True

    @abstractmethod
    def sdf(self, p: torch.Tensor) -> torch.Tensor:
        """
        Compute the signed distance function (SDF) at given points.

        The signed distance is defined as:
        - Negative inside the geometry
        - Zero on the boundary
        - Positive outside the geometry

        Args:
            p (torch.Tensor): Query points.

        Shape:
            - p: (N, dim)
            - return: (N,)

        Returns:
            torch.Tensor: Signed distances for each input point.
        """
        pass

    @abstractmethod
    def get_bounding_box(self) -> List[float]:
        """
        Return the axis-aligned bounding box of the geometry.

        Returns:
            list[float]:
                - 2D: [x_min, x_max, y_min, y_max]
                - 3D: [x_min, x_max, y_min, y_max, z_min, z_max]
        """
        pass

    @abstractmethod
    def in_sample(
            self,
            num_samples: int,
            with_boundary: bool = False,
    ) -> torch.Tensor:
        """
        Sample points inside the geometry.

        Args:
            num_samples (int): Number of points to sample.
            with_boundary (bool): If True, boundary points may be included.

        Shape:
            - return: (N, dim)

        Returns:
            torch.Tensor: Sampled points inside the geometry.
        """
        pass

    @abstractmethod
    def on_sample(
            self,
            num_samples: int,
            with_normal: bool = False,
            separate: bool = False,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, ...],
        Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
    ]:
        """
        Sample points on the boundary of the geometry.

        Args:
            num_samples (int): Target number of boundary samples.
            with_normal (bool): If True, also return outward normal vectors.
            separate (bool): If True, return each boundary component separately.

        Returns:
            If separate is False:
                - with_normal = False:
                    Tensor of shape (N, dim)
                - with_normal = True:
                    Tuple (points, normals), each of shape (N, dim)

            If separate is True:
                - with_normal = False:
                    Tuple of tensors (points_i,), each (Ni, dim)
                - with_normal = True:
                    Tuple of (points_i, normals_i) per boundary component

        Notes:
            For composite geometries (union, intersection, complement),
            multiple boundary components may be returned when `separate=True`.
        """
        pass

    def __and__(self, other: "GeometryBase") -> "GeometryBase":
        """
        Compute the intersection of two geometries.

        Args:
            other (GeometryBase): Another geometry.

        Returns:
            GeometryBase: Intersection geometry.
        """
        return IntersectionGeometry(self, other)

    def __or__(self, other: "GeometryBase") -> "GeometryBase":
        """
        Compute the union of two geometries.

        Args:
            other (GeometryBase): Another geometry.

        Returns:
            GeometryBase: Union geometry.
        """
        return UnionGeometry(self, other)

    def __invert__(self) -> "GeometryBase":
        """
        Compute the complement of the geometry.

        Returns:
            GeometryBase: Complement geometry.
        """
        return ComplementGeometry(self)

    def __add__(self, other: "GeometryBase") -> "GeometryBase":
        """
        Alias for geometry union.

        Returns:
            GeometryBase: Union geometry.
        """
        if isinstance(other, EmptyGeometry):
            return self
        return UnionGeometry(self, other)

    def __sub__(self, other: "GeometryBase") -> "GeometryBase":
        """
        Compute the geometric difference: self \\ other.

        Returns:
            GeometryBase: Difference geometry.
        """
        if isinstance(other, EmptyGeometry):
            return self
        return IntersectionGeometry(self, ComplementGeometry(other))

    def __radd__(self, other: "GeometryBase") -> "GeometryBase":
        """
        Right-addition to support built-in sum().

        Returns:
            GeometryBase: Union geometry.
        """
        return self.__add__(other)


class EmptyGeometry(GeometryBase):
    """
    Geometry representing the empty set.

    This geometry contains no interior points and no boundary.
    It acts as the identity element for geometric union operations.
    """

    def __init__(self):
        """
        Initialize an empty geometry.

        Notes:
            - Both ambient and intrinsic dimensions are set to zero.
            - The boundary is always empty.
        """
        super().__init__(dim=0, intrinsic_dim=0)
        self.boundary = []

    def sdf(self, p: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the signed distance function for the empty geometry.

        For the empty geometry, the signed distance is defined as +infinity
        everywhere.

        Args:
            p (torch.Tensor): Query points.

        Shape:
            - p: (..., dim)
            - return: same shape as `p`

        Returns:
            torch.Tensor: Tensor filled with positive infinity.
        """
        return torch.full_like(p, float("inf"))

    def get_bounding_box(self) -> List[float]:
        """
        Return the bounding box of the empty geometry.

        Returns:
            list[float]: An empty list, since no bounding box exists.
        """
        return []

    def in_sample(
            self,
            num_samples: int,
            with_boundary: bool = False,
    ) -> torch.Tensor:
        """
        Sample points inside the empty geometry.

        Since the geometry is empty, no valid interior samples exist.

        Args:
            num_samples (int): Number of samples requested.
            with_boundary (bool): Ignored for empty geometry.

        Shape:
            - return: (num_samples, 0)

        Returns:
            torch.Tensor: Empty tensor with zero spatial dimension.
        """
        return torch.empty((num_samples, 0), dtype=self.dtype, device=self.device)

    def on_sample(
            self,
            num_samples: int,
            with_normal: bool = False,
            separate: bool = False,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, ...],
        Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
    ]:
        """
        Sample points on the boundary of the empty geometry.

        The empty geometry has no boundary. This method therefore returns
        empty tensors with consistent shapes.

        Args:
            num_samples (int): Number of samples requested.
            with_normal (bool): If True, also return empty normal tensors.
            separate (bool): If True, wrap outputs in tuples.

        Returns:
            If separate is False:
                - with_normal = False:
                    Tensor of shape (num_samples, dim)
                - with_normal = True:
                    Tuple (points, normals), both empty

            If separate is True:
                - with_normal = False:
                    Tuple containing a single empty tensor
                - with_normal = True:
                    Tuple containing a single (points, normals) pair
        """
        pts = torch.empty(
            (num_samples, self.dim), dtype=self.dtype, device=self.device
        )

        if with_normal:
            normals = torch.empty_like(pts)
            if separate:
                return (pts, normals),
            return pts, normals
        else:
            if separate:
                return (pts,)
            return pts

    def __eq__(self, other) -> bool:
        """
        Check equality with another geometry.

        Args:
            other (GeometryBase): Another geometry.

        Returns:
            bool: True if `other` is also an EmptyGeometry.
        """
        return isinstance(other, EmptyGeometry)

    def __add__(self, other: "GeometryBase") -> "GeometryBase":
        """
        Union with another geometry.

        The empty geometry is the identity element for union.

        Returns:
            GeometryBase: The other geometry.
        """
        return other

    def __or__(self, other: "GeometryBase") -> "GeometryBase":
        """
        Union with another geometry.

        The empty geometry is the identity element for union.

        Returns:
            GeometryBase: The other geometry.
        """
        return other

    def __invert__(self) -> "GeometryBase":
        """
        Compute the complement of the empty geometry.

        Returns:
            GeometryBase: Geometry representing the entire space.
        """
        return ComplementGeometry(self)


class UnionGeometry(GeometryBase):
    """
    Geometry representing the union of two geometries.

    The signed distance function (SDF) of the union is defined as:
        sdf(p) = min(sdf_A(p), sdf_B(p))

    This class supports interior sampling and boundary sampling, including
    optional separation of boundary components.
    """

    def __init__(self, geomA: GeometryBase, geomB: GeometryBase):
        """
        Initialize a union geometry.

        Args:
            geomA (GeometryBase): First geometry.
            geomB (GeometryBase): Second geometry.

        Notes:
            - The ambient and intrinsic dimensions are inherited from `geomA`.
            - The boundary list is the concatenation of the boundaries of
              both sub-geometries.
        """
        super().__init__()
        self.geomA = geomA
        self.geomB = geomB
        self.dim = geomA.dim
        self.intrinsic_dim = geomA.intrinsic_dim
        self.boundary = [*geomA.boundary, *geomB.boundary]

    def sdf(self, p: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the signed distance function of the union.

        Args:
            p (torch.Tensor): Query points.

        Shape:
            - p: (N, dim)
            - return: (N,)

        Returns:
            torch.Tensor: Signed distances to the union geometry.
        """
        return torch.min(self.geomA.sdf(p), self.geomB.sdf(p))

    def get_bounding_box(self) -> List[float]:
        """
        Return the axis-aligned bounding box of the union geometry.

        The bounding box is computed as the element-wise union of the
        bounding boxes of the two sub-geometries.

        Returns:
            list[float]:
                Axis-aligned bounding box with length 2 * dim.
        """
        boxA = self.geomA.get_bounding_box()
        boxB = self.geomB.get_bounding_box()
        return [
            min(boxA[i], boxB[i]) if i % 2 == 0 else max(boxA[i], boxB[i])
            for i in range(2 * self.dim)
        ]

    def in_sample(
            self,
            num_samples: int,
            with_boundary: bool = False,
    ) -> torch.Tensor:
        """
        Sample points inside the union geometry.

        Sampling is performed by:
        1. Estimating relative volumes of the bounding boxes
        2. Allocating samples proportionally to each sub-geometry
        3. Concatenating samples
        4. Filtering by the union SDF

        Args:
            num_samples (int): Target number of samples.
            with_boundary (bool): If True, allow points on the boundary.

        Shape:
            - return: (N, dim)

        Returns:
            torch.Tensor: Sampled points inside the union geometry.

        Notes:
            If filtering removes all samples, the unfiltered samples
            are returned as a fallback to avoid empty outputs.
        """
        boxA = self.geomA.get_bounding_box()
        boxB = self.geomB.get_bounding_box()

        VA, VB = 1.0, 1.0
        dim = len(boxA) // 2
        for i in range(dim):
            VA *= max(0.0, boxA[2 * i + 1] - boxA[2 * i])
            VB *= max(0.0, boxB[2 * i + 1] - boxB[2 * i])

        # Sampling ratio based on bounding-box volume estimate
        r = min(2.0, max(0.5, VA / (VB + 1e-12)))

        # Allocate samples
        NA = max(5, int(num_samples * r / (1.0 + r)))
        NB = max(5, num_samples - NA)

        # Sample from sub-geometries
        a = self.geomA.in_sample(NA, with_boundary)
        b = self.geomB.in_sample(NB, with_boundary)
        samples = torch.cat([a, b], dim=0)

        # Filter by union SDF
        if with_boundary:
            mask = (self.sdf(samples) <= 0).squeeze()
        else:
            mask = (self.sdf(samples) < 0).squeeze()

        filtered = samples[mask]

        # Fallback: avoid returning empty tensors
        if filtered.shape[0] == 0:
            return samples

        return filtered

    def on_sample(
            self,
            num_samples: int,
            with_normal: bool = False,
            separate: bool = False,
    ):
        """
        Sample points on the boundary of the union geometry.

        Boundary sampling follows these principles:
        - Samples are first drawn from each sub-geometry.
        - All samples are filtered using the union SDF (sdf == 0).
        - Optional separation preserves original boundary components.

        Args:
            num_samples (int): Target number of boundary samples.
            with_normal (bool): If True, also return outward normal vectors.
            separate (bool): If True, return boundary components separately.

        Returns:
            If separate is False:
                - with_normal = False:
                    Tensor of shape (N, dim)
                - with_normal = True:
                    Tuple (points, normals), both (N, dim)

            If separate is True:
                - with_normal = False:
                    List of tensors, each (Ni, dim)
                - with_normal = True:
                    List of (points_i, normals_i)

        Notes:
            When `separate=True`, boundary components that are fully
            filtered out by the union SDF are discarded.
        """
        boxA = self.geomA.get_bounding_box()
        boxB = self.geomB.get_bounding_box()

        VA, VB = 1.0, 1.0
        dim = len(boxA) // 2
        for i in range(dim):
            VA *= max(0.0, boxA[2 * i + 1] - boxA[2 * i])
            VB *= max(0.0, boxB[2 * i + 1] - boxB[2 * i])

        # Sampling ratio
        r = min(2.0, max(0.5, VA / (VB + 1e-12)))

        # Allocate samples
        NA = max(5, int(num_samples * r / (1.0 + r)))
        NB = max(5, num_samples - NA)

        # =========================================================
        # Case 1: separate == False (flat output)
        # =========================================================
        if not separate:
            if with_normal:
                a, an = self.geomA.on_sample(NA, with_normal=True, separate=False)
                b, bn = self.geomB.on_sample(NB, with_normal=True, separate=False)

                samples = torch.cat([a, b], dim=0)
                normals = torch.cat([an, bn], dim=0)

                mask = torch.isclose(
                    self.sdf(samples),
                    torch.tensor(0.0, device=samples.device),
                )

                if mask.sum() == 0:
                    return samples, normals

                mask = mask.flatten()
                return samples[mask], normals[mask]

            else:
                a = self.geomA.on_sample(NA, with_normal=False, separate=False)
                b = self.geomB.on_sample(NB, with_normal=False, separate=False)

                samples = torch.cat([a, b], dim=0)

                mask = torch.isclose(
                    self.sdf(samples),
                    torch.tensor(0.0, device=samples.device),
                )

                if mask.sum() == 0:
                    return samples

                mask = mask.flatten()
                return samples[mask]

        # =========================================================
        # Case 2: separate == True (preserve boundary components)
        # =========================================================
        if with_normal:
            groups_A = self.geomA.on_sample(NA, with_normal=True, separate=True)
            groups_B = self.geomB.on_sample(NB, with_normal=True, separate=True)

            all_groups = list(groups_A) + list(groups_B)
            if len(all_groups) == 0:
                return []

            pts_list = [g[0] for g in all_groups]
            nrm_list = [g[1] for g in all_groups]

            all_pts = torch.cat(pts_list, dim=0)
            all_nrms = torch.cat(nrm_list, dim=0)

            mask_all = torch.isclose(
                self.sdf(all_pts),
                torch.tensor(0.0, device=all_pts.device),
            ).flatten()

            filtered_groups = []
            start = 0
            for pts, nrms in zip(pts_list, nrm_list):
                n = pts.shape[0]
                submask = mask_all[start:start + n]
                start += n

                if submask.any():
                    filtered_groups.append((pts[submask], nrms[submask]))

            return filtered_groups

        else:
            groups_A = self.geomA.on_sample(NA, with_normal=False, separate=True)
            groups_B = self.geomB.on_sample(NB, with_normal=False, separate=True)

            all_groups = list(groups_A) + list(groups_B)
            if len(all_groups) == 0:
                return []

            all_pts = torch.cat(all_groups, dim=0)

            mask_all = torch.isclose(
                self.sdf(all_pts),
                torch.tensor(0.0, device=all_pts.device),
            ).flatten()

            filtered_groups = []
            start = 0
            for pts in all_groups:
                n = pts.shape[0]
                submask = mask_all[start:start + n]
                start += n

                if submask.any():
                    filtered_groups.append(pts[submask])

            return filtered_groups


class IntersectionGeometry(GeometryBase):
    """
    Geometry representing the intersection of two geometries.

    The signed distance function (SDF) of the intersection is defined as:
        sdf(p) = max(sdf_A(p), sdf_B(p))

    This class also supports set difference operations implicitly via
    intersections with complement geometries:
        A \\ B = Intersection(A, Complement(B))
    """

    def __init__(self, geomA: GeometryBase, geomB: GeometryBase):
        """
        Initialize an intersection geometry.

        Args:
            geomA (GeometryBase): First geometry.
            geomB (GeometryBase): Second geometry.

        Raises:
            ValueError: If ambient or intrinsic dimensions do not match.

        Notes:
            - The ambient and intrinsic dimensions are inherited from `geomA`.
            - The boundary list is the concatenation of both sub-geometries'
              boundaries.
        """
        super().__init__()

        if geomA.dim != geomB.dim:
            raise ValueError("The dimensions of the two geometries must be equal.")
        if geomA.intrinsic_dim != geomB.intrinsic_dim:
            raise ValueError(
                "The intrinsic dimensions of the two geometries must be equal."
            )

        self.geomA = geomA
        self.geomB = geomB
        self.dim = geomA.dim
        self.intrinsic_dim = geomA.intrinsic_dim
        self.boundary = [*geomA.boundary, *geomB.boundary]

    def sdf(self, p: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the signed distance function of the intersection.

        Args:
            p (torch.Tensor): Query points.

        Shape:
            - p: (N, dim)
            - return: (N,)

        Returns:
            torch.Tensor: Signed distances to the intersection geometry.
        """
        return torch.max(self.geomA.sdf(p), self.geomB.sdf(p))

    def get_bounding_box(self) -> List[float]:
        """
        Return the axis-aligned bounding box of the intersection.

        The bounding box is computed as the overlap of the two
        sub-geometries' bounding boxes.

        Returns:
            list[float]: Axis-aligned bounding box with length 2 * dim.
        """
        boxA = self.geomA.get_bounding_box()
        boxB = self.geomB.get_bounding_box()
        return [
            max(boxA[i], boxB[i]) if i % 2 == 0 else min(boxA[i], boxB[i])
            for i in range(2 * self.dim)
        ]

    def in_sample(
            self,
            num_samples: int,
            with_boundary: bool = False,
    ) -> torch.Tensor:
        """
        Sample points inside the intersection geometry.

        Sampling is performed by:
        1. Sampling from each sub-geometry
        2. Concatenating samples
        3. Filtering using the intersection SDF

        Args:
            num_samples (int): Number of samples per sub-geometry.
            with_boundary (bool): If True, include boundary points.

        Shape:
            - return: (N, dim)

        Returns:
            torch.Tensor: Sampled interior points.
        """
        samples = torch.cat(
            [
                self.geomA.in_sample(num_samples, with_boundary),
                self.geomB.in_sample(num_samples, with_boundary),
            ],
            dim=0,
        )

        if with_boundary:
            return samples[(self.sdf(samples) <= 0).squeeze()]

        return samples[(self.sdf(samples) < 0).squeeze()]

    def on_sample(
            self,
            num_samples: int,
            with_normal: bool = False,
            separate: bool = False,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
        List[torch.Tensor],
        List[Tuple[torch.Tensor, torch.Tensor]],
    ]:
        """
        Sample points on the boundary of the intersection geometry.

        Boundary sampling supports both standard intersections and
        set-difference cases (A \\ B).

        Args:
            num_samples (int): Target number of boundary samples.
            with_normal (bool): If True, also return outward normal vectors.
            separate (bool): If True, preserve boundary components.

        Returns:
            If separate is False:
                - with_normal = False:
                    Tensor of shape (N, dim)
                - with_normal = True:
                    Tuple (points, normals)

            If separate is True:
                - with_normal = False:
                    List of tensors, each (Ni, dim)
                - with_normal = True:
                    List of (points_i, normals_i)

        Notes:
            - For standard intersections, boundary points satisfy sdf == 0.
            - For set difference A \\ B, boundary points are selected from:
                * boundary of A outside B
                * boundary of B inside A
            - Boundary components that are fully filtered out are discarded
              when `separate=True`.
        """

        # --------------------------------------------------
        # Allocate samples based on bounding-box volume ratio
        # --------------------------------------------------
        boxA = self.geomA.get_bounding_box()
        boxB = self.geomB.get_bounding_box()

        VA, VB = 1.0, 1.0
        dim = len(boxA) // 2
        for i in range(dim):
            VA *= max(0.0, boxA[2 * i + 1] - boxA[2 * i])
            VB *= max(0.0, boxB[2 * i + 1] - boxB[2 * i])

        r = min(2.0, max(0.5, VA / (VB + 1e-12)))
        NA = max(5, int(num_samples * r / (1.0 + r)))
        NB = max(5, num_samples - NA)

        # --------------------------------------------------
        # Detect set-difference scenario:
        #   A \\ B = Intersection(A, Complement(B))
        # --------------------------------------------------
        is_comp_A = isinstance(self.geomA, ComplementGeometry)
        is_comp_B = isinstance(self.geomB, ComplementGeometry)
        is_difference = is_comp_A ^ is_comp_B

        if is_difference:
            if is_comp_B:
                A_geom = self.geomA
                B_geom = self.geomB.geom
            else:
                A_geom = self.geomB
                B_geom = self.geomA.geom
        else:
            A_geom = self.geomA
            B_geom = self.geomB

        def zero_like(x: torch.Tensor) -> torch.Tensor:
            return torch.zeros(1, dtype=x.dtype, device=x.device).squeeze()

        # ==================================================
        # Case 1: separate == False (flat output)
        # ==================================================
        if not separate:
            if with_normal:
                a_pts, a_nrms = self.geomA.on_sample(
                    NA, with_normal=True, separate=False
                )
                b_pts, b_nrms = self.geomB.on_sample(
                    NB, with_normal=True, separate=False
                )
                samples = torch.cat([a_pts, b_pts], dim=0)
                normals = torch.cat([a_nrms, b_nrms], dim=0)
            else:
                samples = torch.cat(
                    [
                        self.geomA.on_sample(NA, False, False),
                        self.geomB.on_sample(NB, False, False),
                    ],
                    dim=0,
                )
                normals = None

            if samples.numel() == 0:
                return (samples, normals) if with_normal else samples

            if not is_difference:
                mask = torch.isclose(
                    self.sdf(samples), zero_like(samples[..., 0])
                ).flatten()
            else:
                dA = A_geom.sdf(samples)
                dB = B_geom.sdf(samples)

                eps0 = 1e-3
                eps1 = 1e-4

                mask_from_A = torch.isclose(dA, zero_like(dA), atol=eps0) & (dB > eps1)
                mask_from_B = torch.isclose(dB, zero_like(dB), atol=eps0) & (dA < -eps1)

                mask = (mask_from_A | mask_from_B).flatten()

            if mask.sum() == 0:
                if with_normal:
                    return (
                        samples.new_zeros((0, samples.shape[1])),
                        normals.new_zeros((0, normals.shape[1])),
                    )
                return samples.new_zeros((0, samples.shape[1]))

            return (samples[mask], normals[mask]) if with_normal else samples[mask]

        # ==================================================
        # Case 2: separate == True (preserve components)
        # ==================================================
        if with_normal:
            groups_A = self.geomA.on_sample(NA, True, True)
            groups_B = self.geomB.on_sample(NB, True, True)
            all_groups = list(groups_A) + list(groups_B)

            if not all_groups:
                return []

            pts_list = [g[0] for g in all_groups]
            nrm_list = [g[1] for g in all_groups]

            all_pts = torch.cat(pts_list, dim=0)

            if not is_difference:
                mask_all = torch.isclose(
                    self.sdf(all_pts), zero_like(all_pts[..., 0])
                ).flatten()
            else:
                dA = A_geom.sdf(all_pts)
                dB = B_geom.sdf(all_pts)

                eps0 = 1e-1 / NA
                eps1 = 1e-1 / NB

                mask_all = (
                                   torch.isclose(dA, zero_like(dA), atol=eps0) & (dB > eps1)
                           ) | (
                                   torch.isclose(dB, zero_like(dB), atol=eps0) & (dA < -eps1)
                           )
                mask_all = mask_all.flatten()

            filtered = []
            start = 0
            for pts, nrms in zip(pts_list, nrm_list):
                n = pts.shape[0]
                submask = mask_all[start:start + n]
                start += n
                if submask.any():
                    filtered.append((pts[submask], nrms[submask]))

            return filtered

        else:
            groups_A = self.geomA.on_sample(NA, False, True)
            groups_B = self.geomB.on_sample(NB, False, True)
            pts_list = list(groups_A) + list(groups_B)

            if not pts_list:
                return []

            all_pts = torch.cat(pts_list, dim=0)

            if not is_difference:
                mask_all = torch.isclose(
                    self.sdf(all_pts), zero_like(all_pts[..., 0])
                ).flatten()
            else:
                dA = A_geom.sdf(all_pts)
                dB = B_geom.sdf(all_pts)

                eps0 = 1e-1 / NA
                eps1 = 1e-1 / NB

                mask_all = (
                                   torch.isclose(dA, zero_like(dA), atol=eps0) & (dB > eps1)
                           ) | (
                                   torch.isclose(dB, zero_like(dB), atol=eps0) & (dA < -eps1)
                           )
                mask_all = mask_all.flatten()

            filtered = []
            start = 0
            for pts in pts_list:
                n = pts.shape[0]
                submask = mask_all[start:start + n]
                start += n
                if submask.any():
                    filtered.append(pts[submask])

            return filtered


class ComplementGeometry(GeometryBase):
    """
    Geometry representing the complement of a given geometry.

    The complement geometry contains all points that are outside the
    original geometry. Its signed distance function (SDF) is defined as
    the negation of the original SDF:
        sdf_complement(p) = -sdf_original(p)
    """

    def __init__(self, geom: GeometryBase):
        """
        Initialize a complement geometry.

        Args:
            geom (GeometryBase): The geometry to be complemented.

        Notes:
            - Ambient and intrinsic dimensions are inherited from `geom`.
            - The boundary of the complement coincides with the boundary
              of the original geometry.
        """
        super().__init__()
        self.geom = geom
        self.dim = geom.dim
        self.intrinsic_dim = geom.intrinsic_dim
        self.boundary = [*geom.boundary]

    def sdf(self, p: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the signed distance function of the complement geometry.

        Args:
            p (torch.Tensor): Query points.

        Shape:
            - p: (N, dim)
            - return: (N,)

        Returns:
            torch.Tensor: Signed distances to the complement geometry.
        """
        return -self.geom.sdf(p)

    def get_bounding_box(self) -> List[float]:
        """
        Return the axis-aligned bounding box of the complement geometry.

        Since the complement represents an unbounded exterior region,
        the bounding box is defined as infinite in all directions.

        Returns:
            list[float]:
                Bounding box of the form
                [-inf, inf, -inf, inf, ...] with length 2 * dim.
        """
        _ = self.geom.get_bounding_box()  # evaluated for interface consistency
        return [
            float("-inf") if i % 2 == 0 else float("inf")
            for _ in range(self.dim)
            for i in range(2)
        ]

    def in_sample(
            self,
            num_samples: int,
            with_boundary: bool = False,
    ) -> torch.Tensor:
        """
        Sample points inside the complement geometry.

        This method delegates sampling to the underlying geometry.
        The semantic interpretation of "inside" is therefore defined
        by the context in which the complement is used (e.g., in
        intersection or difference operations).

        Args:
            num_samples (int): Number of samples requested.
            with_boundary (bool): Whether to include boundary points.

        Returns:
            torch.Tensor: Sampled points.
        """
        return self.geom.in_sample(num_samples, with_boundary)

    def on_sample(
            self,
            num_samples: int,
            with_normal: bool = False,
            separate: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Sample points on the boundary of the complement geometry.

        The boundary of the complement coincides with the boundary of
        the original geometry, so this method directly forwards the
        call to the wrapped geometry.

        Args:
            num_samples (int): Target number of boundary samples.
            with_normal (bool): If True, also return outward normal vectors.
            separate (bool): If True, preserve boundary components.

        Returns:
            Same return type and structure as `geom.on_sample`.
        """
        return self.geom.on_sample(num_samples, with_normal, separate)


class ExtrudeBody(GeometryBase):
    """
    Geometry representing a 3D solid obtained by extruding a 2D geometry.

    A 2D base geometry is extruded along a given direction vector to form
    a 3D solid with finite thickness.

    Let:
        - d be the extrusion direction vector
        - d̂ = d / ||d|| be the unit direction
        - h = ||d|| / 2 be the half thickness

    The signed distance function (SDF) is defined as:
        sdf(p) = max(d_2D(q), |dot(p, d̂)| - h)

    where:
        q = (dot(p, u), dot(p, v)),
    and (u, v, d̂) form an orthonormal basis.

    Args:
        base2d (GeometryBase): A 2D geometry to be extruded.
        direction (array-like): 3D vector specifying extrusion direction
            and total thickness (its magnitude).

    Notes:
        - `base2d` must be two-dimensional.
        - The resulting geometry is three-dimensional and unbounded
          only through the base geometry.
    """

    # ------------------------------------------------------------------ #
    # Orthonormal basis construction
    # ------------------------------------------------------------------ #
    def _orthonormal(self, n: torch.Tensor) -> torch.Tensor:
        """
        Construct a unit vector orthogonal to a given vector.

        Args:
            n (torch.Tensor): Input vector of shape (3,).

        Returns:
            torch.Tensor: A unit vector orthogonal to `n`.

        Notes:
            This method is numerically robust and works for all nonzero `n`.
        """
        ex = torch.tensor([1.0, 0.0, 0.0], dtype=n.dtype, device=n.device)
        ey = torch.tensor([0.0, 1.0, 0.0], dtype=n.dtype, device=n.device)

        v = torch.linalg.cross(n, ex)
        if torch.norm(v) < 1e-7:
            v = torch.linalg.cross(n, ey)

        return v / torch.norm(v)

    # ------------------------------------------------------------------ #
    # Constructor
    # ------------------------------------------------------------------ #
    def __init__(
            self,
            base2d: GeometryBase,
            direction: Union[torch.Tensor, list, tuple] = (0.0, 0.0, 1.0),
    ):
        """
        Initialize an extruded 3D geometry.

        Args:
            base2d (GeometryBase): Base 2D geometry to be extruded.
            direction (array-like): Extrusion direction vector. Its magnitude
                defines the total thickness.

        Raises:
            ValueError: If `base2d` is not 2D or `direction` is zero.
        """
        super().__init__(dim=3, intrinsic_dim=3)

        if base2d.dim != 2:
            raise ValueError("base2d must be 2-D")

        self.base = base2d

        d = torch.tensor(direction, dtype=self.dtype)
        L = torch.norm(d)
        if L < 1e-8:
            raise ValueError("direction vector must be non-zero")

        self.d = d / L  # Unit extrusion direction
        self.len = L.item()  # Total thickness
        self.h = self.len * 0.5  # Half thickness

        self.u = self._orthonormal(self.d)
        self.v = torch.linalg.cross(self.d, self.u)

    # ------------------------------------------------------------------ #
    # Signed distance function
    # ------------------------------------------------------------------ #
    def sdf(self, p: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the signed distance function of the extruded body.

        Args:
            p (torch.Tensor): Query points.

        Shape:
            - p: (N, 3)
            - return: (N,)

        Returns:
            torch.Tensor: Signed distances to the extruded geometry.
        """
        proj_u = torch.matmul(p, self.u)
        proj_v = torch.matmul(p, self.v)
        q = torch.stack([proj_u, proj_v], dim=1)

        d2 = self.base.sdf(q)
        dz = torch.abs(torch.matmul(p, self.d)) - self.h

        return torch.max(d2, dz.unsqueeze(1))

    # ------------------------------------------------------------------ #
    # Axis-aligned bounding box
    # ------------------------------------------------------------------ #
    def get_bounding_box(self) -> List[float]:
        """
        Return a tight axis-aligned bounding box of the extruded geometry.

        The bounding box is computed by extruding the 2D bounding box
        corners along the extrusion direction.

        Returns:
            list[float]:
                Bounding box in the form
                [x_min, x_max, y_min, y_max, z_min, z_max].
        """
        bx_min, bx_max, by_min, by_max = self.base.get_bounding_box()

        corners_2d = torch.tensor(
            [
                [bx_min, by_min],
                [bx_min, by_max],
                [bx_max, by_min],
                [bx_max, by_max],
            ],
            dtype=self.dtype,
        )

        pts = []
        for s in (-self.h, self.h):
            for x, y in corners_2d:
                pts.append(x * self.u + y * self.v + s * self.d)

        pts = torch.stack(pts, dim=0)

        xyz_min = pts.min(dim=0).values
        xyz_max = pts.max(dim=0).values

        x_min, y_min, z_min = xyz_min.tolist()
        x_max, y_max, z_max = xyz_max.tolist()

        return [x_min, x_max, y_min, y_max, z_min, z_max]

    # ------------------------------------------------------------------ #
    # Interior sampling
    # ------------------------------------------------------------------ #
    def in_sample(
            self,
            num_samples: int,
            with_boundary: bool = False,
    ) -> torch.Tensor:
        """
        Sample points uniformly inside the extruded volume.

        Sampling strategy:
            - Sample (u, v) coordinates inside the base 2D geometry
            - Sample the extrusion coordinate uniformly in [-h, h]

        Args:
            num_samples (int): Number of samples to generate.
            with_boundary (bool): Ignored for volume sampling.

        Returns:
            torch.Tensor: Sampled points of shape (N, 3).
        """
        pts2d = self.base.in_sample(num_samples, with_boundary=False)

        if pts2d.shape[0] < num_samples:
            reps = (num_samples + pts2d.shape[0] - 1) // pts2d.shape[0]
            pts2d = pts2d.repeat(reps, 1)[:num_samples]

        z = (
                torch.rand(pts2d.shape[0], 1, generator=self.gen)
                * self.len
                - self.h
        )

        xyz = (
                pts2d[:, 0:1] * self.u
                + pts2d[:, 1:2] * self.v
                + z * self.d
        )

        return xyz

    # ------------------------------------------------------------------ #
    # Boundary sampling
    # ------------------------------------------------------------------ #
    def on_sample(
            self,
            num_samples: int,
            with_normal: bool = False,
            separate: bool = False,
    ) -> Any:
        """
        Sample points on the boundary of the extruded geometry.

        Sampling strategy:
            - Approximately 2/3 of samples are drawn from the top and bottom
              caps (area sampling of the base geometry).
            - Approximately 1/3 of samples are drawn from the side walls
              (extrusion of the base boundary).

        Args:
            num_samples (int): Target number of boundary samples.
            with_normal (bool): If True, also return outward normal vectors.
            separate (bool): If True, return boundary components separately.

        Returns:
            If separate is False:
                - with_normal = False:
                    Tensor of shape (N, 3)
                - with_normal = True:
                    Tuple (points, normals)

            If separate is True:
                - with_normal = False:
                    Tuple (top_cap, bottom_cap, side_wall)
                - with_normal = True:
                    Tuple of (points, normals) for each boundary component
        """
        n_cap = num_samples // 3
        n_side = num_samples - 2 * n_cap

        # ---- Caps (top and bottom) ----
        cap2d = self.base.in_sample(n_cap, with_boundary=True)
        if cap2d.shape[0] < n_cap:
            reps = (n_cap + cap2d.shape[0] - 1) // cap2d.shape[0]
            cap2d = cap2d.repeat(reps, 1)[:n_cap]

        top_pts = (
                cap2d[:, 0:1] * self.u
                + cap2d[:, 1:2] * self.v
                + self.h * self.d
        )
        bot_pts = (
                cap2d[:, 0:1] * self.u
                + cap2d[:, 1:2] * self.v
                - self.h * self.d
        )
        pts_cap = torch.cat([top_pts, bot_pts], dim=0)

        if with_normal:
            n_top = self.d.expand_as(top_pts)
            n_bot = (-self.d).expand_as(bot_pts)
            normals_cap = torch.cat([n_top, n_bot], dim=0)

        # ---- Side walls ----
        if with_normal:
            edge2d, edge_n2d = self.base.on_sample(
                n_side, with_normal=True
            )
        else:
            edge2d = self.base.on_sample(n_side, with_normal=False)

        m_side = edge2d.shape[0]
        z_side = (
                torch.rand(
                    m_side, 1,
                    device=edge2d.device,
                    dtype=edge2d.dtype,
                    generator=self.gen,
                ) * self.len
                - self.h
        )

        pts_side = (
                edge2d[:, 0:1] * self.u
                + edge2d[:, 1:2] * self.v
                + z_side * self.d
        )

        if with_normal:
            side_norm_vec = (
                    edge_n2d[:, 0:1] * self.u
                    + edge_n2d[:, 1:2] * self.v
            )
            side_normals = side_norm_vec / torch.norm(
                side_norm_vec, dim=1, keepdim=True
            )

        # ---- Merge and return ----
        if separate:
            if with_normal:
                return (
                    (top_pts, n_top),
                    (bot_pts, n_bot),
                    (pts_side, side_normals),
                )
            else:
                return top_pts, bot_pts, pts_side
        else:
            if with_normal:
                points = torch.cat([pts_cap, pts_side], dim=0)
                normals = torch.cat([normals_cap, side_normals], dim=0)
                return points, normals
            else:
                return torch.cat([pts_cap, pts_side], dim=0)


class ImplicitFunctionBase(GeometryBase):
    """
    Base class for geometries defined by implicit functions.

    Subclasses must implement a scalar-valued shape function f(p) whose
    zero level set defines the geometry boundary. This class provides
    utilities to evaluate normalized signed distance functions (SDFs),
    normals, and mean curvature, with optional acceleration via a model
    that supports directional derivatives (`dForward`).
    """

    @abstractmethod
    def shape_func(self, p: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the implicit shape function.

        Args:
            p (torch.Tensor): Query points.

        Shape:
            - p: (N, 3)
            - return: (N,) or (N, 1)

        Returns:
            torch.Tensor: Values of the implicit function f(p).
        """
        pass

    # ------------------------------------------------------------------ #
    # Utilities for locating a model that supports directional derivatives
    # ------------------------------------------------------------------ #
    def _get_model_for_dforward(self):
        """
        Return a model object that provides `dForward`, if available.

        Subclasses may override this method to explicitly return a model
        instance with directional derivative support.

        Returns:
            Any or None: An object exposing `dForward`, or None if unavailable.
        """
        cand = getattr(self, "model", None)
        return cand if (cand is not None and hasattr(cand, "dForward")) else None

    def _has_dforward(self) -> bool:
        """
        Check whether directional derivatives are available.

        Returns:
            bool: True if a compatible `dForward` implementation is found.
        """
        return self._get_model_for_dforward() is not None

    # ------------------------------------------------------------------ #
    # Gradient evaluation
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _eval_grad_dforward(self, p: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the gradient using directional derivatives.

        Args:
            p (torch.Tensor): Query points.

        Returns:
            torch.Tensor: Gradient tensor of shape (N, 3).
        """
        model = self._get_model_for_dforward()
        nx = model.dForward(p, (1, 0, 0)).squeeze(-1)
        ny = model.dForward(p, (0, 1, 0)).squeeze(-1)
        nz = model.dForward(p, (0, 0, 1)).squeeze(-1)
        return torch.stack([nx, ny, nz], dim=-1)

    def _eval_grad(self, p: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the gradient of the implicit function.

        Directional derivatives are used if available; otherwise, PyTorch
        autograd is used as a fallback.

        Args:
            p (torch.Tensor): Query points.

        Returns:
            torch.Tensor: Gradient tensor of shape (N, 3).
        """
        if self._has_dforward():
            return self._eval_grad_dforward(p)

        # Fallback: autograd
        p_req = p.detach().clone().requires_grad_(True)
        f = self.shape_func(p_req)
        if f.ndim == 2 and f.size(-1) == 1:
            f = f.squeeze(-1)

        g = torch.autograd.grad(
            f,
            p_req,
            grad_outputs=torch.ones_like(f),
            create_graph=False,
            retain_graph=False,
        )[0]
        return g

    # ------------------------------------------------------------------ #
    # Hessian and Laplacian (directional-derivative path)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _eval_hessian_dforward(
            self, p: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate the Hessian matrix and Laplacian using directional derivatives.

        This requires the underlying model to support second-order
        multi-index derivatives via `dForward`.

        Args:
            p (torch.Tensor): Query points.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - H: Hessian tensor of shape (N, 3, 3)
                - lap: Laplacian values of shape (N,)
        """
        model = self._get_model_for_dforward()

        f_xx = model.dForward(p, (2, 0, 0)).squeeze(-1)
        f_yy = model.dForward(p, (0, 2, 0)).squeeze(-1)
        f_zz = model.dForward(p, (0, 0, 2)).squeeze(-1)
        f_xy = model.dForward(p, (1, 1, 0)).squeeze(-1)
        f_xz = model.dForward(p, (1, 0, 1)).squeeze(-1)
        f_yz = model.dForward(p, (0, 1, 1)).squeeze(-1)

        H = torch.zeros(
            p.shape[0], 3, 3, device=p.device, dtype=p.dtype
        )
        H[:, 0, 0] = f_xx
        H[:, 1, 1] = f_yy
        H[:, 2, 2] = f_zz
        H[:, 0, 1] = H[:, 1, 0] = f_xy
        H[:, 0, 2] = H[:, 2, 0] = f_xz
        H[:, 1, 2] = H[:, 2, 1] = f_yz

        lap = f_xx + f_yy + f_zz
        return H, lap

    # ------------------------------------------------------------------ #
    # Normalized SDF, normals, and curvature
    # ------------------------------------------------------------------ #
    def sdf(
            self,
            p: torch.Tensor,
            with_normal: bool = False,
            with_curvature: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Evaluate the normalized signed distance function (SDF).

        The normalized SDF is defined as:
            sdf(p) = f(p) / ||∇f(p)||

        Optionally, this method can also return unit normals and mean
        curvature:
            H = 0.5 * div(n),   n = ∇f / ||∇f||

        Args:
            p (torch.Tensor): Query points.
            with_normal (bool): If True, also return unit normals.
            with_curvature (bool): If True, also return mean curvature.

        Returns:
            If with_normal=False and with_curvature=False:
                - sdf: Tensor of shape (N, 1)

            If with_normal=True and with_curvature=False:
                - sdf: (N, 1)
                - normals: (N, 3)

            If with_curvature=True:
                - sdf: (N, 1)
                - normals: (N, 3)
                - mean_curvature: (N, 1)

        Notes:
            - Directional derivatives (`dForward`) are preferred when available.
            - No assumption is made that ||∇f|| ≈ 1.
        """
        eps = torch.finfo(p.dtype).eps

        # Evaluate implicit function
        f = self.shape_func(p)
        if f.ndim == 2 and f.size(-1) == 1:
            f = f.squeeze(-1)

        sdf = f.unsqueeze(-1)
        if not (with_normal or with_curvature):
            return sdf.detach()

        # Evaluate gradient
        g = self._eval_grad(p)
        gnorm = g.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        n = g / gnorm

        if with_normal and not with_curvature:
            return sdf.detach(), n.detach()

        # --------------------------------------------------
        # Mean curvature: div(n)
        # --------------------------------------------------
        if self._has_dforward():
            # Closed-form expression:
            # div(n) = (||g||^2 * tr(H) - g^T H g) / ||g||^3
            H, lap = self._eval_hessian_dforward(p)

            g_col = g.unsqueeze(-1)
            Hg = torch.matmul(H, g_col)
            gT_H_g = torch.matmul(
                g_col.transpose(1, 2), Hg
            ).squeeze(-1).squeeze(-1)

            g2 = gnorm.squeeze(-1) ** 2
            div_n = (g2 * lap - gT_H_g) / (gnorm.squeeze(-1) ** 3 + eps)

            mean_curv = 0.5 * div_n
            return (
                sdf.detach(),
                n.detach(),
                mean_curv.detach().unsqueeze(-1),
            )

        # Fallback: autograd-based divergence of n
        p_req = p.detach().clone().requires_grad_(True)
        f2 = self.shape_func(p_req)
        if f2.ndim == 2 and f2.size(-1) == 1:
            f2 = f2.squeeze(-1)

        g2 = torch.autograd.grad(
            f2,
            p_req,
            grad_outputs=torch.ones_like(f2),
            create_graph=True,
            retain_graph=True,
        )[0]

        g2n = g2.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        n2 = g2 / g2n

        div = 0.0
        for i in range(3):
            di = torch.autograd.grad(
                n2[:, i],
                p_req,
                grad_outputs=torch.ones_like(n2[:, i]),
                create_graph=False,
                retain_graph=True,
            )[0][:, i]
            div = div + di

        mean_curv = 0.5 * div

        return (
            sdf.detach(),
            n.detach(),
            mean_curv.detach().unsqueeze(-1),
        )


class ImplicitSurfaceBase(ImplicitFunctionBase):
    """
    Base class for implicitly defined 2D surfaces embedded in 3D.

    An implicit surface is defined as the zero level set of a scalar
    function f(p):
        S = { p ∈ R³ | f(p) = 0 }

    This class provides a robust surface sampling implementation based
    on:
        - volumetric evaluation on a regular grid
        - zero-crossing detection (Marching Cubes or voxel fallback)
        - normal-based Newton projection onto the surface

    Subclasses must implement `shape_func`.
    """

    def __init__(self):
        """
        Initialize an implicit surface.

        Notes:
            - Ambient dimension is 3.
            - Intrinsic (manifold) dimension is 2.
        """
        super().__init__(dim=3, intrinsic_dim=2)

    @abstractmethod
    def shape_func(self, p: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the implicit surface function.

        Args:
            p (torch.Tensor): Query points.

        Shape:
            - p: (N, 3)
            - return: (N,) or (N, 1)

        Returns:
            torch.Tensor: Values of the implicit function f(p).
        """
        pass

    # ================================================================== #
    # Interior sampling (surface points) via Marching Cubes + projection
    # ================================================================== #
    @torch.no_grad()
    def in_sample(self, num_samples: int, with_boundary: bool = False) -> torch.Tensor:
        """
        Sample points on the implicit surface f(p) = 0.

        This method generates initial surface points using a volumetric
        grid and then projects them onto the zero level set using a
        normal-based Newton iteration.

        Pipeline:
            1. Build a regular grid over the bounding box
            2. Evaluate f on the grid in blocks (to avoid OOM)
            3. Extract zero-crossing candidates:
                - Marching Cubes (if available), or
                - voxel-based zero-crossing fallback
            4. Project candidates onto the surface using
               p_{k+1} = p_k - f(p_k)/||∇f(p_k)|| · n̂
            5. Filter invalid / duplicate / out-of-tolerance points
            6. Subsample or augment to match `num_samples`

        Args:
            num_samples (int): Target number of surface points.
            with_boundary (bool): Ignored for implicit surfaces.

        Returns:
            torch.Tensor: Surface points of shape (num_samples, 3).

        Notes:
            - No assumption is made that ||∇f|| ≈ 1.
            - Computation is automatically chunked to control memory usage.
            - If surface extraction fails, a rejection-sampling fallback
              is used.
        """
        if num_samples <= 0:
            return torch.empty(
                0, self.dim, dtype=self.dtype, device=self.device
            )

        # --------------------------------------------------------------
        # 1) Bounding box and grid resolution estimation
        # --------------------------------------------------------------
        x_min, x_max, y_min, y_max, z_min, z_max = self.get_bounding_box()

        # Expand bounding box slightly to avoid clipping
        margin = 0.1 * max(
            (x_max - x_min), (y_max - y_min), (z_max - z_min)
        )
        x_min -= margin
        x_max += margin
        y_min -= margin
        y_max += margin
        z_min -= margin
        z_max += margin

        # Heuristic grid resolution: O(num_samples)
        c = 2.0
        n_per_axis = int(
            max(128, min(320, round(c * (num_samples ** (1.0 / 3.0)))))
        )
        nx = ny = nz = n_per_axis

        def _linspace(a, b, n):
            return torch.linspace(
                a, b, steps=n, device=self.device, dtype=self.dtype
            )

        xs = _linspace(x_min, x_max, nx)
        ys = _linspace(y_min, y_max, ny)
        zs = _linspace(z_min, z_max, nz)

        # --------------------------------------------------------------
        # 2) Evaluate implicit function on the grid (block-wise)
        # --------------------------------------------------------------
        def _eval_grid():
            sdf_grid = torch.empty(
                (nx, ny, nz), dtype=self.dtype, device=self.device
            )
            bz = max(8, min(nz, 64))
            for z0 in range(0, nz, bz):
                z1 = min(nz, z0 + bz)
                Z = zs[z0:z1]
                X, Y, Zm = torch.meshgrid(xs, ys, Z, indexing="ij")
                pts = torch.stack([X, Y, Zm], dim=-1).reshape(-1, 3)

                f = self.shape_func(pts)
                if f.ndim == 2 and f.size(-1) == 1:
                    f = f.squeeze(-1)

                sdf_grid[:, :, z0:z1] = f.reshape(nx, ny, z1 - z0)
                del X, Y, Zm, pts, f
            return sdf_grid

        sdf_grid = _eval_grid()

        # --------------------------------------------------------------
        # 3) Initial surface candidates (voxel fallback)
        # --------------------------------------------------------------
        init_pts = self._fallback_voxel_zerocross(
            xs, ys, zs, sdf_grid, k=max(num_samples * 2, 512)
        )

        # --------------------------------------------------------------
        # 4) Project candidates to f(p)=0
        # --------------------------------------------------------------
        proj = self._project_to_surface(init_pts, max_iter=60)

        # --------------------------------------------------------------
        # 5) Cleanup and subsampling
        # --------------------------------------------------------------
        f_proj = self.shape_func(proj)
        if f_proj.ndim == 2 and f_proj.size(-1) == 1:
            f_proj = f_proj.squeeze(-1)

        tol = torch.finfo(self.dtype).eps * 100
        mask = torch.isfinite(f_proj) & (f_proj.abs() < tol)
        proj = proj[mask]

        if proj.numel() == 0:
            return self._fallback_rejection_projection(num_samples)

        if proj.shape[0] >= num_samples:
            idx = torch.randperm(
                proj.shape[0], device=proj.device, generator=self.gen
            )
            return proj[idx[:num_samples]].contiguous()

        need = num_samples - proj.shape[0]
        extra = self._fallback_rejection_projection(need)
        return torch.cat([proj, extra], dim=0)[:num_samples].contiguous()

    # ================================================================== #
    # Normal-based Newton projection
    # ================================================================== #
    @torch.no_grad()
    def _project_to_surface(
            self, points: torch.Tensor, max_iter: int = 10
    ) -> torch.Tensor:
        """
        Project points onto the implicit surface using Newton iterations.

        Update rule:
            p_{k+1} = p_k - f(p_k)/||∇f(p_k)|| · n̂

        Args:
            points (torch.Tensor): Initial points, shape (N, 3).
            max_iter (int): Maximum number of projection iterations.

        Returns:
            torch.Tensor: Projected surface points.
        """
        pts = points.clone()
        B = max(2048, min(32768, points.shape[0]))

        for _ in range(max_iter):
            for i in range(0, pts.shape[0], B):
                sl = slice(i, i + B)
                p = pts[sl]

                f = self.shape_func(p)
                if f.ndim == 2 and f.size(-1) == 1:
                    f = f.squeeze(-1)

                g = self._eval_grad(p)
                n_hat = g / g.norm(dim=1, keepdim=True)

                pts[sl] = p - f.unsqueeze(-1) * n_hat

        return pts

    # ================================================================== #
    # Fallback: voxel zero-crossing approximation
    # ================================================================== #
    @torch.no_grad()
    def _fallback_voxel_zerocross(
            self, xs, ys, zs, sdf_grid, k: int
    ) -> torch.Tensor:
        """
        Approximate surface points via voxel sign changes.

        This method detects sign changes of f between neighboring voxels
        and uses face centers as initial surface candidates.

        Args:
            xs, ys, zs (torch.Tensor): Grid coordinates.
            sdf_grid (torch.Tensor): Evaluated implicit function on the grid.
            k (int): Target number of initial points.

        Returns:
            torch.Tensor: Approximate surface points.
        """
        nx, ny, nz = sdf_grid.shape
        pts = []

        sign_x = torch.signbit(sdf_grid[1:, :, :]) != torch.signbit(
            sdf_grid[:-1, :, :]
        )
        ix, iy, iz = torch.where(sign_x)
        if ix.numel():
            pts.append(
                torch.stack(
                    [(xs[ix] + xs[ix + 1]) / 2, ys[iy], zs[iz]], dim=1
                )
            )

        sign_y = torch.signbit(sdf_grid[:, 1:, :]) != torch.signbit(
            sdf_grid[:, :-1, :]
        )
        ix, iy, iz = torch.where(sign_y)
        if iy.numel():
            pts.append(
                torch.stack(
                    [xs[ix], (ys[iy] + ys[iy + 1]) / 2, zs[iz]], dim=1
                )
            )

        sign_z = torch.signbit(sdf_grid[:, :, 1:]) != torch.signbit(
            sdf_grid[:, :, :-1]
        )
        ix, iy, iz = torch.where(sign_z)
        if iz.numel():
            pts.append(
                torch.stack(
                    [xs[ix], ys[iy], (zs[iz] + zs[iz + 1]) / 2], dim=1
                )
            )

        if not pts:
            return self._fallback_rejection_projection(k)

        P = torch.cat(pts, dim=0).to(self.device, self.dtype)
        if P.shape[0] > 3 * k:
            sel = torch.randperm(P.shape[0], device=self.device)[:3 * k]
            P = P[sel]
        return P

    # ================================================================== #
    # Fallback: rejection sampling + projection
    # ================================================================== #
    @torch.no_grad()
    def _fallback_rejection_projection(
            self, num_samples: int
    ) -> torch.Tensor:
        """
        Fallback surface sampling using rejection sampling and projection.

        This method randomly samples points in an expanded bounding box,
        keeps points close to the surface, and projects them onto f(p)=0.

        Args:
            num_samples (int): Number of surface points requested.

        Returns:
            torch.Tensor: Surface points.
        """
        x_min, x_max, y_min, y_max, z_min, z_max = self.get_bounding_box()
        volume = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
        resolution = (volume / max(num_samples, 1)) ** (1 / self.dim)
        eps_band = resolution

        collected = []
        oversample = int(num_samples * 1.5)

        while sum(c.shape[0] for c in collected) < num_samples:
            rand = torch.rand(
                oversample,
                self.dim,
                device=self.device,
                dtype=self.dtype,
                generator=self.gen,
            )

            p = torch.empty(
                oversample, self.dim, dtype=self.dtype, device=self.device
            )
            p[:, 0] = (x_min - 4 * eps_band) + rand[:, 0] * (
                    (x_max + 4 * eps_band) - (x_min - 4 * eps_band)
            )
            p[:, 1] = (y_min - 4 * eps_band) + rand[:, 1] * (
                    (y_max + 4 * eps_band) - (y_min - 4 * eps_band)
            )
            p[:, 2] = (z_min - 4 * eps_band) + rand[:, 2] * (
                    (z_max + 4 * eps_band) - (z_min - 4 * eps_band)
            )

            f = self.shape_func(p)
            if f.ndim == 2 and f.size(-1) == 1:
                f = f.squeeze(-1)

            g = self._eval_grad(p)
            n_hat = g / g.norm(dim=1, keepdim=True)

            near = f.abs() < eps_band
            near_p = p[near]
            near_n = n_hat[near]
            near_f = f[near]

            for _ in range(10):
                if near_p.shape[0] == 0:
                    break
                near_p = near_p - near_f.unsqueeze(-1) * near_n

                f2 = self.shape_func(near_p)
                if f2.ndim == 2 and f2.size(-1) == 1:
                    f2 = f2.squeeze(-1)

                g2 = self._eval_grad(near_p)
                near_n = g2 / g2.norm(dim=1, keepdim=True)
                near_f = f2

            mask = torch.isfinite(near_f) & (
                    near_f.abs()
                    < torch.finfo(self.dtype).eps * resolution
            )
            collected.append(near_p[mask])

        return torch.cat(collected, dim=0)[:num_samples]

    def on_sample(
            self, num_samples: int, with_normal: bool = False, separate: bool = False,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, ...],
        Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
    ]:
        """
        Explicit boundary sampling is not defined for implicit surfaces.

        Returns:
            Empty tensors (and normals if requested).
        """
        empty = torch.empty(
            (0, self.dim), dtype=self.dtype, device=self.device
        )
        if with_normal:
            return empty, empty
        return empty


class Point1D(GeometryBase):
    """
    Class representing a 1D point.

    Attributes:
    ----------
    x : torch.FloatType
        The x-coordinate of the point.
    """

    def __init__(self, x: torch.FloatType):
        """
        Initialize the Point1D object.

        Args:
        ----
        x : torch.FloatType
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

    def on_sample(self, num_samples: int, with_normal: bool = False, separate=False) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, ...],
        Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
    ]:
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
        return torch.tensor([[self.x]] * num_samples) if not separate else (torch.tensor([[self.x]] * num_samples),)


class Point2D(GeometryBase):
    """
    Class representing a 2D point.

    Attributes:
    ----------
    x : torch.FloatType
        The x-coordinate of the point.
    y : torch.FloatType
        The y-coordinate of the point.
    """

    def __init__(self, x: torch.FloatType, y: torch.FloatType):
        """
        Initialize the Point2D object.

        Args:
        ----
        x : torch.FloatType
            The x-coordinate of the point.
        y : torch.FloatType
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

    def on_sample(self, num_samples: int, with_normal: bool = False, separate=False) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, ...],
        Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
    ]:
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
        if not separate:
            if with_normal:
                raise NotImplementedError("Normal vectors are not available for 2D points.")
            return torch.tensor([[self.x, self.y]] * num_samples) if not separate else (
                torch.tensor([[self.x]] * num_samples),)
        else:
            if with_normal:
                raise NotImplementedError("Normal vectors are not available for 2D points.")
            return (torch.tensor([[self.x, self.y]] * num_samples),) if not separate else (
                torch.tensor([[self.x]] * num_samples),)


class Point3D(GeometryBase):
    """
    Class representing a 3D point.

    Attributes:
    ----------
    x : torch.FloatType
        The x-coordinate of the point.
    y : torch.FloatType
        The y-coordinate of the point.
    z : torch.FloatType
        The z-coordinate of the point.
    """

    def __init__(self, x: torch.FloatType, y: torch.FloatType, z: torch.FloatType):
        """
        Initialize the Point3D object.

        Args:
        ----
        x : torch.FloatType
            The x-coordinate of the point.
        y : torch.FloatType
            The y-coordinate of the point.
        z : torch.FloatType
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

    def on_sample(self, num_samples: int, with_normal: bool = False, separate=False) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, ...],
        Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
    ]:
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
        return torch.tensor([[self.x, self.y, self.z]] * num_samples) if not separate else (
            torch.tensor([[self.x, self.y, self.z]] * num_samples),)


class Line1D(GeometryBase):
    """
    Class representing a 1D line segment.

    Attributes:
    ----------
    x1 : torch.FloatType
        The x-coordinate of the first endpoint.
    x2 : torch.FloatType
        The x-coordinate of the second endpoint.
    boundary : list
        The boundary points of the line segment.
    """

    def __init__(self, x1: torch.FloatType, x2: torch.FloatType):
        """
        Initialize the Line1D object.

        Args:
        ----
        x1 : torch.FloatType
            The x-coordinate of the first endpoint.
        x2 : torch.FloatType
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

    def on_sample(self, num_samples: int, with_normal: bool = False, separate=False) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, ...],
        Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
    ]:
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
            return (torch.cat([a, b], dim=0), torch.cat(
                [torch.tensor([[(self.x2 - self.x1) / abs(self.x2 - self.x1)]] * (num_samples // 2)),
                 torch.tensor([[(self.x1 - self.x2) / abs(self.x1 - self.x2)]] * (num_samples // 2))],
                dim=0)) if not separate else (torch.cat([a, b], dim=0), torch.cat(
                [torch.tensor([[(self.x2 - self.x1) / abs(self.x2 - self.x1)]] * (num_samples // 2)),
                 torch.tensor([[(self.x1 - self.x2) / abs(self.x1 - self.x2)]] * (num_samples // 2))],
                dim=0)),
        else:
            return torch.cat([a, b], dim=0) if not separate else (torch.cat([a, b], dim=0),)


class Line2D(GeometryBase):
    def __init__(self, x1: torch.FloatType, y1: torch.FloatType, x2: torch.FloatType, y2: torch.FloatType):
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

    def on_sample(self, num_samples: int, with_normal: bool = False, separate=False) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, ...],
        Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
    ]:
        a = self.boundary[0].in_sample(num_samples // 2, with_boundary=True)
        b = self.boundary[1].in_sample(num_samples // 2, with_boundary=True)
        if with_normal:
            return (torch.cat([a, b], dim=0), torch.cat([torch.tensor(
                [[(self.x2 - self.x1) / abs(self.x2 - self.x1), (self.y2 - self.y1) / abs(self.y2 - self.y1)]] * (
                        num_samples // 2)), torch.tensor(
                [[(self.x1 - self.x2) / abs(self.x1 - self.x2), (self.y1 - self.y2) / abs(self.y1 - self.y2)]] * (
                        num_samples // 2))], dim=0)) if not separate else (torch.cat([a, b], dim=0),
                                                                           torch.cat([torch.tensor(
                                                                               [[(self.x2 - self.x1) / abs(
                                                                                   self.x2 - self.x1),
                                                                                 (self.y2 - self.y1) / abs(
                                                                                     self.y2 - self.y1)]] * (
                                                                                       num_samples // 2)), torch.tensor(
                                                                               [[(self.x1 - self.x2) / abs(
                                                                                   self.x1 - self.x2),
                                                                                 (self.y1 - self.y2) / abs(
                                                                                     self.y1 - self.y2)]] * (
                                                                                       num_samples // 2))], dim=0)),
        else:
            return torch.cat([a, b], dim=0) if not separate else (torch.cat([a, b], dim=0),)


class Line3D(GeometryBase):
    def __init__(self, x1: torch.FloatType, y1: torch.FloatType, z1: torch.FloatType, x2: torch.FloatType,
                 y2: torch.FloatType,
                 z2: torch.FloatType):
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

    def on_sample(self, num_samples: int, with_normal: bool = False, separate=False) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, ...],
        Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
    ]:
        a = self.boundary[0].in_sample(num_samples // 2, with_boundary=True)
        b = self.boundary[1].in_sample(num_samples // 2, with_boundary=True)
        if with_normal:
            return (torch.cat([a, b], dim=0), torch.cat([torch.tensor(
                [[(self.x2 - self.x1) / abs(self.x2 - self.x1), (self.y2 - self.y1) / abs(self.y2 - self.y1),
                  (self.z2 - self.z1) / abs(self.z2 - self.z1)]] * (num_samples // 2)), torch.tensor(
                [[(self.x1 - self.x2) / abs(self.x1 - self.x2), (self.y1 - self.y2) / abs(self.y1 - self.y2),
                  (self.z1 - self.z2) / abs(self.z1 - self.z2)]] * (num_samples // 2))], dim=0)) if not separate else (
                torch.cat([a, b], dim=0), torch.cat([torch.tensor(
                [[(self.x2 - self.x1) / abs(self.x2 - self.x1), (self.y2 - self.y1) / abs(self.y2 - self.y1),
                  (self.z2 - self.z1) / abs(self.z2 - self.z1)]] * (num_samples // 2)), torch.tensor(
                [[(self.x1 - self.x2) / abs(self.x1 - self.x2), (self.y1 - self.y2) / abs(self.y1 - self.y2),
                  (self.z1 - self.z2) / abs(self.z1 - self.z2)]] * (num_samples // 2))], dim=0)),
        else:
            return torch.cat([a, b], dim=0) if not separate else (torch.cat([a, b], dim=0),)


class Square2D(GeometryBase):
    """
    Axis-aligned square in 2D.

    The square is defined by its center and half-lengths along the x and y axes.
    Its boundary consists of four line segments.

    Args:
        center (array-like): Center of the square, shape (2,).
        half (array-like, optional): Half-lengths along x and y directions,
            shape (2,). This is the preferred argument.
        radius (array-like, optional): Deprecated alias of `half`.

    Notes:
        - Exactly one of `half` or `radius` must be provided.
        - `radius` is kept only for backward compatibility.
    """

    def __init__(
            self,
            center: Union[torch.Tensor, List, Tuple],
            half: Union[torch.Tensor, List, Tuple] = None,
            radius: Union[torch.Tensor, List, Tuple] = None,
    ):
        super().__init__(dim=2, intrinsic_dim=2)

        if half is None and radius is None:
            raise ValueError(
                "You must provide `half` (preferred) or `radius` (deprecated)."
            )

        if radius is not None:
            import warnings
            warnings.warn(
                "`radius` is deprecated and will be removed in future versions. "
                "Use `half` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            half = radius

        self.center = torch.tensor(center).view(1, -1)
        self.half = torch.tensor(half).view(1, -1)
        self.radius = self.half  # backward-compatibility alias

        # Boundary edges (counter-clockwise)
        self.boundary = [
            Line2D(
                self.center[0, 0] - self.half[0, 0],
                self.center[0, 1] - self.half[0, 1],
                self.center[0, 0] + self.half[0, 0],
                self.center[0, 1] - self.half[0, 1],
            ),
            Line2D(
                self.center[0, 0] + self.half[0, 0],
                self.center[0, 1] - self.half[0, 1],
                self.center[0, 0] + self.half[0, 0],
                self.center[0, 1] + self.half[0, 1],
            ),
            Line2D(
                self.center[0, 0] + self.half[0, 0],
                self.center[0, 1] + self.half[0, 1],
                self.center[0, 0] - self.half[0, 0],
                self.center[0, 1] + self.half[0, 1],
            ),
            Line2D(
                self.center[0, 0] - self.half[0, 0],
                self.center[0, 1] + self.half[0, 1],
                self.center[0, 0] - self.half[0, 0],
                self.center[0, 1] - self.half[0, 1],
            ),
        ]

    def sdf(self, p: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the signed distance function of the square.

        Args:
            p (torch.Tensor): Query points of shape (N, 2).

        Returns:
            torch.Tensor: Signed distances of shape (N, 1).
        """
        d = torch.abs(p - self.center) - self.half
        return (
                torch.norm(torch.clamp(d, min=0.0), dim=1, keepdim=True)
                + torch.clamp(torch.max(d, dim=1, keepdim=True).values, max=0.0)
        )

    def get_bounding_box(self) -> List[float]:
        """
        Return the axis-aligned bounding box of the square.

        Returns:
            list[float]: [x_min, x_max, y_min, y_max]
        """
        x_min = self.center[0, 0] - self.half[0, 0]
        x_max = self.center[0, 0] + self.half[0, 0]
        y_min = self.center[0, 1] - self.half[0, 1]
        y_max = self.center[0, 1] + self.half[0, 1]
        return [x_min.item(), x_max.item(), y_min.item(), y_max.item()]

    def in_sample(
            self,
            num_samples: Union[int, List[int], Tuple[int, int]],
            with_boundary: bool = False,
    ) -> torch.Tensor:
        """
        Sample points uniformly inside the square.

        Args:
            num_samples (int or tuple): If int, samples are arranged on an
                approximately sqrt(N) x sqrt(N) grid. If a tuple (nx, ny),
                specifies grid resolution explicitly.
            with_boundary (bool): If True, include boundary points.

        Returns:
            torch.Tensor: Sampled points of shape (N, 2).
        """
        if isinstance(num_samples, int):
            num_x = num_y = int(num_samples ** 0.5)
        elif isinstance(num_samples, (list, tuple)) and len(num_samples) == 2:
            num_x, num_y = int(num_samples[0]), int(num_samples[1])
        else:
            raise ValueError(
                "num_samples must be an int or a tuple/list of two integers."
            )

        x_min = self.center[0, 0] - self.half[0, 0]
        x_max = self.center[0, 0] + self.half[0, 0]
        y_min = self.center[0, 1] - self.half[0, 1]
        y_max = self.center[0, 1] + self.half[0, 1]

        if with_boundary:
            x = torch.linspace(x_min, x_max, num_x)
            y = torch.linspace(y_min, y_max, num_y)
        else:
            x = torch.linspace(x_min, x_max, num_x + 2)[1:-1]
            y = torch.linspace(y_min, y_max, num_y + 2)[1:-1]

        X, Y = torch.meshgrid(x, y, indexing="ij")
        return torch.cat(
            [X.reshape(-1, 1), Y.reshape(-1, 1)], dim=1
        )

    def on_sample(
            self,
            num_samples: Union[int, List[int], Tuple[int, ...]],
            with_normal: bool = False,
            separate: bool = False,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, ...],
        Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
    ]:
        """
        Sample points on the boundary of the square.

        Args:
            num_samples (int or sequence): Number of samples per edge.
            with_normal (bool): If True, also return outward normals.
            separate (bool): If True, return samples per edge separately.

        Returns:
            Boundary samples, optionally grouped by edge and/or accompanied
            by normal vectors.
        """
        if isinstance(num_samples, int):
            nums = [num_samples // 4] * 4
        elif isinstance(num_samples, (list, tuple)) and len(num_samples) == 2:
            nums = [
                int(num_samples[0]),
                int(num_samples[1]),
                int(num_samples[0]),
                int(num_samples[1]),
            ]
        elif isinstance(num_samples, (list, tuple)) and len(num_samples) == 4:
            nums = list(map(int, num_samples))
        else:
            raise ValueError(
                "num_samples must be an int or a list/tuple of length 2 or 4."
            )

        a = self.boundary[0].in_sample(nums[0], with_boundary=True)
        b = self.boundary[1].in_sample(nums[1], with_boundary=True)
        c = self.boundary[2].in_sample(nums[2], with_boundary=True)
        d = self.boundary[3].in_sample(nums[3], with_boundary=True)

        if not separate:
            if with_normal:
                normals = torch.cat(
                    [
                        torch.tensor([[0.0, -1.0]] * nums[0]),
                        torch.tensor([[1.0, 0.0]] * nums[1]),
                        torch.tensor([[0.0, 1.0]] * nums[2]),
                        torch.tensor([[-1.0, 0.0]] * nums[3]),
                    ],
                    dim=0,
                )
                return torch.cat([a, b, c, d], dim=0), normals
            return torch.cat([a, b, c, d], dim=0)

        if with_normal:
            return (
                (a, torch.tensor([[0.0, -1.0]] * nums[0])),
                (b, torch.tensor([[1.0, 0.0]] * nums[1])),
                (c, torch.tensor([[0.0, 1.0]] * nums[2])),
                (d, torch.tensor([[-1.0, 0.0]] * nums[3])),
            )
        return a, b, c, d


class Square3D(GeometryBase):
    """
    Axis-aligned square embedded in 3D.

    This geometry represents a 2D square lying in one of the coordinate
    planes. Exactly one component of `radius` must be zero, indicating
    the normal direction of the square.
    """

    def __init__(
            self,
            center: Union[torch.Tensor, List, Tuple],
            radius: Union[torch.Tensor, List, Tuple],
    ):
        """
        Initialize a 3D square.

        Args:
            center (array-like): Center point of the square, shape (3,).
            radius (array-like): Half-lengths along each axis. Exactly one
                entry must be zero.

        Raises:
            ValueError: If no zero-radius axis is found.
        """
        super().__init__(dim=3, intrinsic_dim=2)

        self.center = (
            torch.tensor(center).view(1, -1)
            if isinstance(center, (list, tuple))
            else center.view(1, -1)
        )
        self.radius = (
            torch.tensor(radius).view(1, -1)
            if isinstance(radius, (list, tuple))
            else radius.view(1, -1)
        )

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

                self.boundary = [
                    Line3D(*p1, *p2),
                    Line3D(*p2, *p3),
                    Line3D(*p3, *p4),
                    Line3D(*p4, *p1),
                ]
                break
        else:
            raise ValueError("Square3D requires exactly one zero radius.")

    def sdf(self, p: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the signed distance function of the square.

        Args:
            p (torch.Tensor): Query points of shape (N, 3).

        Returns:
            torch.Tensor: Signed distances of shape (N, 1).
        """
        d = torch.abs(p - self.center) - self.radius
        return (
                torch.norm(torch.clamp(d, min=0.0), dim=1, keepdim=True)
                + torch.clamp(torch.max(d, dim=1, keepdim=True).values, max=0.0)
        )

    def get_bounding_box(self) -> List[float]:
        """
        Return the axis-aligned bounding box of the square.

        Returns:
            list[float]: [x_min, x_max, y_min, y_max, z_min, z_max]
        """
        x_min = self.center[0, 0] - self.radius[0, 0]
        x_max = self.center[0, 0] + self.radius[0, 0]
        y_min = self.center[0, 1] - self.radius[0, 1]
        y_max = self.center[0, 1] + self.radius[0, 1]
        z_min = self.center[0, 2] - self.radius[0, 2]
        z_max = self.center[0, 2] + self.radius[0, 2]
        return [
            x_min.item(),
            x_max.item(),
            y_min.item(),
            y_max.item(),
            z_min.item(),
            z_max.item(),
        ]

    def in_sample(
            self, num_samples: int, with_boundary: bool = False
    ) -> torch.Tensor:
        """
        Uniformly sample points on the square surface.

        Args:
            num_samples (int): Target number of samples.
            with_boundary (bool): If True, include boundary points.

        Returns:
            torch.Tensor: Sampled points of shape (N, 3).
        """
        n = int(num_samples ** 0.5)

        for i in range(3):
            if self.radius[0, i] == 0.0:
                j, k = (i + 1) % 3, (i + 2) % 3
                break

        if with_boundary:
            tj = torch.linspace(-self.radius[0, j], self.radius[0, j], n)
            tk = torch.linspace(-self.radius[0, k], self.radius[0, k], n)
        else:
            tj = torch.linspace(
                -self.radius[0, j], self.radius[0, j], n + 2
            )[1:-1]
            tk = torch.linspace(
                -self.radius[0, k], self.radius[0, k], n + 2
            )[1:-1]

        TJ, TK = torch.meshgrid(tj, tk, indexing="ij")

        pts = torch.zeros((TJ.numel(), 3), dtype=self.center.dtype)
        pts[:, i] = self.center[0, i]
        pts[:, j] = self.center[0, j] + TJ.reshape(-1)
        pts[:, k] = self.center[0, k] + TK.reshape(-1)

        return pts

    def on_sample(
            self,
            num_samples: int,
            with_normal: bool = False,
            separate: bool = False,
    ):
        """
        Sample points on the boundary of the square.

        Args:
            num_samples (int): Total number of boundary samples.
            with_normal (bool): If True, also return outward normals.
            separate (bool): If True, return samples per edge.

        Returns:
            Boundary samples, optionally grouped by edge and/or normals.
        """
        a = self.boundary[0].in_sample(num_samples // 4, with_boundary=True)
        b = self.boundary[1].in_sample(num_samples // 4, with_boundary=True)
        c = self.boundary[2].in_sample(num_samples // 4, with_boundary=True)
        d = self.boundary[3].in_sample(num_samples // 4, with_boundary=True)

        if not with_normal:
            return (
                torch.cat([a, b, c, d], dim=0)
                if not separate
                else (a, b, c, d)
            )

        for i in range(3):
            if self.radius[0, i] == 0.0:
                j, k = (i + 1) % 3, (i + 2) % 3
                break

        an = torch.zeros((num_samples // 4, 3))
        bn = torch.zeros((num_samples // 4, 3))
        cn = torch.zeros((num_samples // 4, 3))
        dn = torch.zeros((num_samples // 4, 3))

        an[:, k] = -1.0
        bn[:, j] = 1.0
        cn[:, k] = 1.0
        dn[:, j] = -1.0

        if not separate:
            return (
                torch.cat([a, b, c, d], dim=0),
                torch.cat([an, bn, cn, dn], dim=0),
            )

        return (a, an), (b, bn), (c, cn), (d, dn)


class Cube3D(GeometryBase):
    """
    Axis-aligned cube in 3D.

    The cube is defined by its center and half-lengths along the x, y, and z
    axes. Its boundary consists of six planar square faces, each represented
    internally by a `Square3D` object.

    Args:
        center (array-like): Center of the cube, shape (3,).
        half (array-like): Half-lengths along x, y, and z directions, shape (3,).
        radius (array-like, optional): Deprecated alias of `half`.

    Notes:
        - Exactly one of `half` or `radius` must be provided.
        - `radius` is kept only for backward compatibility.
    """

    def __init__(
            self,
            center: Union[torch.Tensor, List, Tuple],
            half: Union[torch.Tensor, List, Tuple],
            radius: Union[torch.Tensor, List, Tuple] = None,
    ):
        super().__init__(dim=3, intrinsic_dim=3)

        # Backward compatibility
        if half is None and radius is None:
            raise ValueError(
                "You must provide `half` (preferred) or `radius` (deprecated)."
            )

        if radius is not None:
            import warnings
            warnings.warn(
                "`radius` is deprecated and will be removed in future versions. "
                "Use `half` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            half = radius

        self.center = torch.tensor(center).view(1, -1).to(dtype=self.dtype)
        self.half = torch.tensor(half).view(1, -1).to(dtype=self.dtype)

        # Construct six boundary faces (+x, -x, +y, -y, +z, -z)
        offsets = [
            [self.half[0, 0], 0.0, 0.0],
            [-self.half[0, 0], 0.0, 0.0],
            [0.0, self.half[0, 1], 0.0],
            [0.0, -self.half[0, 1], 0.0],
            [0.0, 0.0, self.half[0, 2]],
            [0.0, 0.0, -self.half[0, 2]],
        ]

        self.boundary = [
            Square3D(
                self.center + torch.tensor(offset),
                torch.tensor(
                    [
                        self.half[0, i] if offset[i] == 0.0 else 0.0
                        for i in range(3)
                    ]
                ),
            )
            for offset in offsets
        ]

    def sdf(self, p: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the signed distance function of the cube.

        Args:
            p (torch.Tensor): Query points of shape (N, 3).

        Returns:
            torch.Tensor: Signed distances of shape (N, 1).
        """
        d = torch.abs(p - self.center) - self.half
        return (
                torch.norm(torch.clamp(d, min=0.0), dim=1, keepdim=True)
                + torch.clamp(torch.max(d, dim=1, keepdim=True).values, max=0.0)
        )

    def get_bounding_box(self) -> List[float]:
        """
        Return the axis-aligned bounding box of the cube.

        Returns:
            list[float]: [x_min, x_max, y_min, y_max, z_min, z_max]
        """
        x_min = self.center[0, 0] - self.half[0, 0]
        x_max = self.center[0, 0] + self.half[0, 0]
        y_min = self.center[0, 1] - self.half[0, 1]
        y_max = self.center[0, 1] + self.half[0, 1]
        z_min = self.center[0, 2] - self.half[0, 2]
        z_max = self.center[0, 2] + self.half[0, 2]
        return [
            x_min.item(),
            x_max.item(),
            y_min.item(),
            y_max.item(),
            z_min.item(),
            z_max.item(),
        ]

    def in_sample(
            self,
            num_samples: Union[int, List[int], Tuple[int, int, int]],
            with_boundary: bool = False,
    ) -> torch.Tensor:
        """
        Uniformly sample points inside the cube.

        Args:
            num_samples (int or tuple): If int, samples are arranged on an
                approximately cubic grid of size N^(1/3) per axis. If a tuple
                (nx, ny, nz), specifies grid resolution explicitly.
            with_boundary (bool): If True, include boundary points.

        Returns:
            torch.Tensor: Sampled points of shape (N, 3).
        """
        if isinstance(num_samples, int):
            num_x = num_y = num_z = int(round(num_samples ** (1.0 / 3.0)))
        elif isinstance(num_samples, (list, tuple)) and len(num_samples) == 3:
            num_x, num_y, num_z = map(int, num_samples)
        else:
            raise ValueError(
                "num_samples must be an int or a list/tuple of three integers."
            )

        x_min = self.center[0, 0] - self.half[0, 0]
        x_max = self.center[0, 0] + self.half[0, 0]
        y_min = self.center[0, 1] - self.half[0, 1]
        y_max = self.center[0, 1] + self.half[0, 1]
        z_min = self.center[0, 2] - self.half[0, 2]
        z_max = self.center[0, 2] + self.half[0, 2]

        if with_boundary:
            x = torch.linspace(x_min, x_max, num_x)
            y = torch.linspace(y_min, y_max, num_y)
            z = torch.linspace(z_min, z_max, num_z)
        else:
            x = torch.linspace(x_min, x_max, num_x + 2)[1:-1]
            y = torch.linspace(y_min, y_max, num_y + 2)[1:-1]
            z = torch.linspace(z_min, z_max, num_z + 2)[1:-1]

        X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")
        return torch.cat(
            [X.reshape(-1, 1), Y.reshape(-1, 1), Z.reshape(-1, 1)], dim=1
        )

    def on_sample(
            self,
            num_samples: int,
            with_normal: bool = False,
            separate: bool = False,
    ):
        """
        Sample points on the boundary of the cube.

        This method generates sample points on the six planar faces of the
        cube (±x, ±y, ±z). Each face is sampled independently using the
        corresponding `Square3D` object stored in `self.boundary`.

        Args:
            num_samples (int): Global sampling budget for the cube boundary.
                The budget is evenly distributed across the six faces as
                ``n_face = max(1, num_samples // 6)``.
            with_normal (bool): If True, also return outward unit normal
                vectors for the sampled boundary points.
            separate (bool): If True, return samples face by face instead
                of concatenating them.

        Returns:
            - If ``separate=False`` and ``with_normal=False``:
                torch.Tensor of shape (N, 3)
            - If ``separate=False`` and ``with_normal=True``:
                (points, normals)
            - If ``separate=True`` and ``with_normal=False``:
                tuple of tensors, one per face
            - If ``separate=True`` and ``with_normal=True``:
                tuple of (points, normals) pairs, one per face

        Notes:
            - The face ordering follows ``self.boundary``:
              +x, -x, +y, -y, +z, -z.
            - Normals are constant per face and aligned with the coordinate
              axes.
        """
        samples = []
        normals = []

        n_face = max(1, num_samples // 6)

        for i, square in enumerate(self.boundary):
            pts = square.in_sample(n_face, with_boundary=True)
            samples.append(pts)

            if with_normal:
                n = torch.zeros(
                    (pts.shape[0], 3),
                    dtype=pts.dtype,
                    device=pts.device,
                )
                axis = i // 2  # 0: x, 1: y, 2: z
                sign = 1.0 if (i % 2 == 0) else -1.0
                n[:, axis] = sign
                normals.append(n)

        if not separate:
            if with_normal:
                return (
                    torch.cat(samples, dim=0),
                    torch.cat(normals, dim=0),
                )
            return torch.cat(samples, dim=0)

        if with_normal:
            return tuple(
                (samples[i], normals[i]) for i in range(len(samples))
            )
        return tuple(samples)


class CircleArc2D(GeometryBase):
    def __init__(self, center: Union[torch.Tensor, List, Tuple], radius: torch.FloatType):
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

    def on_sample(self, num_samples: int, with_normal: bool = False, separate=False) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, ...],
        Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
    ]:
        raise NotImplementedError


class Circle2D(GeometryBase):
    def __init__(self, center: Union[torch.Tensor, List, Tuple], radius: torch.FloatType):
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

    def on_sample(self, num_samples: int, with_normal: bool = False, separate=False) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, ...],
        Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
    ]:
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
        self.center = torch.tensor(center, dtype=self.dtype).view(1, 3) if not isinstance(center,
                                                                                          torch.Tensor) else center.view(
            1, 3)
        self.radius = torch.tensor(radius, dtype=self.dtype) if not isinstance(radius, torch.Tensor) else radius
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

    def on_sample(self, num_samples: int, with_normal: bool = False, separate=False) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, ...],
        Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
    ]:
        empty = torch.empty((0, self.dim), dtype=self.dtype, device=self.device)
        if with_normal:
            return empty, empty
        return empty


class Ball3D(GeometryBase):
    """
    Solid ball (filled sphere) in 3D.

    The geometry represents the closed 3D ball:
        { p ∈ R³ | ||p - center|| ≤ radius }

    Its boundary consists of a single spherical surface.
    """

    def __init__(self, center: Union[torch.Tensor, List, Tuple], radius: float):
        """
        Initialize a 3D ball.

        Args:
            center (array-like or torch.Tensor): Center of the ball, shape (3,).
            radius (float or torch.Tensor): Radius of the ball.
        """
        super().__init__(dim=3, intrinsic_dim=3)

        self.center = (
            torch.tensor(center, dtype=self.dtype).view(1, 3)
            if not isinstance(center, torch.Tensor)
            else center.view(1, 3)
        )
        self.radius = (
            torch.tensor(radius, dtype=self.dtype)
            if not isinstance(radius, torch.Tensor)
            else radius
        )

        # Boundary: a single spherical surface
        self.boundary = [Sphere3D(self.center, self.radius)]

    def sdf(self, p: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the signed distance function of the ball.

        Args:
            p (torch.Tensor): Query points of shape (N, 3).

        Returns:
            torch.Tensor: Signed distances of shape (N, 1).

        Notes:
            - Negative values indicate points inside the ball.
            - Zero corresponds to the spherical boundary.
        """
        return (
                torch.norm(p - self.center.to(p.device), dim=1, keepdim=True)
                - self.radius.to(p.device)
        )

    def get_bounding_box(self) -> List[float]:
        """
        Return the axis-aligned bounding box of the ball.

        Returns:
            list[float]: [x_min, x_max, y_min, y_max, z_min, z_max]
        """
        r = self.radius.item()
        x_min = self.center[0, 0] - r
        x_max = self.center[0, 0] + r
        y_min = self.center[0, 1] - r
        y_max = self.center[0, 1] + r
        z_min = self.center[0, 2] - r
        z_max = self.center[0, 2] + r
        return [
            x_min.item(),
            x_max.item(),
            y_min.item(),
            y_max.item(),
            z_min.item(),
            z_max.item(),
        ]

    def in_sample(
            self, num_samples: int, with_boundary: bool = False
    ) -> torch.Tensor:
        """
        Uniformly sample points inside the ball using spherical coordinates.

        Args:
            num_samples (int): Target number of samples.
            with_boundary (bool): If True, include points on the boundary.

        Returns:
            torch.Tensor: Sampled points of shape (N, 3).

        Notes:
            - Sampling is performed on a regular grid in spherical coordinates.
            - The actual number of returned samples depends on the grid
              resolution derived from `num_samples`.
        """
        device = self.center.device
        n = int(num_samples ** (1.0 / 3.0))

        r = torch.linspace(0.0, 1.0, n, device=device)
        if not with_boundary:
            r = r[:-1]
        r = r * self.radius.to(device)

        theta = torch.linspace(0.0, 2 * torch.pi, n, device=device)
        phi = torch.linspace(0.0, torch.pi, n, device=device)

        R, T, P = torch.meshgrid(r, theta, phi, indexing="ij")

        x = self.center[0, 0] + R * torch.sin(P) * torch.cos(T)
        y = self.center[0, 1] + R * torch.sin(P) * torch.sin(T)
        z = self.center[0, 2] + R * torch.cos(P)

        return torch.stack([x, y, z], dim=-1).reshape(-1, 3)

    def on_sample(
            self,
            num_samples: int,
            with_normal: bool = False,
            separate: bool = False,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, ...],
        Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
    ]:
        """
        Sample points on the spherical boundary of the ball.

        Args:
            num_samples (int): Target number of boundary samples.
            with_normal (bool): If True, also return outward unit normals.
            separate (bool): If True, wrap the result in a tuple to match
                the interface of composite geometries.

        Returns:
            - If ``with_normal=False``:
                * ``separate=False``: torch.Tensor of shape (N, 3)
                * ``separate=True``:  (points,)
            - If ``with_normal=True``:
                * ``separate=False``: (points, normals)
                * ``separate=True``:  ((points, normals),)

        Notes:
            - Boundary points are generated using a tensor-product grid
              in spherical coordinates.
            - Normals are computed analytically as
              ``(p - center) / radius``.
        """
        device = self.center.device
        n = int(num_samples ** 0.5)

        theta = torch.linspace(0.0, 2 * torch.pi, n, device=device)
        phi = torch.linspace(0.0, torch.pi, n, device=device)

        T, P = torch.meshgrid(theta, phi, indexing="ij")
        R = self.radius.to(device)

        x = self.center[0, 0] + R * torch.sin(P) * torch.cos(T)
        y = self.center[0, 1] + R * torch.sin(P) * torch.sin(T)
        z = self.center[0, 2] + R * torch.cos(P)

        points = torch.stack([x, y, z], dim=-1).reshape(-1, 3)
        normals = (points - self.center.to(device)) / self.radius.to(device)

        if with_normal:
            return (
                (points, normals)
                if not separate
                else ((points, normals),)
            )

        return points if not separate else (points,)


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

    def on_sample(self, num_samples: int, with_normal=False, separate=False) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, ...],
        Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
    ]:
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

    def on_sample(self, num_samples: int, with_normal: bool = False, separate=False) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, ...],
        Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
    ]:
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

    def on_sample(self, num_samples: int, with_normal: bool = False, separate=False) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, ...],
        Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
    ]:
        bounding_box = self.get_bounding_box()
        x_on = []
        if not with_normal:
            x_ = self.in_sample(num_samples // (2 * self.dim), with_boundary=True)
            for i in range(self.dim):
                for j in range(2):
                    x = x_.clone()
                    x[:, i] = bounding_box[2 * i + j]
                    x_on.append(x)

        return torch.cat(x_on, dim=0) if not separate else tuple(x_on)

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

    def on_sample(self, num_samples: int = None, with_normal: bool = False, separate=False) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, ...],
        Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
    ]:
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
