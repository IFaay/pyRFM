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

    def __init__(self, dim: Optional[int] = None, intrinsic_dim: Optional[int] = None):
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
    """
    A class to represent the empty geometry.
    """

    def __init__(self):
        super().__init__(dim=0, intrinsic_dim=0)
        self.boundary = []

    def sdf(self, p: torch.Tensor):
        """
        For empty geometry, the signed distance to the geometry is always infinity.
        """
        return torch.full_like(p, float('inf'))

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

    def get_bounding_box(self) -> List[float]:
        bounding_box_geom = self.geom.get_bounding_box()
        return [
            float('-inf') if i % 2 == 0 else float('inf')
            for d in range(self.dim)
            for i in range(2)
        ]

    def in_sample(self, num_samples: int, with_boundary: bool = False) -> torch.Tensor:
        return self.geom.in_sample(num_samples, with_boundary)

    def on_sample(self, num_samples: int, with_normal: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        return self.geom.on_sample(num_samples, with_normal)


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
                [
                    torch.tensor([[(self.x2 - self.x1) / abs(self.x2 - self.x1)]] * (num_samples // 2)),
                    torch.tensor([[(self.x1 - self.x2) / abs(self.x1 - self.x2)]] * (num_samples // 2))
                ], dim=0)
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
            return torch.cat([a, b], dim=0), torch.cat(
                [
                    torch.tensor([[(self.x2 - self.x1) / abs(self.x2 - self.x1),
                                   (self.y2 - self.y1) / abs(self.y2 - self.y1)]] * (num_samples // 2)),
                    torch.tensor([[(self.x1 - self.x2) / abs(self.x1 - self.x2),
                                   (self.y1 - self.y2) / abs(self.y1 - self.y2)]] * (num_samples // 2))
                ], dim=0)
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
            return torch.cat([a, b], dim=0), torch.cat(
                [
                    torch.tensor([[(self.x2 - self.x1) / abs(self.x2 - self.x1),
                                   (self.y2 - self.y1) / abs(self.y2 - self.y1),
                                   (self.z2 - self.z1) / abs(self.z2 - self.z1)]] * (num_samples // 2)),
                    torch.tensor([[(self.x1 - self.x2) / abs(self.x1 - self.x2),
                                   (self.y1 - self.y2) / abs(self.y1 - self.y2),
                                   (self.z1 - self.z2) / abs(self.z1 - self.z2)]] * (num_samples // 2))
                ], dim=0)
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
            torch.max(d, dim=1, keepdim=True).values,
            max=0.0)

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

    def on_sample(
            self,
            num_samples: Union[int, List[int], Tuple],
            with_normal: bool = False,
            separate: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:

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
                normals = torch.cat([
                    torch.tensor([[0.0, -1.0]] * nums[0]),  # bottom
                    torch.tensor([[1.0, 0.0]] * nums[1]),  # right
                    torch.tensor([[0.0, 1.0]] * nums[2]),  # top
                    torch.tensor([[-1.0, 0.0]] * nums[3])  # left
                ], dim=0)
                return torch.cat([a, b, c, d], dim=0), normals
            else:
                return torch.cat([a, b, c, d], dim=0)
        else:
            if with_normal:
                return (
                    a, torch.tensor([[0.0, -1.0]] * nums[0]),
                    b, torch.tensor([[1.0, 0.0]] * nums[1]),
                    c, torch.tensor([[0.0, 1.0]] * nums[2]),
                    d, torch.tensor([[-1.0, 0.0]] * nums[3])
                )
            else:
                return a, b, c, d


class Square3D(GeometryBase):
    def __init__(self, center: torch.Tensor, radius: torch.Tensor):
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
                self.boundary = [
                    Line3D(*p1, *p2),
                    Line3D(*p2, *p3),
                    Line3D(*p3, *p4),
                    Line3D(*p4, *p1),
                ]
                break

    def sdf(self, p: torch.Tensor):
        d = torch.abs(p - self.center) - self.radius
        return torch.norm(torch.clamp(d, min=0.0), dim=1, keepdim=True) + torch.clamp(
            torch.max(d, dim=1, keepdim=True).values,
            max=0.0)

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
        offsets = [
            [self.radius[0, 0], 0.0, 0.0],
            [-self.radius[0, 0], 0.0, 0.0],
            [0.0, self.radius[0, 1], 0.0],
            [0.0, -self.radius[0, 1], 0.0],
            [0.0, 0.0, self.radius[0, 2]],
            [0.0, 0.0, -self.radius[0, 2]]
        ]
        self.boundary = [
            Square3D(self.center + torch.tensor(offset),
                     torch.tensor([self.radius[0, i] if offset[i] == 0.0 else 0.0 for i in range(3)]))
            for offset in offsets
        ]

    def sdf(self, p: torch.Tensor):
        d = torch.abs(p - self.center) - self.radius
        return torch.norm(torch.clamp(d, min=0.0), dim=1, keepdim=True) + torch.clamp(
            torch.max(d, dim=1, keepdim=True).values,
            max=0.0)

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
    def __init__(self, center: Union[torch.Tensor, List, Tuple],
                 radius: torch.float64):
        super().__init__(dim=2, intrinsic_dim=1)
        self.center = torch.tensor(center).view(1, -1)
        self.radius = radius
        self.boundary = [Point2D(self.center[0, 0] + self.radius, self.center[0, 1])]

    def sdf(self, p: torch.Tensor):
        d = torch.norm(p - self.center, dim=1, keepdim=True) - self.radius
        return torch.abs(d)

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
    def __init__(self, center: Union[torch.Tensor, List, Tuple],
                 radius: torch.float64):
        super().__init__(dim=2, intrinsic_dim=2)
        self.center = torch.tensor(center).view(1, -1)
        self.radius = radius
        self.boundary = [CircleArc2D(center, radius)]

    def sdf(self, p: torch.Tensor):
        return torch.norm(p - self.center, dim=1, keepdim=True) - self.radius

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
    def __init__(self, center: Union[torch.Tensor, List, Tuple],
                 radius: torch.float64):
        super().__init__(dim=3, intrinsic_dim=2)
        self.center = torch.tensor(center).view(1, -1)
        self.radius = radius
        self.boundary = [Circle2D(self.center, self.radius)]

    def sdf(self, p: torch.Tensor):
        d = torch.norm(p - self.center, dim=1, keepdim=True) - self.radius
        return torch.abs(d)

    def get_bounding_box(self):
        x_min = self.center[0, 0] - self.radius
        x_max = self.center[0, 0] + self.radius
        y_min = self.center[0, 1] - self.radius
        y_max = self.center[0, 1] + self.radius
        z_min = self.center[0, 2] - self.radius
        z_max = self.center[0, 2] + self.radius
        return [x_min.item(), x_max.item(), y_min.item(), y_max.item(), z_min.item(), z_max.item()]

    def in_sample(self, num_samples: int, with_boundary: bool = False) -> torch.Tensor:
        num_samples = int(num_samples ** (1 / 2))
        theta = torch.linspace(0.0, 2 * torch.pi, num_samples).reshape(-1, 1)
        phi = torch.linspace(0.0, torch.pi, num_samples).reshape(-1, 1)
        R, T, P = torch.meshgrid(self.radius, theta, phi, indexing='ij')
        x = self.center[0, 0] + R * torch.sin(P) * torch.cos(T)
        y = self.center[0, 1] + R * torch.sin(P) * torch.sin(T)
        z = self.center[0, 2] + R * torch.cos(P)
        return torch.cat([x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], dim=1)

    def on_sample(self, num_samples: int, with_normal: bool
    = False) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        raise NotImplementedError


class Ball3D(GeometryBase):
    def __init__(self, center: Union[torch.Tensor, List, Tuple],
                 radius: torch.float64):
        super().__init__(dim=3, intrinsic_dim=3)
        self.center = torch.tensor(center).view(1, -1)
        self.radius = radius
        self.boundary = [Sphere3D(self.center, self.radius)]

    def sdf(self, p: torch.Tensor):
        d = torch.norm(p - self.center, dim=1, keepdim=True) - self.radius
        return d

    def get_bounding_box(self):
        x_min = self.center[0, 0] - self.radius
        x_max = self.center[0, 0] + self.radius
        y_min = self.center[0, 1] - self.radius
        y_max = self.center[0, 1] + self.radius
        z_min = self.center[0, 2] - self.radius
        z_max = self.center[0, 2] + self.radius
        return [x_min.item(), x_max.item(), y_min.item(), y_max.item(), z_min.item(), z_max.item()]

    def in_sample(self, num_samples: int, with_boundary: bool = False) -> torch.Tensor:
        num_samples = int(num_samples ** (1 / 3))
        if with_boundary:
            r = torch.linspace(0.0, self.radius, num_samples).reshape(-1, 1)
        else:
            r = torch.linspace(0.0, self.radius, num_samples + 1)[:-1].reshape(-1, 1)
        theta = torch.linspace(0.0, 2 * torch.pi, num_samples).reshape(-1, 1)
        phi = torch.linspace(0.0, torch.pi, num_samples).reshape(-1, 1)
        R, T, P = torch.meshgrid(r, theta, phi, indexing='ij')
        x = self.center[0, 0] + R * torch.sin(P) * torch.cos(T)
        y = self.center[0, 1] + R * torch.sin(P) * torch.sin(T)
        z = self.center[0, 2] + R * torch.cos(P)
        return torch.cat([x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], dim=1)

    def on_sample(self, num_samples: int, with_normal: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        theta = torch.linspace(0.0, 2 * torch.pi, num_samples).reshape(-1, 1)
        phi = torch.linspace(0.0, torch.pi, num_samples).reshape(-1, 1)
        R, T, P = torch.meshgrid(self.radius, theta, phi, indexing='ij')
        x = self.center[0, 0] + R * torch.sin(P) * torch.cos(T)
        y = self.center[0, 1] + R * torch.sin(P) * torch.sin(T)
        z = self.center[0, 2] + R * torch.cos(P)
        a = torch.cat([x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], dim=1)
        an = (a - self.center) / self.radius
        if with_normal:
            return a, an
        else:
            return a


class Polygon2D(GeometryBase):
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

    def on_sample(self, num_samples: int, with_normal=False) -> Union[
        torch.Tensor, Tuple[torch.Tensor, ...]]:
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
    def __init__(self, vertices: torch.Tensor):
        super().__init__(dim=3, intrinsic_dim=2)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("Vertices must be a tensor of shape (N, 3).")
        self.vertices = vertices
        self.boundary = [Line3D(vertices[i, 0], vertices[i, 1], vertices[i, 2],
                                vertices[(i + 1) % vertices.shape[0], 0],
                                vertices[(i + 1) % vertices.shape[0], 1],
                                vertices[(i + 1) % vertices.shape[0], 2]) for i in range(vertices.shape[0])]

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
        n = torch.cross(v1 - v0, v2 - v0)  # normal = (v1-v0) x (v2-v0)
        if torch.allclose(n, torch.zeros_like(n)):
            raise ValueError("The given vertices are degenerate (normal is zero).")

        # Normalize the normal vector
        n = n / torch.norm(n)

        # 3. Build a local orthonormal frame {e1, e2, n}
        #    We want e1 and e2 to lie in the plane, both perpendicular to n.
        e1 = self._find_orthonormal_vector(n)
        e2 = torch.cross(n, e1)

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
        c1 = torch.cross(n, ex)
        if torch.norm(c1) > 1e-7:
            e1 = c1 / torch.norm(c1)
            return e1

        # Otherwise use ey
        c2 = torch.cross(n, ey)
        if torch.norm(c2) > 1e-7:
            e1 = c2 / torch.norm(c2)
            return e1

        # Fallback: n might be (0, 0, ±1). Then crossing with ex or ey is 0.
        # So let's cross with ez = (0, 0, 1)
        ez = torch.tensor([0.0, 0.0, 1.0], device=n.device, dtype=n.dtype)
        c3 = torch.cross(n, ez)
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
            torch.max(d, dim=1, keepdim=True).values,
            max=0.0)

    def get_bounding_box(self) -> List[float]:
        bounding_box = []
        for i in range(self.dim):
            bounding_box.append((self.center[0, i] - self.radius[0, i]).item())
            bounding_box.append((self.center[0, i] + self.radius[0, i]).item())
        return bounding_box

    def in_sample(self, num_samples: int, with_boundary: bool = False) -> torch.Tensor:
        x_in = torch.rand((num_samples, self.dim), dtype=self.dtype, device=self.device)
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
