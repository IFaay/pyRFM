# -*- coding: utf-8 -*-
"""
Created on 2024/12/15

@author: Yifei Sun
"""
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
        return UnionGeometry(self, other)

    def __sub__(self, other: 'GeometryBase') -> 'GeometryBase':
        return UnionGeometry(self, ComplementGeometry(other))


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
        if self.dim == 1:
            return [max(boxA[0], boxB[0]), min(boxA[1], boxB[1])]


class ComplementGeometry(GeometryBase):
    def __init__(self, geom: GeometryBase):
        super().__init__()
        self.geom = geom
        self.dim = geom.dim
        self.intrinsic_dim = geom.intrinsic_dim
        self.boundary = [*geom.boundary]

    def sdf(self, p: torch.Tensor):
        return -self.geom.sdf(p)

    pass


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

        return torch.abs(p - (self.x1 + self.x2) / 2) - torch.abs(self.x2 - self.x1) / 2

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

    def in_sample(self, num_samples: int, with_boundary: bool = False) -> torch.Tensor:
        num_samples = int(num_samples ** (1 / 2))
        if with_boundary:
            x = torch.linspace(self.center[0, 0] - self.radius[0, 0], self.center[0, 0] + self.radius[0, 0],
                               num_samples)
            y = torch.linspace(self.center[0, 1] - self.radius[0, 1], self.center[0, 1] + self.radius[0, 1],
                               num_samples)
            X, Y = torch.meshgrid(x, y, indexing='ij')
            return torch.cat([X.reshape(-1, 1), Y.reshape(-1, 1)], dim=1)
        else:
            x = torch.linspace(self.center[0, 0] - self.radius[0, 0], self.center[0, 0] + self.radius[0, 0],
                               num_samples + 2)[1:-1]
            y = torch.linspace(self.center[0, 1] - self.radius[0, 1], self.center[0, 1] + self.radius[0, 1],
                               num_samples + 2)[1:-1]
            X, Y = torch.meshgrid(x, y, indexing='ij')
            return torch.cat([X.reshape(-1, 1), Y.reshape(-1, 1)], dim=1)

    def on_sample(self, num_samples: int, with_normal: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        a = self.boundary[0].in_sample(num_samples // 4, with_boundary=True)
        b = self.boundary[1].in_sample(num_samples // 4, with_boundary=True)
        c = self.boundary[2].in_sample(num_samples // 4, with_boundary=True)
        d = self.boundary[3].in_sample(num_samples // 4, with_boundary=True)
        if with_normal:
            return torch.cat([a, b, c, d], dim=0), torch.cat(
                [
                    torch.tensor([[0.0, -1.0]] * (num_samples // 4)),
                    torch.tensor([[1.0, 0.0]] * (num_samples // 4)),
                    torch.tensor([[0.0, 1.0]] * (num_samples // 4)),
                    torch.tensor([[-1.0, 0.0]] * (num_samples // 4))
                ], dim=0)
        else:
            return torch.cat([a, b, c, d], dim=0)


class Square3D(GeometryBase):
    def __init__(self, center: torch.Tensor, radius: torch.Tensor):
        super().__init__(dim=3, intrinsic_dim=2)
        self.center = torch.tensor(center).view(1, -1)
        self.radius = torch.tensor(radius).view(1, -1)

        for i in range(3):
            if radius[0, i] == 0.0:
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
        self.center = torch.tensor(center).view(1, -1)
        self.radius = torch.tensor(radius).view(1, -1)
        offsets = [
            [self.radius[0, 0], 0, 0],
            [-self.radius[0, 0], 0, 0],
            [0, self.radius[0, 1], 0],
            [0, -self.radius[0, 1], 0],
            [0, 0, self.radius[0, 2]],
            [0, 0, -self.radius[0, 2]]
        ]
        self.boundary = [
            Square3D(self.center + torch.tensor(offset),
                     torch.tensor([self.radius[0, i] for i in range(3) if offset[i] == 0]))
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

    def in_sample(self, num_samples: int, with_boundary: bool = False) -> torch.Tensor:
        num_samples = int(num_samples ** (1 / 3))
        if with_boundary:
            samples = []
            for square in self.boundary:
                samples.append(square.in_sample(num_samples, with_boundary=True))
            return torch.cat(samples, dim=0)
        else:
            samples = []
            for square in self.boundary:
                samples.append(square.in_sample(num_samples + 2, with_boundary=False))
            return torch.cat(samples, dim=0)

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
        a = self.boundary[0].in_sample(num_samples, with_boundary=True)
        if with_normal:
            return a, torch.tensor([[1.0, 0.0]] * num_samples)
        else:
            return a


class Circle2D(GeometryBase):
    def __init__(self, center: Union[torch.Tensor, List, Tuple],
                 radius: torch.float64):
        super().__init__(dim=2, intrinsic_dim=1)
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
        if with_boundary:
            r = torch.linspace(0.0, self.radius, num_samples)
        else:
            r = torch.linspace(0.0, self.radius, num_samples + 1)[:-1]

        theta = torch.linspace(0.0, 2 * torch.pi, num_samples)
        R, T = torch.meshgrid(r, theta, indexing='ij')
        x = self.center[0, 0] + R * torch.cos(T)
        y = self.center[0, 1] + R * torch.sin(T)
        return torch.cat([x.reshape(-1, 1), y.reshape(-1, 1)], dim=1)

    def on_sample(self, num_samples: int, with_normal: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        theta = torch.linspace(0.0, 2 * torch.pi, num_samples).reshape(-1, 1)
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
        pass


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
