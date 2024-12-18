# -*- coding: utf-8 -*-
"""
Created on 2024/12/15

@author: Yifei Sun
"""
from .utils import *


class GeometryBase(ABC):
    def __init__(self, dim: Optional[int] = None, intrinsic_dim: Optional[int] = None):
        self.dim = dim if dim is not None else 0
        self.intrinsic_dim = intrinsic_dim if intrinsic_dim is not None else dim
        self.boundary: List = []

    def __eq__(self, other):
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
    def get_bounding_box(self):
        """
        Get the bounding box of the geometry.
        :return:
            For 2D: [x_min, x_max, y_min, y_max];
            For 3D: [x_min, x_max, y_min, y_max, z_min, z_max];
        """
        pass

    @abstractmethod
    def in_sample(self, num_samples: int, with_boundary: bool = False) -> torch.Tensor:
        """
        Generate samples within the geometry.
        Args:
            num_samples: The number of samples to generate.
            with_boundary: Whether to include boundary points in the samples.
        Returns:
            A list of points sampled from the geometry, where each point is a tuple of coordinates.
        """
        pass

    @abstractmethod
    def on_sample(self, num_samples: int) -> torch.Tensor:
        """
        Generate samples on the boundary of the geometry.
        Args:
            num_samples: The number of samples to generate.
        Returns:
            A list of points sampled from the boundary of the geometry, where each point is a tuple of coordinates.
        """
        pass

    @abstractmethod
    def intersection(self, other: "GeometryBase") -> Union["GeometryBase", None]:
        """
        Compute the intersection of this geometry with another geometry.
        Args:
            other: Another geometry object.
        Returns:
            A new GeometryBase object representing the intersection, or None if no intersection exists.
        """
        pass


class Point1D(GeometryBase):
    def __init__(self, x: torch.float64):
        super().__init__(dim=1, intrinsic_dim=0)
        self.x = x

    def get_bounding_box(self):
        return [self.x, self.x]

    def __eq__(self, other):
        if not isinstance(other, Point1D):
            return False

        return self.x == other.x

    def in_sample(self, num_samples: int, with_boundary: bool = False) -> torch.Tensor:
        return torch.tensor([[self.x]] * num_samples)

    def on_sample(self, num_samples: int) -> torch.Tensor:
        return torch.tensor([[self.x]] * num_samples)

    def intersection(self, other: "GeometryBase") -> Union["GeometryBase", None]:
        if isinstance(other, Point1D):
            if self.x == other.x:
                return self
            else:
                return None
        else:
            return other.intersection(self)


class Point2D(GeometryBase):
    def __init__(self, x: torch.float64, y: torch.float64):
        super().__init__(dim=2, intrinsic_dim=0)
        self.x = x
        self.y = y

    def get_bounding_box(self):
        return [self.x, self.x, self.y, self.y]

    def __eq__(self, other):
        if not isinstance(other, Point2D):
            return False

        return self.x == other.x and self.y == other.y

    def in_sample(self, num_samples: int, with_boundary: bool = False) -> torch.Tensor:
        return torch.tensor([[self.x, self.y]] * num_samples)

    def on_sample(self, num_samples: int) -> torch.Tensor:
        return torch.tensor([[self.x, self.y]] * num_samples)

    def intersection(self, other: "GeometryBase") -> Union["GeometryBase", None]:
        if isinstance(other, Point2D):
            if self.x == other.x and self.y == other.y:
                return self
            else:
                return None
        else:
            return other.intersection(self)


class Point3D(GeometryBase):
    def __init__(self, x: torch.float64, y: torch.float64, z: torch.float64):
        super().__init__(dim=3, intrinsic_dim=0)
        self.x = x
        self.y = y
        self.z = z

    def get_bounding_box(self):
        return [self.x, self.x, self.y, self.y, self.z, self.z]

    def __eq__(self, other):
        if not isinstance(other, Point3D):
            return False

        return self.x == other.x and self.y == other.y and self.z == other.z

    def in_sample(self, num_samples: int, with_boundary: bool = False) -> torch.Tensor:
        return torch.tensor([[self.x, self.y, self.z]] * num_samples)

    def on_sample(self, num_samples: int) -> torch.Tensor:
        return torch.tensor([[self.x, self.y, self.z]] * num_samples)

    def intersection(self, other: "GeometryBase") -> Union["GeometryBase", None]:
        if isinstance(other, Point3D):
            if self.x == other.x and self.y == other.y and self.z == other.z:
                return self
            else:
                return None
        else:
            return other.intersection(self)


class Line1D(GeometryBase):
    def __init__(self, x1: torch.float64, x2: torch.float64):
        super().__init__(dim=1, intrinsic_dim=1)
        self.x1 = x1
        self.x2 = x2
        self.boundary = [Point1D(x1), Point1D(x2)]

    def get_bounding_box(self):
        return [self.x1, self.x2] if self.x1 < self.x2 else [self.x2, self.x1]

    def in_sample(self, num_samples: int, with_boundary: bool = False) -> torch.Tensor:
        if with_boundary:
            return torch.linspace(self.x1, self.x2, num_samples).reshape(-1, 1)
        else:
            return torch.linspace(self.x1, self.x2, num_samples + 2)[1:-1].reshape(-1, 1)

    def on_sample(self, num_samples: int) -> torch.Tensor:
        a = self.boundary[0].in_sample(num_samples // 2, with_boundary=True)
        b = self.boundary[1].in_sample(num_samples // 2, with_boundary=True)
        return torch.cat([a, b], dim=0)

    def intersection(self, other: "GeometryBase") -> Union["GeometryBase", None]:
        if isinstance(other, Line1D):
            if self.x1 == other.x1 and self.x2 == other.x2:
                return self
            else:
                return None
        else:
            return other.intersection(self)


class Line2D(GeometryBase):
    def __init__(self, x1: torch.float64, y1: torch.float64, x2: torch.float64, y2: torch.float64):
        super().__init__(dim=2, intrinsic_dim=1)
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.boundary = [Point2D(x1, y1), Point2D(x2, y2)]

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

    def on_sample(self, num_samples: int) -> torch.Tensor:
        a = self.boundary[0].in_sample(num_samples // 2, with_boundary=True)
        b = self.boundary[1].in_sample(num_samples // 2, with_boundary=True)
        return torch.cat([a, b], dim=0)

    def intersection(self, other: "GeometryBase") -> Union["GeometryBase", None]:
        if isinstance(other, Line2D):
            if self.x1 == other.x1 and self.y1 == other.y1 and self.x2 == other.x2 and self.y2 == other.y2:
                return self
            else:
                return None
        else:
            return other.intersection(self)


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

    def get_bounding_box(self):
        x_min = min(self.x1, self.x2)
        x_max = max(self.x1, self.x2)
        y_min = min(self.y1, self.y2)
        y_max = max(self.y1, self.y2)
        z_min = min(self.z1, self.z2)
        z_max = max(self.z1, self.z2)
        return [x_min, x_max, y_min, y_max, z_min, z_max]

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

    def on_sample(self, num_samples: int) -> torch.Tensor:
        a = self.boundary[0].in_sample(num_samples // 2, with_boundary=True)
        b = self.boundary[1].in_sample(num_samples // 2, with_boundary=True)
        return torch.cat([a, b], dim=0)

    def intersection(self, other: "GeometryBase") -> Union["GeometryBase", None]:
        if isinstance(other, Line3D):
            if self.x1 == other.x1 and self.y1 == other.y1 and self.z1 == other.z1 and self.x2 == other.x2 and self.y2 == other.y2 and self.z2 == other.z2:
                return self
            else:
                return None
        else:
            return other.intersection(self)


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

    def get_bounding_box(self):
        x_min = self.center[0, 0] - self.radius[0, 0]
        x_max = self.center[0, 0] + self.radius[0, 0]
        y_min = self.center[0, 1] - self.radius[0, 1]
        y_max = self.center[0, 1] + self.radius[0, 1]
        return [x_min, x_max, y_min, y_max]

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

    def on_sample(self, num_samples: int) -> torch.Tensor:
        a = self.boundary[0].in_sample(num_samples // 4, with_boundary=True)
        b = self.boundary[1].in_sample(num_samples // 4, with_boundary=True)
        c = self.boundary[2].in_sample(num_samples // 4, with_boundary=True)
        d = self.boundary[3].in_sample(num_samples // 4, with_boundary=True)
        return torch.cat([a, b, c, d], dim=0)

    def intersection(self, other: "GeometryBase") -> Union["GeometryBase", None]:
        if isinstance(other, Square2D):
            if torch.all(self.center == other.center) and torch.all(self.radius == other.radius):
                return self
            else:
                return None
        else:
            return other.intersection(self)


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

    def get_bounding_box(self):
        x_min = self.center[0, 0] - self.radius[0, 0]
        x_max = self.center[0, 0] + self.radius[0, 0]
        y_min = self.center[0, 1] - self.radius[0, 1]
        y_max = self.center[0, 1] + self.radius[0, 1]
        z_min = self.center[0, 2] - self.radius[0, 2]
        z_max = self.center[0, 2] + self.radius[0, 2]
        return [x_min, x_max, y_min, y_max, z_min, z_max]

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

    def on_sample(self, num_samples: int) -> torch.Tensor:
        a = self.boundary[0].in_sample(num_samples // 4, with_boundary=True)
        b = self.boundary[1].in_sample(num_samples // 4, with_boundary=True)
        c = self.boundary[2].in_sample(num_samples // 4, with_boundary=True)
        d = self.boundary[3].in_sample(num_samples // 4, with_boundary=True)
        return torch.cat([a, b, c, d], dim=0)

    def intersection(self, other: "GeometryBase") -> Union["GeometryBase", None]:
        if isinstance(other, Square3D):
            if torch.all(self.center == other.center) and torch.all(self.radius == other.radius):
                return self
            else:
                return None
        else:
            return other.intersection(self)


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

    def get_bounding_box(self):
        x_min = self.center[0, 0] - self.radius[0, 0]
        x_max = self.center[0, 0] + self.radius[0, 0]
        y_min = self.center[0, 1] - self.radius[0, 1]
        y_max = self.center[0, 1] + self.radius[0, 1]
        z_min = self.center[0, 2] - self.radius[0, 2]
        z_max = self.center[0, 2] + self.radius[0, 2]
        return [x_min, x_max, y_min, y_max, z_min, z_max]

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

    def on_sample(self, num_samples: int) -> torch.Tensor:
        samples = []
        for square in self.boundary:
            samples.append(square.in_sample(num_samples // 6, with_boundary=True))
        return torch.cat(samples, dim=0)

    def intersection(self, other: "GeometryBase") -> Union["GeometryBase", None]:
        if isinstance(other, Cube3D):
            if torch.all(self.center == other.center) and torch.all(self.radius == other.radius):
                return self
            else:
                return None
        else:
            return other.intersection(self)


class Arc2D(GeometryBase):
    def __init__(self, center: Union[torch.Tensor, List, Tuple], radius: Union[torch.Tensor, List, Tuple],
                 start_angle: torch.float64, end_angle: torch.float64):
        super().__init__(dim=2, intrinsic_dim=1)
        self.center = torch.tensor(center).view(1, -1)
        self.radius = torch.tensor(radius).view(1, -1)


class Arc3D(GeometryBase):
    pass


class Circle2D(GeometryBase):
    pass


class Circle3D(GeometryBase):
    pass
