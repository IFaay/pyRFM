# -*- coding: utf-8 -*-
"""
Created on 2024/12/13

@author: Yifei Sun
"""
import time

import torch

from .geometry import GeometryBase, Line1D, Square2D, Cube3D
from .voronoi import Voronoi
from .utils import *


class RFBase(ABC):
    def __init__(self, dim: int, center: torch.Tensor, radius: torch.Tensor,
                 activation: nn.Module, n_hidden: int,
                 gen: torch.Generator = None,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        self.dtype = dtype if dtype is not None else torch.tensor(0.).dtype
        self.device = device if device is not None else torch.tensor(0.).device

        self.dim: int = dim
        self.center: torch.Tensor = center.to(dtype=self.dtype, device=self.device).view(1, -1)
        self.radius: torch.Tensor = radius.to(dtype=self.dtype, device=self.device).view(1, -1)
        self.activation: nn.Module = activation

        self.n_hidden: int = n_hidden

        if gen is not None:
            self.gen = gen
        else:
            self.gen = torch.Generator(device=self.device)
            self.gen.manual_seed(100)

        self.weights: torch.Tensor = torch.rand((self.dim, self.n_hidden), generator=self.gen, dtype=self.dtype,
                                                device=self.device) * 2 - 1
        self.biases: torch.Tensor = torch.rand((1, self.n_hidden), generator=self.gen, dtype=self.dtype,
                                               device=self.device) * 2 - 1

        self.x_buff_: torch.Tensor or None = None
        self.features_buff_: torch.Tensor or None = None
        pass

    def empty_cache(self):
        self.x_buff_ = None
        self.features_buff_ = None

    def __call__(self, x: torch.Tensor, *args, **kwargs):
        return self.forward(x)

    def __repr__(self):
        return f"\nRFBase(dim={self.dim}, center={self.center}, radius={self.radius}, activation={self.activation}, n_hidden={self.n_hidden})"

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def first_derivative(self, x: torch.Tensor, axis: int) -> torch.Tensor:
        pass

    @abstractmethod
    def second_derivative(self, x: torch.Tensor, axis1: int, axis2: int) -> torch.Tensor:
        pass

    @abstractmethod
    def higher_order_derivative(self, x: torch.Tensor, order: Union[torch.Tensor, List]) -> torch.Tensor:
        pass


class RFTanH(RFBase):
    def __init__(self, dim: int, center: torch.Tensor, radius: torch.Tensor,
                 n_hidden: int,
                 gen: torch.Generator = None,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__(dim, center, radius, nn.Tanh(), n_hidden, gen, dtype, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.dim:
            raise ValueError('Input dimension mismatch')
        with torch.no_grad():
            self.x_buff_ = x
            self.features_buff_ = torch.tanh(
                torch.matmul((x - self.center) / self.radius, self.weights) + self.biases)
            return self.features_buff_

    def first_derivative(self, x: torch.Tensor, axis: int) -> torch.Tensor:
        if x.shape[1] != self.dim:
            raise ValueError('Input dimension mismatch')

        if axis >= self.dim:
            raise ValueError('Axis out of range')

        with torch.no_grad():
            if (self.x_buff_ is not None) and (self.x_buff_ is x or torch.equal(self.x_buff_, x)):
                pass
            else:
                self.forward(x)

            return (1 - torch.pow(self.features_buff_, 2)) * (self.weights[[axis], :] / self.radius[0, axis])

    def second_derivative(self, x: torch.Tensor, axis1: int, axis2: int) -> torch.Tensor:
        if x.shape[1] != self.dim:
            raise ValueError('Input dimension mismatch')

        if axis1 >= self.dim:
            raise ValueError('Axis1 out of range')

        if axis2 >= self.dim:
            raise ValueError('Axis2 out of range')

        with torch.no_grad():
            if (self.x_buff_ is not None) and (self.x_buff_ is x or torch.equal(self.x_buff_, x)):
                pass
            else:
                self.forward(x)

            return -2 * self.features_buff_ * (1 - torch.pow(self.features_buff_, 2)) * \
                (self.weights[[axis1], :] / self.radius[0, axis1]) * (
                        self.weights[[axis2], :] / self.radius[0, axis2])

    def higher_order_derivative(self, x: torch.Tensor, order: Union[torch.Tensor, List]) -> torch.Tensor:
        if isinstance(order, List):
            order = torch.tensor(order, dtype=self.dtype, device=self.device)
        if x.shape[1] != self.dim:
            raise ValueError('Input dimension mismatch')

        if order.shape[0] != self.dim:
            raise ValueError('Order dimension mismatch')

        n_order = order.sum()
        if n_order <= 0:
            raise ValueError('Order must be positive')
        if self.x_buff_ is x or torch.equal(self.x_buff_, x):
            t = self.features_buff_
        else:
            t = torch.tanh(
                torch.matmul((x - self.center) / self.radius, self.weights) + self.biases)
        p_n_minus_1 = 1 - t ** 2
        p_n_minus_2 = t
        p_n = 1
        for n in range(2, n_order + 1):
            p_n = -(2 * n - 1) * t * p_n_minus_1 - (1 - t ** 2) * p_n_minus_2
            p_n_minus_2 = p_n_minus_1
            p_n_minus_1 = p_n

        for i in range(order.shape[0]):
            for _ in range(order[i]):
                p_n *= (self.weights[[i], :] / self.radius[0, i])

        return p_n


class POUBase(ABC):
    def __init__(self, center: torch.Tensor, radius: torch.Tensor,
                 dtype: torch.dtype = None,
                 device: torch.device = None
                 ):
        self.dtype = dtype if dtype is not None else torch.tensor(0.).dtype
        self.device = device if device is not None else torch.tensor(0.).device
        self.center = center.to(dtype=self.dtype, device=self.device).view(1, -1)
        self.radius = radius.to(dtype=self.dtype, device=self.device).view(1, -1)
        self.func = torch.nn.Identity
        self.d_func = torch.nn.Identity
        self.d2_func = torch.nn.Identity
        self.set_func()

    def __call__(self, x: torch.Tensor, *args, **kwargs):
        return self.forward(x)

    def __repr__(self):
        return f"POUBase(center={self.center}, radius={self.radius}, func={self.func}, dtype={self.dtype}, device={self.device})"

    @abstractmethod
    def set_func(self):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.center.shape[1] or x.shape[1] != self.radius.shape[1]:
            raise ValueError('Input dimension mismatch')
        x_ = (x - self.center) / self.radius
        prod = torch.ones((x_.shape[0], 1), dtype=self.dtype, device=self.device)
        for d in range(x_.shape[1]):
            prod *= self.func(x_[:, [d]])

        return prod

    def first_derivative(self, x: torch.Tensor, axis: int) -> torch.Tensor:
        if x.shape[1] != self.center.shape[1] or x.shape[1] != self.radius.shape[1]:
            raise ValueError('Input dimension mismatch')

        if axis >= x.shape[1]:
            raise ValueError('Axis out of range')
        x_ = (x - self.center) / self.radius

        prod = torch.ones((x_.shape[0], 1), dtype=self.dtype, device=self.device)
        for d in range(x_.shape[1]):
            if d == axis:
                prod *= self.d_func(x_[:, [d]]) / self.radius[0, d]
            else:
                prod *= self.func(x_[:, [d]])

        return prod

    def second_derivative(self, x: torch.Tensor, axis1: int, axis2: int) -> torch.Tensor:
        # Check for input dimension mismatch
        if x.shape[1] != self.center.shape[1] or x.shape[1] != self.radius.shape[1]:
            raise ValueError('Input dimension mismatch')

        # Check if axis1 is within valid range
        if axis1 >= x.shape[1]:
            raise ValueError('Axis1 out of range')

        # Check if axis2 is within valid range
        if axis2 >= x.shape[1]:
            raise ValueError('Axis2 out of range')

        x_ = (x - self.center) / self.radius
        prod = torch.ones((x_.shape[0], 1), dtype=self.dtype, device=self.device)
        if axis1 == axis2:
            for d in range(x_.shape[1]):
                if d == axis1:
                    prod *= self.d2_func(x_[:, [d]]) / self.radius[0, d] ** 2
                else:
                    prod *= self.func(x_[:, [d]])
        else:
            for d in range(x_.shape[1]):
                if d == axis1 or d == axis2:
                    prod *= self.d_func(x_[:, [d]]) / self.radius[0, d]
                else:
                    prod *= self.func(x_[:, [d]])
        return prod

        pass

    def higher_order_derivative(self, x: torch.Tensor, order: Union[torch.Tensor, List]) -> torch.Tensor:
        pass


class PsiA(POUBase):
    def set_func(self):
        self.func = lambda x: torch.where(x < -1.0, 0.0, torch.where(x > 1.0, 0.0, 1.0))
        self.d_func = lambda x: torch.zeros((x.shape[0], 1), dtype=self.dtype, device=self.device)
        self.d2_func = lambda x: torch.zeros((x.shape[0], 1), dtype=self.dtype, device=self.device)


class PsiB(POUBase):
    def set_func(self):
        self.func = lambda x: torch.where(x < -5.0 / 4.0, 0.0,
                                          torch.where(x < -3.0 / 4.0,
                                                      1.0 / 2.0 * (1.0 + torch.sin(2 * torch.pi * x)),
                                                      torch.where(x <= 3.0 / 4.0, 1.0,
                                                                  torch.where(x <= 5.0 / 4.0, 1.0 / 2.0 * (
                                                                          1.0 - torch.sin(
                                                                      2 * torch.pi * x)), 0.0)))
                                          )
        self.d_func = lambda x: torch.where(x < -5.0 / 4.0, 0.0,
                                            torch.where(x < -3.0 / 4.0,
                                                        torch.pi * torch.cos(2 * torch.pi * x),
                                                        torch.where(x <= 3.0 / 4.0, 0.0,
                                                                    torch.where(x <= 5.0 / 4.0,
                                                                                -torch.pi * torch.cos(
                                                                                    2 * torch.pi * x), 0.0)))
                                            )
        self.d2_func = lambda x: torch.where(x < -5.0 / 4.0, 0.0,
                                             torch.where(x < -3.0 / 4.0,
                                                         -2 * torch.pi ** 2 * torch.sin(2 * torch.pi * x),
                                                         torch.where(x <= 3.0 / 4.0, 0.0,
                                                                     torch.where(x <= 5.0 / 4.0,
                                                                                 2 * torch.pi ** 2 * torch.sin(
                                                                                     2 * torch.pi * x), 0.0)))
                                             )


class PsiG(POUBase):
    def __init__(self, center: torch.Tensor, radius: torch.Tensor,
                 mu: torch.Tensor, sigma: torch.Tensor,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__(center,
                         radius,
                         dtype=dtype, device=device)

        self.mu = mu.to(dtype=self.dtype, device=self.device)
        self.sigma = sigma.to(dtype=self.dtype, device=self.device)

    def set_func(self):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def first_derivative(self, x: torch.Tensor, axis: int) -> torch.Tensor:
        pass

    def second_derivative(self, x: torch.Tensor, axis1: int, axis2: int) -> torch.Tensor:
        pass

    def higher_order_derivative(self, x: torch.Tensor, order: Union[torch.Tensor, List]) -> torch.Tensor:
        pass


class RFMBase(ABC):
    def __init__(self, dim: int,
                 n_hidden: int,
                 domain: Union[Tuple, List, GeometryBase], n_subdomains: Union[int, Tuple, List] = 1,
                 overlap: torch.float64 = 0.0,
                 rf=RFTanH,
                 pou=PsiB,
                 centers: Optional[torch.Tensor] = None,
                 radii: Optional[torch.Tensor] = None,
                 seed: int = 100,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        """
        Initialize the RFMBase class with arbitrary dimensions.

        :param dim: Number of dimensions.
        :param domain: List or tuple of min and max values for each dimension.
                       Example for 2D: [x_min, x_max, y_min, y_max]
                       Example for 3D: [x_min, x_max, y_min, y_max, z_min, z_max]
        :param n_subdomains: Either an integer (uniform subdivisions in all dimensions)
                             or a list/tuple specifying the subdivisions per dimension.
        :param overlap: Overlap between subdomains, must be between 0 (inclusive) and 1 (exclusive).
        :param rf: Random Feature class, must be a subclass of RFBase.
        :param pou: Partition of Unity class, must be a subclass of POUBase.
        :param centers: Optional tensor specifying the centers of subdomains.
        :param radii: Optional tensor specifying the radii of subdomains.
        :param seed: Random seed for reproducibility.
        :param dtype: Data type for tensors.
        :param device: Device to run the computations on.
        """
        self.dtype = dtype if dtype is not None else torch.tensor(0.).dtype
        self.device = device if device is not None else torch.tensor(0.).device
        self.dim = dim
        if isinstance(domain, GeometryBase):
            if domain.dim != self.dim:
                raise ValueError("Domain dimension mismatch.")
            else:
                self.domain = domain
        else:
            if len(domain) != 2 * dim:
                raise ValueError(f"Domain must contain {2 * dim} values (min and max for each dimension).")
            if dim == 1:
                self.domain = Line1D(domain[0], domain[1])
            elif dim == 2:
                self.domain = Square2D([(domain[0] + domain[1]) / 2.0, (domain[2] + domain[3]) / 2.0],
                                       [(domain[1] - domain[0]) / 2.0, (domain[3] - domain[2]) / 2.0])
            elif dim == 3:
                self.domain = Cube3D(
                    [(domain[0] + domain[1]) / 2.0, (domain[2] + domain[3]) / 2.0, (domain[4] + domain[5]) / 2.0],
                    [(domain[1] - domain[0]) / 2.0, (domain[3] - domain[2]) / 2.0, (domain[5] - domain[4]) / 2.0])
            else:
                raise ValueError("Only 1D, 2D, and 3D domains are supported.")

        # If n_subdomains is an integer, create uniform subdivisions
        if isinstance(n_subdomains, int):
            n_subdomains = [n_subdomains] * self.dim
        elif isinstance(n_subdomains, (list, tuple)) and len(n_subdomains) != self.dim:
            raise ValueError(f"n_subdomains must have {self.dim} elements when provided as a list or tuple.")

        # Validate overlap
        if not (0.0 <= overlap < 1.0):
            raise ValueError("Overlap must be between 0 (inclusive) and 1 (exclusive).")

        self.overlap = overlap

        # Compute centers and radii
        if centers is not None and radii is not None:
            self.centers = torch.tensor(centers, dtype=self.dtype, device=self.device)
            self.radii = torch.tensor(radii, dtype=self.dtype, device=self.device)
            if self.centers.shape[-1] != self.dim or self.radii.shape[-1] != self.dim:
                raise ValueError("Centers and radii must have the same number of dimensions as the domain.")
            elif self.centers.shape != self.radii.shape:
                raise ValueError("Centers and radii must have the same shape.")
        else:
            self.centers, self.radii = self._compute_centers_and_radii(n_subdomains)

        if not issubclass(rf, RFBase):
            raise ValueError("Random Feature must be a subclass of RFBase.")
        submodels = []
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(seed)
        for center, radius in zip(self.centers.view(-1, self.centers.shape[-1]),
                                  self.radii.view(-1, self.radii.shape[-1])):
            submodels.append(rf(dim, center, radius, n_hidden, gen=self.gen, dtype=dtype, device=device))
        self.submodels = Tensor(submodels, shape=n_subdomains)
        self.n_hidden = n_hidden

        if not issubclass(pou, POUBase):
            raise ValueError("Partition of Unity must be a subclass of POUBase.")
        pou_functions = []
        for center, radius in zip(self.centers.view(-1, self.centers.shape[-1]),
                                  self.radii.view(-1, self.radii.shape[-1])):
            pou_functions.append(pou(center, radius, dtype=dtype, device=device))
        self.pou_functions = Tensor(pou_functions, shape=n_subdomains)

        self.W: Union[Tensor, List, torch.tensor] = None
        self.A: Optional[torch.tensor] = None
        self.A_norm: Optional[torch.tensor] = None
        self.tau: Optional[torch.tensor] = None

    def add_c_condition(self, num_samples: int, order: int = 1):
        """
        Add a Continuity (c0 and c1) condition to the model.
        :param num_samples: number of interface points
        :param order: max order of the continuity condition
        :return: feature Tensor
        """
        if not isinstance(self.pou_functions[0], PsiA):
            warnings.warn("The POU function is not PsiA, the continuity condition may not be Appropriate.")

        if order < 0:
            raise ValueError("Order must be non-negative.")

        voronoi = Voronoi(self.domain, self.centers)
        interface_dict, _ = voronoi.interface_sample(num_samples)

        CFeatrues: List[torch.Tensor] = []

        for pair, point in interface_dict.items():
            if order >= 0:
                feature = self.features(point)
                for i in range(feature.numel()):
                    if i >= len(CFeatrues):
                        # Initialize CFeatures[i] if it does not exist
                        if i not in pair:
                            CFeatrues.append(torch.zeros_like(feature[i]))
                        else:
                            CFeatrues.append(feature[i] if i == pair[0] else -feature[i])
                    else:
                        # Update CFeatrues[i] if it exists
                        if i not in pair:
                            CFeatrues[i] = torch.cat([CFeatrues[i], torch.zeros_like(feature[i])], dim=0)
                        else:
                            CFeatrues[i] = torch.cat([CFeatrues[i], feature[i] if i == pair[0] else -feature[i]], dim=0)
            if order >= 1:
                center1 = self.centers.view(-1, self.centers.shape[-1])[pair[1]]
                center0 = self.centers.view(-1, self.centers.shape[-1])[pair[0]]
                normal = (center1 - center0) / torch.linalg.norm(center1 - center0)
                dFeatures = [self.features_derivative(point, d) for d in range(self.dim)]
                d_feature = dFeatures[0] * normal[0]
                for i in range(1, self.dim):
                    d_feature += dFeatures[i] * normal[i]

                for i in range(d_feature.numel()):
                    if i not in pair:
                        CFeatrues[i] = torch.cat([CFeatrues[i], torch.zeros_like(d_feature[i])], dim=0)
                    else:
                        CFeatrues[i] = torch.cat([CFeatrues[i], d_feature[i] if i == pair[0] else -d_feature[i]], dim=0)
        if order > 1:
            raise NotImplementedError("Higher order continuity conditions are not supported.")

        return Tensor(CFeatrues, shape=self.submodels.shape)

    def empty_cache(self):
        """
        Empty the cache for all submodels.
        """
        for submodel in self.submodels.flat_data:
            submodel.empty_cache()

    def __call__(self, x, *args, **kwargs):
        """
        Make the class callable and forward the input tensor.

        :param x: Input tensor.
        :return: Output tensor after forward pass.
        """
        return self.forward(x)

    def compute(self, A: torch.Tensor):
        """
        Compute the QR decomposition of matrix A.

        :param A: Input matrix.
        :return: Self.
        """
        A = A.to(dtype=self.dtype, device=self.device)
        self.A_norm = torch.linalg.norm(A, ord=2, dim=1, keepdim=True)
        A /= self.A_norm
        print("Decomposing the problem size of A: ", A.shape, "with solver QR")

        try:
            self.A, self.tau = torch.geqrf(A)
        except RuntimeError as e:
            if 'cusolver error' in str(e):
                raise RuntimeError("Out Of Memory Error")
            else:
                raise e

        return self

    def solve(self, b: torch.Tensor):
        """
        Solve the linear system Ax = b using the QR decomposition.

        :param b: Right-hand side tensor.
        """
        b = b.view(-1, 1).to(dtype=self.dtype, device=self.device)
        if self.A.shape[0] != b.shape[0]:
            raise ValueError("Input dimension mismatch.")
        b /= self.A_norm

        y = torch.ormqr(self.A, self.tau, b, transpose=True)[:self.A.shape[1]]
        self.W = torch.linalg.solve_triangular(self.A[:self.A.shape[1], :], y, upper=True)
        b_ = torch.ormqr(self.A, self.tau, torch.matmul(torch.triu(self.A), self.W), transpose=False)
        residual = torch.norm(b_ - b) / torch.norm(b)

        # w_set = []
        # b_ = b.clone()
        # for i in range(10):
        #     y = torch.ormqr(self.A, self.tau, b_, transpose=True)[:self.A.shape[1]]
        #     w = torch.linalg.solve_triangular(self.A[:self.A.shape[1], :], y, upper=True)
        #     w_set.append(w)
        #     b_ -= torch.ormqr(self.A, self.tau, torch.matmul(torch.triu(self.A), w), transpose=False)
        #     print(f"Relative residual: {torch.norm(b_) / torch.norm(b):.4e}")

        # # sum up the weights
        # self.W = torch.sum(torch.cat(w_set, dim=1), dim=1, keepdim=True)
        # residual = torch.norm(b_) / torch.norm(b)

        print(f"Relative residual: {residual:.4e}")

        if self.W.numel() % (self.submodels.numel() * self.n_hidden) == 0:
            n_out = int(self.W.numel() / (self.submodels.numel() * self.n_hidden))
            self.W = self.W.view(n_out, -1).T
        else:
            raise ValueError("The output weight mismatch.")

    def forward(self, x):
        """
        Forward pass of the model.

        :param x: Input tensor.
        :return: Output tensor after forward pass.
        """
        if self.W is None:
            raise ValueError("Weights have not been computed yet.")
        elif isinstance(self.W, Tensor):
            self.W = self.W.cat(dim=1)
        elif isinstance(self.W, List) and isinstance(self.W[0], torch.Tensor):
            self.W = torch.cat(self.W, dim=1)

        return torch.matmul(self.features(x).cat(dim=1), self.W)

    def dForward(self, x, order: Union[torch.Tensor, List]):
        """
        Compute the derivative of the forward pass.

        :param x: Input tensor.
        :param order: Order of the derivative.
        :return: Derivative tensor.
        """
        order = torch.tensor(order, dtype=self.dtype, device=self.device).view(1, -1)
        if order.shape[1] != self.dim:
            raise ValueError("Order dimension mismatch.")
        if order.sum() == 0:
            return self.forward(x)
        elif order.sum() == 1:
            for d in range(self.dim):
                if order[0, d] == 1:
                    return torch.matmul(self.features_derivative(x, d).cat(dim=1), self.W.view(-1, 1))
        elif order.sum() == 2:
            for d1 in range(self.dim):
                for d2 in range(self.dim):
                    if order[0, d1] == 1 and order[0, d2] == 1:
                        return torch.matmul(self.features_second_derivative(x, d1, d2).cat(dim=1), self.W.view(-1, 1))
        else:
            pass

    def features(self, x: torch.Tensor, use_sparse: bool = False) -> Tensor:
        """
        Compute the features for the given input.

        :param x: Input tensor.
        :param use_sparse: Whether to use sparse tensors.
        :return: Features Tensor.
        """
        features = []
        pou_coefficients = self.pou_coefficients(x)
        for (submodel, pou_coefficient) in zip(self.submodels.flat_data, pou_coefficients.flat_data):
            if not use_sparse:
                features.append(submodel(x) * pou_coefficient)
            else:
                features.append((submodel(x) * pou_coefficient).to_sparse())
        return Tensor(features, shape=self.submodels.shape)

    def features_derivative(self, x: torch.Tensor, axis: int, use_sparse: bool = False) -> Tensor:
        """
        Compute the feature derivative for the given input along the specified axis.

        :param x: Input tensor.
        :param axis: Axis along which to compute the derivative.
        :param use_sparse: Whether to use sparse tensors.
        :return: Feature derivative Tensor.
        """
        features_derivative = []
        pou_coefficients = self.pou_coefficients(x)
        pou_derivative = self.pou_derivative(x, axis)
        for (submodel, pou_coefficient, pou_derivative) in zip(self.submodels.flat_data,
                                                               pou_coefficients.flat_data,
                                                               pou_derivative.flat_data):
            if not use_sparse:
                features_derivative.append(submodel.first_derivative(x, axis) * pou_coefficient +
                                           submodel(x) * pou_derivative)
            else:
                features_derivative.append((submodel.first_derivative(x, axis) * pou_coefficient).to_sparse() +
                                           (submodel(x) * pou_derivative).to_sparse())
        return Tensor(features_derivative, shape=self.submodels.shape)

    def features_second_derivative(self, x: torch.Tensor, axis1: int, axis2: int, use_sparse: bool = False) -> Tensor:
        """
        Compute the feature second derivative for the given input along the specified axes.

        :param x: Input tensor.
        :param axis1: First axis along which to compute the derivative.
        :param axis2: Second axis along which to compute the derivative.
        :param use_sparse: Whether to use sparse tensors.
        :return: Feature second derivative Tensor.
        """
        features_second_derivative = []
        pou_coefficients = self.pou_coefficients(x)
        pou_first_derivative_axis1 = self.pou_derivative(x, axis1)
        pou_first_derivative_axis2 = self.pou_derivative(x, axis2)
        pou_second_derivative = self.pou_second_derivative(x, axis1, axis2)
        for (submodel, pou_coefficient, pou_first_axis1, pou_first_axis2, pou_second) in zip(
                self.submodels.flat_data,
                pou_coefficients.flat_data,
                pou_first_derivative_axis1.flat_data,
                pou_first_derivative_axis2.flat_data,
                pou_second_derivative.flat_data,
        ):
            if not use_sparse:
                features_second_derivative.append(
                    submodel.second_derivative(x, axis1, axis2) * pou_coefficient +
                    submodel.first_derivative(x, axis1) * pou_first_axis2 +
                    submodel.first_derivative(x, axis2) * pou_first_axis1 +
                    submodel(x) * pou_second
                )
            else:
                features_second_derivative.append(
                    (submodel.second_derivative(x, axis1, axis2) * pou_coefficient).to_sparse() +
                    (submodel.first_derivative(x, axis1) * pou_first_axis2).to_sparse() +
                    (submodel.first_derivative(x, axis2) * pou_first_axis1).to_sparse() +
                    (submodel(x) * pou_second).to_sparse()
                )
        return Tensor(features_second_derivative, shape=self.submodels.shape)

    def _compute_centers_and_radii(self, n_subdomains: Union[int, Tuple, List]):
        """
        Compute the centers and radii for subdomains.

        :param n_subdomains: Either an integer (uniform subdivisions in all dimensions)
                             or a list/tuple specifying the subdivisions per dimension.
        :return: Tuple of centers and radii as tensors.
        """
        centers_list = []
        radii_list = []
        bounding_box = self.domain.get_bounding_box()

        for i in range(self.dim):
            sub_min, sub_max = (bounding_box[2 * i], bounding_box[2 * i + 1])
            n_divisions = n_subdomains[i]

            # Compute the subdomain size and the effective step size
            subdomain_size = (sub_max - sub_min) / n_divisions
            effective_step = subdomain_size * (1 - self.overlap)
            radius_dim = torch.full((n_divisions,), subdomain_size / 2 * (1 + self.overlap), dtype=self.dtype,
                                    device=self.device)
            radii_list.append(radius_dim)

            # Generate the centers along this dimension
            centers_dim = torch.linspace(
                sub_min + effective_step / 2, sub_max - effective_step / 2, steps=n_divisions,
                dtype=self.dtype,
                device=self.device
            )
            centers_list.append(centers_dim)

        # Create a grid of centers for all dimensions as a multi-dimensional tensor
        centers = torch.stack(torch.meshgrid(*centers_list, indexing="ij"), dim=-1)  # Shape: (*n_subdomains, dim)
        radii = torch.stack(torch.meshgrid(*radii_list, indexing="ij"), dim=-1)  # Shape: (*n_subdomains, dim)

        return centers.to(dtype=self.dtype, device=self.device), radii.to(dtype=self.dtype, device=self.device)

    def pou_coefficients(self, x: torch.Tensor) -> Tensor[torch.Tensor]:
        """
        Compute the POU coefficients for the given input.

        :param x: Input tensor.
        :return: POU coefficients tensor.
        """
        if x.shape[1] != self.dim:
            raise ValueError("Input dimension mismatch.")

        # if isinstance(self.pou_functions[0], PsiA):
        #     c = []
        #     for (i, pou_function) in enumerate(self.pou_functions.flat_data):
        #         c_i = pou_function(x)
        #         c.append(c_i)
        #     return Tensor(c, shape=self.submodels.shape)

        c = []
        c_sum = torch.zeros(x.shape[0], 1, dtype=self.dtype, device=self.device)
        for (i, pou_function) in enumerate(self.pou_functions.flat_data):
            c_i = pou_function(x)
            c.append(c_i)
            c_sum += c_i
        c = [c_i / c_sum for c_i in c]

        return Tensor(c, shape=self.submodels.shape)

    def pou_derivative(self, x: torch.Tensor, axis: int) -> Tensor[torch.Tensor]:
        """
        Compute the POU derivative for the given input along the specified axis.

        :param x: Input tensor.
        :param axis: Axis along which to compute the derivative.
        :return: POU derivative Tensor.
        """
        if x.shape[1] != self.dim:
            raise ValueError("Input dimension mismatch.")

        # if isinstance(self.pou_functions[0], PsiA):
        #     c = []
        #     for (i, pou_function) in enumerate(self.pou_functions.flat_data):
        #         c_i = pou_function.first_derivative(x, axis)
        #         c.append(c_i)
        #     return Tensor(c, shape=self.submodels.shape)

        c = []
        c_sum = torch.zeros(x.shape[0], 1, dtype=self.dtype, device=self.device)
        dc_sum = torch.zeros(x.shape[0], 1, dtype=self.dtype, device=self.device)

        for (i, pou_function) in enumerate(self.pou_functions.flat_data):
            c_i = pou_function(x)
            dc_i = pou_function.first_derivative(x, axis)
            c.append((c_i, dc_i))
            c_sum += c_i
            dc_sum += dc_i
        c = [(dc_i - c_i * dc_sum / c_sum) / c_sum for c_i, dc_i in c]
        return Tensor(c, shape=self.submodels.shape)

    def pou_second_derivative(self, x: torch.Tensor, axis1: int, axis2: int) -> Tensor[torch.Tensor]:
        """
        Compute the POU second derivative for the given input along the specified axes.

        :param x: Input tensor.
        :param axis1: First axis along which to compute the derivative.
        :param axis2: Second axis along which to compute the derivative.
        :return: POU second derivative Tensor.
        """
        if x.shape[1] != self.dim:
            raise ValueError("Input dimension mismatch.")

        # if isinstance(self.pou_functions[0], PsiA):
        #     c = []
        #     for (i, pou_function) in enumerate(self.pou_functions.flat_data):
        #         c_i = pou_function.second_derivative(x, axis1, axis2)
        #         c.append(c_i)
        #     return Tensor(c, shape=self.submodels.shape)

        c = []
        c_sum = torch.zeros(x.shape[0], 1, dtype=self.dtype, device=self.device)
        dc_sum_axis1 = torch.zeros(x.shape[0], 1, dtype=self.dtype, device=self.device)
        dc_sum_axis2 = torch.zeros(x.shape[0], 1, dtype=self.dtype, device=self.device)
        d2c_sum = torch.zeros(x.shape[0], 1, dtype=self.dtype, device=self.device)

        # Compute raw values, first derivatives, and second derivatives
        for pou_function in self.pou_functions.flat_data:
            c_i = pou_function(x)
            dc_i_axis1 = pou_function.first_derivative(x, axis1)
            dc_i_axis2 = pou_function.first_derivative(x, axis2)
            d2c_i = pou_function.second_derivative(x, axis1, axis2)

            c.append((c_i, dc_i_axis1, dc_i_axis2, d2c_i))
            c_sum += c_i
            dc_sum_axis1 += dc_i_axis1
            dc_sum_axis2 += dc_i_axis2
            d2c_sum += d2c_i

        # Compute the second derivative with normalization
        d2 = [
            (
                    d2c_i / c_sum
                    - 2 * (dc_i_axis1 * dc_sum_axis2) / (c_sum ** 2)
                    - c_i * d2c_sum / (c_sum ** 2)
                    + 2 * c_i * dc_sum_axis1 * dc_sum_axis2 / (c_sum ** 3)
            )
            for c_i, dc_i_axis1, dc_i_axis2, d2c_i in c
        ]

        return Tensor(d2, shape=self.submodels.shape)

    def pou_higher_order_derivative(self, x: torch.Tensor, order: Union[torch.Tensor, List]) -> Tensor:
        """
        Compute the POU higher-order derivative for the given input.

        :param x: Input tensor.
        :param order: Order of the derivative as a tensor or list.
        :return: POU higher-order derivative Tensor.
        """
        if x.shape[1] != self.dim:
            raise ValueError("Input dimension mismatch.")
        pass

        return Tensor()
