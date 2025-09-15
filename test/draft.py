# -*- coding: utf-8 -*-
"""
Created on 6/17/25

@author: Yifei Sun
"""
from sympy import transpose

import pyrfm

import torch
import numpy as np
import time

import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans2 as Kmeans

from pyrfm.core import *


class RRFBase(RFBase):
    pass


class RRFTanH(RRFBase):
    def __init__(self, dim: int, center: torch.Tensor, radius: torch.Tensor,
                 n_hidden: int,
                 gen: torch.Generator = None,
                 dtype: torch.dtype = None,
                 device: torch.device = None):

        super().__init__(dim, center, radius, nn.Tanh(), n_hidden, gen, dtype, device)
        self.radius = torch.linalg.norm(radius, dim=-1, keepdim=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.dim:
            raise ValueError('Input dimension mismatch')
        with torch.no_grad():
            # Be careful when x in a slice
            if (self.x_buff_ is not None) and (self.x_buff_ is x or torch.equal(self.x_buff_, x)):
                return self.features_buff_
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
            # Be careful when x in a slice
            if (self.x_buff_ is not None) and (self.x_buff_ is x or torch.equal(self.x_buff_, x)):
                pass
            else:
                self.forward(x)

            return (1 - torch.pow(self.features_buff_, 2)) * (self.weights[[axis], :] / self.radius)

    def second_derivative(self, x: torch.Tensor, axis1: int, axis2: int) -> torch.Tensor:
        if x.shape[1] != self.dim:
            raise ValueError('Input dimension mismatch')

        if axis1 >= self.dim:
            raise ValueError('Axis1 out of range')

        if axis2 >= self.dim:
            raise ValueError('Axis2 out of range')

        with torch.no_grad():
            # Be careful when x in a slice
            if (self.x_buff_ is not None) and (self.x_buff_ is x or torch.equal(self.x_buff_, x)):
                pass
            else:
                self.forward(x)

            return -2 * self.features_buff_ * (1 - torch.pow(self.features_buff_, 2)) * \
                (self.weights[[axis1], :] / self.radius) * (
                        self.weights[[axis2], :] / self.radius)

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
                p_n *= self.weights[[i], :]

        return p_n


class RPOUBase(POUBase):
    pass


class RPsiB(RPOUBase):
    def __init__(self, center: torch.Tensor, radius: Union[torch.Tensor, float],
                 dtype: torch.dtype = None,
                 device: torch.device = None
                 ):
        if not isinstance(radius, torch.Tensor):
            radius = torch.tensor(radius, dtype=torch.float64)
        super().__init__(center, radius, dtype, device)
        # self.dtype = dtype if dtype is not None else torch.tensor(0.).dtype
        # self.device = device if device is not None else torch.tensor(0.).device
        # self.center = center.to(dtype=self.dtype, device=self.device).view(1, -1)
        # self.radius = radius.to(dtype=self.dtype, device=self.device).view(1, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.center.shape[1]:
            raise ValueError('Input dimension mismatch')
        if self.radius.shape[1] != 1:
            radius = torch.linalg.norm(self.radius, dim=1, keepdim=True)
        else:
            radius = self.radius

        x_ = torch.linalg.norm((x - self.center), dim=1, keepdim=True) / radius
        return self.func(x_)

    def first_derivative(self, x: torch.Tensor, axis: int) -> torch.Tensor:
        if x.shape[1] != self.center.shape[1]:
            raise ValueError('Input dimension mismatch')
        if self.radius.shape[1] != 1:
            radius = torch.linalg.norm(self.radius, dim=1, keepdim=True)
        else:
            radius = self.radius

        x_ = torch.linalg.norm((x - self.center), dim=1, keepdim=True) / radius
        d_x_ = (x - self.center)[:, [axis]] / torch.linalg.norm((x - self.center), dim=1, keepdim=True) / radius

        return self.d_func(x_) * d_x_

    def second_derivative(self, x: torch.Tensor, axis1: int, axis2: int) -> torch.Tensor:
        if x.shape[1] != self.center.shape[1]:
            raise ValueError('Input dimension mismatch')
        if self.radius.shape[1] != 1:
            radius = torch.linalg.norm(self.radius, dim=1, keepdim=True)
        else:
            radius = self.radius

        r = x - self.center  # shape: [N, D]
        norm_r = torch.linalg.norm(r, dim=1, keepdim=True)  # shape: [N, 1]
        x_ = norm_r / radius  # scalar input to func

        # First derivative: ∂x_/∂x_i
        dx_1 = r[:, [axis1]] / norm_r / radius
        dx_2 = r[:, [axis2]] / norm_r / radius

        # Second derivative of x_: ∂²x_/∂x_i∂x_j
        delta = (axis1 == axis2)
        d2x_ = (-r[:, [axis1]] * r[:, [axis2]] / norm_r ** 3 if not delta else
                (1.0 / norm_r - r[:, [axis1]] ** 2 / norm_r ** 3)) / radius

        term1 = self.d2_func(x_) * dx_1 * dx_2
        term2 = self.d_func(x_) * d2x_

        return term1 + term2

    def higher_order_derivative(self, x: torch.Tensor, order: Union[torch.Tensor, List]) -> torch.Tensor:
        raise NotImplementedError("Higher order derivatives are not implemented for PsiBR.")

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


class SubdomainClusterer:
    def __init__(self, n_subdomains=4, seed=100):
        self.n_subdomains = n_subdomains
        self.seed = seed
        self.centers = None
        self.radii = None
        self.labels = None
        self.x_all = None  # 保存以便绘图

    def fit(self, x_all: torch.Tensor):
        """
        从所有点中聚类并计算每个子域的中心和半径。
        """
        self.x_all = x_all
        rng = np.random.default_rng(seed=self.seed)
        centers, labels = Kmeans(x_all.cpu().numpy(), self.n_subdomains, minit='++', rng=rng)
        centers = torch.tensor(centers, dtype=torch.float64, device=x_all.device)
        labels = torch.tensor(labels, dtype=torch.int64, device=x_all.device)

        distances = torch.cdist(x_all, centers)

        mask = labels.unsqueeze(1) != torch.arange(self.n_subdomains, device=x_all.device).unsqueeze(0)
        distances_by_subdomain = distances.clone()
        distances_by_subdomain[mask] = -1

        max_distances, max_indices = torch.max(distances_by_subdomain, dim=0)
        farthest_points = x_all[max_indices]

        radii = farthest_points - centers

        self.centers = centers
        self.radii = radii
        self.labels = labels

        return centers, radii, labels

    def plot(self, figsize=(8, 8)):
        """
        可视化子域划分、中心和半径向量。
        """
        if self.centers is None or self.radii is None or self.labels is None or self.x_all is None:
            raise ValueError("Must call fit() before plot().")

        x_np = self.x_all.cpu().numpy()
        labels_np = self.labels.cpu().numpy()
        centers_np = self.centers.cpu().numpy()
        radii_np = self.radii.cpu().numpy()

        plt.figure(figsize=figsize)
        plt.scatter(x_np[:, 0], x_np[:, 1], c=labels_np, cmap='viridis', s=10, label='Points')
        plt.scatter(centers_np[:, 0], centers_np[:, 1], c='red', s=100, marker='x', label='Centers')

        # # 绘制 radius 向量为箭头
        # for i in range(self.n_subdomains):
        #     c = centers_np[i]
        #     r = radii_np[i]
        #     plt.arrow(c[0], c[1], r[0], r[1],
        #               head_width=0.02, head_length=0.04,
        #               fc='red', ec='red', linewidth=1.5)

        plt.title('Subdomain Partitioning with Centers and Radii')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.axis('equal')
        plt.colorbar(label='Subdomain Label')
        plt.legend()
        plt.show()


class RRFM(RFMBase):
    def __init__(self, dim: int,
                 n_hidden: int,
                 domain: Union[Tuple, List, GeometryBase],
                 n_subdomains: int,
                 centers: Optional[torch.Tensor] = None,
                 radii: Optional[torch.Tensor] = None,
                 rf=RRFTanH,
                 pou=RPsiB,
                 seed: int = 100,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        if not isinstance(n_subdomains, int):
            raise TypeError("n_subdomains must be an integer.")
        if centers is None or radii is None:
            clusterer = SubdomainClusterer(n_subdomains=n_subdomains, seed=seed)
            x_all = domain.in_sample(10000, with_boundary=True)
            centers, radii, _ = clusterer.fit(x_all)
        else:
            if centers.shape[0] != n_subdomains:
                raise ValueError(f"centers must have {n_subdomains} rows, but got {centers.shape[0]}.")
            if radii.shape[0] != n_subdomains:
                raise ValueError(f"radii must have {n_subdomains} rows, but got {radii.shape[0]}.")

        if not issubclass(rf, RRFBase):
            raise ValueError("Random Feature must be a subclass of RRFBase.")
        if not issubclass(pou, RPOUBase):
            raise ValueError("Point of Unit must be a subclass of RPOUBase.")
        super().__init__(dim=dim, n_hidden=n_hidden, domain=domain,
                         centers=centers, radii=radii,
                         n_subdomains=[n_subdomains],
                         rf=rf, pou=pou, seed=seed,
                         dtype=dtype, device=device)

    def _compute_centers_and_radii(self, n_subdomains: Union[int, Tuple, List]):
        raise NotImplementedError("This method is disabled.")

    def add_c_condition(self, num_samples: int, order: int = 1, with_pts=False):
        raise NotImplementedError("This method is disabled.")


# 简单的解析解 u(x, y)
def u_func(x):
    return torch.sin(math.pi * x[:, [0]]) * torch.sin(math.pi * x[:, [1]])


# 对应的 f(x, y) = -Δu
def f_func(x):
    return 2 * math.pi ** 2 * torch.sin(math.pi * x[:, [0]]) * torch.sin(math.pi * x[:, [1]])


# 边界值函数
def g_func(x):
    return u_func(x)


# def u_func(x):
#     return -0.5 * (1.5 * torch.cos(torch.pi * x[:, [0]] + 2 * torch.pi / 5) +
#                    2 * torch.cos(2 * torch.pi * x[:, [0]] - torch.pi / 5)) * \
#         (1.5 * torch.cos(torch.pi * x[:, [1]] + 2 * torch.pi / 5) +
#          2 * torch.cos(2 * torch.pi * x[:, [1]] - torch.pi / 5)) - \
#         0.5 * (1.5 * torch.cos(2 * torch.pi * x[:, [0]] + 2 * torch.pi / 5) +
#                2 * torch.cos(4 * torch.pi * x[:, [0]] - torch.pi / 5)) * \
#         (1.5 * torch.cos(2 * torch.pi * x[:, [1]] + 2 * torch.pi / 5) +
#          2 * torch.cos(4 * torch.pi * x[:, [1]] - torch.pi / 5))
#
#
# # -(uxx + uyy) = f
# def f_func(x):
#     return -(-0.5 * (-1.5 * torch.pi ** 2 * torch.cos(torch.pi * x[:, [0]] + 2 * torch.pi / 5) -
#                      2 * (2 * torch.pi) ** 2 * torch.cos(2 * torch.pi * x[:, [0]] - torch.pi / 5)) * \
#              (1.5 * torch.cos(torch.pi * x[:, [1]] + 2 * torch.pi / 5) +
#               2 * torch.cos(2 * torch.pi * x[:, [1]] - torch.pi / 5)) - \
#              0.5 * (1.5 * torch.cos(torch.pi * x[:, [0]] + 2 * torch.pi / 5) +
#                     2 * torch.cos(2 * torch.pi * x[:, [0]] - torch.pi / 5)) * \
#              (-1.5 * torch.pi ** 2 * torch.cos(torch.pi * x[:, [1]] + 2 * torch.pi / 5) -
#               2 * (2 * torch.pi) ** 2 * torch.cos(2 * torch.pi * x[:, [1]] - torch.pi / 5)) - \
#              0.5 * (-1.5 * (2 * torch.pi) ** 2 * torch.cos(2 * torch.pi * x[:, [0]] + 2 * torch.pi / 5) -
#                     2 * (4 * torch.pi) ** 2 * torch.cos(4 * torch.pi * x[:, [0]] - torch.pi / 5)) * \
#              (1.5 * torch.cos(2 * torch.pi * x[:, [1]] + 2 * torch.pi / 5) +
#               2 * torch.cos(4 * torch.pi * x[:, [1]] - torch.pi / 5)) - \
#              0.5 * (1.5 * torch.cos(2 * torch.pi * x[:, [0]] + 2 * torch.pi / 5) +
#                     2 * torch.cos(4 * torch.pi * x[:, [0]] - torch.pi / 5)) * \
#              (-1.5 * (2 * torch.pi) ** 2 * torch.cos(2 * torch.pi * x[:, [1]] + 2 * torch.pi / 5) -
#               2 * (4 * torch.pi) ** 2 * torch.cos(4 * torch.pi * x[:, [1]] - torch.pi / 5)))
#
#
# def g_func(x):
#     return u_func(x)


class IterativeSolverBase:
    def __init__(self, max_iter: int = 1000, tol: float = 1e-8, preconditioner=None, dtype: torch.dtype = None,
                 device: torch.device = None):
        self.max_iter = max_iter
        self.tol = tol
        self.preconditioner = preconditioner
        self.dtype = dtype if dtype is not None else torch.tensor(0.).dtype
        self.device = device if device is not None else torch.tensor(0.).device

    def compute(self, A):
        pass

    def solve(self, b):
        pass


class PreconditionerBase:
    def __init__(self, dtype: torch.dtype = None, device: torch.device = None):
        self.dtype = dtype if dtype is not None else torch.tensor(0.).dtype
        self.device = device if device is not None else torch.tensor(0.).device

    def compute(self, A):
        pass

    def solve(self, b):
        pass


class TriangularPreconditioner(PreconditionerBase):
    def __init__(self, dtype: torch.dtype = None, device: torch.device = None):
        super().__init__(dtype=dtype, device=device)
        self.R = None

    def compute(self, A: torch.Tensor):
        """
        Compute the QR decomposition of matrix A.

        :param A: Input matrix.
        :return: Self.
        """
        A = A.to(dtype=self.dtype, device=self.device)
        try:
            R, _ = torch.geqrf(A)
        except RuntimeError as e:
            if 'cusolver error' in str(e):
                raise RuntimeError("Out Of Memory Error")
            else:
                raise e
        self.R = R[:R.shape[1], :]
        self.R.diagonal().add_((self.R.diagonal() >= 0).float() * torch.finfo(self.dtype).eps)

        return self

    def solve(self, b: torch.Tensor, transpose: bool = False):
        """
        Solve the linear system Ax = b using the QR decomposition.

        :param b: Right-hand side tensor.
        :param check_condition: Whether to check the condition number of A, and switch to SVD if necessary.
        :param complex: Whether to use complex numbers.
        """
        b = b.view(-1, 1).to(dtype=self.dtype, device=self.device)
        if self.R.shape[0] != b.shape[0]:
            raise ValueError("Input dimension mismatch.")
        if not transpose:
            w = torch.linalg.solve_triangular(self.R, b, upper=True)
        else:
            w = torch.linalg.solve_triangular(self.R.T, b, upper=False)

        return w


class LinearOperatorBase:
    @abstractmethod
    def matvec(self, x):
        raise NotImplementedError("matvec method must be implemented.")

    @abstractmethod
    def rmatvec(self, x):
        raise NotImplementedError("rmatvec method must be implemented.")

    @abstractmethod
    def matmat(self, X):
        raise NotImplementedError("matmat method must be implemented.")

    @abstractmethod
    def rmatmat(self, X):
        raise NotImplementedError("rmatmat method must be implemented.")


class TorchMatrix(LinearOperatorBase):
    def __init__(self, A: torch.Tensor):
        self.A = A

    def matvec(self, x: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor.")
        if self.A.shape[1] != x.shape[0]:
            raise ValueError("Input dimension mismatch.")

        return torch.matmul(self.A, x)

    def rmatvec(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor.")
        if self.A.shape[0] != x.shape[0]:
            raise ValueError("Input dimension mismatch.")

        return torch.matmul(self.A.T, x)

    def matmat(self, X: torch.Tensor):
        if not isinstance(X, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor.")
        if self.A.shape[1] != X.shape[0]:
            raise ValueError("Input dimension mismatch.")

        return torch.matmul(self.A, X)

    def rmatmat(self, X: torch.Tensor):
        if not isinstance(X, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor.")
        if self.A.shape[0] != X.shape[0]:
            raise ValueError("Input dimension mismatch.")

        return torch.matmul(self.A.T, X)


class RowCompressedMatrix(LinearOperatorBase):
    def __init__(self, A_sparse: torch.Tensor, mask: torch.Tensor, dtype: torch.dtype = None,
                 device: torch.device = None):
        """
        Initialize a row-compressed matrix.
        Args:
            A_sparse:
            mask: A.any(dim=1, keepdim=True) is used to create the mask.
            dtype:
            device:
        """
        self.dtype = dtype if dtype is not None else A_sparse.dtype
        self.device = device if device is not None else A_sparse.device
        self.data = A_sparse
        self.mask = mask
        self.shape = (self.mask.shape[0], self.data.shape[1])

    def todense(self):
        A_dense = torch.zeros((self.mask.shape[0], self.data.shape[1]), dtype=self.dtype, device=self.device)
        A_dense[self.mask] = self.data
        return A_dense

    def matvec(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor.")
        if self.data.shape[1] != x.shape[0]:
            raise ValueError("Input dimension mismatch.")
        x = x.to(dtype=self.dtype, device=self.device)
        Ax = torch.zeros((self.mask.shape[0], 1), dtype=self.dtype, device=self.device)
        Ax[self.mask] = torch.matmul(self.data, x)

        return Ax

    def rmatvec(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor.")
        if self.mask.shape[0] != x.shape[0]:
            raise ValueError("Input dimension mismatch.")
        x = x.to(dtype=self.dtype, device=self.device)
        rAx = torch.matmul(self.data.T, x[self.mask])

        return rAx

    def matmat(self, X):
        if not isinstance(X, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor.")
        if self.data.shape[1] != X.shape[0]:
            raise ValueError("Input dimension mismatch.")
        X = X.to(dtype=self.dtype, device=self.device)
        AX = torch.zeros((self.mask.shape[0], X.shape[1]), dtype=self.dtype, device=self.device)
        AX[self.mask] = torch.matmul(self.data, X)

        return AX

    def rmatmat(self, X):
        if not isinstance(X, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor.")
        if self.mask.shape[0] != X.shape[0]:
            raise ValueError("Input dimension mismatch.")
        X = X.to(dtype=self.dtype, device=self.device)
        rAX = torch.matmul(self.data.T, X[self.mask])

        return rAX


class ColBlockMatrix(LinearOperatorBase):
    def __init__(self, A: Tensor[LinearOperatorBase]):
        self.A = A
        self.shape = (self.A[0].shape[0], sum(A[i].shape[1] for i in range(A.numel())))
        self.dtype = A[0].dtype if A.numel() > 0 else torch.tensor(0.).dtype
        self.device = A[0].device if A.numel() > 0 else torch.tensor(0.).device

    def compute(self):
        for i in range(self.A.numel()):
            if not isinstance(self.A[i], LinearOperatorBase):
                raise TypeError(f"Element {i} of A must be a LinearOperatorBase.")
            self.A[i].compute()

    def matvec(self, x: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor.")
        if x.ndim == 1:
            x = x.unsqueeze(-1)
        if x.shape[0] != self.shape[1]:
            raise ValueError(f"Input dimension mismatch: expected {self.shape[1]}, got {x.shape[0]}")

        Ax = torch.zeros((self.shape[0], x.shape[1]), dtype=self.dtype, device=self.device)

        col_offset = 0
        for i in range(self.A.numel()):
            block_cols = self.A[i].shape[1]
            x_block = x[col_offset:col_offset + block_cols]

            Ax += self.A[i].matvec(x_block)
            col_offset += block_cols

        return Ax

    def rmatvec(self, x: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor.")
        if self.shape[0] != x.shape[0]:
            raise ValueError("Input dimension mismatch.")

        rAx = torch.cat([self.A[i].rmatvec(x) for i in range(self.A.numel())], dim=0)

        return rAx

    def matmat(self, X: torch.Tensor):
        if not isinstance(X, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor.")
        if X.ndim != 2:
            raise ValueError("Input must be a 2D tensor.")
        if X.shape[0] != self.shape[1]:
            raise ValueError(f"Input dimension mismatch: expected {self.shape[1]}, got {X.shape[0]}")

        AX = torch.zeros((self.shape[0], X.shape[1]), dtype=self.dtype, device=self.device)

        col_offset = 0
        for i in range(self.A.numel()):
            block_cols = self.A[i].shape[1]
            X_block = X[col_offset:col_offset + block_cols, :]

            AX += self.A[i].matmat(X_block)
            col_offset += block_cols

        return AX

    def rmatmat(self, X: torch.Tensor):
        if not isinstance(X, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor.")
        if self.shape[0] != X.shape[0]:
            raise ValueError("Input dimension mismatch.")

        rAX = torch.cat([self.A[i].rmatmat(X) for i in range(self.A.numel())], dim=0)

        return rAX


class RowCompressedQRMatrix(LinearOperatorBase):
    def __init__(self, A_sparse: torch.Tensor,
                 mask: torch.Tensor,
                 dtype: torch.dtype = None, device: torch.device = None):
        self.dtype = dtype if dtype is not None else A_sparse.dtype
        self.device = device if device is not None else A_sparse.device
        self.data = A_sparse
        self.tau = None
        self.mask = mask
        self.shape = (self.mask.shape[0], self.data.shape[1])

    def todense(self):
        A_dense = torch.zeros((self.mask.shape[0], self.data.shape[1]), dtype=self.dtype, device=self.device)
        A_dense[self.mask] = self.data
        return A_dense

    def compute(self):
        self.data, self.tau = torch.geqrf(self.data)
        return self

    def solve(self, x):
        return torch.linalg.solve_triangular(self.data[:self.data.shape[1], :], x, upper=True)

    def matvec(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor.")
        if self.data.shape[1] != x.shape[0]:
            raise ValueError("Input dimension mismatch.")
        x = x.to(dtype=self.dtype, device=self.device)
        x = torch.cat([x, torch.zeros(self.data.shape[0] - self.data.shape[1], 1)], dim=0)
        Ax = torch.zeros((self.mask.shape[0], 1), dtype=self.dtype, device=self.device)
        Ax[self.mask] = torch.ormqr(self.data, self.tau,
                                    x,
                                    transpose=False)

        return Ax

    def rmatvec(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor.")
        if self.mask.shape[0] != x.shape[0]:
            raise ValueError("Input dimension mismatch.")
        x = x.to(dtype=self.dtype, device=self.device)
        rAx = torch.ormqr(self.data, self.tau, x[self.mask], transpose=True)[:self.data.shape[1], :]

        return rAx

    def matmat(self, X: torch.Tensor):
        if not isinstance(X, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor.")
        if self.data.shape[1] != X.shape[0]:
            raise ValueError("Input dimension mismatch.")
        X = X.to(dtype=self.dtype, device=self.device)
        AX = torch.zeros((self.mask.shape[0], X.shape[1]), dtype=self.dtype, device=self.device)
        AX[self.mask] = torch.ormqr(self.data, self.tau,
                                    torch.cat([X, torch.zeros((self.data.shape[0] - self.data.shape[1], X.shape[1]))],
                                              dim=0), transpose=False)

        return AX

    def rmatmat(self, X: torch.Tensor):
        if not isinstance(X, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor.")
        if self.mask.shape[0] != X.shape[0]:
            raise ValueError("Input dimension mismatch.")
        X = X.to(dtype=self.dtype, device=self.device)
        rAX = torch.ormqr(self.data, self.tau, X[self.mask], transpose=True)[:self.data.shape[1], :]

        return rAX


class PLSQR(IterativeSolverBase):
    def __init__(self, max_iter: int = 1000, tol: float = 1e-8):
        super().__init__(max_iter, tol)
        self.A = None

    def compute(self, A: Tensor):
        if isinstance(A[0], torch.Tensor):
            masks = [A[i].any(dim=1) for i in range(A.numel())]
            A = Tensor([
                RowCompressedQRMatrix(A[i][masks[i]],
                                      masks[i],
                                      dtype=self.dtype,
                                      device=self.device)
                for i in range(A.numel())], A.shape)

        self.A = ColBlockMatrix(A)
        self.A.compute()

        return self

    def solve(self, b: torch.Tensor):

        Ap = self.A
        m, n = Ap.shape

        x = torch.zeros(n, 1, dtype=self.dtype, device=self.device)
        u = b.clone()
        beta = torch.norm(u)
        u /= beta

        v = Ap.rmatvec(u)
        alpha = torch.norm(v)
        v /= alpha

        w = v.clone()
        phi_bar = beta
        rho_bar = alpha

        for it in range(self.max_iter):
            u = Ap.matvec(v) - alpha * u
            beta = torch.norm(u)
            if beta != 0:
                u /= beta

            v = Ap.rmatvec(u) - beta * v
            alpha = torch.norm(v)
            if alpha != 0:
                v /= alpha

            rho = torch.sqrt(rho_bar ** 2 + beta ** 2)
            c = rho_bar / rho
            s = beta / rho
            theta = s * alpha
            rho_bar = -c * alpha
            phi = c * phi_bar
            phi_bar = s * phi_bar

            x = x + (phi / rho) * w
            w = v - (theta / rho) * w

            # 输出迭代信息
            print(
                f"Iter {it + 1:4d}: phi_bar = {phi_bar.item():.2e}, alpha = {alpha.item():.2e}, beta = {beta.item():.2e}, residual ≈ {phi_bar.abs().item():.2e}")

            if phi_bar.abs() < self.tol:
                print("Converged.")
                break

        x_true = []
        col_offset = 0
        for i in range(Ap.A.numel()):
            block_size = Ap.A[i].shape[1]
            x_block = x[col_offset:col_offset + block_size]
            x_true.append(Ap.A[i].solve(x_block))
            col_offset += block_size
        x = torch.cat(x_true, dim=0)

        return x.squeeze(1) if b.ndim == 1 else x


def lsqr(A, b, max_iter=100, atol=1e-6, btol=1e-6, x0=None):
    m, n = A.shape
    if x0 is None:
        x = torch.zeros((n, 1), dtype=A.dtype, device=A.device)
    else:
        x = x0.clone().view(n, 1).to(dtype=A.dtype, device=A.device)

    u = b.clone() - A @ x
    beta = torch.norm(u)
    if beta != 0:
        u = u / beta

    v = A.T @ u
    alpha = torch.norm(v)
    if alpha != 0:
        v = v / alpha

    w = v.clone()
    phi_bar = beta
    rho_bar = alpha
    prev_phi_bar = phi_bar  # 初始化上一轮的 phi_bar

    for i in range(max_iter):
        u = A @ v - alpha * u
        beta = torch.norm(u)
        if beta != 0:
            u = u / beta

        v = A.T @ u - beta * v
        alpha = torch.norm(v)
        if alpha != 0:
            v = v / alpha

        rho = torch.sqrt(rho_bar ** 2 + beta ** 2)
        c = rho_bar / rho
        s = beta / rho
        theta = s * alpha
        rho_bar = -c * alpha
        phi = c * phi_bar
        prev_phi_bar_value = phi_bar.item()
        phi_bar = s * phi_bar

        x = x + (phi / rho) * w
        w = v - (theta / rho) * w

        print(f"Iter {i + 1:4d}: phi_bar = {phi_bar.item():.2e}, alpha = {alpha.item():.2e}, beta = {beta.item():.2e}")

        # 收敛判断：phi_bar 相对变化小于 1e-4
        if prev_phi_bar_value != 0:
            rel_change = abs((phi_bar.item() - prev_phi_bar_value) / prev_phi_bar_value)
            if rel_change < 1e-4:
                print(f"Converged at iteration {i + 1} (relative change in phi_bar < 1e-4)")
                break

    return x


def lscg(A, b, max_iter=100, atol=1e-6, btol=1e-6, x0=None):
    m, n = A.shape
    b = b.view(-1, 1).to(dtype=A.dtype, device=A.device)
    At = A.T

    if x0 is None:
        x = torch.zeros((n, 1), dtype=A.dtype, device=A.device)
    else:
        x = x0.clone().view(n, 1).to(dtype=A.dtype, device=A.device)

    r = At @ (b - A @ x)  # 初始残差 r0 = A^T(b - A x)
    p = r.clone()
    rs_old = torch.sum(r * r)

    residual = b - A @ x
    phi_bar = torch.norm(residual)
    prev_phi_bar_value = phi_bar.item()

    for i in range(max_iter):
        Ap = At @ (A @ p)
        alpha = rs_old / torch.sum(p * Ap)

        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = torch.sum(r * r)

        # 更新 phi_bar = ||b - Ax||，与 lsqr 一致
        residual = b - A @ x
        phi_bar = torch.norm(residual)
        rel_change = abs((phi_bar.item() - prev_phi_bar_value) / (prev_phi_bar_value + 1e-12))
        prev_phi_bar_value = phi_bar.item()

        print(f"Iter {i + 1:4d}: phi_bar = {phi_bar.item():.2e}, rel_change = {rel_change:.2e}")

        if rel_change < 1e-4:
            print(f"Converged at iteration {i + 1} (relative change in phi_bar < 1e-4)")
            break

        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new

    return x


def lsmr(A, b, max_iter=100, atol=1e-6, btol=1e-6, x0=None):
    m, n = A.shape
    if x0 is None:
        x = torch.zeros((n, 1), dtype=A.dtype, device=A.device)
    else:
        x = x0.clone().view(n, 1).to(dtype=A.dtype, device=A.device)

    u = b.clone() - A @ x
    beta = torch.norm(u)
    if beta != 0:
        u = u / beta

    v = A.T @ u
    alpha = torch.norm(v)
    if alpha != 0:
        v = v / alpha

    w = v.clone()
    z = torch.zeros_like(x)
    phi = beta
    rho = alpha
    prev_phi = phi.item()

    for i in range(max_iter):
        u = A @ v - alpha * u
        beta = torch.norm(u)
        if beta != 0:
            u = u / beta

        v = A.T @ u - beta * v
        alpha = torch.norm(v)
        if alpha != 0:
            v = v / alpha

        rho_hat = torch.sqrt(rho ** 2 + beta ** 2)
        c = rho / rho_hat
        s = beta / rho_hat
        theta = s * alpha
        rho = -c * alpha
        phi_hat = c * phi
        phi = s * phi

        z = z + (phi_hat / rho_hat) * w
        w = v - (theta / rho_hat) * w
        x = x + z
        z.zero_()

        print(f"Iter {i + 1:4d}: phi = {phi.item():.2e}, alpha = {alpha.item():.2e}, beta = {beta.item():.2e}")

        # 收敛判断：phi 相对变化小于阈值
        if prev_phi != 0:
            rel_change = abs((phi.item() - prev_phi) / prev_phi)
            if rel_change < 1e-4:
                print(f"Converged at iteration {i + 1} (relative change in phi < 1e-4)")
                break
        prev_phi = phi.item()

    return x


def gmres(A, b, x0=None, tol=1e-5, max_iter=100, restart=None, verbose=False):
    """
    GMRES solver for A x = b where A is (n x n) and b is (n x 1).

    Parameters:
    - A: torch.Tensor of shape (n, n)
    - b: torch.Tensor of shape (n, 1)
    - x0: initial guess, shape (n, 1)
    - tol: convergence tolerance
    - max_iter: max number of iterations
    - restart: subspace restart (optional)
    - verbose: print residuals

    Returns:
    - x: solution of shape (n, 1)
    - residuals: list of residual norms
    """
    n = b.shape[0]
    dtype = b.dtype
    device = b.device

    if x0 is None:
        x = torch.zeros((n, 1), dtype=dtype, device=device)
    else:
        x = x0.clone()

    r = b - A @ x
    beta = torch.norm(r)
    residuals = [beta.item()]
    if verbose:
        print(f"Iter 0: Residual = {beta.item():.3e}")
    if beta < tol:
        return x, residuals

    if restart is None:
        restart = min(n, 50)

    for outer in range(0, max_iter, restart):
        V = [r / beta]  # list of (n x 1) vectors
        H = torch.zeros((restart + 1, restart), dtype=dtype, device=device)
        g = torch.zeros((restart + 1, 1), dtype=dtype, device=device)
        g[0, 0] = beta

        for j in range(restart):
            w = A @ V[j]
            for i in range(j + 1):
                H[i, j] = (V[i].T @ w).item()
                w = w - H[i, j] * V[i]
            H[j + 1, j] = torch.norm(w)
            if H[j + 1, j] < 1e-14:
                break
            V.append(w / H[j + 1, j])

            # Solve least squares: min ||H y - g|| via QR
            H_sub = H[:j + 2, :j + 1]  # (j+2) x (j+1)
            g_sub = g[:j + 2]  # (j+2) x 1
            y = torch.linalg.lstsq(H_sub, g_sub).solution  # (j+1) x 1

            dx = torch.cat(V[:j + 1], dim=1) @ y  # (n x j+1) @ (j+1 x 1) => (n x 1)
            x_new = x + dx
            r_new = b - A @ x_new
            res_norm = torch.norm(r_new).item()
            residuals.append(res_norm)

            if verbose:
                print(f"Iter {outer + j + 1}: Residual = {res_norm:.3e}")

            if res_norm < tol:
                return x_new, residuals

        x = x_new
        r = r_new
        beta = torch.norm(r)

    return x, residuals


def cg(A, b, x0=None, tol=1e-5, max_iter=100, verbose=False):
    """
    Conjugate Gradient solver for symmetric positive-definite A x = b.

    Parameters:
    - A: torch.Tensor of shape (n, n), symmetric and positive-definite
    - b: torch.Tensor of shape (n, 1)
    - x0: initial guess, shape (n, 1)
    - tol: convergence tolerance
    - max_iter: max number of iterations
    - verbose: print residuals

    Returns:
    - x: solution of shape (n, 1)
    - residuals: list of residual norms
    """
    n = b.shape[0]
    dtype = b.dtype
    device = b.device

    if x0 is None:
        x = torch.zeros((n, 1), dtype=dtype, device=device)
    else:
        x = x0.clone()

    r = b - A @ x
    p = r.clone()
    rs_old = torch.sum(r * r)
    residuals = [torch.sqrt(rs_old).item()]

    if verbose:
        print(f"Iter 0: Residual = {residuals[-1]:.3e}")

    for i in range(1, max_iter + 1):
        Ap = A @ p
        alpha = rs_old / (torch.sum(p * Ap) + 1e-12)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = torch.sum(r * r)
        residual = torch.sqrt(rs_new).item()
        residuals.append(residual)

        if verbose:
            print(f"Iter {i}: Residual = {residual:.3e}")

        if residual < tol:
            break

        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new

    return x, residuals


if __name__ == '__main__':
    torch.set_default_device('cuda') if torch.cuda.is_available() else torch.set_default_device('cpu')
    start_time = time.time()

    # domain = pyrfm.Square2D(center=[0, 0], radius=[1, 1])
    domain = pyrfm.Circle2D((0, 0), 1) - pyrfm.Circle2D((0, 0), 0.5)
    # domain = (pyrfm.Square2D(center=[-0.75, 0], radius=[0.25, 1])
    #           + pyrfm.Square2D(center=[0.75, 0], radius=[0.25, 1])
    #           + pyrfm.Square2D(center=[0, 0.75], radius=[1.0, 0.25]))

    # domain = pyrfm.Square2D(center=[0, 0.5], radius=[1, 0.5]) + pyrfm.Square2D(center=[0.5, 0.0], radius=[0.5, 1.0])
    # domain = pyrfm.Square2D(center=[0, 0], radius=[1, 0.1])

    x_in = domain.in_sample(20000, with_boundary=False)

    x_on = domain.on_sample(2000)

    n_subdomains = 8

    x_all = [x_in, x_on]

    point_type_labels = []
    for label, x_ in enumerate(x_all):
        # print(f"Label {label} has {x_.shape[0]} points.")
        point_type_labels.append(torch.ones(x_.shape[0], 1, dtype=torch.int64) * label)

    x_all = torch.cat(x_all, dim=0)
    point_type_labels = torch.cat(point_type_labels, dim=0)

    clusterer = SubdomainClusterer(n_subdomains=n_subdomains, seed=100)
    centers, radii, labels = clusterer.fit(x_all)
    # clusterer.plot()

    model = RRFM(dim=2, n_hidden=200, domain=domain, n_subdomains=n_subdomains, centers=centers, radii=radii)

    # A_in_xx = model.features_second_derivative(x_in, axis1=0, axis2=0).cat(dim=1)
    # A_in_yy = model.features_second_derivative(x_in, axis1=1, axis2=1).cat(dim=1)
    # A_on = model.features(x_on).cat(dim=1)

    A_in_xx = model.features_second_derivative(x_in, axis1=0, axis2=0)
    A_in_yy = model.features_second_derivative(x_in, axis1=1, axis2=1)
    A_on = model.features(x_on)

    A = pyrfm.concat_blocks([[-(A_in_xx + A_in_yy)], [A_on]])

    f_in = f_func(x_in).view(-1, 1)
    f_on = g_func(x_on).view(-1, 1)
    f = pyrfm.concat_blocks([[f_in], [f_on]])

    A_norm = torch.linalg.norm(A.cat(dim=1), ord=2, dim=1, keepdim=True)

    # sampled_x = []
    # sampled_labels = []
    # sampled_point_type_labels = []
    # # 计算每个子域的目标采样数量
    # target_fraction = 1.0 / n_subdomains

    # for i in range(n_subdomains):
    #     # 找到属于当前子域的点
    #     idx = (labels == i).nonzero(as_tuple=True)[0]
    #     num_points = idx.shape[0]
    #
    #     # 计算采样数量（向下取整）
    #     sample_num = int(num_points * target_fraction)
    #     if sample_num == 0 and num_points > 0:
    #         sample_num = 1  # 至少采一个点
    #
    #     # 随机采样
    #     perm = torch.randperm(num_points)[:sample_num]
    #     selected_idx = idx[perm]
    #
    #     sampled_x.append(x_all[selected_idx])
    #     sampled_labels.append(labels[selected_idx])
    #     sampled_point_type_labels.append(point_type_labels[selected_idx])
    #
    # # 合并采样后的结果
    # sampled_x = torch.cat(sampled_x, dim=0)
    # sampled_labels = torch.cat(sampled_labels, dim=0)
    # sampled_point_type_labels = torch.cat(sampled_point_type_labels, dim=0)
    #
    # sampled_x_in = sampled_x[sampled_point_type_labels.squeeze() == 0]
    # sampled_x_on = sampled_x[sampled_point_type_labels.squeeze() == 1]
    #
    # A_in_xx_sampled = model.features_second_derivative(sampled_x_in, axis1=0, axis2=0)
    # A_in_yy_sampled = model.features_second_derivative(sampled_x_in, axis1=1, axis2=1)
    # A_on_sampled = model.features(sampled_x_on)
    # A_sampled = pyrfm.concat_blocks([[-(A_in_xx_sampled + A_in_yy_sampled)], [A_on_sampled]]).cat(dim=1)
    # f_in_sampled = f_func(sampled_x_in).view(-1, 1)
    # f_on_sampled = g_func(sampled_x_on).view(-1, 1)
    # f_sampled = pyrfm.concat_blocks([[f_in_sampled], [f_on_sampled]])
    #
    # num_sampled_cols = 200
    # sampled_indices = torch.randperm(A_sampled.shape[1])[:num_sampled_cols]
    #
    # # 提取采样后的子矩阵
    # A_sampled_reduced = A_sampled[:, sampled_indices]
    # print(A_sampled_reduced.shape)
    #
    # A_sampled_reduced_norm = torch.linalg.norm(A_sampled_reduced, ord=2, dim=1, keepdim=True)
    # w_reduced = \
    #     torch.linalg.lstsq(A_sampled_reduced / A_sampled_reduced_norm, f_sampled / A_sampled_reduced_norm,
    #                        rcond=None)[
    #         0].view(-1, 1)
    # print("Residual : ", torch.linalg.norm(A_sampled_reduced @ w_reduced - f_sampled) / torch.linalg.norm(f_sampled))
    # w0 = torch.zeros((A_sampled.shape[1], 1), dtype=A_sampled.dtype, device=A_sampled.device)
    # w0[sampled_indices] = w_reduced

    # w0 = torch.linalg.lstsq(A.cat(dim=1) / A_norm, f / A_norm, rcond=None)[0]

    Qs = []
    Rs = []

    k = 1  # 可调整为任意正整数

    # A_reduced = torch.cat([A[i].mean(dim=1, keepdim=True) / A_norm for i in range(A.numel())], dim=1)
    # w = torch.linalg.lstsq(A_reduced, f / A_norm, rcond=None)[0].view(-1, 1)
    # w0 = torch.cat([torch.ones(A[i].shape[1], 1) * w[i] for i in range(A.numel())],
    #                dim=0)

    for i in range(0, A.numel(), k):
        # 收集 k 个子块，注意不要越界
        A_group = [A[j] / A_norm for j in range(i, min(i + k, A.numel()))]
        # 沿列拼接
        A_concat = torch.cat(A_group, dim=1)
        Q, R = torch.linalg.qr(A_concat)

        Qs.append(Q)
        Rs.append(R)

    QQ = torch.cat(Qs, dim=1)

    RR = torch.block_diag(*Rs)

    A_all = A.cat(dim=1) / A_norm
    sampled_indices = torch.randperm(A_all.shape[1])[:200]
    A_reduced = A_all[:, sampled_indices]
    w_reduced = torch.linalg.lstsq(A_reduced, (f / A_norm), rcond=None)[
        0].view(-1, 1)
    w = torch.zeros((A_all.shape[1], 1), dtype=A_all.dtype, device=A_all.device)
    w[sampled_indices] = w_reduced

    w0 = RR @ w

    # sampled_indices = torch.randperm(QQ.shape[1])[:1600]
    # Q_reduced = QQ[:, sampled_indices]
    # w = torch.linalg.lstsq(Q_reduced, f / A_norm, rcond=None)[0].view(-1, 1)
    # print("Residual : ", torch.linalg.norm(Q_reduced @ w - f / A_norm) / torch.linalg.norm(f / A_norm))
    # w0 = torch.zeros((QQ.shape[1], 1), dtype=QQ.dtype, device=QQ.device)
    # w0[sampled_indices] = w
    # print("Residual : ", torch.linalg.norm(QQ @ w0 - f / A_norm) / torch.linalg.norm(f / A_norm))

    # w0 = lsqr(QQ, f / A_norm, max_iter=100)
    # f_res = f / A_norm - QQ @ w0

    # sampled_indices = torch.randperm(QQ.shape[0])[:1000]
    #
    # Q_reduced = torch.cat([Qs[i].mean(dim=1, keepdim=True) for i in range(len(Qs))], dim=1)
    # w_reduced = torch.linalg.lstsq(Q_reduced, f_res)[0].view(-1, 1)
    # print("Residual : ", torch.linalg.norm(Q_reduced @ w_reduced - f_res) / torch.linalg.norm(f_res))
    # w_res = torch.cat([torch.ones(Qs[i].shape[1], 1) * w_reduced[i] / Qs[i].shape[1] for i in range(len(Qs))], dim=0)
    # w0 = w0 + w_res
    # print("Residual : ", torch.linalg.norm(QQ @ w0 - f / A_norm) / torch.linalg.norm(f / A_norm))

    # w = torch.linalg.lstsq(Q_reduced, f / A_norm, rcond=None)[0].view(-1, 1)
    # print("Residual : ", torch.linalg.norm(Q_reduced @ w - f / A_norm) / torch.linalg.norm(f / A_norm))
    # w0 = torch.cat([torch.ones(Qs[i].shape[1], 1) * w[i] / Qs[i].shape[1] for i in range(len(Qs))], dim=0)
    # print("Residual : ", torch.linalg.norm(QQ @ w0 - f / A_norm) / torch.linalg.norm(f / A_norm))

    # print("Condition number of A:", torch.linalg.cond(A.cat(dim=1) / A_norm))
    # print("Condition number of QQ:", torch.linalg.cond(QQ))

    # w0 = QQ.T @ (f / A_norm)

    w = lsqr(QQ, f / A_norm, max_iter=A.cat(dim=1).shape[1] * 10, x0=w0)
    # w = lsqr(QQ, f / A_norm, max_iter=A.cat(dim=1).shape[1] * 10)
    # w = lsmr(QQ, f / A_norm, max_iter=10000, atol=1e-8, btol=1e-8)
    # w = gmres(QQ.T @ QQ, QQ.T @ f, max_iter=10000, tol=1e-8, verbose=True)[0]
    # w = cg(QQ.T @ QQ, QQ.T @ f, max_iter=10000, tol=1e-8, verbose=True)[0]

    model.W = torch.linalg.solve_triangular(RR, w, upper=True)

    # model.compute(A.cat(dim=1)).solve(f)

    x_test = domain.in_sample(40, with_boundary=True)
    u_test = u_func(x_test).view(-1, 1)
    u_pred = model(x_test)

    print((u_test - u_pred).norm() / u_test.norm())

    print('Time:', time.time() - start_time, ", with", torch.tensor(0.).device)
