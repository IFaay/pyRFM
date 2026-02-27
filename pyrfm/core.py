# -*- coding: utf-8 -*-
"""
Created on 2024/12/13

@author: Yifei Sun
"""
import time

import torch

from .geometry import GeometryBase, Line1D, Square2D, Cube3D, Point1D, Line2D, Square3D
from .voronoi import Voronoi
from .utils import *


class RFBase(ABC):
    def clone(self, deep: bool = True):
        """
        Create a (deep) clone of the current RFBase instance.
        :param deep: Whether to deep copy tensors and submodules.
        :return: A new RFBase instance with the same configuration.
        """
        import copy
        return copy.deepcopy(self) if deep else copy.copy(self)

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
                 device: torch.device = None,
                 use_cache: bool = True):
        super().__init__(dim, center, radius, nn.Tanh(), n_hidden, gen, dtype, device)
        self.use_cache = use_cache

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.dim:
            raise ValueError('Input dimension mismatch')
        # Be careful when x in a slice
        if (self.x_buff_ is not None and self.use_cache) and (self.x_buff_ is x or torch.equal(self.x_buff_, x)):
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

        # Be careful when x in a slice
        if (self.x_buff_ is not None and self.use_cache) and (self.x_buff_ is x or torch.equal(self.x_buff_, x)):
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

        # Be careful when x in a slice
        if (self.x_buff_ is not None and self.use_cache) and (self.x_buff_ is x or torch.equal(self.x_buff_, x)):
            pass
        else:
            self.forward(x)

        return -2 * self.features_buff_ * (1 - torch.pow(self.features_buff_, 2)) * \
            (self.weights[[axis1], :] / self.radius[0, axis1]) * (
                    self.weights[[axis2], :] / self.radius[0, axis2])

    def higher_order_derivative(self, x: torch.Tensor, order: Union[torch.Tensor, List]) -> torch.Tensor:
        if isinstance(order, list):
            order = torch.tensor(order, dtype=torch.int, device=self.device)
        if x.shape[1] != self.dim:
            raise ValueError('Input dimension mismatch')
        if order.shape[0] != self.dim:
            raise ValueError('Order dimension mismatch')

        n_order = int(order.sum())
        if n_order <= 0:
            raise ValueError('Order must be positive')

        if (self.x_buff_ is not None and self.use_cache) and (self.x_buff_ is x or torch.equal(self.x_buff_, x)):
            t = self.features_buff_
        else:
            t = torch.tanh(
                torch.matmul((x - self.center) / self.radius, self.weights) + self.biases)

        # 用多项式系数追踪 P_n(t)
        # P_1(t) = 1 - t^2, 系数: c[0]=1, c[1]=0, c[2]=-1
        # 索引 k 对应 t^k
        max_deg = n_order + 1
        c = torch.zeros(max_deg + 1, dtype=self.dtype if hasattr(self, 'dtype') else t.dtype,
                        device=self.device)
        # P_1: 1 - t^2
        c[0] = 1.0
        c[2] = -1.0

        for _ in range(1, n_order):
            # Step1: 对 c 求导 -> dc[k] = (k+1)*c[k+1]
            dc = torch.zeros_like(c)
            for k in range(max_deg):
                dc[k] = (k + 1) * c[k + 1]
            # Step2: 乘以 (1-t^2) -> new_c[k] = dc[k] - dc[k-2]
            new_c = dc.clone()
            new_c[2:] -= dc[:-2]
            c = new_c

        # 用 Horner 方法计算 P_n(t) 的值
        # t shape: (batch, n_hidden)
        result = torch.zeros_like(t)
        for k in range(max_deg, -1, -1):
            result = result * t + c[max_deg - k] if False else None  # 需要正向Horner

        # 正向 Horner: P = c[d]*t^d + ... + c[0]
        deg = max_deg
        result = torch.full_like(t, c[deg].item())
        for k in range(deg - 1, -1, -1):
            result = result * t + c[k]

        # 乘以链式法则权重
        weight_factor = torch.ones(1, self.weights.shape[1], dtype=t.dtype, device=self.device)
        for i in range(order.shape[0]):
            for _ in range(int(order[i])):
                weight_factor = weight_factor * (self.weights[[i], :] / self.radius[0, i])

        return result * weight_factor


class RFTanH2(RFBase):
    """Two-layer: y = tanh( inner(x) @ W2 + b2 ), inner(x) = tanh( ((x-c)/r) @ W1 + b1 )"""

    def __init__(self, dim: int, center: torch.Tensor, radius: torch.Tensor,
                 n_hidden: int,
                 gen: torch.Generator = None,
                 dtype: torch.dtype = None,
                 device: torch.device = None,
                 use_cache: bool = True):
        super().__init__(n_hidden, center, radius, nn.Tanh(), n_hidden, gen, dtype, device)
        self.inner = RFTanH(dim, center, radius, n_hidden, gen, dtype, device, use_cache)
        self.use_cache = use_cache

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.inner.dim:
            raise ValueError('Input dimension mismatch')
        # Be careful when x in a slice
        if (self.features_buff_ is not None and self.use_cache) and (self.x_buff_ is x or torch.equal(self.x_buff_, x)):
            return self.features_buff_
        self.x_buff_ = x
        inner_features = self.inner.forward(x)
        self.features_buff_ = torch.tanh(
            torch.matmul(inner_features, self.weights) + self.biases)
        return self.features_buff_

    def first_derivative(self, x: torch.Tensor, axis: int) -> torch.Tensor:
        # 轴检查在原始输入维度空间
        if x.shape[1] != self.inner.dim:
            raise ValueError('Input dimension mismatch')
        if axis >= self.inner.dim:
            raise ValueError('Axis out of range')

        # 准备外层前向和内层一阶
        if (self.features_buff_ is None and self.use_cache) or not (self.x_buff_ is x or torch.equal(self.x_buff_, x)):
            self.forward(x)

        y = self.features_buff_  # (N, n_hidden)
        dh_dx_i = self.inner.first_derivative(x, axis)  # (N, n_hidden)

        # d y / d x_i = (1 - y^2) ⊙ ( dh/dx_i @ W2 )
        return (1 - y ** 2) * (dh_dx_i @ self.weights)  # 逐元素乘，最终 (N, n_hidden)

    def second_derivative(self, x: torch.Tensor, axis1: int, axis2: int) -> torch.Tensor:
        if x.shape[1] != self.inner.dim:
            raise ValueError('Input dimension mismatch')
        if axis1 >= self.inner.dim:
            raise ValueError('Axis1 out of range')
        if axis2 >= self.inner.dim:
            raise ValueError('Axis2 out of range')

        # 确保缓存
        if (self.features_buff_ is None and self.use_cache) or not (self.x_buff_ is x or torch.equal(self.x_buff_, x)):
            self.forward(x)

        y = self.features_buff_  # (N, n_hidden)
        # 需要两种内层导数
        dh_dxi = self.inner.first_derivative(x, axis1)  # (N, n_hidden)
        dh_dxj = self.inner.first_derivative(x, axis2)  # (N, n_hidden)
        d2h_dxidxj = self.inner.second_derivative(x, axis1, axis2)  # (N, n_hidden)

        # 先得到 dy/dx_j 以便构造项1
        dy_dxj = (1 - y ** 2) * (dh_dxj @ self.weights)  # (N, n_hidden)

        # 项1：(-2*y ⊙ dy/dx_j) ⊙ (dh/dx_i @ W2)
        term1 = (-2 * y * dy_dxj) * (dh_dxi @ self.weights)  # (N, n_hidden)

        # 项2：(1 - y^2) ⊙ ( (d2h/dxidxj) @ W2 )
        term2 = (1 - y ** 2) * (d2h_dxidxj @ self.weights)  # (N, n_hidden)

        return term1 + term2

    def higher_order_derivative(self, x: torch.Tensor, order: Union[torch.Tensor, List]) -> torch.Tensor:
        # 与单层保持相同的输入校验
        if isinstance(order, List):
            order = torch.tensor(order, dtype=self.dtype, device=self.device)
        if x.shape[1] != self.inner.dim:
            raise ValueError('Input dimension mismatch')
        if order.shape[0] != self.inner.dim:
            raise ValueError('Order dimension mismatch')

        n_order = int(order.sum().item() if isinstance(order, torch.Tensor) else sum(order))
        if n_order <= 0:
            raise ValueError('Order must be positive')

        # 只实现到 2 阶；更高阶需要 Faà di Bruno 型组合，若需要可继续扩展
        if n_order == 1:
            axis = int(torch.nonzero(order, as_tuple=False)[0].item())
            return self.first_derivative(x, axis)
        if n_order == 2:
            nz = torch.nonzero(order, as_tuple=False).flatten()
            if len(nz) == 1:
                a = int(nz[0].item())
                return self.second_derivative(x, a, a)
            elif len(nz) == 2:
                a, b = int(nz[0].item()), int(nz[1].item())
                return self.second_derivative(x, a, b)
            else:
                # e.g., [1,1,0,...] 已覆盖；其它组合理论上也可拆分成 axis1, axis2
                a, b = int(nz[0].item()), int(nz[1].item())
                return self.second_derivative(x, a, b)

        raise NotImplementedError("higher_order_derivative for order >= 3 is not implemented yet.")


class RFReLU(RFBase):
    def __init__(self, dim: int, center: torch.Tensor, radius: torch.Tensor,
                 n_hidden: int,
                 gen: torch.Generator = None,
                 dtype: torch.dtype = None,
                 device: torch.device = None,
                 use_cache: bool = True):
        super().__init__(dim, center, radius, nn.ReLU(), n_hidden, gen, dtype, device)
        self.use_cache = use_cache

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.dim:
            raise ValueError("Input dimension mismatch")
        if (self.x_buff_ is not None and self.use_cache) and (self.x_buff_ is x or torch.equal(self.x_buff_, x)):
            return self.features_buff_

        self.x_buff_ = x
        z = torch.matmul((x - self.center) / self.radius, self.weights) + self.biases
        self.features_buff_ = torch.relu(z)
        return self.features_buff_

    def first_derivative(self, x: torch.Tensor, axis: int) -> torch.Tensor:
        if x.shape[1] != self.dim:
            raise ValueError("Input dimension mismatch")
        if axis >= self.dim:
            raise ValueError("Axis out of range")

        if (self.x_buff_ is None or not (self.x_buff_ is x or torch.equal(self.x_buff_, x))):
            self.forward(x)

        # ReLU'(z) = 1 if z > 0 else 0
        grad_mask = (self.features_buff_ > 0).to(self.dtype)
        return grad_mask * (self.weights[[axis], :] / self.radius[0, axis])

    def second_derivative(self, x: torch.Tensor, axis1: int, axis2: int) -> torch.Tensor:
        # ReLU 的二阶导在几乎所有点都为 0（除了 z=0 的不可导点）
        if x.shape[1] != self.dim:
            raise ValueError("Input dimension mismatch")
        if axis1 >= self.dim or axis2 >= self.dim:
            raise ValueError("Axis out of range")
        return torch.zeros(x.shape[0], self.n_hidden, dtype=self.dtype, device=self.device)

    def higher_order_derivative(self, x: torch.Tensor, order: Union[torch.Tensor, List]) -> torch.Tensor:
        if isinstance(order, List):
            order = torch.tensor(order, dtype=self.dtype, device=self.device)
        if x.shape[1] != self.dim:
            raise ValueError("Input dimension mismatch")
        if order.shape[0] != self.dim:
            raise ValueError("Order dimension mismatch")

        n_order = order.sum()
        if n_order <= 0:
            raise ValueError("Order must be positive")
        if n_order == 1:
            axis = int(torch.nonzero(order, as_tuple=False)[0].item())
            return self.first_derivative(x, axis)

        # ReLU 高阶导数在几乎所有点都为 0
        return torch.zeros(x.shape[0], self.n_hidden, dtype=self.dtype, device=self.device)


class RFReLUTanH(RFBase):
    """Two-layer: y = tanh( inner(x) @ W2 + b2 ), inner(x) = ReLU( ((x-c)/r) @ W1 + b1 )"""

    def __init__(self, dim: int, center: torch.Tensor, radius: torch.Tensor,
                 n_hidden: int,
                 gen: torch.Generator = None,
                 dtype: torch.dtype = None,
                 device: torch.device = None,
                 use_cache: bool = True):
        super().__init__(n_hidden, center, radius, nn.Tanh(), n_hidden, gen, dtype, device)
        self.inner = RFReLU(dim, center, radius, n_hidden, gen, dtype, device, use_cache)
        self.use_cache = use_cache

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.inner.dim:
            raise ValueError("Input dimension mismatch")

        if (self.features_buff_ is not None and self.use_cache) and (self.x_buff_ is x or torch.equal(self.x_buff_, x)):
            return self.features_buff_

        self.x_buff_ = x
        inner_features = self.inner.forward(x)  # (N, n_hidden)
        self.features_buff_ = torch.tanh(
            torch.matmul(inner_features, self.weights) + self.biases
        )
        return self.features_buff_

    def first_derivative(self, x: torch.Tensor, axis: int) -> torch.Tensor:
        if x.shape[1] != self.inner.dim:
            raise ValueError("Input dimension mismatch")
        if axis >= self.inner.dim:
            raise ValueError("Axis out of range")

        if (self.features_buff_ is None and self.use_cache) or not (self.x_buff_ is x or torch.equal(self.x_buff_, x)):
            self.forward(x)

        y = self.features_buff_  # (N, n_hidden)
        dh_dx_i = self.inner.first_derivative(x, axis)  # (N, n_hidden)

        # dy/dx_i = (1 - y^2) ⊙ (dh/dx_i @ W2)
        return (1 - y ** 2) * (dh_dx_i @ self.weights)

    def second_derivative(self, x: torch.Tensor, axis1: int, axis2: int) -> torch.Tensor:
        if x.shape[1] != self.inner.dim:
            raise ValueError("Input dimension mismatch")
        if axis1 >= self.inner.dim or axis2 >= self.inner.dim:
            raise ValueError("Axis out of range")

        if (self.features_buff_ is None and self.use_cache) or not (self.x_buff_ is x or torch.equal(self.x_buff_, x)):
            self.forward(x)

        y = self.features_buff_  # (N, n_hidden)
        dh_dxi = self.inner.first_derivative(x, axis1)  # (N, n_hidden)
        dh_dxj = self.inner.first_derivative(x, axis2)  # (N, n_hidden)
        d2h_dxidxj = self.inner.second_derivative(x, axis1, axis2)  # (N, n_hidden)

        # dy/dx_j
        dy_dxj = (1 - y ** 2) * (dh_dxj @ self.weights)

        # term1: (-2*y ⊙ dy/dx_j) ⊙ (dh/dx_i @ W2)
        term1 = (-2 * y * dy_dxj) * (dh_dxi @ self.weights)

        # term2: (1 - y^2) ⊙ (d2h/dxidxj @ W2)
        term2 = (1 - y ** 2) * (d2h_dxidxj @ self.weights)

        return term1 + term2

    def higher_order_derivative(self, x: torch.Tensor, order: Union[torch.Tensor, List]) -> torch.Tensor:
        if isinstance(order, List):
            order = torch.tensor(order, dtype=self.dtype, device=self.device)
        if x.shape[1] != self.inner.dim:
            raise ValueError("Input dimension mismatch")
        if order.shape[0] != self.inner.dim:
            raise ValueError("Order dimension mismatch")

        n_order = int(order.sum().item() if isinstance(order, torch.Tensor) else sum(order))
        if n_order <= 0:
            raise ValueError("Order must be positive")

        if n_order == 1:
            axis = int(torch.nonzero(order, as_tuple=False)[0].item())
            return self.first_derivative(x, axis)
        if n_order == 2:
            nz = torch.nonzero(order, as_tuple=False).flatten()
            if len(nz) == 1:
                a = int(nz[0].item())
                return self.second_derivative(x, a, a)
            elif len(nz) == 2:
                a, b = int(nz[0].item()), int(nz[1].item())
                return self.second_derivative(x, a, b)
            else:
                a, b = int(nz[0].item()), int(nz[1].item())
                return self.second_derivative(x, a, b)

        raise NotImplementedError("higher_order_derivative for order >= 3 is not implemented yet.")


class RFCos(RFBase):
    def __init__(self, dim: int, center: torch.Tensor, radius: torch.Tensor,
                 n_hidden: int,
                 gen: torch.Generator = None,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        class Cos(nn.Module):
            def __init__(self):
                super(Cos, self).__init__()

            def forward(self, x):
                return torch.cos(x)

        super().__init__(dim, center, radius, Cos(), n_hidden, gen, dtype, device)
        self.features_cos_buff_ = None
        self.features_sin_buff_ = None

    def empty_cache(self):
        super().empty_cache()
        self.features_cos_buff_ = None
        self.features_sin_buff_ = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.dim:
            raise ValueError('Input dimension mismatch')
        # Be careful when x in a slice
        if (self.features_cos_buff_ is not None) and (self.x_buff_ is x or torch.equal(self.x_buff_, x)):
            return self.features_cos_buff_
        self.x_buff_ = x
        self.features_cos_buff_ = torch.cos(
            torch.matmul((x - self.center) / self.radius, self.weights) + self.biases)
        self.features_sin_buff_ = None
        return self.features_cos_buff_

    def first_derivative(self, x: torch.Tensor, axis: int) -> torch.Tensor:
        if x.shape[1] != self.dim:
            raise ValueError('Input dimension mismatch')

        if axis >= self.dim:
            raise ValueError('Axis out of range')

        # Be careful when x in a slice
        if (self.features_sin_buff_ is not None) and (self.x_buff_ is x or torch.equal(self.x_buff_, x)):
            pass
        else:
            self.x_buff_ = x
            self.features_cos_buff_ = None
            self.features_sin_buff_ = torch.sin(
                torch.matmul((x - self.center) / self.radius, self.weights) + self.biases)

        return -self.features_sin_buff_ * (self.weights[[axis], :] / self.radius[0, axis])

    def second_derivative(self, x: torch.Tensor, axis1: int, axis2: int) -> torch.Tensor:
        if x.shape[1] != self.dim:
            raise ValueError('Input dimension mismatch')

        if axis1 >= self.dim:
            raise ValueError('Axis1 out of range')

        if axis2 >= self.dim:
            raise ValueError('Axis2 out of range')

        # Be careful when x in a slice
        if (self.features_cos_buff_ is not None) and (self.x_buff_ is x or torch.equal(self.x_buff_, x)):
            pass
        else:
            self.x_buff_ = x
            self.features_cos_buff_ = torch.cos(
                torch.matmul((x - self.center) / self.radius, self.weights) + self.biases)
            self.features_sin_buff_ = None

        return -self.features_cos_buff_ * (self.weights[[axis1], :] / self.radius[0, axis1]) * \
            (self.weights[[axis2], :] / self.radius[0, axis2])

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
            if self.features_cos_buff_ is not None:
                t1 = self.features_cos_buff_
            else:
                t1 = torch.cos(
                    torch.matmul((x - self.center) / self.radius, self.weights) + self.biases)
            if self.features_sin_buff_ is not None:
                t2 = self.features_sin_buff_
            else:
                t2 = torch.sin(
                    torch.matmul((x - self.center) / self.radius, self.weights) + self.biases)
        else:
            self.x_buff_ = x
            self.features_cos_buff_ = torch.cos(
                torch.matmul((x - self.center) / self.radius, self.weights) + self.biases)
            self.features_sin_buff_ = torch.sin(
                torch.matmul((x - self.center) / self.radius, self.weights) + self.biases)
            t1 = self.features_cos_buff_
            t2 = self.features_sin_buff_

        if n_order % 4 == 0:
            p_n = t1
        elif n_order % 4 == 1:
            p_n = -t2
        elif n_order % 4 == 2:
            p_n = -t1
        else:
            p_n = t2

        for i in range(order.shape[0]):
            for _ in range(order[i]):
                p_n *= (self.weights[[i], :] / self.radius[0, i])

        return p_n


class RFSin(RFBase):
    def __init__(self, dim: int, center: torch.Tensor, radius: torch.Tensor,
                 n_hidden: int,
                 gen: torch.Generator = None,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        class Sin(nn.Module):
            def __init__(self):
                super(Sin, self).__init__()

            def forward(self, x):
                return torch.sin(x)

        super().__init__(dim, center, radius, Sin(), n_hidden, gen, dtype, device)
        self.features_cos_buff_ = None
        self.features_sin_buff_ = None

    def empty_cache(self):
        super().empty_cache()
        self.features_cos_buff_ = None
        self.features_sin_buff_ = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.dim:
            raise ValueError('Input dimension mismatch')
        # Be careful when x in a slice
        if (self.features_sin_buff_ is not None) and (self.x_buff_ is x or torch.equal(self.x_buff_, x)):
            return self.features_sin_buff_
        self.x_buff_ = x
        self.features_sin_buff_ = torch.sin(
            torch.matmul((x - self.center) / self.radius, self.weights) + self.biases)
        self.features_cos_buff_ = None
        return self.features_sin_buff_

    def first_derivative(self, x: torch.Tensor, axis: int) -> torch.Tensor:
        if x.shape[1] != self.dim:
            raise ValueError('Input dimension mismatch')

        if axis >= self.dim:
            raise ValueError('Axis out of range')

        # Be careful when x in a slice
        if (self.features_cos_buff_ is not None) and (self.x_buff_ is x or torch.equal(self.x_buff_, x)):
            pass
        else:
            self.x_buff_ = x
            self.features_sin_buff_ = None
            self.features_cos_buff_ = torch.cos(
                torch.matmul((x - self.center) / self.radius, self.weights) + self.biases)

        return self.features_cos_buff_ * (self.weights[[axis], :] / self.radius[0, axis])

    def second_derivative(self, x: torch.Tensor, axis1: int, axis2: int) -> torch.Tensor:
        if x.shape[1] != self.dim:
            raise ValueError('Input dimension mismatch')

        if axis1 >= self.dim:
            raise ValueError('Axis1 out of range')

        if axis2 >= self.dim:
            raise ValueError('Axis2 out of range')

        with torch.no_grad():
            # Be careful when x in a slice
            if (self.features_sin_buff_ is not None) and (self.x_buff_ is x or torch.equal(self.x_buff_, x)):
                pass
            else:
                self.x_buff_ = x
                self.features_sin_buff_ = torch.sin(
                    torch.matmul((x - self.center) / self.radius, self.weights) + self.biases)
                self.features_cos_buff_ = None

            return -self.features_sin_buff_ * (self.weights[[axis1], :] / self.radius[0, axis1]) * \
                (self.weights[[axis2], :] / self.radius[0, axis2])

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
            if self.features_sin_buff_ is not None:
                t1 = self.features_sin_buff_
            else:
                t1 = torch.sin(
                    torch.matmul((x - self.center) / self.radius, self.weights) + self.biases)
            if self.features_cos_buff_ is not None:
                t2 = self.features_cos_buff_
            else:
                t2 = torch.cos(
                    torch.matmul((x - self.center) / self.radius, self.weights) + self.biases)
        else:
            self.x_buff_ = x
            self.features_sin_buff_ = torch.sin(
                torch.matmul((x - self.center) / self.radius, self.weights) + self.biases)
            self.features_cos_buff_ = torch.cos(
                torch.matmul((x - self.center) / self.radius, self.weights) + self.biases)
            t1 = self.features_sin_buff_
            t2 = self.features_cos_buff_

        if n_order % 4 == 0:
            p_n = t1
        elif n_order % 4 == 1:
            p_n = t2
        elif n_order % 4 == 2:
            p_n = -t1
        else:
            p_n = -t2

        for i in range(order.shape[0]):
            for _ in range(order[i]):
                p_n *= (self.weights[[i], :] / self.radius[0, i])

        return p_n


class POUBase(ABC):
    def clone(self, deep: bool = True):
        """
        Create a (deep) clone of the current POUBase instance.
        :param deep: Whether to deep copy tensors and submodules.
        :return: A new POUBase instance with the same configuration.
        """
        import copy
        return copy.deepcopy(self) if deep else copy.copy(self)

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

    def higher_order_derivative(self, x: torch.Tensor, order: Union[torch.Tensor, List]) -> torch.Tensor:
        pass


class PsiA(POUBase):
    def set_func(self):
        self.func = lambda x: torch.where(x < -1.0, 0.0, torch.where(x > 1.0, 0.0, 1.0))
        self.d_func = lambda x: torch.zeros((x.shape[0], 1), dtype=self.dtype, device=self.device)
        self.d2_func = lambda x: torch.zeros((x.shape[0], 1), dtype=self.dtype, device=self.device)

    def higher_order_derivative(self, x: torch.Tensor, order: Union[torch.Tensor, List]) -> torch.Tensor:
        return torch.zeros((x.shape[0], 1), dtype=self.dtype, device=self.device)


class PsiBW(POUBase):
    # A wider partition of unity function of Type B
    def set_func(self):
        self.func = lambda x: torch.where(x <= -3.0 / 2.0, 0.0,
                                          torch.where(x <= -1.0 / 2.0,
                                                      1.0 / 2.0 * (1.0 - torch.sin(torch.pi * x)),
                                                      torch.where(x <= 1.0 / 2.0, 1.0,
                                                                  torch.where(x <= 3.0 / 2.0, 1.0 / 2.0 * (
                                                                          1.0 + torch.sin(
                                                                      torch.pi * x)), 0.0)))
                                          )
        self.d_func = lambda x: torch.where(x <= -3.0 / 2.0, 0.0,
                                            torch.where(x <= -1.0 / 2.0,
                                                        -1.0 / 2.0 * torch.pi * torch.cos(torch.pi * x),
                                                        torch.where(x <= 1.0 / 2.0, 0.0,
                                                                    torch.where(x <= 3.0 / 2.0,
                                                                                + 1.0 / 2.0 * torch.pi * torch.cos(
                                                                                    torch.pi * x), 0.0)))
                                            )
        self.d2_func = lambda x: torch.where(x <= -3.0 / 2.0, 0.0,
                                             torch.where(x <= -1.0 / 2.0,
                                                         1.0 / 2.0 * torch.pi ** 2 * torch.sin(torch.pi * x),
                                                         torch.where(x <= 1.0 / 2.0, 0.0,
                                                                     torch.where(x <= 3.0 / 2.0,
                                                                                 - 1.0 / 2.0 * torch.pi ** 2 * torch.sin(
                                                                                     torch.pi * x), 0.0)))
                                             )


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

    def _nth_derivative_1d(self, xd: torch.Tensor, n: int) -> torch.Tensor:
        """
        1D n-th derivative of PsiB w.r.t. normalized coordinate xd.
        n must be >= 1.
        """
        phase = n * torch.pi / 2.0
        factor = (2.0 * torch.pi) ** n / 2.0
        sin_term = factor * torch.sin(2.0 * torch.pi * xd + phase)

        return torch.where(xd < -1.25, torch.zeros_like(xd),
                           torch.where(xd < -0.75, sin_term,
                                       torch.where(xd <= 0.75, torch.zeros_like(xd),
                                                   torch.where(xd <= 1.25, -sin_term,
                                                               torch.zeros_like(xd)))))

    def higher_order_derivative(self, x: torch.Tensor, order: Union[torch.Tensor, List]) -> torch.Tensor:
        if isinstance(order, list):
            order = torch.tensor(order, dtype=torch.long, device=self.device)
        if x.shape[1] != self.center.shape[1]:
            raise ValueError('Input dimension mismatch')
        if order.numel() != x.shape[1]:
            raise ValueError('Order dimension mismatch')

        x_ = (x - self.center) / self.radius
        prod = torch.ones((x_.shape[0], 1), dtype=self.dtype, device=self.device)

        for d in range(x_.shape[1]):
            n = int(order[d].item())
            xd = x_[:, [d]]
            if n == 0:
                prod *= self.func(xd)
            elif n == 1:
                prod *= self.d_func(xd) / self.radius[0, d]
            elif n == 2:
                prod *= self.d2_func(xd) / self.radius[0, d] ** 2
            else:
                prod *= self._nth_derivative_1d(xd, n) / self.radius[0, d] ** n

        return prod


class PsiG(POUBase):
    def __init__(self, center: torch.Tensor, radius: torch.Tensor,
                 mu: torch.Tensor = None, sigma: torch.Tensor = None,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__(center,
                         radius,
                         dtype=dtype, device=device)

        # 默认值设置
        if mu is None:
            mu = torch.zeros_like(center)
        if sigma is None:
            # sigma = 0.5 使得在 x = ±1 处,高斯函数衰减到 exp(-4) ≈ 0.0183
            # 这样在标准化坐标 [-1, 1] 的边界处函数值接近0
            sigma = torch.full_like(center, 0.5)

        self.mu = mu.to(dtype=self.dtype, device=self.device)
        self.sigma = sigma.to(dtype=self.dtype, device=self.device)

    def set_func(self):
        # 高斯函数: exp(-((x-mu)/sigma)^2)
        # 为了构造紧支撑的POU函数,使用截断的高斯函数
        def gaussian(x):
            z = (x - self.mu) / self.sigma
            return torch.exp(-z ** 2)

        # 归一化函数,使其在支撑域内积分为1
        self.func = lambda x: torch.where(
            torch.abs(x) < 1.2,  # 紧支撑在 [-1, 1]
            gaussian(x),
            torch.zeros_like(x)
        )

        # 一阶导数: d/dx[exp(-z^2)] = -2z/sigma * exp(-z^2)
        self.d_func = lambda x: torch.where(
            torch.abs(x) < 1.2,
            -2.0 * (x - self.mu) / (self.sigma ** 2) * gaussian(x),
            torch.zeros_like(x)
        )

        # 二阶导数: d^2/dx^2[exp(-z^2)] = (-2/sigma^2 + 4z^2/sigma^2) * exp(-z^2)
        self.d2_func = lambda x: torch.where(
            torch.abs(x) < 1.2,
            (-2.0 / (self.sigma ** 2) + 4.0 * (x - self.mu) ** 2 / (self.sigma ** 4)) * gaussian(x),
            torch.zeros_like(x)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 标准化坐标到 [-1, 1]
        x_normalized = (x - self.center) / self.radius
        return self.func(x_normalized)

    def first_derivative(self, x: torch.Tensor, axis: int) -> torch.Tensor:
        # 标准化坐标
        x_normalized = (x[..., axis:axis + 1] - self.center[axis]) / self.radius[axis]
        # 链式法则: d/dx = d/dx_norm * dx_norm/dx = d/dx_norm * (1/radius)
        return self.d_func(x_normalized) / self.radius[axis]

    def second_derivative(self, x: torch.Tensor, axis1: int, axis2: int) -> torch.Tensor:
        x_normalized = (x[..., axis1:axis1 + 1] - self.center[axis1]) / self.radius[axis1]

        if axis1 == axis2:
            # 同一轴的二阶导数
            return self.d2_func(x_normalized) / (self.radius[axis1] ** 2)
        else:
            # 不同轴的混合偏导数 (高斯函数可分离)
            x_norm1 = (x[..., axis1:axis1 + 1] - self.center[axis1]) / self.radius[axis1]
            x_norm2 = (x[..., axis2:axis2 + 1] - self.center[axis2]) / self.radius[axis2]

            return (self.d_func(x_norm1) / self.radius[axis1]) * \
                (self.d_func(x_norm2) / self.radius[axis2])

    def higher_order_derivative(self, x: torch.Tensor, order: Union[torch.Tensor, List]) -> torch.Tensor:
        """
        计算高阶导数
        order: 每个维度的导数阶数,例如 [2, 1, 0] 表示对第0维求2阶,第1维求1阶,第2维不求导
        """
        if isinstance(order, list):
            order = torch.tensor(order, dtype=torch.long, device=self.device)

        result = torch.ones_like(x[..., 0:1])

        for axis in range(len(order)):
            n = order[axis].item()
            if n == 0:
                continue

            x_normalized = (x[..., axis:axis + 1] - self.center[axis]) / self.radius[axis]
            z = (x_normalized - self.mu) / self.sigma
            gaussian_val = torch.exp(-z ** 2)

            # 使用Hermite多项式计算高阶导数
            # d^n/dx^n[exp(-z^2)] = (-1)^n / sigma^n * H_n(z) * exp(-z^2)
            # 其中 H_n 是物理学家的Hermite多项式
            hermite_val = self._hermite_polynomial(z, n)
            deriv = ((-1) ** n) / (self.sigma ** n) * hermite_val * gaussian_val

            result = result * deriv / (self.radius[axis] ** n)

        return result

    def _hermite_polynomial(self, z: torch.Tensor, n: int) -> torch.Tensor:
        """计算物理学家的Hermite多项式 H_n(z)"""
        if n == 0:
            return torch.ones_like(z)
        elif n == 1:
            return 2.0 * z

        # 递推关系: H_{n+1}(z) = 2z*H_n(z) - 2n*H_{n-1}(z)
        H_prev = torch.ones_like(z)
        H_curr = 2.0 * z

        for i in range(2, n + 1):
            H_next = 2.0 * z * H_curr - 2.0 * (i - 1) * H_prev
            H_prev = H_curr
            H_curr = H_next

        return H_curr


class PsiBump(POUBase):
    """
    C^∞ bump with compact support in normalized coordinate [-r, r], r=1.25:

        b(x) = exp( -1 / (1 - (x/r)^2) ),  |x| < r
             = 0                           otherwise

    POUBase builds ND as product over dimensions.
    """

    def __init__(self, center: torch.Tensor, radius: torch.Tensor,
                 r: float = 1.2,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        self.r_value = float(r)
        super().__init__(center, radius, dtype=dtype, device=device)

    def set_func(self):
        r = torch.tensor(self.r_value, dtype=self.dtype, device=self.device)

        def _bump_inside(x):
            z = x / r
            t = torch.clamp(1.0 - z * z, min=1e-10)  # 防止边界处 t->0 时梯度爆炸
            return torch.exp(-1.0 / t)

        # C∞ bump (piecewise)
        def bump(x):
            return torch.where(torch.abs(x) < r, _bump_inside(x), torch.zeros_like(x))

        # keep these for POUBase first/second derivative usage (closed-form 1st/2nd)
        def dbump(x):
            z = x / r
            t = 1.0 - z * z
            b = torch.exp(-1.0 / t)
            val = b * (-2.0 * x) / (r * r * t * t)
            return torch.where(torch.abs(x) < r, val, torch.zeros_like(x))

        def d2bump(x):
            z = x / r
            t = 1.0 - z * z
            b = torch.exp(-1.0 / t)

            r2 = r * r
            r4 = r2 * r2
            t2 = t * t
            t3 = t2 * t
            t4 = t2 * t2

            term1 = (-2.0) / (r2 * t2)
            term2 = (-8.0 * x * x) / (r4 * t3)
            term3 = (4.0 * x * x) / (r4 * t4)
            val = b * (term1 + term2 + term3)
            return torch.where(torch.abs(x) < r, val, torch.zeros_like(x))

        self.func = bump
        self.d_func = dbump
        self.d2_func = d2bump

    # ---------- helper: 1D n-th derivative in normalized coordinate ----------
    def _nth_derivative_1d(self, x1d: torch.Tensor, n: int) -> torch.Tensor:
        """
        Compute d^n/dx^n self.func(x) for 1D input x1d (shape [N,1]) using autograd recursion.
        Note: This differentiates the *normalized-coordinate* bump (already compactly supported).
        """
        if n == 0:
            return self.func(x1d)

        # Ensure a leaf with grad
        x = x1d.clone().detach().requires_grad_(True)

        y = self.func(x)
        for _ in range(n):
            # grad_outputs must match y
            g = torch.autograd.grad(
                outputs=y,
                inputs=x,
                grad_outputs=torch.ones_like(y),
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]
            y = g
        return y

    # ---------- ND higher-order derivative (multi-index) ----------
    def higher_order_derivative(self, x: torch.Tensor, order: Union[torch.Tensor, List]) -> torch.Tensor:
        """
        Multi-index derivative:
            order = [n0, n1, ..., n_{D-1}]
        returns:
            Π_d  ( d^{n_d}/dx_d^{n_d}  φ_d( (x_d-c_d)/R_d ) )
        where chain rule gives division by R_d^{n_d}.
        """
        if isinstance(order, list):
            order = torch.tensor(order, dtype=torch.long, device=self.device)
        else:
            order = order.to(dtype=torch.long, device=self.device)

        if x.shape[1] != self.center.shape[1] or x.shape[1] != self.radius.shape[1]:
            raise ValueError('Input dimension mismatch')
        if order.numel() != x.shape[1]:
            raise ValueError('Order dimension mismatch')

        x_ = (x - self.center) / self.radius  # normalized coords, shape [N, D]

        prod = torch.ones((x_.shape[0], 1), dtype=self.dtype, device=self.device)

        for d in range(x_.shape[1]):
            n = int(order[d].item())
            xd = x_[:, [d]]

            # n-th derivative w.r.t normalized coordinate
            deriv_nd = self._nth_derivative_1d(xd, n)

            # chain rule: d^n/dx_d^n = (1/R_d^n) * d^n/dx_norm^n
            if n > 0:
                deriv_nd = deriv_nd / (self.radius[0, d] ** n)

            prod = prod * deriv_nd

        return prod.detach()


class RFMBase(ABC):

    def __init__(self, dim: int,
                 n_hidden: int,
                 domain: Union[Tuple, List, GeometryBase],
                 n_subdomains: Union[int, Tuple, List] = 1,
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
        elif isinstance(n_subdomains, float):
            n_subdomains = [int(n_subdomains)] * self.dim
        elif isinstance(n_subdomains, (list, tuple)) and len(n_subdomains) != self.dim:
            if len(n_subdomains) != self.dim:
                if len(n_subdomains) != 1:
                    raise ValueError(f"n_subdomains must have {self.dim} elements when provided as a list or tuple.")

        # Validate overlap
        if not (0.0 <= overlap < 1.0):
            raise ValueError("Overlap must be between 0 (inclusive) and 1 (exclusive).")

        self.overlap = overlap

        # Compute centers and radii
        if centers is not None and radii is not None:
            if not isinstance(centers, torch.Tensor) or not isinstance(radii, torch.Tensor):
                self.centers = torch.tensor(centers, dtype=self.dtype, device=self.device)
                self.radii = torch.tensor(radii, dtype=self.dtype, device=self.device)
            else:
                self.centers = centers.to(dtype=self.dtype, device=self.device)
                self.radii = radii.to(dtype=self.dtype, device=self.device)
            if self.centers.shape[-1] != self.dim or not (
                    self.radii.shape[-1] == self.dim or self.radii.shape[-1] == 1):
                raise ValueError("Centers and radii must have the same number of dimensions as the domain.")
            elif self.centers.shape[:-1] != self.radii.shape[:-1]:
                raise ValueError("Centers and radii must have the same shape.")
            if self.domain.sdf(self.centers.view(-1, self.centers.shape[-1])).max() > 0:
                logger.warning("Assigned centers are not inside the domain.")
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
        self.A_backup: Optional[torch.tensor] = None
        self.A_norm: Optional[torch.tensor] = None
        self.tau: Optional[torch.tensor] = None

    def __call__(self, x, *args, **kwargs):
        """
        Make the class callable and forward the input tensor.

        :param x: Input tensor.
        :return: Output tensor after forward pass.
        """
        return self.forward(x)

    def add_c_condition(self, num_samples: int, order: int = 1, with_pts=False):
        """
        Add a Continuity (c0 and c1) condition to the model.
        :param num_samples: number of interface points
        :param order: max order of the continuity condition
        :param with_pts: whether to return the interface points
        :return: feature Tensor
        """
        if not isinstance(self.pou_functions[0], PsiA):
            logger.warning("The POU function is not PsiA, the continuity condition may not be Appropriate.")

        if order < 0:
            raise ValueError("Order must be non-negative.")

        n_subdomains = self.submodels.shape
        interface_dict = {}
        if len(n_subdomains) == self.dim:
            n_interface = 0
            for d in range(self.dim):
                n_interface += (n_subdomains[d] - 1) * (prod(n_subdomains) // n_subdomains[d])
            num_samples = max(int(num_samples / n_interface), 3) if self.dim > 1 else max(
                int(num_samples / n_interface), 1)
            for d in range(self.dim):
                if n_subdomains[d] <= 1:
                    continue

                for k in range(n_subdomains[d] - 1):
                    indices1 = [slice(None)] * self.dim
                    indices1[d] = slice(k, k + 1, 1)
                    indices2 = [slice(None)] * self.dim
                    indices2[d] = slice(k + 1, k + 2, 1)

                    centers1 = self.centers[indices1].view(-1, self.dim)
                    radii1 = self.radii[indices1].view(-1, self.dim)
                    centers2 = self.centers[indices2].view(-1, self.dim)
                    radii2 = self.radii[indices2].view(-1, self.dim)

                    indices1 = ravel_multi_index(indices1, n_subdomains)
                    indices2 = ravel_multi_index(indices2, n_subdomains)

                    for (idx1, idx2, center1, radius1, center2, radius2) in zip(indices1, indices2, centers1, radii1,
                                                                                centers2, radii2):

                        if not torch.abs(center1[d] - center2[d]) < (radius1[d] + radius2[d]) * (1 + self.overlap) * (
                                1 + 1e-6):
                            raise ValueError("Subdomains are not adjacent.")

                        interface_center = center1.clone()
                        interface_center[d] = center1[d] + radius1[d]
                        interface_radius = radius1.clone()
                        interface_radius[d] = 0.0
                        if self.dim == 1:
                            interface = Point1D(interface_center[0].item())
                        elif self.dim == 2:
                            interface = Line2D(interface_center[0].item() - interface_radius[0].item(),
                                               interface_center[1].item() - interface_radius[1].item(),
                                               interface_center[0].item() + interface_radius[0].item(),
                                               interface_center[1].item() + interface_radius[1].item())
                        elif self.dim == 3:
                            interface = Square3D(interface_center, interface_radius)
                        else:
                            interface = None
                            raise NotImplementedError("Higher dimension continuity conditions are not supported.")

                        points = interface.in_sample(num_samples, with_boundary=False)
                        points = points[torch.where(self.domain.sdf(points) < 0)[0]]
                        interface_dict[(idx1, idx2)] = points

        else:
            voronoi = Voronoi(self.domain, self.centers)
            interface_dict, _ = voronoi.interface_sample(num_samples)

        n_interface = len(interface_dict)
        n_points = sum([len(points) for points in interface_dict.values()])
        logger.info(f"Number of interface points: {n_points} for {n_interface} interfaces.")

        all_pts = []
        CFeatrues: List[torch.Tensor] = []

        for pair, point in interface_dict.items():
            if order >= 0:
                feature = self.features(point)
                all_pts.append(point)
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
                center1 = self.centers.view(-1, self.centers.shape[-1])[int(pair[1])]
                center0 = self.centers.view(-1, self.centers.shape[-1])[int(pair[0])]
                normal = (center1 - center0) / torch.linalg.norm(center1 - center0)
                dFeatures = [self.features_derivative(point, d) for d in range(self.dim)]
                d_feature = dFeatures[0] * normal[0]
                all_pts.append(point)
                for i in range(1, self.dim):
                    d_feature += dFeatures[i] * normal[i]

                for i in range(d_feature.numel()):
                    if i not in pair:
                        CFeatrues[i] = torch.cat([CFeatrues[i], torch.zeros_like(d_feature[i])], dim=0)
                    else:
                        CFeatrues[i] = torch.cat([CFeatrues[i], d_feature[i] if i == pair[0] else -d_feature[i]], dim=0)
        if order > 1:
            raise NotImplementedError("Higher order continuity conditions are not supported.")

        if with_pts:
            return Tensor(CFeatrues, shape=self.submodels.shape), torch.cat(all_pts, dim=0)
        return Tensor(CFeatrues, shape=self.submodels.shape)

    def empty_cache(self):
        """
        Empty the cache for all submodels.
        """
        for submodel in self.submodels.flat_data:
            submodel.empty_cache()

    def clone(self, deep: bool = True):
        """
        Create a (deep) clone of the current RFMBase instance.
        :param deep: Whether to deep copy tensors and submodules.
        :return: A new RFMBase instance with the same configuration.
        """
        import copy
        new_obj = copy.deepcopy(self) if deep else copy.copy(self)
        return new_obj

    def compute(self, A: torch.Tensor, damp: float = 0.0, use_complex: bool = False, verbose=True):
        """
        Compute the QR decomposition of matrix A.

        :param A: Input matrix.
        :param damp: Damping factor for regularization.
        :param use_complex: Whether to use complex numbers.
        :return: Self.
        """
        if use_complex:
            self.dtype = torch.complex128 if self.dtype == torch.float64 else torch.complex64
        A = A.to(dtype=self.dtype, device=self.device)
        self.A_norm = torch.linalg.norm(A, ord=2, dim=1, keepdim=True)
        # self.A_norm = torch.ones((A.shape[0], 1))
        A /= self.A_norm
        self.A_backup = A.clone().cpu()
        if abs(damp) > 0.0:
            A = torch.cat(
                [A, damp * torch.eye(A.shape[1], dtype=self.dtype, device=self.device)],
                dim=0)
        if verbose:
            print("Decomposing the problem size of A: ", A.shape, "with solver QR")

        try:
            self.A, self.tau = torch.geqrf(A)
        except RuntimeError as e:
            if 'cusolver error' in str(e):
                raise RuntimeError("Out Of Memory Error")
            else:
                raise e

        return self

    def solve(self, b: torch.Tensor, check_condition=False, verbose=True):
        """
        Solve the linear system Ax = b using the QR decomposition.

        :param b: Right-hand side tensor.
        :param check_condition: Whether to check the condition number of A, and switch to SVD if necessary.
        :param complex: Whether to use complex numbers.
        """
        with_damping = None
        b = b.clone().view(-1, 1).to(dtype=self.dtype, device=self.device) if b.dim() == 1 else b.clone().to(
            dtype=self.dtype, device=self.device)
        if self.A.shape[0] != b.shape[0]:
            if b.shape[0] + self.A.shape[1] == self.A.shape[0]:
                with_damping = True
            else:
                raise ValueError("Input dimension mismatch.")
        b /= self.A_norm
        b_backup = b.clone().cpu()
        b = torch.cat([b, torch.zeros((self.A.shape[0] - b.shape[0], 1), dtype=self.dtype, device=self.device)],
                      dim=0) if with_damping else b

        # detect whether A is from a damped system

        try:
            y = torch.ormqr(self.A, self.tau, b, transpose=True)[:self.A.shape[1]]
            # diag = self.A.diagonal()
            # if torch.is_complex(diag):
            #     diag_to_compare = diag.real  # 只用实部来判断正负
            # else:
            #     diag_to_compare = diag
            # diag.add_((diag_to_compare >= 0).float() * torch.finfo(self.dtype).eps)
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
            #
            # # sum up the weights
            # self.W = torch.sum(torch.cat(w_set, dim=1), dim=1, keepdim=True)
            # residual = torch.norm(b_) / torch.norm(b)

            if check_condition and torch.linalg.cond(self.A_backup) > 1.0 / torch.finfo(self.dtype).eps:
                logger.info(
                    f"The condition number {torch.linalg.cond(self.A_backup):.4e} exceeds 1/eps; switching to SVD.")
                self.W = torch.linalg.lstsq(self.A_backup, b.cpu(), driver='gelsd')[0].to(dtype=self.dtype,
                                                                                          device=self.device)
                residual = torch.norm(
                    torch.matmul(self.A_backup.to(dtype=self.dtype, device=self.device), self.W) - b) / torch.norm(b)

        except RuntimeError as e:
            # Add support for minium norm solution
            logger.warning(str(e))
            logger.warning("Switching to torch.linalg.lstsq solver.")
            self.A = self.A_backup.to(dtype=self.dtype, device=self.device)
            b = b_backup.to(dtype=self.dtype, device=self.device)
            self.W = torch.linalg.lstsq(self.A, b,
                                        driver='gels').solution
            residual = torch.norm(torch.matmul(self.A, self.W) - b) / torch.norm(b)

        if verbose:
            print(f"Least Square Relative residual: {residual:.4e}")

        if self.W.numel() % (self.submodels.numel() * self.n_hidden) == 0:
            n_out = int(self.W.numel() / (self.submodels.numel() * self.n_hidden))
            self.W = self.W.view(n_out, -1).T
        else:
            raise ValueError("The output weight mismatch.")

        return self.W

    def forward(self, x: torch.Tensor, batch_size: int = 32768):
        if self.W is None:
            raise ValueError("Weights have not been computed yet.")
        elif isinstance(self.W, Tensor):
            W = self.W.cat(dim=1)
        elif isinstance(self.W, List) and isinstance(self.W[0], torch.Tensor):
            W = torch.cat(self.W, dim=1)
        else:
            W = self.W

        outputs = []
        while True:
            try:
                for i in range(0, x.shape[0], batch_size):
                    x_batch = x[i:i + batch_size]
                    feats = self.features(x_batch).cat(dim=1)
                    out = torch.matmul(feats, W)
                    outputs.append(out)
                return torch.cat(outputs, dim=0)
            except RuntimeError as e:
                if any(keyword in str(e).lower() for keyword in
                       ["out of memory", "can't allocate", "not enough memory", "std::bad_alloc"]) and batch_size > 1:
                    batch_size //= 2
                    outputs = []  # Clear outputs to retry
                    torch.cuda.empty_cache()  # Clear GPU memory
                else:
                    raise e

    def dForward(self, x, order: Union[torch.Tensor, List, Tuple], batch_size: int = 32768):
        """
        Compute the derivative of the forward pass in batches.

        :param x: Input tensor.
        :param order: Order of the derivative, e.g. (1,0,0), (2,0,0), (1,1,0).
        :param batch_size: Max number of points per batch to avoid OOM.
        :return: Derivative tensor of shape (N, out_dim).
        """
        if self.W is None:
            raise ValueError("Weights have not been computed yet.")
        # --- 规范化 multi-index ---
        if not isinstance(order, torch.Tensor):
            order = torch.tensor(order, dtype=self.dtype, device=self.device)
        else:
            order = order.to(device=self.device, dtype=self.dtype)
        order = order.view(1, -1)

        if order.shape[1] != self.dim:
            raise ValueError(f"Order dimension mismatch: got {order.shape[1]}, expected {self.dim}")

        ord_sum = int(order.sum().item())
        if ord_sum == 0:
            return self.forward(x, batch_size=batch_size)

        outputs = []
        while True:
            try:
                if ord_sum == 1:
                    # 只会有一个方向为 1
                    d = int(torch.nonzero(order[0] == 1, as_tuple=False).squeeze(1).item())
                    for i in range(0, x.shape[0], batch_size):
                        x_batch = x[i:i + batch_size]
                        feat = self.features_derivative(x_batch, d).cat(dim=1)
                        outputs.append(torch.matmul(feat, self.W))
                    return torch.cat(outputs, dim=0)

                elif ord_sum == 2:
                    # 可能是 (2,0,0) 或 (1,1,0) 这两类
                    idx2 = torch.nonzero(order[0] == 2, as_tuple=False).squeeze(1).tolist()
                    idx1 = torch.nonzero(order[0] == 1, as_tuple=False).squeeze(1).tolist()

                    if len(idx2) == 1 and len(idx1) == 0:
                        # 纯二阶 ∂^2/∂x_d^2
                        d = idx2[0]
                        for i in range(0, x.shape[0], batch_size):
                            x_batch = x[i:i + batch_size]
                            feat = self.features_second_derivative(x_batch, d, d).cat(dim=1)
                            outputs.append(torch.matmul(feat, self.W))
                        return torch.cat(outputs, dim=0)

                    elif len(idx2) == 0 and len(idx1) == 2:
                        # 混合二阶 ∂^2/(∂x_d1 ∂x_d2)；对称，不需要双算
                        d1, d2 = idx1[0], idx1[1]
                        for i in range(0, x.shape[0], batch_size):
                            x_batch = x[i:i + batch_size]
                            feat = self.features_second_derivative(x_batch, d1, d2).cat(dim=1)
                            outputs.append(torch.matmul(feat, self.W))
                        return torch.cat(outputs, dim=0)

                    else:
                        raise NotImplementedError(
                            f"Unsupported second-order multi-index: {tuple(order.view(-1).tolist())}")

                else:
                    raise NotImplementedError("Higher-order derivatives not supported in batch mode.")

            except RuntimeError as e:
                # OOM 回退：批量减半重试
                msg = str(e).lower()
                if any(k in msg for k in
                       ["out of memory", "can't allocate", "not enough memory", "std::bad_alloc"]) and batch_size > 1:
                    batch_size //= 2
                    outputs.clear()
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    continue
                else:
                    raise

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
        for (submodel, pou_coefficient, pou_axis) in zip(self.submodels.flat_data,
                                                         pou_coefficients.flat_data,
                                                         pou_derivative.flat_data):
            if not use_sparse:
                features_derivative.append(submodel(x) * pou_axis
                                           + submodel.first_derivative(x, axis) * pou_coefficient)
            else:
                features_derivative.append((submodel(x) * pou_axis).to_sparse()
                                           + (submodel.first_derivative(x, axis) * pou_coefficient).to_sparse())
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
                    submodel(x) * pou_second +
                    submodel.second_derivative(x, axis1, axis2) * pou_coefficient +
                    submodel.first_derivative(x, axis1) * pou_first_axis2 +
                    submodel.first_derivative(x, axis2) * pou_first_axis1
                )
            else:
                features_second_derivative.append(
                    (submodel(x) * pou_second).to_sparse() +
                    (submodel.second_derivative(x, axis1, axis2) * pou_coefficient).to_sparse() +
                    (submodel.first_derivative(x, axis1) * pou_first_axis2).to_sparse() +
                    (submodel.first_derivative(x, axis2) * pou_first_axis1).to_sparse()
                )
        return Tensor(features_second_derivative, shape=self.submodels.shape)

    def features_high_order_derivative(self, x: torch.Tensor, order: Union[torch.Tensor, List],
                                       use_sparse: bool = False) -> Tensor:
        """
        Compute the high-order feature derivative for the given input.

        :param x: Input tensor.
        :param order: Multi-index order as list/tensor, e.g. [2,1] means d^3/dx0^2 dx1.
        :param use_sparse: Whether to use sparse tensors.
        :return: High-order feature derivative Tensor.
        """
        order = list(order) if not isinstance(order, list) else order
        assert len(order) == self.dim, "order length must match input dimension."

        from itertools import product as iproduct
        from math import comb
        from functools import reduce

        alpha = tuple(order)

        def iter_sub_multiindices(alpha):
            ranges = [range(int(a) + 1) for a in alpha]
            for beta in iproduct(*ranges):
                yield beta

        def multinomial_coeff(alpha, beta):
            return reduce(lambda acc, ab: acc * comb(int(ab[0]), int(ab[1])), zip(alpha, beta), 1)

        def call_submodel_derivative(submodel, x, beta):
            """Call the appropriate derivative method on submodel given multi-index beta."""
            total = sum(beta)
            if total == 0:
                return submodel(x)
            elif total == 1:
                axis = beta.index(1)
                return submodel.first_derivative(x, axis)
            elif total == 2:
                axes = [i for i, b in enumerate(beta) for _ in range(b)]
                return submodel.second_derivative(x, axes[0], axes[1])
            else:
                return submodel.higher_order_derivative(x, list(beta))

        # Pre-compute all needed pou derivatives D^beta(phi_i) for beta <= alpha
        pou_derivative_cache = {}  # beta -> Tensor of pou derivatives
        zero = tuple([0] * self.dim)

        for beta in iter_sub_multiindices(alpha):
            total = sum(beta)
            if total == 0:
                pou_derivative_cache[beta] = self.pou_coefficients(x)
            elif total == 1:
                axis = list(beta).index(1)
                pou_derivative_cache[beta] = self.pou_derivative(x, axis)
            elif total == 2:
                axes = [i for i, b in enumerate(beta) for _ in range(b)]
                pou_derivative_cache[beta] = self.pou_second_derivative(x, axes[0], axes[1])
            else:
                pou_derivative_cache[beta] = self.pou_higher_order_derivative(x, list(beta))

        # Apply Leibniz rule: D^alpha(f_i * phi_i) = sum_{beta<=alpha} C(alpha,beta) * D^beta(f_i) * D^{alpha-beta}(phi_i)
        features_hod = []

        for i, submodel in enumerate(self.submodels.flat_data):
            result = torch.zeros(x.shape[0], submodel.output_dim if hasattr(submodel, 'output_dim') else 1,
                                 dtype=self.dtype, device=self.device)

            for beta in iter_sub_multiindices(alpha):
                coeff = multinomial_coeff(alpha, beta)
                if coeff == 0:
                    continue

                beta_minus = tuple(a - b for a, b in zip(alpha, beta))

                # D^beta(f_i)
                df = call_submodel_derivative(submodel, x, list(beta))

                # D^{alpha-beta}(phi_i)
                dphi = pou_derivative_cache[beta_minus].flat_data[i]

                result = result + coeff * df * dphi

            if use_sparse:
                features_hod.append(result.to_sparse())
            else:
                features_hod.append(result)

        return Tensor(features_hod, shape=self.submodels.shape)

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
        # print(torch.cat([x, c[0], c_sum], dim=1))

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
        :param order: Order of the derivative as a multi-index list/tensor, e.g. [2,1,0] means
                      d^3/dx0^2 dx1^1 for a 3D input.
        :return: POU higher-order derivative Tensor.
        """
        if x.shape[1] != self.dim:
            raise ValueError("Input dimension mismatch.")

        order = list(order) if not isinstance(order, list) else order
        assert len(order) == self.dim, "order length must match input dimension."

        # Build list of (axis, repeat) derivative calls from multi-index
        # e.g. order=[2,1] -> axes_sequence = [0,0,1]
        axes_sequence = []
        for axis, cnt in enumerate(order):
            axes_sequence.extend([axis] * int(cnt))
        total_order = len(axes_sequence)

        if total_order == 0:
            return self.pou_coefficients(x)

        # ---------------------------------------------------------------
        # We use the recursive formula based on the quotient rule:
        #   D^alpha(psi_i / S) = (1/S) * [D^alpha(psi_i) - sum_{0<beta<=alpha} C(alpha,beta)*D^beta(S)*D^{alpha-beta}(phi_i)]
        #
        # Strategy: iterate over all sub-multiindices of `order` in
        # increasing total order, caching D^beta(phi_i) and D^beta(S).
        # ---------------------------------------------------------------

        from itertools import product as iproduct
        from math import comb, prod
        from functools import reduce

        def multiindex_leq(a, b):
            return all(ai <= bi for ai, bi in zip(a, b))

        def iter_sub_multiindices(alpha):
            ranges = [range(int(a) + 1) for a in alpha]
            for beta in iproduct(*ranges):
                yield beta

        def multinomial_coeff(alpha, beta):
            return reduce(lambda acc, ab: acc * comb(int(ab[0]), int(ab[1])), zip(alpha, beta), 1)

        alpha = tuple(order)

        # Cache D^beta(S) for all sub-multiindices 0 < beta <= alpha
        # Cache D^beta(phi_i) for 0 <= beta < alpha (strict)
        n = x.shape[0]
        zero = tuple([0] * self.dim)

        # First, gather raw psi_i and their mixed derivatives D^beta(psi_i)
        # We need D^beta(psi_i) for all beta <= alpha via pou_function.higher_order_derivative
        psi_cache = {}  # beta -> sum over i (i.e. D^beta S), and per-function list
        psi_per_func = {}  # beta -> list of D^beta(psi_i)

        all_betas = list(iter_sub_multiindices(alpha))

        for beta in all_betas:
            vals = []
            s = torch.zeros(n, 1, dtype=self.dtype, device=self.device)
            for pou_function in self.pou_functions.flat_data:
                total = sum(beta)
                if total == 0:
                    v = pou_function(x)
                elif total == 1:
                    axis = beta.index(1)
                    v = pou_function.first_derivative(x, axis)
                elif total == 2:
                    ones = [i for i, b in enumerate(beta) for _ in range(b)]
                    v = pou_function.second_derivative(x, ones[0], ones[1])
                else:
                    # Assume pou_function has higher_order_derivative(x, order_list)
                    v = pou_function.higher_order_derivative(x, list(beta))
                vals.append(v)
                s += v
            psi_per_func[beta] = vals
            psi_cache[beta] = s  # D^beta(S)

        S = psi_cache[zero]  # S(x), shape (n,1)

        # Recursively compute D^beta(phi_i) for increasing |beta|
        # phi_cache[beta] -> list of D^beta(phi_i) for each i
        phi_cache = {}

        # Sort all_betas by total order
        all_betas_sorted = sorted(all_betas, key=lambda b: sum(b))

        for beta in all_betas_sorted:
            phi_vals = []
            total_beta = sum(beta)

            for i, _ in enumerate(self.pou_functions.flat_data):
                # Compute sum over 0 < gamma <= beta of C(beta,gamma)*D^gamma(S)*D^{beta-gamma}(phi_i)
                correction = torch.zeros(n, 1, dtype=self.dtype, device=self.device)
                for gamma in iter_sub_multiindices(beta):
                    if gamma == beta:
                        continue  # 只跳过 gamma=beta，不跳过 gamma=0
                    beta_minus_gamma = tuple(b - g for b, g in zip(beta, gamma))
                    coeff = multinomial_coeff(beta, gamma)
                    d_alpha_minus_gamma_S = psi_cache[beta_minus_gamma]  # 修正：用 D^{beta-gamma}(S)
                    d_gamma_phi = phi_cache[gamma][i]  # 修正：用 D^gamma(phi_i)
                    correction += coeff * d_gamma_phi * d_alpha_minus_gamma_S

                phi_i_beta = (psi_per_func[beta][i] - correction) / S
                phi_vals.append(phi_i_beta)

            phi_cache[beta] = phi_vals

        result = phi_cache[alpha]
        return Tensor(result, shape=self.submodels.shape)


class STRFMBase(ABC):
    def __init__(self, dim: int,
                 n_hidden: int,
                 domain: Union[Tuple, List, GeometryBase],
                 time_interval: Union[Tuple[float, float], List[float]],
                 n_spatial_subdomains: Union[int, Tuple, List] = 1,
                 n_temporal_subdomains: int = 1,
                 st_type: str = "STC",
                 overlap: torch.float64 = 0.0,
                 space_rf=RFTanH,
                 time_rf=RFTanH,
                 pou=PsiB,
                 centers: Optional[torch.Tensor] = None,
                 radii: Optional[torch.Tensor] = None,
                 seed: int = 100,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        """
        Initialize the RFMBase class with arbitrary dimensions.

        :param dim: Number of spatial dimensions.
        :param domain: List or tuple of min and max values for each dimension.
                       Example for 2D: [x_min, x_max, y_min, y_max]
                       Example for 3D: [x_min, x_max, y_min, y_max, z_min, z_max]
        :param time_interval: List or tuple of min and max values for time.
        :param n_spatial_subdomains: Either an integer (uniform subdivisions in all dimensions)
                             or a list/tuple specifying the subdivisions per dimension.
        :param n_temporal_subdomains: Number of time subdomains.
        :param st_type: Define the construction method of space-time random feature functions, either "STC" (Space-Time Concatenation) or "SoV" (Separation of Variables).
        :param overlap: Overlap between subdomains, must be between 0 (inclusive) and 1 (exclusive).
        :param space_rf: Random Feature class for spatial part, must be a subclass of RFBase.
        :param time_rf: Random Feature class for temporal part, must be a subclass of RFBase.
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

        if isinstance(time_interval, (list, tuple)) and len(time_interval) != 2:
            raise ValueError("Time interval must contain two values (start_time and end_time).")
        self.time_interval = (float(time_interval[0]), float(time_interval[1]))

        # If n_spatial_subdomains is an integer, create uniform subdivisions
        if isinstance(n_spatial_subdomains, int):
            n_spatial_subdomains = [n_spatial_subdomains] * self.dim
        elif isinstance(n_spatial_subdomains, float):
            n_spatial_subdomains = [int(n_spatial_subdomains)] * self.dim
        elif isinstance(n_spatial_subdomains, (list, tuple)):
            if len(n_spatial_subdomains) != self.dim:
                if len(n_spatial_subdomains) != 1:
                    raise ValueError(
                        f"n_spatial_subdomains must have {self.dim} elements when provided as a list or tuple.")

        # Validate overlap
        if not (0.0 <= overlap < 1.0):
            raise ValueError("Overlap must be between 0 (inclusive) and 1 (exclusive).")

        self.overlap = overlap

        # Compute centers and radii
        if centers is not None and radii is not None:
            self.centers = torch.tensor(centers, dtype=self.dtype, device=self.device)
            self.radii = torch.tensor(radii, dtype=self.dtype, device=self.device)
            if self.centers.shape[-1] != self.dim or not (
                    self.radii.shape[-1] == self.dim or self.radii.shape[-1] != 1):
                raise ValueError("Centers and radii must have the same number of dimensions as the domain.")
            elif self.centers.shape[:-1] != self.radii.shape[:-1]:
                raise ValueError("Centers and radii must have the same shape.")
            if self.domain.sdf(self.centers.view(-1, self.centers.shape[-1])).max() > 0:
                logger.warning("Assigned centers are not inside the domain.")
        else:
            self.centers, self.radii = self._compute_centers_and_radii(n_spatial_subdomains)

        if not issubclass(space_rf, RFBase) or not issubclass(time_rf, RFBase):
            raise ValueError("Random Feature must be a subclass of RFBase.")

        submodels = []
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(seed)

        if not isinstance(st_type, str) or st_type.upper() not in ["STC", "SOV"]:
            raise ValueError("st_type must be either 'STC' or 'SoV'.")

        self.st_type = st_type.upper()

        submodels = []
        for center, radius in zip(self.centers.view(-1, self.centers.shape[-1]),
                                  self.radii.view(-1, self.radii.shape[-1])):
            time_stamp = torch.linspace(*time_interval, n_temporal_subdomains + 1)
            for (t0, t1) in zip(time_stamp[:-1], time_stamp[1:]):
                if self.st_type == "STC":
                    center_ = torch.cat([center, torch.tensor([(t1 + t0) / 2.0])], dim=0)
                    radius_ = torch.cat([radius, torch.tensor([(t1 - t0) / 2.0])], dim=0)
                    # print(f"center = {center_} and radius = {radius_}")
                    submodels.append(space_rf(dim + 1, center_, radius_, n_hidden, gen=self.gen, dtype=self.dtype,
                                              device=self.device))
                elif self.st_type == "SOV":
                    submodels.append(
                        (space_rf(dim, center, radius, n_hidden, gen=self.gen, dtype=dtype, device=self.device),
                         time_rf(1, (t0 + t1) / 2.0, (t1 - t0) / 2.0, n_hidden, gen=self.gen, dtype=self.dtype,
                                 device=self.device)))
                else:
                    raise ValueError("st_type must be either 'STC' or 'SoV'.")

        self.submodels = Tensor(submodels, shape=n_spatial_subdomains.append(n_temporal_subdomains) if isinstance(
            n_spatial_subdomains, list) else n_spatial_subdomains * n_temporal_subdomains)
        self.n_hidden = n_hidden

        if not issubclass(pou, POUBase):
            raise ValueError("Partition of Unity must be a subclass of POUBase.")
        pou_functions = []
        for center, radius in zip(self.centers.view(-1, self.centers.shape[-1]),
                                  self.radii.view(-1, self.radii.shape[-1])):
            time_stamp = torch.linspace(*time_interval, n_temporal_subdomains + 1)
            for (t0, t1) in zip(time_stamp[:-1], time_stamp[1:]):
                center_ = torch.cat([center, torch.tensor([(t1 + t0) / 2.0])], dim=0)
                radius_ = torch.cat([radius, torch.tensor([(t1 - t0) / 2.0])], dim=0)
                pou_functions.append(pou(center_, radius_, dtype=dtype, device=device))
        self.pou_functions = Tensor(pou_functions,
                                    shape=n_spatial_subdomains.append(n_temporal_subdomains) if isinstance(
                                        n_spatial_subdomains, list) else n_spatial_subdomains * n_temporal_subdomains)

        self.W: Union[Tensor, List, torch.tensor] = None
        self.A: Optional[torch.tensor] = None
        self.A_backup: Optional[torch.tensor] = None
        self.A_norm: Optional[torch.tensor] = None
        self.tau: Optional[torch.tensor] = None

    def __call__(self, x, *args, **kwargs):
        """
        Make the class callable and forward the input tensor.

        :param x: Input tensor.
        :return: Output tensor after forward pass.
        """
        return self.forward(x)

    def clone(self, deep: bool = True):
        """
        Create a (deep) clone of the current STRFMBase instance.
        :param deep: Whether to deep copy tensors and submodules.
        :return: A new STRFMBase instance with the same configuration.
        """
        import copy
        return copy.deepcopy(self) if deep else copy.copy(self)

    def compute(self, A: torch.Tensor):
        """
        Compute the QR decomposition of matrix A.

        :param A: Input matrix.
        :param complex: Whether to use complex numbers.
        :return: Self.
        """
        A = A.to(dtype=self.dtype, device=self.device)
        self.A_norm = torch.linalg.norm(A, ord=2, dim=1, keepdim=True)
        A /= self.A_norm
        self.A_backup = A.clone().cpu()
        print("Decomposing the problem size of A: ", A.shape, "with solver QR")

        try:
            self.A, self.tau = torch.geqrf(A)
        except RuntimeError as e:
            if 'cusolver error' in str(e):
                raise RuntimeError("Out Of Memory Error")
            else:
                raise e

        return self

    def solve(self, b: torch.Tensor, check_condition=False):
        """
        Solve the linear system Ax = b using the QR decomposition.

        :param b: Right-hand side tensor.
        :param check_condition: Whether to check the condition number of A, and switch to SVD if necessary.
        :param complex: Whether to use complex numbers.
        """
        b = b.view(-1, 1).to(dtype=self.dtype, device=self.device)
        if self.A.shape[0] != b.shape[0]:
            raise ValueError("Input dimension mismatch.")
        b /= self.A_norm

        try:
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
            #
            # # sum up the weights
            # self.W = torch.sum(torch.cat(w_set, dim=1), dim=1, keepdim=True)
            # residual = torch.norm(b_) / torch.norm(b)

            if check_condition and torch.linalg.cond(self.A_backup) > 1.0 / torch.finfo(self.dtype).eps:
                logger.info(f"The condition number exceeds 1/eps; switching to SVD.")
                self.W = torch.linalg.lstsq(self.A_backup, b.cpu(), driver='gelsd')[0].to(dtype=self.dtype,
                                                                                          device=self.device)
                residual = torch.norm(
                    torch.matmul(self.A_backup.to(dtype=self.dtype, device=self.device), self.W) - b) / torch.norm(b)

        except RuntimeError as e:
            # Add support for minium norm solution
            self.A = self.A_backup.to(dtype=self.dtype, device=self.device)
            self.W = torch.linalg.lstsq(self.A, b, driver='gels').solution
            residual = torch.norm(torch.matmul(self.A, self.W) - b) / torch.norm(b)

        print(f"Least Square Relative residual: {residual:.4e}")

        if self.W.numel() % (self.submodels.numel() * self.n_hidden) == 0:
            n_out = int(self.W.numel() / (self.submodels.numel() * self.n_hidden))
            self.W = self.W.view(n_out, -1).T
        else:
            raise ValueError("The output weight mismatch.")

    def _compute_centers_and_radii(self, n_spatial_subdomains: Union[int, Tuple, List]):
        """
        Compute the centers and radii for subdomains.

        :param n_spatial_subdomains: Either an integer (uniform subdivisions in all dimensions)
                             or a list/tuple specifying the subdivisions per dimension.
        :return: Tuple of centers and radii as tensors.
        """
        centers_list = []
        radii_list = []
        bounding_box = self.domain.get_bounding_box()

        for i in range(self.dim):
            sub_min, sub_max = (bounding_box[2 * i], bounding_box[2 * i + 1])
            n_divisions = n_spatial_subdomains[i]

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

    def forward(self, x: torch.Tensor = None, t: torch.Tensor = None, xt: torch.Tensor = None) -> torch.Tensor:
        xt = self.validate_and_prepare_xt(x, t, xt)
        if self.W is None:
            raise ValueError("Weights have not been computed yet.")
        elif isinstance(self.W, Tensor):
            self.W = self.W.cat(dim=1)
        elif isinstance(self.W, List) and isinstance(self.W[0], torch.Tensor):
            self.W = torch.cat(self.W, dim=1)

        return torch.matmul(self.features(xt=xt).cat(dim=1), self.W)

    def dForward(
            self,
            x: torch.Tensor = None,
            t: torch.Tensor = None,
            xt: torch.Tensor = None,
            order: Union[torch.Tensor, List] = None,
            batch_size: int = 32768,
    ):
        """
        Compute the derivative of the forward pass in batches.

        :param x: Spatial input tensor.
        :param t: Temporal input tensor.
        :param xt: Combined input tensor.
        :param order: Derivative order, e.g. (1,0,0,0), (0,0,0,2), (1,0,0,1).
                      Length must be dim+1 (space dims + time).
        :param batch_size: Max batch size to avoid OOM.
        :return: Derivative tensor of shape (N, 1) (following self.W.view(-1,1)).
        """
        xt = self.validate_and_prepare_xt(x, t, xt)

        # --- 规范化 multi-index ---
        if not isinstance(order, torch.Tensor):
            order = torch.tensor(order, dtype=self.dtype, device=self.device)
        else:
            order = order.to(device=self.device, dtype=self.dtype)
        order = order.view(1, -1)

        n_axes = self.dim + 1
        if order.shape[1] != n_axes:
            raise ValueError(f"Order dimension mismatch: got {order.shape[1]}, expected {n_axes}")

        ord_sum = int(order.sum().item())
        if ord_sum == 0:
            return self.forward(xt)

        outputs = []
        while True:
            try:
                if ord_sum == 1:
                    # 仅一个轴为 1（可能是空间或时间轴）
                    d = int(torch.nonzero(order[0] == 1, as_tuple=False).squeeze(1).item())
                    for i in range(0, xt.shape[0], batch_size):
                        xt_batch = xt[i:i + batch_size]
                        feat = self.features_derivative(xt=xt_batch, axis=d).cat(dim=1)
                        outputs.append(torch.matmul(feat, self.W))
                    return torch.cat(outputs, dim=0)

                elif ord_sum == 2:
                    # 可能是纯二阶 (axis with 2) 或混合二阶 (two axes with 1)
                    idx2 = torch.nonzero(order[0] == 2, as_tuple=False).squeeze(1).tolist()
                    idx1 = torch.nonzero(order[0] == 1, as_tuple=False).squeeze(1).tolist()

                    if len(idx2) == 1 and len(idx1) == 0:
                        # 纯二阶 ∂^2 / ∂axis^2
                        d = idx2[0]
                        for i in range(0, xt.shape[0], batch_size):
                            xt_batch = xt[i:i + batch_size]
                            feat = self.features_second_derivative(xt=xt_batch, axis1=d, axis2=d).cat(dim=1)
                            outputs.append(torch.matmul(feat, self.W))
                        return torch.cat(outputs, dim=0)

                    elif len(idx2) == 0 and len(idx1) == 2:
                        # 混合二阶 ∂^2 / (∂axis1 ∂axis2)
                        d1, d2 = idx1[0], idx1[1]
                        for i in range(0, xt.shape[0], batch_size):
                            xt_batch = xt[i:i + batch_size]
                            feat = self.features_second_derivative(xt=xt_batch, axis1=d1, axis2=d2).cat(dim=1)
                            outputs.append(torch.matmul(feat, self.W))
                        return torch.cat(outputs, dim=0)

                    else:
                        raise NotImplementedError(
                            f"Unsupported second-order multi-index: {tuple(order.view(-1).tolist())}")

                else:
                    raise NotImplementedError("Higher-order derivatives not supported in batch mode.")

            except RuntimeError as e:
                # OOM 回退：批量减半重试
                msg = str(e).lower()
                if any(k in msg for k in
                       ["out of memory", "can't allocate", "not enough memory", "std::bad_alloc"]) and batch_size > 1:
                    batch_size //= 2
                    outputs.clear()
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    continue
                else:
                    raise

    def features(self, x: torch.Tensor = None, t: torch.Tensor = None, xt: torch.Tensor = None,
                 use_sparse: bool = False) -> Tensor[torch.Tensor]:
        xt = self.validate_and_prepare_xt(x, t, xt)

        features = []
        pou_coefficients = self.pou_coefficients(xt=xt)
        for (submodel, pou_coefficient) in zip(self.submodels.flat_data, pou_coefficients.flat_data):
            if self.st_type == "STC":
                if not use_sparse:
                    features.append(submodel(xt) * pou_coefficient)
                else:
                    features.append((submodel(xt) * pou_coefficient).to_sparse())
            elif self.st_type == "SOV":
                x_submodel, t_submodel = submodel
                if not use_sparse:
                    features.append(x_submodel(xt[:, :-1]) * t_submodel(xt[:, [-1]]) * pou_coefficient)
                else:
                    features.append((x_submodel(xt[:, :-1]) * t_submodel(xt[:, [-1]]) * pou_coefficient).to_sparse())

        return Tensor(features, shape=self.submodels.shape)

    def features_derivative(self, x: torch.Tensor = None,
                            t: torch.Tensor = None,
                            xt: torch.Tensor = None,
                            axis: int = 0,
                            use_sparse: bool = False) -> Tensor[
        torch.Tensor]:
        xt = self.validate_and_prepare_xt(x, t, xt)
        axis = self.dim + axis if axis < 0 else axis
        features_derivative = []
        pou_coefficients = self.pou_coefficients(xt=xt)
        pou_derivative = self.pou_derivative(xt=xt, axis=axis)

        for (submodel, pou_coefficient, pou_axis) in zip(self.submodels.flat_data, pou_coefficients.flat_data,
                                                         pou_derivative.flat_data):
            if self.st_type == "STC":
                if not use_sparse:
                    features_derivative.append(submodel(xt) * pou_axis
                                               + submodel.first_derivative(xt, axis) * pou_coefficient)
                else:
                    features_derivative.append((submodel(xt) * pou_axis).to_sparse()
                                               + (submodel.first_derivative(xt, axis) * pou_coefficient).to_sparse())

            elif self.st_type == "SOV":
                if not use_sparse:
                    x_submodel, t_submodel = submodel
                    if axis < self.dim:
                        features_derivative.append(
                            x_submodel(xt[:, :-1]) * t_submodel(xt[:, [-1]]) * pou_axis
                            + x_submodel.first_derivative(xt[:, :-1], axis) * t_submodel(xt[:, [-1]]) * pou_coefficient
                        )
                    else:
                        features_derivative.append(
                            x_submodel(xt[:, :-1]) * t_submodel(xt[:, [-1]]) * pou_axis
                            + x_submodel(xt[:, :-1]) * t_submodel.first_derivative(xt[:, [-1]], 0) * pou_coefficient
                        )
                else:
                    x_submodel, t_submodel = submodel
                    if axis < self.dim:
                        features_derivative.append(
                            (x_submodel(xt[:, :-1]) * t_submodel(xt[:, [-1]]) * pou_axis).to_sparse()
                            + (x_submodel.first_derivative(xt[:, :-1], axis) * t_submodel(
                                xt[:, [-1]]) * pou_coefficient).to_sparse()
                        )
                    else:
                        features_derivative.append(
                            (x_submodel(xt[:, :-1]) * t_submodel(xt[:, [-1]]) * pou_axis).to_sparse()
                            + (x_submodel(xt[:, :-1]) * t_submodel.first_derivative(
                                xt[:, [-1], 0]) * pou_coefficient).to_sparse()
                        )

        return Tensor(features_derivative, shape=self.submodels.shape)

    def features_second_derivative(self, x: torch.Tensor = None,
                                   t: torch.Tensor = None,
                                   xt: torch.Tensor = None,
                                   axis1: int = 0,
                                   axis2: int = 0,
                                   use_sparse: bool = False) -> Tensor[torch.Tensor]:
        xt = self.validate_and_prepare_xt(x, t, xt)
        if axis1 < 0:
            axis1 = self.dim + axis1
        if axis2 < 0:
            axis2 = self.dim + axis2
        if axis1 > axis2:
            axis1, axis2 = axis2, axis1

        features_second_derivative = []
        pou_coefficients = self.pou_coefficients(xt=xt)
        pou_first_derivative_axis1 = self.pou_derivative(xt=xt, axis=axis1)
        pou_first_derivative_axis2 = self.pou_derivative(xt=xt, axis=axis2)
        pou_second_derivative = self.pou_second_derivative(xt=xt, axis1=axis1, axis2=axis2)

        for (submodel, pou_coefficient, pou_first_axis1, pou_first_axis2, pou_second) \
                in zip(self.submodels.flat_data,
                       pou_coefficients.flat_data,
                       pou_first_derivative_axis1.flat_data,
                       pou_first_derivative_axis2.flat_data,
                       pou_second_derivative.flat_data):
            if self.st_type == "STC":
                if not use_sparse:
                    features_second_derivative.append(
                        submodel(xt) * pou_second +
                        submodel.second_derivative(xt, axis1, axis2) * pou_coefficient +
                        submodel.first_derivative(xt, axis1) * pou_first_axis2 +
                        submodel.first_derivative(xt, axis2) * pou_first_axis1
                    )
                else:
                    features_second_derivative.append(
                        (submodel(xt) * pou_second).to_sparse() +
                        (submodel.second_derivative(xt, axis1, axis2) * pou_coefficient).to_sparse() +
                        (submodel.first_derivative(xt, axis1) * pou_first_axis2).to_sparse() +
                        (submodel.first_derivative(xt, axis2) * pou_first_axis1).to_sparse()
                    )
            elif self.st_type == "SOV":
                x_submodel, t_submodel = submodel
                if axis2 < self.dim:
                    if not use_sparse:
                        features_second_derivative.append(
                            x_submodel(xt[:, :-1]) * t_submodel(xt[:, [-1]]) * pou_second +
                            x_submodel.second_derivative(xt[:, :-1], axis1, axis2) * t_submodel(
                                xt[:, [-1]]) * pou_coefficient +
                            x_submodel.first_derivative(xt[:, :-1], axis1) * t_submodel(xt[:, [-1]]) * pou_first_axis2 +
                            x_submodel.first_derivative(xt[:, :-1], axis2) * t_submodel(xt[:, [-1]]) * pou_first_axis1
                        )
                    else:
                        features_second_derivative.append(
                            (x_submodel(xt[:, :-1]) * t_submodel(xt[:, [-1]]) * pou_second).to_sparse() +
                            (x_submodel.second_derivative(xt[:, :-1], axis1, axis2) * t_submodel(
                                xt[:, [-1]]) * pou_coefficient).to_sparse() +
                            (x_submodel.first_derivative(xt[:, :-1], axis1) * t_submodel(
                                xt[:, [-1]]) * pou_first_axis2).to_sparse() +
                            (x_submodel.first_derivative(xt[:, :-1], axis2) * t_submodel(
                                xt[:, [-1]]) * pou_first_axis1).to_sparse()
                        )
                elif axis2 == self.dim:
                    axis2 = 0
                    if axis1 == self.dim:
                        axis1 = 0
                        if not use_sparse:
                            features_second_derivative.append(
                                x_submodel(xt[:, :-1]) * t_submodel(xt[:, [-1]]) * pou_second +
                                x_submodel(xt[:, :-1]) * t_submodel.second_derivative(xt[:, [-1]], axis1,
                                                                                      axis2) * pou_coefficient +
                                x_submodel(xt[:, :-1]) * t_submodel.first_derivative(xt[:, [-1]],
                                                                                     axis1) * pou_first_axis2 +
                                x_submodel(xt[:, :-1]) * t_submodel.first_derivative(xt[:, [-1]],
                                                                                     axis2) * pou_first_axis1
                            )
                        else:
                            features_second_derivative.append(
                                (x_submodel(xt[:, :-1]) * t_submodel(xt[:, [-1]]) * pou_second).to_sparse() +
                                (x_submodel(xt[:, :-1]) * t_submodel.second_derivative(xt[:, [-1]], axis1,
                                                                                       axis2) * pou_coefficient).to_sparse() +
                                (x_submodel(xt[:, :-1]) * t_submodel.first_derivative(xt[:, [-1]],
                                                                                      axis1) * pou_first_axis2).to_sparse() +
                                (x_submodel(xt[:, :-1]) * t_submodel.first_derivative(xt[:, [-1]],
                                                                                      axis2) * pou_first_axis1).to_sparse()
                            )

                    else:
                        if not use_sparse:
                            features_second_derivative.append(
                                x_submodel(xt[:, :-1]) * t_submodel(xt[:, [-1]]) * pou_second +
                                x_submodel.first_derivative(xt[:, :-1], axis1) * t_submodel.first_derivative(
                                    xt[:, [-1]], axis2) * pou_coefficient +
                                x_submodel.first_derivative(xt[:, :-1], axis1) * t_submodel(
                                    xt[:, [-1]]) * pou_first_axis2 +
                                x_submodel(xt[:, :-1]) * t_submodel.first_derivative(xt[:, [-1]],
                                                                                     axis2) * pou_first_axis1
                            )
                        else:
                            features_second_derivative.append(
                                (x_submodel(xt[:, :-1]) * t_submodel(xt[:, [-1]]) * pou_second).to_sparse() +
                                (x_submodel.first_derivative(xt[:, :-1], axis1) * t_submodel.first_derivative(
                                    xt[:, [-1]], axis2) * pou_coefficient).to_sparse() +
                                (x_submodel.first_derivative(xt[:, :-1], axis1) * t_submodel(
                                    xt[:, [-1]]) * pou_first_axis2).to_sparse() +
                                (x_submodel(xt[:, :-1]) * t_submodel.first_derivative(xt[:, [-1]],
                                                                                      axis2) * pou_first_axis1).to_sparse()
                            )

                else:
                    raise ValueError("axis out of range")

        return Tensor(features_second_derivative, shape=self.submodels.shape)

    def pou_coefficients(self, x: torch.Tensor = None,
                         t: torch.Tensor = None,
                         xt: torch.Tensor = None) -> Tensor[torch.Tensor]:
        xt = self.validate_and_prepare_xt(x, t, xt)
        c = []
        c_sum = torch.zeros(xt.shape[0], 1, dtype=self.dtype, device=self.device)
        for (i, pou_function) in enumerate(self.pou_functions.flat_data):
            c_i = pou_function(xt)
            c.append(c_i)
            c_sum += c_i
        c = [c_i / c_sum for c_i in c]
        # print(torch.cat([x, c[0], c_sum], dim=1))

        return Tensor(c, shape=self.submodels.shape)

    def pou_derivative(self, x: torch.Tensor = None,
                       t: torch.Tensor = None,
                       xt: torch.Tensor = None,
                       axis: int = 0) -> Tensor[
        torch.Tensor]:
        xt = self.validate_and_prepare_xt(x, t, xt)
        c = []
        c_sum = torch.zeros(xt.shape[0], 1, dtype=self.dtype, device=self.device)
        dc_sum = torch.zeros(xt.shape[0], 1, dtype=self.dtype, device=self.device)

        for (i, pou_function) in enumerate(self.pou_functions.flat_data):
            c_i = pou_function(xt)
            dc_i = pou_function.first_derivative(xt, axis)
            c.append((c_i, dc_i))
            c_sum += c_i
            dc_sum += dc_i
        c = [(dc_i - c_i * dc_sum / c_sum) / c_sum for c_i, dc_i in c]
        return Tensor(c, shape=self.submodels.shape)

    def pou_second_derivative(self, x: torch.Tensor = None,
                              t: torch.Tensor = None,
                              xt: torch.Tensor = None,
                              axis1: int = 0,
                              axis2: int = 0) -> Tensor[torch.Tensor]:
        xt = self.validate_and_prepare_xt(x, t, xt)
        c = []
        c_sum = torch.zeros(xt.shape[0], 1, dtype=self.dtype, device=self.device)
        dc_sum_axis1 = torch.zeros(xt.shape[0], 1, dtype=self.dtype, device=self.device)
        dc_sum_axis2 = torch.zeros(xt.shape[0], 1, dtype=self.dtype, device=self.device)
        d2c_sum = torch.zeros(xt.shape[0], 1, dtype=self.dtype, device=self.device)

        # Compute raw values, first derivatives, and second derivatives
        for pou_function in self.pou_functions.flat_data:
            c_i = pou_function(xt)
            dc_i_axis1 = pou_function.first_derivative(xt, axis1)
            dc_i_axis2 = pou_function.first_derivative(xt, axis2)
            d2c_i = pou_function.second_derivative(xt, axis1, axis2)

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

    def validate_and_prepare_xt(self, x: torch.Tensor = None,
                                t: torch.Tensor = None,
                                xt: torch.Tensor = None) -> torch.Tensor:
        """
        Validate and prepare the combined tensor xt from x and t if not provided.

        Args:
            x (torch.Tensor, optional): Spatial input tensor.
            t (torch.Tensor, optional): Temporal input tensor.
            xt (torch.Tensor, optional): Combined space-time input tensor.

        Returns:
            torch.Tensor: Combined space-time input tensor.

        Raises:
            ValueError: If input dimensions do not match or if neither x and t nor xt are provided.
        """
        if xt is not None:
            if xt.shape[1] != self.dim + 1:
                raise ValueError("Input dimension mismatch")
            return xt
        elif x is not None and t is not None:
            if x.shape[1] != self.dim or t.shape[1] != 1:
                raise ValueError("Input dimension mismatch.")
            x, t = x.view(-1, self.dim), t.view(-1, 1)
            xt = torch.cat([x.unsqueeze(1).expand(-1, t.shape[0], -1).reshape(-1, self.dim),
                            t.unsqueeze(0).expand(x.shape[0], -1, -1).reshape(-1, 1)], dim=1)
        else:
            raise ValueError("Either x and t or xt must be provided.")
        return xt
