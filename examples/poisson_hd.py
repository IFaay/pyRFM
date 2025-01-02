# -*- coding: utf-8 -*-
"""
Created on 2025/1/2

@author: Yifei Sun
"""
from torch.utils.backcompat import keepdim_warning

import pyrfm
import torch


def u(x):
    # $u(\mathbf{x})=$ $\left(\frac{1}{d} \sum_{i=1}^d x_i\right)^2+\sin \left(\frac{1}{d} \sum_{i=1}^d x_i\right), \forall \mathbf{x} \in \mathbf{R}^d$,
    return ((1.0 / x.shape[1] * x.sum(dim=1, keepdim=True)) ** 2
            + torch.sin(1.0 / x.shape[1] * x.sum(dim=1, keepdim=True)))


def f(x):
    return - (2.0 - torch.sin(1.0 / x.shape[1] * x.sum(dim=1, keepdim=True))) / x.shape[1]


def g(x):
    return u(x)


class CustomRF(pyrfm.RFTanH):
    def __init__(self, dim: int, center: torch.Tensor, radius: torch.Tensor,
                 n_hidden: int,
                 gen: torch.Generator = None,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__(dim, center, radius, n_hidden, gen, dtype, device)

        torch.nn.init.xavier_normal_(self.weights, gain=torch.nn.init.calculate_gain('tanh'))
        torch.nn.init.normal_(self.biases, mean=0.0, std=1 / n_hidden)

    def forward(self, x):
        return super(CustomRF, self).forward(x)


if __name__ == '__main__':
    dim = 5
    seed = 2025
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.set_default_device('cuda') if torch.cuda.is_available() else torch.set_default_device('cpu')
    domain = pyrfm.HyperCube(dim=dim)

    model = pyrfm.RFMBase(rf=CustomRF, dim=dim, n_hidden=1000, domain=domain, n_subdomains=1, seed=seed)

    x_in = domain.in_sample(3000, with_boundary=False)
    x_on = domain.on_sample(400 * 2 * dim)

    A_in_xx = model.features_second_derivative(x_in, axis1=0, axis2=0).cat(dim=1)
    for i in range(1, dim):
        A_in_xx += model.features_second_derivative(x_in, axis1=i, axis2=i).cat(dim=1)

    A_on = model.features(x_on).cat(dim=1)

    A = pyrfm.concat_blocks([[-A_in_xx], [A_on]])

    f_in = f(x_in).view(-1, 1)
    f_on = g(x_on).view(-1, 1)

    f = pyrfm.concat_blocks([[f_in], [f_on]])

    model.compute(A).solve(f)

    x_test = domain.in_sample(40, with_boundary=True)

    u_test = u(x_test).view(-1, 1)

    u_pred = model(x_test)

    print((u_test - u_pred).norm() / u_test.norm())
