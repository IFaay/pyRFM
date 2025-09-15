# -*- coding: utf-8 -*-
"""
Created on 2024/12/17

@author: Yifei Sun
"""
import time

import pyrfm
import torch
import os


def u(x):
    return -0.5 * (1.5 * torch.cos(torch.pi * x[:, [0]] + 2 * torch.pi / 5) +
                   2 * torch.cos(2 * torch.pi * x[:, [0]] - torch.pi / 5)) * \
        (1.5 * torch.cos(torch.pi * x[:, [1]] + 2 * torch.pi / 5) +
         2 * torch.cos(2 * torch.pi * x[:, [1]] - torch.pi / 5)) - \
        0.5 * (1.5 * torch.cos(2 * torch.pi * x[:, [0]] + 2 * torch.pi / 5) +
               2 * torch.cos(4 * torch.pi * x[:, [0]] - torch.pi / 5)) * \
        (1.5 * torch.cos(2 * torch.pi * x[:, [1]] + 2 * torch.pi / 5) +
         2 * torch.cos(4 * torch.pi * x[:, [1]] - torch.pi / 5))


# -(uxx + uyy) = f
def f(x):
    return -(-0.5 * (-1.5 * torch.pi ** 2 * torch.cos(torch.pi * x[:, [0]] + 2 * torch.pi / 5) -
                     2 * (2 * torch.pi) ** 2 * torch.cos(2 * torch.pi * x[:, [0]] - torch.pi / 5)) * \
             (1.5 * torch.cos(torch.pi * x[:, [1]] + 2 * torch.pi / 5) +
              2 * torch.cos(2 * torch.pi * x[:, [1]] - torch.pi / 5)) - \
             0.5 * (1.5 * torch.cos(torch.pi * x[:, [0]] + 2 * torch.pi / 5) +
                    2 * torch.cos(2 * torch.pi * x[:, [0]] - torch.pi / 5)) * \
             (-1.5 * torch.pi ** 2 * torch.cos(torch.pi * x[:, [1]] + 2 * torch.pi / 5) -
              2 * (2 * torch.pi) ** 2 * torch.cos(2 * torch.pi * x[:, [1]] - torch.pi / 5)) - \
             0.5 * (-1.5 * (2 * torch.pi) ** 2 * torch.cos(2 * torch.pi * x[:, [0]] + 2 * torch.pi / 5) -
                    2 * (4 * torch.pi) ** 2 * torch.cos(4 * torch.pi * x[:, [0]] - torch.pi / 5)) * \
             (1.5 * torch.cos(2 * torch.pi * x[:, [1]] + 2 * torch.pi / 5) +
              2 * torch.cos(4 * torch.pi * x[:, [1]] - torch.pi / 5)) - \
             0.5 * (1.5 * torch.cos(2 * torch.pi * x[:, [0]] + 2 * torch.pi / 5) +
                    2 * torch.cos(4 * torch.pi * x[:, [0]] - torch.pi / 5)) * \
             (-1.5 * (2 * torch.pi) ** 2 * torch.cos(2 * torch.pi * x[:, [1]] + 2 * torch.pi / 5) -
              2 * (4 * torch.pi) ** 2 * torch.cos(4 * torch.pi * x[:, [1]] - torch.pi / 5)))


def g(x):
    return u(x)


# class CustomRF(pyrfm.RFTanH):
#     def __init__(self, *args, **kwargs):
#         super(CustomRF, self).__init__(*args, **kwargs)
#         self.weights: torch.Tensor = torch.rand((self.dim, self.n_hidden), generator=self.gen, dtype=self.dtype,
#                                                 device=self.device) * 50 - 25
#         self.biases: torch.Tensor = torch.rand((1, self.n_hidden), generator=self.gen, dtype=self.dtype,
#                                                device=self.device) * 50 - 25


if __name__ == '__main__':
    torch.set_default_device('cuda') if torch.cuda.is_available() else torch.set_default_device('cpu')
    domain = pyrfm.Square2D(center=[0, 0], radius=[1, 1])
    model = pyrfm.RFMBase(dim=2, n_hidden=1000, domain=domain, n_subdomains=1)

    x_in = domain.in_sample(10000, with_boundary=False)

    x_on = domain.on_sample(4 * 100)

    A_in_xx = model.features_second_derivative(x_in, axis1=0, axis2=0).cat(dim=1)
    A_in_yy = model.features_second_derivative(x_in, axis1=1, axis2=1).cat(dim=1)

    A_on = model.features(x_on).cat(dim=1)

    A = pyrfm.concat_blocks([[-(A_in_xx + A_in_yy)], [A_on]])

    f_in = f(x_in).view(-1, 1)
    f_on = g(x_on).view(-1, 1)

    f = pyrfm.concat_blocks([[f_in], [f_on]])

    model.compute(A).solve(f)
    x_test = domain.in_sample(40, with_boundary=True)
    u_test = u(x_test).view(-1, 1)
    u_pred = model(x_test)

    print((u_test - u_pred).norm() / u_test.norm())

    U, S, V = torch.svd(A, some=True)
    threshold = 1e-14
    mask = S > threshold
    U_k = U[:, mask]
    S_k = S[mask]
    V_k = V[:, mask]
    U, S, V = U_k, S_k, V_k
    print("Shape of U", U.shape)
    print("Shape of S", S.shape)
    print("Shape of V", V.shape)
    print(torch.dist(A, torch.mm(torch.mm(U, torch.diag(S)), V.t())))

    w = V @ ((U.t() @ f) / S.view(-1, 1))
    print("Residual: ", (f - A @ w).norm())

    w_set = []
    f_ = f.clone()
    for i in range(100):
        w = V @ ((U.t() @ f_) / S.view(-1, 1))
        w_set.append(w)
        f_ = f_ - A @ w
        print("Residual: ", f_.norm() / f.norm())
    w = torch.cat(w_set, dim=1).sum(dim=1).view(-1, 1)

    model.W = w

    x_test = domain.in_sample(40, with_boundary=True)
    u_test = u(x_test).view(-1, 1)
    u_pred = model(x_test)

    print((u_test - u_pred).norm() / u_test.norm())
