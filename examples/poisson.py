# -*- coding: utf-8 -*-
"""
Created on 2024/12/17

@author: Yifei Sun
"""
import pyrfm
import torch


# -(uxx + uyy) = f

def g(x):
    return torch.sin(torch.pi * x[:, 0]) * torch.sin(torch.pi * x[:, 1])


def u(x):
    return torch.sin(torch.pi * x[:, 0]) * torch.sin(torch.pi * x[:, 1])


def f(x):
    return 2 * torch.pi ** 2 * g(x)


if __name__ == '__main__':
    torch.set_default_device('cuda') if torch.cuda.is_available() else torch.set_default_device('cpu')
    domain = pyrfm.Square2D(center=[0, 0], radius=[1, 1])
    model = pyrfm.RFMBase(dim=2, n_hidden=100, domain=domain, n_subdomains=4)

    x_in = domain.in_sample(100, with_boundary=False)

    x_on = domain.on_sample(400)

    A_in_xx = model.features_second_derivative(x_in, axis1=0, axis2=0).cat(dim=1)
    A_in_yy = model.features_second_derivative(x_in, axis1=1, axis2=1).cat(dim=1)

    A_on = model.features(x_on).cat(dim=1)

    A = pyrfm.concat_blocks([[- (A_in_xx + A_in_yy)], [A_on]])

    f_in = f(x_in).view(-1, 1)
    f_on = g(x_on).view(-1, 1)

    f = pyrfm.concat_blocks([[f_in], [f_on]])

    model.compute(A).solve(f)
    # model.solve(A, f)

    x_test = domain.in_sample(40, with_boundary=True)
    u_test = u(x_test).view(-1, 1)
    u_pred = model(x_test)

    print((u_test - u_pred).norm() / u_test.norm())
