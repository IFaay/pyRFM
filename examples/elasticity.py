# -*- coding: utf-8 -*-
"""
Created on 2024/12/17

@author: Yifei Sun
"""
import time

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
    model = pyrfm.RFMBase(dim=2, n_hidden=400, domain=domain, n_subdomains=2)

    x_in = domain.in_sample(2500, with_boundary=False)

    x_on = domain.on_sample(100)

    u_in = model
