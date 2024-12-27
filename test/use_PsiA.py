# -*- coding: utf-8 -*-
"""
Created on 2024/12/27

@author: Yifei Sun
"""
import torch

import pyrfm

if __name__ == '__main__':
    x = torch.linspace(-1, 1, 100)
    y = torch.linspace(-1, 1, 100)
    X, Y = torch.meshgrid(x, y)
    points = torch.stack([X.flatten(), Y.flatten()], dim=1)

    domain = pyrfm.Square2D(center=[0, 0], radius=[1, 1])
    model = pyrfm.RFMBase(dim=2, n_hidden=400, domain=domain, n_subdomains=4, pou=pyrfm.PsiA)

    value = model.pou_coefficients(torch.tensor([[-0.5, 0.5]]))
    print(value)
    # value = value.reshape(100, 100).cpu().detach().numpy()

    # import matplotlib.pyplot as plt
    #
    # plt.imshow(value, extent=[-1, 1, -1, 1])
    # plt.colorbar()
    # plt.show()
