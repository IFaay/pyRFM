# -*- coding: utf-8 -*-
"""
Created on 2024/12/17

@author: Yifei Sun
"""
import pyrfm
import torch

from matplotlib import pyplot as plt

from pyrfm import Square2D, concat_blocks


# -(uxx + uyy) = f

def g(x):
    return torch.sin(torch.pi * x[:, 0]) * torch.sin(torch.pi * x[:, 1])


def u(x):
    return torch.sin(torch.pi * x[:, 0]) * torch.sin(torch.pi * x[:, 1])


def f(x):
    return 2 * torch.pi ** 2 * g(x)


if __name__ == '__main__':
    domain = Square2D(center=[0, 0], radius=[1, 1])
    model = pyrfm.RFMBase(dim=2, n_hidden=100, domain=domain, n_subdomains=4)

    x_in = domain.in_sample(50, with_boundary=False)

    x_on = domain.on_sample(400)

    A_in_xx = model.features_second_derivative(x_in, axis1=0, axis2=0).cat(dim=1)
    A_in_yy = model.features_second_derivative(x_in, axis1=1, axis2=1).cat(dim=1)

    A_on = model.features(x_on).cat(dim=1)

    A = concat_blocks([[- (A_in_xx + A_in_yy)], [A_on]])

    f_in = f(x_in).view(-1, 1)
    f_on = g(x_on).view(-1, 1)

    f = concat_blocks([[f_in], [f_on]])

    model.Train(A, f)

    # x_test = domain.in_sample(50, with_boundary=True)
    # model.Train(model.features(x_test).cat(), u(x_test).view(-1, 1))

    x_test = domain.in_sample(40, with_boundary=True)
    u_test = u(x_test).view(-1, 1)
    u_pred = model(x_test)

    # print(u_test)
    # print(u_pred)

    print((u_test - u_pred).norm() / u_test.norm())

    # print(model.submodels[3])
    # print(model.radii)
    # print(id(model.submodels.flat_data[0]) == id(model.submodels[0, 0]))
    # x = torch.rand((3000, 2))
    # u = model.features(x, use_sparse=True).cat()
    # uxx = model.features_second_derivative(x, axis1=0, axis2=0)
    # print(u)

    # # 生成输入数据 (2D 网格)
    # x = torch.linspace(-1, 1, 100)
    # y = torch.linspace(-1, 1, 100)
    # X, Y = torch.meshgrid(x, y)
    # inputs = torch.cat([X.reshape(-1, 1), Y.reshape(-1, 1)], dim=1)
    #
    # # 计算 POU 系数、一阶导数和二阶导数
    # coefficients = model.pou_coefficients(inputs)[6].view(100, 100)
    # derivatives_x = model.pou_derivative(inputs, axis=0)[6].view(100, 100)
    # derivatives_y = model.pou_derivative(inputs, axis=1)[6].view(100, 100)
    # second_derivatives_xx = model.pou_second_derivative(inputs, axis1=0, axis2=0)[6].view(100, 100)
    # second_derivatives_yy = model.pou_second_derivative(inputs, axis1=1, axis2=1)[6].view(100, 100)
    # second_derivatives_xy = model.pou_second_derivative(inputs, axis1=0, axis2=1)[6].view(100, 100)
    #
    # # 绘制图像
    # plt.figure(figsize=(10, 6))
    # plt.pcolor(X, Y, second_derivatives_xy, cmap='coolwarm')
    # plt.colorbar()
    # plt.tight_layout()
    # plt.show()
