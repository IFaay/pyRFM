# -*- coding: utf-8 -*-
"""
Created on 2025/2/23

@author: Yifei Sun
"""

import pyrfm
import torch
import os
import argparse
import sys
import time
import matplotlib.pyplot as plt

from scipy.optimize import least_squares
import numpy as np

"""
Consider the burgers equation with Dirichlet boundary condition over (x, t) ∈ [-1, 1] × [0, 1]:

    du/dt + u * du/dx - v * d²u/dx² = 0,    x ∈ [-1, 1]，   t ∈ [0, 1]，

with boundary conditions:

    u(x, 0) = -sin(pi * x) = h(x),   
    u(-1, t) = u(1, t) = 0 = g(t)

"""

v = 0.01
Nx = 2
Nt = 5
Qx = 20
Qt = 20
Jn = 400
Nb = 10


def func_g(xt):
    return torch.zeros(xt.shape[0], 1)


def func_h(xt):
    x = xt[:, :-1]
    return - torch.sin(torch.pi * x)


if __name__ == '__main__':
    torch.set_default_device('cuda') if torch.cuda.is_available() else torch.set_default_device('cpu')
    start_time = time.time()
    time_stamp = torch.linspace(start=0, end=1, steps=Nb + 1)
    domain = pyrfm.Line1D(x1=-1, x2=1)
    models = []
    for (t0, t1) in zip(time_stamp[:-1], time_stamp[1:]):
        models.append(pyrfm.STRFMBase(dim=1,
                                      n_hidden=Jn,
                                      domain=domain,
                                      time_interval=[t0, t1],
                                      n_spatial_subdomains=Nx,
                                      n_temporal_subdomains=Nt,
                                      st_type="SOV"))

    x_in = domain.in_sample(Qx * Nx, with_boundary=False)
    x_on = domain.on_sample(2)

    for i, model in enumerate(models):
        t0 = torch.tensor(model.time_interval[0]).reshape(-1, 1)
        t = torch.linspace(*model.time_interval, (Qt * Nt) + 1)[1:].reshape(-1, 1)

        x_t0 = model.validate_and_prepare_xt(x=torch.cat([x_in, x_on], dim=0),
                                             t=t0)
        x_in_t = model.validate_and_prepare_xt(x=x_in, t=t)
        x_on_t = model.validate_and_prepare_xt(x=x_on, t=t)

        A_init = model.features(xt=x_t0).cat(dim=1)
        A_boundary = model.features(xt=x_on_t).cat(dim=1)
        A = model.features(xt=x_in_t).cat(dim=1)
        A_t = model.features_derivative(xt=x_in_t, axis=1).cat(dim=1)
        A_x = model.features_derivative(xt=x_in_t, axis=0).cat(dim=1)
        A_xx = model.features_second_derivative(xt=x_in_t, axis1=0, axis2=0).cat(dim=1)

        if i == 0:
            f_init = func_h(x_t0)
        else:
            f_init = models[i - 1].forward(xt=x_t0)

        f_boundary = func_g(x_on_t)


        def fcn(w):
            u = A @ w
            u_t = A_t @ w
            u_x = A_x @ w
            u_xx = A_xx @ w
            u_init = A_init @ w
            u_boundary = A_boundary @ w
            return torch.cat([u_t + u * u_x - v * u_xx, u_boundary - f_boundary, u_init - f_init])


        def jac(w):
            return torch.cat([A_t + (A @ w) * A_x + (A_x @ w) * A - v * A_xx, A_boundary, A_init], dim=0)


        # tol = torch.finfo(torch.float64).eps
        tol = 1e-6
        result = pyrfm.nonlinear_least_square(fcn=fcn,
                                              x0=torch.zeros((A.shape[1], 1)),
                                              jac=jac,
                                              ftol=tol,
                                              gtol=tol,
                                              xtol=tol,
                                              method='newton',
                                              verbose=2)

        model.W = result[0]

    print("Training completed.")

    data = np.load("burgers_data_in.npy")
    xt_test = data[:, :2]
    xt_test_tensor = torch.from_numpy(xt_test).float().to(torch.tensor(0.).device)
    u_test = data[:, -1]
    u_test_tensor = torch.from_numpy(u_test).float().view(-1, 1).to(torch.tensor(0.).device)
    t_test = xt_test_tensor[:, -1:]
    n_groups = Nb
    boundaries = torch.linspace(0, 1, n_groups + 1)
    groups = torch.bucketize(t_test.contiguous(), boundaries.contiguous(), right=False) - 1
    xt_groups = torch.cat([xt_test_tensor, groups], dim=1)
    u_pred = []

    batch_size = 49  # 每次取出的行数
    num_rows = xt_groups.size(0)  # xt_groups 的总行数

    for i in range(xt_groups.size(0)):
        row_subtensor = xt_groups[i:i + 1]
        u_pred.append(models[int(row_subtensor[:, -1])].forward(xt=row_subtensor[:, :-1]))
    u_pred_tensor = torch.tensor(u_pred)
    print("u_pred_tensor contains NaN:", torch.isnan(u_pred_tensor).any())
    print("Relative Error: ", torch.linalg.norm(u_pred_tensor - u_test_tensor) / torch.linalg.norm(u_test_tensor))

    print('Time:', time.time() - start_time, ", with", torch.tensor(0.).device)

    '''
    for start in range(0, num_rows, batch_size):
        end = min(start + batch_size, num_rows)  # 确保不超过总行数
        batch = xt_groups[start:end]  # 取出当前批次的行
        u_pred.append(models[int(batch[1, -1])].forward(xt=batch[:, :-1]))
    u_pred_tensor = torch.cat(u_pred, dim=0)
    print("u_pred_tensor contains NaN:", torch.isnan(u_pred_tensor).any())
    print("Relative Error: ", torch.linalg.norm(u_pred_tensor - u_test_tensor) / torch.linalg.norm(u_test_tensor))

    print('Time:', time.time() - start_time, ", with", torch.tensor(0.).device)
    '''

    '''
    x = torch.linspace(start=-1, end=1, steps=200).reshape(-1, 1)  # 在 [-1, 1] 之间生成200个点
    t = 0.2 * torch.ones(200).reshape(-1, 1)  # 时间点为0.2
    x_t = torch.cat([x, t], dim=1)  # 将 x 和 t 直接拼接在一起
    y = models[0].forward(xt=x_t).cpu().numpy()  # 计算模型输出
    x = x.cpu().numpy()  # 将数据转换为 NumPy 数组
    plt.figure(figsize=(8, 4))  # 设置图像大小（可选）
    #plt.scatter(x, y, label="u(x)", color="red", marker="o")  # 绘制散点图
    plt.plot(x, y, label="t = 0.2", color="blue", linestyle="--")  # 绘制曲线
    

    # 添加标题、坐标轴标签和图例
    plt.title("Function Plot")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.grid(True)  # 显示网格
    plt.legend()

    plt.show()  # 显示图像
    '''
