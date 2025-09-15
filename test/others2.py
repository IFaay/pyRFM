# -*- coding: utf-8 -*-
"""
Created on 2025/4/21

@author: Yifei Sun
"""
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 09:19:18 2025

@author: Administrator
"""

import time

import pyrfm
import torch

"""
Consider the Laplace equation with Dirichlet and Neumann boundary condition:
∇²u = 0, x∈ Ω,  
u = 1, x∈Γ₁,
∂u/∂n = 4/ln4, x∈Γ₂.
where Ω is the annular region between two circles with radii R₁ = 1(Γ₁) and R₂ = 1/4(Γ₂).
Exact solutions:
u = 1 - ln(√(x² + y²)) / ln4.
"""


# 精确解定义
def u(x):
    return 1 - torch.log(torch.sqrt((x ** 2).sum(dim=1, keepdim=True))) / torch.log(torch.tensor(4))


# 设置默认设备
torch.set_default_device('cuda') if torch.cuda.is_available() else torch.set_default_device('cpu')

# 定义问题域
Outer_circle = pyrfm.Circle2D(center=[0.0, 0.0], radius=1.0)
Inner_circle = pyrfm.Circle2D(center=[0.0, 0.0], radius=0.25)
domain = Outer_circle - Inner_circle

# 创建RFM模型
model = pyrfm.RFMBase(dim=2, n_hidden=100, domain=domain, n_subdomains=2)

# 内部点采样（不包含边界）
x_in = domain.in_sample(5000, with_boundary=False)

# 构建PDE方程部分：∇²u = 0
uxx = model.features_second_derivative(x_in, axis1=0, axis2=0).cat(dim=1)
uyy = model.features_second_derivative(x_in, axis1=1, axis2=1).cat(dim=1)

A_in = uxx + uyy
f_in = torch.zeros((A_in.shape[0], 1))

# 外边界点采样
x_on_outer = Outer_circle.on_sample(1000)

# Dirichlet边界条件：u=1
u_on_outer = model.features(x_on_outer).cat(dim=1)
f_outer = torch.ones(x_on_outer.shape[0], 1)

# Neumann边界条件：∂u/∂n = 4/ln4
x_on_inner, n_inner = Inner_circle.on_sample(1000, with_normal=True)
n_inner = - n_inner
u_x_on_inner = model.features_derivative(x_on_inner, axis=0).cat(dim=1)
u_y_on_inner = model.features_derivative(x_on_inner, axis=1).cat(dim=1)

u_on_inner = u_x_on_inner * n_inner[:, [0]] + u_y_on_inner * n_inner[:, [1]]
f_inner = torch.full((u_on_inner.shape[0], 1), 4.0 / torch.log(torch.tensor(4.0)))

# 构建系统矩阵A和向量b
A = pyrfm.concat_blocks([[A_in],
                         [u_on_outer],
                         [u_on_inner]])
b = pyrfm.concat_blocks([[f_in],
                         [f_outer],
                         [f_inner]])

# 求解
model.compute(A).solve(b)

# 测试与计算误差
x_test = domain.in_sample(100, with_boundary=True)
u_pre = model(x_test)
u_true = u(x_test)

# 打印误差
print('Error:', (u_pre - u_true).norm() / u_true.norm())
