# -*- coding: utf-8 -*-
"""
Created on 2024/12/21

@author: Yifei Sun
"""
import pyrfm
import torch
import matplotlib.pyplot as plt


def plot_sdf(sdf_func, bounding_box, resolution=100):
    x_min, x_max, y_min, y_max = bounding_box
    x = torch.linspace(x_min, x_max, resolution)
    y = torch.linspace(y_min, y_max, resolution)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    points = torch.cat([X.reshape(-1, 1), Y.reshape(-1, 1)], dim=1)
    sdf_values = sdf_func(points).reshape(resolution, resolution)
    print(sdf_values.min(), sdf_values.max())

    plt.figure(figsize=(8, 8))
    plt.contourf(X, Y, sdf_values, levels=20, cmap='coolwarm')
    plt.colorbar(label="SDF Value")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Signed Distance Function (SDF) Distribution")
    plt.axis('equal')
    plt.show()

# 示例使用
# square = pyrfm.Square2D(center=[0.0, 0.0], radius=[1.0, 1.0])
# x = square.in_sample(100, with_boundary=True)
# print(square.sdf(x))
# bounding_box = [-2.0, 2.0, -2.0, 2.0]
# plot_sdf(square.sdf, bounding_box)

# circle = pyrfm.Circle2D(center=[0.0, 0.0], radius=1.0)
# x = circle.in_sample(100, with_boundary=True)
# print(circle.sdf(x))
