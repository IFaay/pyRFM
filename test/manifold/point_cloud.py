# -*- coding: utf-8 -*-
"""
Created on 7/31/25

@author: Yifei Sun
"""
from typing import List

import pyrfm
import torch
import time


class Ellipsoid(pyrfm.ImplicitSurfaceBase):

    def __init__(self):
        super().__init__()
        self.a, self.b, self.c = 1.5, 1.0, 0.5

    def get_bounding_box(self) -> List[float]:
        return [-self.a, self.a, -self.b, self.b, -self.c, self.c]

    def shape_func(self, p: torch.Tensor) -> torch.Tensor:
        # Ellipsoid equation: (x/a)^2 + (y/b)^2 + (z/c)^2 = 1

        return ((p[:, 0] / self.a) ** 2 + (p[:, 1] / self.b) ** 2 + (p[:, 2] / self.c) ** 2 - 1).unsqueeze(-1)


if __name__ == '__main__':
    torch.set_default_device('cuda') if torch.cuda.is_available() else torch.set_default_device('cpu')
    domain = Ellipsoid()
    model = pyrfm.RFMBase(dim=3, n_hidden=1000, domain=domain, n_subdomains=1)

    num_samples = 100
    oversample = 1000
    x_min, x_max, y_min, y_max, z_min, z_max = domain.get_bounding_box()
    volume = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
    resolution = (volume / num_samples) ** (1 / domain.dim)
    eps = 0.0
    rand = torch.rand(oversample, domain.dim)
    p = torch.empty(oversample, domain.dim)
    p[:, 0] = (x_min - eps) + rand[:, 0] * ((x_max + eps) - (x_min - eps))
    p[:, 1] = (y_min - eps) + rand[:, 1] * ((y_max + eps) - (y_min - eps))
    p[:, 2] = (z_min - eps) + rand[:, 2] * ((z_max + eps) - (z_min - eps))

    # Step 1: 采样点 & 几何
    x_in = domain.in_sample(num_samples=num_samples)
    x_all = torch.cat((p, x_in), dim=0)
    sdf, normal, mean_curvature = domain.sdf(x_all, with_normal=True, with_curvature=True)

    # plot sdf
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_all[:, 0].cpu(), x_all[:, 1].cpu(),
               x_all[:, 2].cpu(), c=sdf.cpu().squeeze(), cmap='viridis', marker='o', s=1)
    vmin = sdf.min().item()
    vmax = sdf.max().item()
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    plt.colorbar(sm, ax=ax)
    plt.show()

    A = model.features(x_all).cat(dim=1)
    b = domain.shape_func(x_all)
    # b = sdf
    model.compute(A).solve(b)

    normal_estimate = torch.cat([model.dForward(x_in, order=(1, 0, 0)),
                                 model.dForward(x_in, order=(0, 1, 0)),
                                 model.dForward(x_in, order=(0, 0, 1))],
                                dim=-1)

    normal_estimate = normal_estimate / torch.norm(normal_estimate, dim=-1, keepdim=True)

    _, normal_exact, _ = domain.sdf(x_in, with_normal=True, with_curvature=True)

    print(normal_exact.shape, normal_estimate.shape)

    normal_error = torch.norm(normal_exact - normal_estimate) / torch.norm(normal_exact)

    print(normal_error)
