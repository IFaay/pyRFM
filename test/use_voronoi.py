from typing import Union, Tuple

import matplotlib.pyplot as plt
import torch
from scipy.spatial import voronoi_plot_2d

import pyrfm


class Heart(pyrfm.GeometryBase):
    def __init__(self):
        super(Heart, self).__init__(2, 2)

    def sdf(self, p: torch.Tensor):
        pass

    def get_bounding_box(self):
        return -1.5, 1.5, -1.5, 1.5

    def in_sample(self, num_samples: int, with_boundary: bool = False):
        points = []
        count = 0
        while count < num_samples:
            x = torch.rand(1) * 3 - 1.5
            y = torch.rand(1) * 3 - 1.5
            if (x ** 2 + y ** 2 - 1) ** 3 - x ** 2 * (y ** 3) <= 0 and (x ** 2 + y ** 2 - 0.99) >= 0:
                points.append([x, y])
                count += 1
        return torch.tensor(points)

    def on_sample(self, num_samples: int, with_normal: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        pass


if __name__ == '__main__':
    # domain = Heart()
    domain = pyrfm.Cube3D(center=[0, 0, 0], radius=[1, 1, 1])
    model = pyrfm.RFMBase(dim=3, n_hidden=100, n_subdomains=2, pou=pyrfm.PsiA, domain=domain)
    voronoi = pyrfm.Voronoi(domain, centers=model.centers)
    points = voronoi.points
    print(points)
    bounding_box = torch.tensor(domain.get_bounding_box()).view(-1, 2)
    D = (bounding_box[:, 1] - bounding_box[:, 0]).norm(p=2)
    center = bounding_box.mean(dim=1).view(1, domain.dim)
    alpha = 2 * domain.dim
    R = D * alpha
    virtual_points = [center + R * torch.eye(domain.dim)[i] for i in range(domain.dim)]
    virtual_points.append(center - R * torch.ones(domain.dim))
    print(center)
    print(torch.cat(virtual_points, dim=0))

    centers = torch.cat([points, torch.cat(virtual_points, dim=0)], dim=0)
    voronoi = pyrfm.Voronoi(domain, centers=centers)

    n_domain = 8
    vertices = voronoi.vertices
    ridge_points = voronoi.ridge_points
    ridge_vertices = voronoi.ridge_vertices
    mask = []
    for i in range(len(ridge_points)):
        for j in range(len(ridge_points[i])):
            if ridge_points[i][j] >= n_domain:
                mask.append(i)
                break

    ridge_vertices = [ridge_vertices[i] for i in range(len(ridge_vertices)) if i not in mask]
    ridge_points = [list(ridge_points[i]) for i in range(len(ridge_points)) if i not in mask]

    print(ridge_vertices)
    print(ridge_points)
    print(len(ridge_vertices))
    print(len(ridge_points))

    # voronoi_plot_2d(voronoi.voronoi_, show_points=True, show_vertices=True, line_alpha=0.6)
    # p = domain.in_sample(1000)
    # plt.scatter(x=p[:, 0], y=p[:, 1])
    # plt.xlim(-1.5, 1.5)
    # plt.ylim(-1.5, 1.5)
    # plt.show()
