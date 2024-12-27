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
    domain = Heart()
    voronoi = pyrfm.Voronoi(domain, k=8)
    points = voronoi.points
    x_min, x_max, y_min, y_max = domain.get_bounding_box()
    delta = max(x_max - x_min, y_max - y_min)
    virtual_points = [[x_min - delta, y_min - delta], [x_max + delta, y_min - delta],
                      [0.5 * (x_max + x_min), y_max + 2 * delta]]
    centers = torch.cat([points, torch.tensor(virtual_points)], dim=0)
    voronoi = pyrfm.Voronoi(domain, centers=centers)

    print(voronoi.vertices)
    voronoi_plot_2d(voronoi.voronoi_, show_points=True, show_vertices=True, line_alpha=0.6)
    p = domain.in_sample(1000)
    plt.scatter(x=p[:, 0], y=p[:, 1])
    plt.xlim(-1.5, 1.5)
    plt.ylim(-1.5, 1.5)
    plt.show()
