from typing import List
import pyrfm
import torch
import time
import math

import matplotlib.pyplot as plt


class CheeseLike(pyrfm.ImplicitSurfaceBase):
    def __init__(self):
        super().__init__()

    def get_bounding_box(self) -> List[float]:
        return [-1.25, 1.25, -1.25, 1.25, -1.25, 1.25]

    def shape_func(self, p: torch.Tensor) -> torch.Tensor:
        """
        ψ(x, y, z) = (4x² − 1)² + (4y² − 1)² + (4z² − 1)²
                               + 16(x² + y² − 1)² + 16(x² + z² − 1)² + 16(y² + z² − 1)² − 16
        """
        return ((4 * p[:, 0] ** 2 - 1) ** 2 + (4 * p[:, 1] ** 2 - 1) ** 2 + (4 * p[:, 2] ** 2 - 1) ** 2
                + 16 * (p[:, 0] ** 2 + p[:, 1] ** 2 - 1) ** 2 + 16 * (p[:, 0] ** 2 + p[:, 2] ** 2 - 1) ** 2
                + 16 * (p[:, 1] ** 2 + p[:, 2] ** 2 - 1) ** 2 - 16).unsqueeze(-1)


if __name__ == '__main__':
    torch.set_default_device('cuda') if torch.cuda.is_available() else torch.set_default_device('cpu')

    domain = CheeseLike()
    x = domain.in_sample(100000)
    _, normal, mean_curvature = domain.sdf(x, with_normal=True, with_curvature=True)

    # save the points, normals, and mean curvature in a file
    torch.save((x, normal, mean_curvature), '../../data/cheese_in.pth')

    # # load the points, normals, and mean curvature from the file
    # x, normal, mean_curvature = torch.load('../../data/cheese_in.pth', map_location=torch.tensor(0.).device)
