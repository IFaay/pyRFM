from typing import List
import pyrfm
import torch
import time


class Torus(pyrfm.ImplicitSurfaceBase):
    def __init__(self):
        super().__init__()

    def get_bounding_box(self) -> List[float]:
        return [-1.25, 1.25, -1.25, 1.25, -0.25, 0.25]

    def shape_func(self, p: torch.Tensor) -> torch.Tensor:
        # ψ(x, y, z) = (√(x² + y²) − 1)² + z² − 1 / 16
        return (((p[:, 0] ** 2 + p[:, 1] ** 2) ** 0.5 - 1.0) ** 2 + (p[:, 2] ** 2) - 1 / 16).unsqueeze(-1)


if __name__ == '__main__':
    torch.set_default_device('cuda') if torch.cuda.is_available() else torch.set_default_device('cpu')
    domain = Torus()
    x = domain.in_sample(num_samples=100000)

    _, normal, mean_curvature = domain.sdf(x, with_normal=True, with_curvature=True)

    # save the points, normals, and mean curvature in a file
    torch.save((x, normal, mean_curvature), '../../data/torus_in.pth')

    # load the points, normals, and mean curvature from the file
    # x, normal, mean_curvature = torch.load('../../data/torus_in.pth', map_location=torch.tensor(0.).device)
