from typing import List
import pyrfm
import torch
import time


class Genus2Torus(pyrfm.ImplicitSurfaceBase):
    def __init__(self):
        super().__init__()

    def get_bounding_box(self) -> List[float]:
        return [-1.5, 1.5, -0.6, 0.6, -0.1, 0.1]

    def shape_func(self, p: torch.Tensor) -> torch.Tensor:
        # ψ(x, y, z) = [(x + 1)x²(x − 1) + y²]² + z² − 0.01
        return (((p[:, 0] + 1.0) * p[:, 0] ** 2 * (p[:, 0] - 1.0) + p[:, 1] ** 2) ** 2 + p[:, 2] ** 2 - 0.01).unsqueeze(
            -1)


if __name__ == '__main__':
    torch.set_default_device('cuda') if torch.cuda.is_available() else torch.set_default_device('cpu')

    domain = Genus2Torus()
    x = domain.in_sample(100000)
    _, normal, mean_curvature = domain.sdf(x, with_normal=True, with_curvature=True)

    # save the points, normals, and mean curvature in a file
    torch.save((x, normal, mean_curvature), '../../data/genus2_in.pth')

    # # load the points, normals, and mean curvature from the file
    # x, normal, mean_curvature = torch.load('../../data/genus2_in.pth', map_location=torch.tensor(0.).device)
