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
    x = domain.in_sample(100000)
    _, normal, mean_curvature = domain.sdf(x, with_normal=True, with_curvature=True)

    # save the points, normals, and mean curvature in a file
    torch.save((x, normal, mean_curvature), '../../data/ellipsoid_in.pth')

    print(x.shape, normal.shape, mean_curvature.shape)

    # # load the points, normals, and mean curvature from the file
    x, normal, mean_curvature = torch.load('../../data/ellipsoid_in.pth', map_location=torch.tensor(0.).device)
