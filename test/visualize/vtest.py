from typing import List

import pyrfm
import torch


class Ellipsoid(pyrfm.ImplicitSurfaceBase):
    def __init__(self):
        super().__init__()

    def get_bounding_box(self) -> List[float]:
        return [-1.5, 1.5, -1.0, 1.0, -0.5, 0.5]

    def shape_func(self, p: torch.Tensor) -> torch.Tensor:
        # Ellipsoid equation: (x/a)^2 + (y/b)^2 + (z/c)^2 = 1
        a, b, c = 1.5, 1.0, 0.5
        return ((p[:, 0] / a) ** 2 + (p[:, 1] / b) ** 2 + (p[:, 2] / c) ** 2 - 1).unsqueeze(-1)


if __name__ == '__main__':
    # domain = pyrfm.Ball3D(center=(0, 0, 0), radius=1.0) + pyrfm.Ball3D(center=(0, 0, 1.5), radius=0.5) + pyrfm.Square3D(
    #     center=(1.0, 0, 0), radius=(1.0, 0.5, 0.5))
    #
    # model = pyrfm.RFMBase(dim=3, n_hidden=200, domain=domain, n_subdomains=2)
    #
    # model.W = torch.randn((2 * 2 * 2 * 200, 1))
    # # model.W = torch.zeros((model.submodels.numel() * 200, 1))
    #
    # visualizer = pyrfm.RFMVisualizer3D(model, view='front-right', resolution=(800, 800))
    # visualizer.plot()
    # visualizer.show()
    # visualizer.savefig('test.png')

    domain = Ellipsoid()
    x_in = domain.in_sample(num_samples=1000)
    print(domain.sdf(x_in).abs().max().item())

    # # plot x_in as scatter plot in 3D
    # from mpl_toolkits.mplot3d import Axes3D
    # import matplotlib.pyplot as plt
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(x_in[:, 0].numpy(), x_in[:, 1].numpy(), x_in[:, 2].numpy(), s=1)
    # ax.set_xlabel('X axis')
    # ax.set_ylabel('Y axis')
    # ax.set_zlabel('Z axis')
    # plt.show()
    #
    # print(x_in)
