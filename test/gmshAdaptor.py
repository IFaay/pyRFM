# -*- coding: utf-8 -*-
"""
Created on 2025/9/27

@author: Yifei Sun
"""
from typing import Union, Tuple, Dict, List

import torch
import numpy as np

import pyrfm

import numpy as np
import torch
import meshio
from typing import Dict, List, Tuple, Union, Optional

import pyrfm  # 你现有的基类

import numpy as np
import torch
import meshio
from typing import Dict, List, Tuple, Union, Optional

import pyrfm

# —— 示例用法 ——
if __name__ == "__main__":
    torch.set_default_device(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    domain = pyrfm.GmshAdaptor("circle_2d.msh")
    x_in = domain.in_sample(with_boundary=True)
    x_on, x_on_n = domain.on_sample(with_normal=True)

    model = pyrfm.RFMBase(dim=2, domain=domain, n_hidden=100, n_subdomains=1)
    model.W = torch.randn((100, 1))

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 8))

    # interior+boundary 点
    x_in_np = x_in.cpu().numpy()
    sdf_in = domain.sdf(torch.as_tensor(x_in_np, dtype=torch.float32, device=domain.device)).detach().cpu().numpy()
    sc1 = plt.scatter(x_in_np[:, 0], x_in_np[:, 1], s=10, c=sdf_in, cmap="coolwarm", label="interior+boundary pts")

    # boundary 点
    x_on_np = x_on.cpu().numpy()
    sdf_on = domain.sdf(torch.as_tensor(x_on_np, dtype=torch.float32, device=domain.device)).detach().cpu().numpy()
    sc2 = plt.scatter(x_on_np[:, 0], x_on_np[:, 1], s=10, c=sdf_on, cmap="coolwarm", label="boundary pts")

    # 法向
    x_on_n_np = x_on_n.cpu().numpy()
    plt.quiver(x_on_np[:, 0], x_on_np[:, 1], x_on_n_np[:, 0], x_on_n_np[:, 1],
               angles="xy", scale_units="xy", scale=10.0, color="k", width=0.005)

    plt.gca().set_aspect("equal", "box")
    plt.legend()
    plt.colorbar(sc1, label="Signed Distance Function (SDF)")
    plt.show()

    viz = pyrfm.RFMVisualizer2D(model)
    viz.plot()
    viz.show()

    # import matplotlib.pyplot as plt
    #
    # # domain = GmshAdaptor("circle_2d.msh")
    # #
    # # print(domain.on_sample(with_normal=True))
    #
    # d = GmshAdaptor("circle_2d.msh")
    #
    # pts_in = d.in_sample(with_boundary=True)
    #
    # plt.figure(figsize=(8, 8))
    # pts_in = pts_in.cpu().numpy()
    # plt.scatter(pts_in[:, 0], pts_in[:, 1], s=10, label='interior+boundary pts')
    # plt.gca().set_aspect('equal', 'box')
    # plt.legend()
    # plt.show()
    #
    # pts_on, normals = d.on_sample(with_normal=True)
    #
    # print(pts_on.shape)
    # print(normals.shape)
    #
    # pts_on = pts_on.cpu().numpy()
    # normals = normals.cpu().numpy()
    #
    # plt.figure(figsize=(6, 6))
    # plt.scatter(pts_on[:, 0], pts_on[:, 1], s=10, label='boundary pts')
    # plt.quiver(pts_on[:, 0], pts_on[:, 1], normals[:, 0], normals[:, 1], angles='xy', scale_units='xy', scale=1,
    #            color='r', width=0.005)
    # plt.gca().set_aspect('equal', 'box')
    # plt.legend()
    # plt.show()

    # import meshio, numpy as np
    #
    # m = meshio.read("circle_2d.msh")
    # for blk in m.cells:
    #     print(blk.type, blk.data.shape)
    #
    # # 看看 line 覆盖点的数量
    # line = next((blk.data for blk in m.cells if blk.type == "line"), None)
    # if line is not None:
    #     print("unique line vertices:", np.unique(line).size)
    #
    # # 用三角形推断边界覆盖点数量
    # tri = next((blk.data for blk in m.cells if blk.type == "triangle"), None)
    # if tri is not None:
    #     edges = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    #     e_sorted = np.sort(edges, axis=1)
    #     uniq, counts = np.unique(e_sorted, axis=0, return_counts=True)
    #     b_edges = uniq[counts == 1]
    #     print("triangle-derived boundary vertices:", np.unique(b_edges).size)
