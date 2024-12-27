# -*- coding: utf-8 -*-
"""
Created on 2024/12/24

@author: Yifei Sun
"""

import torch
import pyrfm
from scipy.spatial import voronoi_plot_2d
import matplotlib.pyplot as plt

if __name__ == '__main__':
    torch.set_default_device('cuda') if torch.cuda.is_available() else torch.set_default_device('cpu')
    # domain = pyrfm.Circle2D(center=[0, 0], radius=1)
    domain = pyrfm.Square2D(center=[0, 0], radius=[1, 1])
    model = pyrfm.RFMBase(dim=2, n_hidden=400, domain=domain, n_subdomains=4)

    boundary = domain.on_sample(1000)

    # voronoi = pyrfm.Voronoi(domain)
    voronoi = pyrfm.Voronoi(domain, centers=model.centers)
    # print(voronoi.ridge_vertices)
    interface_dict, all_pts = voronoi.interface_sample(400)
    # print(interface_dict)
    # print(all_pts.size())

    voronoi_plot_2d(voronoi.voronoi_)
    # plot scatter points
    plt.scatter(x=all_pts[:, 0].cpu().numpy(), y=all_pts[:, 1].cpu().numpy(), marker="x", color="red")
    # plot boundary points
    plt.scatter(x=boundary[:, 0].cpu().numpy(), y=boundary[:, 1].cpu().numpy(), marker="o", color="blue")
    bounding_box = domain.get_bounding_box()
    plt.xlim(bounding_box[0], bounding_box[1])
    plt.ylim(bounding_box[2], bounding_box[3])
    plt.show()
