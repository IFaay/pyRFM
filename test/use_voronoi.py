from typing import Union, Tuple, List

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


import torch
from typing import List

import torch
from typing import List


def _polygon_box_intersection(polygon: torch.Tensor, bbox: List[float]) -> torch.Tensor:
    """
    Clips a 3D polygon against an axis-aligned bounding box using a Sutherland–Hodgman-like approach.

    Args:
        polygon (torch.Tensor): A tensor of shape (N, 3), representing vertices of the input polygon.
        bbox (List[float]): The bounding box specified as [xmin, xmax, ymin, ymax, zmin, zmax].

    Returns:
        torch.Tensor: A tensor of shape (M, 3), representing the vertices of the clipped polygon.
                      If there is no intersection, returns an empty tensor of shape (0, 3).
    """
    # Unpack bounding box values
    xmin, xmax, ymin, ymax, zmin, zmax = bbox

    # Define the six clipping planes:
    #   (plane_value, axis_index, keep_greater_or_equal)
    #   e.g., (xmin, 0, True) means we keep the region where x >= xmin
    planes = [
        (xmin, 0, True),  # x >= xmin
        (xmax, 0, False),  # x <= xmax
        (ymin, 1, True),  # y >= ymin
        (ymax, 1, False),  # y <= ymax
        (zmin, 2, True),  # z >= zmin
        (zmax, 2, False),  # z <= zmax
    ]

    # Initialize the polygon to be clipped
    clipped_polygon = polygon.clone()

    # Iteratively clip against each plane
    for plane_val, axis, keep_greater in planes:
        if clipped_polygon.shape[0] == 0:
            # If there are no vertices, the polygon is fully clipped
            break

        new_polygon = []
        num_points = clipped_polygon.shape[0]

        # Traverse each edge of the current polygon
        for i in range(num_points):
            p1 = clipped_polygon[i]
            p2 = clipped_polygon[(i + 1) % num_points]

            # Check whether a point is inside or outside the plane
            # keep_greater = True  --> keep p[axis] >= plane_val
            # keep_greater = False --> keep p[axis] <= plane_val
            if keep_greater:
                p1_inside = p1[axis] >= plane_val
                p2_inside = p2[axis] >= plane_val
            else:
                p1_inside = p1[axis] <= plane_val
                p2_inside = p2[axis] <= plane_val

            # Sutherland–Hodgman clipping logic:
            # 1) p1 and p2 both inside --> keep p2
            if p1_inside and p2_inside:
                new_polygon.append(p2)
            # 2) p1 inside, p2 outside --> keep intersection
            elif p1_inside and not p2_inside:
                t = (plane_val - p1[axis]) / (p2[axis] - p1[axis])
                inter_point = p1 + t * (p2 - p1)
                new_polygon.append(inter_point)
            # 3) p1 outside, p2 inside --> keep intersection, then p2
            elif not p1_inside and p2_inside:
                t = (plane_val - p1[axis]) / (p2[axis] - p1[axis])
                inter_point = p1 + t * (p2 - p1)
                new_polygon.append(inter_point)
                new_polygon.append(p2)
            # 4) p1 and p2 both outside --> discard this edge

        # Update the polygon after clipping against this plane
        if len(new_polygon) > 0:
            clipped_polygon = torch.stack(new_polygon, dim=0)
        else:
            # If empty, no intersection remains
            clipped_polygon = torch.empty((0, 3))
            break

    return clipped_polygon


if __name__ == '__main__':
    # domain = pyrfm.Line1D(0, 10)
    # model = pyrfm.RFMBase(dim=1, n_hidden=100, n_subdomains=4, pou=pyrfm.PsiA, domain=domain)
    # print(model.centers)
    # voronoi = pyrfm.Voronoi(domain, centers=model.centers)
    # print(voronoi.vertices)

    # domain = Heart()
    domain = pyrfm.Cube3D(center=[0, 0, 0], radius=[1, 1, 1])
    model = pyrfm.RFMBase(dim=3, n_hidden=100, n_subdomains=2, pou=pyrfm.PsiA, domain=domain)
    voronoi = pyrfm.Voronoi(domain, centers=model.centers)
    # print(_polygon_box_intersection(voronoi.vertices[voronoi.ridge_vertices[0]], domain.get_bounding_box()))
    torch.set_printoptions(profile="full")
    print(voronoi.interface_sample(1000)[1])

    # points = voronoi.points
    # print(points)
    # bounding_box = torch.tensor(domain.get_bounding_box()).view(-1, 2)
    # D = (bounding_box[:, 1] - bounding_box[:, 0]).norm(p=2)
    # center = bounding_box.mean(dim=1).view(1, domain.dim)
    # alpha = 2 * domain.dim
    # R = D * alpha
    # virtual_points = [center + R * torch.eye(domain.dim)[i] for i in range(domain.dim)]
    # virtual_points.append(center - R * torch.ones(domain.dim))
    # print(center)
    # print(torch.cat(virtual_points, dim=0))
    #
    # centers = torch.cat([points, torch.cat(virtual_points, dim=0)], dim=0)
    # voronoi = pyrfm.Voronoi(domain, centers=centers)
    #
    # n_domain = 8
    # vertices = voronoi.vertices
    # ridge_points = voronoi.ridge_points
    # ridge_vertices = voronoi.ridge_vertices
    # mask = []
    # for i in range(len(ridge_points)):
    #     for j in range(len(ridge_points[i])):
    #         if ridge_points[i][j] >= n_domain:
    #             mask.append(i)
    #             break
    #
    # ridge_vertices = [ridge_vertices[i] for i in range(len(ridge_vertices)) if i not in mask]
    # ridge_points = [[item for item in ridge_points[i]] for i in range(len(ridge_points)) if i not in mask]
    #
    # print(ridge_vertices)
    # print(ridge_points)
    # print(len(ridge_vertices))
    # print(len(ridge_points))
    #
    # polygon = torch.cat([vertices[ridge_vertices[0][i]].view(1, -1) for i in range(len(ridge_vertices[0]))], dim=0)
    # print(polygon)
    # polygon = _polygon_box_intersection(polygon, domain.get_bounding_box())
    # print(polygon)
    #
    # polygon = pyrfm.Polygon3D(polygon)
    # print(polygon.in_sample(1000, with_boundary=True))

    # print(vertices[ridge_vertices[0][0]], vertices[ridge_vertices[0][1]], vertices[ridge_vertices[0][2]],
    #       vertices[ridge_vertices[0][3]])
    # voronoi_plot_2d(voronoi.voronoi_, show_points=True, show_vertices=True, line_alpha=0.6)
    # p = domain.in_sample(1000)
    # plt.scatter(x=p[:, 0], y=p[:, 1])
    # plt.xlim(-1.5, 1.5)
    # plt.ylim(-1.5, 1.5)
    # plt.show()
