# -*- coding: utf-8 -*-
"""
Created on 2024/12/23

@author: Yifei Sun
"""
from typing import Tuple

from torch import Tensor

from .geometry import GeometryBase
from .utils import *

from scipy.spatial import Voronoi as SciVoronoi
from scipy.cluster.vq import kmeans as Kmeans


def _line_box_intersection(point: torch.Tensor,
                           direction: torch.Tensor,
                           bbox: List[float]) -> Optional[torch.Tensor]:
    """
    Computes the intersection between a parametric line x(t) = point + direction * t
    and a given 2D bounding box (xmin, xmax, ymin, ymax).

    Args:
        point (torch.Tensor): A 2D point (x0, y0) on the line.
        direction (torch.Tensor): A 2D direction vector (dx, dy).
        bbox (List[float]): The bounding box given as [xmin, xmax, ymin, ymax].

    Returns:
        Optional[torch.Tensor]: A 2 x 2 Tensor containing two intersection points
        with the bounding box, or None if there is no valid intersection.
        Each row represents one intersection point.
    """
    xmin, xmax, ymin, ymax = bbox
    p = point.clone().float()
    d = direction.clone().float()

    eps = 1e-14
    t_min = -float('inf')
    t_max = float('inf')

    # Check intersection in x-direction
    if abs(d[0]) < eps:
        # If direction has no x component and the current x is out of [xmin, xmax], no intersection
        if not (xmin <= p[0] <= xmax):
            return None
    else:
        t1 = (xmin - p[0]) / d[0]
        t2 = (xmax - p[0]) / d[0]
        t_low, t_high = min(t1, t2), max(t1, t2)
        t_min = max(t_min, t_low)
        t_max = min(t_max, t_high)
        if t_max < t_min:
            return None

    # Check intersection in y-direction
    if abs(d[1]) < eps:
        if not (ymin <= p[1] <= ymax):
            return None
    else:
        t1 = (ymin - p[1]) / d[1]
        t2 = (ymax - p[1]) / d[1]
        t_low, t_high = min(t1, t2), max(t1, t2)
        t_min = max(t_min, t_low)
        t_max = min(t_max, t_high)
        if t_max < t_min:
            return None

    # Two intersection points: parametric values t_min and t_max
    p1 = p + d * t_min
    p2 = p + d * t_max
    return torch.stack([p1, p2], dim=0)


def _sample_segment(p1: torch.Tensor, p2: torch.Tensor, n_samples: int) -> torch.Tensor:
    """
    Samples points on the line segment between p1 and p2 using linear interpolation.

    Args:
        p1 (torch.Tensor): The starting point of the segment, shape (2,).
        p2 (torch.Tensor): The ending point of the segment, shape (2,).
        n_samples (int): Number of sample points to generate along [p1, p2].

    Returns:
        torch.Tensor: A tensor of shape (n_samples, 2), containing the sampled coordinates.
    """
    ts = torch.linspace(0., 1., n_samples)
    seg = p1.unsqueeze(0) + (p2 - p1).unsqueeze(0) * ts.unsqueeze(1)
    return seg


class Voronoi:
    def __init__(self, domain: GeometryBase, centers: Optional[torch.Tensor] = None, k: Optional[int] = None):
        """
        The Voronoi diagram class.

        Args:
            domain: The domain of the Voronoi diagram.
            centers: The initial centers of the Voronoi diagram.
            k: The number of centers if the initial centers are not provided.

        Attributes:
            points: The coordinates of the input points.
            vertices: The coordinates of the Voronoi vertices.
            ridge_points: Indices of the points between which each Voronoi ridge lies.
            ridge_vertices: Indices of the Voronoi vertices forming each Voronoi ridge.
            regions: Indices of the Voronoi vertices forming each Voronoi region. -1 indicates vertex outside the Voronoi diagram.
            point_region: Index of the Voronoi region for each input point.
        """
        self.domain = domain

        if centers is not None:
            if isinstance(centers, torch.Tensor):
                self.points: torch.Tensor = centers.view(-1, 2)
            else:
                self.points: torch.Tensor = torch.tensor(centers)
        else:
            dim = domain.dim
            if k is None:
                k = 2 ** dim
            samples = domain.in_sample(int(20 ** dim), with_boundary=True)
            self.points = torch.tensor(Kmeans(samples.cpu().numpy(), int(k))[0])

        if domain.dim == 2:
            voronoi_ = SciVoronoi(self.points.cpu().numpy())
        else:
            raise NotImplementedError("Only 2D Voronoi diagrams are supported.")
        self.voronoi_ = voronoi_
        self.vertices: torch.Tensor = torch.tensor(voronoi_.vertices)
        self.ridge_points: List[List[int]] = voronoi_.ridge_points
        self.ridge_vertices: List[List[int]] = voronoi_.ridge_vertices
        self.regions: List[List[int]] = voronoi_.regions
        self.point_region: List[List[int]] = voronoi_.point_region

    def interface_sample(self, num_samples: int) -> Tuple[Dict, Tensor]:
        """
        Samples points on the Voronoi interface (the boundaries between Voronoi cells).
        For each ridge (edge) in the Voronoi diagram:
          - If both vertices are valid (no -1), sample on the finite line segment.
          - If one vertex is -1, treat the edge as a ray and intersect with the domain boundary.
          - If both vertices are -1, treat it as an infinite line and intersect with the domain boundary.

        Args:
            num_samples (int): Number of sample points to generate on each ridge segment.

        Returns:
            torch.Tensor: A collection of all sampled points on the Voronoi interfaces,
                          concatenated into a single tensor of shape (M, 2), where M
                          depends on how many ridges intersect the domain.
        """
        bbox = self.domain.get_bounding_box()  # [xmin, xmax, ymin, ymax]

        interface_points = []
        region_pairs = []
        num_samples = max(int(num_samples / len(self.ridge_points)), 3)  # Ensure at least two points per segment

        center = torch.mean(self.points, dim=0)
        # Loop over each ridge in the Voronoi diagram
        for ridge_idx, (v1_idx, v2_idx) in enumerate(self.ridge_vertices):
            # Retrieve the two generator points that gave rise to this ridge
            c1_idx, c2_idx = self.ridge_points[ridge_idx]
            region_pairs.append((c1_idx, c2_idx))
            p1 = self.points[c1_idx]  # (2,)
            p2 = self.points[c2_idx]  # (2,)

            # Case 1: Both vertices are inside the diagram (no -1), a finite line segment
            if v1_idx >= 0 and v2_idx >= 0:
                v1 = self.vertices[v1_idx]
                v2 = self.vertices[v2_idx]
                seg_pts = _sample_segment(v1, v2, num_samples)
                interface_points.append(seg_pts)


            # Case 2: Exactly one vertex is -1 => half-infinite ray
            elif (v1_idx == -1 and v2_idx >= 0) or (v2_idx == -1 and v1_idx >= 0):
                if v1_idx == -1:
                    finite_v_idx = v2_idx
                else:
                    finite_v_idx = v1_idx
                v_finite = self.vertices[finite_v_idx]

                # Compute the direction of the perpendicular bisector (p1, p2)
                mid = 0.5 * (p1 + p2)
                dp = p2 - p1  # (dx, dy)
                dir_candidate = torch.tensor([dp[1], -dp[0]], dtype=p1.dtype)

                # Determine direction sign so that it points from v_finite outward
                if torch.dot(mid - center, dir_candidate) < 0:
                    dir_candidate = -dir_candidate

                # Intersect ray with bounding box
                inter = _line_box_intersection(v_finite, dir_candidate, bbox)
                if inter is not None:
                    # We have up to two intersection points
                    d2 = torch.sum(dir_candidate ** 2).item()
                    t_candidates = []
                    for ip in inter:
                        delta = ip - v_finite
                        t_val = torch.dot(delta, dir_candidate) / d2
                        t_candidates.append((t_val.item(), ip))

                    # Only keep intersection points where t >= 0 (forward along the ray)
                    valid_ts = [x for x in t_candidates if x[0] >= 0]

                    if len(valid_ts) == 0:
                        # No valid intersection in t>=0
                        continue
                    elif len(valid_ts) == 1:
                        # Ray from v_finite to that single intersection
                        _, p_int = valid_ts[0]
                        seg_pts = _sample_segment(v_finite, p_int, num_samples)
                        interface_points.append(seg_pts)
                    else:
                        # Two intersections in t>=0 => take the farthest
                        valid_ts.sort(key=lambda x: x[0])
                        _, p_int_max = valid_ts[-1]
                        seg_pts = _sample_segment(v_finite, p_int_max, num_samples)
                        interface_points.append(seg_pts)

            # Case 3: Both vertices are -1 => infinite line
            elif v1_idx == -1 and v2_idx == -1:
                mid = 0.5 * (p1 + p2)
                dp = p2 - p1
                dir_ = torch.tensor([dp[1], -dp[0]], dtype=p1.dtype)

                # Intersect the entire line x(t) = mid + t * dir with bounding box
                inter = _line_box_intersection(mid, dir_, bbox)
                if inter is not None:
                    p_int1, p_int2 = inter[0], inter[1]
                    seg_pts = _sample_segment(p_int1, p_int2, num_samples)
                    interface_points.append(seg_pts)
                else:
                    continue

        # Combine all sampled interface points
        if len(interface_points) == 0:
            return {}, torch.empty((0, 2), dtype=self.points.dtype)

        interface_dict = {}
        filtered_pts_list = []
        for (pair, points) in zip(region_pairs, interface_points):
            # remove points out of domain (that is sdf > 0)
            points = points[torch.where(self.domain.sdf(points) < 0)[0]]
            interface_dict[pair] = points
            filtered_pts_list.append(points)
        all_pts = torch.cat(filtered_pts_list, dim=0)
        return interface_dict, all_pts
