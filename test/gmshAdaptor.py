# -*- coding: utf-8 -*-
"""
Created on 2025/9/27

@author: Yifei Sun
"""
from typing import Union, Tuple, Dict, List

import torch
import numpy as np

import pyrfm


class GmshAdaptor(pyrfm.GeometryBase):

    def __init__(self, msh_path: str):
        import meshio
        # 1) read mesh
        self.mesh = meshio.read(msh_path)
        self.coord_dim = self.mesh.points.shape[1]
        self.topo_dim = self._infer_topo_dim_from_cells(self.mesh)
        super().__init__(dim=self.topo_dim, intrinsic_dim=self.topo_dim)
        # 3) points
        self.points = self.mesh.points.astype(np.float64)
        self.points_torch = torch.as_tensor(self.points, dtype=self.dtype, device=self.device)
        # 4) boundary mask/indices
        self.boundary_vertex_mask = self._find_boundary_vertices_from_cells(self.mesh, self.topo_dim)
        self.boundary_vertex_idx = np.nonzero(self.boundary_vertex_mask)[0]
        all_idx = np.arange(len(self.points))
        self.interior_vertex_idx = all_idx[~self.boundary_vertex_mask]
        # 5) boundary normals (only when we can compute reliably)
        self.boundary_normals = None
        self.boundary_normals_torch = None
        cells_d = self._cells_dict(self.mesh)
        # choose triangle faces for normals (surface)
        tri_faces = None
        if self.topo_dim == 3:
            # surface triangles may be present in mesh cells
            if 'triangle' in cells_d:
                tri_faces = cells_d['triangle']
            else:
                # derive from tets: faces with multiplicity 1 are boundary
                tets = cells_d.get('tetra')
                if tets is not None and self.coord_dim == 3:
                    faces = np.vstack([
                        tets[:, [0, 1, 2]],
                        tets[:, [0, 1, 3]],
                        tets[:, [0, 2, 3]],
                        tets[:, [1, 2, 3]],
                    ])
                    # unique rows regardless of orientation
                    faces_sorted = np.sort(faces, axis=1)
                    uniq, counts = np.unique(faces_sorted, axis=0, return_counts=True)
                    tri_faces = uniq[counts == 1]
        elif self.topo_dim == 2:
            # normals for surface triangles themselves (not 1D boundary normals)
            tri_faces = cells_d.get('triangle')
        # compute normals if possible
        if tri_faces is not None and self.coord_dim == 3:
            normals = self._compute_vertex_normals(self.points, tri_faces)
            if normals is not None:
                self.boundary_normals = normals
                # boundary vertex normals subset
                if self.boundary_vertex_idx.size > 0:
                    self.boundary_normals_torch = torch.as_tensor(normals[self.boundary_vertex_idx],
                                                                  dtype=self.dtype)
        # 6) torch views for outputs
        idx_b = torch.as_tensor(self.boundary_vertex_idx, dtype=torch.long, device=self.points_torch.device)
        idx_i = torch.as_tensor(self.interior_vertex_idx, dtype=torch.long, device=self.points_torch.device)
        self._boundary_points_torch = self.points_torch[idx_b]
        self._interior_points_torch = self.points_torch[idx_i]

    def in_sample(self, num_samples: int = None, with_boundary: bool = False) -> torch.Tensor:
        """返回内点；如果 with_boundary=True，则将边界点也包含在内。忽略 num_samples。"""
        if with_boundary:
            if self._interior_points_torch.numel() == 0:
                return self._boundary_points_torch
            return torch.vstack([self._interior_points_torch, self._boundary_points_torch])
        return self._interior_points_torch

    def on_sample(self, num_samples: int = None, with_normal: bool = False) -> Union[
        torch.Tensor, Tuple[torch.Tensor, ...]]:
        """返回边界点；如果 with_normal=True，且可计算法向，则同时返回法向量。忽略 num_samples。"""
        if not with_normal or self.boundary_normals_torch is None:
            return self._boundary_points_torch
        return self._boundary_points_torch, self.boundary_normals_torch

    def get_bounding_box(self) -> List[float]:
        pts = self.points
        # ensure 3D shape
        if pts.shape[1] == 2:
            zmin = zmax = 0.0
            xmin, ymin = pts.min(axis=0)
            xmax, ymax = pts.max(axis=0)
        else:
            xmin, ymin, zmin = pts.min(axis=0)
            xmax, ymax, zmax = pts.max(axis=0)
        return [float(xmin), float(xmax), float(ymin), float(ymax), float(zmin), float(zmax)]

    def sdf(self, p: torch.Tensor) -> torch.Tensor:
        """Return (unsigned) distance from points p to the mesh.
        p: (N, D) tensor, D can be 2 or 3. If 2, z is assumed 0.
        Note: This is an unsigned distance. Sign is not inferred.
        """
        if p.ndim != 2:
            raise ValueError("p must be a 2D tensor of shape (N, D)")
        device = p.device
        # Ensure (N,3)
        if p.shape[1] == 2:
            p3 = torch.cat([p, torch.zeros((p.shape[0], 1), dtype=p.dtype, device=device)], dim=1)
        elif p.shape[1] == 3:
            p3 = p
        else:
            raise ValueError("sdf currently supports D in {2,3}")

        cells = self._cells_dict(self.mesh)
        # Prefer triangles if available
        if 'triangle' in cells and self.points.shape[1] >= 2:
            tri = cells['triangle']
            a = torch.as_tensor(self.points[tri[:, 0]], dtype=p3.dtype, device=device)
            b = torch.as_tensor(self.points[tri[:, 1]], dtype=p3.dtype, device=device)
            c = torch.as_tensor(self.points[tri[:, 2]], dtype=p3.dtype, device=device)
            # Ensure 3D for computation
            if a.shape[1] == 2:
                z = torch.zeros((a.shape[0], 1), dtype=p3.dtype, device=device)
                a = torch.cat([a, z], dim=1)
                b = torch.cat([b, z], dim=1)
                c = torch.cat([c, z], dim=1)
            dists = self._pointset_to_triangles_distance(p3, a, b, c)
            return dists
        # Fall back to line segments
        if 'line' in cells:
            seg = cells['line']
            a = torch.as_tensor(self.points[seg[:, 0]], dtype=p3.dtype, device=device)
            b = torch.as_tensor(self.points[seg[:, 1]], dtype=p3.dtype, device=device)
            if a.shape[1] == 2:
                z = torch.zeros((a.shape[0], 1), dtype=p3.dtype, device=device)
                a = torch.cat([a, z], dim=1)
                b = torch.cat([b, z], dim=1)
            dists = self._pointset_to_segments_distance(p3, a, b)
            return dists
        # Last resort: distance to vertices
        verts = torch.as_tensor(self.points, dtype=p3.dtype, device=device)
        if verts.shape[1] == 2:
            z = torch.zeros((verts.shape[0], 1), dtype=p3.dtype, device=device)
            verts = torch.cat([verts, z], dim=1)
        # compute min Euclidean distance to all vertices
        # (N,1,3) - (1,M,3) -> (N,M,3)
        diff = p3[:, None, :] - verts[None, :, :]
        d2 = (diff * diff).sum(dim=2)
        return torch.sqrt(d2.min(dim=1).values)

    # ===================== helpers inside class =====================
    @staticmethod
    def _cells_dict(mesh) -> Dict[str, np.ndarray]:
        d: Dict[str, np.ndarray] = {}
        for block in mesh.cells:
            d[block.type] = block.data
        return d

    def _find_boundary_vertices_from_cells(self, mesh, topo_dim: int) -> np.ndarray:
        """返回一个布尔数组，标记每个顶点是否为边界顶点。"""
        n_pts = mesh.points.shape[0]
        mask = np.zeros(n_pts, dtype=bool)
        cd = self._cells_dict(mesh)

        if topo_dim == 3:
            tri = cd.get('triangle')
            if tri is None and 'tetra' in cd:
                tets = cd['tetra']
                faces = np.vstack([
                    tets[:, [0, 1, 2]],
                    tets[:, [0, 1, 3]],
                    tets[:, [0, 2, 3]],
                    tets[:, [1, 2, 3]],
                ])
                faces_sorted = np.sort(faces, axis=1)
                uniq, counts = np.unique(faces_sorted, axis=0, return_counts=True)
                tri = uniq[counts == 1]
            if tri is not None:
                mask[np.unique(tri)] = True
            return mask

        if topo_dim == 2:
            line = cd.get('line')
            if line is not None:
                mask[np.unique(line)] = True
                return mask
            tri = cd.get('triangle')
            if tri is not None:
                edges = np.vstack([
                    tri[:, [0, 1]],
                    tri[:, [1, 2]],
                    tri[:, [2, 0]],
                ])
                edges_sorted = np.sort(edges, axis=1)
                uniq, counts = np.unique(edges_sorted, axis=0, return_counts=True)
                boundary_edges = uniq[counts == 1]
                mask[np.unique(boundary_edges)] = True
            return mask

        if topo_dim == 1:
            line = cd.get('line')
            if line is not None:
                deg = np.zeros(n_pts, dtype=int)
                for a, b in line:
                    deg[a] += 1
                    deg[b] += 1
                mask[deg == 1] = True
            return mask

        return mask

    @staticmethod
    def _compute_vertex_normals(points: np.ndarray, faces: np.ndarray):
        """对三角面片计算顶点法向（面积加权平均）。仅当 points 为 3D 时可用。"""
        if points.shape[1] != 3 or faces is None or len(faces) == 0:
            return None
        normals = np.zeros((points.shape[0], 3), dtype=np.float64)
        p = points
        f = faces
        v1 = p[f[:, 1]] - p[f[:, 0]]
        v2 = p[f[:, 2]] - p[f[:, 0]]
        fn = np.cross(v1, v2)
        for i in range(3):
            np.add.at(normals, f[:, i], fn)
        lens = np.linalg.norm(normals, axis=1, keepdims=True)
        lens[lens == 0.0] = 1.0
        normals /= lens
        return normals

    def _infer_topo_dim_from_cells(self, mesh) -> int:
        has3 = has2 = has1 = has0 = False
        for block in mesh.cells:
            ct = block.type.lower()
            if ct.startswith(("tetra", "hexahedron", "wedge", "pyramid", "polyhedron")):
                has3 = True
            elif ct.startswith(("triangle", "quad", "polygon")):
                has2 = True
            elif ct.startswith(("line", "edge")) or ct in ("line",):
                has1 = True
            elif ct in ("vertex", "point"):
                has0 = True
        if has3:
            return 3
        if has2:
            return 2
        if has1:
            return 1
        if has0:
            return 0
        # fallback: if there are points only but no explicit cell blocks
        return 0

    @staticmethod
    def _pointset_to_segments_distance(P: torch.Tensor, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Minimal distance from point set P (N,3) to segments AB (M,3).
        Returns (N,) distances (unsigned)."""
        # Vector math adapted from closest-point on segment
        AB = B - A  # (M,3)
        AP = P[:, None, :] - A[None, :, :]  # (N,M,3)
        AB_len2 = (AB * AB).sum(dim=1)  # (M,)
        # Avoid div by zero for degenerate segments
        AB_len2 = torch.clamp(AB_len2, min=1e-30)
        t = (AP * AB[None, :, :]).sum(dim=2) / AB_len2[None, :]  # (N,M)
        t = torch.clamp(t, 0.0, 1.0)
        closest = A[None, :, :] + t[:, :, None] * AB[None, :, :]  # (N,M,3)
        d2 = ((P[:, None, :] - closest) ** 2).sum(dim=2)  # (N,M)
        return torch.sqrt(d2.min(dim=1).values)

    @staticmethod
    def _pointset_to_triangles_distance(P: torch.Tensor, A: torch.Tensor, B: torch.Tensor,
                                        C: torch.Tensor) -> torch.Tensor:
        """Minimal distance from point set P (N,3) to triangles ABC (M,3).
        Uses algorithm from 'Real-Time Collision Detection' (Christer Ericson)."""
        # Expand for vectorized edge tests
        # Shapes: P(N,3), A/B/C(M,3) -> broadcast to (N,M,3)
        PA = P[:, None, :] - A[None, :, :]
        PB = P[:, None, :] - B[None, :, :]
        PC = P[:, None, :] - C[None, :, :]
        AB = B[None, :, :] - A[None, :, :]
        AC = C[None, :, :] - A[None, :, :]
        BC = C[None, :, :] - B[None, :, :]
        CA = A[None, :, :] - C[None, :, :]

        # Compute normals for plane projection
        N = torch.cross(AB, AC, dim=2)  # (N,M,3) but AB/AC are (1,M,3) so it broadcasts to (N,M,3)
        N_len2 = (N * N).sum(dim=2)
        # Handle degenerate triangles by small epsilon
        eps = 1e-30
        N_len2 = torch.clamp(N_len2, min=eps)

        # Region tests via barycentric technique
        # Compute signed areas / barycentrics
        # v = dot(N, cross(AB, AP)) / |N|^2 etc.
        # But we can use the projection method: project P onto triangle plane and then check if inside using edge tests.
        # Edge tests: sign of dot(N, cross(edge, vec_to_point))
        # Project P onto plane of triangle
        # distance along normal:
        dist_plane = ((PA * N).sum(dim=2)) / torch.sqrt(N_len2)  # (N,M) signed distance to plane magnitude
        # Closest point on plane
        proj = P[:, None, :] - ((PA * N).sum(dim=2) / N_len2)[:, :, None] * N  # (N,M,3)

        # Edge inside tests for proj
        # For edge AB
        C1 = torch.cross(AB, proj - A[None, :, :], dim=2)
        C2 = torch.cross(BC, proj - B[None, :, :], dim=2)
        C3 = torch.cross(CA, proj - C[None, :, :], dim=2)
        s1 = (C1 * N).sum(dim=2) >= 0
        s2 = (C2 * N).sum(dim=2) >= 0
        s3 = (C3 * N).sum(dim=2) >= 0
        inside = s1 & s2 & s3  # (N,M)

        # Distance if inside: |dist to plane|
        d_inside = torch.abs(dist_plane)

        # If outside, distance is min to triangle edges
        d_edge_ab = GmshAdaptor._pointset_to_segments_distance(P, A, B)  # (N,)
        d_edge_bc = GmshAdaptor._pointset_to_segments_distance(P, B, C)  # (N,)
        d_edge_ca = GmshAdaptor._pointset_to_segments_distance(P, C, A)  # (N,)
        d_outside = torch.min(torch.min(d_edge_ab, d_edge_bc), d_edge_ca)  # (N,)

        # Combine per triangle: for each P, per M triangle distance
        # Build full per-triangle distances: if inside[k,m] -> d_inside[k,m], else -> d_outside[k]
        d_full = torch.where(inside, d_inside, d_outside[:, None])  # (N,M)
        return d_full.min(dim=1).values


# —— 示例用法 ——
if __name__ == "__main__":
    domain = GmshAdaptor("circle_2d.msh")
    x_in = domain.in_sample()
    x_on = domain.on_sample()

    print(x_in)
