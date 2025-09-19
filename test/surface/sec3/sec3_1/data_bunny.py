# -*- coding: utf-8 -*-
"""
Created on 2025/9/18

@author: Yifei Sun
"""
import warnings
from math import inf
import logging
import sys
from typing import Literal, SupportsFloat, Union, List, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
import shutil
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader  # noqa: F401
from scipy.spatial import cKDTree
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import point_cloud_utils as pcu
import open3d as o3d
from matplotlib.colors import TwoSlopeNorm

import math  # noqa: F401
import time  # ★ 新增

from torch.optim.lr_scheduler import LRScheduler
from typing_extensions import override
from torch.optim import Optimizer

# 配置/解析
import yaml
import argparse

# 可选 3D 渲染（pyrfm）
import pyrfm


# ---------------------- 法向处理 ----------------------
@dataclass
class NormalParams:
    radius: float = 0.02
    max_nn: int = 30
    orient_k: int = 20  # <=0 不做一致化


def _to_o3d_mesh(v: np.ndarray, f: np.ndarray) -> o3d.geometry.TriangleMesh:
    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(v.astype(np.float64))
    m.triangles = o3d.utility.Vector3iVector(f.astype(np.int32))
    return m


def _o3d_clean(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    return mesh


class PLYNormalHelper:
    def __init__(self, path: str | Path, params: NormalParams = NormalParams(), remeshing=False):
        self.path = Path(path);
        self.params = params
        self._pcd: o3d.geometry.PointCloud | None = None
        self._mesh: o3d.geometry.TriangleMesh | None = None
        self._loaded_type: str | None = None  # "pointcloud" | "mesh"
        self._changed: bool = False
        self.load(remeshing=remeshing)

    def load(self, remeshing=False) -> "PLYNormalHelper":
        pcd = o3d.io.read_point_cloud(str(self.path))
        mesh = o3d.io.read_triangle_mesh(str(self.path))

        if remeshing and len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
            v, f = pcu.make_mesh_watertight(np.asarray(mesh.vertices), np.asarray(mesh.triangles), 40000)
            mesh = _to_o3d_mesh(v, f);
            mesh = _o3d_clean(mesh)

        if len(pcd.points) > 0 and (len(mesh.vertices) == 0 or len(mesh.triangles) == 0):
            self._pcd, self._mesh, self._loaded_type = pcd, None, "pointcloud"
        elif len(mesh.vertices) > 0 and len(mesh.triangles) > 0:
            self._pcd, self._mesh, self._loaded_type = None, mesh, "mesh"
        else:
            raise ValueError("无法识别为有效的点云或网格")
        return self

    def ensure_normals(self) -> "PLYNormalHelper":
        self._require_loaded()
        if self._loaded_type == "pointcloud":
            if self._ensure_pointcloud_normals(self._pcd): self._changed = True
        else:
            rv, rf = self._ensure_mesh_normals(self._mesh)
            if rv or rf: self._changed = True
        return self

    def get_points_and_normals(self, stack: bool = False) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        self._require_loaded()
        if self._loaded_type == "pointcloud":
            pts = np.asarray(self._pcd.points);
            nrm = np.asarray(self._pcd.normals)
        else:
            pts = np.asarray(self._mesh.vertices);
            nrm = np.asarray(self._mesh.vertex_normals)

        if self._invalid_normals(nrm) or (nrm.shape[0] != pts.shape[0]):
            raise RuntimeError("缺少有效法向，请先调用 ensure_normals()")
        return np.hstack([pts, nrm]) if stack else (pts, nrm)

    def visualize(self):
        self._require_loaded()
        geo = self._pcd if self._loaded_type == "pointcloud" else self._mesh
        o3d.visualization.draw_geometries([geo])

    def save(self, suffix: str = "_with_normals") -> Path | None:
        out = self.path.with_name(self.path.stem + suffix + ".ply")
        if self._loaded_type == "pointcloud":
            o3d.io.write_point_cloud(str(out), self._pcd)
        else:
            o3d.io.write_triangle_mesh(str(out), self._mesh)
        print(f"文件已保存：{out}");
        return out

    @staticmethod
    def _invalid_normals(arr: np.ndarray) -> bool:
        if arr.size == 0: return True
        if np.isnan(arr).any(): return True
        if np.allclose(arr, 0): return True
        return False

    def _ensure_pointcloud_normals(self, pcd: o3d.geometry.PointCloud) -> bool:
        need = (not pcd.has_normals())
        if not need:
            normals = np.asarray(pcd.normals)
            need = self._invalid_normals(normals) or (normals.shape[0] != np.asarray(pcd.points).shape[0])
        if not need: return False
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=self.params.radius, max_nn=self.params.max_nn))
        if self.params.orient_k > 0:
            pcd.orient_normals_consistent_tangent_plane(k=self.params.orient_k)
        return True

    def _ensure_mesh_normals(self, mesh: o3d.geometry.TriangleMesh) -> tuple[bool, bool]:
        recomputed_v = False;
        recomputed_f = False
        if not mesh.has_vertex_normals():
            recomputed_v = True
        else:
            vn = np.asarray(mesh.vertex_normals)
            if self._invalid_normals(vn) or (vn.shape[0] != np.asarray(mesh.vertices).shape[0]): recomputed_v = True
        if recomputed_v: mesh.compute_vertex_normals()

        if not mesh.has_triangle_normals():
            recomputed_f = True
        else:
            fn = np.asarray(mesh.triangle_normals)
            if self._invalid_normals(fn) or (fn.shape[0] != np.asarray(mesh.triangles).shape[0]): recomputed_f = True
        if recomputed_f: mesh.compute_triangle_normals()
        return recomputed_v, recomputed_f

    def _require_loaded(self):
        if self._loaded_type is None: raise RuntimeError("请先调用 load()")


if __name__ == "__main__":
    torch.set_default_device('cuda') if torch.cuda.is_available() else torch.set_default_device('cpu')

    ply_path = '../../data/bunny.ply'
    remeshing = True

    # 读取 & 法向
    helper = PLYNormalHelper(ply_path, remeshing=remeshing)
    helper.ensure_normals()
    x, normal = helper.get_points_and_normals()
    print(f"点数量: {x.shape[0]}, 法向数量: {normal.shape[0]}")

    x, normal = torch.tensor(x), torch.tensor(normal)
    mean_curvature = torch.zeros(x.shape[0], 1)  # 占位

    # save the points, normals, and mean curvature in a file
    torch.save((x, normal, mean_curvature), '../../data/bunny_in.pth')

    # # load the points, normals, and mean curvature from the file
    x, normal, mean_curvature = torch.load('../../data/bunny_in.pth', map_location=torch.tensor(0.).device)

    print(x.shape, normal.shape, mean_curvature.shape, x.dtype, x.device)
