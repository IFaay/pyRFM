from typing import List, Union, Tuple
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import open3d as o3d
from scipy.spatial import cKDTree
import point_cloud_utils as pcu

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


# ===================== 可视化 =====================
def plot_model_slice(model,
                     bbox: List[float],
                     axis: str = 'z',
                     value: float = 0.0,
                     res: int = 256,
                     level: float = 0.0,
                     cmap: str = 'RdBu_r',
                     vmin: float = None,
                     vmax: float = None,
                     symmetric: bool = True,
                     device: Union[torch.device, str, None] = None):
    """
    在给定轴向的截面上可视化 model(x,y,z) 的标量场，并在 level 映射白色。
    """
    assert axis in ('x', 'y', 'z')
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    xr, yr, zr = (bbox[0], bbox[1]), (bbox[2], bbox[3]), (bbox[4], bbox[5])

    if axis == 'x':
        value = float(np.clip(value, *xr))
    elif axis == 'y':
        value = float(np.clip(value, *yr))
    else:
        value = float(np.clip(value, *zr))

    axis2idx = {'x': 0, 'y': 1, 'z': 2}
    fixed = axis2idx[axis]
    free = [i for i in range(3) if i != fixed]
    ranges = [xr, yr, zr]

    u = torch.linspace(ranges[free[0]][0], ranges[free[0]][1], res, device=device)
    v = torch.linspace(ranges[free[1]][0], ranges[free[1]][1], res, device=device)
    U, V = torch.meshgrid(u, v, indexing='ij')

    P = torch.zeros((res * res, 3), device=device)
    P[:, free[0]] = U.reshape(-1)
    P[:, free[1]] = V.reshape(-1)
    P[:, fixed] = value

    with torch.no_grad():
        out = model(P)
        if out.ndim == 2 and out.size(-1) == 1:
            out = out.squeeze(-1)
        Z = out.reshape(res, res).detach().cpu().numpy()

    data_min, data_max = Z.min(), Z.max()
    if symmetric:
        bound = max(abs(data_min), abs(data_max))
        data_vmin, data_vmax = -bound, bound
    else:
        data_vmin = data_min if vmin is None else vmin
        data_vmax = data_max if vmax is None else vmax

    if not (data_vmin < level < data_vmax):
        pad = 1e-6 * max(1.0, abs(level))
        data_vmin = min(data_vmin, level - pad)
        data_vmax = max(data_vmax, level + pad)

    norm = TwoSlopeNorm(vmin=data_vmin, vcenter=level, vmax=data_vmax)

    extent = (ranges[free[0]][0], ranges[free[0]][1],
              ranges[free[1]][0], ranges[free[1]][1])

    plt.figure(figsize=(6, 5))
    im = plt.imshow(Z.T, origin='lower', extent=extent,
                    aspect='equal', cmap=cmap, norm=norm)

    cbar = plt.colorbar(im, label='model value')
    ticks = np.linspace(data_vmin, data_vmax, 7)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.2f}" for t in ticks])

    cs = plt.contour(np.linspace(*ranges[free[0]], res),
                     np.linspace(*ranges[free[1]], res),
                     Z.T, levels=[level], colors='k', linewidths=2.0)
    plt.clabel(cs, fmt=f'{level:g}')

    labels = ['x', 'y', 'z']
    plt.xlabel(labels[free[0]])
    plt.ylabel(labels[free[1]])
    plt.title(f"{labels[fixed]} = {value:.3f} slice (white at {level:g})")
    plt.tight_layout()
    plt.show()


# ===================== KDTree 封装 =====================
class TorchCKDTree:
    def __init__(self, leafsize=16, balanced_tree=True, compact_nodes=True):
        self.leafsize = leafsize
        self.balanced_tree = balanced_tree
        self.compact_nodes = compact_nodes
        self.tree = None
        self._fit_device = None
        self._fit_dtype = None
        self.n = 0
        self.dim = 0

    @staticmethod
    def _to_numpy(x):
        if torch.is_tensor(x):
            return x.detach().cpu().to(torch.float64).numpy()
        return np.asarray(x, dtype=np.float64)

    def fit(self, x_in):
        if torch.is_tensor(x_in):
            self._fit_device = x_in.device
            self._fit_dtype = x_in.dtype
        else:
            self._fit_device = None
            self._fit_dtype = torch.float32

        xin_np = self._to_numpy(x_in)
        assert xin_np.ndim == 2, "x_in 必须是 (N, D)"
        self.n, self.dim = xin_np.shape

        self.tree = cKDTree(
            xin_np,
            leafsize=self.leafsize,
            balanced_tree=self.balanced_tree,
            compact_nodes=self.compact_nodes,
        )
        return self

    def query(self, x_bnd, k=1, workers=-1, return_numpy=False):
        assert self.tree is not None, "请先调用 fit(x_in) 建树"
        xb_np = self._to_numpy(x_bnd)
        if xb_np.ndim == 1:
            xb_np = xb_np[None, :]
        assert xb_np.shape[1] == self.dim, "x_bnd 维度与 x_in 不一致"

        dists, idx = self.tree.query(xb_np, k=k, workers=workers)

        if return_numpy:
            return dists, idx

        d_tensor = torch.from_numpy(np.asarray(dists))
        i_tensor = torch.from_numpy(np.asarray(idx, dtype=np.int64))

        if self._fit_device is not None:
            d_tensor = d_tensor.to(device=self._fit_device, dtype=self._fit_dtype)
            i_tensor = i_tensor.to(device=self._fit_device)
        return d_tensor, i_tensor


# ===================== 法向处理 =====================
@dataclass
class NormalParams:
    radius: float = 0.02
    max_nn: int = 30
    orient_k: int = 20  # <=0 不做


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
        self.path = Path(path)
        self.params = params
        self._pcd: o3d.geometry.PointCloud | None = None
        self._mesh: o3d.geometry.TriangleMesh | None = None
        self._loaded_type: str | None = None  # "pointcloud" | "mesh"
        self._changed: bool = False
        self.load(remeshing=remeshing)

    def load(self, remeshing=False) -> "PLYNormalHelper":
        pcd = o3d.io.read_point_cloud(str(self.path))
        mesh = o3d.io.read_triangle_mesh(str(self.path))

        if remeshing:
            v, f = pcu.make_mesh_watertight(
                np.asarray(mesh.vertices),
                np.asarray(mesh.triangles),
                20000
            )
            mesh = _to_o3d_mesh(v, f)
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
            if self._ensure_pointcloud_normals(self._pcd):
                self._changed = True
        else:
            rv, rf = self._ensure_mesh_normals(self._mesh)
            if rv or rf:
                self._changed = True
        return self

    def get_points_and_normals(self, stack: bool = False) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        self._require_loaded()
        if self._loaded_type == "pointcloud":
            pts = np.asarray(self._pcd.points)
            nrm = np.asarray(self._pcd.normals)
        else:
            pts = np.asarray(self._mesh.vertices)
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
        print(f"文件已保存：{out}")
        return out

    @staticmethod
    def _invalid_normals(arr: np.ndarray) -> bool:
        if arr.size == 0:
            return True
        if np.isnan(arr).any():
            return True
        if np.allclose(arr, 0):
            return True
        return False

    def _ensure_pointcloud_normals(self, pcd: o3d.geometry.PointCloud) -> bool:
        need = (not pcd.has_normals())
        if not need:
            normals = np.asarray(pcd.normals)
            need = self._invalid_normals(normals) or (normals.shape[0] != np.asarray(pcd.points).shape[0])
        if not need:
            return False
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.params.radius, max_nn=self.params.max_nn
            )
        )
        if self.params.orient_k > 0:
            pcd.orient_normals_consistent_tangent_plane(k=self.params.orient_k)
        return True

    def _ensure_mesh_normals(self, mesh: o3d.geometry.TriangleMesh) -> tuple[bool, bool]:
        recomputed_v = False
        recomputed_f = False

        if not mesh.has_vertex_normals():
            recomputed_v = True
        else:
            vn = np.asarray(mesh.vertex_normals)
            if self._invalid_normals(vn) or (vn.shape[0] != np.asarray(mesh.vertices).shape[0]):
                recomputed_v = True
        if recomputed_v:
            mesh.compute_vertex_normals()

        if not mesh.has_triangle_normals():
            recomputed_f = True
        else:
            fn = np.asarray(mesh.triangle_normals)
            if self._invalid_normals(fn) or (fn.shape[0] != np.asarray(mesh.triangles).shape[0]):
                recomputed_f = True
        if recomputed_f:
            mesh.compute_triangle_normals()

        return recomputed_v, recomputed_f

    def _require_loaded(self):
        if self._loaded_type is None:
            raise RuntimeError("请先调用 load()")


# ===================== AABB & 采样 =====================
class BoundingBox:
    def __init__(self, x_min, x_max, y_min, y_max, z_min, z_max):
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.y_min = float(y_min)
        self.y_max = float(y_max)
        self.z_min = float(z_min)
        self.z_max = float(z_max)

    def get_bounding_box(self):
        return (self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max)

    def sample(self, num_samples):
        n_samples_per_dim = int(num_samples ** (1 / 3)) + 1
        x = torch.linspace(self.x_min, self.x_max, n_samples_per_dim)
        y = torch.linspace(self.y_min, self.y_max, n_samples_per_dim)
        z = torch.linspace(self.z_min, self.z_max, n_samples_per_dim)
        xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
        grid_points = torch.stack([xx, yy, zz], dim=-1).view(-1, 3)
        return grid_points

    def contains(self, point):
        x, y, z = point
        return (self.x_min <= x <= self.x_max) and (self.y_min <= y <= self.y_max) and (self.z_min <= z <= self.z_max)

    def expand(self, margin=None, ratio=None):
        if margin is None and ratio is None:
            raise ValueError("必须指定 margin 或 ratio")
        if margin is not None:
            self.x_min -= margin;
            self.x_max += margin
            self.y_min -= margin;
            self.y_max += margin
            self.z_min -= margin;
            self.z_max += margin
        if ratio is not None:
            center_x = (self.x_min + self.x_max) / 2
            center_y = (self.y_min + self.y_max) / 2
            center_z = (self.z_min + self.z_max) / 2
            half_x = (self.x_max - self.x_min) / 2 * ratio
            half_y = (self.y_max - self.y_min) / 2 * ratio
            half_z = (self.z_max - self.z_min) / 2 * ratio
            self.x_min, self.x_max = center_x - half_x, center_x + half_x
            self.y_min, self.y_max = center_y - half_y, center_y + half_y
            self.z_min, self.z_max = center_z - half_z, center_z + half_z
        return self

    def to_array(self):
        return np.array([[self.x_min, self.y_min, self.z_min], [self.x_max, self.y_max, self.z_max]])


def sample_uniform_in_bbox(bbox: BoundingBox, n: int, device, dtype):
    xs = torch.rand(n, 3, device=device, dtype=dtype)
    xs[:, 0] = xs[:, 0] * (bbox.x_max - bbox.x_min) + bbox.x_min
    xs[:, 1] = xs[:, 1] * (bbox.y_max - bbox.y_min) + bbox.y_min
    xs[:, 2] = xs[:, 2] * (bbox.z_max - bbox.z_min) + bbox.z_min
    return xs


# ===================== 模型 =====================
class SDFModel(torch.nn.Module):
    def __init__(self, inner_dim=512, input_dim=3, output_dim=1):
        super(SDFModel, self).__init__()
        self.inner_dim = inner_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_layer = nn.Sequential(
            nn.utils.parametrizations.weight_norm(nn.Linear(input_dim, inner_dim)),
            nn.Tanh()
        )
        self.final_layer = nn.Linear(inner_dim, output_dim)

    def forward(self, x):
        x = self.input_layer(x)
        sdf = self.final_layer(x)
        return sdf


# ===================== 损失 =====================
def eikonal_loss(model, x):
    x = x.clone().detach().requires_grad_(True)
    fx = model(x)  # (N,1)
    ones = torch.ones_like(fx, device=fx.device)
    grads = torch.autograd.grad(
        outputs=fx, inputs=x, grad_outputs=ones,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]  # (N,3)
    return (grads.norm(dim=1) - 1.0).pow(2).mean()


def surface_zero_loss(model, xs):
    fx = model(xs)
    return fx.abs().mean()


def normal_alignment_loss(model, xs, ns):
    xs = xs.clone().detach().requires_grad_(True)
    fx = model(xs)
    ones = torch.ones_like(fx, device=fx.device)
    grads = torch.autograd.grad(
        outputs=fx, inputs=xs, grad_outputs=ones,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    g = torch.nn.functional.normalize(grads, dim=1, eps=1e-8)
    n = torch.nn.functional.normalize(ns, dim=1, eps=1e-8)
    cos2 = (g * n).sum(dim=1).clamp(-1, 1).pow(2)
    return (1.0 - cos2).mean()


# ===================== 主程序 =====================
if __name__ == "__main__":

    torch.set_default_device('cuda') if torch.cuda.is_available() else torch.set_default_device('cpu')
    device = torch.tensor(0.0).device
    dtype = torch.tensor(0.0).dtype

    helper = PLYNormalHelper("bottle_with_watertight.ply", remeshing=True)
    # helper.save("_with_watertight")
    helper.ensure_normals()
    pts, nrms = helper.get_points_and_normals()
    print(f"点数量: {pts.shape[0]}, 法向数量: {nrms.shape[0]}")

    # -------- 1) 直接对原始几何归一化（中心到原点，最长半径=1）--------
    pts_t = torch.tensor(pts, device=device, dtype=dtype)
    nrms_t = torch.tensor(nrms, device=device, dtype=dtype)

    mins = pts_t.min(dim=0).values
    maxs = pts_t.max(dim=0).values
    center = (mins + maxs) * 0.5  # 几何中心
    half_ranges = (maxs - mins) * 0.5  # 各轴半径
    scale = torch.max(half_ranges)  # 统一缩放：最长半径 -> 1
    pts_n = (pts_t - center) / scale  # 归一化点
    nrms_n = nrms_t  # 统一缩放不改变法向方向（已单位化）

    # -------- 2) 用归一化点云构造采样边界与样本 --------
    mins_n = pts_n.min(dim=0).values
    maxs_n = pts_n.max(dim=0).values
    bbox = BoundingBox(mins_n[0].item(), maxs_n[0].item(),
                       mins_n[1].item(), maxs_n[1].item(),
                       mins_n[2].item(), maxs_n[2].item())
    # bbox = BoundingBox(-1, 1, -1, 1, -1, 1)  # 直接用单位球盒
    bbox.expand(ratio=1.5)  # 归一化坐标下的适度外扩
    x_bnd = bbox.sample(num_samples=100000).to(device=device, dtype=dtype)

    # # -------- 3) 归一化坐标系内计算有符号距离 --------
    # tree = TorchCKDTree(leafsize=32).fit(pts_n)
    # dists, idx_nn = tree.query(x_bnd, k=1, workers=-1)
    #
    # nn_normals = nrms_n[idx_nn]
    # nn_points = pts_n[idx_nn]
    # vec = x_bnd - nn_points
    # sign = torch.sign(torch.sum(vec * nn_normals, dim=1))
    # signed_dists = (dists * sign).unsqueeze(-1)  # 归一化单位下的SDF

    # —— 两路 batch 的迭代（沿用你“随机打乱 + 手切 batch”的风格）——
    batch_size_eik = 32
    batch_size_surf = 32
    num_epochs = 100
    lambda_eik, lambda_surf, lambda_norm = 1.0, 100.0, 1.0
    sigma_surf = 0.01  # 表面附近扰动，提升稳定性；设 0 则不扰动

    model = SDFModel(inner_dim=512, input_dim=3, output_dim=1).to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=20)

    N_bnd = x_bnd.size(0)
    N_pts = pts_n.size(0)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        seen = 0

        # 用于统计每个分量
        epoch_eik_loss = 0.0
        epoch_surf_loss = 0.0
        epoch_norm_loss = 0.0

        idx_b = torch.randperm(N_bnd, device=device)
        idx_s = torch.randperm(N_pts, device=device)

        steps = max((N_bnd + batch_size_eik - 1) // batch_size_eik,
                    (N_pts + batch_size_surf - 1) // batch_size_surf)

        for t in range(steps):
            # eikonal batch
            s1, e1 = t * batch_size_eik, min((t + 1) * batch_size_eik, N_bnd)
            xb = x_bnd[idx_b[s1:e1]] if s1 < e1 else None

            # surface batch
            s2, e2 = t * batch_size_surf, min((t + 1) * batch_size_surf, N_pts)
            xs = pts_n[idx_s[s2:e2]] if s2 < e2 else None
            ns = nrms_n[idx_s[s2:e2]] if s2 < e2 else None
            if xs is not None and sigma_surf > 0:
                xs = xs + torch.randn_like(xs) * sigma_surf

            # 分别计算分量
            eik_loss = torch.tensor(0., device=device)
            surf_loss = torch.tensor(0., device=device)
            norm_loss = torch.tensor(0., device=device)

            if xb is not None:
                eik_loss = lambda_eik * eikonal_loss(model, xb)
            if xs is not None:
                surf_loss = lambda_surf * surface_zero_loss(model, xs)
                if lambda_norm > 0:
                    norm_loss = lambda_norm * normal_alignment_loss(model, xs, ns)

            loss = eik_loss + surf_loss + norm_loss

            if (xb is None) and (xs is None):
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            bs = (0 if xb is None else xb.size(0)) + (0 if xs is None else xs.size(0))
            epoch_loss += loss.item() * max(1, bs)
            epoch_eik_loss += eik_loss.item() * max(1, bs)
            epoch_surf_loss += surf_loss.item() * max(1, bs)
            epoch_norm_loss += norm_loss.item() * max(1, bs)
            seen += max(1, bs)

        epoch_loss /= max(1, seen)
        epoch_eik_loss /= max(1, seen)
        epoch_surf_loss /= max(1, seen)
        epoch_norm_loss /= max(1, seen)

        scheduler.step(epoch_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}] "
              f"loss={epoch_loss:.4e} "
              f"eik={epoch_eik_loss:.4e} "
              f"surf={epoch_surf_loss:.4e} "
              f"norm={epoch_norm_loss:.4e} "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

    print("Training finished.")
    # ---------- 简单检查与可视化 ----------
    with torch.no_grad():
        val_on_surface = model(pts_n).abs()
        print("表面点 |f(x)| 的最大/均值：",
              val_on_surface.max().item(), val_on_surface.mean().item())

    plot_model_slice(model, bbox=bbox.get_bounding_box())
