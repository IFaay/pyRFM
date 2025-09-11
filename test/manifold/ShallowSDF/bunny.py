from typing import List, Union, Tuple

import pyrfm
import open3d as o3d
import numpy as np
from pathlib import Path
from dataclasses import dataclass
import torch
from torch.utils.data import TensorDataset, DataLoader
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import point_cloud_utils as pcu


class TorchCKDTree:
    """
    轻量封装：用 SciPy cKDTree 做最近邻，并自动处理
    PyTorch Tensor <-> NumPy 的转换与设备/dtype 回传。
    """

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
        """
        x_in: (N, D) torch.Tensor 或 np.ndarray
        """
        if torch.is_tensor(x_in):
            self._fit_device = x_in.device
            self._fit_dtype = x_in.dtype
        else:
            self._fit_device = None
            self._fit_dtype = torch.float32  # 若 query 要转回 torch，就用默认 float32

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
        """
        x_bnd: (M, D) torch.Tensor 或 np.ndarray
        k:    最近邻个数（k=1 返回一维；k>1 返回二维）
        workers: cKDTree 并行查询线程（-1 表示用尽可能多的核）
        return_numpy: True 则直接返回 numpy；否则尽量转回 torch（到 fit 的设备和 dtype）

        返回：
          dists, idx
          - 若 k==1: 形状 (M,)
          - 若 k>1 : 形状 (M, k)
        """
        assert self.tree is not None, "请先调用 fit(x_in) 建树"
        xb_np = self._to_numpy(x_bnd)
        if xb_np.ndim == 1:
            xb_np = xb_np[None, :]
        assert xb_np.shape[1] == self.dim, "x_bnd 维度与 x_in 不一致"

        dists, idx = self.tree.query(xb_np, k=k, workers=workers)

        if return_numpy:
            return dists, idx

        # 转回 torch（距离用训练时的 dtype，索引用 long）
        d_tensor = torch.from_numpy(np.asarray(dists))
        i_tensor = torch.from_numpy(np.asarray(idx, dtype=np.int64))

        # 若 fit 时输入是 torch，就回到原设备/精度
        if self._fit_device is not None:
            d_tensor = d_tensor.to(device=self._fit_device, dtype=self._fit_dtype)
            i_tensor = i_tensor.to(device=self._fit_device)
        return d_tensor, i_tensor


@dataclass
class NormalParams:
    # 点云法向估计参数
    radius: float = 0.02
    max_nn: int = 30
    orient_k: int = 20  # 方向一致化邻域大小；<=0 不做


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
    """
    - 自动识别点云或网格
    - 只在需要时计算法向
    - 显式调用 save() 时才写文件
    """

    def __init__(self, path: str | Path, params: NormalParams = NormalParams(), remeshing=False):
        self.path = Path(path)
        self.params = params
        self._pcd: o3d.geometry.PointCloud | None = None
        self._mesh: o3d.geometry.TriangleMesh | None = None
        self._loaded_type: str | None = None  # "pointcloud" | "mesh"
        self._changed: bool = False  # 是否修改过法向
        self.load(remeshing=remeshing)

    # ----------- 公共 API -----------
    def load(self, remeshing=False) -> "PLYNormalHelper":
        """读取并识别为点云或网格"""
        pcd = o3d.io.read_point_cloud(str(self.path))
        mesh = o3d.io.read_triangle_mesh(str(self.path))

        if remeshing:
            v, f = pcu.make_mesh_watertight(
                np.asarray(mesh.vertices),
                np.asarray(mesh.triangles),
                50000
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
        """确保有有效法向；只标记修改状态，不立即保存"""
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
        """返回所有点与法向"""
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
        """可视化点云或网格"""
        self._require_loaded()
        geo = self._pcd if self._loaded_type == "pointcloud" else self._mesh
        o3d.visualization.draw_geometries([geo])

    def save(self, suffix: str = "_with_normals") -> Path | None:
        """保存到新文件，仅当有修改过法向时才写文件"""
        # self._require_loaded()
        # if not self._changed:
        #     print("未检测到修改，无需保存")
        #     return None

        out = self.path.with_name(self.path.stem + suffix + ".ply")
        if self._loaded_type == "pointcloud":
            o3d.io.write_point_cloud(str(out), self._pcd)
        else:
            o3d.io.write_triangle_mesh(str(out), self._mesh)

        print(f"文件已保存：{out}")
        # self._changed = False
        return out

    # ----------- 内部工具 -----------
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
        """返回是否新计算法向"""
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
        """返回 (recomputed_vertex, recomputed_triangle)"""
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
            self.x_min -= margin
            self.x_max += margin
            self.y_min -= margin
            self.y_max += margin
            self.z_min -= margin
            self.z_max += margin

        if ratio is not None:
            center_x = (self.x_min + self.x_max) / 2
            center_y = (self.y_min + self.y_max) / 2
            center_z = (self.z_min + self.z_max) / 2

            half_x = (self.x_max - self.x_min) / 2 * ratio
            half_y = (self.y_max - self.y_min) / 2 * ratio
            half_z = (self.z_max - self.z_min) / 2 * ratio

            x_min = center_x - half_x
            x_max = center_x + half_x
            y_min = center_y - half_y
            y_max = center_y + half_y
            z_min = center_z - half_z
            z_max = center_z + half_z

            self.x_min, self.x_max = x_min, x_max
            self.y_min, self.y_max = y_min, y_max
            self.z_min, self.z_max = z_min, z_max

        return self

    def to_array(self):
        return np.array([[self.x_min, self.y_min, self.z_min], [self.x_max, self.y_max, self.z_max]])


class TrainableRFBase(pyrfm.RFTanH, torch.nn.Module):
    def __init__(self, *args, **kwargs):
        torch.nn.Module.__init__(self)
        pyrfm.RFTanH.__init__(self, *args, **kwargs)
        self.weights = torch.nn.Parameter(self.weights)
        self.biases = torch.nn.Parameter(self.biases)


import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def plot_model_slice(model,
                     bbox: List[float],
                     axis: str = 'z',
                     value: float = 0.0,
                     res: int = 256,
                     level: float = 0.0,
                     cmap: str = 'RdBu_r',
                     vmin: float = None,
                     vmax: float = None,
                     symmetric: bool = True,  # 新增：自动对称色条
                     device: Union[torch.device, str, None] = None):
    """
    在给定轴向的截面上可视化 model(x,y,z) 的标量场，并在 level 映射白色。
    """
    assert axis in ('x', 'y', 'z')
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 解析 bbox
    xr, yr, zr = (bbox[0], bbox[1]), (bbox[2], bbox[3]), (bbox[4], bbox[5])

    # 裁剪截面 value
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

    # 生成网格
    u = torch.linspace(ranges[free[0]][0], ranges[free[0]][1], res, device=device)
    v = torch.linspace(ranges[free[1]][0], ranges[free[1]][1], res, device=device)
    U, V = torch.meshgrid(u, v, indexing='ij')

    P = torch.zeros((res * res, 3), device=device, dtype=torch.float32)
    P[:, free[0]] = U.reshape(-1)
    P[:, free[1]] = V.reshape(-1)
    P[:, fixed] = value

    # 前向
    with torch.no_grad():
        out = model(P)
        if out.ndim == 2 and out.size(-1) == 1:
            out = out.squeeze(-1)
        Z = out.reshape(res, res).detach().cpu().numpy()

    # 自动设置范围
    data_min, data_max = Z.min(), Z.max()
    if symmetric:
        bound = max(abs(data_min), abs(data_max))
        data_vmin, data_vmax = -bound, bound
    else:
        data_vmin = data_min if vmin is None else vmin
        data_vmax = data_max if vmax is None else vmax

    # 确保 level 在范围内
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

    # 颜色条，显示负值刻度
    cbar = plt.colorbar(im, label='model value')
    ticks = np.linspace(data_vmin, data_vmax, 7)  # 7 个刻度
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.2f}" for t in ticks])

    # 等值线
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


if __name__ == "__main__":
    torch.set_default_device('cuda') if torch.cuda.is_available() else torch.set_default_device('cpu')
    device = torch.tensor(0.0).device
    dtype = torch.tensor(0.0).dtype

    helper = PLYNormalHelper("bun_zipper.ply", remeshing=False)
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

    # (可选) 若后续需要把预测的SDF还原到原单位：  d_orig = d_norm * scale

    # -------- 2) 用归一化点云构造采样边界与样本 --------
    mins_n = pts_n.min(dim=0).values
    maxs_n = pts_n.max(dim=0).values
    bbox = BoundingBox(mins_n[0].item(), maxs_n[0].item(),
                       mins_n[1].item(), maxs_n[1].item(),
                       mins_n[2].item(), maxs_n[2].item())
    # bbox = BoundingBox(-1, 1, -1, 1, -1, 1)  # 直接用单位球盒
    bbox.expand(ratio=1.5)  # 归一化坐标下的适度外扩
    x_bnd = bbox.sample(num_samples=500000).to(device=device, dtype=dtype)

    # -------- 3) 归一化坐标系内计算有符号距离 --------
    tree = TorchCKDTree(leafsize=32).fit(pts_n)
    dists, idx_nn = tree.query(x_bnd, k=1, workers=-1)

    nn_normals = nrms_n[idx_nn]
    nn_points = pts_n[idx_nn]
    vec = x_bnd - nn_points
    sign = torch.sign(torch.sum(vec * nn_normals, dim=1))
    signed_dists = (dists * sign).unsqueeze(-1)  # 归一化单位下的SDF

    # -------- 4) RFM 的 domain 也使用归一化后的范围 --------
    x_min, x_max, y_min, y_max, z_min, z_max = bbox.get_bounding_box()
    center_n = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
    radius_n = [(x_max - x_min) / 2, (y_max - y_min) / 2, (z_max - z_min) / 2]
    domain = pyrfm.Square3D(center=center_n, radius=radius_n)

    n_hidden = 256
    n_subdomains = 1
    model = pyrfm.RFMBase(dim=3, n_hidden=n_hidden, rf=TrainableRFBase, domain=domain, n_subdomains=n_subdomains)
    model.W = torch.nn.Parameter(torch.zeros(model.submodels.numel() * model.n_hidden, 1))

    # -------- 5) 训练数据改用归一化坐标 --------
    all_params = []
    for submodel in model.submodels.flat_data:
        all_params += list(submodel.parameters())
    all_params.append(model.W)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(all_params, lr=1e-3, weight_decay=0, eps=1e-14)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.9, patience=20, threshold=0.0001, threshold_mode='rel'
    )

    x_train = torch.cat([x_bnd, pts_n], dim=0)
    y_train = torch.cat([signed_dists, torch.zeros((pts_n.shape[0], 1), device=device, dtype=dtype)], dim=0)

    batch_size = 20000
    num_samples = x_train.size(0)
    num_epochs = 20


    def clamp_loss(pred, target, delta):
        pred_clamped = torch.clamp(pred, -delta, delta)
        target_clamped = torch.clamp(target, -delta, delta)
        return torch.abs(pred_clamped - target_clamped).mean()


    def exp_loss(pred, target, delta):
        pred_clamped = torch.clamp(pred, -delta, delta)
        target_clamped = torch.clamp(target, -delta, delta)
        L = torch.exp(-torch.abs(pred_clamped - target_clamped)).mean()
        return L


    def exp_mse_loss(pred, target):
        # 计算均方误差
        mse = torch.mean((pred - target) ** 2)
        # 按论文形式转成 exp(-MSE)
        return torch.exp(-mse)


    def weighted_mse_gauss(pred, target, sigma=0.05, eps=1e-8):
        # 基础误差
        mse = (pred - target) ** 2
        # 权重：0附近最大，远离0指数衰减
        w = torch.exp(-(target ** 2) / (sigma ** 2))
        # 归一化，避免整体权重漂移
        w = w / (w.mean() + eps)
        w[w < 0.1] = 0.1  # 避免过小权重
        return (w * mse).mean()


    # def rel_mse_loss(pred, target, eps=1e-8):
    #     mse = (torch.abs(pred - target) / (torch.abs(target) + eps)) ** 2
    #     return mse.mean()

    # print(model.submodels.flat_data[0].weights)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        indices = torch.randperm(num_samples)
        for start in range(0, num_samples, batch_size):
            end = start + batch_size
            idx = indices[start:end]
            x_batch = x_train[idx]
            y_batch = y_train[idx]

            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            # loss = rel_mse_loss(y_pred, y_batch)
            # loss = weighted_mse_gauss(y_pred, y_batch, sigma=0.05)
            # loss = exp_mse_loss(y_pred, y_batch)
            # 如需使用截断损失，在归一化坐标下可以用固定阈值：
            # loss = clamp_loss(y_pred, y_batch, delta=0.1)
            # loss = exp_loss(y_pred, y_batch, delta=0.1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * x_batch.size(0)

        epoch_loss /= num_samples
        scheduler.step(epoch_loss)
        if (epoch + 1) % 1 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.6e}, LR: {current_lr:.6e}")

    print("Training finished.")

    print(model(pts_n).abs().max())
    print(model(pts_n).mean())

    # print(model.submodels.flat_data[0].weights)

    with torch.no_grad():
        batchQR = pyrfm.BatchQR(m=n_hidden * n_subdomains ** 3, n_rhs=1)
        for start in range(0, num_samples, batch_size):
            x_batch = x_train[start:start + batch_size]
            y_batch = y_train[start:start + batch_size]
            A_batch = model.features(x_batch).cat(dim=1)
            batchQR.add_rows(A_batch, y_batch)
            torch.cuda.empty_cache()

        model.W = batchQR.get_solution()

    print(model(pts_n).abs().max())
    print(model(pts_n).mean())

    # with torch.no_grad():
    #     b_bnd = signed_dists
    #     # delta = 0.2 * extent
    #     # mask = torch.abs(b_bnd) <= delta
    #     # x_bnd = x_bnd[mask.squeeze()]
    #     # b_bnd = b_bnd[mask.squeeze()]
    #     A_bnd = model.features(x_bnd).cat(dim=1)
    #     A_pts = model.features(pts).cat(dim=1)
    #     A = torch.cat((A_bnd, A_pts), dim=0)
    #     b = torch.cat((b_bnd, torch.zeros((pts.shape[0], 1), device=device, dtype=dtype)), dim=0)
    #     model.compute(A).solve(b)

    # print(model(pts))

    plot_model_slice(model, bbox=bbox.get_bounding_box())


    class NearShape(pyrfm.GeometryBase):

        def __init__(self):
            super().__init__(dim=3, intrinsic_dim=2)

        def get_bounding_box(self) -> List[float]:
            return domain.get_bounding_box()

        def in_sample(self, num_samples: int, with_boundary: bool = False) -> torch.Tensor:
            pass

        def on_sample(self, num_samples: int, with_normal: bool = False) -> Union[
            torch.Tensor, Tuple[torch.Tensor, ...]]:
            pass

        def sdf(self, p: torch.Tensor) -> torch.Tensor:
            return model(p)
            # dists, idx_nn = tree.query(p, k=1, workers=-1)
            #
            # nn_normals = nrms_n[idx_nn]
            # nn_points = pts_n[idx_nn]
            # vec = p - nn_points
            # sign = torch.sign(torch.sum(vec * nn_normals, dim=1))
            # signed_dists = (dists * sign).unsqueeze(-1)  # 归一化单位下的SDF
            #
            # return signed_dists

    # model.domain = NearShape()
    #
    # # near_shape = NearShape()
    # # near_model = pyrfm.RFMBase(dim=3, n_hidden=100, domain=near_shape, n_subdomains=1)
    # # near_model.W = torch.rand((100, 1))
    #
    # viz = pyrfm.RFMVisualizer3DMC(model, t=0.0, resolution=(800, 800), component_idx=0, view='bottom')
    # # viz = pyrfm.RFMVisualizer3D(model, t=0.0, resolution=(1920, 1080), component_idx=0, view='iso')
    #
    # # 可选：调整网格与等值面（默认为 level=0.0，grid=(128,128,128)）
    # viz.plot(cmap='viridis', level=0.0, grid=(128, 128, 128))
    # viz.show()
