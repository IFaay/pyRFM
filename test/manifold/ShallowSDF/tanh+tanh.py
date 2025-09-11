import warnings
from math import inf
from typing import Literal, SupportsFloat, Union, List, Tuple, Any
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import shutil
import pyrfm
import open3d as o3d
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import point_cloud_utils as pcu
from matplotlib.colors import TwoSlopeNorm

import math
from torch.optim.lr_scheduler import LRScheduler
from typing_extensions import override
from torch.optim import Optimizer

EPOCH_DEPRECATION_WARNING = (
    "The 'epoch' parameter in 'step()' is deprecated and will be removed."
)


class BiDirectionalLROnPlateau(LRScheduler):
    """Reduce LR on plateau; boost LR on sustained improvement.

    This scheduler extends PyTorch's ReduceLROnPlateau by also *increasing* the
    learning rate when the monitored metric keeps improving for a consecutive
    number of epochs (``up_patience``). This can accelerate training in phases
    where the model is confidently descending the loss landscape.

    Args:
        optimizer (Optimizer): Wrapped optimizer.

        mode (str): One of {'min', 'max'}.
            - In 'min' mode the monitored quantity is expected to decrease (e.g. loss).
            - In 'max' mode it is expected to increase (e.g. accuracy).
            Default: 'min'.

        # ↓↓↓ standard (down) controls, mirroring ReduceLROnPlateau ↓↓↓
        factor (float): Multiplicative factor for *reducing* LR.
            new_lr = lr * factor. Must be < 1. Default: 0.1.
        patience (int): Number of bad epochs allowed before reducing LR.
            Default: 10.
        threshold (float): Minimum significant change to qualify as improvement.
            Default: 1e-4.
        threshold_mode (str): {'rel', 'abs'}. Default: 'rel'.
        cooldown (int): Cooldown epochs after a reduction before resuming normal
            operation. Default: 0.
        min_lr (float or list[float]): Per-group lower bound(s). Default: 0.
        eps (float): Minimal change in lr to apply an update. Default: 1e-8.

        # ↑↑↑ standard (down) controls ↑↑↑

        # ↓↓↓ new (up) controls for boosting LR on sustained improvement ↓↓↓
        up_factor (float): Multiplicative factor for *increasing* LR.
            new_lr = lr * up_factor. Must be > 1. Default: 1.1.
        up_patience (int): Number of consecutive *good* epochs before boosting.
            Default: 5.
        up_cooldown (int): Cooldown epochs after an increase. Default: 0.
        max_lr (float or list[float]): Per-group upper bound(s). Default: inf.

    Example:
        >>> # xdoctest: +SKIP
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = BiDirectionalLROnPlateau(
        ...     optimizer,
        ...     mode="min",
        ...     factor=0.5, patience=5,  # decrease controls
        ...     up_factor=1.2, up_patience=3  # increase controls
        ... )
        >>> for epoch in range(100):
        ...     train(...)
        ...     val_loss = validate(...)
        ...     scheduler.step(val_loss)  # call AFTER validate()
    """

    def __init__(
            self,
            optimizer: Optimizer,
            mode: Literal["min", "max"] = "min",
            *,
            # down controls (same semantics as ReduceLROnPlateau)
            factor: float = 0.1,
            patience: int = 10,
            threshold: float = 1e-4,
            threshold_mode: Literal["rel", "abs"] = "rel",
            cooldown: int = 0,
            min_lr: Union[list[float], float] = 0.0,
            eps: float = 1e-8,
            # up controls (new)
            up_factor: float = 1.1,
            up_patience: int = 5,
            up_cooldown: int = 0,
            max_lr: Union[list[float], float] = float("inf"),
    ):
        # --- validate factors
        if factor >= 1.0:
            raise ValueError("factor (down) should be < 1.0.")
        if up_factor <= 1.0:
            raise ValueError("up_factor should be > 1.0.")
        self.factor = factor
        self.up_factor = up_factor

        # --- attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f"{type(optimizer).__name__} is not an Optimizer")
        self.optimizer = optimizer

        # --- min/max LR per param group
        # min
        if isinstance(min_lr, (list, tuple)):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError(
                    f"expected {len(optimizer.param_groups)} min_lrs, got {len(min_lr)}"
                )
            self.default_min_lr = None
            self.min_lrs = list(min_lr)
        else:
            self.default_min_lr = min_lr
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        # max
        if isinstance(max_lr, (list, tuple)):
            if len(max_lr) != len(optimizer.param_groups):
                raise ValueError(
                    f"expected {len(optimizer.param_groups)} max_lrs, got {len(max_lr)}"
                )
            self.default_max_lr = None
            self.max_lrs = list(max_lr)
        else:
            self.default_max_lr = max_lr
            self.max_lrs = [max_lr] * len(optimizer.param_groups)

        # --- core controls
        self.patience = patience
        self.cooldown = cooldown
        self.eps = eps

        # --- up controls
        self.up_patience = up_patience
        self.up_cooldown = up_cooldown

        # --- book-keeping
        self.last_epoch = 0
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

        self._init_is_better(
            mode=mode,
            threshold=threshold,
            threshold_mode=threshold_mode,
        )
        self._reset()

    def _reset(self):
        """Reset counters and best state."""
        self.best = self.mode_worse
        # 'bad' epochs (for reduction)
        self.num_bad_epochs = 0
        # 'good' epochs (for boosting) must be consecutive
        self.num_good_epochs = 0
        # separate cooldowns for down and up
        self.cooldown_counter_down = 0
        self.cooldown_counter_up = 0

    def step(self, metrics: SupportsFloat, epoch=None) -> None:  # type: ignore[override]
        """Update internal state given the latest monitored value and adjust LR."""
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch + 1
        else:
            warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
        self.last_epoch = epoch

        improved = self.is_better(current, self.best)
        if improved:
            self.best = current
            self.num_bad_epochs = 0
            self.num_good_epochs += 1
        else:
            self.num_bad_epochs += 1
            self.num_good_epochs = 0

        # handle cooldowns
        if self.in_cooldown_down:
            self.cooldown_counter_down -= 1
            # ignore bad epochs while cooling down from a reduction
            self.num_bad_epochs = 0

        if self.in_cooldown_up:
            self.cooldown_counter_up -= 1
            # ignore good epochs while cooling down from an increase
            self.num_good_epochs = 0

        # --- down path: plateau/worse for too long
        if (self.num_bad_epochs > self.patience) and (not self.in_cooldown_down):
            self._reduce_lr(epoch)
            self.cooldown_counter_down = self.cooldown
            self.num_bad_epochs = 0

        # --- up path: consecutive improvements
        if (self.num_good_epochs > self.up_patience) and (not self.in_cooldown_up):
            self._increase_lr(epoch)
            self.cooldown_counter_up = self.up_cooldown
            self.num_good_epochs = 0

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    # -------------------- LR updates --------------------

    def _sync_bounds_if_groups_changed(self):
        """Keep min/max lists aligned with current param_groups length."""
        n = len(self.optimizer.param_groups)

        if len(self.min_lrs) != n:
            if self.default_min_lr is None:
                raise RuntimeError(
                    "optimizer.param_groups changed; please update 'min_lrs' to match."
                )
            self.min_lrs = [self.default_min_lr] * n

        if len(self.max_lrs) != n:
            if self.default_max_lr is None:
                raise RuntimeError(
                    "optimizer.param_groups changed; please update 'max_lrs' to match."
                )
            self.max_lrs = [self.default_max_lr] * n

    def _reduce_lr(self, epoch):
        self._sync_bounds_if_groups_changed()
        for i, group in enumerate(self.optimizer.param_groups):
            old_lr = float(group["lr"])
            new_lr = max(old_lr * self.factor, float(self.min_lrs[i]))
            if old_lr - new_lr > self.eps:
                group["lr"] = new_lr

    def _increase_lr(self, epoch):
        self._sync_bounds_if_groups_changed()
        for i, group in enumerate(self.optimizer.param_groups):
            old_lr = float(group["lr"])
            new_lr = min(old_lr * self.up_factor, float(self.max_lrs[i]))
            if new_lr - old_lr > self.eps:
                group["lr"] = new_lr

    # -------------------- helpers --------------------

    @property
    def in_cooldown_down(self) -> bool:
        return self.cooldown_counter_down > 0

    @property
    def in_cooldown_up(self) -> bool:
        return self.cooldown_counter_up > 0

    def is_better(self, a, best) -> bool:
        # identical to ReduceLROnPlateau's comparison logic
        if self.mode == "min" and self.threshold_mode == "rel":
            return a < best * (1.0 - self.threshold)
        elif self.mode == "min" and self.threshold_mode == "abs":
            return a < best - self.threshold
        elif self.mode == "max" and self.threshold_mode == "rel":
            return a > best * (1.0 + self.threshold)
        else:  # mode == 'max' and threshold_mode == 'abs'
            return a > best + self.threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {"min", "max"}:
            raise ValueError(f"mode {mode} is unknown!")
        if threshold_mode not in {"rel", "abs"}:
            raise ValueError(f"threshold mode {threshold_mode} is unknown!")

        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.mode_worse = inf if mode == "min" else -inf

    @override
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load the scheduler's state."""
        self.__dict__.update(state_dict)
        self._init_is_better(
            mode=self.mode, threshold=self.threshold, threshold_mode=self.threshold_mode
        )


# ---------------------- 可视化切片 ----------------------
def plot_model_slice(model,
                     bbox,
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
    extent = (ranges[free[0]][0], ranges[free[0]][1], ranges[free[1]][0], ranges[free[1]][1])

    plt.figure(figsize=(6, 5))
    im = plt.imshow(Z.T, origin='lower', extent=extent, aspect='equal', cmap=cmap, norm=norm)

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


# ---------------------- KDTree 封装 ----------------------
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
            self._fit_dtype = torch.float64

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
        self._changed: bool = False
        self.load(remeshing=remeshing)

    def load(self, remeshing=False) -> "PLYNormalHelper":
        pcd = o3d.io.read_point_cloud(str(self.path))
        mesh = o3d.io.read_triangle_mesh(str(self.path))

        if remeshing:
            v, f = pcu.make_mesh_watertight(
                np.asarray(mesh.vertices),
                np.asarray(mesh.triangles),
                40000
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


# ---------------------- AABB 采样盒 ----------------------
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


# ---------------------- SDF 模型 ----------------------
class SDFModel(torch.nn.Module):
    def __init__(self, inner_dim=512, input_dim=3, output_dim=1):
        super(SDFModel, self).__init__()
        self.inner_dim = inner_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_layer = nn.Sequential(nn.utils.parametrizations.weight_norm(nn.Linear(input_dim, inner_dim)),
                                         nn.Tanh())
        self.hidden_layer = nn.Sequential(nn.utils.parametrizations.weight_norm(nn.Linear(inner_dim, inner_dim)),
                                          nn.Tanh())
        self.final_layer = nn.Sequential(nn.Linear(inner_dim, output_dim))

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layer(x)
        sdf = self.final_layer(x)
        return sdf


# ---------------------- Checkpoint I/O ----------------------
def save_checkpoint(model, optimizer, epoch, best_metric, path: str):
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict() if optimizer is not None else None,
        "best_metric": float(best_metric),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    torch.save(ckpt, path)
    print(f"[Checkpoint] saved to: {path} (epoch={epoch}, best={best_metric:.6e})")


def load_checkpoint(model, optimizer=None, path: str = None, map_location=None):
    if path is None or not Path(path).exists():
        print(f"[Checkpoint] path not found: {path}")
        return None
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and ckpt.get("optim_state") is not None:
        optimizer.load_state_dict(ckpt["optim_state"])
    print(f"[Checkpoint] loaded from: {path} (epoch={ckpt.get('epoch')}, best={ckpt.get('best_metric')})")
    return ckpt


def render_isosurface(save_path: Union[str, Path],
                      grid=(128, 128, 128),
                      level: float = 0.0,
                      resolution: Tuple[int, int] = (800, 800),
                      view: str = 'bottom'):
    """
    使用 pyrfm 的 3D Monte Carlo 等值面可视化器，直接保存图片到 save_path。
    """

    class NearShapeForViz(pyrfm.GeometryBase):
        def __init__(self):
            super().__init__(dim=3, intrinsic_dim=2)

        def get_bounding_box(self):
            return bbox.get_bounding_box()

        def in_sample(self, num_samples: int, with_boundary: bool = False):
            # 可按需实现；本可视化流程不依赖
            raise NotImplementedError

        def on_sample(self, num_samples: int, with_normal: bool = False):
            # 可按需实现；本可视化流程不依赖
            raise NotImplementedError

        def sdf(self, p: torch.Tensor) -> torch.Tensor:
            # 保证设备匹配，且不追踪梯度
            with torch.no_grad():
                dev = next(model.parameters()).device
                out = model(p.to(dev))
                return out.to(p.device)

    near_shape = NearShapeForViz()
    near_model = pyrfm.RFMBase(dim=3, n_hidden=100, domain=near_shape, n_subdomains=1)
    # 与你的示例一致，W 随机初始化在 CPU 即可（可视化不依赖训练）
    near_model.W = torch.rand((100, 1))

    viz = pyrfm.RFMVisualizer3DMC(
        near_model, t=0.0, resolution=resolution, component_idx=0, view=view
    )
    viz.plot(cmap='viridis', level=level, grid=grid)
    viz.savefig(str(save_path))  # 关键：直接保存文件，不 show
    ## remove .png suffix and save as .ply
    save_path = Path(save_path).with_suffix('.ply')
    viz.save_ply(str(save_path))
    try:
        import matplotlib.pyplot as _plt
        _plt.close('all')  # 尝试释放可视化资源
    except Exception:
        pass


# ---------------------- 训练与可视化 ----------------------
if __name__ == "__main__":
    torch.set_default_device('cuda') if torch.cuda.is_available() else torch.set_default_device('cpu')
    device = torch.tensor(0.0).device
    dtype = torch.tensor(0.0).dtype

    helper = PLYNormalHelper("bunny.ply", remeshing=True)
    helper.ensure_normals()
    pts, nrms = helper.get_points_and_normals()
    print(f"点数量: {pts.shape[0]}, 法向数量: {nrms.shape[0]}")

    # ---- 归一化 ----
    pts_t = torch.tensor(pts, device=device, dtype=dtype)
    nrms_t = torch.tensor(nrms, device=device, dtype=dtype)

    mins = pts_t.min(dim=0).values
    maxs = pts_t.max(dim=0).values
    center = (mins + maxs) * 0.5
    half_ranges = (maxs - mins) * 0.5
    scale = torch.max(half_ranges)
    pts_n = (pts_t - center) / scale
    nrms_n = nrms_t

    # ---- 采样与 SDF 计算 ----
    mins_n = pts_n.min(dim=0).values
    maxs_n = pts_n.max(dim=0).values
    bbox = BoundingBox(mins_n[0].item(), maxs_n[0].item(),
                       mins_n[1].item(), maxs_n[1].item(),
                       mins_n[2].item(), maxs_n[2].item())
    bbox.expand(ratio=1.2)
    x_bnd = bbox.sample(num_samples=100000).to(device=device, dtype=dtype)

    tree = TorchCKDTree(leafsize=32).fit(pts_n)
    dists, idx_nn = tree.query(x_bnd, k=1, workers=-1)

    nn_normals = nrms_n[idx_nn]
    nn_points = pts_n[idx_nn]
    vec = x_bnd - nn_points
    sign = torch.sign(torch.sum(vec * nn_normals, dim=1))
    signed_dists = (dists * sign).unsqueeze(-1)

    x_train = torch.cat([x_bnd, pts_n], dim=0)
    y_train = torch.cat([signed_dists, torch.zeros((pts_n.shape[0], 1), device=device, dtype=dtype)], dim=0)

    batch_size = 256
    num_samples = x_train.size(0)
    num_epochs = 2000

    model = SDFModel(inner_dim=512, input_dim=3, output_dim=1).to(device=device, dtype=dtype)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0, eps=1e-14)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='min', factor=0.9, patience=20, threshold=0.0001, threshold_mode='rel', eps=1e-14
    # )

    scheduler = BiDirectionalLROnPlateau(optimizer, mode='min', factor=0.9, patience=20, threshold=0.0001,
                                         threshold_mode='rel',
                                         up_factor=1.1, up_patience=30, eps=1e-14)

    # ---- Checkpoint 配置 ----
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_best = ckpt_dir / "sdf_best.pt"
    ckpt_last = ckpt_dir / "sdf_last.pt"

    snap_dir = Path("renders")
    snap_dir.mkdir(parents=True, exist_ok=True)

    RESUME = True  # 断点续训：True 将从 ckpt_last 恢复
    LOAD_BEST_FOR_PLOT = False  # 训练后用 best 权重重载再作图：True 开启

    best_metric = float("inf")

    if RESUME and ckpt_last.exists():
        ck = load_checkpoint(model, optimizer, str(ckpt_last), map_location=device)
        start_epoch = int(ck.get("epoch", 0))
        if ck is not None and "best_metric" in ck and ck["best_metric"] is not None:
            best_metric = float(ck["best_metric"])
        # for param_group in optimizer.param_groups:
        #     param_group["lr"] = 1e-3  # 你想要的新学习率
    else:
        start_epoch = 0
        # —— 清理 checkpoints 目录（只清理权重文件 *.pt）——
        if ckpt_dir.exists():
            # 先删除指定的 best/last
            for f in [ckpt_best, ckpt_last]:
                try:
                    if f.exists():
                        f.unlink()
                        print(f"[Clean] removed: {f}")
                except Exception as e:
                    print(f"[Clean] failed to remove {f}: {e}")

            # 再安全地删除目录下所有 *.pt
            for f in ckpt_dir.glob("*.pt"):
                try:
                    f.unlink()
                    print(f"[Clean] removed: {f}")
                except Exception as e:
                    print(f"[Clean] failed to remove {f}: {e}")
        else:
            ckpt_dir.mkdir(parents=True, exist_ok=True)

        # —— 清理 renders 目录（图片/视频/子目录全部清掉）——
        if snap_dir.exists():
            for p in snap_dir.iterdir():
                try:
                    if p.is_file():
                        p.unlink()
                        print(f"[Clean] removed: {p}")
                    elif p.is_dir():
                        shutil.rmtree(p)
                        print(f"[Clean] rmtree: {p}")
                except Exception as e:
                    print(f"[Clean] failed to remove {p}: {e}")
        else:
            snap_dir.mkdir(parents=True, exist_ok=True)

        # 再次确保目录存在（即使刚被清空/删除）
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        snap_dir.mkdir(parents=True, exist_ok=True)

        # 重置最佳指标（避免沿用旧值）
        best_metric = float("inf")
        print("[Clean] checkpoints/ and renders/ are reset.")

    best_count = 0  # 每次出现 BEST 就 +1；累计到 10、20、30... 时触发保存
    # ---- 训练循环 ----
    for epoch in range(start_epoch, start_epoch + num_epochs):
        epoch_loss = 0.0
        indices = torch.randperm(num_samples, device=device)
        for start in range(0, num_samples, batch_size):
            end = start + batch_size
            idx = indices[start:end]
            x_batch = x_train[idx]
            y_batch = y_train[idx]

            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * x_batch.size(0)

        epoch_loss /= num_samples
        scheduler.step(epoch_loss)

        # ---- 保存 last & best ----
        improved = epoch_loss < best_metric
        if improved:
            best_metric = epoch_loss
            save_checkpoint(model, optimizer, epoch + 1, best_metric, str(ckpt_best))
            best_count += 1
            if best_count % 10 == 0:
                img_path = snap_dir / f"iso_e{epoch + 1:04d}_best{best_count:03d}.png"
                try:
                    plot_model_slice(model, bbox=bbox.get_bounding_box(), device=device, axis='y')
                    render_isosurface(img_path, grid=(128, 128, 128), level=0.0,
                                      resolution=(800, 800), view='front')
                    print(f"[Viz] Snapshot saved -> {img_path}")
                except Exception as e:
                    print(f"[Viz] Snapshot failed: {e}")
        save_checkpoint(model, optimizer, epoch + 1, best_metric, str(ckpt_last))

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch + 1}/{start_epoch + num_epochs}], Loss: {epoch_loss:.6e}, "
              f"Best: {best_metric:.6e}, LR: {current_lr:.6e}, "
              f"{'BEST' if improved else ''}")

    print("Training finished.")

    # ---- 可选：为可视化加载最佳权重 ----
    if LOAD_BEST_FOR_PLOT and ckpt_best.exists():
        load_checkpoint(model, None, str(ckpt_best), map_location=device)

    # ---- 简单评估与可视化 ----
    with torch.no_grad():
        print("Max |SDF(pts_n)|:", model(pts_n).abs().max().item())
        print("Mean SDF(pts_n):", model(pts_n).mean().item())
