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

EPOCH_DEPRECATION_WARNING = (
    "The 'epoch' parameter in 'step()' is deprecated and will be removed."
)


# ---------------------- BiDirectionalLROnPlateau ----------------------
class BiDirectionalLROnPlateau(LRScheduler):
    def __init__(
            self, optimizer: Optimizer, mode: Literal["min", "max"] = "min", *,
            factor: float = 0.1, patience: int = 10, threshold: float = 1e-4,
            threshold_mode: Literal["rel", "abs"] = "rel", cooldown: int = 0,
            min_lr: Union[list[float], float] = 0.0, eps: float = 1e-8,
            up_factor: float = 1.1, up_patience: int = 5, up_cooldown: int = 0,
            max_lr: Union[list[float], float] = float("inf"),
    ):
        if factor >= 1.0: raise ValueError("factor (down) should be < 1.0.")
        if up_factor <= 1.0: raise ValueError("up_factor should be > 1.0.")
        self.factor = factor;
        self.up_factor = up_factor
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f"{type(optimizer).__name__} is not an Optimizer")
        self.optimizer = optimizer

        # min/max bounds per param group
        if isinstance(min_lr, (list, tuple)):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError(f"expected {len(optimizer.param_groups)} min_lrs, got {len(min_lr)}")
            self.default_min_lr = None;
            self.min_lrs = list(min_lr)
        else:
            self.default_min_lr = min_lr;
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        if isinstance(max_lr, (list, tuple)):
            if len(max_lr) != len(optimizer.param_groups):
                raise ValueError(f"expected {len(optimizer.param_groups)} max_lrs, got {len(max_lr)}")
            self.default_max_lr = None;
            self.max_lrs = list(max_lr)
        else:
            self.default_max_lr = max_lr;
            self.max_lrs = [max_lr] * len(optimizer.param_groups)

        self.patience = patience;
        self.cooldown = cooldown;
        self.eps = eps
        self.up_patience = up_patience;
        self.up_cooldown = up_cooldown
        self.last_epoch = 0
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

        self._init_is_better(mode=mode, threshold=threshold, threshold_mode=threshold_mode)
        self._reset()

    def _reset(self):
        self.best = self.mode_worse
        self.num_bad_epochs = 0
        self.num_good_epochs = 0
        self.cooldown_counter_down = 0
        self.cooldown_counter_up = 0

    def step(self, metrics: SupportsFloat, epoch=None) -> None:  # type: ignore[override]
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch + 1
        else:
            warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
        self.last_epoch = epoch

        improved = self.is_better(current, self.best)
        if improved:
            self.best = current;
            self.num_bad_epochs = 0;
            self.num_good_epochs += 1
        else:
            self.num_bad_epochs += 1;
            self.num_good_epochs = 0

        if self.in_cooldown_down:
            self.cooldown_counter_down -= 1;
            self.num_bad_epochs = 0
        if self.in_cooldown_up:
            self.cooldown_counter_up -= 1;
            self.num_good_epochs = 0

        if (self.num_bad_epochs > self.patience) and (not self.in_cooldown_down):
            self._reduce_lr(epoch);
            self.cooldown_counter_down = self.cooldown;
            self.num_bad_epochs = 0

        if (self.num_good_epochs > self.up_patience) and (not self.in_cooldown_up):
            self._increase_lr(epoch);
            self.cooldown_counter_up = self.up_cooldown;
            self.num_good_epochs = 0

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def _sync_bounds_if_groups_changed(self):
        n = len(self.optimizer.param_groups)
        if len(self.min_lrs) != n:
            if self.default_min_lr is None: raise RuntimeError("optimizer.param_groups changed; update 'min_lrs'.")
            self.min_lrs = [self.default_min_lr] * n
        if len(self.max_lrs) != n:
            if self.default_max_lr is None: raise RuntimeError("optimizer.param_groups changed; update 'max_lrs'.")
            self.max_lrs = [self.default_max_lr] * n

    def _reduce_lr(self, epoch):
        self._sync_bounds_if_groups_changed()
        for i, group in enumerate(self.optimizer.param_groups):
            old_lr = float(group["lr"]);
            new_lr = max(old_lr * self.factor, float(self.min_lrs[i]))
            if old_lr - new_lr > self.eps: group["lr"] = new_lr

    def _increase_lr(self, epoch):
        self._sync_bounds_if_groups_changed()
        for i, group in enumerate(self.optimizer.param_groups):
            old_lr = float(group["lr"]);
            new_lr = min(old_lr * self.up_factor, float(self.max_lrs[i]))
            if new_lr - old_lr > self.eps: group["lr"] = new_lr

    @property
    def in_cooldown_down(self) -> bool:
        return self.cooldown_counter_down > 0

    @property
    def in_cooldown_up(self) -> bool:
        return self.cooldown_counter_up > 0

    def is_better(self, a, best) -> bool:
        if self.mode == "min" and self.threshold_mode == "rel":
            return a < best * (1.0 - self.threshold)
        elif self.mode == "min" and self.threshold_mode == "abs":
            return a < best - self.threshold
        elif self.mode == "max" and self.threshold_mode == "rel":
            return a > best * (1.0 + self.threshold)
        else:
            return a > best + self.threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {"min", "max"}: raise ValueError(f"mode {mode} is unknown!")
        if threshold_mode not in {"rel", "abs"}: raise ValueError(f"threshold mode {threshold_mode} is unknown!")
        self.mode = mode;
        self.threshold = threshold;
        self.threshold_mode = threshold_mode
        self.mode_worse = inf if mode == "min" else -inf

    @override
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.__dict__.update(state_dict)
        self._init_is_better(mode=self.mode, threshold=self.threshold, threshold_mode=self.threshold_mode)


# ---------------------- 2D 截面渲染（保存） ----------------------
def save_model_slice_png(
        save_path: Union[str, Path],
        model,
        bbox,
        *,
        axis: str = "y",
        value: float = 0.0,
        res: int = 256,
        level: float = 0.0,
        cmap: str = "RdBu_r",
        vmin: float | None = None,
        vmax: float | None = None,
        symmetric: bool = True,
        device: Union[torch.device, str, None] = None,
):
    assert axis in ("x", "y", "z")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    xr, yr, zr = (bbox[0], bbox[1]), (bbox[2], bbox[3]), (bbox[4], bbox[5])
    if axis == "x":
        value = float(np.clip(value, *xr))
    elif axis == "y":
        value = float(np.clip(value, *yr))
    else:
        value = float(np.clip(value, *zr))

    axis2idx = {"x": 0, "y": 1, "z": 2}
    fixed = axis2idx[axis]
    free = [i for i in range(3) if i != fixed]
    ranges = [xr, yr, zr]

    u = torch.linspace(ranges[free[0]][0], ranges[free[0]][1], res, device=device)
    v = torch.linspace(ranges[free[1]][0], ranges[free[1]][1], res, device=device)
    U, V = torch.meshgrid(u, v, indexing="ij")

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
    im = plt.imshow(Z.T, origin="lower", extent=extent, aspect="equal", cmap=cmap, norm=norm)
    cbar = plt.colorbar(im, label="model value")
    ticks = np.linspace(data_vmin, data_vmax, 7)
    cbar.set_ticks(ticks);
    cbar.set_ticklabels([f"{t:.2f}" for t in ticks])

    cs = plt.contour(np.linspace(*ranges[free[0]], res),
                     np.linspace(*ranges[free[1]], res),
                     Z.T, levels=[level], colors="k", linewidths=2.0)
    plt.clabel(cs, fmt=f"{level:g}")

    labels = ["x", "y", "z"]
    plt.xlabel(labels[free[0]]);
    plt.ylabel(labels[free[1]])
    plt.title(f"{labels[fixed]} = {value:.3f} slice (white at {level:g})")
    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"[Viz-2D] Saved: {save_path}")


# ---------------------- KDTree 封装 ----------------------
class TorchCKDTree:
    def __init__(self, leafsize=16, balanced_tree=True, compact_nodes=True):
        self.leafsize = leafsize;
        self.balanced_tree = balanced_tree;
        self.compact_nodes = compact_nodes
        self.tree = None;
        self._fit_device = None;
        self._fit_dtype = None
        self.n = 0;
        self.dim = 0

    @staticmethod
    def _to_numpy(x):
        if torch.is_tensor(x): return x.detach().cpu().to(torch.float64).numpy()
        return np.asarray(x, dtype=np.float64)

    def fit(self, x_in):
        if torch.is_tensor(x_in):
            self._fit_device = x_in.device;
            self._fit_dtype = x_in.dtype
        else:
            self._fit_device = None;
            self._fit_dtype = torch.float64

        xin_np = self._to_numpy(x_in)
        assert xin_np.ndim == 2, "x_in 必须是 (N, D)"
        self.n, self.dim = xin_np.shape
        self.tree = cKDTree(xin_np, leafsize=self.leafsize,
                            balanced_tree=self.balanced_tree, compact_nodes=self.compact_nodes)
        return self

    def query(self, x_bnd, k=1, workers=-1, return_numpy=False):
        assert self.tree is not None, "请先调用 fit(x_in) 建树"
        xb_np = self._to_numpy(x_bnd)
        if xb_np.ndim == 1: xb_np = xb_np[None, :]
        assert xb_np.shape[1] == self.dim, "x_bnd 维度与 x_in 不一致"

        dists, idx = self.tree.query(xb_np, k=k, workers=workers)
        if return_numpy: return dists, idx

        d_tensor = torch.from_numpy(np.asarray(dists))
        i_tensor = torch.from_numpy(np.asarray(idx, dtype=np.int64))
        if self._fit_device is not None:
            d_tensor = d_tensor.to(device=self._fit_device, dtype=self._fit_dtype)
            i_tensor = i_tensor.to(device=self._fit_device)
        return d_tensor, i_tensor


# ---------------------- AABB 采样盒 ----------------------
class BoundingBox:
    def __init__(self, x_min, x_max, y_min, y_max, z_min, z_max):
        self.x_min = float(x_min);
        self.x_max = float(x_max)
        self.y_min = float(y_min);
        self.y_max = float(y_max)
        self.z_min = float(z_min);
        self.z_max = float(z_max)

    def get_bounding_box(self):
        return (self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max)

    def sample(self, num_samples):
        n = int(num_samples ** (1 / 3)) + 1
        x = torch.linspace(self.x_min, self.x_max, n)
        y = torch.linspace(self.y_min, self.y_max, n)
        z = torch.linspace(self.z_min, self.z_max, n)
        xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")
        grid_points = torch.stack([xx, yy, zz], dim=-1).view(-1, 3)
        return grid_points

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
            cx = (self.x_min + self.x_max) / 2
            cy = (self.y_min + self.y_max) / 2
            cz = (self.z_min + self.z_max) / 2
            hx = (self.x_max - self.x_min) / 2 * ratio
            hy = (self.y_max - self.y_min) / 2 * ratio
            hz = (self.z_max - self.z_min) / 2 * ratio
            self.x_min, self.x_max = cx - hx, cx + hx
            self.y_min, self.y_max = cy - hy, cy + hy
            self.z_min, self.z_max = cz - hz, cz + hz
        return self


# ---------------------- SDF 模型 ----------------------
class SDFModel(nn.Module):
    def __init__(self, inner_dim=512, input_dim=3, output_dim=1, activation_cfg=("ReLU", "TanH")):
        super().__init__()
        act_map = {"ReLU": nn.ReLU(), "TanH": nn.Tanh()}
        self.input_layer = nn.Sequential(
            nn.utils.parametrizations.weight_norm(nn.Linear(input_dim, inner_dim)),
            act_map[activation_cfg[0]],
        )
        self.hidden_layer = nn.Sequential(
            nn.utils.parametrizations.weight_norm(nn.Linear(inner_dim, inner_dim)),
            act_map[activation_cfg[1]],
        )
        self.final_layer = nn.Linear(inner_dim, output_dim, bias=False)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layer(x)
        return self.final_layer(x)


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
    print(f"[Checkpoint] saved: {path} (epoch={epoch}, best={best_metric:.6e})")


def load_checkpoint(model, optimizer=None, path: str = None, map_location=None):
    if path is None or not Path(path).exists():
        print(f"[Checkpoint] not found: {path}");
        return None
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and ckpt.get("optim_state") is not None:
        optimizer.load_state_dict(ckpt["optim_state"])
    print(f"[Checkpoint] loaded: {path} (epoch={ckpt.get('epoch')}, best={ckpt.get('best_metric')})")
    return ckpt


# ---------------------- YAML 配置 ----------------------
@dataclass
class ModelCfg: inner_dim: int = 512; input_dim: int = 3; output_dim: int = 1


@dataclass
class OptimCfg: lr: float = 1e-4; weight_decay: float = 0.0; eps: float = 1e-14


@dataclass
class SchedulerCfg:
    mode: str = "min";
    factor: float = 0.9;
    patience: int = 20;
    threshold: float = 1e-4
    threshold_mode: str = "rel";
    up_factor: float = 1.1;
    up_patience: int = 30;
    eps: float = 1e-14


@dataclass
class VizCfg:
    enable_2d: bool = True;
    enable_3d: bool = True
    slice_axis: str = "y";
    slice_value: float = 0.0;
    slice_res: int = 256
    slice_level: float = 0.0;
    slice_cmap: str = "RdBu_r";
    slice_symmetric: bool = True
    iso_level: float = 0.0;
    iso_grid: Tuple[int, int, int] = (128, 128, 128)
    iso_resolution: Tuple[int, int] = (800, 800);
    iso_view: str = "front"


@dataclass
class OutCfg:
    checkpoints_dir: str = "checkpoints";
    renders_dir: str = "renders"


@dataclass
class LoggingCfg:
    dir: str = "logs"
    level: str = "INFO"
    also_print: bool = True
    filename_prefix: str = "train"
    timestamp_in_name: bool = True


@dataclass
class Config:
    pth_path: str = "../../data/cheese_in.pth"
    batch_size: int = 256;
    num_epochs: int = 2000
    activation_cfg: Tuple[str, str] = ("ReLU", "TanH")
    model: ModelCfg = field(default_factory=ModelCfg)
    optim: OptimCfg = field(default_factory=OptimCfg)
    scheduler: SchedulerCfg = field(default_factory=SchedulerCfg)
    logging: LoggingCfg = field(default_factory=LoggingCfg)
    remeshing: bool = True;
    bbox_expand_ratio: float = 1.2
    sample_num_bounding: int = 100000;
    kdtree_leafsize: int = 32
    viz: VizCfg = field(default_factory=VizCfg)
    out: OutCfg = field(default_factory=OutCfg)
    resume: bool = True;
    load_best_for_plot: bool = False
    save_every_best: int = 10;
    save_every_epoch: int = 0


def _as_tuple2(x) -> Tuple[str, str]:
    if isinstance(x, (list, tuple)) and len(x) == 2: return (str(x[0]), str(x[1]))
    if isinstance(x, str) and "," in x:
        a, b = x.split(",", 1);
        return (a.strip(), b.strip())
    raise ValueError("activation_cfg 需要两个激活，例如 ['ReLU','TanH']")


def load_config() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    lg = raw.get("logging", {}) or {}

    cfg = Config(
        pth_path=str(raw.get("pth_path", "../../data/cheese_in.pth")),
        batch_size=int(raw.get("batch_size", 256)),
        num_epochs=int(raw.get("num_epochs", 2000)),
        activation_cfg=_as_tuple2(raw.get("activation_cfg", ["ReLU", "TanH"])),
        model=ModelCfg(
            **{k: raw.get("model", {}).get(k, getattr(ModelCfg, k)) for k in ["inner_dim", "input_dim", "output_dim"]}),
        optim=OptimCfg(**{k: raw.get("optim", {}).get(k, getattr(OptimCfg, k)) for k in ["lr", "weight_decay", "eps"]}),
        scheduler=SchedulerCfg(**{
            "mode": raw.get("scheduler", {}).get("mode", "min"),
            "factor": float(raw.get("scheduler", {}).get("factor", 0.9)),
            "patience": int(raw.get("scheduler", {}).get("patience", 20)),
            "threshold": float(raw.get("scheduler", {}).get("threshold", 1e-4)),
            "threshold_mode": raw.get("scheduler", {}).get("threshold_mode", "rel"),
            "up_factor": float(raw.get("scheduler", {}).get("up_factor", 1.1)),
            "up_patience": int(raw.get("scheduler", {}).get("up_patience", 30)),
            "eps": float(raw.get("scheduler", {}).get("eps", 1e-14)),
        }),
        remeshing=bool(raw.get("remeshing", True)),
        bbox_expand_ratio=float(raw.get("bbox_expand_ratio", 1.2)),
        sample_num_bounding=int(raw.get("sample_num_bounding", 100000)),
        kdtree_leafsize=int(raw.get("kdtree_leafsize", 32)),
        viz=VizCfg(
            enable_2d=bool(raw.get("viz", {}).get("enable_2d", True)),
            enable_3d=bool(raw.get("viz", {}).get("enable_3d", True)),
            slice_axis=str(raw.get("viz", {}).get("slice_axis", "y")),
            slice_value=float(raw.get("viz", {}).get("slice_value", 0.0)),
            slice_res=int(raw.get("viz", {}).get("slice_res", 256)),
            slice_level=float(raw.get("viz", {}).get("slice_level", 0.0)),
            slice_cmap=str(raw.get("viz", {}).get("slice_cmap", "RdBu_r")),
            slice_symmetric=bool(raw.get("viz", {}).get("slice_symmetric", True)),
            iso_level=float(raw.get("viz", {}).get("iso_level", 0.0)),
            iso_grid=tuple(raw.get("viz", {}).get("iso_grid", [128, 128, 128])),
            iso_resolution=tuple(raw.get("viz", {}).get("iso_resolution", [800, 800])),
            iso_view=str(raw.get("viz", {}).get("iso_view", "front")),
        ),
        out=OutCfg(
            checkpoints_dir=str(raw.get("out", {}).get("checkpoints_dir", "checkpoints")),
            renders_dir=str(raw.get("out", {}).get("renders_dir", "renders")),
        ),
        resume=bool(raw.get("resume", True)),
        load_best_for_plot=bool(raw.get("load_best_for_plot", False)),
        save_every_best=int(raw.get("save_every_best", 10)),
        save_every_epoch=int(raw.get("save_every_epoch", 0)),
        logging=LoggingCfg(  # <--- 加上这段
            dir=str(lg.get("dir", "logs")),
            level=str(lg.get("level", "INFO")),
            also_print=bool(lg.get("also_print", True)),
            filename_prefix=str(lg.get("filename_prefix", "train")),
            timestamp_in_name=bool(lg.get("timestamp_in_name", True)),
        ),
    )
    return cfg


# ---------------------- 3D 等值面渲染（保存） ----------------------
def save_isosurface_png_and_ply(
        save_png_path: Union[str, Path],
        save_ply_path: Union[str, Path],
        *,
        model,
        bbox: BoundingBox,
        level: float = 0.0,
        grid: Tuple[int, int, int] = (128, 128, 128),
        resolution: Tuple[int, int] = (800, 800),
        view: str = "front",
):
    class NearShapeForViz(pyrfm.GeometryBase):
        def __init__(self): super().__init__(dim=3, intrinsic_dim=2)

        def get_bounding_box(self): return bbox.get_bounding_box()

        def in_sample(self, num_samples: int, with_boundary: bool = False): raise NotImplementedError

        def on_sample(self, num_samples: int, with_normal: bool = False): raise NotImplementedError

        def sdf(self, p: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                dev = next(model.parameters()).device
                out = model(p.to(dev));
                return out.to(p.device)

    near_shape = NearShapeForViz()
    near_model = pyrfm.RFMBase(dim=3, n_hidden=100, domain=near_shape, n_subdomains=1)
    near_model.W = torch.rand((100, 1))

    viz = pyrfm.RFMVisualizer3DMC(near_model, t=0.0, resolution=resolution, component_idx=0, view=view)
    viz.plot(cmap="viridis", level=level, grid=grid)

    save_png_path = Path(save_png_path);
    save_png_path.parent.mkdir(parents=True, exist_ok=True)
    viz.savefig(str(save_png_path))

    save_ply_path = Path(save_ply_path);
    save_ply_path.parent.mkdir(parents=True, exist_ok=True)
    viz.save_ply(str(save_ply_path))

    try:
        import matplotlib.pyplot as _plt
        _plt.close('all')
    except Exception:
        pass
    print(f"[Viz-3D] Saved PNG: {save_png_path} | PLY: {save_ply_path}")


class _TeeStream:
    """把写到 stdout/stderr 的内容同时写到原流和文件。"""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass
        return len(data)

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass


def setup_logging_and_tee(log_dir: Path, level: str = "INFO",
                          also_print: bool = True,
                          filename_prefix: str = "train",
                          timestamp_in_name: bool = True):
    """
    1) 创建 logs/<tag>/xxx.log
    2) 配置 logging 到该文件（同时可选打印到控制台）
    3) 用 Tee 把 sys.stdout / sys.stderr 也导入同一个文件（捕获 print 和第三方库输出）
    返回 (logger, logfile_handle, orig_stdout, orig_stderr)
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{filename_prefix}_{ts}.log" if timestamp_in_name else f"{filename_prefix}.log"
    log_path = log_dir / fname

    # 1) logging
    logger = logging.getLogger("train")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    fmt = logging.Formatter(fmt="%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(getattr(logging, level.upper(), logging.INFO))
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    if also_print:
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(getattr(logging, level.upper(), logging.INFO))
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    # 2) tee stdout/stderr，确保所有 print/第三方库输出都进文件
    logfile_handle = open(log_path, "a", encoding="utf-8")
    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    if also_print:
        sys.stdout = _TeeStream(orig_stdout, logfile_handle)
        sys.stderr = _TeeStream(orig_stderr, logfile_handle)
    else:
        sys.stdout = _TeeStream(logfile_handle)
        sys.stderr = _TeeStream(logfile_handle)

    logger.info(f"Logging to: {log_path}")
    return logger, logfile_handle, orig_stdout, orig_stderr


# ---------------------- SDF 法向计算（由梯度得到） ----------------------
def sdf_unit_normals(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    返回与 x 同 shape 的单位法向（N,3）。
    注意：会对 x 设置 requires_grad，且不保留计算图（不会额外占用显存）。
    """
    x_local = x.detach().clone().requires_grad_(True)
    y = model(x_local)
    if y.ndim == 2 and y.size(-1) == 1:
        y = y.squeeze(-1)
    ones = torch.ones_like(y, device=y.device, dtype=y.dtype)
    # 对输出对输入求梯度
    grads = torch.autograd.grad(
        y, x_local, grad_outputs=ones,
        create_graph=False, retain_graph=False, only_inputs=True
    )[0]
    # 归一化，防止零梯度数值问题
    n = torch.nn.functional.normalize(grads, dim=-1, eps=1e-9)
    return n


# ---------------------- 训练主流程 ----------------------
def main():
    t0 = time.perf_counter()  # ★ 计时开始
    cfg = load_config()

    torch.set_default_device('cuda') if torch.cuda.is_available() else torch.set_default_device('cpu')
    device = torch.tensor(0.0).device
    dtype = torch.tensor(0.0).dtype

    # 读取 & 法向（直接从 .pth 加载 torch 张量）
    # 期望文件中包含: x (N,3), normal (N,3), mean_curvature (可选)
    x, normal, mean_curvature = torch.load(cfg.pth_path, map_location=device)
    if not torch.is_tensor(x) or not torch.is_tensor(normal):
        raise TypeError("pth 文件应返回 (x, normal, mean_curvature) 的 torch 张量")
    print(f"点数量: {x.shape[0]}, 法向数量: {normal.shape[0]}")

    # 归一化（与原逻辑一致）：把点居中到 [-1,1] 盒内，法向保持单位方向
    pts_t = x.to(device=device, dtype=dtype)
    nrms_t = normal.to(device=device, dtype=dtype)
    mins = pts_t.min(dim=0).values;
    maxs = pts_t.max(dim=0).values
    center = (mins + maxs) * 0.5;
    half = (maxs - mins) * 0.5
    scale = torch.max(half)
    pts_n = (pts_t - center) / scale
    nrms_n = torch.nn.functional.normalize(nrms_t, dim=-1, eps=1e-9)

    # ---- Train/Test split (10% test) ----
    N_total = pts_n.size(0)
    perm_all = torch.randperm(N_total, device=device)
    n_test = max(1, int(0.1 * N_total))
    test_idx = perm_all[:n_test]
    train_idx = perm_all[n_test:]
    pts_train, pts_test = pts_n[train_idx], pts_n[test_idx]
    nrms_train, nrms_test = nrms_n[train_idx], nrms_n[test_idx]
    print(f"[Split] train={pts_train.shape[0]}, test={pts_test.shape[0]} (10% test)")

    # AABB & 训练数据
    mins_n = pts_n.min(dim=0).values;
    maxs_n = pts_n.max(dim=0).values
    bbox = BoundingBox(mins_n[0].item(), maxs_n[0].item(),
                       mins_n[1].item(), maxs_n[1].item(),
                       mins_n[2].item(), maxs_n[2].item()).expand(ratio=cfg.bbox_expand_ratio)
    x_bnd = bbox.sample(num_samples=cfg.sample_num_bounding).to(device=device, dtype=dtype)

    tree = TorchCKDTree(leafsize=cfg.kdtree_leafsize).fit(pts_n)
    dists, idx_nn = tree.query(x_bnd, k=1, workers=-1)
    nn_normals = nrms_n[idx_nn];
    nn_points = pts_n[idx_nn]
    vec = x_bnd - nn_points;
    sign = torch.sign(torch.sum(vec * nn_normals, dim=1))
    signed_dists = (dists * sign).unsqueeze(-1)

    x_train = torch.cat([x_bnd, pts_train], dim=0)
    y_train = torch.cat([signed_dists, torch.zeros((pts_train.shape[0], 1), device=device, dtype=dtype)], dim=0)

    batch_size = cfg.batch_size;
    num_samples = x_train.size(0);
    num_epochs = cfg.num_epochs

    # 激活/模型
    activation_cfg = cfg.activation_cfg

    def _act_tag(cfg_tuple: tuple[str, str]) -> str:
        safe = [s.lower().replace("+", "").replace("/", "-") for s in cfg_tuple]
        return f"{safe[0]}-{safe[1]}"

    tag = _act_tag(activation_cfg)
    # Dataset-specific tag so different pth inputs get separate folders
    data_tag = Path(cfg.pth_path).stem

    model = SDFModel(
        inner_dim=cfg.model.inner_dim, input_dim=cfg.model.input_dim,
        output_dim=cfg.model.output_dim, activation_cfg=activation_cfg
    ).to(device=device, dtype=dtype)

    # 优化 & 调度
    # criterion = torch.nn.MSELoss()
    criterion = torch.nn.L1Loss()

    def L1Loss_clamp(input, target, min_val=-0.1, max_val=0.1, reduction="mean"):
        input = torch.clamp(input, min_val, max_val)
        target = torch.clamp(target, min_val, max_val)
        loss = torch.abs(input - target)
        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        else:
            return loss

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optim.lr,
                                 weight_decay=cfg.optim.weight_decay, eps=cfg.optim.eps)
    scheduler = BiDirectionalLROnPlateau(
        optimizer,
        mode=cfg.scheduler.mode,
        factor=cfg.scheduler.factor,
        patience=cfg.scheduler.patience,
        threshold=cfg.scheduler.threshold,
        threshold_mode=cfg.scheduler.threshold_mode,
        up_factor=cfg.scheduler.up_factor,
        up_patience=cfg.scheduler.up_patience,
        eps=cfg.scheduler.eps
    )

    # 目录
    ckpt_dir = Path(cfg.out.checkpoints_dir) / data_tag / tag
    snap_dir = Path(cfg.out.renders_dir) / data_tag / tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    snap_dir.mkdir(parents=True, exist_ok=True)

    logs_dir = Path(cfg.logging.dir) / data_tag / tag
    logger, _logfp, _orig_out, _orig_err = setup_logging_and_tee(
        logs_dir,
        level=cfg.logging.level,
        also_print=cfg.logging.also_print,
        filename_prefix=cfg.logging.filename_prefix,
        timestamp_in_name=cfg.logging.timestamp_in_name,
    )

    ckpt_best = ckpt_dir / "sdf_best.pt"
    ckpt_last = ckpt_dir / "sdf_last.pt"

    RESUME = bool(cfg.resume);
    LOAD_BEST_FOR_PLOT = bool(cfg.load_best_for_plot)
    best_metric = float("inf")

    if RESUME and ckpt_last.exists():
        ck = load_checkpoint(model, optimizer, str(ckpt_last), map_location=device)
        start_epoch = int(ck.get("epoch", 0)) if ck is not None else 0
        if ck is not None and "best_metric" in ck and ck["best_metric"] is not None:
            best_metric = float(ck["best_metric"])
    else:
        start_epoch = 0
        for f in [ckpt_best, ckpt_last]:
            try:
                if f.exists(): f.unlink(); print(f"[Clean-{tag}] removed: {f}")
            except Exception as e:
                print(f"[Clean-{tag}] failed to remove {f}: {e}")
        for f in ckpt_dir.glob("*.pt"):
            try:
                f.unlink();
                print(f"[Clean-{tag}] removed: {f}")
            except Exception as e:
                print(f"[Clean-{tag}] failed to remove {f}: {e}")
        if snap_dir.exists():
            for p in snap_dir.iterdir():
                try:
                    if p.is_file():
                        p.unlink()
                    elif p.is_dir():
                        shutil.rmtree(p)
                except Exception as e:
                    print(f"[Clean-{tag}] failed to remove {p}: {e}")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        snap_dir.mkdir(parents=True, exist_ok=True)
        best_metric = float("inf")
        print(f"[Clean] checkpoints/ and renders/ are reset for data={data_tag}, tag={tag}.")

    best_count = 0
    epochs_done = 0  # ★ 新增计数器

    try:

        # 训练
        for epoch in range(start_epoch, start_epoch + num_epochs):
            epoch_loss = 0.0
            indices = torch.randperm(num_samples, device=device)
            for start in range(0, num_samples, batch_size):
                end = start + batch_size
                idx = indices[start:end]
                x_batch = x_train[idx];
                y_batch = y_train[idx]
                y_pred = model(x_batch)
                if epoch < 100:
                    loss = criterion(y_pred, y_batch)
                else:
                    loss = L1Loss_clamp(y_pred, y_batch, min_val=-0.1, max_val=0.1, reduction="mean")
                optimizer.zero_grad();
                loss.backward();
                optimizer.step()
                epoch_loss += loss.item() * x_batch.size(0)
            epoch_loss /= num_samples

            # 验证指标：平均法向夹角（度）
            with torch.no_grad():
                # 真值法向先确保单位长度
                gt_n = torch.nn.functional.normalize(nrms_n, dim=-1, eps=1e-9)

            # 需要梯度来算预测法向（由 SDF 梯度得）
            # 验证指标：法向夹角（度）：平均 + 最大
            with torch.no_grad():
                gt_n = torch.nn.functional.normalize(nrms_n, dim=-1, eps=1e-9)

            pred_n = sdf_unit_normals(model, pts_n)  # (N,3)
            dot = (pred_n * gt_n).sum(dim=-1).clamp(-1.0, 1.0)
            angles_rad = torch.acos(dot)
            angles_deg = angles_rad * (180.0 / math.pi)

            val_angle_mean = angles_deg.mean().item()
            val_angle_max = angles_deg.max().item()  # ← 新增最大角度
            scheduler.step(epoch_loss)  # 仍然用平均角度驱动 LR

            # 保存 last
            save_checkpoint(model, optimizer, epoch + 1, best_metric, str(ckpt_last))

            # 条件保存：best
            improved = epoch_loss < best_metric
            if improved:
                best_metric = epoch_loss
                save_checkpoint(model, optimizer, epoch + 1, best_metric, str(ckpt_best))
                best_count += 1
                if cfg.save_every_best > 0 and (best_count % cfg.save_every_best == 0):
                    # 2D & 3D
                    if cfg.viz.enable_2d:
                        png2d = snap_dir / f"slice_best{best_count:04d}_e{epoch + 1:04d}.png"
                        save_model_slice_png(
                            png2d, model, bbox.get_bounding_box(),
                            axis=cfg.viz.slice_axis, value=cfg.viz.slice_value,
                            res=cfg.viz.slice_res, level=cfg.viz.slice_level,
                            cmap=cfg.viz.slice_cmap, symmetric=cfg.viz.slice_symmetric,
                            device=device
                        )
                    if cfg.viz.enable_3d:
                        png3d = snap_dir / f"iso_best{best_count:04d}_e{epoch + 1:04d}.png"
                        ply3d = snap_dir / f"iso_best{best_count:04d}_e{epoch + 1:04d}.ply"
                        save_isosurface_png_and_ply(
                            png3d, ply3d, model=model, bbox=bbox,
                            level=cfg.viz.iso_level,
                            grid=cfg.viz.iso_grid,
                            resolution=cfg.viz.iso_resolution,
                            view=cfg.viz.iso_view
                        )

            # 固定 epoch 间隔保存
            if cfg.save_every_epoch > 0 and ((epoch + 1) % cfg.save_every_epoch == 0):
                if cfg.viz.enable_2d:
                    png2d = snap_dir / f"slice_e{epoch + 1:04d}.png"
                    save_model_slice_png(
                        png2d, model, bbox.get_bounding_box(),
                        axis=cfg.viz.slice_axis, value=cfg.viz.slice_value,
                        res=cfg.viz.slice_res, level=cfg.viz.slice_level,
                        cmap=cfg.viz.slice_cmap, symmetric=cfg.viz.slice_symmetric,
                        device=device
                    )
                if cfg.viz.enable_3d:
                    png3d = snap_dir / f"iso_e{epoch + 1:04d}.png"
                    ply3d = snap_dir / f"iso_e{epoch + 1:04d}.ply"
                    save_isosurface_png_and_ply(
                        png3d, ply3d, model=model, bbox=bbox,
                        level=cfg.viz.iso_level, grid=cfg.viz.iso_grid,
                        resolution=cfg.viz.iso_resolution, view=cfg.viz.iso_view
                    )

            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch [{epoch + 1}/{start_epoch + num_epochs}] "
                  f"Loss: {epoch_loss:.6e} | Best: {best_metric:.6e} | "
                  f"LR: {current_lr:.6e} | ValAngle(deg): mean={val_angle_mean:.3f}, max={val_angle_max:.3f} "
                  f"{' <BEST>' if improved else ''}")

            epochs_done += 1  # ★ 每个 epoch 结束 +1

        print("Training finished.")

        if cfg.load_best_for_plot and ckpt_best.exists():
            load_checkpoint(model, None, str(ckpt_best), map_location=device)

        with torch.no_grad():
            sdf_abs_test = model(pts_test).abs()
            print("[Eval/Test] Max |SDF(pts_test)|:", sdf_abs_test.max().item())
            print("[Eval/Test] Mean |SDF(pts_test)|:", sdf_abs_test.mean().item())

        total_sec = time.perf_counter() - t0
        h = int(total_sec // 3600)
        m = int((total_sec % 3600) // 60)
        s = total_sec % 60
        avg_per_epoch = total_sec / max(1, epochs_done)

        print(f"[Time] Total elapsed: {h:02d}:{m:02d}:{s:05.2f} "
              f"({total_sec:.2f}s) | epochs_done={epochs_done} | "
              f"avg/epoch={avg_per_epoch:.2f}s")

    finally:
        # 一定要恢复 & 关闭
        try:
            sys.stdout = _orig_out
            sys.stderr = _orig_err
            _logfp.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
