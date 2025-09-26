# -*- coding: utf-8 -*-
"""
Created on 2025/9/19

@author: Yifei Sun
"""
from collections import defaultdict
import re
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
from pyrfm import RFMVisualizer3DMC
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


class SDFModel1L(nn.Module):
    def __init__(self, inner_dim=512, input_dim=3, output_dim=1, activation: str = "TanH"):
        super().__init__()
        act_map = {"ReLU": nn.ReLU(), "TanH": nn.Tanh()}
        if activation not in act_map:
            raise ValueError(f"Unsupported activation: {activation}. Choose from {list(act_map.keys())}")
        self.net = nn.Sequential(
            nn.utils.parametrizations.weight_norm(nn.Linear(input_dim, inner_dim)),
            act_map[activation],
            nn.Linear(inner_dim, output_dim, bias=False),
        )

    def forward(self, x):
        return self.net(x)


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


class NormalNormRFMVisualizer3DMC(RFMVisualizer3DMC):
    def _compute_field_values_points(self, pts_world):
        """
        复用你在 ray-marching 版本中的字段取值逻辑，但针对任意点集合。
        返回 numpy (N,) 的标量数组（取 component_idx 分量；若 ref 存在，做绝对差）。
        """
        pts_t = torch.tensor(pts_world, device=self.device, dtype=self.dtype)
        if isinstance(self.model, pyrfm.RFMBase):
            if self.ref is not None:
                # field_vals = self.model(pts_t)
                nx = self.model.dForward(pts_t, (1, 0, 0))
                ny = self.model.dForward(pts_t, (0, 1, 0))
                nz = self.model.dForward(pts_t, (0, 0, 1))
                field_vals = torch.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
                ref_vals = self.ref(pts_t)
                field_vals = (field_vals - ref_vals).abs().detach().cpu().numpy()[:, self.component_idx]
            else:
                nx = self.model.dForward(pts_t, (1, 0, 0))
                ny = self.model.dForward(pts_t, (0, 1, 0))
                nz = self.model.dForward(pts_t, (0, 0, 1))
                field_vals = torch.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
                field_vals = field_vals.detach().cpu().numpy()[:, self.component_idx]

        elif isinstance(self.model, pyrfm.STRFMBase):
            xt = self.model.validate_and_prepare_xt(x=pts_t,
                                                    t=torch.tensor([[self.t]], device=self.device, dtype=self.dtype))
            if self.ref is not None:
                field_vals = self.model.forward(xt=xt)
                ref_vals = self.ref(xt=xt)
                field_vals = (field_vals - ref_vals).abs().detach().cpu().numpy()[:, self.component_idx]
            else:
                field_vals = self.model.forward(xt=xt).detach().cpu().numpy()[:, self.component_idx]
        else:
            raise NotImplementedError("Model type not supported for visualization.")
        return field_vals


class MeanCurvatureRFMVisualizer3DMC(RFMVisualizer3DMC):
    def _compute_field_values_points(self, pts_world):
        """
        复用你在 ray-marching 版本中的字段取值逻辑，但针对任意点集合。
        返回 numpy (N,) 的标量数组（取 component_idx 分量；若 ref 存在，做绝对差）。
        """
        pts_t = torch.tensor(pts_world, device=self.device, dtype=self.dtype)
        if isinstance(self.model, pyrfm.RFMBase):
            if self.ref is not None:
                # field_vals = self.model(pts_t)
                field_vals = self.model.dForward(pts_t, (2, 0, 0))
                field_vals += self.model.dForward(pts_t, (0, 2, 0))
                field_vals += self.model.dForward(pts_t, (0, 0, 2))
                field_vals *= 0.5
                ref_vals = self.ref(pts_t)
                field_vals = (field_vals - ref_vals).abs().detach().cpu().numpy()[:, self.component_idx]
            else:
                field_vals = self.model.dForward(pts_t, (2, 0, 0))
                field_vals += self.model.dForward(pts_t, (0, 2, 0))
                field_vals += self.model.dForward(pts_t, (0, 0, 2))
                field_vals *= 0.5
                field_vals = field_vals.detach().cpu().numpy()[:, self.component_idx]

        elif isinstance(self.model, pyrfm.STRFMBase):
            xt = self.model.validate_and_prepare_xt(x=pts_t,
                                                    t=torch.tensor([[self.t]], device=self.device, dtype=self.dtype))
            if self.ref is not None:
                field_vals = self.model.forward(xt=xt)
                ref_vals = self.ref(xt=xt)
                field_vals = (field_vals - ref_vals).abs().detach().cpu().numpy()[:, self.component_idx]
            else:
                field_vals = self.model.forward(xt=xt).detach().cpu().numpy()[:, self.component_idx]
        else:
            raise NotImplementedError("Model type not supported for visualization.")
        return field_vals


def restore_plain_state_dict(state_dict, eps=1e-12):
    """
    将带有 `parametrizations.weight.original*` 的权重还原为普通 `...weight`。
    假设采用的是 PyTorch 的 weight_norm，默认 dim=0（与官方默认一致）。
    对于 Linear/Conv/Embedding 都可用：对 v 在 dims = (1..N-1) 上做 L2 范数。
    """
    # 收集所有层的 (g, v) 指针
    buckets = defaultdict(dict)
    pat = re.compile(r"^(.*)\.parametrizations\.weight\.original([01])$")
    for k in state_dict.keys():
        m = pat.match(k)
        if m:
            prefix, idx = m.group(1), m.group(2)
            buckets[prefix][idx] = k  # 记录 original0 / original1 的完整 key

    new_sd = {}

    # 先复制原来就已是“普通”的参数（例如 final_layer.weight/bias、任意 bias）
    for k, v in state_dict.items():
        if ".parametrizations.weight.original" in k:
            continue  # 稍后还原后会以 ...weight 的新 key 放进去
        new_sd[k] = v

    # 对每个需要还原的权重执行 weight_norm 逆变换
    for prefix, pair in buckets.items():
        if "0" not in pair or "1" not in pair:
            raise ValueError(f"{prefix} 缺少 original0/1，无法还原。")

        g = state_dict[pair["0"]].clone()
        v = state_dict[pair["1"]].clone()

        # 计算 ||v||，对除第 0 维以外的全部维度做范数（等价于 weight_norm 的默认 dim=0）
        if v.dim() == 1:
            # 例如某些特殊情况：把 1D 当作 (out,) —— 这时范数就是绝对值
            v_norm = v.abs() + eps
            scale = g / v_norm
            w = v * scale
        else:
            reduce_dims = tuple(range(1, v.dim()))
            v_norm = v.norm(dim=reduce_dims, keepdim=True) + eps
            # g 形状通常是 (out,)；需要 reshape 成 (out, 1, 1, ...) 才能广播
            shape = [g.shape[0]] + [1] * (v.dim() - 1)
            scale = g.view(*shape) / v_norm
            w = v * scale

        # 把还原后的权重写回为 `<prefix>.weight`
        new_sd[f"{prefix}.weight"] = w

    return new_sd


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
        ref=None
):
    # viz = pyrfm.RFMVisualizer3DMCMC(model, t=0.0, resolution=resolution, component_idx=0, view=view, ref=ref)
    viz = NormalNormRFMVisualizer3DMC(model, t=0.0, resolution=resolution, component_idx=0, view=view, ref=ref)
    # viz = MeanCurvatureRFMVisualizer3DMC(model, t=0.0, resolution=resolution, component_idx=0, view=view, ref=ref)
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


if __name__ == "__main__":
    torch.set_default_device('cuda') if torch.cuda.is_available() else torch.set_default_device('cpu')
    device = torch.tensor(0.0).device
    dtype = torch.tensor(0.0).dtype

    # section3.1 results

    # # Draw the mean curvature from the trained shapes
    # # ellipsoid
    # pth_path = "../data/ellipsoid_in.pth"
    # pt_path = "./sec3_2/checkpoints/ellipsoid_in/tanh-tanh/sdf_best.pt"
    # x, normal, mean_curvature = torch.load(pth_path, map_location=device)
    # pts_t = x.to(device=device, dtype=dtype)
    # nrms_t = normal.to(device=device, dtype=dtype)
    # mins = pts_t.min(dim=0).values;
    # maxs = pts_t.max(dim=0).values
    # center = (mins + maxs) * 0.5;
    # half = (maxs - mins) * 0.5
    # scale = torch.max(half)
    # pts_n = (pts_t - center) / scale
    # nrms_n = torch.nn.functional.normalize(nrms_t, dim=-1, eps=1e-9)
    #
    # mins_n = pts_n.min(dim=0).values;
    # maxs_n = pts_n.max(dim=0).values
    # bbox = BoundingBox(mins_n[0].item(), maxs_n[0].item(),
    #                    mins_n[1].item(), maxs_n[1].item(),
    #                    mins_n[2].item(), maxs_n[2].item()).expand(ratio=1.2)
    #
    # ckpt = torch.load(pt_path, map_location=device)
    # plain_state_dict = restore_plain_state_dict(ckpt["model_state"])
    # domain = pyrfm.Square3D(center=(0.0, 0.0, 0.0), radius=(1, 1, 1))
    # model = pyrfm.RFMBase(dim=3, n_hidden=512, domain=domain, n_subdomains=1, rf=pyrfm.RFTanH2)
    # model.submodels[0].inner.weights = plain_state_dict["input_layer.0.weight"].t()
    # model.submodels[0].inner.biases = plain_state_dict["input_layer.0.bias"]
    # model.submodels[0].weights = plain_state_dict["hidden_layer.0.weight"].t()
    # model.submodels[0].biases = plain_state_dict["hidden_layer.0.bias"]
    # model.W = plain_state_dict["final_layer.weight"].t()
    #
    #
    # class NearShapeForViz(pyrfm.GeometryBase):
    #     def __init__(self): super().__init__(dim=3, intrinsic_dim=2)
    #
    #     def get_bounding_box(self): return bbox.get_bounding_box()
    #
    #     def in_sample(self, num_samples: int, with_boundary: bool = False): raise NotImplementedError
    #
    #     def on_sample(self, num_samples: int, with_normal: bool = False): raise NotImplementedError
    #
    #     def sdf(self, p: torch.Tensor) -> torch.Tensor:
    #         return model(p).squeeze()
    #
    #
    # near_shape = NearShapeForViz()
    # model.domain = near_shape
    #
    # save_isosurface_png_and_ply("figures/ellipsoid_tanh2_iso.png", "/dev/null",
    #                             model=model, bbox=bbox, level=0.0, grid=(256, 256, 256),
    #                             resolution=(800, 800), view="iso")
    # save_model_slice_png("figures/ellipsoid_tanh2_slice.png", model=model, bbox=bbox.get_bounding_box(),
    #                      axis="y", device=device)
    #
    # ## torus
    # pth_path = "../data/torus_in.pth"
    # pt_path = "./sec3_2/checkpoints/torus_in/tanh-tanh/sdf_best.pt"
    # x, normal, mean_curvature = torch.load(pth_path, map_location=device)
    # pts_t = x.to(device=device, dtype=dtype)
    # nrms_t = normal.to(device=device, dtype=dtype)
    # mins = pts_t.min(dim=0).values;
    # maxs = pts_t.max(dim=0).values
    # center = (mins + maxs) * 0.5;
    # half = (maxs - mins) * 0.5
    # scale = torch.max(half)
    # pts_n = (pts_t - center) / scale
    # nrms_n = torch.nn.functional.normalize(nrms_t, dim=-1, eps=1e-9)
    #
    # mins_n = pts_n.min(dim=0).values;
    # maxs_n = pts_n.max(dim=0).values
    # bbox = BoundingBox(mins_n[0].item(), maxs_n[0].item(),
    #                    mins_n[1].item(), maxs_n[1].item(),
    #                    mins_n[2].item(), maxs_n[2].item()).expand(ratio=1.2)
    #
    # ckpt = torch.load(pt_path, map_location=device)
    # plain_state_dict = restore_plain_state_dict(ckpt["model_state"])
    # domain = pyrfm.Square3D(center=(0.0, 0.0, 0.0), radius=(1, 1, 1))
    # model = pyrfm.RFMBase(dim=3, n_hidden=512, domain=domain, n_subdomains=1, rf=pyrfm.RFTanH2)
    # model.submodels[0].inner.weights = plain_state_dict["input_layer.0.weight"].t()
    # model.submodels[0].inner.biases = plain_state_dict["input_layer.0.bias"]
    # model.submodels[0].weights = plain_state_dict["hidden_layer.0.weight"].t()
    # model.submodels[0].biases = plain_state_dict["hidden_layer.0.bias"]
    # model.W = plain_state_dict["final_layer.weight"].t()
    #
    #
    # class NearShapeForViz(pyrfm.GeometryBase):
    #     def __init__(self): super().__init__(dim=3, intrinsic_dim=2)
    #
    #     def get_bounding_box(self): return bbox.get_bounding_box()
    #
    #     def in_sample(self, num_samples: int, with_boundary: bool = False): raise NotImplementedError
    #
    #     def on_sample(self, num_samples: int, with_normal: bool = False): raise NotImplementedError
    #
    #     def sdf(self, p: torch.Tensor) -> torch.Tensor:
    #         return model(p).squeeze()
    #
    #
    # near_shape = NearShapeForViz()
    # model.domain = near_shape
    #
    # save_isosurface_png_and_ply("figures/torus_tanh2_iso.png", "/dev/null",
    #                             model=model, bbox=bbox, level=0.0, grid=(256, 256, 256),
    #                             resolution=(800, 800), view="iso")
    # save_model_slice_png("figures/torus_tanh2_slice.png", model=model, bbox=bbox.get_bounding_box(),
    #                      axis="y", device=device)
    #
    # ## genus2
    # pth_path = "../data/genus2_in.pth"
    # pt_path = "./sec3_2/checkpoints/genus2_in/tanh-tanh/sdf_best.pt"
    # x, normal, mean_curvature = torch.load(pth_path, map_location=device)
    # pts_t = x.to(device=device, dtype=dtype)
    # nrms_t = normal.to(device=device, dtype=dtype)
    # mins = pts_t.min(dim=0).values;
    # maxs = pts_t.max(dim=0).values
    # center = (mins + maxs) * 0.5;
    # half = (maxs - mins) * 0.5
    # scale = torch.max(half)
    # pts_n = (pts_t - center) / scale
    # nrms_n = torch.nn.functional.normalize(nrms_t, dim=-1, eps=1e-9)
    #
    # mins_n = pts_n.min(dim=0).values;
    # maxs_n = pts_n.max(dim=0).values
    # bbox = BoundingBox(mins_n[0].item(), maxs_n[0].item(),
    #                    mins_n[1].item(), maxs_n[1].item(),
    #                    mins_n[2].item(), maxs_n[2].item()).expand(ratio=1.2)
    #
    # ckpt = torch.load(pt_path, map_location=device)
    # plain_state_dict = restore_plain_state_dict(ckpt["model_state"])
    # domain = pyrfm.Square3D(center=(0.0, 0.0, 0.0), radius=(1, 1, 1))
    # model = pyrfm.RFMBase(dim=3, n_hidden=512, domain=domain, n_subdomains=1, rf=pyrfm.RFTanH2)
    # model.submodels[0].inner.weights = plain_state_dict["input_layer.0.weight"].t()
    # model.submodels[0].inner.biases = plain_state_dict["input_layer.0.bias"]
    # model.submodels[0].weights = plain_state_dict["hidden_layer.0.weight"].t()
    # model.submodels[0].biases = plain_state_dict["hidden_layer.0.bias"]
    # model.W = plain_state_dict["final_layer.weight"].t()
    #
    #
    # class NearShapeForViz(pyrfm.GeometryBase):
    #     def __init__(self): super().__init__(dim=3, intrinsic_dim=2)
    #
    #     def get_bounding_box(self): return bbox.get_bounding_box()
    #
    #     def in_sample(self, num_samples: int, with_boundary: bool = False): raise NotImplementedError
    #
    #     def on_sample(self, num_samples: int, with_normal: bool = False): raise NotImplementedError
    #
    #     def sdf(self, p: torch.Tensor) -> torch.Tensor:
    #         return model(p).squeeze()
    #
    #
    # near_shape = NearShapeForViz()
    # model.domain = near_shape
    #
    # save_isosurface_png_and_ply("figures/genus2_tanh2_iso.png", "/dev/null",
    #                             model=model, bbox=bbox, level=0.0, grid=(256, 256, 256),
    #                             resolution=(800, 800), view="iso")
    # save_model_slice_png("figures/genus2_tanh2_slice.png", model=model, bbox=bbox.get_bounding_box(),
    #                      axis="y", device=device)
    #
    # ## cheese
    # pth_path = "../data/cheese_in.pth"
    # pt_path = "./sec3_2/checkpoints/cheese_in/tanh-tanh/sdf_best.pt"
    # x, normal, mean_curvature = torch.load(pth_path, map_location=device)
    # pts_t = x.to(device=device, dtype=dtype)
    # nrms_t = normal.to(device=device, dtype=dtype)
    # mins = pts_t.min(dim=0).values;
    # maxs = pts_t.max(dim=0).values
    # center = (mins + maxs) * 0.5;
    # half = (maxs - mins) * 0.5
    # scale = torch.max(half)
    # pts_n = (pts_t - center) / scale
    # nrms_n = torch.nn.functional.normalize(nrms_t, dim=-1, eps=1e-9)
    #
    # mins_n = pts_n.min(dim=0).values;
    # maxs_n = pts_n.max(dim=0).values
    # bbox = BoundingBox(mins_n[0].item(), maxs_n[0].item(),
    #                    mins_n[1].item(), maxs_n[1].item(),
    #                    mins_n[2].item(), maxs_n[2].item()).expand(ratio=1.2)
    #
    # ckpt = torch.load(pt_path, map_location=device)
    # plain_state_dict = restore_plain_state_dict(ckpt["model_state"])
    # domain = pyrfm.Square3D(center=(0.0, 0.0, 0.0), radius=(1, 1, 1))
    # model = pyrfm.RFMBase(dim=3, n_hidden=512, domain=domain, n_subdomains=1, rf=pyrfm.RFTanH2)
    # model.submodels[0].inner.weights = plain_state_dict["input_layer.0.weight"].t()
    # model.submodels[0].inner.biases = plain_state_dict["input_layer.0.bias"]
    # model.submodels[0].weights = plain_state_dict["hidden_layer.0.weight"].t()
    # model.submodels[0].biases = plain_state_dict["hidden_layer.0.bias"]
    # model.W = plain_state_dict["final_layer.weight"].t()
    #
    #
    # class NearShapeForViz(pyrfm.GeometryBase):
    #     def __init__(self): super().__init__(dim=3, intrinsic_dim=2)
    #
    #     def get_bounding_box(self): return bbox.get_bounding_box()
    #
    #     def in_sample(self, num_samples: int, with_boundary: bool = False): raise NotImplementedError
    #
    #     def on_sample(self, num_samples: int, with_normal: bool = False): raise NotImplementedError
    #
    #     def sdf(self, p: torch.Tensor) -> torch.Tensor:
    #         return model(p).squeeze()
    #
    #
    # near_shape = NearShapeForViz()
    # model.domain = near_shape
    #
    # save_isosurface_png_and_ply("figures/cheese_tanh2_iso.png", "/dev/null",
    #                             model=model, bbox=bbox, level=0.0, grid=(256, 256, 256),
    #                             resolution=(800, 800), view="iso")
    # save_model_slice_png("figures/cheese_tanh2_slice.png", model=model, bbox=bbox.get_bounding_box(),
    #                      axis="y", device=device)
    #
    # # bottle
    # pth_path = "../data/bottle_in.pth"
    # pt_path = "./sec3_2/checkpoints/bottle_in/tanh-tanh/sdf_best.pt"
    # x, normal, mean_curvature = torch.load(pth_path, map_location=device)
    # pts_t = x.to(device=device, dtype=dtype)
    # nrms_t = normal.to(device=device, dtype=dtype)
    # mins = pts_t.min(dim=0).values;
    # maxs = pts_t.max(dim=0).values
    # center = (mins + maxs) * 0.5;
    # half = (maxs - mins) * 0.5
    # scale = torch.max(half)
    # pts_n = (pts_t - center) / scale
    # nrms_n = torch.nn.functional.normalize(nrms_t, dim=-1, eps=1e-9)
    #
    # mins_n = pts_n.min(dim=0).values;
    # maxs_n = pts_n.max(dim=0).values
    # bbox = BoundingBox(mins_n[0].item(), maxs_n[0].item(),
    #                    mins_n[1].item(), maxs_n[1].item(),
    #                    mins_n[2].item(), maxs_n[2].item()).expand(ratio=1.2)
    #
    # ckpt = torch.load(pt_path, map_location=device)
    # plain_state_dict = restore_plain_state_dict(ckpt["model_state"])
    # domain = pyrfm.Square3D(center=(0.0, 0.0, 0.0), radius=(1, 1, 1))
    # model = pyrfm.RFMBase(dim=3, n_hidden=512, domain=domain, n_subdomains=1, rf=pyrfm.RFTanH2)
    # model.submodels[0].inner.weights = plain_state_dict["input_layer.0.weight"].t()
    # model.submodels[0].inner.biases = plain_state_dict["input_layer.0.bias"]
    # model.submodels[0].weights = plain_state_dict["hidden_layer.0.weight"].t()
    # model.submodels[0].biases = plain_state_dict["hidden_layer.0.bias"]
    # model.W = plain_state_dict["final_layer.weight"].t()
    #
    #
    # class NearShapeForViz(pyrfm.GeometryBase):
    #     def __init__(self): super().__init__(dim=3, intrinsic_dim=2)
    #
    #     def get_bounding_box(self): return bbox.get_bounding_box()
    #
    #     def in_sample(self, num_samples: int, with_boundary: bool = False): raise NotImplementedError
    #
    #     def on_sample(self, num_samples: int, with_normal: bool = False): raise NotImplementedError
    #
    #     def sdf(self, p: torch.Tensor) -> torch.Tensor:
    #         return model(p).squeeze()
    #
    #
    # near_shape = NearShapeForViz()
    # model.domain = near_shape
    #
    # save_isosurface_png_and_ply("figures/bottle_tanh2_iso.png", "/dev/null",
    #                             model=model, bbox=bbox, level=0.0, grid=(256, 256, 256),
    #                             resolution=(800, 800), view="top")
    # save_model_slice_png("figures/bottle_tanh2_slice.png", model=model, bbox=bbox.get_bounding_box(),
    #                      axis="z", device=device)
    #
    # # bunny
    # pth_path = "../data/bunny_in.pth"
    # pt_path = "./sec3_2/checkpoints/bunny_in/tanh-tanh/sdf_best.pt"
    # x, normal, mean_curvature = torch.load(pth_path, map_location=device)
    # pts_t = x.to(device=device, dtype=dtype)
    # nrms_t = normal.to(device=device, dtype=dtype)
    # mins = pts_t.min(dim=0).values;
    # maxs = pts_t.max(dim=0).values
    # center = (mins + maxs) * 0.5;
    # half = (maxs - mins) * 0.5
    # scale = torch.max(half)
    # pts_n = (pts_t - center) / scale
    # nrms_n = torch.nn.functional.normalize(nrms_t, dim=-1, eps=1e-9)
    #
    # mins_n = pts_n.min(dim=0).values;
    # maxs_n = pts_n.max(dim=0).values
    # bbox = BoundingBox(mins_n[0].item(), maxs_n[0].item(),
    #                    mins_n[1].item(), maxs_n[1].item(),
    #                    mins_n[2].item(), maxs_n[2].item()).expand(ratio=1.2)
    #
    # ckpt = torch.load(pt_path, map_location=device)
    # plain_state_dict = restore_plain_state_dict(ckpt["model_state"])
    # domain = pyrfm.Square3D(center=(0.0, 0.0, 0.0), radius=(1, 1, 1))
    # model = pyrfm.RFMBase(dim=3, n_hidden=512, domain=domain, n_subdomains=1, rf=pyrfm.RFTanH2)
    # model.submodels[0].inner.weights = plain_state_dict["input_layer.0.weight"].t()
    # model.submodels[0].inner.biases = plain_state_dict["input_layer.0.bias"]
    # model.submodels[0].weights = plain_state_dict["hidden_layer.0.weight"].t()
    # model.submodels[0].biases = plain_state_dict["hidden_layer.0.bias"]
    # model.W = plain_state_dict["final_layer.weight"].t()
    #
    #
    # class NearShapeForViz(pyrfm.GeometryBase):
    #     def __init__(self): super().__init__(dim=3, intrinsic_dim=2)
    #
    #     def get_bounding_box(self): return bbox.get_bounding_box()
    #
    #     def in_sample(self, num_samples: int, with_boundary: bool = False): raise NotImplementedError
    #
    #     def on_sample(self, num_samples: int, with_normal: bool = False): raise NotImplementedError
    #
    #     def sdf(self, p: torch.Tensor) -> torch.Tensor:
    #         return model(p).squeeze()
    #
    #
    # near_shape = NearShapeForViz()
    # model.domain = near_shape
    #
    # save_isosurface_png_and_ply("figures/bunny_tanh2_iso.png", "/dev/null",
    #                             model=model, bbox=bbox, level=0.0, grid=(256, 256, 256),
    #                             resolution=(800, 800), view="front")
    # save_model_slice_png("figures/bunny_tanh2_slice.png", model=model, bbox=bbox.get_bounding_box(),
    #                      axis="y", device=device)

    # # extract the last three lines from all log files under ./sec3_2/logs
    # from pathlib import Path
    # import os
    #
    #
    # def extract_last_lines(base_dir="./sec3_2"):
    #     base_path = Path(base_dir)
    #     log_files = base_path.rglob("logs/**/*.log")  # 递归查找 logs 下的 .log 文件
    #
    #     results = []
    #
    #     for log_file in log_files:
    #         try:
    #             with open(log_file, "r", encoding="utf-8") as f:
    #                 lines = f.readlines()
    #                 last_three = lines[-3:] if len(lines) >= 3 else lines
    #
    #             # 从路径中提取标签
    #             parts = log_file.parts
    #             # 例子: sec3_2/logs/cheese_in/relu-tanh/train_xxx.log
    #             bottle_like = ""
    #             tanh_like = ""
    #
    #             for p in parts:
    #                 if p.endswith("_in"):  # cheese_in, bottle_in 等
    #                     bottle_like = p.replace("_in", "")
    #                 if "-" in p:  # relu-tanh, tanh-tanh 等
    #                     tanh_like = p
    #
    #             results.append({
    #                 "file": log_file,
    #                 "label1": bottle_like,
    #                 "label2": tanh_like,
    #                 "last_lines": [line.strip() for line in last_three]
    #             })
    #
    #         except Exception as e:
    #             print(f"读取 {log_file} 出错: {e}")
    #
    #     # 打印结果
    #     for res in results:
    #         print(f"\n文件: {res['file']}")
    #         print(f"标签: {res['label1']} | {res['label2']}")
    #         print("最后三行:")
    #         for line in res["last_lines"]:
    #             print("    ", line)
    #
    #
    # extract_last_lines("./sec3_2")

    # section 3.3 results: compare tanh-tanh and tanh

    # cheese-like

    pth_path = "../data/cheese_in.pth"
    pt_path = "./sec3_2/checkpoints/cheese_in/tanh-tanh/sdf_best.pt"
    x, normal, mean_curvature = torch.load(pth_path, map_location=device)
    pts_t = x.to(device=device, dtype=dtype)
    nrms_t = normal.to(device=device, dtype=dtype)
    mins = pts_t.min(dim=0).values;
    maxs = pts_t.max(dim=0).values
    center = (mins + maxs) * 0.5;
    half = (maxs - mins) * 0.5
    scale = torch.max(half)

    pts_n = (pts_t - center) / scale
    nrms_n = torch.nn.functional.normalize(nrms_t, dim=-1, eps=1e-9)

    mins_n = pts_n.min(dim=0).values;
    maxs_n = pts_n.max(dim=0).values
    bbox = BoundingBox(mins_n[0].item(), maxs_n[0].item(),
                       mins_n[1].item(), maxs_n[1].item(),
                       mins_n[2].item(), maxs_n[2].item()).expand(ratio=1.2)

    ckpt = torch.load(pt_path, map_location=device)
    plain_state_dict = restore_plain_state_dict(ckpt["model_state"])
    domain = pyrfm.Square3D(center=(0.0, 0.0, 0.0), radius=(1, 1, 1))
    model = pyrfm.RFMBase(dim=3, n_hidden=512, domain=domain, n_subdomains=1, rf=pyrfm.RFTanH2)
    model.submodels[0].inner.weights = plain_state_dict["input_layer.0.weight"].t()
    model.submodels[0].inner.biases = plain_state_dict["input_layer.0.bias"]
    model.submodels[0].weights = plain_state_dict["hidden_layer.0.weight"].t()
    model.submodels[0].biases = plain_state_dict["hidden_layer.0.bias"]
    model.W = plain_state_dict["final_layer.weight"].t()


    class NearShapeForViz(pyrfm.GeometryBase):
        def __init__(self): super().__init__(dim=3, intrinsic_dim=2)

        def get_bounding_box(self): return bbox.get_bounding_box()

        def in_sample(self, num_samples: int, with_boundary: bool = False): raise NotImplementedError

        def on_sample(self, num_samples: int, with_normal: bool = False): raise NotImplementedError

        def sdf(self, p: torch.Tensor) -> torch.Tensor:
            return model(p).squeeze()


    near_shape = NearShapeForViz()
    model.domain = near_shape

    save_isosurface_png_and_ply("figures/cheese_tanh2_iso_normal.png", "/dev/null",
                                model=model, bbox=bbox, level=0.0, grid=(256, 256, 256),
                                resolution=(800, 800), view="iso")
    save_model_slice_png("figures/cheese_tanh2_slice.png", model=model, bbox=bbox.get_bounding_box(),
                         axis="y", device=device)

    pth_path = "../data/cheese_in.pth"
    pt_path = "./sec3_3/checkpoints/cheese_in/tanh/sdf_best.pt"
    x, normal, mean_curvature = torch.load(pth_path, map_location=device)
    pts_t = x.to(device=device, dtype=dtype)
    nrms_t = normal.to(device=device, dtype=dtype)
    mins = pts_t.min(dim=0).values;
    maxs = pts_t.max(dim=0).values
    center = (mins + maxs) * 0.5;
    half = (maxs - mins) * 0.5
    scale = torch.max(half)
    pts_n = (pts_t - center) / scale
    nrms_n = torch.nn.functional.normalize(nrms_t, dim=-1, eps=1e-9)

    mins_n = pts_n.min(dim=0).values;
    maxs_n = pts_n.max(dim=0).values
    bbox = BoundingBox(mins_n[0].item(), maxs_n[0].item(),
                       mins_n[1].item(), maxs_n[1].item(),
                       mins_n[2].item(), maxs_n[2].item()).expand(ratio=1.2)

    ckpt = torch.load(pt_path, map_location=device)
    plain_state_dict = restore_plain_state_dict(ckpt["model_state"])
    print(plain_state_dict.keys())
    domain = pyrfm.Square3D(center=(0.0, 0.0, 0.0), radius=(1, 1, 1))
    model = pyrfm.RFMBase(dim=3, n_hidden=512, domain=domain, n_subdomains=1, rf=pyrfm.RFTanH)
    model.submodels[0].weights = plain_state_dict["net.0.weight"].t()
    model.submodels[0].biases = plain_state_dict["net.0.bias"]
    model.W = plain_state_dict["net.2.weight"].t()


    class NearShapeForViz(pyrfm.GeometryBase):
        def __init__(self): super().__init__(dim=3, intrinsic_dim=2)

        def get_bounding_box(self): return bbox.get_bounding_box()

        def in_sample(self, num_samples: int, with_boundary: bool = False): raise NotImplementedError

        def on_sample(self, num_samples: int, with_normal: bool = False): raise NotImplementedError

        def sdf(self, p: torch.Tensor) -> torch.Tensor:
            return model(p).squeeze()


    near_shape = NearShapeForViz()
    model.domain = near_shape

    save_isosurface_png_and_ply("figures/cheese_tanh1_iso_normal.png", "/dev/null",
                                model=model, bbox=bbox, level=0.0, grid=(256, 256, 256),
                                resolution=(800, 800), view="iso")
    save_model_slice_png("figures/cheese_tanh1_slice.png", model=model, bbox=bbox.get_bounding_box(),
                         axis="y", device=device)

    # bunny
    pth_path = "../data/bunny_in.pth"
    pt_path = "./sec3_2/checkpoints/bunny_in/tanh-tanh/sdf_best.pt"
    x, normal, mean_curvature = torch.load(pth_path, map_location=device)
    pts_t = x.to(device=device, dtype=dtype)
    nrms_t = normal.to(device=device, dtype=dtype)
    mins = pts_t.min(dim=0).values;
    maxs = pts_t.max(dim=0).values
    center = (mins + maxs) * 0.5;
    half = (maxs - mins) * 0.5
    scale = torch.max(half)
    pts_n = (pts_t - center) / scale
    nrms_n = torch.nn.functional.normalize(nrms_t, dim=-1, eps=1e-9)

    mins_n = pts_n.min(dim=0).values;
    maxs_n = pts_n.max(dim=0).values
    bbox = BoundingBox(mins_n[0].item(), maxs_n[0].item(),
                       mins_n[1].item(), maxs_n[1].item(),
                       mins_n[2].item(), maxs_n[2].item()).expand(ratio=1.2)

    ckpt = torch.load(pt_path, map_location=device)
    plain_state_dict = restore_plain_state_dict(ckpt["model_state"])
    domain = pyrfm.Square3D(center=(0.0, 0.0, 0.0), radius=(1, 1, 1))
    model = pyrfm.RFMBase(dim=3, n_hidden=512, domain=domain, n_subdomains=1, rf=pyrfm.RFTanH2)
    model.submodels[0].inner.weights = plain_state_dict["input_layer.0.weight"].t()
    model.submodels[0].inner.biases = plain_state_dict["input_layer.0.bias"]
    model.submodels[0].weights = plain_state_dict["hidden_layer.0.weight"].t()
    model.submodels[0].biases = plain_state_dict["hidden_layer.0.bias"]
    model.W = plain_state_dict["final_layer.weight"].t()


    class NearShapeForViz(pyrfm.GeometryBase):
        def __init__(self): super().__init__(dim=3, intrinsic_dim=2)

        def get_bounding_box(self): return bbox.get_bounding_box()

        def in_sample(self, num_samples: int, with_boundary: bool = False): raise NotImplementedError

        def on_sample(self, num_samples: int, with_normal: bool = False): raise NotImplementedError

        def sdf(self, p: torch.Tensor) -> torch.Tensor:
            return model(p).squeeze()


    near_shape = NearShapeForViz()
    model.domain = near_shape

    save_isosurface_png_and_ply("figures/bunny_tanh2_front_normal.png", "/dev/null",
                                model=model, bbox=bbox, level=0.0, grid=(256, 256, 256),
                                resolution=(800, 800), view="front")
    save_model_slice_png("figures/bunny_tanh2_slice.png", model=model, bbox=bbox.get_bounding_box(),
                         axis="y", device=device)

    pth_path = "../data/bunny_in.pth"
    pt_path = "./sec3_3/checkpoints/bunny_in/tanh/sdf_best.pt"
    x, normal, mean_curvature = torch.load(pth_path, map_location=device)
    pts_t = x.to(device=device, dtype=dtype)
    nrms_t = normal.to(device=device, dtype=dtype)
    mins = pts_t.min(dim=0).values;
    maxs = pts_t.max(dim=0).values
    center = (mins + maxs) * 0.5;
    half = (maxs - mins) * 0.5
    scale = torch.max(half)
    pts_n = (pts_t - center) / scale
    nrms_n = torch.nn.functional.normalize(nrms_t, dim=-1, eps=1e-9)

    mins_n = pts_n.min(dim=0).values;
    maxs_n = pts_n.max(dim=0).values
    bbox = BoundingBox(mins_n[0].item(), maxs_n[0].item(),
                       mins_n[1].item(), maxs_n[1].item(),
                       mins_n[2].item(), maxs_n[2].item()).expand(ratio=1.2)

    ckpt = torch.load(pt_path, map_location=device)
    plain_state_dict = restore_plain_state_dict(ckpt["model_state"])
    print(plain_state_dict.keys())
    domain = pyrfm.Square3D(center=(0.0, 0.0, 0.0), radius=(1, 1, 1))
    model = pyrfm.RFMBase(dim=3, n_hidden=512, domain=domain, n_subdomains=1, rf=pyrfm.RFTanH)
    model.submodels[0].weights = plain_state_dict["net.0.weight"].t()
    model.submodels[0].biases = plain_state_dict["net.0.bias"]
    model.W = plain_state_dict["net.2.weight"].t()


    class NearShapeForViz(pyrfm.GeometryBase):
        def __init__(self): super().__init__(dim=3, intrinsic_dim=2)

        def get_bounding_box(self): return bbox.get_bounding_box()

        def in_sample(self, num_samples: int, with_boundary: bool = False): raise NotImplementedError

        def on_sample(self, num_samples: int, with_normal: bool = False): raise NotImplementedError

        def sdf(self, p: torch.Tensor) -> torch.Tensor:
            return model(p).squeeze()


    near_shape = NearShapeForViz()
    model.domain = near_shape

    save_isosurface_png_and_ply("figures/bunny_tanh1_front_normal.png", "/dev/null",
                                model=model, bbox=bbox, level=0.0, grid=(256, 256, 256),
                                resolution=(800, 800), view="front")
    save_model_slice_png("figures/bunny_tanh1_slice.png", model=model, bbox=bbox.get_bounding_box(),
                         axis="y", device=device)
