# -*- coding: utf-8 -*-
"""
Created on 2025/10/8

@author: Yifei Sun
"""

"""
Train a time-dependent random feature model for laplace-beltrami operator

A classical problem in surface PDEs is the surface diffusion equation (𝓛 = Δₛ):

    ∂ₜ u(x, t) = Δₛ u(x, t) + f(x, t),   x ∈ Γ,  t ∈ (0, T].

To evaluate accuracy, we consider the exact solution:
    u(x, y, z, t) = sin(x + sin(t)) · exp(cos(y − z)),
from which the initial condition and source term f can be derived.

The Crank–Nicolson (CN) scheme is:
    (uⁿ⁺¹ − uⁿ)/Δt = ½ [ 𝓛(uⁿ) + 𝓛(uⁿ⁺¹) ] + ½ [ fⁿ + fⁿ⁺¹ ],

where Δt is the time step size, uⁿ = u(x, nΔt), fⁿ = f(x, nΔt).

Rearranging gives the system for uⁿ⁺¹:
    (I − Δt/2 · 𝓛) uⁿ⁺¹ = (I + Δt/2 · 𝓛) uⁿ + Δt/2 (fⁿ + fⁿ⁺¹).

"""

from collections import defaultdict
import re
import torch
import numpy as np
import pyrfm
from typing import Union, Tuple, List, Optional, Dict

from pathlib import Path
import matplotlib

# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.colors import TwoSlopeNorm

import time


def compute_rhs_fast(model, p: torch.Tensor, p0: torch.Tensor, t: torch.Tensor, dt, A_lap_beltrami):
    u_n = model(p)
    f_np12 = func_f(p, p0, t - dt / 2, normal, mean_curvature)
    b = u_n + (dt / 2) * (A_lap_beltrami @ model.W) + dt * f_np12
    return b


def compute_rhs(model, p: torch.Tensor, p0: torch.Tensor, t: torch.Tensor, dt, normal: torch.Tensor,
                mean_curvature: torch.Tensor) -> torch.Tensor:
    """
    Compute the right-hand side of the CN scheme:
        b = (I + Δt/2 · 𝓛) uⁿ + Δt/2 (fⁿ + fⁿ⁺¹)
    where 𝓛 = Δₛ is the Laplace–Beltrami operator.
    """
    u_n = model(p)
    # f_n = func_f(p, t - dt, normal, mean_curvature)
    # f_np1 = func_f(p, t, normal, mean_curvature)
    f_np12 = func_f(p, p0, t - dt / 2, normal, mean_curvature)
    A_lap_beltrami = compute_laplace_beltrami_matrix(model, p, normal, mean_curvature)
    b = u_n + (dt / 2) * (A_lap_beltrami @ model.W) + dt * f_np12
    # b = u_n + (dt / 2) * (A_lap_beltrami @ model.W) + dt * (f_n + f_np1) / 2
    return b


def func_u(p: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Example function u(x, y, z) = sin(x + sin(t)) · exp(cos(y − z))
    """
    return torch.zeros(p.shape[0], 1)


def func_f(p: torch.Tensor, p0: torch.Tensor, t: torch.Tensor, normal: torch.Tensor,
           mean_curvature: torch.Tensor) -> torch.Tensor:
    """
    f(𝐱) = exp(−4‖𝐱 − 𝐱₀‖)
    """
    return torch.exp(-(4 * (p - p0).norm(dim=1, keepdim=True, p=2) ** 2))


def compute_laplace_beltrami_matrix(
        model: pyrfm.RFMBase,
        x_in: torch.Tensor,
        normal: torch.Tensor,
        mean_curvature: torch.Tensor,
) -> torch.Tensor:
    A_xx = model.features_second_derivative(x_in, axis1=0, axis2=0).cat(dim=1)
    A_yy = model.features_second_derivative(x_in, axis1=1, axis2=1).cat(dim=1)
    A_zz = model.features_second_derivative(x_in, axis1=2, axis2=2).cat(dim=1)
    A_xy = model.features_second_derivative(x_in, axis1=0, axis2=1).cat(dim=1)
    A_xz = model.features_second_derivative(x_in, axis1=0, axis2=2).cat(dim=1)
    A_yz = model.features_second_derivative(x_in, axis1=1, axis2=2).cat(dim=1)

    A_lap = A_xx + A_yy + A_zz

    # 手动展开 einsum('ni,nijk,nj->nk') 等价于：
    A_nHn = (
            normal[:, 0:1] * (A_xx * normal[:, 0:1] + A_xy * normal[:, 1:2] + A_xz * normal[:, 2:3]) +
            normal[:, 1:2] * (A_xy * normal[:, 0:1] + A_yy * normal[:, 1:2] + A_yz * normal[:, 2:3]) +
            normal[:, 2:3] * (A_xz * normal[:, 0:1] + A_yz * normal[:, 1:2] + A_zz * normal[:, 2:3])
    )

    # 可选释放内存
    del A_xx, A_yy, A_zz, A_xy, A_xz, A_yz
    torch.cuda.empty_cache()

    A_in_x = model.features_derivative(x_in, axis=0).cat(dim=1)
    A_in_y = model.features_derivative(x_in, axis=1).cat(dim=1)
    A_in_z = model.features_derivative(x_in, axis=2).cat(dim=1)

    A_grad = torch.stack([A_in_x, A_in_y, A_in_z], dim=1)
    A_partial_n = (
            normal[:, 0:1] * A_in_x +
            normal[:, 1:2] * A_in_y +
            normal[:, 2:3] * A_in_z
    )
    del A_in_x, A_in_y, A_in_z
    torch.cuda.empty_cache()

    A_lap_beltrami = A_lap - 2 * mean_curvature * A_partial_n - A_nHn

    return A_lap_beltrami


class PreciseRFMVisualizer3DMC(pyrfm.RFMVisualizer3DMC):
    @torch.no_grad()
    def _compute_field_values_points(self, pts_world):
        """
        复用你在 ray-marching 版本中的字段取值逻辑，但针对任意点集合。
        返回 numpy (N,) 的标量数组（取 component_idx 分量；若 ref 存在，做绝对差）。
        """
        pts_t = torch.tensor(pts_world, device=self.device, dtype=self.dtype)

        pts_t = self._project_to_surface(
            pts_t,
            max_iter=40,  # 可按需调大
            atol=torch.finfo(self.dtype).eps,
            rtol=torch.finfo(self.dtype).eps,  # 与几何尺度相对
            batch_size=1 << 15  # 避免 OOM
        )
        if isinstance(self.model, pyrfm.RFMBase):
            if self.ref is not None:
                field_vals = self.model(pts_t)
                ref_vals = self.ref(pts_t)
                print(field_vals.shape, ref_vals.shape)
                print((field_vals - ref_vals).abs().max())
                field_vals = (field_vals - ref_vals).abs().detach().cpu().numpy()[:, self.component_idx]
            else:
                field_vals = self.model(pts_t).detach().cpu().numpy()[:, self.component_idx]
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

    @torch.no_grad()
    def _project_to_surface(
            self,
            pts: torch.Tensor,
            max_iter: int = 40,
            atol: float = None,
            rtol: float = None,
            batch_size: int = 1 << 15,
    ) -> torch.Tensor:
        """
        用 self.sdf 的有符号距离与单位法向，将任意 3D 点更精确地投影到 φ=0 曲面上。
        停止条件：max(|d|) < max(atol, rtol * L)，其中 L 为几何特征尺度（来自包围盒直径）。
        """
        # 1) 设备/精度对齐
        pts = pts.to(device=self.device, dtype=self.dtype)

        # 2) 估计几何尺寸（来自包围盒）
        try:
            x_min, x_max, y_min, y_max, z_min, z_max = self.get_bounding_box()
            L = float(max(x_max - x_min, y_max - y_min, z_max - z_min))
        except Exception:
            # fallback：单位尺度
            L = 1.0
        if atol is None:
            # 绝对阈值随 dtype/尺度自适应
            atol = max(torch.finfo(self.dtype).eps * L * 10, 1e-12)

        # 3) 分批迭代
        N = pts.shape[0]
        out = torch.empty_like(pts)
        for i0 in range(0, N, batch_size):
            i1 = min(N, i0 + batch_size)
            p = pts[i0:i1].clone()

            for _ in range(max_iter):
                # sdf: (B,1); n: (B,3)
                sdf_val, n = self.sdf(p, with_normal=True)
                if sdf_val.ndim == 2 and sdf_val.size(-1) == 1:
                    sdf_val = sdf_val.squeeze(-1)  # (B,)

                # 步长：d * n
                step = (sdf_val.unsqueeze(-1)) * n
                p = p - step

                # 收敛检查（相对+绝对）
                thresh = max(atol, rtol * L)
                if torch.max(sdf_val.abs()).item() < thresh:
                    break

            out[i0:i1] = p

        return out


class NormalNormRFMVisualizer3DMC(pyrfm.RFMVisualizer3DMC):
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


class MeanCurvatureRFMVisualizer3DMC(pyrfm.RFMVisualizer3DMC):
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


def save_isosurface_png_and_ply(
        save_png_path: Union[str, Path],
        save_ply_path: Union[str, Path],
        *,
        model,
        bbox,
        level: float = 0.0,
        grid: Tuple[int, int, int] = (128, 128, 128),
        resolution: Tuple[int, int] = (800, 800),
        view: str = "front",
        ref=None,
        vmin=None,
        vmax=None
):
    # viz = pyrfm.RFMVisualizer3DMC(model, t=0.0, resolution=resolution, component_idx=0, view=view, ref=ref)
    viz = PreciseRFMVisualizer3DMC(model, t=0.0, resolution=resolution, component_idx=0, view=view, ref=ref)
    # viz = NormalNormRFMVisualizer3DMC(model, t=0.0, resolution=resolution, component_idx=0, view=view, ref=ref)
    # viz = MeanCurvatureRFMVisualizer3DMC(model, t=0.0, resolution=resolution, component_idx=0, view=view, ref=ref)
    viz.plot(cmap="viridis", level=level, grid=grid, vmin=vmin, vmax=vmax)

    save_png_path = Path(save_png_path);
    save_png_path.parent.mkdir(parents=True, exist_ok=True)
    viz.savefig(str(save_png_path))
    viz.show()
    viz.close()

    # save_ply_path = Path(save_ply_path);
    # save_ply_path.parent.mkdir(parents=True, exist_ok=True)
    # viz.save_ply(str(save_ply_path))
    #
    # try:
    #     import matplotlib.pyplot as _plt
    #     _plt.close('all')
    # except Exception:
    #     pass
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


if __name__ == '__main__':
    torch.set_default_device('cuda') if torch.cuda.is_available() else torch.set_default_device('cpu')
    device = torch.tensor(0.0).device
    dtype = torch.tensor(0.0).dtype

    for shape in ["bunny"]:
        print(shape)
        pth_path = '../../data/{}_m.pth'.format(shape)
        pt_path = "../../sec3/sec3_2/checkpoints/{}_in/tanh-tanh/sdf_best.pt".format(shape)
        ckpt = torch.load(pt_path, map_location=device)
        plain_state_dict = restore_plain_state_dict(ckpt["model_state"])
        domain = pyrfm.Square3D(center=(0.0, 0.0, 0.0), radius=(1, 1, 1))
        model = pyrfm.RFMBase(dim=3, n_hidden=512, domain=domain, n_subdomains=1, rf=pyrfm.RFTanH2)
        model.submodels[0].inner.weights = plain_state_dict["input_layer.0.weight"].t()
        model.submodels[0].inner.biases = plain_state_dict["input_layer.0.bias"]
        model.submodels[0].weights = plain_state_dict["hidden_layer.0.weight"].t()
        model.submodels[0].biases = plain_state_dict["hidden_layer.0.bias"]
        # model.W = plain_state_dict["final_layer.weight"].t()

        x_in, normal, mean_curvature = torch.load(pth_path, map_location=device)

        mins = x_in.min(dim=0).values
        maxs = x_in.max(dim=0).values
        bbox = BoundingBox(mins[0].item(), maxs[0].item(),
                           mins[1].item(), maxs[1].item(),
                           mins[2].item(), maxs[2].item()).expand(
            ratio=1.5 if shape == "cheese" else 1.1 if shape == "genus2" else 1.2)
        ## adjust ratio to look better
        idx_max_y = torch.argmin(x_in[:, 1])  # 单个索引
        p_max_y = x_in[idx_max_y]  # 对应的点 (1,3)

        # -----------------------------------

        # , 5e-2, 2e-2, 1e-2, 5e-3
        for t_end in [1.0, 0.8, 0.6, 0.4, 0.2]:
            for dt in [5e-3]:
                t0 = time.time()
                t = 0.0

                print("dt = {:.4e}".format(dt))

                import io
                import sys

                backup = sys.stdout
                sys.stdout = io.StringIO()

                u0 = func_u(x_in, torch.tensor([[t]]))
                A = model.features(x_in).cat(dim=1)
                b = u0
                model.compute(A).solve(b)

                A_lap_beltrami = compute_laplace_beltrami_matrix(model, x_in, normal, mean_curvature)
                A = model.features(x_in).cat(dim=1) - (dt / 2) * A_lap_beltrami
                model.compute(A, damp=1e-14)

                while t < t_end:
                    """
                    (I − Δt/2 · 𝓛) uⁿ⁺¹
                    """
                    t += dt
                    print("t = {:.6f}".format(t))

                    # b = compute_rhs(model, x_in, torch.tensor([[t]]), dt, normal, mean_curvature)
                    # model.compute(A, damp=1e-14).solve(b)

                    b = compute_rhs_fast(model, x_in, p_max_y, torch.tensor([[t]]), dt, A_lap_beltrami)
                    model.solve(b)

                sys.stdout = backup

                # -----------------------------------
                # 🧮 计时结束
                t3 = time.time()

                print(f'[Timer] Total time: {t3 - t0:.2f} seconds')


                class NearShapeForViz(pyrfm.ImplicitSurfaceBase):
                    def __init__(self, model, domain):
                        super().__init__()
                        self.model = model  # 这里一定要挂上支持 dForward 的 RFM 模型
                        self._domain = domain

                    def get_bounding_box(self):
                        return bbox.get_bounding_box()

                    def shape_func(self, p: torch.Tensor) -> torch.Tensor:
                        # 返回模型预测的 SDF
                        return self.model(p).squeeze(-1)


                shape_model = pyrfm.RFMBase(dim=3, n_hidden=512, domain=domain, n_subdomains=1, rf=pyrfm.RFTanH2)
                shape_model.submodels[0].inner.weights = plain_state_dict["input_layer.0.weight"].t()
                shape_model.submodels[0].inner.biases = plain_state_dict["input_layer.0.bias"]
                shape_model.submodels[0].weights = plain_state_dict["hidden_layer.0.weight"].t()
                shape_model.submodels[0].biases = plain_state_dict["hidden_layer.0.bias"]
                shape_model.W = plain_state_dict["final_layer.weight"].t()

                near_shape = NearShapeForViz(shape_model, domain)

                model.domain = near_shape

                save_isosurface_png_and_ply("../figures/{}_heat_{:.6g}.png".format(shape, t), "/dev/null",
                                            model=model, bbox=domain, level=0.0, grid=(256, 256, 256),
                                            resolution=(800, 800), view="front", vmin=0.0, vmax=0.2)
