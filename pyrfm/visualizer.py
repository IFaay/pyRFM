# -*- coding: utf-8 -*-
"""
Created on 7/21/25

@author: Yifei Sun
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from skimage.measure import marching_cubes as mc
except Exception as e:
    mc = None
from pyrfm.core import *


class VisualizerBase:
    def __init__(self):
        self.fig, self.ax = plt.subplots(dpi=150)

    def plot(self, *args, **kwargs):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def show(self, *args, **kwargs):
        # self.fig.show(*args, **kwargs)
        plt.show(*args, **kwargs)

    def close(self, *args, **kwargs):
        plt.close(self.fig)

    def savefig(self, fname, dpi=600, *args, **kwargs):
        self.fig.savefig(fname=fname, dpi=dpi, *args, **kwargs)

    def xlabel(self, label, **kwargs):
        self.ax.set_xlabel(label, **kwargs)

    def ylabel(self, label, **kwargs):
        self.ax.set_ylabel(label, **kwargs)

    def title(self, title, **kwargs):
        self.ax.set_title(title, **kwargs)

    def xlim(self, left=None, right=None):
        self.ax.set_xlim(left, right)

    def ylim(self, bottom=None, top=None):
        self.ax.set_ylim(bottom, top)

    def grid(self, b=None, **kwargs):
        self.ax.grid(b=b, **kwargs)

    def axis_equal(self):
        self.ax.set_aspect('auto', adjustable='box')

    def xticks(self, ticks, labels=None, **kwargs):
        self.ax.set_xticks(ticks)
        if labels is not None:
            self.ax.set_xticklabels(labels, **kwargs)


class RFMVisualizer(VisualizerBase):
    def __init__(self, model: Union[RFMBase, STRFMBase], t=0.0, resolution=(1920, 1080), component_idx=0,
                 ref: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
        super().__init__()
        self.dtype = torch.tensor(0.).dtype
        self.device = torch.tensor(0.).device
        self.model = model
        self.resolution = resolution
        self.component_idx = component_idx
        self.bounding_box = model.domain.get_bounding_box()
        self.t = t

        self.sdf = model.domain.sdf if hasattr(model.domain, 'sdf') else None
        if ref is not None:
            if callable(ref):
                self.ref: Optional[..., torch.Tensor] = ref
            else:
                raise TypeError("ref must be a callable function.")
        else:
            self.ref = None


class RFMVisualizer2D(RFMVisualizer):
    def __init__(self, model: Union[RFMBase, STRFMBase], t=0.0, resolution=(1920, 1080), component_idx=0, ref=None):
        super().__init__(model, t, resolution, component_idx, ref)

    def compute_field_vals(self, grid_points):
        if isinstance(self.model, RFMBase):
            if self.ref is not None:
                Z = (self.model(grid_points) - self.ref(grid_points)).abs().detach().cpu().numpy()
            else:
                Z = self.model(grid_points).detach().cpu().numpy()
        elif isinstance(self.model, STRFMBase):
            xt = self.model.validate_and_prepare_xt(x=grid_points, t=torch.tensor([[self.t]]))
            if self.ref is not None:
                Z = (self.model.forward(xt=xt) - self.ref(xt=xt)).abs().detach().cpu().numpy()
            else:
                Z = self.model.forward(xt=xt).detach().cpu().numpy()

        else:
            raise NotImplementedError

        return Z

    def plot(self, cmap='viridis', **kwargs):
        x = torch.linspace(self.bounding_box[0], self.bounding_box[1], self.resolution[0])
        y = torch.linspace(self.bounding_box[2], self.bounding_box[3], self.resolution[1])
        X, Y = torch.meshgrid(x, y, indexing='ij')
        grid_points = torch.column_stack([X.ravel(), Y.ravel()])

        Z = self.compute_field_vals(grid_points)
        Z = Z[:, self.component_idx].reshape(X.shape)

        # mark SDF > 0 as white
        if self.sdf is not None:
            sdf_values = self.sdf(grid_points).detach().cpu().numpy().reshape(X.shape)
            Z[sdf_values > 0] = np.nan
        Z = Z.T[::-1]

        self.ax.imshow(Z, extent=self.bounding_box, origin='lower', cmap=cmap, aspect='auto', **kwargs)
        self.ax.set_xlabel('X-axis')
        self.ax.set_ylabel('Y-axis')
        # add colorbar
        self.fig.colorbar(self.ax.images[0], ax=self.ax)


class RFMVisualizer3D(RFMVisualizer):
    _CAMERA_TABLE = {'front': {'view_dir': torch.tensor([0.0, -1.0, 0.0]), 'up': torch.tensor([0.0, 0.0, 1.0])},
                     'back': {'view_dir': torch.tensor([0.0, 1.0, 0.0]), 'up': torch.tensor([0.0, 0.0, 1.0])},
                     'left': {'view_dir': torch.tensor([-1.0, 0.0, 0.0]), 'up': torch.tensor([0.0, 0.0, 1.0])},
                     'right': {'view_dir': torch.tensor([1.0, 0.0, 0.0]), 'up': torch.tensor([0.0, 0.0, 1.0])},
                     'top': {'view_dir': torch.tensor([0.0, 0.0, 1.0]), 'up': torch.tensor([0.0, 1.0, 0.0])},
                     'bottom': {'view_dir': torch.tensor([0.0, 0.0, -1.0]), 'up': torch.tensor([0.0, 1.0, 0.0])},
                     'iso': {'view_dir': torch.tensor([-1.0, -1.0, 1.25]), 'up': torch.tensor([0.5, 0.5, 1 / 1.25])},
                     'front-right': {'view_dir': torch.tensor([0.5, -1.0, 0.0]), 'up': torch.tensor([0.0, 0.0, 1.0])},
                     'front-left': {'view_dir': torch.tensor([-0.5, -1.0, 0.0]), 'up': torch.tensor([0.0, 0.0, 1.0])}, }

    def __init__(self, model: RFMBase, t=0.0, resolution=(1920, 1080), component_idx=0, view='iso', ref=None):
        super().__init__(model, t, resolution, component_idx, ref)
        cam = self._CAMERA_TABLE.get(str(view).lower())
        if cam is None:
            raise ValueError(f"Unknown view: {view}")
        view_dir = cam['view_dir']
        up = cam['up']
        self.view_dir = (view_dir / torch.linalg.norm(view_dir)).to(dtype=self.dtype, device=self.device)
        self.up = (up / torch.linalg.norm(up)).to(dtype=self.dtype, device=self.device)

    def generate_rays(self):
        W, H = self.resolution
        i, j = torch.meshgrid(torch.linspace(0, W - 1, W), torch.linspace(0, H - 1, H), indexing='ij')
        uv = torch.stack([(i - W / 2) / H, (j - H / 2) / H], dim=-1)  # shape: (W, H, 2)

        # Compute camera basis
        forward = -self.view_dir
        right = torch.linalg.cross(forward, self.up)
        dirs = forward[None, None, :] + uv[..., 0:1] * right + uv[..., 1:2] * self.up
        dirs /= torch.linalg.norm(dirs, dim=-1, keepdim=True)
        return dirs

    def ray_march(self, origins, directions, max_steps=128, epsilon=1e-3, far=200.0):
        if self.ref is not None:
            max_steps = 512
            bbox = self.bounding_box
            diag_len = max(max(bbox[1] - bbox[0], bbox[3] - bbox[2]), bbox[5] - bbox[4])
            epsilon = torch.finfo(self.dtype).eps * (10.0 + diag_len)
        hits = torch.zeros(origins.shape[:-1], dtype=torch.bool, device=self.device)
        t_vals = torch.zeros_like(hits, dtype=self.dtype, device=self.device)

        for step in range(max_steps):
            # Current sample positions along each ray
            pts = origins + t_vals[..., None] * directions  # (..., 3)

            # Signed‑distance values at those positions
            dists = self.sdf(pts.reshape(-1, 3)).reshape(*pts.shape[:-1])

            # Hit detection: surface reached when |SDF| < epsilon
            # hit_mask = torch.abs(dists) < epsilon
            hit_mask = torch.abs(dists) <= epsilon
            hits |= hit_mask

            # Continue marching only on rays that have not yet hit
            # and are still within the far clipping distance
            active_mask = (~hits) & (t_vals < far)
            if not active_mask.any():
                break  # All rays terminated

            step_scale = 1.0
            t_vals = torch.where(active_mask, t_vals + dists * step_scale, t_vals)

        return t_vals, hits

    def estimate_normal(self, pts, epsilon=1e-4):
        """
        Estimate outward normals at given 3‑D points using central finite differences
        of the domain's signed‑distance function (SDF).

        Parameters
        ----------
        pts : torch.Tensor
            Tensor of shape (..., 3) containing query positions.
        epsilon : float, optional
            Finite‑difference step size used for gradient estimation.

        Returns
        -------
        torch.Tensor
            Normalized normal vectors with the same leading shape as `pts`.
        """
        # SDF must be available to compute gradients
        if self.sdf is None:
            raise RuntimeError("Domain SDF is not defined; cannot estimate normals.")

        try:
            _, normal = self.sdf(pts.reshape(-1, 3), with_normal=True)
            return normal.reshape(*pts.shape[:-1], 3)
        except TypeError:
            pass  # Fallback to finite differences if with_normal is not supported

        # Build coordinate offsets (shape: (3, 3))
        offsets = torch.eye(3, device=self.device) * epsilon

        # Central finite differences for ∂SDF/∂x, ∂SDF/∂y, ∂SDF/∂z
        grads = []
        for i in range(3):
            d_plus = self.sdf((pts + offsets[i]).reshape(-1, 3)).reshape(pts.shape[:-1])
            d_minus = self.sdf((pts - offsets[i]).reshape(-1, 3)).reshape(pts.shape[:-1])
            grads.append((d_plus - d_minus) / (2 * epsilon))

        # Stack into a vector field and normalize
        normal = torch.stack(grads, dim=-1)  # (..., 3)
        normal = normal / torch.clamp(torch.norm(normal, dim=-1, keepdim=True), min=1e-10)
        return normal

    def compute_field_values(self, pts_hit, hits):
        if isinstance(self.model, RFMBase):
            if self.ref is not None:
                field_vals = self.model(pts_hit.reshape(-1, 3))
                ref_vals = self.ref(pts_hit.reshape(-1, 3))
                field_vals[hits.ravel()] -= ref_vals[hits.ravel()]
                field_vals = field_vals.abs().detach().cpu().numpy()[:, self.component_idx]
            else:
                field_vals = self.model(pts_hit.reshape(-1, 3)).detach().cpu().numpy()[:, self.component_idx]
        elif isinstance(self.model, STRFMBase):
            xt = self.model.validate_and_prepare_xt(x=pts_hit.reshape(-1, 3), t=torch.tensor([[self.t]]))
            if self.ref is not None:
                field_vals = self.model.forward(xt=xt)
                ref_vals = self.ref(xt=xt)
                field_vals[hits.ravel()] -= ref_vals[hits.ravel()]
                field_vals = field_vals.abs().detach().cpu().numpy()[:, self.component_idx]
            else:
                field_vals = self.model.forward(xt=xt).detach().cpu().numpy()[:, self.component_idx]

        else:
            raise NotImplementedError("Model type not supported for visualization.")

        return field_vals

    def plot(self, cmap='viridis', **kwargs):
        directions = self.generate_rays()  # (W, H, 3)
        bbox = self.bounding_box
        center = torch.tensor([(bbox[0] + bbox[1]) / 2, (bbox[2] + bbox[3]) / 2, (bbox[4] + bbox[5]) / 2, ])
        diag_len = max(max(bbox[1] - bbox[0], bbox[3] - bbox[2]), bbox[5] - bbox[4])
        # view_dir = self.get_view_matrix() @ torch.tensor([0.0, 0.0, 1.0])
        eye = center + self.view_dir * (1.2 * diag_len + 1)
        origins = eye[None, None, :].expand(*directions.shape[:2], 3)

        t_vals, hits = self.ray_march(origins, directions)
        pts_hit = origins + t_vals.unsqueeze(-1) * directions
        pts_normal = self.estimate_normal(pts_hit)

        field_vals = self.compute_field_values(pts_hit, hits)
        # field_vals = torch.norm(pts_hit.reshape(-1, 3) - eye, dim=-1).cpu().numpy()  # debug: distance to eye
        field_vals[~hits.detach().cpu().numpy().ravel()] = np.nan

        # vmin = np.nanmin(field_vals)
        # vmax = np.nanmax(field_vals)
        vmin = np.nanpercentile(field_vals, 1)
        vmax = np.nanpercentile(field_vals, 99)
        normed = (field_vals - vmin) / (vmax - vmin)
        normed = np.clip(normed, 0.0, 1.0)

        cmap = plt.get_cmap(cmap)
        base = cmap(normed.reshape(self.resolution))[..., :3]
        base = torch.tensor(base, dtype=self.dtype, device=self.device)
        light_dir = self.view_dir + torch.tensor([-1.0, -0.0, 1.0], dtype=self.dtype, device=self.device)
        light_dir /= torch.norm(light_dir)
        view_dir = self.view_dir
        half_vector = (light_dir + view_dir).unsqueeze(0).unsqueeze(0)
        half_vector = half_vector / torch.norm(half_vector, dim=-1, keepdim=True)
        diff = torch.clamp(torch.sum(pts_normal * light_dir[None, None, :], dim=-1), min=0.0)
        spec = torch.clamp(torch.sum(pts_normal * half_vector, dim=-1), min=0.0)
        spec = torch.pow(spec, 32)
        col = (0.8 * base + 0.2) * diff[..., None] + base * 0.3 + spec[..., None] * 0.5
        col = torch.clamp(col, 0.0, 1.0)
        col[~hits] = 1.0  # background color (white)
        colors = col.cpu().numpy()

        self.ax.imshow(colors.transpose(1, 0, 2), origin='lower', interpolation='bilinear')
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        plt.colorbar(sm, ax=self.ax, shrink=0.6, pad=-0.05)
        self.ax.set_axis_off()
        plt.tight_layout()

        self.draw_view_axes()

        return self.fig, self.ax

    def draw_view_axes(self, length=0.5, offset=0.1):
        """
        Draws a 3D coordinate axes indicator aligned with the view direction,
        projected using the same camera setup as the main plot.
        """

        # Define 3D coordinate axes
        axes_3d = {'X': (torch.tensor([1.0, 0.0, 0.0]), 'red'), 'Y': (torch.tensor([0.0, 1.0, 0.0]), 'green'),
                   'Z': (torch.tensor([0.0, 0.0, 1.0]), 'blue')}

        # Get bounding box center and camera vectors
        bbox = self.bounding_box
        center = torch.tensor([(bbox[0] + bbox[1]) / 2, (bbox[2] + bbox[3]) / 2, (bbox[4] + bbox[5]) / 2, ])
        diag_len = max(max(bbox[1] - bbox[0], bbox[3] - bbox[2]), bbox[5] - bbox[4])
        forward = -self.view_dir
        right = torch.linalg.cross(forward, self.up)
        right = right / torch.norm(right)
        up = torch.linalg.cross(right, forward)
        origin = center + self.view_dir * (1.2 * diag_len + 0.05)

        # Project function: perspective projection with depth
        def project(pt3):
            rel = pt3 - origin
            depth = torch.dot(rel, forward)
            scale = 1.0 / (1.0 + 0.4 * depth)
            x = torch.dot(rel, right) * scale
            y = torch.dot(rel, up) * scale
            return torch.tensor([x.item(), y.item()]), depth

        base = np.array([offset, offset])
        trans = self.ax.transAxes

        axes_draw = []
        for label, (vec, color) in axes_3d.items():
            tip = center + vec * diag_len * 0.25
            p0, _ = project(center)
            p1, d1 = project(tip)
            axes_draw.append((d1.item(), label, vec, color, p0, p1))
        axes_draw.sort(reverse=True)  # Sort by depth from farthest to nearest

        for d1, label, vec, color, p0, p1 in axes_draw:
            dir2d = p1 - p0
            if torch.norm(dir2d) < 1e-5:
                continue
            dir2d = dir2d * length
            end = base + dir2d.detach().cpu().numpy()
            self.ax.annotate('', xy=end, xytext=base, xycoords='axes fraction', textcoords='axes fraction',
                             arrowprops=dict(arrowstyle='-|>', lw=2.5, color=color, alpha=0.8))
            self.ax.text(end[0], end[1], label, transform=trans, fontsize=10, color=color, fontweight='bold',
                         ha='center', va='center')


class RFMVisualizer3DMC(RFMVisualizer3D):
    """
    Marching Cubes 可视化，与 RFMVisualizer3D 的参数和相机模型保持一致。
    - 使用 SDF 的 0 等值面。
    - 颜色映射与光照模型与 ray marching 版本一致。
    - 投影与像素坐标系与 generate_rays() 一致（同一 pinhole 相机）。
    """

    def __init__(self, model: RFMBase, t=0.0, resolution=(1920, 1080), component_idx=0, view='iso', ref=None):
        super().__init__(model, t, resolution, component_idx, view, ref)

    # --------- 核心：在 bbox 上采样 SDF，提取 0 等值面 ---------
    def _eval_sdf_grid(self, bbox, grid=(128, 128, 128), chunk_pts=300_000):
        """
        在边界盒上以规则网格采样 SDF。
        返回:
            volume: (Nz, Ny, Nx) numpy float32
            axes: (xs, ys, zs) 1D numpy arrays
            spacing: (dz, dy, dx) floats (供 marching_cubes)
        """
        if self.sdf is None:
            raise RuntimeError("需要可用的 self.sdf 来进行 Marching Cubes。")

        xmin, xmax, ymin, ymax, zmin, zmax = bbox
        Nx, Ny, Nz = grid  # 注意：这里用 (Nx, Ny, Nz) 的语义，但 volume 存储为 (Nz, Ny, Nx)

        xs = torch.linspace(xmin, xmax, Nx, device=self.device, dtype=self.dtype)
        ys = torch.linspace(ymin, ymax, Ny, device=self.device, dtype=self.dtype)
        zs = torch.linspace(zmin, zmax, Nz, device=self.device, dtype=self.dtype)

        # 生成 (Nz, Ny, Nx, 3) 的点（最后一维按 (x,y,z) 排列）
        zz, yy, xx = torch.meshgrid(zs, ys, xs, indexing='ij')
        pts = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)

        volume = torch.empty((Nz * Ny * Nx,), device=self.device, dtype=self.dtype)

        def _sdf_as_1d(x: torch.Tensor) -> torch.Tensor:
            """
            统一 SDF 输出为 (N,)，并确保在需要时启用梯度。
            注意：geometry.sdf 内部会用 autograd.grad，所以必须 enable_grad + requires_grad(True)。
            """
            # 确保需要梯度
            return self.sdf(x).reshape(-1)
            # x = x.requires_grad_(True)
            # with torch.enable_grad():
            #     out = self.sdf(x)
            # # 兼容返回 (dist, normal) 或 list/tuple 的情况
            # if isinstance(out, (tuple, list)):
            #     out = out[0]
            # if not torch.is_tensor(out):
            #     out = torch.tensor(out, device=x.device)
            # out = out.to(device=x.device, dtype=self.dtype)
            # out = out.reshape(-1)  # (N,1)/(N,) -> (N,)
            # return out.detach()  # 立刻与图断开，避免图增大

        n_total = pts.shape[0]
        for start in range(0, n_total, chunk_pts):
            end = min(start + chunk_pts, n_total)
            vals = _sdf_as_1d(pts[start:end])
            assert vals.numel() == (end - start), f"SDF batch size mismatch: got {vals.shape}"
            volume[start:end] = vals

        volume = volume.reshape(Nz, Ny, Nx).detach().cpu().numpy().astype(np.float64)
        dx = (xmax - xmin) / max(1, Nx - 1)
        dy = (ymax - ymin) / max(1, Ny - 1)
        dz = (zmax - zmin) / max(1, Nz - 1)
        return volume, (xs.detach().cpu().numpy(), ys.detach().cpu().numpy(), zs.detach().cpu().numpy()), (dz, dy, dx)

    @torch.no_grad()
    def _compute_field_values_points(self, pts_world):
        """
        复用你在 ray-marching 版本中的字段取值逻辑，但针对任意点集合。
        返回 numpy (N,) 的标量数组（取 component_idx 分量；若 ref 存在，做绝对差）。
        """
        pts_t = torch.tensor(pts_world, device=self.device, dtype=self.dtype)
        if isinstance(self.model, RFMBase):
            if self.ref is not None:
                field_vals = self.model(pts_t)
                ref_vals = self.ref(pts_t)
                print(field_vals.shape, ref_vals.shape)
                print((field_vals - ref_vals).abs().max())
                field_vals = (field_vals - ref_vals).abs().detach().cpu().numpy()[:, self.component_idx]
            else:
                field_vals = self.model(pts_t).detach().cpu().numpy()[:, self.component_idx]
        elif isinstance(self.model, STRFMBase):
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

    # --------- 相机投影（与 generate_rays() 完全一致的针孔模型） ---------
    def _project_points(self, pts_world, eye, forward, right, up_cam):
        """
        将 3D 点投影到像素坐标：
          uv_x = dot((p-eye), right) / dot((p-eye), forward)
          uv_y = dot((p-eye), up)   / dot((p-eye), forward)
          i = uv_x * H + W/2,  j = uv_y * H + H/2
        返回:
          pix: (N,2) 像素坐标 (i, j)
          depth: (N,) 与 forward 的投影距离（越大越远）
        """
        W, H = self.resolution
        P = pts_world.shape[0]

        p = torch.tensor(pts_world, device=self.device, dtype=self.dtype)
        rel = p - eye[None, :]
        depth = torch.sum(rel * forward[None, :], dim=-1)  # 前方为正
        # 避免除零/背面：留给上层剔除
        uvx = torch.sum(rel * right[None, :], dim=-1) / depth
        uvy = torch.sum(rel * up_cam[None, :], dim=-1) / depth

        i = uvx * H + W / 2.0
        j = uvy * H + H / 2.0
        pix = torch.stack([i, j], dim=-1)
        return pix, depth

    def plot(self, cmap='viridis', level=0.0, grid=(128, 128, 128), chunk_pts=300_000, vmin=None, vmax=None, **kwargs):
        """
        Marching Cubes 绘制（视角/亮度与 ray-marching 版完全对齐）
        参数：
            cmap:    colormap 名称
            level:   等值面（默认 0.0，SDF 零水平面）
            grid:    体素网格分辨率 (Nx, Ny, Nz)
            chunk_pts: SDF 批评估大小
        """
        # 依赖检查
        try:
            from skimage.measure import marching_cubes as _mc
        except Exception:
            _mc = None
        if _mc is None:
            if 'mc' not in globals() or mc is None:
                raise RuntimeError("需要 scikit-image：请先 `pip install scikit-image`。")

        # ---- 相机 / 视图（与 ray 版一致）----
        W, H = self.resolution
        bbox = self.bounding_box
        center = torch.tensor([(bbox[0] + bbox[1]) / 2,
                               (bbox[2] + bbox[3]) / 2,
                               (bbox[4] + bbox[5]) / 2], device=self.device, dtype=self.dtype)
        diag_len = max(max(bbox[1] - bbox[0], bbox[3] - bbox[2]), bbox[5] - bbox[4])
        eye = center + self.view_dir * (1.2 * diag_len + 0.1)

        forward = -self.view_dir
        right = torch.linalg.cross(forward, self.up)  # 不归一化，保持与 generate_rays 一致
        up_cam = self.up

        # ---- 体素化 + Marching Cubes ----
        volume, (xs, ys, zs), (dz, dy, dx) = self._eval_sdf_grid(bbox, grid=grid, chunk_pts=chunk_pts)
        try:
            verts_v, faces, _norms_unused, _ = mc(volume, level=level, spacing=(dz, dy, dx))
        except Exception:
            raise RuntimeError("在当前网格或等值面参数下未提取到有效表面，请调整 level 或 grid。")

        # skimage 顶点坐标顺序 (z,y,x) -> 世界坐标 (x,y,z)
        zmin, ymin, xmin = zs[0], ys[0], xs[0]
        verts_world = np.column_stack([
            verts_v[:, 2] + xmin,
            verts_v[:, 1] + ymin,
            verts_v[:, 0] + zmin
        ])

        # ---- 顶点法线（SDF 梯度）----
        vnorm = self.estimate_normal(torch.tensor(verts_world, device=self.device, dtype=self.dtype))
        # 单位化（保险）
        vnorm = vnorm / torch.clamp(torch.norm(vnorm, dim=-1, keepdim=True), min=1e-10)

        # ---- 标量场 -> colormap 基色（顶点）----
        vfield = self._compute_field_values_points(verts_world)  # numpy (N,)
        vmin = np.nanpercentile(vfield, 1) if vmin is None else vmin
        vmax = np.nanpercentile(vfield, 99) if vmax is None else vmax
        denom = (vmax - vmin) if (vmax > vmin) else 1.0
        vnormed = np.clip((vfield - vmin) / denom, 0.0, 1.0)

        cmap_obj = plt.get_cmap(cmap)
        base_rgb_np = cmap_obj(vnormed)[..., :3]  # numpy (N,3)
        base_rgb = torch.tensor(base_rgb_np, device=self.device, dtype=self.dtype)  # torch (N,3)

        # ---- 光照向量（与 ray 版一致）----
        light_dir = self.view_dir + torch.tensor([-1.0, -0.0, 1.0], dtype=self.dtype, device=self.device)
        light_dir = light_dir / torch.norm(light_dir)
        view_dir = self.view_dir
        half_vec = (light_dir + view_dir)
        half_vec = half_vec / torch.norm(half_vec)

        # ---- 投影到像素坐标 ----
        pix_t, depth_t = self._project_points(verts_world, eye, forward, right, up_cam)  # torch
        pix = pix_t.detach().cpu().numpy()
        depth = depth_t.detach().cpu().numpy()

        # 剔除后方顶点，并据此过滤面
        valid_v = depth > 1e-6
        valid_f = valid_v[faces].all(axis=1)
        if not np.any(valid_f):
            raise RuntimeError("所有三角形都被相机剔除了（可能视角在模型内部/背面）。请调整 view 或 bbox。")
        faces = faces[valid_f]

        # ---- 面级着色：与 ray 版系数完全一致 ----
        tri2d = pix[faces]  # (F,3,2)
        tri_depth = depth[faces].mean(axis=1)  # (F,)

        # 面级 base（平均顶点 colormap）
        tri_base_t = base_rgb[faces].mean(axis=1)  # torch (F,3)

        # 面级法线（平均后单位化）
        tri_n = vnorm[faces].mean(axis=1)  # torch (F,3)
        tri_n = tri_n / torch.clamp(torch.norm(tri_n, dim=-1, keepdim=True), min=1e-10)

        diff_f = torch.clamp((tri_n * light_dir).sum(dim=-1), min=0.0)  # (F,)
        spec_f = torch.clamp((tri_n * half_vec).sum(dim=-1), min=0.0) ** 32  # (F,)

        # 与 ray 版一致的合成公式（无 gamma）
        tri_color = (0.8 * tri_base_t + 0.2) * diff_f[:, None] + \
                    tri_base_t * 0.3 + \
                    spec_f[:, None] * 0.5

        tri_color = torch.clamp(tri_color, 0.0, 1.0).detach().cpu().numpy()

        # Painter：远->近
        order = np.argsort(tri_depth)[::-1]
        tri2d = tri2d[order]
        tri_color = tri_color[order]

        # ---- 绘制 ----
        self.ax.clear()
        from matplotlib.collections import PolyCollection
        from matplotlib.ticker import ScalarFormatter, FuncFormatter, FormatStrFormatter
        polys = [tri for tri in tri2d]
        coll = PolyCollection(polys, facecolors=tri_color, edgecolors='none', closed=True, antialiased=False,
                              linewidths=0)
        self.ax.add_collection(coll)
        self.ax.set_xlim([0, W])
        self.ax.set_ylim([0, H])
        self.ax.set_aspect('equal')
        # 不翻转 y 轴；保持与 imshow(origin='lower') 一致
        self.ax.set_axis_off()

        # 背景白
        self.ax.set_facecolor((1, 1, 1))

        # 颜色条（字段值）
        sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cb = plt.colorbar(sm, ax=self.ax, shrink=0.6, pad=-0.05)

        # —— 关键部分：科学计数法 ——
        formatter = ScalarFormatter(useMathText=True)  # 用 1×10^{k} 的数学字体
        formatter.set_powerlimits((0, 0))  # 强制所有刻度都用科学计数
        cb.formatter = formatter

        cb.update_ticks()  # 应用到 colorbar
        plt.tight_layout()
        self.draw_view_axes()

        return self.fig, self.ax

    def save_ply(self, filepath,
                 level=0.0,
                 grid=(128, 128, 128),
                 chunk_pts=300_000,
                 cmap='viridis',
                 color_mode='lit',  # 'field' 或 'lit'
                 binary=True):
        """
        将 Marching Cubes 提取的网格保存为 PLY，包含：顶点位置/法向/颜色 + 三角面。
        参数：
            filepath: 输出路径（.ply）
            level:    等值面，默认 0.0
            grid:     体素网格 (Nx, Ny, Nz)
            chunk_pts:SDF 批大小
            cmap:     colormap 名称（用于 'field' 与 'lit' 基色）
            color_mode: 'field'（仅按标量场着色）或 'lit'（烘焙光照）
            binary:   True -> binary_little_endian；False -> ASCII
        """
        import numpy as np
        import torch
        import matplotlib.pyplot as plt

        # ---- 体素化 + Marching Cubes 与 plot() 对齐 ----
        try:
            from skimage.measure import marching_cubes as _mc
        except Exception:
            _mc = None
        if _mc is None:
            if 'mc' not in globals() or mc is None:
                raise RuntimeError("需要 scikit-image：请先 `pip install scikit-image`。")

        bbox = self.bounding_box
        volume, (xs, ys, zs), (dz, dy, dx) = self._eval_sdf_grid(bbox, grid=grid, chunk_pts=chunk_pts)
        try:
            verts_v, faces, _norms_unused, _ = mc(volume, level=level, spacing=(dz, dy, dx))
        except Exception:
            raise RuntimeError("在当前网格或等值面参数下未提取到有效表面，请调整 level 或 grid。")

        # skimage 顶点坐标顺序 (z,y,x) -> 世界坐标 (x,y,z)
        zmin, ymin, xmin = zs[0], ys[0], xs[0]
        verts_world = np.column_stack([
            verts_v[:, 2] + xmin,
            verts_v[:, 1] + ymin,
            verts_v[:, 0] + zmin
        ])  # (V,3) float64

        # ---- 顶点法向（SDF 梯度）----
        vnorm_t = self.estimate_normal(torch.tensor(verts_world, device=self.device, dtype=self.dtype))
        vnorm_t = vnorm_t / torch.clamp(torch.norm(vnorm_t, dim=-1, keepdim=True), min=1e-10)
        vnorm = vnorm_t.detach().cpu().numpy().astype(np.float64)  # (V,3)

        # ---- 标量场 -> colormap 基色（顶点）----
        vfield = self._compute_field_values_points(verts_world)  # numpy (V,)
        vmin = np.nanpercentile(vfield, 1)
        vmax = np.nanpercentile(vfield, 99.1)
        denom = (vmax - vmin) if (vmax > vmin) else 1.0
        vnormed = np.clip((vfield - vmin) / denom, 0.0, 1.0)

        cmap_obj = plt.get_cmap(cmap)
        base_rgb = cmap_obj(vnormed)[..., :3].astype(np.float64)  # (V,3) in [0,1]

        # ---- 颜色模式：field or lit（与 plot() 的光照尽量一致，但按顶点近似）----
        if color_mode not in ('field', 'lit'):
            raise ValueError("color_mode 只能是 'field' 或 'lit'")

        if color_mode == 'lit':
            # 与 plot() 保持一致的光照向量
            light_dir = self.view_dir + torch.tensor([-1.0, -0.0, 1.0], dtype=self.dtype, device=self.device)
            light_dir = light_dir / torch.norm(light_dir)
            view_dir = self.view_dir
            half_vec = (light_dir + view_dir)
            half_vec = half_vec / torch.norm(half_vec)

            n = vnorm_t  # (V,3) torch
            diff = torch.clamp((n * light_dir).sum(dim=-1), min=0.0)  # (V,)
            spec = torch.clamp((n * half_vec).sum(dim=-1), min=0.0) ** 32  # (V,)

            base_rgb_t = torch.tensor(base_rgb, device=self.device, dtype=self.dtype)  # (V,3)
            rgb_lit = (0.8 * base_rgb_t + 0.2) * diff[:, None] + \
                      base_rgb_t * 0.3 + \
                      spec[:, None] * 0.5
            rgb_lit = torch.clamp(rgb_lit, 0.0, 1.0).detach().cpu().numpy().astype(np.float64)
            rgb = rgb_lit
        else:
            rgb = base_rgb  # 直接用字段 colormap

        # 转为 0-255 uint8
        rgb_u8 = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)  # (V,3)

        # ---- 写 PLY：顶点位置/法向/颜色 + 面 ----
        V = verts_world.shape[0]
        F = faces.shape[0]

        # ---- 写 PLY（二进制）修正版：逐面交错写入 ----
        if binary:
            header = (
                "ply\n"
                "format binary_little_endian 1.0\n"
                f"element vertex {V}\n"
                "property float x\n"
                "property float y\n"
                "property float z\n"
                "property float nx\n"
                "property float ny\n"
                "property float nz\n"
                "property uchar red\n"
                "property uchar green\n"
                "property uchar blue\n"
                f"element face {F}\n"
                "property list uchar int vertex_indices\n"
                "end_header\n"
            ).encode('ascii')

            # 顶点打包
            vert_pack = np.empty(V, dtype=[
                ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                ('nx', '<f4'), ('ny', '<f4'), ('nz', '<f4'),
                ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
            ])
            vert_pack['x'] = verts_world[:, 0].astype(np.float64)
            vert_pack['y'] = verts_world[:, 1].astype(np.float64)
            vert_pack['z'] = verts_world[:, 2].astype(np.float64)
            vert_pack['nx'] = vnorm[:, 0]
            vert_pack['ny'] = vnorm[:, 1]
            vert_pack['nz'] = vnorm[:, 2]
            vert_pack['red'] = rgb_u8[:, 0]
            vert_pack['green'] = rgb_u8[:, 1]
            vert_pack['blue'] = rgb_u8[:, 2]

            # 面索引检查 & 转小端 int32
            faces_i32 = faces.astype('<i4', copy=False)
            if faces_i32.min() < 0 or faces_i32.max() >= V:
                raise ValueError(f"Face index out of range: min={faces_i32.min()} max={faces_i32.max()} V={V}")

            with open(filepath, 'wb') as f:
                f.write(header)
                vert_pack.tofile(f)

                # 逐面交错写： [uchar(3), int32, int32, int32] * F
                # 用结构化 dtype 一次性写，效率更高也更不易出错
                face_dtype = np.dtype([('n', 'u1'),
                                       ('i0', '<i4'), ('i1', '<i4'), ('i2', '<i4')])
                face_pack = np.empty(F, dtype=face_dtype)
                face_pack['n'] = 3
                face_pack['i0'] = faces_i32[:, 0]
                face_pack['i1'] = faces_i32[:, 1]
                face_pack['i2'] = faces_i32[:, 2]
                face_pack.tofile(f)
        else:
            # ASCII
            header = (
                "ply\n"
                "format ascii 1.0\n"
                f"element vertex {V}\n"
                "property float x\n"
                "property float y\n"
                "property float z\n"
                "property float nx\n"
                "property float ny\n"
                "property float nz\n"
                "property uchar red\n"
                "property uchar green\n"
                "property uchar blue\n"
                f"element face {F}\n"
                "property list uchar int vertex_indices\n"
                "end_header\n"
            )
            with open(filepath, 'w') as f:
                f.write(header)
                for i in range(V):
                    x, y, z = verts_world[i]
                    nx, ny, nz = vnorm[i]
                    r, g, b = rgb_u8[i]
                    f.write(f"{x:.6f} {y:.6f} {z:.6f} {nx:.6f} {ny:.6f} {nz:.6f} {int(r)} {int(g)} {int(b)}\n")
                for i0, i1, i2 in faces:
                    f.write(f"3 {int(i0)} {int(i1)} {int(i2)}\n")

        return {
            "vertices": V,
            "faces": F,
            "filepath": filepath,
            "mode": "binary_little_endian" if binary else "ascii",
            "color_mode": color_mode,
            "vmin_vmax": (float(vmin), float(vmax))
        }
