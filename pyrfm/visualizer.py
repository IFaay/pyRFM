# -*- coding: utf-8 -*-
"""
Created on 7/21/25

@author: Yifei Sun
"""
import os

import matplotlib.pyplot as plt
import numpy as np

from pyrfm.core import *


class VisualizerBase:
    def __init__(self):
        self.fig, self.ax = plt.subplots(dpi=150)

    def plot(self, *args, **kwargs):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def show(self, *args, **kwargs):
        # self.fig.show(*args, **kwargs)
        plt.show(*args, **kwargs)

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
    def __init__(self, model: RFMBase, resolution=(1920, 1080), component_idx=0):
        super().__init__()
        self.model = model
        self.resolution = resolution
        self.component_idx = component_idx
        self.bounding_box = model.domain.get_bounding_box()
        self.sdf = model.domain.sdf if hasattr(model.domain, 'sdf') else None


class RFMVisualizer2D(RFMVisualizer):
    def __init__(self, model: RFMBase, resolution=(1920, 1080), component_idx=0):
        super().__init__(model, resolution, component_idx)

    def plot(self, cmap='viridis', **kwargs):
        x = torch.linspace(self.bounding_box[0], self.bounding_box[1], self.resolution[0])
        y = torch.linspace(self.bounding_box[2], self.bounding_box[3], self.resolution[1])
        X, Y = torch.meshgrid(x, y, indexing='ij')
        grid_points = torch.column_stack([X.ravel(), Y.ravel()])

        Z = self.model(grid_points).detach().cpu().numpy()
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
    _CAMERA_TABLE = {
        'front': {'view_dir': torch.tensor([0.0, -1.0, 0.0]), 'up': torch.tensor([0.0, 0.0, 1.0])},
        'back': {'view_dir': torch.tensor([0.0, 1.0, 0.0]), 'up': torch.tensor([0.0, 0.0, 1.0])},
        'left': {'view_dir': torch.tensor([-1.0, 0.0, 0.0]), 'up': torch.tensor([0.0, 0.0, 1.0])},
        'right': {'view_dir': torch.tensor([1.0, 0.0, 0.0]), 'up': torch.tensor([0.0, 0.0, 1.0])},
        'top': {'view_dir': torch.tensor([0.0, 0.0, -1.0]), 'up': torch.tensor([0.0, 1.0, 0.0])},
        'bottom': {'view_dir': torch.tensor([0.0, 0.0, 1.0]), 'up': torch.tensor([0.0, 1.0, 0.0])},
        'iso': {'view_dir': torch.tensor([-1.0, -1.0, 1.0]), 'up': torch.tensor([0.0, 1.0, 1.0])},
        'front-right': {'view_dir': torch.tensor([0.5, -1.0, 0.0]), 'up': torch.tensor([0.0, 0.0, 1.0])},
        'front-left': {'view_dir': torch.tensor([-0.5, -1.0, 0.0]), 'up': torch.tensor([0.0, 0.0, 1.0])},
    }

    def __init__(self, model: RFMBase, resolution=(1920, 1080), component_idx=0, view='iso'):
        super().__init__(model, resolution, component_idx)
        cam = self._CAMERA_TABLE.get(str(view).lower())
        if cam is None:
            raise ValueError(f"Unknown view: {view}")
        view_dir = cam['view_dir']
        up = cam['up']
        self.view_dir = view_dir / torch.linalg.norm(view_dir)
        self.up = up / torch.linalg.norm(up)

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

    def ray_march(self, origins, directions, max_steps=128, epsilon=1e-3, far=100.0):
        hits = torch.zeros(origins.shape[:-1], dtype=torch.bool)
        t_vals = torch.zeros_like(hits, dtype=torch.float32)

        for step in range(max_steps):
            pts = origins + t_vals[..., None] * directions  # shape: (..., 3)
            dists = self.sdf(pts.reshape(-1, 3)).reshape(*pts.shape[:-1])  # (...,)

            mask = (dists > epsilon) & (t_vals < far)
            t_vals = torch.where(mask, t_vals + dists, t_vals)
            hits |= (dists < epsilon)

        return t_vals, hits

    def estimate_normal(self, pts, epsilon=1e-3):
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

        # Build coordinate offsets (shape: (3, 3))
        offsets = torch.eye(3, device=pts.device) * epsilon

        # Central finite differences for ∂SDF/∂x, ∂SDF/∂y, ∂SDF/∂z
        grads = []
        for i in range(3):
            d_plus = self.sdf((pts + offsets[i]).reshape(-1, 3)).reshape(pts.shape[:-1])
            d_minus = self.sdf((pts - offsets[i]).reshape(-1, 3)).reshape(pts.shape[:-1])
            grads.append((d_plus - d_minus) / (2 * epsilon))

        # Stack into a vector field and normalize
        normal = torch.stack(grads, dim=-1)  # (..., 3)
        normal = normal / torch.clamp(torch.norm(normal, dim=-1, keepdim=True), min=1e-8)
        return normal

    def plot(self, cmap='viridis', **kwargs):
        directions = self.generate_rays()  # (W, H, 3)
        bbox = self.bounding_box
        center = torch.tensor([
            (bbox[0] + bbox[1]) / 2,
            (bbox[2] + bbox[3]) / 2,
            (bbox[4] + bbox[5]) / 2,
        ])
        diag_len = max(max(bbox[1] - bbox[0], bbox[3] - bbox[2]), bbox[5] - bbox[4])
        # view_dir = self.get_view_matrix() @ torch.tensor([0.0, 0.0, 1.0])
        eye = center + self.view_dir * (1.2 * diag_len + 0.1)
        origins = eye[None, None, :].expand(*directions.shape[:2], 3)

        t_vals, hits = self.ray_march(origins, directions)
        pts_hit = origins + t_vals.unsqueeze(-1) * directions
        pts_normal = self.estimate_normal(pts_hit)
        field_vals = self.model(pts_hit.reshape(-1, 3)).detach().cpu().numpy()[:, self.component_idx]
        field_vals[~hits.numpy().ravel()] = np.nan

        vmin = np.nanmin(field_vals)
        vmax = np.nanmax(field_vals)
        normed = (field_vals - vmin) / (vmax - vmin)
        normed = np.clip(normed, 0.0, 1.0)

        cmap = plt.get_cmap(cmap)
        base = cmap(normed.reshape(self.resolution))[..., :3]
        base = torch.tensor(base, dtype=pts_normal.dtype, device=pts_normal.device)
        light_dir = self.view_dir + torch.tensor([-1.0, -0.0, 1.0], dtype=pts_normal.dtype,
                                                 device=pts_normal.device)
        light_dir /= torch.norm(light_dir)
        view_dir = self.view_dir
        half_vector = (light_dir + view_dir).unsqueeze(0).unsqueeze(0)
        half_vector = half_vector / torch.norm(half_vector, dim=-1, keepdim=True)
        diff = torch.clamp(
            torch.sum(pts_normal * light_dir[None, None, :], dim=-1), min=0.0
        )
        spec = torch.clamp(torch.sum(pts_normal * half_vector, dim=-1), min=0.0)
        spec = torch.pow(spec, 32)
        col = (0.8 * base + 0.2) * diff[..., None] + base * 0.3 + spec[..., None] * 0.5
        col = torch.clamp(col, 0.0, 1.0)
        col[~hits] = 1.0  # background color (white)
        colors = col.cpu().numpy()

        self.ax.imshow(colors.transpose(1, 0, 2), origin='lower', interpolation='bilinear')
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        plt.colorbar(sm, ax=self.ax)
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
        axes_3d = {
            'X': (torch.tensor([1.0, 0.0, 0.0]), 'red'),
            'Y': (torch.tensor([0.0, 1.0, 0.0]), 'green'),
            'Z': (torch.tensor([0.0, 0.0, 1.0]), 'blue')
        }

        # Get bounding box center and camera vectors
        bbox = self.bounding_box
        center = torch.tensor([
            (bbox[0] + bbox[1]) / 2,
            (bbox[2] + bbox[3]) / 2,
            (bbox[4] + bbox[5]) / 2,
        ])
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
            end = base + dir2d.numpy()
            self.ax.annotate(
                '', xy=end, xytext=base, xycoords='axes fraction',
                textcoords='axes fraction',
                arrowprops=dict(arrowstyle='-|>', lw=2.5, color=color, alpha=0.8)
            )
            self.ax.text(
                end[0], end[1], label, transform=trans,
                fontsize=10, color=color, fontweight='bold',
                ha='center', va='center'
            )
