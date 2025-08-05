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
    _CAMERA_TABLE = {
        'front': {'view_dir': torch.tensor([0.0, -1.0, 0.0]), 'up': torch.tensor([0.0, 0.0, 1.0])},
        'back': {'view_dir': torch.tensor([0.0, 1.0, 0.0]), 'up': torch.tensor([0.0, 0.0, 1.0])},
        'left': {'view_dir': torch.tensor([-1.0, 0.0, 0.0]), 'up': torch.tensor([0.0, 0.0, 1.0])},
        'right': {'view_dir': torch.tensor([1.0, 0.0, 0.0]), 'up': torch.tensor([0.0, 0.0, 1.0])},
        'top': {'view_dir': torch.tensor([0.0, 0.0, 1.0]), 'up': torch.tensor([0.0, 1.0, 0.0])},
        'bottom': {'view_dir': torch.tensor([0.0, 0.0, -1.0]), 'up': torch.tensor([0.0, 1.0, 0.0])},
        'iso': {'view_dir': torch.tensor([-1.0, -1.0, 1.25]), 'up': torch.tensor([0.5, 0.5, 1 / 1.25])},
        'front-right': {'view_dir': torch.tensor([0.5, -1.0, 0.0]), 'up': torch.tensor([0.0, 0.0, 1.0])},
        'front-left': {'view_dir': torch.tensor([-0.5, -1.0, 0.0]), 'up': torch.tensor([0.0, 0.0, 1.0])},
    }

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

        field_vals = self.compute_field_values(pts_normal, hits)
        # field_vals = torch.norm(pts_hit.reshape(-1, 3) - eye, dim=-1).cpu().numpy()  # debug: distance to eye
        field_vals[~hits.detach().cpu().numpy().ravel()] = np.nan

        # vmin = np.nanmin(field_vals)
        # vmax = np.nanmax(field_vals)
        vmin = np.nanpercentile(field_vals, 2)
        vmax = np.nanpercentile(field_vals, 98)
        normed = (field_vals - vmin) / (vmax - vmin)
        normed = np.clip(normed, 0.0, 1.0)

        cmap = plt.get_cmap(cmap)
        base = cmap(normed.reshape(self.resolution))[..., :3]
        base = torch.tensor(base, dtype=self.dtype, device=self.device)
        light_dir = self.view_dir + torch.tensor([-1.0, -0.0, 1.0], dtype=self.dtype,
                                                 device=self.device)
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
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
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
            end = base + dir2d.detach().cpu().numpy()
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
