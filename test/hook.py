# -*- coding: utf-8 -*-
"""
Created on 2025/9/26

@author: Yifei Sun
"""
from cgitb import Hook
from typing import List, Tuple, Union
import math
import torch

import pyrfm
# If pyrfm exposes the following symbols, import them here.
from pyrfm import GeometryBase, CircleArc2D, Line2D


class Hook3D(GeometryBase):
    """
    3D 钩子（实体）：(大椭球 - 小椭球) 与一段逐渐变细的竖直胶囊平滑并联，再减去 Z 轴方向的盖帽圆柱孔。

    约定（与给定 GLSL 一致）：
      • 世界坐标 pWorld 先做 TRS 到局部：先平移到 origin，再绕 X 旋转、再绕 Z 旋转，最后做等比缩放的“逆变换”，
        即 p_local = applyTRS(pWorld, origin, rotXZ, scale)。
      • SDF 在局部空间计算，最后把距离乘回 scale。

    主要参数（均在“局部空间”中定义，数值默认与题干一致）：
      - origin, rot_xz(rx, rz), scale : 世界→局部的 TRS
      - B_C, B_R : 大椭球（主体）
      - T_C, T_R : 小椭球（挖空）
      - SMOOTH_K : 平滑并联 smin 的参数
      - TIP_C, TIP_H, TIP_R0, TIP_RT : 顶端胶囊（竖直 +y）基点、高度、基半径、线性锥度
      - HOLE_C, HOLE_HH, HOLE_R : 眼孔（沿 Z 的盖帽圆柱）中心、半高和半径
    """

    # ------------------------- construction ------------------------- #
    def __init__(self,
                 origin=(0.0, -0.2, 0.0),
                 rot_xz=(0.0, 0.0),  # (rx, rz)
                 scale: float = 1.0,
                 # body (big ellipsoid - small ellipsoid)
                 B_C=(-0.30, -0.40, 0.0),
                 B_R=(0.70, 0.80, 10.0),
                 T_C=(-0.30, -0.15, 0.0),
                 T_R=(0.80, 0.90, 0.10),
                 # blending
                 SMOOTH_K: float = 0.20,
                 # tip capsule (with mild taper r(y))
                 TIP_C=(0.35, 0.00, 0.0),
                 TIP_H: float = 3.0,
                 TIP_R0: float = 0.10,
                 TIP_RT: float = 0.05,
                 # eye hole (capped cylinder along Z)
                 HOLE_C=(-0.35, 3.00, 0.00),
                 HOLE_HH: float = 0.20,
                 HOLE_R: float = 0.05):
        super().__init__(dim=3, intrinsic_dim=3)

        t = lambda x: torch.as_tensor(x, dtype=self.dtype, device=self.device).view(1, -1)
        s = lambda x: torch.as_tensor(float(x), dtype=self.dtype, device=self.device)

        # TRS
        self.origin = t(origin)  # (1,3)
        self.rot_xz = t(rot_xz).view(1, 2)
        self.scale = s(scale)

        # body
        self.B_C = t(B_C)
        self.B_R = t(B_R)
        self.T_C = t(T_C)
        self.T_R = t(T_R)

        # blending
        self.SMOOTH_K = s(SMOOTH_K)

        # tip
        self.TIP_C = t(TIP_C)
        self.TIP_H = s(TIP_H)
        self.TIP_R0 = s(TIP_R0)
        self.TIP_RT = s(TIP_RT)

        # hole
        self.HOLE_C = t(HOLE_C)
        self.HOLE_HH = s(HOLE_HH)
        self.HOLE_R = s(HOLE_R)

        # conservative world bbox
        self._bbox_min, self._bbox_max = self._compute_bbox()

    # --------------------------- helpers ---------------------------- #
    def _world_to_local(self, p: torch.Tensor) -> torch.Tensor:
        # p: (N,3) -> 先平移，再绕 X、绕 Z，最后等比缩放逆（与 GLSL applyTRS 同序）
        q = p + self.origin
        rx, rz = self.rot_xz[0, 0], self.rot_xz[0, 1]
        cx, sx = torch.cos(rx), torch.sin(rx)
        cz, sz = torch.cos(rz), torch.sin(rz)
        R_x = torch.tensor([[1.0, 0.0, 0.0],
                            [0.0, cx, -sx],
                            [0.0, sx, cx]], dtype=p.dtype, device=p.device)
        R_z = torch.tensor([[cz, -sz, 0.0],
                            [sz, cz, 0.0],
                            [0.0, 0.0, 1.0]], dtype=p.dtype, device=p.device)
        q = q @ R_x.T
        q = q @ R_z.T
        q = q / torch.clamp(self.scale, min=torch.finfo(self.dtype).eps)
        return q

    @staticmethod
    def _smin_poly(d1: torch.Tensor, d2: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        # iq 的 polynomial smooth-min
        h = torch.clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0)
        return d2.lerp(d1, h) - k * h * (1.0 - h)

    @staticmethod
    def _sub(d1: torch.Tensor, d2: torch.Tensor) -> torch.Tensor:
        # CSG 差：d1 \ d2
        return torch.maximum(-d1, d2)

    def _sd_ellipsoid(self, p: torch.Tensor, rad: torch.Tensor) -> torch.Tensor:
        k0 = torch.linalg.norm(p / rad, dim=1, keepdim=True)
        k1 = torch.linalg.norm(p / (rad * rad), dim=1, keepdim=True)
        return (k0 * (k0 - 1.0)) / k1

    def _sd_capsule_y(self, p: torch.Tensor, h: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        # 竖直 (+y) 胶囊，允许 r 为逐点半径（广播）
        py = p[:, 1:2]
        q = p.clone()
        q[:, 1:2] = py - torch.clamp(py, 0.0, h)
        return torch.linalg.norm(q, dim=1, keepdim=True) - r

    def _sd_capped_cylinder_z(self, p: torch.Tensor, h: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        # Z 轴盖帽圆柱：高度 2h，半径 r
        dxy = torch.linalg.norm(p[:, :2], dim=1, keepdim=True)
        dz = p[:, 2:3]
        d = torch.abs(torch.cat([dxy, dz], dim=1)) - torch.cat([r.expand_as(dxy), h.expand_as(dz)], dim=1)
        mxy = torch.maximum(d[:, :1], d[:, 1:2])
        return torch.minimum(torch.maximum(mxy, torch.tensor(0.0, dtype=d.dtype, device=d.device)),
                             torch.tensor(0.0, dtype=d.dtype, device=d.device)) + \
            torch.linalg.norm(torch.clamp(d, min=0.0), dim=1, keepdim=True)

    # ----------------------------- SDF ------------------------------- #
    def sdf(self, p: torch.Tensor):
        p = p.to(dtype=self.dtype, device=self.device)
        q = self._world_to_local(p)

        # 主体：大椭球 - 小椭球
        hb = self._sd_ellipsoid(q + self.B_C, self.B_R)
        ht = self._sd_ellipsoid(q + self.T_C, self.T_R)
        base = self._sub(hb, ht)

        # 顶端胶囊（线性锥度）
        tipR = self.TIP_R0 + self.TIP_RT * (q[:, 1:2] / 2.0)
        top = self._sd_capsule_y(q + self.TIP_C, self.TIP_H, tipR)

        # 平滑并联
        hook = self._smin_poly(base, top, self.SMOOTH_K)

        # 减去眼孔
        hole = self._sd_capped_cylinder_z(q - self.HOLE_C, self.HOLE_HH, self.HOLE_R)
        hook = torch.maximum(hook, -hole)

        # 缩放回距离
        return hook * self.scale

    # --------------------------- bounding box ------------------------ #
    def _bbox_from_local_aabb(self, bmin: torch.Tensor, bmax: torch.Tensor):
        # (local → world uses R = R_x @ R_z, matching _world_to_local order)
        rx, rz = self.rot_xz[0, 0], self.rot_xz[0, 1]
        cx, sx = torch.cos(rx), torch.sin(rx)
        cz, sz = torch.cos(rz), torch.sin(rz)
        R_x = torch.tensor([[1.0, 0.0, 0.0],
                            [0.0, cx, -sx],
                            [0.0, sx, cx]], dtype=self.dtype, device=self.device)
        R_z = torch.tensor([[cz, -sz, 0.0],
                            [sz, cz, 0.0],
                            [0.0, 0.0, 1.0]], dtype=self.dtype, device=self.device)
        R = R_x @ R_z
        corners = torch.tensor([[bmin[0], bmin[1], bmin[2]],
                                [bmin[0], bmin[1], bmax[2]],
                                [bmin[0], bmax[1], bmin[2]],
                                [bmin[0], bmax[1], bmax[2]],
                                [bmax[0], bmin[1], bmin[2]],
                                [bmax[0], bmin[1], bmax[2]],
                                [bmax[0], bmax[1], bmin[2]],
                                [bmax[0], bmax[1], bmax[2]]],
                               dtype=self.dtype, device=self.device)
        world = (corners @ R.T) * self.scale + self.origin
        wmin = world.min(dim=0).values.squeeze(0)
        wmax = world.max(dim=0).values.squeeze(0)
        return wmin, wmax

    def _compute_bbox(self):
        """Tighter conservative bbox in *world* coordinates.
        Notes on centers (because SDF uses p + C or p - C):
          • Ellipsoids are called as elipsoid(p + C, R)  → center at (−C)
          • Capsule is called as capsuleY(p + TIP_C, ...) → segment base at (−TIP_C)
          • Hole is called as sdCappedCylinderZ(p − HOLE_C, ...) → center at (+HOLE_C)
        Smooth union doesn't grow support beyond the union; we add a small padding
        proportional to SMOOTH_K for safety.
        """
        # -------- body: big ellipsoid dominates base bbox -------- #
        c_big = (-self.B_C).squeeze(0)  # center at −B_C
        r_big = self.B_R.squeeze(0)
        bmin1 = c_big - r_big * 0.1
        bmax1 = c_big + r_big * 0.1

        # -------- tip: vertical capsule with linear radius r(y) -------- #
        c0 = (-self.TIP_C).squeeze(0)  # base point (y = c0.y)
        c1 = c0 + torch.tensor([0.0, self.TIP_H.item(), 0.0], dtype=self.dtype, device=self.device)
        # radius varies with global/local y as r(y) = r0 + rt*(y/2)
        r0 = (self.TIP_R0 + self.TIP_RT * (c0[1] / 2.0)).item()
        r1 = (self.TIP_R0 + self.TIP_RT * (c1[1] / 2.0)).item()
        rmax = float(max(r0, r1))
        # y-extent includes spherical caps
        ymin2 = float(min(c0[1] - r0, c1[1] - r1))
        ymax2 = float(max(c0[1] + self.TIP_H.item() + 0.0, c1[1] + r1))  # c1 already at top; include top cap radius
        # Because the segment is vertical along +y, x/z extents are centered at c0.x/z
        xmin2 = float(c0[0] - rmax)
        xmax2 = float(c0[0] + rmax)
        zmin2 = float(c0[2] - rmax)
        zmax2 = float(c0[2] + rmax)
        bmin2 = torch.tensor([xmin2, ymin2, zmin2], dtype=self.dtype, device=self.device)
        bmax2 = torch.tensor([xmax2, ymax2, zmax2], dtype=self.dtype, device=self.device)

        # -------- hole: capped cylinder along Z (center at +HOLE_C) -------- #
        c_h = self.HOLE_C.squeeze(0)
        r_h = self.HOLE_R.item()
        h_h = self.HOLE_HH.item()
        bmin3 = torch.tensor([c_h[0] - r_h, c_h[1] - r_h, c_h[2] - h_h], dtype=self.dtype, device=self.device)
        bmax3 = torch.tensor([c_h[0] + r_h, c_h[1] + r_h, c_h[2] + h_h], dtype=self.dtype, device=self.device)

        # transform each local AABB to world AABB and merge
        wmin1, wmax1 = self._bbox_from_local_aabb(bmin1, bmax1)
        wmin2, wmax2 = self._bbox_from_local_aabb(bmin2, bmax2)
        wmin3, wmax3 = self._bbox_from_local_aabb(bmin3, bmax3)
        wmin = torch.minimum(torch.minimum(wmin1, wmin2), wmin3)
        wmax = torch.maximum(torch.maximum(wmax1, wmax2), wmax3)

        # Smooth-min band padding (conservative): add a small band ≈ SMOOTH_K in world units
        pad = float(self.SMOOTH_K.item() * self.scale.item())
        if pad > 0:
            wmin = wmin - pad
            wmax = wmax + pad

        return wmin, wmax

    def get_bounding_box(self) -> List[float]:
        return [self._bbox_min[0].item(), self._bbox_max[0].item(),
                self._bbox_min[1].item(), self._bbox_max[1].item(),
                self._bbox_min[2].item(), self._bbox_max[2].item()]

    # ----------------------- sampling utilities ---------------------- #
    def in_sample(self, num_samples: int, with_boundary: bool = False) -> torch.Tensor:
        # 粗网格 + 内部筛选（够用且与 Boolean 兼容）
        nx = ny = nz = int(max(3, round(num_samples ** (1 / 3))))
        xs = torch.linspace(self._bbox_min[0], self._bbox_max[0], nx, generator=self.gen, dtype=self.dtype,
                            device=self.device)
        ys = torch.linspace(self._bbox_min[1], self._bbox_max[1], ny, generator=self.gen, dtype=self.dtype,
                            device=self.device)
        zs = torch.linspace(self._bbox_min[2], self._bbox_max[2], nz, generator=self.gen, dtype=self.dtype,
                            device=self.device)
        X, Y, Z = torch.meshgrid(xs, ys, zs, indexing='ij')
        pts = torch.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], dim=1)
        d = self.sdf(pts).squeeze()
        return pts[(d <= 0) if with_boundary else (d < 0)]

    def on_sample(self, num_samples: int, with_normal: bool = False):
        # 近边带采样 + 有限差分法向（与其它几何的一致用法）
        nx = ny = nz = int(max(6, round(num_samples ** (1 / 3))))
        xs = torch.linspace(self._bbox_min[0], self._bbox_max[0], nx, generator=self.gen, dtype=self.dtype,
                            device=self.device)
        ys = torch.linspace(self._bbox_min[1], self._bbox_max[1], ny, generator=self.gen, dtype=self.dtype,
                            device=self.device)
        zs = torch.linspace(self._bbox_min[2], self._bbox_max[2], nz, generator=self.gen, dtype=self.dtype,
                            device=self.device)
        X, Y, Z = torch.meshgrid(xs, ys, zs, indexing='ij')
        pts = torch.stack([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)], dim=1)
        d = self.sdf(pts).squeeze()
        eps = 0.5 * max((self._bbox_max - self._bbox_min).tolist()) / max(nx, ny, nz)
        mask = d.abs() <= eps
        pts = pts[mask]
        if not with_normal:
            return pts
        # 中心差分估计法向
        h = eps
        ex = torch.tensor([[h, 0.0, 0.0]], dtype=pts.dtype, device=pts.device)
        ey = torch.tensor([[0.0, h, 0.0]], dtype=pts.dtype, device=pts.device)
        ez = torch.tensor([[0.0, 0.0, h]], dtype=pts.dtype, device=pts.device)
        nx_ = (self.sdf(pts + ex) - self.sdf(pts - ex)).squeeze() / (2 * h)
        ny_ = (self.sdf(pts + ey) - self.sdf(pts - ey)).squeeze() / (2 * h)
        nz_ = (self.sdf(pts + ez) - self.sdf(pts - ez)).squeeze() / (2 * h)
        n = torch.stack([nx_, ny_, nz_], dim=1)
        n = n / (torch.norm(n, dim=1, keepdim=True) + 1e-12)
        return pts, n

    # ----------------------------- GLSL ------------------------------ #
    def glsl_sdf(self) -> str:
        # 将当前参数烘焙成一段可内联的 GLSL 表达式；假定 shader 里已有 vec3 p
        def f3(v): return ", ".join(f"{float(x):.9g}" for x in v.squeeze().tolist())

        def f1(x): return f"{float(x):.9g}"

        ORIGIN = f"vec3({f3(self.origin)})"
        ROT_XZ = f"vec2({f3(self.rot_xz)})"
        SCALE = f1(self.scale)
        B_C = f"vec3({f3(self.B_C)})";
        B_R = f"vec3({f3(self.B_R)})"
        T_C = f"vec3({f3(self.T_C)})";
        T_R = f"vec3({f3(self.T_R)})"
        SMOOTH_K = f1(self.SMOOTH_K)
        TIP_C = f"vec3({f3(self.TIP_C)})";
        TIP_H = f1(self.TIP_H);
        TIP_R0 = f1(self.TIP_R0);
        TIP_RT = f1(self.TIP_RT)
        HOLE_C = f"vec3({f3(self.HOLE_C)})";
        HOLE_HH = f1(self.HOLE_HH);
        HOLE_R = f1(self.HOLE_R)

        return (
            "({\n"
            "  float smin(float d1, float d2, float k){ float h = clamp(0.5 + 0.5*(d2-d1)/k, 0.0, 1.0); return mix(d2,d1,h) - k*h*(1.0-h); }\n"
            "  float subf(float d1, float d2){ return max(-d1, d2); }\n"
            "  float elipsoid(vec3 p, vec3 rad){ float k0 = length(p / rad); float k1 = length(p / (rad*rad)); return (k0*(k0-1.0))/k1; }\n"
            "  float capsuleY(vec3 p, float h, float r){ p.y -= clamp(p.y, 0.0, h); return length(p) - r; }\n"
            "  float sdCappedCylinderZ(vec3 p, float h, float r){ vec2 d = abs(vec2(length(p.xy), p.z)) - vec2(r, h); return min(max(d.x, d.y), 0.0) + length(max(d, 0.0)); }\n"
            "  mat2 rot(float a){ float c=cos(a), s=sin(a); return mat2(c,-s,s,c); }\n"
            "  vec3 applyTRS(vec3 p, vec3 origin, vec2 rotXZ, float scale){ p += origin; p.yz = rot(rotXZ.x)*p.yz; p.xy = rot(rotXZ.y)*p.xy; return p/max(scale,1e-6); }\n"
            f"  vec3 ORIGIN = {ORIGIN}; vec2 ROT_XZ = {ROT_XZ}; float SCALE = {SCALE};\n"
            f"  vec3 B_C = {B_C}; vec3 B_R = {B_R}; vec3 T_C = {T_C}; vec3 T_R = {T_R};\n"
            f"  float SMOOTH_K = {SMOOTH_K};\n"
            f"  vec3 TIP_C = {TIP_C}; float TIP_H = {TIP_H}; float TIP_R0 = {TIP_R0}; float TIP_RT = {TIP_RT};\n"
            f"  vec3 HOLE_C = {HOLE_C}; float HOLE_HH = {HOLE_HH}; float HOLE_R = {HOLE_R};\n"
            "  vec3 p = applyTRS(p, ORIGIN, ROT_XZ, SCALE);\n"
            "  float hb = elipsoid(p + B_C, B_R);\n"
            "  float ht = elipsoid(p + T_C, T_R);\n"
            "  float base = subf(hb, ht);\n"
            "  float tipR = TIP_R0 + TIP_RT * (p.y/2.0);\n"
            "  float top = capsuleY(p + TIP_C, TIP_H, tipR);\n"
            "  float hook = smin(base, top, SMOOTH_K);\n"
            "  float hole = sdCappedCylinderZ(p - HOLE_C, HOLE_HH, HOLE_R);\n"
            "  hook = max(hook, -hole);\n"
            "  hook * SCALE;\n"
            "})"
        )


if __name__ == "__main__":
    hook = Hook3D()
    print(hook.get_bounding_box())
    model = pyrfm.RFMBase(dim=3, domain=hook, n_subdomains=1, n_hidden=100)

    model.W = torch.randn((100, 1))

    viz = pyrfm.RFMVisualizer3D(model=model, view='top')

    viz.plot()
    viz.show()
