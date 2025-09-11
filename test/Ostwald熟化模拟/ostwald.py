# -*- coding: utf-8 -*-
"""
Created on 2025/8/22

@author: Yifei Sun
"""
import torch
import pyrfm
import math
import random
import os, glob, shutil

try:
    import cv2
except Exception:
    cv2 = None

from typing import Tuple, List, Union


def constants():
    edif = 0.3
    kB = 1.380649e-23
    D0 = 1e-6 * 1e18
    ediff = edif * 1.602e-19

    temp = 500
    vm = 6.5e-20 * 1e18
    gamma = 1.0e-10 * 1e-9
    Rg = 8.314
    C_inf = 10
    D = D0 * math.exp(-ediff / (kB * temp))

    return vm, gamma, kB, temp, C_inf, D


def init_particles(
        n_particles: int,
        bounding_box: Union[Tuple[float, float, float, float], List[float]],
        mean_radius: float,
        std_radius: float,
        min_dist: float,
        *,
        seed: int = None,
        max_global_tries: int = 20000
) -> Tuple[List[Tuple[float, float]], List[float]]:
    """
    生成 n 个圆形粒子（中心与半径）：
    1) r ~ N(mean_radius, std_radius)，并截断在 [mean - std, mean + std]
    2) 任意两粒子中心距 >= min_dist
    3) 粒子中心到边界距离 >= 30 * r     （即 `10 * 3 * r`）

    返回:
        centers: [(x, y), ...]
        radii:   [r1, r2, ...]
    """
    if seed is not None:
        random.seed(seed)

    x_min, x_max, y_min, y_max = bounding_box
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("Invalid bounding_box.")

    r_lo = max(0.0, mean_radius - std_radius)
    r_hi = mean_radius + std_radius
    if r_hi <= 0:
        raise ValueError("mean_radius + std_radius must be > 0.")

    # 检查最大半径是否几何可行（需要预留 30r 的边距）
    if (x_max - x_min) <= 60 * r_hi or (y_max - y_min) <= 60 * r_hi:
        raise ValueError("Box too small for the required 30*r margin with the largest allowed radius.")

    centers: List[Tuple[float, float]] = []
    radii: List[float] = []

    def sample_radius() -> float:
        # 截断正态：直到落在 [r_lo, r_hi] 且 > 0
        for _ in range(10000):
            r = random.gauss(mean_radius, std_radius)
            if r_lo <= r <= r_hi and r > 0:
                return r
        # 兜底（极端情况下）
        return max(r_lo, 1e-9)

    def valid_position(x: float, y: float, r: float) -> bool:
        # 边界条件：到边距离 >= 30*r
        if not (x_min + 30 * r <= x <= x_max - 30 * r and y_min + 30 * r <= y <= y_max - 30 * r):
            return False
        # 中心距条件：>= min_dist  （不再与半径相关）
        for (cx, cy) in centers:
            dx = x - cx
            dy = y - cy
            if dx * dx + dy * dy < (min_dist ** 2):
                return False
        return True

    tries = 0
    while len(centers) < n_particles and tries < max_global_tries:
        tries += 1
        r = sample_radius()
        # 在可放置范围内均匀采样中心
        x = random.uniform(x_min + 30 * r, x_max - 30 * r)
        y = random.uniform(y_min + 30 * r, y_max - 30 * r)
        if valid_position(x, y, r):
            centers.append((x, y))
            radii.append(r)

    if len(centers) < n_particles:
        raise RuntimeError(
            f"Failed to place all particles: placed {len(centers)}/{n_particles}. "
            "Try reducing n_particles, mean/std radius, or min_dist, or enlarge bounding_box."
        )

    return centers, radii


def init_particles_growth_style(
        n_particles: int,
        bounding_box: Union[Tuple[float, float, float, float], List[float]],
        mean_radius: float,
        std_radius: float,
        min_dist: float,
        *,
        seed: int = None,
        max_global_tries: int = 20000
) -> Tuple[List[Tuple[float, float]], List[float]]:
    """
    生成 n 个圆形粒子（中心与半径），放置策略按“后者”生长式：
    - 第1个粒子：在缩了 30*r 边界的矩形内均匀采样；
    - 之后的每个粒子 i：随机挑一个已有粒子 j，在距离 j 的圆周（半径 = min_dist）上按随机方向取候选，
      若候选满足：
        1) 与边界距离 >= 30*r_i
        2) 与除了 j 以外的所有粒子中心距均 >= min_dist
      则接受，否则继续尝试。
    其它设定：
    - r ~ N(mean_radius, std_radius)，截断在 [mean-std, mean+std] 且 r>0。
    - 全局几何可行性检查：箱体需能容下最大半径 r_hi 的 30r 边距。
    - 全局最多尝试次数 max_global_tries，超过即报错。
    """
    if seed is not None:
        random.seed(seed)

    x_min, x_max, y_min, y_max = bounding_box
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("Invalid bounding_box.")

    r_lo = max(0.0, mean_radius - std_radius)
    r_hi = mean_radius + std_radius
    if r_hi <= 0:
        raise ValueError("mean_radius + std_radius must be > 0.")

    # 全局几何可行性（需预留 30*r 的边距）
    if (x_max - x_min) <= 60 * r_hi or (y_max - y_min) <= 60 * r_hi:
        raise ValueError("Box too small for the required 30*r margin with the largest allowed radius.")

    centers: List[Tuple[float, float]] = []
    radii: List[float] = []

    def sample_radius() -> float:
        # 截断正态：直到落在 [r_lo, r_hi] 且 > 0
        for _ in range(10000):
            r = random.gauss(mean_radius, std_radius)
            if r_lo <= r <= r_hi and r > 0:
                return r
        # 兜底（极端情况下）
        return max(r_lo, 1e-9)

    def inside_bounds_with_margin(x: float, y: float, r: float) -> bool:
        return (x_min + 30 * r <= x <= x_max - 30 * r) and (y_min + 30 * r <= y <= y_max - 30 * r)

    def respects_mindist_except_anchor(x: float, y: float, anchor_idx: int) -> bool:
        # 与除 anchor 以外的所有点的距离 >= min_dist
        for k, (cx, cy) in enumerate(centers):
            if k == anchor_idx:
                continue
            dx = x - cx
            dy = y - cy
            # 允许极小数值误差
            if dx * dx + dy * dy < (min_dist ** 2) - 1e-12:
                return False
        return True

    tries = 0
    while len(centers) < n_particles and tries < max_global_tries:
        r = sample_radius()
        if not centers:
            # 第一个粒子：在缩了 30*r 的矩形内均匀采样
            tries += 1
            x = random.uniform(x_min + 30 * r, x_max - 30 * r)
            y = random.uniform(y_min + 30 * r, y_max - 30 * r)
            centers.append((x, y))
            radii.append(r)
            continue

        # 后续粒子：生长式（圆周上取候选）
        placed = False
        # 为了避免在某个粒子上无限打转，这里也占用全局 tries
        while tries < max_global_tries and not placed:
            tries += 1
            j = random.randrange(len(centers))  # 随机选锚点
            cjx, cjy = centers[j]
            theta = random.uniform(0.0, 2.0 * math.pi)
            x = cjx + min_dist * math.cos(theta)
            y = cjy + min_dist * math.sin(theta)

            if not inside_bounds_with_margin(x, y, r):
                continue
            if not respects_mindist_except_anchor(x, y, j):
                continue

            centers.append((x, y))
            radii.append(r)
            placed = True

        if not placed:
            break  # 触发全局失败

    if len(centers) < n_particles:
        raise RuntimeError(
            f"Failed to place all particles (growth-style): placed {len(centers)}/{n_particles} "
            f"after {tries} attempts. Try reducing n_particles, mean/std radius, or min_dist, "
            "or enlarge bounding_box."
        )

    return centers, radii


class CustomVisualizer(pyrfm.RFMVisualizer2D):
    def compute_field_vals(self, grid_points):
        if isinstance(self.model, pyrfm.RFMBase):
            if self.ref is not None:
                Z = (self.model(grid_points) - self.ref(grid_points)).abs().detach().cpu().numpy()
            else:
                Z = self.model(grid_points).detach().cpu().numpy()
        elif isinstance(self.model, pyrfm.STRFMBase):
            xt = self.model.validate_and_prepare_xt(x=grid_points, t=torch.tensor([[self.t]]))
            if self.ref is not None:
                Z = (self.model.forward(xt=xt) - self.ref(xt=xt)).abs().detach().cpu().numpy()
            else:
                Z = self.model.forward(xt=xt).detach().cpu().numpy()

        else:
            raise NotImplementedError

        # let Z be non-negative
        Z[Z < C_inf] = C_inf

        return Z


class RFReLU(pyrfm.RFBase):
    def __init__(self, dim: int, center: torch.Tensor, radius: torch.Tensor,
                 n_hidden: int,
                 gen: torch.Generator = None,
                 dtype: torch.dtype = None,
                 device: torch.device = None):
        super().__init__(dim, center, radius, torch.nn.ReLU(), n_hidden, gen, dtype, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.dim:
            raise ValueError('Input dimension mismatch')
        with torch.no_grad():
            # Be careful when x in a slice
            if (self.x_buff_ is not None) and (self.x_buff_ is x or torch.equal(self.x_buff_, x)):
                return self.features_buff_
            self.x_buff_ = x
            m = self.activation
            self.features_buff_ = m(
                torch.matmul((x - self.center) / self.radius, self.weights) + self.biases)
            return self.features_buff_

    def first_derivative(self, x: torch.Tensor, axis: int) -> torch.Tensor:
        if x.shape[1] != self.dim:
            raise ValueError('Input dimension mismatch')

        if axis >= self.dim:
            raise ValueError('Axis out of range')

        with torch.no_grad():
            # Be careful when x in a slice
            if (self.x_buff_ is not None) and (self.x_buff_ is x or torch.equal(self.x_buff_, x)):
                pass
            else:
                self.forward(x)

            return torch.where(self.features_buff_ >= 0, 1.0, 0.0) * (self.weights[[axis], :] / self.radius[0, axis])

    def second_derivative(self, x: torch.Tensor, axis1: int, axis2: int) -> torch.Tensor:
        if x.shape[1] != self.dim:
            raise ValueError('Input dimension mismatch')

        if axis1 >= self.dim:
            raise ValueError('Axis1 out of range')

        if axis2 >= self.dim:
            raise ValueError('Axis2 out of range')

        with torch.no_grad():
            # Be careful when x in a slice
            if (self.x_buff_ is not None) and (self.x_buff_ is x or torch.equal(self.x_buff_, x)):
                pass
            else:
                self.forward(x)

            return torch.zeros_like(self.features_buff_)

    def higher_order_derivative(self, x: torch.Tensor, order: Union[torch.Tensor, List]) -> torch.Tensor:
        pass


if __name__ == "__main__":
    torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")

    # === 帧目录准备 ===
    frames_dir = "frames"
    try:
        shutil.rmtree(frames_dir)
        print(f"已清理帧目录：{frames_dir}")
    except Exception as e:
        print(f"清理帧目录时出错（可忽略）：{e}")
    os.makedirs(frames_dir, exist_ok=True)
    frame_paths = []
    fps = 10
    video_path = "simulation.mp4"

    V_m, gamma, kB, T, C_inf, D = constants()
    print("V_m = {:e}, gamma = {:e}, kB = {:e}, T = {:e}, C_inf = {:e}, D = {:e}".format(V_m, gamma, kB, T, C_inf, D))

    base = pyrfm.Square2D(center=(0, 0), radius=(710.0, 710.0))
    centers, radii = init_particles_growth_style(
        n_particles=10,
        bounding_box=base.get_bounding_box(),
        mean_radius=15.0,
        std_radius=1.5,
        min_dist=180,
        seed=42
    )
    print(f"Placed {len(centers)} particles.")

    dt = 1e-6
    domain = None
    model = None
    circles = []
    big_flag = False
    for i in range(2000):
        if i == 0:
            for center, radius in zip(centers, radii):
                c = pyrfm.Circle2D(center=center, radius=radius)
                circles.append(c)
            domain = base - sum(circles, pyrfm.EmptyGeometry())
            model = pyrfm.RFMBase(dim=2, domain=domain, n_hidden=200, n_subdomains=2, pou=pyrfm.PsiB)
            x_in = domain.in_sample(2000)
            # x_on, x_on_normal = base.on_sample(4000, with_normal=True)
            # A_on = model.features(x_on).cat(dim=1)
            # b_on = C_inf * torch.ones((x_on.shape[0], 1))

            (a, an), (b, bn), (c, cn), (d, dn) = base.on_sample(4000, with_normal=True, separate=True)
            c = c.flip(0)
            cn = cn.flip(0)
            d = d.flip(0)
            dn = dn.flip(0)
            A_a = model.features(a).cat(dim=1)
            A_c = model.features(c).cat(dim=1)
            A_b = model.features(b).cat(dim=1)
            A_d = model.features(d).cat(dim=1)
            A_a_y = model.features_derivative(a, axis=1).cat(dim=1)
            A_c_y = model.features_derivative(c, axis=1).cat(dim=1)
            A_b_x = model.features_derivative(b, axis=0).cat(dim=1)
            A_d_x = model.features_derivative(d, axis=0).cat(dim=1)
            A_on = torch.cat([A_a - A_c, A_b - A_d, A_a_y - A_c_y, A_b_x - A_d_x], dim=0)
            b_on = torch.zeros(A_on.shape[0], 1)

            r_circles = [c.radius for c in circles]
            x_circles = [c.on_sample(400) for c in circles]
            f_circles = [torch.ones((x_c.shape[0], 1)) * C_inf * math.exp(V_m * gamma / (kB * T * max(radius, 0.1)))
                         for (x_c, radius) in zip(x_circles, r_circles)]

            A = torch.cat([model.features(torch.cat([x_in, torch.cat(x_circles, dim=0)])).cat(dim=1), A_on], dim=0)
            b = torch.cat(
                [C_inf * torch.ones((x_in.shape[0], 1)),
                 torch.cat(f_circles, dim=0), b_on],
                dim=0)

            A_regularity = torch.eye(A.shape[1]) * 1e-8
            b_regularity = torch.zeros(A.shape[1], 1)
            A = torch.cat([A, A_regularity], dim=0)
            b = torch.cat([b, b_regularity], dim=0)

            model.W = torch.linalg.lstsq(A, b)[0]
            residual = torch.norm(torch.matmul(A, model.W) - b) / torch.norm(b)
            print(f"Step {i:05d}, Least Square Relative residual: {residual:.4e}")

            visualizer = CustomVisualizer(model, resolution=(800, 800))
            visualizer.plot()
            frame_path = os.path.join(frames_dir, f"frame_{i:05d}.png")
            visualizer.savefig(frame_path, dpi=300)
            frame_paths.append(frame_path)
            try:
                visualizer.close()
            except Exception:
                del visualizer

        else:
            circles_new = []
            for c in circles:
                x_on, x_on_normal = c.on_sample(400, with_normal=True)
                mean_val = torch.mean(
                    model.dForward(x_on, order=(1, 0)) * x_on_normal[:, [0]] +
                    model.dForward(x_on, order=(0, 1)) * x_on_normal[:, [1]]
                )
                integral = mean_val.item() * 2 * math.pi * c.radius
                dr = dt * V_m * D / (2 * math.pi * c.radius) * integral
                if c.radius + dr < 1.5:
                    continue
                if c.radius + dr > 150.0:
                    big_flag = True
                r_new = c.radius + dr
                circles_new.append(pyrfm.Circle2D(center=c.center, radius=r_new))
            if big_flag or len(circles_new) == 0:
                break
            circles = circles_new
            domain = base - sum(circles, pyrfm.EmptyGeometry())
            model_old = model
            model = pyrfm.RFMBase(dim=2, domain=domain, n_hidden=200, n_subdomains=2, pou=pyrfm.PsiB)
            x_in = domain.in_sample(2000)
            # x_on, x_on_normal = base.on_sample(4000, with_normal=True)
            # A_on = model.features(x_on).cat(dim=1)
            # b_on = C_inf * torch.ones((x_on.shape[0], 1))
            (a, an), (b, bn), (c, cn), (d, dn) = base.on_sample(4000, with_normal=True, separate=True)
            c = c.flip(0)
            cn = cn.flip(0)
            d = d.flip(0)
            dn = dn.flip(0)
            A_a = model.features(a).cat(dim=1)
            A_c = model.features(c).cat(dim=1)
            A_b = model.features(b).cat(dim=1)
            A_d = model.features(d).cat(dim=1)
            A_a_y = model.features_derivative(a, axis=1).cat(dim=1)
            A_c_y = model.features_derivative(c, axis=1).cat(dim=1)
            A_b_x = model.features_derivative(b, axis=0).cat(dim=1)
            A_d_x = model.features_derivative(d, axis=0).cat(dim=1)
            A_on = torch.cat([A_a - A_c, A_b - A_d, A_a_y - A_c_y, A_b_x - A_d_x], dim=0)
            b_on = torch.zeros(A_on.shape[0], 1)

            r_circles = [c.radius for c in circles]
            x_circles = [c.on_sample(400) for c in circles]
            x_circles = [c[(domain.sdf(c) <= 0.01).flatten()] for c in x_circles]
            f_circles = [torch.ones((x_c.shape[0], 1)) * C_inf * math.exp(V_m * gamma / (kB * T * radius))
                         for (x_c, radius) in zip(x_circles, r_circles)]
            A_circles = torch.cat([model.features(x_c).cat(dim=1) for x_c in x_circles], dim=0)
            b_circles = torch.cat(f_circles, dim=0)

            A_in = model.features(x_in).cat(dim=1)
            A_in_lap = (model.features_second_derivative(x_in, axis1=0, axis2=0)
                        + model.features_second_derivative(x_in, axis1=1, axis2=1)).cat(dim=1)
            A_pde = A_in - dt * D * A_in_lap
            b_pde = model_old.forward(x_in)
            b_pde[b_pde < C_inf] = C_inf  # 保障非负

            A = torch.cat([A_pde, A_on, A_circles], dim=0)
            b = torch.cat([b_pde, b_on, b_circles], dim=0)

            A_regularity = torch.eye(A.shape[1]) * 1e-8
            b_regularity = torch.zeros(A.shape[1], 1)
            A = torch.cat([A, A_regularity], dim=0)
            b = torch.cat([b, b_regularity], dim=0)

            model.W = torch.linalg.lstsq(A, b)[0]
            residual = torch.norm(torch.matmul(A, model.W) - b) / torch.norm(b)
            print(f"Step {i:05d}, Least Square Relative residual: {residual:.4e}")

            visualizer = CustomVisualizer(model, resolution=(800, 800))
            visualizer.plot()
            frame_path = os.path.join(frames_dir, f"frame_{i:05d}.png")
            visualizer.savefig(frame_path, dpi=200)
            frame_paths.append(frame_path)
            try:
                visualizer.close()
            except Exception:
                del visualizer

    # === 仅用 OpenCV 合成视频并清理帧目录 ===
    if cv2 is None:
        raise RuntimeError("未安装 OpenCV（cv2）。请先 pip install opencv-python 再运行。")

    frame_paths = sorted(frame_paths) if frame_paths else sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
    if not frame_paths:
        raise RuntimeError("没有生成任何帧图，检查上面的保存逻辑。")

    # 读取首帧确定尺寸，并统一到 BGR 三通道
    first = cv2.imread(frame_paths[0], cv2.IMREAD_UNCHANGED)
    if first is None:
        raise RuntimeError(f"读取首帧失败：{frame_paths[0]}")
    if first.ndim == 2:
        first = cv2.cvtColor(first, cv2.COLOR_GRAY2BGR)
    elif first.shape[2] == 4:
        first = cv2.cvtColor(first, cv2.COLOR_BGRA2BGR)
    H, W = first.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 若需要 H.264 可尝试 "avc1"
    vw = cv2.VideoWriter(video_path, fourcc, fps, (W, H))
    if not vw.isOpened():
        raise RuntimeError("OpenCV VideoWriter 打不开，可能缺少编码器或路径无写权限。")


    def _read_norm(path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"读取帧失败：{path}")
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        if (img.shape[1], img.shape[0]) != (W, H):
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        return img


    bad = 0
    for fp in frame_paths:
        try:
            vw.write(_read_norm(fp))
        except Exception as e:
            bad += 1
            print(f"警告：跳过坏帧 {fp}，原因：{e}")

    vw.release()
    if not os.path.isfile(video_path) or os.path.getsize(video_path) == 0:
        raise RuntimeError("视频文件为空或未生成，可能编码器不可用。")

    print(f"(OpenCV) 视频已保存：{video_path}，有效帧数={len(frame_paths) - bad}, 跳过={bad}, fps={fps}")

    # 成功后清理帧目录
    try:
        shutil.rmtree(frames_dir)
        print(f"已清理帧目录：{frames_dir}")
    except Exception as e:
        print(f"清理帧目录时出错：{e}")
