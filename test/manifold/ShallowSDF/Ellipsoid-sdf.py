from typing import List, Union, Tuple

import pyrfm
import torch
import time
import numpy as np
from scipy.spatial import cKDTree

from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


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


class Ellipsoid(pyrfm.ImplicitSurfaceBase):

    def __init__(self):
        super().__init__()
        self.a, self.b, self.c = 1.5, 1.0, 0.5

    def get_bounding_box(self) -> List[float]:
        ratio = 1.1
        return [-self.a * ratio, self.a * ratio, -self.b * ratio, self.b * ratio, -self.c * ratio, self.c * ratio]

    def shape_func(self, p: torch.Tensor) -> torch.Tensor:
        # Ellipsoid equation: (x/a)^2 + (y/b)^2 + (z/c)^2 = 1

        return ((p[:, 0] / self.a) ** 2 + (p[:, 1] / self.b) ** 2 + (p[:, 2] / self.c) ** 2 - 1).unsqueeze(-1)


class CheeseLike(pyrfm.ImplicitSurfaceBase):
    def __init__(self):
        super().__init__()

    def get_bounding_box(self) -> List[float]:
        return [-1.25, 1.25, -1.25, 1.25, -1.25, 1.25]

    def shape_func(self, p: torch.Tensor) -> torch.Tensor:
        """
        ψ(x, y, z) = (4x² − 1)² + (4y² − 1)² + (4z² − 1)²
                               + 16(x² + y² − 1)² + 16(x² + z² − 1)² + 16(y² + z² − 1)² − 16
        """
        return ((4 * p[:, 0] ** 2 - 1) ** 2 + (4 * p[:, 1] ** 2 - 1) ** 2 + (4 * p[:, 2] ** 2 - 1) ** 2
                + 16 * (p[:, 0] ** 2 + p[:, 1] ** 2 - 1) ** 2 + 16 * (p[:, 0] ** 2 + p[:, 2] ** 2 - 1) ** 2
                + 16 * (p[:, 1] ** 2 + p[:, 2] ** 2 - 1) ** 2 - 16).unsqueeze(-1)


def sdf_from_shape_and_dists(domain, pts: torch.Tensor, dists: torch.Tensor, zero_eps: float = 0.0):
    """
    根据 domain.shape_func(pts) 的符号，为无符号距离 dists 加符号，返回 SDF。
    - domain.shape_func: 需返回形状 (M, 1)，外部为正，内部为负（如你的 Ellipsoid）。
    - pts:  查询点 (M, D)（这里是 x_bnd）
    - dists: 最近距离，形状 (M,) 或 (M, k)
    - zero_eps: 将 |phi|<=zero_eps 的点视作 0（可选，默认不改）
    """
    phi = domain.shape_func(pts).squeeze(-1)  # (M,)
    if zero_eps > 0:
        # 让靠近零层的点直接置 0（数值稳一点，可选）
        phi = torch.where(phi.abs() <= zero_eps, torch.zeros_like(phi), phi)

    sign = torch.sign(phi)  # -1, 0, +1
    # 按约定：外正内负；若你想内正外负，改成 sign = -torch.sign(phi)

    # 兼容 dists 形状 (M, k)
    while sign.ndim < dists.ndim:
        sign = sign.unsqueeze(-1)  # 广播到 (M, k)
    sdf = dists * sign.to(dists.dtype)

    return sdf


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


class TrainableRFBase(pyrfm.RFTanH, torch.nn.Module):
    def __init__(self, *args, **kwargs):
        torch.nn.Module.__init__(self)
        pyrfm.RFTanH.__init__(self, *args, **kwargs)
        self.weights = torch.nn.Parameter(self.weights)
        self.biases = torch.nn.Parameter(self.biases)


if __name__ == '__main__':
    torch.set_default_device('cuda') if torch.cuda.is_available() else torch.set_default_device('cpu')
    # domain = Ellipsoid()
    domain = CheeseLike()
    model = pyrfm.RFMBase(dim=3, n_hidden=512, rf=TrainableRFBase, domain=domain, n_subdomains=1)
    model.W = torch.nn.Parameter(torch.zeros(model.submodels.numel() * model.n_hidden, 1))

    all_params = []
    for submodel in model.submodels.flat_data:
        all_params += list(submodel.parameters())
    all_params.append(model.W)

    # 1) 采样
    pts = domain.in_sample(num_samples=40000)  # (N,3)
    bbox = domain.get_bounding_box()
    x_min, x_max, y_min, y_max, z_min, z_max = domain.get_bounding_box()
    extent = max(x_max - x_min, y_max - y_min, z_max - z_min)
    x_bnd = torch.rand((10000, 3), device=pts.device)
    x_bnd[:, 0] = bbox[0] + (bbox[1] - bbox[0]) * x_bnd[:, 0]
    x_bnd[:, 1] = bbox[2] + (bbox[3] - bbox[2]) * x_bnd[:, 1]
    x_bnd[:, 2] = bbox[4] + (bbox[5] - bbox[4]) * x_bnd[:, 2]

    # 2) 建树 + 查询（k=1 最近邻）
    tree = TorchCKDTree(leafsize=32).fit(pts)
    dists, idx_nn = tree.query(x_bnd, k=1, workers=-1)  # dists: (M,), idx_nn: (M,)

    # 最近邻点坐标（仍在原设备）
    x_bnd_nn = pts[idx_nn]

    signed_dists = sdf_from_shape_and_dists(domain, x_bnd, dists, zero_eps=0.0).unsqueeze(-1)
    print(signed_dists)

    pts = domain.in_sample(num_samples=500000)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(all_params, lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',  # 根据最小化loss调整
        factor=0.9,  # 学习率乘以0.9
        patience=20,  # 20轮没有改善才降低学习率
    )
    x_train = torch.cat([x_bnd, pts], dim=0)
    y_train = torch.cat([signed_dists, torch.zeros((pts.shape[0], 1))],
                        dim=0)
    batch_size = 4096
    num_samples = x_train.size(0)
    num_epochs = 1000


    def clamp_loss(pred, target, delta):
        pred_clamped = torch.clamp(pred, -delta, delta)
        target_clamped = torch.clamp(target, -delta, delta)
        return torch.abs(pred_clamped - target_clamped).mean()


    for epoch in range(num_epochs):
        epoch_loss = 0.0

        # 打乱索引
        indices = torch.randperm(num_samples)

        for start in range(0, num_samples, batch_size):
            end = start + batch_size
            idx = indices[start:end]

            x_batch = x_train[idx]
            y_batch = y_train[idx]

            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            # loss = clamp_loss(y_pred, y_batch, delta=0.2 * extent)  # Set delta as needed

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

    # A = model.features(torch.cat([x_bnd, pts], dim=0)).cat(dim=1)
    # b = torch.cat([signed_dists, torch.zeros_like(pts[:, :1])], dim=0)
    #
    # model.compute(A).solve(b)

    pts = domain.in_sample(num_samples=10000)
    print(model(pts))


    # plot_model_slice(model, bbox=domain.get_bounding_box(), axis='z', value=0.0, res=300, device=x_in.device)

    # class NearEllipsoid(pyrfm.ImplicitSurfaceBase):
    #
    #     def __init__(self):
    #         super().__init__()
    #         self.a, self.b, self.c = 1.5, 1.0, 0.5
    #
    #     def get_bounding_box(self) -> List[float]:
    #         ratio = 2.0
    #         return [-self.a * ratio, self.a * ratio, -self.b * ratio, self.b * ratio, -self.c * ratio, self.c * ratio]
    #
    #     def shape_func(self, p: torch.Tensor) -> torch.Tensor:
    #         # Ellipsoid equation: (x/a)^2 + (y/b)^2 + (z/c)^2 = 1
    #
    #         return model(p)

    class NearShape(pyrfm.GeometryBase):

        def __init__(self):
            super().__init__(dim=3, intrinsic_dim=2)
            self.a, self.b, self.c = 1.5, 1.0, 0.5

        def get_bounding_box(self) -> List[float]:
            return domain.get_bounding_box()

        def in_sample(self, num_samples: int, with_boundary: bool = False) -> torch.Tensor:
            pass

        def on_sample(self, num_samples: int, with_normal: bool = False) -> Union[
            torch.Tensor, Tuple[torch.Tensor, ...]]:
            pass

        def sdf(self, p: torch.Tensor) -> torch.Tensor:
            return model(p)


    near_shape = NearShape()
    near_model = pyrfm.RFMBase(dim=3, n_hidden=100, domain=near_shape, n_subdomains=1)
    near_model.W = torch.rand((100, 1))

    viz = pyrfm.RFMVisualizer3DMC(near_model, t=0.0, resolution=(1920, 1080), component_idx=0, view='iso')
    # viz = pyrfm.RFMVisualizer3D(near_model, t=0.0, resolution=(1920, 1080), component_idx=0, view='iso')

    # 可选：调整网格与等值面（默认为 level=0.0，grid=(128,128,128)）
    viz.plot(cmap='viridis', level=0.0, grid=(160, 160, 160))
    viz.show()
