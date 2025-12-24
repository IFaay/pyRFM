"""
Geometric-Regularized Mixture-of-Experts (MoE) demo
Author: Yifei Sun + ChatGPT
Date: 2025-11
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
import matplotlib
import matplotlib.pyplot as plt

import pyrfm


# ============================================================
# 1. 数据采样 + K-Means 初始化
# ============================================================

def init_centers(x, k=6):
    """K-Means 初始化几何中心（仅用于对比）"""
    km = KMeans(n_clusters=k, n_init=10)
    km.fit(x)
    return torch.tensor(km.cluster_centers_, device=x.device)


# ============================================================
# 2. 路由网络 Router
# ============================================================

class Router(nn.Module):
    def __init__(self, dim_in, num_experts, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_experts)
        )

    def forward(self, x, temp=0.1):
        logits = self.net(x)
        return F.softmax(logits / temp, dim=-1)


# ============================================================
# 3. Experts
# ============================================================

class Expert(nn.Module):
    def __init__(self, dim_in, hidden=400):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, hidden),
            nn.Tanh(),
            nn.Linear(hidden, dim_in)
        )

    def forward(self, x):
        return self.net(x)


class MoERegressor(nn.Module):
    def __init__(self, dim_in, num_experts):
        super().__init__()
        self.router = Router(dim_in, num_experts)
        self.experts = nn.ModuleList([Expert(dim_in) for _ in range(num_experts)])

    def forward(self, x, temp=0.1):
        w = self.router(x, temp=temp)  # [N,K]
        outs = torch.stack([e(x) for e in self.experts], dim=1)  # [N,K,2]
        y_pred = (w.unsqueeze(-1) * outs).sum(dim=1)
        return y_pred, w


# ============================================================
# 4. 几何正则：基于 Top-1 的组内方差（最终版）
# ============================================================

def geometric_intra_variance_loss(x, w, min_points=3):
    """
    使用 w 的 Top-1 assignment，将 x 分组，
    计算每组的方差，并对所有有效组取均值。
    """
    top1 = torch.argmax(w, dim=-1)
    K = w.shape[1]

    vars_list = []
    for i in range(K):
        pts = x[top1 == i]
        if pts.shape[0] <= min_points:
            continue
        vars_list.append(pts.var(dim=0).mean())

    if not vars_list:
        return torch.tensor(0.0, device=x.device)

    return torch.stack(vars_list).mean()


# ============================================================
# 5. Loss 组合
# ============================================================

def loss_fn(x, y_pred, w, lam_geo=1e-1, lam_ent=1e-3, lam_bal=1.0):
    L_reg = ((x - y_pred) ** 2).mean()
    L_geo = geometric_intra_variance_loss(x, w)
    L_ent = -(w * torch.log(w + 1e-8)).sum(-1).mean()
    L_bal = w.mean(0).var()

    total_loss = L_reg + lam_geo * L_geo + lam_ent * L_ent + lam_bal * L_bal

    return total_loss, {
        "L_reg": f"{L_reg.item():.4e}",
        "L_geo": f"{L_geo.item():.4e}",
        "L_ent": f"{L_ent.item():.4e}",
        "L_bal": f"{L_bal.item():.4e}",
    }


# ============================================================
# 6. 可视化（Top-1 & Top-2 + 动态中心点）
# ============================================================

def visualize_router_vs_kmeans(x, w, km_centers, step):
    """
    三联图可视化:
        (1) MoE Top-1 Router Partition
        (2) MoE Top-2 Router Partition
        (3) K-Means Top-1 Partition (用于对比)
    """
    with torch.no_grad():
        N, K = w.shape

        # ------------------------------------------------------------
        # MoE Top-1 / Top-2
        # ------------------------------------------------------------
        top2_val, top2_idx = torch.topk(w, 2, dim=-1)
        router_top1 = top2_idx[:, 0].cpu().numpy()
        router_top2 = top2_idx[:, 1].cpu().numpy()

        # ------------------------------------------------------------
        # 动态重计算 MoE 中心点（基于 Top-1）
        # ------------------------------------------------------------
        top1_t = torch.tensor(router_top1, device=x.device)
        moe_centers = []
        for i in range(K):
            pts = x[top1_t == i]
            moe_centers.append(pts.mean(dim=0) if pts.shape[0] > 0 else km_centers[i])
        moe_centers = torch.stack(moe_centers)

        # ------------------------------------------------------------
        # K-Means Top-1 assignment (对比用)
        # ------------------------------------------------------------
        km_top1 = torch.cdist(x, km_centers).argmin(dim=-1).cpu().numpy()

        # ------------------------------------------------------------
        # 颜色方案（与 MoE 完全一致）
        # ------------------------------------------------------------
        if K <= 10:
            base_cmap = matplotlib.colormaps['tab10']
        elif K <= 20:
            base_cmap = matplotlib.colormaps['tab20']
        else:
            base_cmap = matplotlib.colormaps['viridis']

        colors = [base_cmap(i / max(K - 1, 1)) for i in range(K)]
        discrete_cmap = matplotlib.colors.ListedColormap(colors)
        bounds = np.arange(-0.5, K + 0.5, 1)
        norm = matplotlib.colors.BoundaryNorm(bounds, discrete_cmap.N)

        # ------------------------------------------------------------
        # 三联图
        # ------------------------------------------------------------
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        names = [
            "MoE Top-1 Router Partition",
            "MoE Top-2 Router Partition",
            "K-Means Top-1 Partition"
        ]

        datas = [router_top1, router_top2, km_top1]
        centers_list = [moe_centers, moe_centers, km_centers]

        for ax, data, title, centers_used in zip(axes, datas, names, centers_list):
            sc = ax.scatter(
                x[:, 0].cpu(), x[:, 1].cpu(),
                c=data, s=8,
                cmap=discrete_cmap,
                norm=norm,
                edgecolors="none"
            )

            # 绘制对应中心点
            for i, c in enumerate(centers_used):
                ax.scatter(
                    c[0].cpu(), c[1].cpu(),
                    color=colors[i],
                    marker="x",
                    s=120,
                    linewidths=2.5
                )

            cbar = plt.colorbar(sc, ax=ax, boundaries=bounds, ticks=range(K))
            cbar.ax.set_yticklabels([f"E{i}" for i in range(K)])

            ax.set_title(f"{title} (step {step})")
            ax.set_aspect('equal')

        plt.tight_layout()
        plt.show()


# ============================================================
# 7. 训练流程
# ============================================================

def train_moe(num_experts=6, steps=2000, lr=1e-3, plot_interval=200):
    # ==== 几何域采样 ====
    # base = pyrfm.Square2D(center=[2.0, 1.5], half=[0.5, 0.5])
    #
    # region1 = pyrfm.Circle2D(center=[1.9576, 1.2456], radius=0.0502)
    # region2 = pyrfm.Circle2D(center=[2.1249, 1.5534], radius=0.2743)
    # region3 = pyrfm.Circle2D(center=[1.9883, 1.5191], radius=0.2666)
    #
    # cut1 = pyrfm.Circle2D(center=[2.1955, 1.7093], radius=0.1413)
    # cut2 = pyrfm.Circle2D(center=[2.0465, 1.0623], radius=0.1595)
    # cut3 = pyrfm.Circle2D(center=[1.7483, 1.4980], radius=0.1915)
    # cut4 = pyrfm.Circle2D(center=[1.9530, 1.6899], radius=0.0919)
    # domain = pyrfm.IntersectionGeometry(base, region1 + region2 + region3) - (cut1 + cut2 + cut3 + cut4)

    domain = (pyrfm.Square2D(center=[-0.75, 0], half=[0.25, 1])
              + pyrfm.Square2D(center=[0.75, 0], half=[0.25, 1])
              + pyrfm.Square2D(center=[0, 0.75], half=[1.0, 0.25]))

    # domain = pyrfm.Circle2D((0, 0), 1) - pyrfm.Circle2D((0, 0), 0.5)

    # domain = pyrfm.Square2D(center=[0, 0], half=[1, 1])

    x = domain.in_sample(10000, with_boundary=True)

    # ==== K-Means 初始化，用于对比 ====
    km_centers = init_centers(x.cpu().numpy(), num_experts).to(x.device)

    model = MoERegressor(2, num_experts)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for step in range(steps + 1):
        opt.zero_grad()
        y_pred, w = model(x)
        loss, log_dict = loss_fn(x, y_pred, w)
        loss.backward()
        opt.step()

        if step % plot_interval == 0:
            print(f"Step {step:4d}: "
                  f"L={loss.item():.4e}, "
                  f"L_reg={log_dict['L_reg']} "
                  f"L_geo={log_dict['L_geo']} "
                  f"L_ent={log_dict['L_ent']} "
                  f"L_bal={log_dict['L_bal']}"
                  )
            visualize_router_vs_kmeans(x, w, km_centers, step)

    return model, km_centers


# ============================================================
# 8. 运行
# ============================================================

if __name__ == "__main__":
    torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')
    model, km_centers = train_moe(num_experts=7, steps=8000, lr=1e-3)
