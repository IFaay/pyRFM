"""
Free-form MoE domain decomposition (repulsion-based, no collapse, with KMeans comparison)
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
# 1. Router
# ============================================================

class Router(nn.Module):
    def __init__(self, dim_in, num_experts, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_experts)
        )

    def forward(self, x, temp=1.0):
        logits = self.net(x)
        return F.softmax(logits / temp, dim=-1)


# ============================================================
# 2. Experts（不重要，仅用于任务）
# ============================================================

class Expert(nn.Module):
    def __init__(self, dim_in, hidden=128):
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

    def forward(self, x):
        w = self.router(x)
        outs = torch.stack([exp(x) for exp in self.experts], dim=1)
        top_val, top_idx = torch.topk(w, 2, dim=-1)
        mask = torch.zeros_like(w)
        mask.scatter_(1, top_idx, 1.0)
        w_top = w * mask
        w_top = w_top / (w_top.sum(dim=-1, keepdim=True) + 1e-8)
        y_pred = (w_top.unsqueeze(-1) * outs).sum(dim=-2)
        return y_pred, w


# ============================================================
# 3. Regularizations
# ============================================================

def laplacian_smoothness(x, w, k=20):
    dist = torch.cdist(x, x)
    knn_idx = dist.topk(k, largest=False).indices
    w_neighbor = w[knn_idx]
    diff = (w_neighbor - w.unsqueeze(1)).pow(2).mean()
    return diff


def area_balance(w):
    A = w.mean(dim=0)
    target = torch.full_like(A, 1.0 / w.shape[1])
    return ((A - target) ** 2).mean()


def sharpness(w):
    return (w * torch.log(w + 1e-8)).sum(dim=-1).mean()


def repulsive_centers(x, w, sigma=0.3):
    N, K = w.shape
    centers = []

    for i in range(K):
        wi = w[:, i:i + 1]
        ci = (wi * x).sum(dim=0) / (wi.sum() + 1e-8)
        centers.append(ci)

    centers = torch.stack(centers)
    dist = torch.cdist(centers, centers) + torch.eye(K, device=x.device)
    repulse = torch.exp(-(dist ** 2) / sigma ** 2)
    repulse = repulse.sum() - torch.trace(repulse)
    return repulse / (K * K)


# ============================================================
# 4. Loss
# ============================================================

def loss_fn(x, y_pred, w,
            lam_rep=5e-2, lam_smooth=2e-3, lam_area=1e-1, lam_sharp=5e-4):
    L_reg = ((x - y_pred) ** 2).mean()
    L_smooth = laplacian_smoothness(x, w)
    L_area = area_balance(w)
    L_sharp = sharpness(w)
    L_rep = repulsive_centers(x, w)

    total = (
            L_reg
            + lam_rep * L_rep
            + lam_smooth * L_smooth
            + lam_area * L_area
            + lam_sharp * L_sharp
    )

    return total, {
        "L_reg": f"{L_reg.item():.4e}",
        "L_rep": f"{L_rep.item():.4e}",
        "L_smooth": f"{L_smooth.item():.4e}",
        "L_area": f"{L_area.item():.4e}",
        "L_sharp": f"{L_sharp.item():.4e}",
    }


# ============================================================
# 5. KMeans baseline
# ============================================================

def kmeans_partition(x, num_experts):
    km = KMeans(n_clusters=num_experts, n_init=10)
    labels = km.fit_predict(x.cpu().numpy())
    centers = km.cluster_centers_
    return labels, centers


# ============================================================
# 6. Router Top-1 visualization
# ============================================================

def visualize_router_top1(x, w, step):
    top1 = torch.argmax(w, dim=-1).cpu().numpy()
    K = w.shape[1]
    cmap = matplotlib.colormaps['tab10'] if K <= 10 else matplotlib.colormaps['tab20']

    plt.figure(figsize=(6, 5))
    plt.scatter(x[:, 0].cpu(), x[:, 1].cpu(), c=top1, cmap=cmap, s=6)
    plt.title(f"Free-form MoE Top-1 Partition (step {step})")
    plt.gca().set_aspect('equal', 'box')
    plt.colorbar()
    plt.show()


# ============================================================
# 7. Full Top-1 & Top-2 Comparison (Router vs KMeans)
# ============================================================

def visualize_router_vs_kmeans_top12(x, w_router, kmeans_labels, kmeans_centers, step):
    """
    六联图对比：
      行 1：Top-1
        (1) Router Top-1
        (2) KMeans Top-1
        (3) Difference Top-1 (router - kmeans)

      行 2：Top-2
        (4) Router Top-2
        (5) KMeans Top-2
        (6) Difference Top-2
    """
    with torch.no_grad():
        N, K = w_router.shape

        # Router Top-1 / Top-2
        r_top2_val, r_top2_idx = torch.topk(w_router, k=2, dim=-1)
        router_top1 = r_top2_idx[:, 0].cpu().numpy()
        router_top2 = r_top2_idx[:, 1].cpu().numpy()

        # KMeans Top-1（直接 labels）
        kmeans_top1 = kmeans_labels.copy()

        # KMeans Top-2（按距离排序）
        dist = np.linalg.norm(
            x.cpu().numpy()[:, None, :] - kmeans_centers[None, :, :], axis=-1
        )
        km_sorted = np.argsort(dist, axis=1)
        kmeans_top2 = km_sorted[:, 1]

        # Differences
        diff1 = router_top1 - kmeans_top1
        diff2 = router_top2 - kmeans_top2

        # Discrete colormap
        if K <= 10:
            base_cmap = matplotlib.colormaps['tab10']
        else:
            base_cmap = matplotlib.colormaps['tab20']
        colors = [base_cmap(i / max(K - 1, 1)) for i in range(K)]
        discrete_cmap = matplotlib.colors.ListedColormap(colors)
        bounds = np.arange(-0.5, K + 0.5, 1)
        norm = matplotlib.colors.BoundaryNorm(bounds, discrete_cmap.N)

        # Plot 2×3
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))

        titles = [
            "Router Top-1", "KMeans Top-1", "Top-1 Difference",
            "Router Top-2", "KMeans Top-2", "Top-2 Difference"
        ]
        datasets = [
            router_top1, kmeans_top1, diff1,
            router_top2, kmeans_top2, diff2
        ]
        cmaps = [
            discrete_cmap, discrete_cmap, "coolwarm",
            discrete_cmap, discrete_cmap, "coolwarm"
        ]
        norms = [
            norm, norm, None,
            norm, norm, None
        ]

        for ax, data, title, cmap, nm in zip(axes.flatten(), datasets, titles, cmaps, norms):

            sc = ax.scatter(
                x[:, 0].cpu(), x[:, 1].cpu(),
                c=data, cmap=cmap, norm=nm,
                s=6, edgecolors='none'
            )

            # KMeans centers (only for KMeans plots)
            if "KMeans" in title:
                for c in kmeans_centers:
                    ax.scatter(c[0], c[1], color='black', marker='x', s=120, linewidths=2)

            # Colorbar
            if "Difference" not in title:
                cbar = plt.colorbar(sc, ax=ax, boundaries=bounds, ticks=range(K))
                cbar.ax.set_yticklabels([f"E{i}" for i in range(K)])
            else:
                plt.colorbar(sc, ax=ax)

            ax.set_title(f"{title} (step {step})")
            ax.set_aspect('equal', 'box')

        plt.tight_layout()
        plt.show()


# ============================================================
# 8. Training Loop
# ============================================================

def train_moe(num_experts=7, steps=4000, lr=1e-3, plot_interval=200):
    # 建域
    base = pyrfm.Square2D(center=[2.0, 1.5], half=[0.5, 0.5])
    region1 = pyrfm.Circle2D([1.95, 1.24], 0.05)
    region2 = pyrfm.Circle2D([2.12, 1.55], 0.27)
    region3 = pyrfm.Circle2D([1.99, 1.52], 0.26)

    cut1 = pyrfm.Circle2D([2.19, 1.70], 0.14)
    cut2 = pyrfm.Circle2D([2.04, 1.06], 0.16)
    cut3 = pyrfm.Circle2D([1.75, 1.49], 0.19)
    cut4 = pyrfm.Circle2D([1.95, 1.69], 0.09)

    domain = pyrfm.IntersectionGeometry(base, region1 + region2 + region3) - (cut1 + cut2 + cut3 + cut4)

    x = domain.in_sample(6000, with_boundary=True)
    x = x[torch.randperm(x.shape[0])][:6000]

    model = MoERegressor(2, num_experts)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # ✅ 一次性 baseline
    kmeans_labels, kmeans_centers = kmeans_partition(x, num_experts)

    for step in range(steps + 1):
        opt.zero_grad()
        y_pred, w = model(x)
        loss, logs = loss_fn(x, y_pred, w)
        loss.backward()
        opt.step()

        if step % plot_interval == 0:
            print(f"[step {step}] "
                  f"L={loss.item():.4e} | "
                  f"Reg={logs['L_reg']} | "
                  f"Rep={logs['L_rep']} | "
                  f"S={logs['L_smooth']} | "
                  f"A={logs['L_area']} | "
                  f"Sh={logs['L_sharp']}")

            visualize_router_top1(x, w, step)
            visualize_router_vs_kmeans_top12(x, w, kmeans_labels, kmeans_centers, step)

    return model


# ============================================================
# 9. Run
# ============================================================

if __name__ == "__main__":
    torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')
    train_moe(num_experts=7, steps=4000, lr=1e-3)
