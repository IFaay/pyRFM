import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# ----------------------------
# Helper
# ----------------------------
def sample_square(n):
    return np.random.rand(n, 2)


def is_on_boundary(p, eps=1e-2):
    return (
            abs(p[0] - 0) < eps or
            abs(p[0] - 1) < eps or
            abs(p[1] - 0) < eps or
            abs(p[1] - 1) < eps
    )


def farthest_point_step(existing_all, domain_sampler, candidates=5000):
    """从 existing_all 计算最远点"""
    Xcand = domain_sampler(candidates)
    dists = np.linalg.norm(Xcand[:, None, :] - existing_all[None, :, :], axis=-1)
    min_dist = dists.min(axis=1)
    idx = np.argmax(min_dist)
    return Xcand[idx]


# ----------------------------
# Initial 50 via KMeans
# ----------------------------
x_raw = sample_square(2000)
kmeans = KMeans(n_clusters=50, n_init=10).fit(x_raw)
initial_50 = kmeans.cluster_centers_

existing_all = initial_50.copy()  # 所有点都会加入这里用于距离计算
accepted = []  # 最终有效 FPS 点
rejected = []  # 落在边界的点（用于显示）

target_new = 50

# ----------------------------
# Dynamic visualization
# ----------------------------
plt.ion()
fig, ax = plt.subplots(figsize=(6, 6))

while len(accepted) < target_new:
    newp = farthest_point_step(existing_all, sample_square)

    on_bd = is_on_boundary(newp)
    if on_bd:
        rejected.append(newp)  # 不加入 accepted
    else:
        accepted.append(newp)  # 加入最终有效集合

    existing_all = np.vstack([existing_all, newp])  # 关键：所有点都加入计算集合

    # --- 绘图 ---
    ax.clear()
    ax.scatter(x_raw[:, 0], x_raw[:, 1], s=2, alpha=0.08)

    # 初始50
    ax.scatter(initial_50[:, 0], initial_50[:, 1], s=30, c='red', label='KMeans 50')

    # 有效 FPS
    if accepted:
        acc = np.array(accepted)
        ax.scatter(acc[:, 0], acc[:, 1], s=30, c='blue', label='FPS accepted')

    # 边界拒绝点（半透明）
    if rejected:
        rej = np.array(rejected)
        ax.scatter(rej[:, 0], rej[:, 1],
                   s=50, color=(0.3, 0.3, 0.3, 0.3),
                   label='Boundary (ignored final result)')

    # 高亮当前点
    ax.scatter(newp[0], newp[1], s=120,
               facecolors='none', edgecolors='black', linewidths=2)

    ax.set_title(f"FPS Step = {len(accepted) + len(rejected)}  |  Accepted = {len(accepted)}/{target_new}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc='lower left')

    plt.pause(0.25)

plt.ioff()
plt.show()
