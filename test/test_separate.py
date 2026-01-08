import pyrfm
import torch
import matplotlib.pyplot as plt

if __name__ == "__main__":
    box1 = pyrfm.Square2D(center=(0, 0), half=(1, 1))
    box2 = pyrfm.Square2D(center=(0.5, 0.5), half=(0.5, 0.5))

    domain = box1 - box2
    groups = domain.on_sample(1000, with_normal=True, separate=True)

    # 简单 sanity check
    print(f"number of groups: {len(groups)}")
    for i, (pts, nrm) in enumerate(groups):
        print(f"group {i}: {pts.shape[0]} points")

    # ------- 画图 -------
    fig, ax = plt.subplots(figsize=(6, 6))

    for i, (pts, nrm) in enumerate(groups):
        # torch -> numpy
        pts_np = pts.detach().cpu().numpy()
        nrm_np = nrm.detach().cpu().numpy()

        # 只取前两个维度 (2D)
        x = pts_np[:, 0]
        y = pts_np[:, 1]
        u = nrm_np[:, 0]
        v = nrm_np[:, 1]

        # 画采样点
        ax.scatter(x, y, s=5, alpha=0.3, label=f"group {i} pts")

        # 画法向量（可以适当缩放，不然箭头太长）
        # scale 越大，箭头越短；可根据实际情况调一下
        ax.quiver(
            x, y, u, v,
            angles="xy",
            scale_units="xy",
            scale=10.0,
            width=0.002,
            alpha=0.7,
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Boundary samples and normals per group")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.show()
