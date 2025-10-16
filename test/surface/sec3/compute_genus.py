# -*- coding: utf-8 -*-
"""
Created on 2025/10/16

@author: Yifei Sun
"""
# -*- coding: utf-8 -*-
"""
从隐式曲面 phi(x,y,z)=0 生成网格并计算亏格。
支持几何：
- Ellipsoid
- Torus
- Genus2 (genus-2 torus-like)
- Cheese (cheese-like surface)

依赖（任装其一即可）：
- scikit-image  (preferred)
- PyMCubes

保存：可选择将网格保存为 Wavefront OBJ（无纹理/法向，便于通用查看）。

用法示例：
    python implicit_mesh_genus.py --geom all --res 160 --save_obj
    python implicit_mesh_genus.py --geom torus --res 128
"""

import argparse
import math
import sys
from collections import deque, defaultdict

import numpy as np

# 尝试导入网格提取库（优先 skimage）
_HAS_SKIMAGE = False
_HAS_PYMCUBES = False
try:
    from skimage import measure as _sk_measure  # marching_cubes

    _HAS_SKIMAGE = True
except Exception:
    _HAS_SKIMAGE = False

if not _HAS_SKIMAGE:
    try:
        import mcubes as _mcubes  # PyMCubes

        _HAS_PYMCUBES = True
    except Exception:
        _HAS_PYMCUBES = False


# ----------------------------
# 定义四种隐式函数 phi(x,y,z)
# ----------------------------
def phi_ellipsoid(x, y, z):
    # (x/1.5)^2 + y^2 + (z/0.5)^2 - 1
    return (x / 1.5) ** 2 + y ** 2 + (z / 0.5) ** 2 - 1.0


def phi_torus(x, y, z):
    # ((sqrt(x^2 + y^2) - 1)^2 + z^2) - 1/16
    R = np.sqrt(x * x + y * y)
    return (R - 1.0) ** 2 + z * z - (1.0 / 16.0)


def phi_genus2(x, y, z):
    # ([(x+1) x^2 (x-1) + y^2]^2 + z^2) - 0.01
    val = ((x + 1.0) * (x ** 2) * (x - 1.0) + y ** 2) ** 2 + z ** 2 - 0.01
    return val


def phi_cheese(x, y, z):
    # ((4x^2-1)^2 + (4y^2-1)^2 + (4z^2-1)^2
    #  + 16(x^2+y^2-1)^2 + 16(x^2+z^2-1)^2 + 16(y^2+z^2-1)^2) - 16
    term1 = (4 * x * x - 1) ** 2 + (4 * y * y - 1) ** 2 + (4 * z * z - 1) ** 2
    term2 = 16 * (x * x + y * y - 1) ** 2
    term3 = 16 * (x * x + z * z - 1) ** 2
    term4 = 16 * (y * y + z * z - 1) ** 2
    return term1 + term2 + term3 + term4 - 16.0


PHI_DICT = {
    "ellipsoid": phi_ellipsoid,
    "torus": phi_torus,
    "genus2": phi_genus2,
    "cheese": phi_cheese,
}


# ----------------------------
# 体素采样与 Marching Cubes
# ----------------------------
def sample_phi_grid(phi_func, bounds=(-2.0, 2.0), res=128):
    """
    在立方体 [bmin,bmax]^3 上采样 phi 到 3D 网格。
    返回：
        vol: (res,res,res) 栅格值
        xs, ys, zs: 每个轴上的坐标数组（长度res）
        spacing: (dx, dy, dz)
    """
    bmin, bmax = bounds
    xs = np.linspace(bmin, bmax, res, dtype=np.float32)
    ys = np.linspace(bmin, bmax, res, dtype=np.float32)
    zs = np.linspace(bmin, bmax, res, dtype=np.float32)
    # 注意：为节省内存，逐层计算
    vol = np.empty((res, res, res), dtype=np.float32)
    for k, z in enumerate(zs):
        zz = np.full((res, res), z, dtype=np.float32)
        yy, xx = np.meshgrid(ys, xs, indexing="xy")
        vol[:, :, k] = phi_func(xx, yy, zz).astype(np.float32)
    spacing = (
        float(xs[1] - xs[0]),
        float(ys[1] - ys[0]),
        float(zs[1] - zs[0]),
    )
    return vol, xs, ys, zs, spacing


def marching_cubes_zero_level(vol, spacing, origin):
    """
    从体素标量场 vol 中提取等值面 level=0 的三角网格。
    返回：
        verts: (N,3) float 顶点坐标
        faces: (M,3) int   三角面顶点索引
    """
    if _HAS_SKIMAGE:
        # skimage.measure.marching_cubes 返回的坐标基于体素索引空间，
        # 需要乘 spacing 并加 origin 以转到世界坐标
        verts, faces, _, _ = _sk_measure.marching_cubes(
            vol, level=0.0, spacing=spacing, gradient_direction="descent"
        )
        verts = verts + origin  # origin 平移
        return verts.astype(np.float64), faces.astype(np.int64)

    if _HAS_PYMCUBES:
        # PyMCubes: 输入的是函数/体数据 & isovalue
        # 直接传体数据
        verts, faces = _mcubes.marching_cubes(vol, 0.0)
        # PyMCubes 顶点在体素索引空间，需要线性变换到世界坐标
        # 体素索引空间 [0, res-1]，坐标变换：x_world = origin + idx * spacing
        verts = verts.astype(np.float64)
        faces = faces.astype(np.int64)
        verts[:, 0] = origin[0] + verts[:, 0] * spacing[0]
        verts[:, 1] = origin[1] + verts[:, 1] * spacing[1]
        verts[:, 2] = origin[2] + verts[:, 2] * spacing[2]
        return verts, faces

    raise RuntimeError(
        "未找到可用的 Marching Cubes 库。请安装 scikit-image 或 PyMCubes。"
    )


# ----------------------------
# 网格拓扑与亏格计算
# ----------------------------
def connected_components_from_faces(n_vertices, faces):
    """
    基于三角面构造顶点邻接并分解连通分量（按顶点连通）。
    返回：list[ np.ndarray ]，每个为该分量内的顶点索引数组。
    """
    adj = [[] for _ in range(n_vertices)]
    for (a, b, c) in faces:
        adj[a].extend([b, c])
        adj[b].extend([a, c])
        adj[c].extend([a, b])
    visited = np.zeros(n_vertices, dtype=bool)
    comps = []
    for v in range(n_vertices):
        if visited[v]:
            continue
        dq = deque([v])
        visited[v] = True
        comp = [v]
        while dq:
            u = dq.popleft()
            for w in adj[u]:
                if not visited[w]:
                    visited[w] = True
                    dq.append(w)
                    comp.append(w)
        comps.append(np.array(comp, dtype=np.int64))
    return comps


def submesh_by_vertices(vertices, faces, keep_vertices):
    """
    从全局网格中提取仅包含 keep_vertices 的子网格（按顶点诱导）。
    会丢弃使用到外部顶点的三角形。
    同时会压缩顶点索引，使之从 0..n-1。
    返回：
        v_sub: (n_sub,3)
        f_sub: (m_sub,3)
    """
    keep_mask = np.zeros(len(vertices), dtype=bool)
    keep_mask[keep_vertices] = True
    # 保留三角形：三个顶点都在 keep 内
    tri_mask = keep_mask[faces].all(axis=1)
    f_sub_global = faces[tri_mask]
    # 压缩索引
    old_to_new = -np.ones(len(vertices), dtype=np.int64)
    old_to_new[keep_vertices] = np.arange(len(keep_vertices), dtype=np.int64)
    f_sub = old_to_new[f_sub_global]
    v_sub = vertices[keep_vertices]
    return v_sub, f_sub


def euler_characteristic(vertices, faces):
    """
    计算欧拉示性数 chi = V - E - F ？（注意应为 V - E + F）
    注意：E 用无向唯一边计数。
    """
    V = vertices.shape[0]
    F = faces.shape[0]
    # 累积无向边
    edges = set()
    for (a, b, c) in faces:
        e1 = (a, b) if a < b else (b, a)
        e2 = (b, c) if b < c else (c, b)
        e3 = (c, a) if c < a else (a, c)
        edges.add(e1);
        edges.add(e2);
        edges.add(e3)
    E = len(edges)
    chi = V - E + F
    return chi, V, E, F


def genus_from_chi(chi):
    """
    对封闭、可定向曲面：g = (2 - chi)/2
    """
    return (2 - chi) / 2.0


# ----------------------------
# OBJ 写出（最简单版本）
# ----------------------------
def save_obj(path, vertices, faces):
    """
    将三角网格保存为最简 OBJ（只写 v / f）。
    OBJ 索引是 1-based。
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write("# OBJ file generated from implicit surface\n")
        for v in vertices:
            f.write(f"v {v[0]:.9f} {v[1]:.9f} {v[2]:.9f}\n")
        for tri in faces:
            a, b, c = tri.tolist()
            f.write(f"f {a + 1} {b + 1} {c + 1}\n")


# ----------------------------
# 主流程
# ----------------------------
def process_one_geometry(name, phi_func, res, bounds, save_obj_flag):
    print(f"\n=== {name.upper()} ===")
    vol, xs, ys, zs, spacing = sample_phi_grid(phi_func, bounds=bounds, res=res)
    origin = np.array([xs[0], ys[0], zs[0]], dtype=np.float64)
    verts, faces = marching_cubes_zero_level(vol, spacing, origin)

    nV, nF = verts.shape[0], faces.shape[0]
    print(f"mesh: V={nV}, F={nF}")

    # 连通分量
    comps = connected_components_from_faces(nV, faces)
    print(f"connected components: {len(comps)}")

    total_report = []
    for idx, comp_vertices in enumerate(comps, start=1):
        v_sub, f_sub = submesh_by_vertices(verts, faces, comp_vertices)
        chi, V, E, F = euler_characteristic(v_sub, f_sub)
        g = genus_from_chi(chi)
        print(f" - component {idx}: V={V}, E={E}, F={F}, chi={chi}, genus={g}")
        total_report.append((idx, V, E, F, chi, g))

        if save_obj_flag:
            out_path = f"{name}_c{idx}_res{res}.obj"
            save_obj(out_path, v_sub, f_sub)
            print(f"   saved OBJ -> {out_path}")

    # 若只有一个分量，给出总述
    if len(comps) == 1:
        chi, V, E, F = euler_characteristic(verts, faces)
        g = genus_from_chi(chi)
        print(f"==> {name}: χ={chi}, genus={g}")

    return total_report


def main():
    parser = argparse.ArgumentParser(description="Implicit surfaces meshing & genus computation")
    parser.add_argument("--geom", type=str, default="all",
                        help="选择几何：ellipsoid | torus | genus2 | cheese | all")
    parser.add_argument("--res", type=int, default=256, help="采样分辨率（建议 96~256）")
    parser.add_argument("--bmin", type=float, default=-2.0, help="采样立方体最小坐标")
    parser.add_argument("--bmax", type=float, default=2.0, help="采样立方体最大坐标")
    parser.add_argument("--save_obj", action="store_true", help="保存 OBJ 网格")
    args = parser.parse_args()

    if not (_HAS_SKIMAGE or _HAS_PYMCUBES):
        print("错误：未检测到可用的 Marching Cubes 库。请安装 scikit-image 或 PyMCubes。")
        sys.exit(1)

    bounds = (args.bmin, args.bmax)

    if args.geom.lower() == "all":
        names = ["ellipsoid", "torus", "genus2", "cheese"]
    else:
        names = [args.geom.lower()]
        for n in names:
            if n not in PHI_DICT:
                print(f"未知几何类型：{n}")
                sys.exit(1)

    for n in names:
        process_one_geometry(n, PHI_DICT[n], args.res, bounds, args.save_obj)


if __name__ == "__main__":
    main()
