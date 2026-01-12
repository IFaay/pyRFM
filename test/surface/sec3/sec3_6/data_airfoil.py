# -*- coding: utf-8 -*-
"""
Gmsh-based sampling of NACA0012 STEP:

- Use Gmsh to:
    * import Naca0012.STEP
    * generate a surface mesh (2D on CAD surfaces)
    * extract nodes, normals, and mean curvature on each surface

- Organize point sets:
    x               : all points on Γ'            (all surfaces)
    x_trim_in       : points on Γ                (wing surfaces)
    x_trim_out      : points on Γ' \ Γ           (section surfaces)
    x_trim_boundary : points on ∂Γ               (intersection of Γ and section)

Saved files:
    ../../data/airfoil_in.pth         -> (x, normal, mean_curvature)
    ../../data/airfoil_trim_sets.pth  -> (x_trim_in, x_trim_out, x_trim_boundary)
"""

from pathlib import Path
from typing import Iterable, Tuple, Dict

import numpy as np
import torch
import matplotlib.pyplot as plt

import gmsh


# =========================================================
# Gmsh sampling helpers
# =========================================================

def generate_surface_mesh_with_gmsh(
        step_file: str | Path,
        mesh_size: float = 10.0,
) -> None:
    """
    Use Gmsh to import STEP and generate a 2D surface mesh.

    After this call, the current Gmsh model contains:
        - imported CAD geometry
        - a 2D mesh on all surfaces
    """
    step_file = Path(step_file)
    if not step_file.exists():
        raise FileNotFoundError(step_file)

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)

    gmsh.model.add("NACA0012")
    gmsh.model.occ.importShapes(str(step_file))
    gmsh.model.occ.synchronize()

    # Report surfaces for debugging / tag inspection
    surfaces = gmsh.model.getEntities(dim=2)
    print(f"[INFO] Surfaces (dim=2): {surfaces}")

    # Uniform mesh size (coarse/medium, adjust as needed)
    gmsh.option.setNumber("Mesh.MeshSizeMax", mesh_size)
    # gmsh.option.setNumber("Mesh.MeshSizeMin", mesh_size)

    print("[INFO] Generating 2D surface mesh...")
    gmsh.model.mesh.generate(2)
    print("[INFO] Mesh generation done.")


def _accumulate_node_data_from_surface(
        surf_tag: int,
        section_surface_tags: Iterable[int],
        all_pos: Dict[int, np.ndarray],
        all_normal: Dict[int, np.ndarray],
        all_H: Dict[int, float],
        all_tags: set[int],
        wing_tags: set[int],
        section_tags: set[int],
) -> None:
    """
    For a single surface (dim=2, tag=surf_tag), collect all nodes, normals, and
    mean curvature, and update:
        - global maps: all_pos, all_normal, all_H
        - global tag sets: all_tags, wing_tags, section_tags
    """
    # Get nodes on this surface, including boundary nodes, with parametric coords
    nodeTags, coord, param = gmsh.model.mesh.getNodes(
        dim=2, tag=surf_tag, includeBoundary=True, returnParametricCoord=True
    )

    if len(nodeTags) == 0:
        return

    coord = np.asarray(coord, dtype=np.float64).reshape(-1, 3)

    # param is [u1, v1, u2, v2, ...]
    param = np.asarray(param, dtype=np.float64)
    if param.size != 2 * coord.shape[0]:
        # Fallback: if no param coords are available (discrete surface),
        # we skip normals/curvature and set zeros.
        print(f"[WARN] Surface {surf_tag}: parametric coords missing/inconsistent.")
        normals = np.zeros_like(coord)
        H = np.zeros(coord.shape[0], dtype=np.float64)
    else:
        # Get normals from CAD
        normals = gmsh.model.getNormal(surf_tag, param.tolist())
        normals = np.asarray(normals, dtype=np.float64).reshape(-1, 3)

        # Principal curvatures -> mean curvature H = (k_max + k_min)/2
        k_max, k_min, dir_max, dir_min = gmsh.model.getPrincipalCurvatures(
            surf_tag, param.tolist()
        )
        k_max = np.asarray(k_max, dtype=np.float64).ravel()
        k_min = np.asarray(k_min, dtype=np.float64).ravel()
        H = 0.5 * (k_max + k_min)

    # Update maps and sets
    is_section = surf_tag in section_surface_tags

    for i, tag in enumerate(nodeTags):
        tag = int(tag)

        # Global unique storage for node position/normal/curvature
        if tag not in all_pos:
            all_pos[tag] = coord[i]
            all_normal[tag] = normals[i]
            all_H[tag] = float(H[i])

        all_tags.add(tag)
        if is_section:
            section_tags.add(tag)
        else:
            wing_tags.add(tag)


def extract_airfoil_points_from_gmsh(
        section_surface_tags: Iterable[int] = (10, 11),
        device: str = "cpu",
        dtype=torch.float64,
) -> Tuple[
    torch.Tensor,  # x  (all)
    torch.Tensor,  # normal
    torch.Tensor,  # mean curvature
    torch.Tensor,  # x_trim_in
    torch.Tensor,  # x_trim_out
    torch.Tensor,  # x_trim_boundary
]:
    """
    From the current Gmsh model (must already contain a 2D surface mesh),
    build the point sets:

        x               : all points on Γ'
        normal          : unit normals on Γ'
        mean_curvature  : mean curvature on Γ'
        x_trim_in       : points on Γ         (wing surfaces)
        x_trim_out      : points on Γ'\Γ      (section surfaces)
        x_trim_boundary : intersection ∂Γ     (nodes shared by wing & section)

    section_surface_tags:
        Surface tags to be treated as "section faces" (like face 10, 11 before).
        All other surfaces are treated as "wing faces".
    """
    section_surface_tags = set(int(t) for t in section_surface_tags)

    # Global maps: nodeTag -> data
    all_pos: Dict[int, np.ndarray] = {}
    all_normal: Dict[int, np.ndarray] = {}
    all_H: Dict[int, float] = {}

    # Global tag sets
    all_tags: set[int] = set()
    wing_tags: set[int] = set()
    section_tags: set[int] = set()

    # Traverse all 2D entities (surfaces)
    surfaces = gmsh.model.getEntities(dim=2)
    print(f"[INFO] Found {len(surfaces)} surfaces for sampling.")

    for dim, sTag in surfaces:
        assert dim == 2
        print(f"[INFO] Sampling surface tag = {sTag} "
              f"{'(section)' if sTag in section_surface_tags else '(wing/other)'}")
        _accumulate_node_data_from_surface(
            surf_tag=sTag,
            section_surface_tags=section_surface_tags,
            all_pos=all_pos,
            all_normal=all_normal,
            all_H=all_H,
            all_tags=all_tags,
            wing_tags=wing_tags,
            section_tags=section_tags,
        )

    # Set operations for trim sets
    boundary_tags = wing_tags & section_tags
    trim_in_tags = wing_tags - boundary_tags
    trim_out_tags = section_tags - boundary_tags

    print(f"[INFO] #all nodes       = {len(all_tags)}")
    print(f"[INFO] #wing nodes      = {len(wing_tags)}")
    print(f"[INFO] #section nodes   = {len(section_tags)}")
    print(f"[INFO] #boundary nodes  = {len(boundary_tags)}")
    print(f"[INFO] #trim_in nodes   = {len(trim_in_tags)}")
    print(f"[INFO] #trim_out nodes  = {len(trim_out_tags)}")

    # Helper to pack tags -> torch tensor
    def pack_positions(tags: Iterable[int]) -> torch.Tensor:
        tags_sorted = sorted(tags)
        P = np.array([all_pos[t] for t in tags_sorted], dtype=np.float64)
        return torch.as_tensor(P, device=device, dtype=dtype)

    def pack_all() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tags_sorted = sorted(all_tags)
        P = np.array([all_pos[t] for t in tags_sorted], dtype=np.float64)
        N = np.array([all_normal[t] for t in tags_sorted], dtype=np.float64)
        H = np.array([[all_H[t]] for t in tags_sorted], dtype=np.float64)
        return (
            torch.as_tensor(P, device=device, dtype=dtype),
            torch.as_tensor(N, device=device, dtype=dtype),
            torch.as_tensor(H, device=device, dtype=dtype),
        )

    x, normal, mean_curvature = pack_all()
    x_trim_in = pack_positions(trim_in_tags)
    x_trim_out = pack_positions(trim_out_tags)
    x_trim_boundary = pack_positions(boundary_tags)

    return x, normal, mean_curvature, x_trim_in, x_trim_out, x_trim_boundary


# =========================================================
# Visualization (optional, same接口)
# =========================================================

def plot_airfoil_points(P_wing: torch.Tensor,
                        P_section: torch.Tensor,
                        P_wing_boundary: torch.Tensor,
                        subsample: int = 5) -> None:
    Pw = P_wing[::subsample].cpu().numpy()
    Ps = P_section[::subsample].cpu().numpy()
    Pb = P_wing_boundary[::max(1, subsample // 2)].cpu().numpy()

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(Pw[:, 0], Pw[:, 1], Pw[:, 2], s=1, alpha=0.6, label="Wing (Γ)")
    ax.scatter(Ps[:, 0], Ps[:, 1], Ps[:, 2], s=2, alpha=0.9, label="Section (Γ'\\Γ)")
    ax.scatter(Pb[:, 0], Pb[:, 1], Pb[:, 2], s=6, c="k", label="Boundary (∂Γ)")

    # equal aspect ratio
    max_range = np.array([
        Pw[:, 0].max() - Pw[:, 0].min(),
        Pw[:, 1].max() - Pw[:, 1].min(),
        Pw[:, 2].max() - Pw[:, 2].min()
    ]).max() / 2.0
    mid_x = (Pw[:, 0].max() + Pw[:, 0].min()) * 0.5
    mid_y = (Pw[:, 1].max() + Pw[:, 1].min()) * 0.5
    mid_z = (Pw[:, 2].max() + Pw[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_airfoil_all(Pw: torch.Tensor, subsample: int = 5) -> None:
    Pw = Pw[::subsample].cpu().numpy()

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(Pw[:, 0], Pw[:, 1], Pw[:, 2], s=1, alpha=0.6, label="ALL")

    max_range = np.array([
        Pw[:, 0].max() - Pw[:, 0].min(),
        Pw[:, 1].max() - Pw[:, 1].min(),
        Pw[:, 2].max() - Pw[:, 2].min()
    ]).max() / 2.0
    mid_x = (Pw[:, 0].max() + Pw[:, 0].min()) * 0.5
    mid_y = (Pw[:, 1].max() + Pw[:, 1].min()) * 0.5
    mid_z = (Pw[:, 2].max() + Pw[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.legend()
    plt.tight_layout()
    plt.show()


# =========================================================
# Entry
# =========================================================

if __name__ == "__main__":
    step_path = "Naca0012.STEP"
    mesh_size = 5.0

    # Default device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)

    # 1) 用 Gmsh 生成表面网格
    generate_surface_mesh_with_gmsh(step_path, mesh_size=mesh_size)

    try:
        # 2) 从 Gmsh 提取点集 + 法向 + 曲率
        (
            x,
            normal,
            mean_curvature,
            x_trim_in,
            x_trim_out,
            x_trim_boundary,
        ) = extract_airfoil_points_from_gmsh(
            section_surface_tags=(10, 11),  # 如有需要可改
            device=device,
            dtype=torch.float64,
        )

        # 3) 保存与之前保持兼容
        torch.save((x, normal, mean_curvature), "../../data/airfoil_in.pth")
        torch.save(
            (x_trim_in, x_trim_out, x_trim_boundary),
            "../../data/airfoil_trim_sets.pth",
        )

        print("[INFO] Saved ../../data/airfoil_in.pth")
        print("[INFO] Saved ../../data/airfoil_trim_sets.pth")
        print(x.shape, normal.shape, mean_curvature.shape,
              x.dtype, x.device)
        print(x_trim_in.shape, x_trim_out.shape, x_trim_boundary.shape)

        # 可选可视化：用几何分组而不是直接 x_trim_*，
        # 这里简单地用 trim 集合重构 P_wing/P_section/P_boundary
        P_wing = x_trim_in
        P_section = x_trim_out
        P_boundary = x_trim_boundary

        plot_airfoil_points(P_wing, P_section, P_boundary)
        plot_airfoil_all(x)

    finally:
        gmsh.finalize()
