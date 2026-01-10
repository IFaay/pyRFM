# -*- coding: utf-8 -*-
"""
Dense sampling of NACA airfoil STEP surfaces with:
- strict face-inside test
- exact surface normal
- exact mean curvature
- final torch.tensor conversion

Each surface sample:
[x, y, z, nx, ny, nz, H]
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone

from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import (
    TopAbs_FACE,
    TopAbs_EDGE,
    TopAbs_IN,
    TopAbs_ON,
)

from OCC.Core.TopoDS import topods
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.BRepTools import breptools
from OCC.Core.BRepClass import BRepClass_FaceClassifier
from OCC.Core.GeomLProp import GeomLProp_SLProps


# =========================================================
# STEP loader
# =========================================================

def load_step(filename):
    reader = STEPControl_Reader()
    if reader.ReadFile(filename) != IFSelect_RetDone:
        raise RuntimeError(f"Failed to read STEP file: {filename}")
    reader.TransferRoots()
    return reader.Shape()


# =========================================================
# Dense sampling on a face
# =========================================================

def sample_face(face, nu=80, nv=80, tol=1e-7):
    """
    Returns np.ndarray of shape (N, 7):
    [x, y, z, nx, ny, nz, H]
    """
    umin, umax, vmin, vmax = breptools.UVBounds(face)

    surf_adaptor = BRepAdaptor_Surface(face, True)
    geom_surf = surf_adaptor.Surface().Surface()

    classifier = BRepClass_FaceClassifier()

    us = np.linspace(umin, umax, nu)
    vs = np.linspace(vmin, vmax, nv)

    data = []

    for u in us:
        for v in vs:
            p = surf_adaptor.Value(u, v)

            classifier.Perform(face, p, tol)
            if classifier.State() not in (TopAbs_IN, TopAbs_ON):
                continue

            props = GeomLProp_SLProps(geom_surf, u, v, 2, tol)
            if not props.IsNormalDefined():
                continue

            n = props.Normal()
            H = props.MeanCurvature()

            data.append([
                p.X(), p.Y(), p.Z(),
                n.X(), n.Y(), n.Z(),
                H
            ])

    return np.asarray(data, dtype=np.float64)


# =========================================================
# Dense sampling on an edge (geometry only)
# =========================================================

def sample_edge(edge, n=300):
    curve = BRepAdaptor_Curve(edge)
    umin = curve.FirstParameter()
    umax = curve.LastParameter()

    us = np.linspace(umin, umax, n)
    pts = []
    for u in us:
        p = curve.Value(u)
        pts.append([p.X(), p.Y(), p.Z()])

    return np.asarray(pts, dtype=np.float64)


# =========================================================
# Main extraction logic (NumPy level)
# =========================================================

def extract_airfoil_points_numpy(shape):
    """
    Returns NumPy arrays:
    - P_all_faces      (N, 7)
    - P_wing           (Γ interior)
    - P_section        (section faces)
    - P_wing_boundary  (∂Γ, geometry only)
    """

    face_exp = TopExp_Explorer(shape, TopAbs_FACE)

    P_all_faces = []
    P_wing = []
    P_section = []

    section_faces = []

    face_id = 1
    while face_exp.More():
        face = topods.Face(face_exp.Current())

        pts = sample_face(face, 40, 40)
        if pts.size == 0:
            face_id += 1
            face_exp.Next()
            continue

        if face_id not in (10, 11):
            pts = sample_face(face, 30, 400)

        P_all_faces.append(pts)

        if face_id in (10, 11):
            P_section.append(pts)
            section_faces.append(face)
        else:
            P_wing.append(pts)

        face_id += 1
        face_exp.Next()

    P_wing_boundary = []
    for face in section_faces:
        edge_exp = TopExp_Explorer(face, TopAbs_EDGE)
        while edge_exp.More():
            edge = topods.Edge(edge_exp.Current())
            pts = sample_edge(edge)
            if pts.size > 0:
                P_wing_boundary.append(pts)
            edge_exp.Next()

    return (
        np.vstack(P_all_faces),
        np.vstack(P_wing),
        np.vstack(P_section),
        np.vstack(P_wing_boundary),
    )


# =========================================================
# Torch wrapper (recommended entry)
# =========================================================

def extract_airfoil_points_torch(
        shape,
        device="cpu",
        dtype=torch.float64,
):
    """
    Returns torch.Tensor:
    - P_all_faces      (N, 7)
    - P_wing           (Nw, 7)
    - P_section        (Ns, 7)
    - P_wing_boundary  (Nb, 3)
    """

    P_all, P_wing, P_sec, P_bnd = extract_airfoil_points_numpy(shape)

    return (
        torch.as_tensor(P_all, device=device, dtype=dtype),
        torch.as_tensor(P_wing, device=device, dtype=dtype),
        torch.as_tensor(P_sec, device=device, dtype=dtype),
        torch.as_tensor(P_bnd, device=device, dtype=dtype),
    )


# =========================================================
# Visualization (optional)
# =========================================================

def plot_airfoil_points(P_wing, P_section, P_wing_boundary, subsample=5):
    Pw = P_wing[::subsample].cpu().numpy()
    Ps = P_section[::subsample].cpu().numpy()
    Pb = P_wing_boundary[::max(1, subsample // 2)].cpu().numpy()

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(Pw[:, 0], Pw[:, 1], Pw[:, 2], s=1, alpha=0.6, label="Wing")
    ax.scatter(Ps[:, 0], Ps[:, 1], Ps[:, 2], s=2, alpha=0.9, label="Section")
    ax.scatter(Pb[:, 0], Pb[:, 1], Pb[:, 2], s=6, c="k", label="Boundary")

    # equal aspect ratio
    max_range = np.array([Pw[:, 0].max() - Pw[:, 0].min(),
                          Pw[:, 1].max() - Pw[:, 1].min(),
                          Pw[:, 2].max() - Pw[:, 2].min()]).max() / 2.0
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
    torch.set_default_device('cuda') if torch.cuda.is_available() else torch.set_default_device('cpu')
    shape = load_step("Naca0012.STEP")

    P_all, P_wing, P_section, P_boundary = extract_airfoil_points_torch(
        shape,
        device="cpu",  # or "cuda"
        dtype=torch.float64,
    )

    x, normal, mean_curvature = P_all[:, :3], P_all[:, 3:6], P_all[:, 6:]
    torch.save((x, normal, mean_curvature), '../../data/airfoil_in.pth')

    x, normal, mean_curvature = torch.load('../../data/airfoil_in.pth', map_location=torch.tensor(0.).device)

    print(x.shape, normal.shape, mean_curvature.shape, x.dtype, x.device)

    # x, normal = torch.tensor(x), torch.tensor(normal)
    # mean_curvature = torch.zeros(x.shape[0], 1)  # 占位
    #
    # # save the points, normals, and mean curvature in a file
    # torch.save((x, normal, mean_curvature), '../../data/bottle_in.pth')
    #
    # # # load the points, normals, and mean curvature from the file

    # print("Torch tensors:")
    # print("  All faces     :", P_all.shape)
    # print("  Wing surface  :", P_wing.shape)
    # print("  Section faces :", P_section.shape)
    # print("  Boundary      :", P_boundary.shape)
    #
    # print("Mean curvature range (wing):",
    #       P_wing[:, 6].min().item(),
    #       P_wing[:, 6].max().item())
    #
    # plot_airfoil_points(P_wing, P_section, P_boundary)
