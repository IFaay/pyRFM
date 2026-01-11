# -*- coding: utf-8 -*-
"""
Dense sampling of NACA airfoil STEP surfaces with:
- strict face-inside test
- exact surface normal
- exact mean curvature
- final torch.tensor conversion

Each surface sample:
[x, y, z, nx, ny, nz, H]

We organize the point sets as:
- x               : all points on the STEP surface (Γ')       (geometry + normal + H stored in airfoil_in.pth)
- x_trim_in       : points on the target trimmed surface Γ    (wing faces, interior of Γ)
- x_trim_out      : points on Γ' but not in Γ                 (here: section faces)
- x_trim_boundary : boundary points ∂Γ (geometry only, from section edges)
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
    TopAbs_REVERSED
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
    Dense sampling on a single face.

    Returns
    -------
    np.ndarray of shape (N, 7):
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

            # strict inside test on the face
            classifier.Perform(face, p, tol)
            if classifier.State() not in (TopAbs_IN, TopAbs_ON):
                continue

            props = GeomLProp_SLProps(geom_surf, u, v, 2, tol)
            if not props.IsNormalDefined():
                continue

            n = props.Normal()
            if face.Orientation() == TopAbs_REVERSED:
                n.Reverse()

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
    """
    Dense sampling on a single edge (geometry only).

    Returns
    -------
    np.ndarray of shape (N, 3):
        [x, y, z]
    """
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
    Extract dense samples from the STEP shape.

    Returns
    -------
    P_all_faces : (N, 7)
        All sampled points on all faces (Γ'), with [x, y, z, nx, ny, nz, H].
    P_wing      : (Nw, 7)
        Points on wing faces (interpreted as trimmed surface Γ).
    P_section   : (Ns, 7)
        Points on section faces (here interpreted as Γ' \ Γ).
    P_bnd       : (Nb, 3)
        Boundary points ∂Γ sampled on edges of section faces (geometry only).
    """

    face_exp = TopExp_Explorer(shape, TopAbs_FACE)

    P_all_faces = []
    P_wing = []
    P_section = []

    section_faces = []

    face_id = 1
    while face_exp.More():
        face = topods.Face(face_exp.Current())

        if face_id not in (10, 11):
            pts = sample_face(face, 30, 400)
        else:
            pts = sample_face(face, 100, 100)

        if pts.size == 0:
            face_id += 1
            face_exp.Next()
            continue

        P_all_faces.append(pts)

        # Face 10, 11: section faces
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
    Torch wrapper for airfoil point extraction.

    Returns
    -------
    P_all      : torch.Tensor, (N, 7)
    P_wing     : torch.Tensor, (Nw, 7)
    P_section  : torch.Tensor, (Ns, 7)
    P_bnd      : torch.Tensor, (Nb, 3)
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

    ax.scatter(Pw[:, 0], Pw[:, 1], Pw[:, 2], s=1, alpha=0.6, label="Wing (Γ)")
    ax.scatter(Ps[:, 0], Ps[:, 1], Ps[:, 2], s=2, alpha=0.9, label="Section (Γ'\\Γ)")
    ax.scatter(Pb[:, 0], Pb[:, 1], Pb[:, 2], s=6, c="k", label="Boundary (∂Γ)")

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


def plot_airfoil_all(Pw, subsample=5):
    Pw = Pw[::subsample].cpu().numpy()

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(Pw[:, 0], Pw[:, 1], Pw[:, 2], s=1, alpha=0.6, label="ALL")
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
    # Default device setup
    torch.set_default_device('cuda') if torch.cuda.is_available() else torch.set_default_device('cpu')

    shape = load_step("Naca0012.STEP")

    P_all, P_wing, P_section, P_boundary = extract_airfoil_points_torch(
        shape,
        device="cpu",  # or "cuda"
        dtype=torch.float64,
    )

    # ------------------------------------------------------------------
    # 1) Untrimmed surface Γ': all faces, kept as before
    # ------------------------------------------------------------------
    x = P_all[:, :3]  # (N, 3)
    normal = P_all[:, 3:6]  # (N, 3)
    mean_curvature = P_all[:, 6:]  # (N, 1)

    # This file keeps the original convention: all points on Γ'
    torch.save((x, normal, mean_curvature), '../../data/airfoil_in.pth')

    x, normal, mean_curvature = torch.load(
        '../../data/airfoil_in.pth',
        map_location=torch.tensor(0.).device
    )

    print(x.shape, normal.shape, mean_curvature.shape, x.dtype, x.device)

    # ------------------------------------------------------------------
    # 2) Trimmed sets for Γ, Γ'\Γ, and ∂Γ (geometry only)
    #    These use the naming:
    #       x_trim_in       : points on Γ        (wing faces)
    #       x_trim_out      : points on Γ'\Γ    (section faces)
    #       x_trim_boundary : points on ∂Γ      (boundary edges)
    # ------------------------------------------------------------------
    x_trim_in = P_wing[:, :3]  # (N_in, 3)
    x_trim_out = P_section[:, :3]  # (N_out, 3)
    x_trim_boundary = P_boundary  # (N_bnd, 3), already geometry only

    # 如需要，可以单独保存 trim 相关集合：
    torch.save(
        (x_trim_in, x_trim_out, x_trim_boundary),
        '../../data/airfoil_trim_sets.pth'
    )

    x_trim_in, x_trim_out, x_trim_boundary = torch.load(
        '../../data/airfoil_trim_sets.pth',
        map_location=torch.tensor(0.).device
    )

    print(x_trim_in.shape, x_trim_out.shape, x_trim_boundary.shape)

    # # Optional: check / visualization
    plot_airfoil_points(P_wing, P_section, P_boundary)
    plot_airfoil_all(P_all)
