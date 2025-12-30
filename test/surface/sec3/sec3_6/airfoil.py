import sys
from pathlib import Path

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE
from OCC.Core.TopoDS import topods
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepTools import breptools

from OCC.Core.Geom import (
    Geom_BSplineSurface,
    Geom_Plane,
    Geom_CylindricalSurface,
    Geom_SphericalSurface,
    Geom_ConicalSurface,
    Geom_ToroidalSurface,
)


# ------------------------------------------------------------
# STEP 读取
# ------------------------------------------------------------
def load_step(step_path: str):
    reader = STEPControl_Reader()
    status = reader.ReadFile(step_path)

    if status != IFSelect_RetDone:
        raise RuntimeError(f"Failed to read STEP file: {step_path}")

    reader.TransferRoots()
    return reader.OneShape()


# ------------------------------------------------------------
# 单个 Face 分析
# ------------------------------------------------------------
def analyze_face(face, tol=1e-12):
    surface = BRep_Tool.Surface(face)

    # ---------- 几何类型（OCCT 动态类型，最稳） ----------
    surface_type = surface.DynamicType().Name()
    is_nurbs = (surface_type == "Geom_BSplineSurface")

    # ---------- 是否有理 ----------
    if is_nurbs:
        bspline = Geom_BSplineSurface.DownCast(surface)
        is_rational = bspline.IsURational() or bspline.IsVRational()
    else:
        is_rational = False

    # ---------- Trim 判断 ----------
    u_min, u_max, v_min, v_max = breptools.UVBounds(face)
    su_min, su_max, sv_min, sv_max = surface.Bounds()

    is_trimmed = (
            abs(u_min - su_min) > tol or
            abs(u_max - su_max) > tol or
            abs(v_min - sv_min) > tol or
            abs(v_max - sv_max) > tol
    )

    return {
        "surface_type": surface_type,
        "is_nurbs": is_nurbs,
        "is_rational": is_rational,
        "is_trimmed": is_trimmed,
        "uv_bounds": (u_min, u_max, v_min, v_max),
        "surface_bounds": (su_min, su_max, sv_min, sv_max),
    }


# ------------------------------------------------------------
# STEP 文件分析
# ------------------------------------------------------------
def analyze_step_file(step_path: str):
    shape = load_step(step_path)

    explorer = TopExp_Explorer(shape, TopAbs_FACE)

    faces_info = []
    face_id = 0

    while explorer.More():
        face = topods.Face(explorer.Current())
        info = analyze_face(face)
        info["face_id"] = face_id
        faces_info.append(info)

        explorer.Next()
        face_id += 1

    return faces_info


# ------------------------------------------------------------
# 输出统计信息
# ------------------------------------------------------------
def print_summary(faces_info):
    total = len(faces_info)
    nurbs_faces = [f for f in faces_info if f["is_nurbs"]]
    trimmed_nurbs_faces = [
        f for f in faces_info if f["is_nurbs"] and f["is_trimmed"]
    ]

    print("=" * 80)
    print(f"Total faces             : {total}")
    print(f"NURBS faces             : {len(nurbs_faces)}")
    print(f"Trimmed NURBS faces     : {len(trimmed_nurbs_faces)}")
    print("=" * 80)

    for f in faces_info:
        print(
            f"Face {f['face_id']:3d} | "
            f"type={f['surface_type']:20s} | "
            f"NURBS={str(f['is_nurbs']):5s} | "
            f"rational={str(f['is_rational']):5s} | "
            f"trimmed={str(f['is_trimmed']):5s}"
        )


# ------------------------------------------------------------
# CLI 入口
# ------------------------------------------------------------
if __name__ == "__main__":
    # if len(sys.argv) != 2:
    #     print("Usage: python analyze_step_faces.py model.stp")
    #     sys.exit(1)

    # step_file = Path(sys.argv[1])
    # if not step_file.exists():
    #     raise FileNotFoundError(step_file)
    #
    faces_info = analyze_step_file(str("Naca0012.STEP"))
    print_summary(faces_info)
