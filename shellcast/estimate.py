"""Print-time and filament estimates: an instant geometric model.

Geometric model (per part, in its print orientation):
    grams      = V_part * density                       (thin shells are nearly all perimeters)
    t_print    = V_part / Q_eff + N_layers * t_layer + V_support / (Q_eff * support_eff)
    V_support  = Σ over faces overhanging beyond 50° of (area * drop height to the bed)   (the Oh et al. feature)
Orientation is chosen by score = t_print + λ_cavity * A(cavity faces needing support) + bed-fit penalty.
Expect ±15–25 % uncalibrated; the slicer's own estimate on the exported OBJ is the ground truth.
"""

from __future__ import annotations

import math

import numpy as np
import trimesh

from .params import PrinterProfile

OVERHANG_DEG = 50                     # slicers default to 45–55°; the 45° cathedral roof must not count
OVERHANG_COS = math.cos(math.radians(OVERHANG_DEG))
LAMBDA_CAVITY_S_PER_MM2 = 30.0        # a support scar on the cavity face is very expensive (it prints onto the mask)


def _rot(axis_from: np.ndarray, axis_to: np.ndarray) -> np.ndarray:
    a, b = axis_from / np.linalg.norm(axis_from), axis_to / np.linalg.norm(axis_to)
    v, c = np.cross(a, b), float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        if c > 0:
            return np.eye(4)
        # 180°: rotate about any perpendicular axis
        perp = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
        perp = np.cross(a, perp)
        return trimesh.transformations.rotation_matrix(math.pi, perp)
    ang = math.atan2(np.linalg.norm(v), c)
    return trimesh.transformations.rotation_matrix(ang, v)


def y_axis():
    return np.array([0, 1.0, 0])


def candidates(kind: str, pull_normal: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Named 4x4 transforms to try for a part kind."""
    n = np.asarray(pull_normal, dtype=np.float64)
    z = np.array([0, 0, 1.0])
    if kind == "shell_A":      # A lies on the +n side; its flange face (at the plane) faces -n
        flange_dir = -n
    elif kind == "shell_B":
        flange_dir = n
    elif kind in ("mouth_insert", "mouth_cover", "mouth_wedge"):   # a plug: its outer (mold-outside) face is toward the front, -Y; print it face down
        return [("face-down", _rot(-y_axis(), -z)), ("face-up", _rot(-y_axis(), z))]
    else:                      # core / neck plug: collar plate is the lowest face already
        return [("neck-down", np.eye(4)), ("neck-up", _rot(z, -z))]
    out = [("flange-up", _rot(flange_dir, z)), ("flange-down", _rot(flange_dir, -z))]
    for tilt in (20, -20, 35, -35):
        for axis, tag in (([1, 0, 0], "x"), ([0, 1, 0], "y")):
            r = trimesh.transformations.rotation_matrix(math.radians(tilt), axis) @ _rot(flange_dir, z)
            out.append((f"flange-up tilt {tilt:+d}{tag}", r))
    return out


def support_volume(t: trimesh.Trimesh, cavity: np.ndarray | None):
    """(support volume proxy mm3, cavity area needing support mm2, overhang area mm2)."""
    nz = t.face_normals[:, 2]
    over = nz < -OVERHANG_COS
    drop = t.triangles_center[:, 2] - t.bounds[0][2]
    area = t.area_faces
    vol = float((area[over] * drop[over]).sum())
    cav = float(area[over & cavity].sum()) if cavity is not None else 0.0
    return vol, cav, float(area[over].sum())


def printed_volume(t: trimesh.Trimesh, pr: PrinterProfile) -> tuple[float, float]:
    """(mm3 actually extruded, interior mm3). Thin shells are all perimeters, but a thick region (a flange, a
    solid block inside a puppet mouth) prints as walls plus sparse infill. The interior beyond the walls is found
    by eroding the part by the wall thickness on a coarse signed-distance grid."""
    from .geom import Grid
    vol = abs(float(t.volume))
    wall = pr.wall_lines * pr.line_width + 2 * pr.layer_height * 3    # side walls, plus top/bottom skins averaged in
    try:
        g = Grid.for_bounds(t.bounds[0], t.bounds[1], 2.5, margin=3.0, max_cells=4_000_000)
        d = g.eval(t.vertices, t.faces)
        interior = float((d <= -wall).sum()) * g.pitch ** 3
    except Exception:  # noqa: BLE001
        interior = 0.0
    interior = min(interior, vol)
    return vol - interior * (1.0 - pr.infill_pct / 100.0), interior


def estimate_part(t: trimesh.Trimesh, pr: PrinterProfile, cavity: np.ndarray | None = None, printed=None) -> dict:
    vol, interior = printed if printed is not None else printed_volume(t, pr)
    height = float(t.extents[2])
    layers = max(1, int(math.ceil(height / pr.layer_height)))
    sup_vol, cav_sup, over_area = support_volume(t, cavity)
    # sparse support region printed at support_density; time at the lower flow efficiency
    sup_printed = sup_vol * pr.support_density
    t_print = vol / pr.flow_mm3_s + layers * pr.layer_overhead_s + sup_printed / (pr.flow_mm3_s * pr.support_efficiency)
    grams = (vol + sup_printed) * pr.density_g_cm3 / 1000.0
    fil_len_m = (vol + sup_printed) / (math.pi * (1.75 / 2) ** 2) / 1000.0
    fits = bool(t.extents[0] <= pr.bed_x and t.extents[1] <= pr.bed_y and t.extents[2] <= pr.bed_z)
    return {"volume_mm3": abs(float(t.volume)), "printed_mm3": vol, "interior_mm3": interior, "height_mm": height, "layers": layers, "filament_g": grams, "filament_m": fil_len_m,
            "support_volume_mm3": sup_vol, "support_g": sup_printed * pr.density_g_cm3 / 1000.0,
            "overhang_area_mm2": over_area, "cavity_support_area_mm2": cav_sup, "time_s_geom": t_print, "fits_bed": fits,
            "footprint_mm": [float(t.extents[0]), float(t.extents[1])]}


def best_orientation(t: trimesh.Trimesh, kind: str, pull_normal, pr: PrinterProfile, cavity: np.ndarray | None):
    best = None
    tried = []
    printed = printed_volume(t, pr)       # rotation-invariant: one SDF per part, not one per candidate
    for name, M in candidates(kind, pull_normal):
        u = t.copy()
        u.apply_transform(M)
        e = estimate_part(u, pr, cavity, printed)
        score = e["time_s_geom"] + LAMBDA_CAVITY_S_PER_MM2 * e["cavity_support_area_mm2"] + (0 if e["fits_bed"] else 3600 * 24)
        tried.append({"orientation": name, "score_s": score, "time_s_geom": e["time_s_geom"], "cavity_support_area_mm2": e["cavity_support_area_mm2"], "fits_bed": e["fits_bed"]})
        if best is None or score < best[0]:
            best = (score, name, M, u, e)
    score, name, M, u, e = best
    # sit on the bed, centred on XY
    shift = np.array([-u.bounds[0][0] - u.extents[0] / 2, -u.bounds[0][1] - u.extents[1] / 2, -u.bounds[0][2]])
    u.apply_translation(shift)
    M = trimesh.transformations.translation_matrix(shift) @ M
    e.update({"orientation": name, "transform": M.tolist(), "orientations_tried": tried})
    return u, e


def fmt_hms(s: float) -> str:
    s = int(round(s))
    return f"{s // 3600}h {(s % 3600) // 60:02d}m"


