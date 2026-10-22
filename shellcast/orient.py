"""Put a sculpt into the head frame (+Z crown, neck at -Z, face toward -Y).

Sculpts arrive in whatever frame the slicer or DCC left them in — this one was lying on its back. Two cues,
each snapped to a principal axis because sculpts are almost always axis-aligned:

  up     the neck/base cut is the largest flat region on the mesh; its outward normal points down. Fallback:
         the longest axis, with the crown on the side that holds more volume.
  front  in the nose band (55–75 % of the height) the surface reaches farthest from the vertical axis toward
         the face. Among the four horizontal axis directions the one with the largest protrusion wins.

`orient` param: "auto", "keep" (already in the head frame), or an explicit "<up>,<front>" such as "+y,+z".
"""

from __future__ import annotations

import numpy as np

AXES = {"+x": np.array([1.0, 0, 0]), "-x": np.array([-1.0, 0, 0]), "+y": np.array([0, 1.0, 0]),
        "-y": np.array([0, -1.0, 0]), "+z": np.array([0, 0, 1.0]), "-z": np.array([0, 0, -1.0])}


def _name(v):
    i = int(np.argmax(np.abs(v)))
    return ("+" if v[i] > 0 else "-") + "xyz"[i]


def detect(t) -> dict:
    """Return {'up': '+y', 'front': '+z', 'planar_area_frac': …, 'protrusion': {…}} for a trimesh."""
    n = t.face_normals
    area = t.area_faces
    total = area.sum()
    # --- up: biggest cluster of faces sharing one axis-aligned normal (the flat cut) ---
    best_axis, best_frac = None, 0.0
    for name, ax in AXES.items():
        frac = float(area[(n @ ax) > 0.985].sum() / total)
        if frac > best_frac:
            best_axis, best_frac = name, frac
    if best_frac >= 0.02:
        down = best_axis
        up = ("-" if down[0] == "+" else "+") + down[1]
        up_reason = f"flat cut ({best_frac * 100:.1f} % of the surface faces {down})"
    else:
        i = int(np.argmax(t.extents))
        c = t.center_mass if t.is_volume else t.centroid
        mid = (t.bounds[0][i] + t.bounds[1][i]) / 2
        up = ("+" if c[i] > mid else "-") + "xyz"[i]
        up_reason = "no flat cut found; longest axis, crown on the heavier side"
    upv = AXES[up]
    # --- front: the face is where the detail is. Weight every edge by the dihedral angle between its two faces
    # (curvature) and see which way the curvature-weighted centre is displaced from the bounding-box centre in the
    # band between the chin and the brows. Works for uniform (dynamesh) and adaptive meshes alike; the back of a
    # head is smooth, the face is not.
    h = t.vertices @ upv
    lo, hi = h.min(), h.max()
    adj = t.face_adjacency
    ang = t.face_adjacency_angles
    mid = t.vertices[t.face_adjacency_edges].mean(axis=1)
    hm = mid @ upv
    band = (hm > lo + 0.35 * (hi - lo)) & (hm < lo + 0.80 * (hi - lo))
    w = ang[band]
    cen = (mid[band] * w[:, None]).sum(axis=0) / max(w.sum(), 1e-9)
    bb_mid = (mid[band].min(axis=0) + mid[band].max(axis=0)) / 2
    off = cen - bb_mid
    prot = {}
    for name, ax in AXES.items():
        if abs(float(ax @ upv)) > 0.5:
            continue
        prot[name] = float(off @ ax)
    front = max(prot, key=prot.get)
    return {"up": up, "front": front, "up_reason": up_reason, "detail_offset_mm": {k: round(v, 1) for k, v in prot.items()},
            "planar_area_frac": round(best_frac, 4)}


def rotation(up: str, front: str) -> np.ndarray:
    """3x3 matrix R with R @ v mapping mesh coordinates to the head frame (up→+Z, front→-Y)."""
    u, f = AXES[up], AXES[front]
    if abs(float(u @ f)) > 0.5:
        raise ValueError("up and front axes must be perpendicular")
    zt, yt = u, -f
    xt = np.cross(yt, zt)
    R = np.stack([xt, yt, zt])          # rows: where mesh x/y/z basis goes → R maps v to (v·xt, v·yt, v·zt)
    return R


def resolve(spec: str, t) -> tuple[np.ndarray, dict]:
    spec = (spec or "auto").strip().lower()
    if spec in ("keep", "none", ""):
        return np.eye(3), {"mode": "keep", "up": "+z", "front": "-y"}
    if spec == "auto":
        d = detect(t)
        return rotation(d["up"], d["front"]), {"mode": "auto", **d}
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 2 or parts[0] not in AXES or parts[1] not in AXES:
        raise ValueError("orient must be auto, keep, or '<up>,<front>' like '+y,+z'")
    return rotation(parts[0], parts[1]), {"mode": "manual", "up": parts[0], "front": parts[1]}
