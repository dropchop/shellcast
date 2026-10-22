"""Rescale the sculpt so the finished mask fits a real head.

What has to fit is the mask's *inner* surface, which is the core: the sculpt offset inward by the mask thickness t.
The standard fitting measurement is head circumference at the hat line (the widest horizontal loop above the ears
and brows). We measure the sculpt's largest horizontal convex-hull perimeter in the upper part of the head,
subtract 2πt for the wall, and choose one uniform scale factor so that inner circumference equals the target
(times 1 + ease). Uniform scaling keeps the sculpt's proportions; a stretchy Shore-2A mask absorbs the rest.

Targets are 50th-percentile adult values from ANSUR II (2012, US Army), rounded; they are starting points — a
custom circumference (tape measure over the brows and the bump at the back of the head) is better.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial import ConvexHull

TARGETS = {
    "male": {"circumference": 573.0, "length": 197.0, "breadth": 152.0, "label": "average adult male (ANSUR II 50th percentile)"},
    "female": {"circumference": 549.0, "length": 187.0, "breadth": 146.0, "label": "average adult female (ANSUR II 50th percentile)"},
}


def hat_line(m, z_lo_frac=0.72, z_hi_frac=0.92, steps=28):
    """(circumference_mm, z, length_mm, breadth_mm) of the largest horizontal convex-hull section in the upper head.

    The convex hull ignores concavities (eye sockets, the gap between ear and skull); the z band skips the crown
    and everything below the ear tops, which is where a hat band sits. Horns, hair and big ears still count, so
    the report shows the measurement for a sanity check."""
    bb = m.bounding_box()
    z0, z1 = bb[2], bb[5]
    best = (0.0, None, 0.0, 0.0)
    for z in np.linspace(z0 + (z1 - z0) * z_lo_frac, z0 + (z1 - z0) * z_hi_frac, steps):
        cs = m.slice(float(z))
        if cs.is_empty():
            continue
        polys = [np.asarray(p, dtype=np.float64) for p in cs.to_polygons()]
        pts = max(polys, key=len)
        if len(pts) < 3:
            continue
        hull = ConvexHull(pts)
        hp = pts[hull.vertices]
        per = float(np.sum(np.linalg.norm(np.roll(hp, -1, axis=0) - hp, axis=1)))
        if per > best[0]:
            best = (per, float(z), float(np.ptp(hp[:, 1])), float(np.ptp(hp[:, 0])))
    return best


def plan(m, mask_thickness: float, target: str, custom_circumference: float, ease_pct: float, manual_scale: float) -> dict:
    """Decide the scale factor. Returns a report dict with 'scale' (1.0 when nothing is requested)."""
    circ, z, length, breadth = hat_line(m)
    inner = circ - 2 * math.pi * mask_thickness
    rep = {"target": target, "sculpt_hat_circumference_mm": circ, "hat_line_z": z, "sculpt_hat_length_mm": length,
           "sculpt_hat_breadth_mm": breadth, "mask_inner_circumference_mm": inner, "manual_scale": manual_scale, "ease_pct": ease_pct}
    goal = None
    if target in TARGETS:
        goal = TARGETS[target]["circumference"]
        rep["target_label"] = TARGETS[target]["label"]
    elif target == "custom" and custom_circumference > 0:
        goal = float(custom_circumference)
        rep["target_label"] = f"custom head circumference {goal:.0f} mm"
    scale = float(manual_scale) if manual_scale > 0 else 1.0
    if goal and circ > 0:
        want_inner = goal * (1 + ease_pct / 100.0)
        scale *= (want_inner + 2 * math.pi * mask_thickness) / circ
        rep.update(target_circumference_mm=goal, wanted_inner_circumference_mm=want_inner)
    rep["scale"] = scale
    rep["mask_inner_circumference_after_mm"] = circ * scale - 2 * math.pi * mask_thickness
    rep["sculpt_hat_circumference_after_mm"] = circ * scale
    return rep
