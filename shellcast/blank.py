"""A generic smoothed human head form to use as the core ("head blank") regardless of the sculpt.

Mask makers cast on a standard head form: the inside of the mask is a human head, the outside is whatever the
sculpt is, and the silicone in between is as thick as it needs to be. The form here is built from a few
ellipsoids and a neck cylinder in ANSUR-like proportions (head length ≈ 197, breadth ≈ 152, vertex–menton ≈ 232
at scale 1), then softened with an 8 mm blur of its distance field. It is scaled so its hat-line circumference
matches the fit target and placed with its hat line on the sculpt's hat line.

The core field is max(blank, sculpt + t): the blank, clipped wherever the sculpt is too tight to keep the mask
`t` thick. So the blank never breaks through, and where the creature is bigger than a head the mask is solid.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

import manifold3d as m3

from .fit import hat_line
from .geom import Grid, Manifold, manifold_from_arrays, tm_from_manifold


def generic_head(scale: float = 1.0) -> Manifold:
    """Head frame: +Z crown, face toward -Y, neck at -Z. Hat line at roughly z = +55·scale."""
    s = scale
    skull = Manifold.sphere(1.0, 96).scale((74 * s, 96 * s, 82 * s)).translate((0, 6 * s, 38 * s))
    face = Manifold.sphere(1.0, 64).scale((64 * s, 70 * s, 76 * s)).translate((0, -20 * s, -26 * s))
    jaw = Manifold.sphere(1.0, 64).scale((56 * s, 58 * s, 46 * s)).translate((0, -22 * s, -66 * s))
    back = Manifold.sphere(1.0, 64).scale((66 * s, 60 * s, 66 * s)).translate((0, 44 * s, 20 * s))
    neck = Manifold.cylinder(260 * s, 57 * s, 60 * s, 64).translate((0, 18 * s, -330 * s))
    head = m3.Manifold.batch_boolean([skull, face, jaw, back, neck], m3.OpType.Add)
    # soften: distance field, blur, iso — a smooth organic blank with no facets or creases
    t = tm_from_manifold(head)
    g = Grid.for_bounds(t.bounds[0], t.bounds[1], 2.0 * s, margin=20 * s)
    g.fill(t.vertices, t.faces)
    from scipy import ndimage
    field = ndimage.gaussian_filter(g.d, sigma=8.0 * s / g.pitch)
    return g.iso(0.0, field=field, smooth_iters=2, largest_only=True)


_CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "shellcast" / "generic_head_v1.npz"


def generic_head_cached() -> Manifold:
    """generic_head(1.0) never changes; building it costs an SDF fill and a blur (~20 s), so it is kept on disk.
    Bump the file name when generic_head changes."""
    try:
        z = np.load(_CACHE)
        return manifold_from_arrays(z["v"], z["f"])
    except (OSError, KeyError, ValueError):
        pass
    m = generic_head(1.0)
    try:
        _CACHE.parent.mkdir(parents=True, exist_ok=True)
        mesh = m.to_mesh()
        tmp = _CACHE.with_suffix(".tmp.npz")
        np.savez(tmp, v=np.asarray(mesh.vert_properties)[:, :3], f=np.asarray(mesh.tri_verts))
        tmp.replace(_CACHE)
    except OSError:
        pass
    return m


def place_blank(sculpt: Manifold, target_circumference: float, log=None) -> tuple[Manifold, dict]:
    """Scale the generic head to `target_circumference` at its hat line and align it with the sculpt's hat line."""
    base = generic_head_cached()
    c0, z0, _, _ = hat_line(base, z_lo_frac=0.55, z_hi_frac=0.9)
    scale = target_circumference / c0
    blank = base.scale((scale, scale, scale))
    cb, zb, _, _ = hat_line(blank, z_lo_frac=0.55, z_hi_frac=0.9)
    cs, zs, _, _ = hat_line(sculpt)
    def centre(m, z):
        cs_ = m.slice(float(z))
        polys = [np.asarray(p) for p in cs_.to_polygons()]
        p = max(polys, key=len)
        return (p.min(axis=0) + p.max(axis=0)) / 2
    cb_xy, cs_xy = centre(blank, zb), centre(sculpt, zs)
    shift = (float(cs_xy[0] - cb_xy[0]), float(cs_xy[1] - cb_xy[1]), float(zs - zb))
    blank = blank.translate(shift)
    info = {"blank_scale": scale, "blank_hat_circumference_mm": cb, "sculpt_hat_circumference_mm": cs,
            "hat_z": float(zs), "shift": list(shift)}
    if log:
        log(f"head blank: generic head ×{scale:.3f} (hat {cb:.0f} mm) aligned to the sculpt's hat line at z={zs:.0f}")
    return blank, info
