"""Remove any geometry inside the head: the sculpt becomes its outer skin only.

A mask sculpt can carry an inner mouth, teeth, a tongue, a hollow interior — all irrelevant, because the inside
of the mask is a human head. Every region of "outside" that cannot be reached from far away without passing a
slit narrower than 2·seal (a closed mouth's interior behind the lips, a hollow behind a thin skin) is made solid.
Open crevices on the surface (wrinkles, ear folds, nostrils) are small and stay as they are.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .geom import Grid, Manifold, tm_from_manifold


def remove_interior(m: Manifold, proxy, pitch=1.5, seal=6.0, min_volume_mm3=2000.0, log=None) -> tuple[Manifold, dict]:
    g = Grid.for_bounds(proxy.bounds[0], proxy.bounds[1], pitch, margin=seal + 4 * pitch, max_cells=25_000_000)
    g.fill(proxy.vertices, proxy.faces, log=log)
    # Closing over `seal` (slits sealed, chambers behind them filled). Only a voxel label is needed here, never a
    # mesh, so it is done on masks: the dilation is exact from the SDF, the erosion is an EDT. (Grid.morph_field
    # would cost a second full SDF evaluation — minutes on a big sculpt — for the same labels.)
    dil = g.d <= seal
    out_lab, _ = ndimage.label(~dil)
    rim = [out_lab[0], out_lab[-1], out_lab[:, 0], out_lab[:, -1], out_lab[:, :, 0], out_lab[:, :, -1]]
    outside = set(np.unique(np.concatenate([f.ravel() for f in rim]))) - {0}
    dil = ~np.isin(out_lab, list(outside))                              # enclosed voids of the dilation filled
    # + half a cell: the voxel erosion otherwise seals borderline crevices the exact closing leaves open (on a real
    # sculpt it found a 4 cm³ "pocket" the exact method did not); this way it errs toward leaving the sculpt alone
    closed_in = ndimage.distance_transform_edt(dil, sampling=g.pitch) > seal + 0.5 * g.pitch
    lab, n = ndimage.label(~closed_in)
    faces = [lab[0], lab[-1], lab[:, 0], lab[:, -1], lab[:, :, 0], lab[:, :, -1]]
    border = set(np.unique(np.concatenate([f.ravel() for f in faces]))) - {0}
    reachable = np.isin(lab, list(border))
    pocket = (g.d > 0) & ~reachable
    plab, pn = ndimage.label(pocket)
    info = {"count": 0, "volume_cm3": 0.0}
    if pn == 0:
        return m, info
    vols = ndimage.sum(pocket, plab, np.arange(1, pn + 1)) * g.pitch ** 3
    keep = [i + 1 for i, v in enumerate(vols) if v >= min_volume_mm3]
    if not keep:
        return m, info
    mask = ndimage.binary_dilation(np.isin(plab, keep), iterations=2)   # overlap the chamber walls so the union seals
    field = ndimage.gaussian_filter(g.signed_edt(mask), sigma=0.8)
    try:
        solid = g.iso(0.0, field=field, smooth_iters=2, largest_only=False, min_fraction=0.05)
    except ValueError as e:
        if log:
            log(f"(interior removal skipped: {e})")
        return m, info
    out = m + solid
    info = {"count": len(keep), "volume_cm3": float(sum(vols[i - 1] for i in keep) / 1000)}
    if log:
        log(f"removed interior geometry: {info['count']} enclosed region(s), {info['volume_cm3']:.0f} cm³ made solid")
    return out, info
