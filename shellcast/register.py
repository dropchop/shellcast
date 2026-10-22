"""Register a second version of the same sculpt (e.g. the closed-mouth edit) onto the working sculpt.

Similarity ICP (rotation, translation, uniform scale) on points sampled from the upper part of the head, where
both versions are the same geometry; the mouth and jaw, which differ, are left out of the fit and outliers are
dropped after the first iterations."""
from __future__ import annotations

import numpy as np
import trimesh
import igl


def _closest(P, V, F):
    d, _, c = igl.signed_distance(np.ascontiguousarray(P, dtype=np.float64), V, F, sign_type=igl.SIGNED_DISTANCE_TYPE_UNSIGNED)[:3]
    return np.asarray(d), np.asarray(c)


def register(src: trimesh.Trimesh, dst: trimesh.Trimesh, upper_fraction: float = 0.45, n_points: int = 8000, iterations: int = 40, log=None):
    """4x4 similarity transform taking `src` onto `dst`, fitted on surface points in the upper `upper_fraction` of
    each mesh's own height (the two may sit in different frames and scales). Returns (M, rms_mm, scale)."""
    rng = np.random.default_rng(7)

    def upper(m, n, seed):
        z0, z1 = m.bounds[0][2], m.bounds[1][2]
        P, _ = trimesh.sample.sample_surface(m, n * 3, seed=seed)
        P = P[P[:, 2] > z1 - upper_fraction * (z1 - z0)]
        if len(P) > n:
            P = P[rng.choice(len(P), n, replace=False)]
        return P

    pts, dpts = upper(src, n_points, 7), upper(dst, n_points, 11)
    if len(pts) < 100 or len(dpts) < 100:
        raise ValueError(f"too few upper-head points to register ({len(pts)} / {len(dpts)})")
    # initial guess: match centroid and rms radius of the two upper-head samples (both are already in the head frame)
    cs, cd = pts.mean(0), dpts.mean(0)
    s0 = np.sqrt(((dpts - cd) ** 2).sum(1).mean() / ((pts - cs) ** 2).sum(1).mean())
    M = np.eye(4)
    M[:3, :3] *= s0
    M[:3, 3] = cd - s0 * cs
    V, F = np.ascontiguousarray(dst.vertices, dtype=np.float64), np.ascontiguousarray(dst.faces, dtype=np.int64)
    P = trimesh.transform_points(pts, M)
    rms = None
    for it in range(iterations):
        d, C = _closest(P, V, F)
        keep = np.ones(len(P), bool)
        if it >= 3:
            keep = d < max(3.0 * np.median(d), 0.5)          # drop the mouth/jaw region and any other edits
        T, moved, cost = trimesh.registration.procrustes(P[keep], C[keep], reflection=False, translation=True, scale=True, return_cost=True)
        M = T @ M
        P = trimesh.transform_points(pts, M)
        new_rms = float(np.sqrt(np.mean(_closest(P, V, F)[0][keep] ** 2)))
        if rms is not None and abs(rms - new_rms) < 1e-4:
            rms = new_rms
            break
        rms = new_rms
    scale = float(np.cbrt(abs(np.linalg.det(M[:3, :3]))))
    if not (np.isfinite(rms) and np.isfinite(M).all() and 0.2 < scale < 5.0):
        raise ValueError(f"registration did not converge (rms {rms}, scale {scale})")
    if log:
        log(f"registered second sculpt: scale ×{scale:.4f}, rms {rms:.3f} mm on {int(keep.sum())} upper-head points")
    return M, rms, scale
