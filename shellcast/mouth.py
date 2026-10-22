"""Close the mouth of a sculpt: bring the lips together and remove the teeth, tongue and inner-mouth geometry.

How it works (head frame: +Z crown, face toward -Y):
  locate   a signed-distance grid over the lower front of the head; a morphological closing with `radius` fills
           every concavity narrower than 2·radius. The fill voxels (inside the closed solid, outside the sculpt)
           form blobs: eye sockets, nostrils, and the mouth — the largest blob in the mouth band. Per x column the
           blob gives the lip gap g(x), its mid-height m(x) and the mouth corners.
  classify sculpt vertices deeper than ~1.5 mm inside the closed solid are interior (teeth, tongue, gums, cheeks
           inside); faces made only of interior vertices are deleted.
  deform   exterior vertices within an influence distance of the fill blob move along Z toward m(x), with weight
           1 at the lip edge falling smoothly to 0 at the influence radius — so the lip edges meet on the mid-line
           and the lip surfaces stretch to follow, keeping their shape.
  reseal   the mouth box is re-wrapped from a fine SDF of the deformed mesh (the slit seals, the hole where the
           interior was closes) and swapped into the sculpt with booleans; the rest of the sculpt is untouched.
"""

from __future__ import annotations

import numpy as np
import trimesh
from scipy import ndimage

import manifold3d as m3

from .geom import Grid, Manifold, keep_components, manifold_from_tm, tm_from_manifold
from .ingest import proxy_of


def locate(t: trimesh.Trimesh, radius: float = 12.0, pitch: float = 1.0, band=(0.12, 0.58), log=None) -> dict:
    """Find the mouth fill blob. Returns a dict with the grid, masks and per-column lip statistics."""
    b0, b1 = t.bounds
    h = b1[2] - b0[2]
    zlo, zhi = b0[2] + band[0] * h, b0[2] + band[1] * h
    ymid = (b0[1] + b1[1]) / 2
    bmin = np.array([b0[0], b0[1], zlo]); bmax = np.array([b1[0], ymid, zhi])
    g = Grid.for_bounds(bmin, bmax, pitch, margin=radius + 3 * pitch, max_cells=30_000_000)
    if log:
        log(f"mouth grid {g.shape} = {g.n / 1e6:.1f}M cells at {g.pitch} mm")
    g.fill(t.vertices, t.faces, log=log)
    closed = g.morph_field(0.0, close_radius=radius, log=log, floor=0.0)
    fill = (closed <= 0) & (g.d > 0.6 * g.pitch)
    lab, n = ndimage.label(fill)
    if n == 0:
        raise ValueError("no closed concavity found — is the mouth open, and is the sculpt in the head frame?")
    # candidate blobs: volume, centroid; the mouth is the biggest blob whose centroid is in front of the head centre
    idx = np.arange(1, n + 1)
    vols = ndimage.sum(fill, lab, idx) * g.pitch ** 3
    cents = np.array(ndimage.center_of_mass(fill, lab, idx))
    cent_mm = g.origin + cents * g.pitch
    xc = (b0[0] + b1[0]) / 2
    score = vols.copy()
    score[np.abs(cent_mm[:, 0] - xc) > 0.25 * (b1[0] - b0[0])] = 0        # off-centre: eye sockets / ears
    k = int(np.argmax(score)) + 1
    mouth = lab == k
    # per-column statistics (x = mouth width direction)
    cols = np.where(mouth.any(axis=(1, 2)))[0]
    zi = np.arange(g.shape[2])
    stats = {}
    for ix in cols:
        m = mouth[ix]                       # (ny, nz)
        zs = zi[m.any(axis=0)]
        stats[int(ix)] = {"z_lo": g.origin[2] + zs.min() * g.pitch, "z_hi": g.origin[2] + zs.max() * g.pitch,
                          "z_mid": g.origin[2] + ((zs.min() + zs.max()) / 2) * g.pitch, "n": int(m.sum())}
    xs = g.origin[0] + cols * g.pitch
    gap = np.array([stats[int(i)]["z_hi"] - stats[int(i)]["z_lo"] for i in cols])
    zmid = np.array([stats[int(i)]["z_mid"] for i in cols])
    # distance from every voxel to the mouth blob (for influence weights)
    dist = ndimage.distance_transform_edt(~mouth, sampling=g.pitch).astype(np.float32)
    bb = np.argwhere(mouth)
    box = (g.origin + bb.min(axis=0) * g.pitch, g.origin + (bb.max(axis=0) + 1) * g.pitch)
    return {"grid": g, "closed": closed, "mouth": mouth, "dist": dist, "xs": xs, "gap": gap, "zmid": zmid,
            "volume_mm3": float(vols[k - 1]), "box": box, "n_blobs": int(n),
            "width_mm": float(xs.max() - xs.min() + g.pitch), "max_gap_mm": float(gap.max()), "mean_gap_mm": float(gap.mean())}


def _smooth(y, k):
    k = max(1, int(k))
    if k <= 1 or len(y) < 3:
        return y
    ker = np.ones(k) / k
    pad = np.pad(y, (k // 2, k - 1 - k // 2), mode="edge")
    return np.convolve(pad, ker, mode="valid")


def _snap_detail(patch: Manifold, ref: trimesh.Trimesh, log=None, tol=1.0) -> Manifold:
    """The re-wrap loses the fine skin texture. Wherever the wrapped surface lies within `tol` of the deformed
    original mesh (everywhere except the sealed lip zone) snap its vertices back onto that mesh, restoring the
    detail exactly. Falls back to the unsnapped patch if the snapped mesh is not manifold."""
    import igl
    pt = tm_from_manifold(patch)
    V0 = np.ascontiguousarray(ref.vertices, dtype=np.float64)
    F0 = np.ascontiguousarray(ref.faces, dtype=np.int64)
    P = np.ascontiguousarray(pt.vertices, dtype=np.float64)
    closest = np.empty_like(P)
    dist = np.empty(len(P))
    for a in range(0, len(P), 500_000):                       # libigl: AABB tree, chunked, small memory
        d, _, c = igl.signed_distance(P[a:a + 500_000], V0, F0, sign_type=igl.SIGNED_DISTANCE_TYPE_UNSIGNED)[:3]
        dist[a:a + 500_000] = np.abs(d)
        closest[a:a + 500_000] = c
    near = dist < tol
    # blend at the boundary of the snapped region so there is no step
    wgt = np.clip((tol - dist) / (0.5 * tol), 0, 1)
    V = pt.vertices + (closest - pt.vertices) * wgt[:, None]
    try:
        out = manifold_from_tm(trimesh.Trimesh(V, pt.faces, process=False))
        if log:
            log(f"detail restored on {int(near.sum())} of {len(V)} patch vertices")
        return out
    except ValueError:
        if log:
            log("(detail snap produced a non-manifold patch; keeping the smooth re-wrap)")
        return patch


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def deform(t: trimesh.Trimesh, radius=12.0, pitch=1.0, mode="auto", influence=None, log=None):
    """Move the lips/jaw of `t` (head frame) so the mouth closes. Returns (deformed mesh, info for reseal())."""
    # locating and closing only need the shape, not the detail: run them on a ~150k-triangle proxy (5–8× faster)
    loc = locate(proxy_of(t, 150_000), radius=radius, pitch=pitch, log=log)
    g, closed, dist = loc["grid"], loc["closed"], loc["dist"]
    xs, gap, zmid = loc["xs"], _smooth(loc["gap"], 9 / g.pitch), _smooth(loc["zmid"], 9 / g.pitch)
    zlo_col = _smooth(np.array([loc["zmid"][i] - loc["gap"][i] / 2 for i in range(len(xs))]), 9 / g.pitch)
    if mode == "auto":
        mode = "hinge" if loc["max_gap_mm"] > 25 else "pinch"
    R = influence if influence else float(np.clip(1.6 * loc["max_gap_mm"], 8.0, 40.0))
    if log:
        log(f"mouth: width {loc['width_mm']:.0f} mm, gap max {loc['max_gap_mm']:.1f} / mean {loc['mean_gap_mm']:.1f} mm → {mode} mode, {loc['n_blobs']} concavities")

    V = t.vertices.copy()
    ijk = np.round((V - g.origin) / g.pitch).astype(int)
    inside = np.all((ijk >= 0) & (ijk < np.asarray(g.shape)), axis=1)
    ii = np.clip(ijk, 0, np.asarray(g.shape) - 1)
    v_dist = np.where(inside, dist[ii[:, 0], ii[:, 1], ii[:, 2]], 1e9)
    x_lo, x_hi = xs.min() - 3.0, xs.max() + 3.0
    zt = np.interp(V[:, 0], xs, zmid)
    corner = _smoothstep(np.minimum(V[:, 0] - x_lo + 12, x_hi + 12 - V[:, 0]) / 12.0)   # fade past the corners
    # Nothing is deleted: teeth, tongue and inner lips are deformed along with their jaw and end up inside the
    # closed head, where the re-wrap (a winding-number SDF of the deformed, self-overlapping mesh) drops them.
    if mode == "pinch":
        lip = (V[:, 0] >= x_lo) & (V[:, 0] <= x_hi) & (v_dist < R)
        # full weight for anything touching the cavity (the lip edges and inner walls), so they really meet
        w = (1.0 - np.maximum(0.0, v_dist[lip] - 1.5 * g.pitch) / R) ** 2 * corner[lip]
        dz = zt[lip] - V[lip, 2]
        V[lip, 2] += (dz + np.sign(dz) * 1.5) * w                 # 1.5 mm past the mid-line so the lips overlap
        moved = int(lip.sum())
        hinge = None
    else:
        # jaw joint: at the mouth-line height, 62 % of the head depth back from the face; axis along x
        b0, b1 = t.bounds
        z_h = float(zmid.mean())
        y_h = float(b0[1] + 0.62 * (b1[1] - b0[1]))
        # front profile: the crease under the chin is where the front-most surface recedes the most below the mouth
        zi = np.arange(g.shape[2]); z_axis = g.origin[2] + zi * g.pitch
        solid = g.d <= 0
        yfront = np.full(g.shape[2], np.nan)
        for k in range(g.shape[2]):
            cols = np.where(solid[:, :, k].any(axis=0))[0]
            yfront[k] = g.origin[1] + cols.min() * g.pitch if len(cols) else np.nan
        # chin bottom: lowest z below the mouth where the front is still within 12 mm of its front-most value
        cand = np.where((z_axis < z_h - 10) & (z_axis > z_h - 130) & ~np.isnan(yfront))[0]
        ymin = np.nanmin(yfront[cand]) if len(cand) else np.nan
        near = cand[yfront[cand] <= ymin + 12.0] if len(cand) else cand
        z_chin = float(z_axis[near.min()]) if len(near) else z_h - 60
        r_front = np.hypot(np.nanmin(yfront) - y_h, zlo_col.mean() - z_h)
        theta = float(np.clip(np.arctan2(gap.mean() + 4.0, r_front), 0, np.radians(45)))   # +4 mm: the lips press together
        below = V[:, 2] < zt + 2.0                                   # the lower jaw: everything under the mouth line
        w = np.where(below, 1.0, 0.0)
        w *= _smoothstep((V[:, 2] - (z_chin - 18)) / 22.0)           # fade into the neck under the chin
        w *= _smoothstep((y_h - V[:, 1]) / 25.0)                     # fade toward the joint
        w *= corner
        c, sn = np.cos(-theta), np.sin(-theta)                     # negative angle about +x lifts the front (y < y_h)
        py, pz = V[:, 1] - y_h, V[:, 2] - z_h
        ry, rz = py * c - pz * sn, py * sn + pz * c
        V[:, 1] += (ry - py) * w
        V[:, 2] += (rz - pz) * w
        moved = int((w > 0.01).sum())
        hinge = {"y": y_h, "z": z_h, "angle_deg": float(np.degrees(theta)), "z_chin": z_chin, "r_front_mm": float(r_front)}
        if log:
            log(f"hinge at y={y_h:.0f} z={z_h:.0f}, chin crease z={z_chin:.0f}, rotate {np.degrees(theta):.1f}°, {moved} jaw vertices")
    t2 = trimesh.Trimesh(V, t.faces, process=False)
    if log:
        log(f"mouth: moved {moved} vertices")
    info = {"mode": mode, "loc": loc, "R": R, "hinge": hinge, "moved": moved}
    return t2, info


def reseal(S: Manifold, t: trimesh.Trimesh, t2: trimesh.Trimesh, info: dict, seal_pitch=0.6, log=None):
    """Re-wrap the deformed region from a winding-number SDF of the deformed (self-overlapping) mesh and swap it
    into the sculpt with booleans. Everything outside the box is the original geometry."""
    loc, R, hinge, mode = info["loc"], info["R"], info["hinge"], info["mode"]
    pad = R + 10.0
    bmin, bmax = loc["box"][0] - pad, loc["box"][1] + pad
    bmin[1] = min(bmin[1], t.bounds[0][1] - 1)
    if mode == "hinge":
        bmin[2] = min(bmin[2], hinge["z_chin"] - 40)
        bmax[1] = max(bmax[1], hinge["y"] + 30)
        bmax[2] = max(bmax[2], hinge["z"] + 45)
    sel = np.where(np.all((t2.triangles_center > bmin - 30) & (t2.triangles_center < bmax + 30), axis=1))[0]
    sub_full = t2.submesh([sel], append=True)
    sub = proxy_of(sub_full, 300_000) if len(sub_full.faces) > 300_000 else sub_full   # the SDF only needs ~0.5 mm
    gs = Grid.for_bounds(bmin, bmax, seal_pitch, margin=2 * seal_pitch, max_cells=30_000_000)
    if log:
        log(f"reseal grid {gs.shape} = {gs.n / 1e6:.1f}M cells at {gs.pitch} mm")
    gs.fill(sub.vertices, sub.faces)
    patch = gs.iso(0.0, smooth_iters=2, largest_only=True)
    patch = keep_components(patch, largest_only=True, fill_voids=True)   # the sealed mouth cavity becomes solid
    patch = _snap_detail(patch, sub_full, log)
    box = Manifold.cube(tuple(bmax - bmin)).translate(tuple(bmin))
    patch = patch ^ box
    rest = S - box
    S2 = keep_components(rest + patch, largest_only=True)
    return S2, [bmin.tolist(), bmax.tolist()]


def close_mouth(S: Manifold, t: trimesh.Trimesh, radius=12.0, pitch=1.0, mode="auto", influence=None, seal_pitch=0.6, log=None):
    """Return (new Manifold, new trimesh, report). `t` must be the trimesh of `S` in the head frame.

    mode "pinch": both lips move toward the mid-line (parted lips, gaps up to ~25 mm).
    mode "hinge": the lower jaw rotates up about the jaw joint until the lips meet (open jaws); the upper lip stays.
    "auto" picks hinge when the largest gap exceeds 25 mm."""
    t2, info = deform(t, radius=radius, pitch=pitch, mode=mode, influence=influence, log=log)
    S2, box = reseal(S, t, t2, info, seal_pitch=seal_pitch, log=log)
    loc = info["loc"]
    t3 = tm_from_manifold(S2)
    rep = {"mode": info["mode"], "width_mm": loc["width_mm"], "max_gap_mm": loc["max_gap_mm"], "mean_gap_mm": loc["mean_gap_mm"],
           "influence_mm": info["R"], "vertices_moved": info["moved"], "box": box, "hinge": info["hinge"],
           "volume_before_mm3": float(S.volume()), "volume_after_mm3": float(S2.volume())}
    return S2, t3, rep
