"""Geometry kernel helpers: trimesh <-> manifold3d, signed-distance grids, iso-surfaces, primitives.

Sign convention everywhere: d < 0 inside the sculpt, d > 0 outside (libigl's).
"""

from __future__ import annotations

import math
import time

import igl
import manifold3d as m3
import numpy as np
import trimesh
from scipy import ndimage
from skimage import measure

Manifold = m3.Manifold


# ---------------------------------------------------------------------------
# trimesh <-> manifold
# ---------------------------------------------------------------------------

def tm_from_manifold(m: Manifold) -> trimesh.Trimesh:
    mesh = m.to_mesh()
    v = np.asarray(mesh.vert_properties, dtype=np.float64)[:, :3]
    f = np.asarray(mesh.tri_verts, dtype=np.int64)
    return trimesh.Trimesh(v, f, process=False)


def manifold_from_arrays(v: np.ndarray, f: np.ndarray) -> Manifold:
    """Try to build a Manifold; raises ValueError with the manifold3d status if the mesh isn't boolean-ready."""
    mesh = m3.Mesh(vert_properties=np.ascontiguousarray(v, dtype=np.float32), tri_verts=np.ascontiguousarray(f, dtype=np.uint32))
    m = Manifold(mesh)
    if m.status() == m3.Error.NoError and not m.is_empty():
        return m
    mesh.merge()
    m = Manifold(mesh)
    if m.status() == m3.Error.NoError and not m.is_empty():
        return m
    raise ValueError(f"not manifold: {m.status()}")


def manifold_from_tm(t: trimesh.Trimesh) -> Manifold:
    return manifold_from_arrays(t.vertices, t.faces)


def keep_components(m: Manifold, min_fraction: float = 0.01, largest_only: bool = False, fill_voids: bool = False) -> Manifold:
    """Drop stray bodies. manifold3d's decompose() splits by *surface* connectivity, so a hollow solid comes
    back as an outer body (positive volume) plus its cavity surface (negative volume). Cavities are kept with
    the body whose bounding box contains them; only small positive bodies (and tiny void bubbles) are dropped."""
    parts = m.decompose()
    if len(parts) <= 1:
        return m
    solids = [(p, p.volume()) for p in parts if p.volume() > 0]
    if fill_voids and largest_only and solids:
        return max(solids, key=lambda pv: pv[1])[0]          # the biggest solid, with any enclosed cavities filled
    if len(solids) <= 1:
        return m
    vmax = max(v for _, v in solids)
    drop = [p for p, v in solids if (v < vmax if largest_only else v < min_fraction * vmax)]
    if not drop:
        return m
    # Subtracting the stray bodies (they are disjoint from the main one) leaves the main body's cavities intact.
    return m - Manifold.batch_boolean(drop, m3.OpType.Add)


def strip_slivers(m: Manifold, min_volume=1e-3) -> Manifold:
    """Remove degenerate zero-volume surface fragments left behind by coincident-face booleans."""
    t = tm_from_manifold(m)
    comps = t.split(only_watertight=False)
    if len(comps) <= 1:
        return m
    keep = [c for c in comps if abs(c.volume) > min_volume]
    if len(keep) == len(comps) or not keep:
        return m
    return manifold_from_tm(trimesh.util.concatenate(keep))


# ---------------------------------------------------------------------------
# Signed-distance grid
# ---------------------------------------------------------------------------

class Grid:
    """A regular grid of signed distances, indexed d[ix, iy, iz]."""

    def __init__(self, origin, pitch, shape):
        self.origin = np.asarray(origin, dtype=np.float64)
        self.pitch = float(pitch)
        self.shape = tuple(int(s) for s in shape)
        self.d = None

    @classmethod
    def for_bounds(cls, bmin, bmax, pitch, margin, max_cells=None):
        """Grid covering the box plus margin. With max_cells the pitch is coarsened (in 0.25 mm steps) until the
        cell count fits — large sculpts then trade a little surface fidelity for bounded memory."""
        # the origin is offset by an irrational fraction of the pitch so flat faces of CAD-like inputs
        # (fixtures, boxes) never land exactly on grid points, which would put marching cubes on a knife edge
        bmin = np.asarray(bmin, dtype=np.float64) - margin
        bmax = np.asarray(bmax, dtype=np.float64) + margin
        pitch = float(pitch)
        while max_cells and np.prod(np.ceil((bmax - bmin) / pitch) + 2) > max_cells:
            pitch = round(pitch + 0.25, 2)
        origin = bmin - 0.3819660 * pitch
        shape = np.ceil((bmax - origin) / pitch).astype(int) + 2
        return cls(origin, pitch, shape)

    def points(self):
        ax = [self.origin[i] + self.pitch * np.arange(self.shape[i]) for i in range(3)]
        X, Y, Z = np.meshgrid(*ax, indexing="ij")
        return np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    @property
    def n(self):
        return int(np.prod(self.shape))

    def fill(self, v, f, chunk=2_000_000, log=None):
        """Evaluate the signed distance of every grid point against mesh (v, f) using fast winding numbers."""
        self.d = self.eval(v, f, chunk=chunk, log=log)
        return self

    def eval(self, v, f, chunk=2_000_000, log=None, only=None, known=None):
        """Signed distance of every grid point to mesh (v, f), as an array shaped like the grid (not stored).
        Points are generated one z-slab at a time so the full point cloud is never materialised. With a boolean
        mask `only`, just those points are evaluated and every other cell is taken from `known`."""
        v = np.ascontiguousarray(v, dtype=np.float64)
        f = np.ascontiguousarray(f, dtype=np.int64)
        nx, ny, nz = self.shape
        out = np.empty(self.shape, dtype=np.float32) if only is None else known.astype(np.float32, copy=True)
        xs = self.origin[0] + self.pitch * np.arange(nx)
        ys = self.origin[1] + self.pitch * np.arange(ny)
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        slab_xy = np.stack([X.ravel(), Y.ravel()], axis=1)
        per_slab = nx * ny
        zs_per_chunk = max(1, chunk // per_slab)
        t0 = time.time()
        for z0 in range(0, nz, zs_per_chunk):
            z1 = min(nz, z0 + zs_per_chunk)
            zs = self.origin[2] + self.pitch * np.arange(z0, z1)
            pts = np.empty(((z1 - z0) * per_slab, 3))
            pts[:, :2] = np.tile(slab_xy, ((z1 - z0), 1))
            pts[:, 2] = np.repeat(zs, per_slab)
            if only is None:
                d = igl.signed_distance(pts, v, f, sign_type=igl.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER)[0]
                out[:, :, z0:z1] = d.reshape((z1 - z0), nx, ny).transpose(1, 2, 0)
            else:
                sel = only[:, :, z0:z1].transpose(2, 0, 1).ravel()
                if sel.any():
                    d = igl.signed_distance(pts[sel], v, f, sign_type=igl.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER)[0]
                    blk = out[:, :, z0:z1].transpose(2, 0, 1).reshape(-1)
                    blk[sel] = d
                    out[:, :, z0:z1] = blk.reshape((z1 - z0), nx, ny).transpose(1, 2, 0)
            if log:
                log(f"sdf {z1}/{nz} slabs ({time.time() - t0:.0f}s)")
        return out

    def sample(self, xyz, field=None):
        """Nearest-cell value of the field at a point (or +inf outside the grid)."""
        field = self.d if field is None else field
        i = np.round((np.asarray(xyz, dtype=np.float64) - self.origin) / self.pitch).astype(int)
        if (i < 0).any() or (i >= np.asarray(self.shape)).any():
            return float("inf")
        return float(field[i[0], i[1], i[2]])

    def iso(self, level, field=None, smooth_iters=0, largest_only=False, min_fraction=0.01) -> Manifold:
        """Marching-cubes iso-surface of `field` (default: self.d) at `level`, as a Manifold with outward normals.

        Voxel-quantised fields (from EDT) can give marching cubes ambiguous saddle cells whose mesh is not
        manifold; on failure the field is re-extracted after a light Gaussian blur (0.6 then 1.2 cells)."""
        field = self.d if field is None else field
        if not (field.min() < level < field.max()):
            raise ValueError(f"iso level {level} is outside the field range [{field.min():.1f}, {field.max():.1f}]")
        last = None
        eps = 1e-4 * self.pitch
        for sigma in (0.0, 0.6, 1.2):
            f = field if sigma == 0 else ndimage.gaussian_filter(field, sigma=sigma)
            f = np.where(np.abs(f - level) < eps, level + eps, f)      # never sample exactly on the level
            # one layer of "far outside" around the grid so a surface that runs off the grid is capped, not left open
            f = np.pad(f, 1, constant_values=level + 10 * self.pitch)
            try:
                verts, faces, _, _ = measure.marching_cubes(f, level=level, spacing=(self.pitch,) * 3, allow_degenerate=False)
                t = trimesh.Trimesh(verts + self.origin - self.pitch, faces, process=True)
                if smooth_iters > 0:
                    trimesh.smoothing.filter_laplacian(t, lamb=0.5, iterations=smooth_iters, volume_constraint=True)
                if t.volume < 0:
                    t.invert()
                m = manifold_from_tm(t)
                return keep_components(m, min_fraction=min_fraction, largest_only=largest_only)
            except ValueError as e:
                last = e
        raise ValueError(f"iso-surface is not manifold even after smoothing: {last}")

    def signed_edt(self, inside_mask):
        """Signed distance (mm) of a binary set: negative inside, positive outside."""
        inside = inside_mask.astype(bool)
        dist_out = ndimage.distance_transform_edt(~inside, sampling=self.pitch)   # for outside points: distance to set
        dist_in = ndimage.distance_transform_edt(inside, sampling=self.pitch)     # for inside points: distance to complement
        return (dist_out - dist_in).astype(np.float32)

    def morph_field(self, level, close_radius=0.0, open_radius=0.0, log=None, base=None, floor=None):
        """Signed field of {d <= level} after a morphological *closing* (fills slots, folds, nostrils and ear canals
        narrower than 2·close_radius) followed by an *opening* (removes fins and protrusions thinner than
        2·open_radius, e.g. thin ears, which then become solid silicone). Both radii 0 → the exact field.

        Done exactly rather than on voxel masks: dilate = offset iso-surface of the true SDF, then a fresh signed
        distance to that surface gives a true SDF of the dilated set, whose -r level is the erosion. The result is
        smooth, so marching cubes meshes it cleanly.

        Closing only needs the slow SDF where the answer is not already known: outside the dilation the closed
        field equals the input exactly (distance to the dilated set + r), and with `floor` set (the lowest level
        the caller will ever extract from the result) cells more than two cells below it keep the input value,
        which is already on the correct side (closing only adds material)."""
        f = (self.d if base is None else base) - level     # <= 0 inside {d <= level}, a true signed distance
        if close_radius > 0:
            # the dilation seals any slit; a chamber behind it becomes an enclosed void of the dilated solid, which a
            # distance field would still treat as "outside" — fill such voids so closing really fills pockets
            m = keep_components(self.iso(close_radius, field=f), largest_only=True, fill_voids=True)
            t = tm_from_manifold(m)
            # cells outside the dilation and reachable from the grid border (not in a void the fill just closed)
            out_lab, _ = ndimage.label(f > close_radius + self.pitch)
            rim = [out_lab[0], out_lab[-1], out_lab[:, 0], out_lab[:, -1], out_lab[:, :, 0], out_lab[:, :, -1]]
            known = np.isin(out_lab, list(set(np.unique(np.concatenate([q.ravel() for q in rim]))) - {0}))
            if floor is not None and open_radius <= 0:
                known |= f < floor - 2 * self.pitch
            if log:
                log(f"closing: exact SDF on {100 * (1 - known.mean()):.0f} % of the grid")
            f = self.eval(t.vertices, t.faces, log=log, only=~known, known=f - close_radius) + close_radius
        if open_radius > 0:
            t = tm_from_manifold(self.iso(-open_radius, field=f))
            f = self.eval(t.vertices, t.faces, log=log) - open_radius
        return f


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

def _rotation_to(z_to: np.ndarray) -> np.ndarray:
    """3x3 rotation taking +Z to unit vector z_to."""
    z = np.asarray(z_to, dtype=np.float64)
    z = z / np.linalg.norm(z)
    a = np.array([0.0, 0.0, 1.0])
    v = np.cross(a, z)
    c = float(np.dot(a, z))
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 / (1 + c))


def transform_matrix(rot: np.ndarray, t) -> np.ndarray:
    """manifold3d wants a 3x4 (rows = xyz, cols = basis + translation)."""
    m = np.zeros((3, 4))
    m[:, :3] = rot
    m[:, 3] = t
    return m


def cylinder_along(base, direction, radius_base, radius_top, height, segments=32) -> Manifold:
    """Cylinder/cone with its base disc centred at `base`, extending `height` along `direction`."""
    c = Manifold.cylinder(height, radius_base, radius_top, segments)
    return c.transform(transform_matrix(_rotation_to(direction), base))


def slab(center, normal, thickness, size=5000.0) -> Manifold:
    """A big square slab of `thickness` centred on the plane (center, normal)."""
    b = Manifold.cube((size, size, thickness), center=True)
    return b.transform(transform_matrix(_rotation_to(normal), center))


def halfspace_keep(m: Manifold, normal, point) -> Manifold:
    """Keep the part of m on the side the normal points to."""
    n = np.asarray(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    return m.trim_by_plane(tuple(n), float(np.dot(n, point)))


def cross_section_polys(cs: m3.CrossSection):
    return [np.asarray(p, dtype=np.float64) for p in cs.to_polygons()]


def resample_polyline(pts: np.ndarray, spacing: float, closed: bool, phase: float = 0.5) -> np.ndarray:
    """Points every `spacing` mm along a polyline (arc length) at (i + phase) * step; phase 0.5 = centred samples,
    phase 1.0 = the points half-way between those (keys between bolts, measured along the curve, not the chord)."""
    pts = np.asarray(pts, dtype=np.float64)
    if closed:
        pts = np.vstack([pts, pts[:1]])
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = seg.sum()
    if total < spacing * 0.6:
        return pts[:1]
    n = max(1, int(round(total / spacing)))
    targets = (np.arange(n) + phase) * (total / n)
    targets = targets[targets < total - 1e-9]
    cum = np.concatenate([[0], np.cumsum(seg)])
    out = []
    for tgt in targets:
        i = int(np.searchsorted(cum, tgt, side="right") - 1)
        i = min(i, len(seg) - 1)
        f = (tgt - cum[i]) / seg[i] if seg[i] > 0 else 0
        out.append(pts[i] + f * (pts[i + 1] - pts[i]))
    return np.asarray(out)


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / max(np.linalg.norm(v), 1e-12)


def deg(x):
    return math.degrees(x)
