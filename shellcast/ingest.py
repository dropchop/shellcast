"""Ingest and repair: load a sculpt, clean it, gate it through manifold3d, rewrap if needed, build a proxy."""

from __future__ import annotations

import hashlib
import re
import zipfile
from pathlib import Path

import numpy as np
import trimesh

from .geom import Grid, Manifold, manifold_from_tm, tm_from_manifold
from .orient import resolve as resolve_orientation
from .interior import remove_interior


_VERT_RE = re.compile(rb'<vertex\s+x="([^"]+)"\s+y="([^"]+)"\s+z="([^"]+)"')
_TRI_RE = re.compile(rb'<triangle\s+v1="(\d+)"\s+v2="(\d+)"\s+v3="(\d+)"')
_OBJ_RE = re.compile(rb'<object\s[^>]*?\bid="(\d+)"[^>]*>')
_COMP_RE = re.compile(rb'<component\s([^>]*)>')
_ITEM_RE = re.compile(rb'<item\s([^>]*)>')
_ATTR_RE = re.compile(rb'([\w:]+)="([^"]*)"')


def _3mf_transform(txt):
    if not txt:
        return np.eye(4)
    v = np.array([float(x) for x in txt.decode().split()], dtype=np.float64)
    if len(v) != 12:
        return np.eye(4)
    M = np.eye(4)
    M[:3, :3] = v[:9].reshape(3, 3).T      # 3MF stores the matrix row-major as three basis rows
    M[:3, 3] = v[9:]
    return M


def _scan_chunks(buf: bytes, regex, dtype, chunk=48 << 20):
    """Regex findall over a big byte buffer in chunks (split at tag boundaries) → one numpy array. Peak memory
    is one chunk's worth of match tuples, not the whole file's."""
    parts, pos, n = [], 0, len(buf)
    while pos < n:
        end = min(n, pos + chunk)
        if end < n:
            cut = buf.rfind(b"<", pos, end)
            end = cut if cut > pos else end
        m = regex.findall(buf[pos:end])
        if m:
            parts.append(np.array(m, dtype=dtype))
        pos = end
    return np.concatenate(parts) if parts else np.zeros((0, 3), dtype=dtype)


def load_3mf_streaming(path: str | Path) -> trimesh.Trimesh:
    """Memory-bounded 3MF reader. Handles the core spec and the production extension used by OrcaSlicer /
    Bambu Studio / PrusaSlicer, where the root model only holds components that point (p:path) at per-object
    model parts, and the build item carries the placement/scale transform. Vertices and triangles are pulled
    with chunked regex scans, so a 7 M-triangle sculpt loads in ~1–2 GB instead of the >7 GB trimesh needs."""
    objects = {}          # (part path, id) -> ("mesh", V, F) | ("components", [(part path, id, M)])
    with zipfile.ZipFile(path) as z:
        parts = [n for n in z.namelist() if n.lower().endswith(".model")]
        if not parts:
            raise ValueError("3MF has no .model part")
        root = next((n for n in parts if n.lower() == "3d/3dmodel.model"), min(parts, key=len))
        items = []
        for part in parts:
            buf = z.read(part)
            key_path = "/" + part.lstrip("/")
            starts = [(m.start(), m.group(1).decode()) for m in _OBJ_RE.finditer(buf)]
            for start, oid in starts:
                end = buf.find(b"</object>", start)
                seg = buf[start:end if end >= 0 else len(buf)]
                comps = _COMP_RE.findall(seg)
                if comps:
                    lst = []
                    for c in comps:
                        a = {k.decode(): v for k, v in _ATTR_RE.findall(c)}
                        cpath = a.get("p:path", b"").decode() or key_path
                        lst.append((cpath, a.get("objectid", b"").decode(), _3mf_transform(a.get("transform"))))
                    objects[(key_path, oid)] = ("components", lst)
                    continue
                V = _scan_chunks(seg, _VERT_RE, np.float64)
                F = _scan_chunks(seg, _TRI_RE, np.int64)
                if len(V) and len(F):
                    objects[(key_path, oid)] = ("mesh", V, F)
            if part == root:
                b_at = buf.find(b"<build")
                for it in (_ITEM_RE.findall(buf[b_at:]) if b_at >= 0 else []):
                    a = {k.decode(): v for k, v in _ATTR_RE.findall(it)}
                    ipath = a.get("p:path", b"").decode() or key_path
                    items.append((ipath, a.get("objectid", b"").decode(), _3mf_transform(a.get("transform"))))
            del buf
    root_path = "/" + root.lstrip("/")
    if not items:
        items = [(pth, oid, np.eye(4)) for (pth, oid) in objects if pth == root_path] or [(pth, oid, np.eye(4)) for (pth, oid) in objects]
    meshes = []

    def emit(pth, oid, M, depth=0):
        o = objects.get((pth, oid))
        if not o or depth > 8:
            return
        if o[0] == "mesh":
            V = o[1] @ M[:3, :3].T + M[:3, 3]
            meshes.append(trimesh.Trimesh(V, o[2], process=False))
        else:
            for cpath, cid, cm in o[1]:
                emit(cpath, cid, M @ cm, depth + 1)

    for pth, oid, M in items:
        emit(pth, oid, M)
    if not meshes:
        raise ValueError("3MF contains no triangle meshes")
    return meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    if str(path).lower().endswith(".3mf"):
        try:
            return load_3mf_streaming(path)
        except Exception:  # noqa: BLE001 — unusual 3MF layout: fall back to trimesh's reader
            pass
    obj = trimesh.load(str(path), force="mesh", process=False)
    if isinstance(obj, trimesh.Scene):
        geoms = [g for g in obj.dump() if isinstance(g, trimesh.Trimesh)]
        if not geoms:
            raise ValueError("no triangle geometry in file")
        obj = trimesh.util.concatenate(geoms)
    if not isinstance(obj, trimesh.Trimesh) or len(obj.faces) == 0:
        raise ValueError("no triangle geometry in file")
    return obj


NOMINAL_HEIGHT_MM = 250.0


def guess_units(t: trimesh.Trimesh) -> tuple[str, float]:
    """(unit guess, scale-to-mm). A head is 150–350 mm tall; sculpts arrive in mm, cm, m, or Nomad/ZBrush "scene
    units" that are a few hundredths across. Anything that ends up under 100 mm is scaled to a nominal 250 mm
    head so the geometry is numerically sane; a fit target then sets the real size."""
    ext = float(np.max(t.extents))
    if ext <= 0:
        raise ValueError("mesh has no size")
    if 100 <= ext <= 600:
        return "mm", 1.0
    if 600 < ext <= 6000:
        return "mm (large)", 1.0
    if 10 <= ext < 100:
        return "cm", 10.0
    if 0.05 <= ext < 10:
        return "m", 1000.0
    return f"scene units ({ext:.4g} across) → scaled to {NOMINAL_HEIGHT_MM:.0f} mm", NOMINAL_HEIGHT_MM / ext


def clean(t: trimesh.Trimesh) -> trimesh.Trimesh:
    """Cheap, memory-light cleanup. Hole filling and normal repair only run when the mesh actually needs them —
    on a 7 M-triangle sculpt they cost gigabytes."""
    t = t.copy()
    t.merge_vertices()
    t.update_faces(t.nondegenerate_faces())
    t.update_faces(t.unique_faces())
    t.remove_unreferenced_vertices()
    if not t.is_watertight:
        try:
            t.fill_holes()
        except Exception:  # noqa: BLE001
            pass
    if t.volume < 0:
        t.invert()
    elif not t.is_watertight or len(t.faces) < 1_500_000:
        try:
            t.fix_normals()
        except Exception:  # noqa: BLE001
            pass
    return t


def mesh_hash(t: trimesh.Trimesh) -> str:
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(t.vertices, dtype=np.float32).tobytes())
    h.update(np.ascontiguousarray(t.faces, dtype=np.int32).tobytes())
    return h.hexdigest()[:16]


def proxy_of(t: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    if len(t.faces) <= target_faces:
        return t
    p = t.simplify_quadric_decimation(face_count=target_faces)
    p.merge_vertices()
    p.update_faces(p.nondegenerate_faces())
    return p


def rewrap(t: trimesh.Trimesh, pitch: float, log=None, max_cells=40_000_000) -> tuple[Manifold, dict]:
    """Voxel rewrap: signed distance from fast winding numbers (works on soups), iso-surface at 0.

    Returns the manifold and a deviation report (how far the new surface moved from the input).
    """
    g = Grid.for_bounds(t.bounds[0], t.bounds[1], pitch, margin=3 * pitch, max_cells=max_cells)
    pitch = g.pitch
    if log:
        log(f"rewrap grid {g.shape} = {g.n / 1e6:.1f}M cells at {pitch} mm")
    g.fill(t.vertices, t.faces, log=log)
    m = g.iso(0.0, smooth_iters=1, largest_only=False, min_fraction=0.001)
    # deviation: distance from rewrapped vertices to the original surface
    rw = tm_from_manifold(m)
    sample = rw.vertices[:: max(1, len(rw.vertices) // 20000)]
    closest, dist, _ = trimesh.proximity.closest_point(t, sample)
    rep = {"pitch": pitch, "cells": g.n, "deviation_mean": float(dist.mean()), "deviation_p95": float(np.percentile(dist, 95)),
           "deviation_max": float(dist.max()), "triangles_out": int(m.num_tri())}
    return m, rep


def ingest(path: str | Path, rewrap_pitch: float, proxy_faces: int, log=None, force_rewrap: bool = False,
           cavity_max_faces: int = 2_000_000, max_grid_cells: int = 40_000_000, orient: str = "auto") -> dict:
    """Load → units → clean → manifold gate (or rewrap) → proxy. Returns a dict of everything later stages need."""
    raw = load_mesh(path)
    unit, scale = guess_units(raw)
    if scale != 1.0:
        raw.apply_scale(scale)
        if log:
            log(f"units: {unit}, scaled ×{scale:.4g} → {np.max(raw.extents):.0f} mm across")
    if float(np.max(raw.extents)) < 100:                      # cm/m guess still tiny: normalise to a nominal head
        f = NOMINAL_HEIGHT_MM / float(np.max(raw.extents))
        raw.apply_scale(f)
        scale *= f
        unit += f" (then ×{f:.3g} to a nominal {NOMINAL_HEIGHT_MM:.0f} mm)"
    info = {"file": str(path), "triangles_in": int(len(raw.faces)), "unit_guess": unit, "scale_applied": scale,
            "bounds": raw.bounds.tolist(), "extents": raw.extents.tolist(), "watertight_in": bool(raw.is_watertight),
            "bodies_in": int(raw.body_count) if len(raw.faces) < 3_000_000 else None}
    # Decimate FIRST. Every later step (cleaning, manifold gate, body union, cavity boolean) scales with triangle
    # count in both time and memory; a 7 M-triangle bust needs >10 GB and many minutes at full resolution, while
    # 1 M triangles (≈0.35 mm edges on a head) is invisible on an FDM print and runs in a few minutes.
    if cavity_max_faces and len(raw.faces) > cavity_max_faces:
        if log:
            log(f"decimating {len(raw.faces):,} → {cavity_max_faces:,} triangles before repair")
        raw = proxy_of(raw, cavity_max_faces)
        info["decimated_to"] = int(len(raw.faces))
    t = clean(raw)
    method = "manifold"
    try:
        if force_rewrap:
            raise ValueError("rewrap forced")
        m = manifold_from_tm(t)
        m, merged = _union_bodies(_drop_tiny_bodies(m))
        if merged:
            method = "manifold+union"
        info["rewrap"] = None
    except ValueError as e:
        if log:
            log(f"sculpt is not boolean-ready ({e}); rewrapping at {rewrap_pitch} mm")
        m, rep = rewrap(t, rewrap_pitch, log=log, max_cells=max_grid_cells)
        info["rewrap"] = rep
        method = "rewrap"
        t = tm_from_manifold(m)
    m, interior = remove_interior(m, proxy_of(t, 150_000), log=log)
    if interior["count"]:
        info["interior_removed"] = interior
        t = tm_from_manifold(m)
    m, n_voids, v_voids = fill_voids(m)
    if n_voids:
        info["cavities_filled"] = {"count": n_voids, "volume_mm3": v_voids}
        if log:
            log(f"filled {n_voids} internal cavit{'y' if n_voids == 1 else 'ies'} ({v_voids / 1000:.0f} cm³) — a sculpt is solid")
    if cavity_max_faces and m.num_tri() > cavity_max_faces:
        if log:
            log(f"decimating cavity mesh {m.num_tri():,} → {cavity_max_faces:,} triangles")
        dec = proxy_of(tm_from_manifold(m), cavity_max_faces)
        try:
            m = manifold_from_tm(dec)
            t = dec
            info["cavity_decimated"] = True
        except ValueError:
            info["cavity_decimated"] = False
            if log:
                log("decimated mesh is not manifold; keeping the full-resolution sculpt")
    # union, stray-body drop and void fill only touched m: t must be the repaired solid before orienting or exporting
    t = tm_from_manifold(m)
    # put the sculpt in the head frame (+Z crown, face toward -Y)
    R, oinfo = resolve_orientation(orient, t)
    info["orientation"] = oinfo
    if not np.allclose(R, np.eye(3)):
        t = t.copy()
        t.vertices = t.vertices @ R.T
        m = manifold_from_tm(t)
        if log:
            log(f"oriented: crown was {oinfo['up']}, face was {oinfo['front']} → rotated into the head frame")
    info["method"] = method
    info["triangles_used"] = int(m.num_tri())
    info["volume_mm3"] = float(m.volume())
    info["surface_mm2"] = float(m.surface_area())
    info["hash"] = mesh_hash(t)
    proxy = proxy_of(t, proxy_faces)
    info["proxy_triangles"] = int(len(proxy.faces))
    return {"info": info, "mesh": t, "manifold": m, "proxy": proxy}


def fill_voids(m: Manifold):
    """A sculpt is a solid: drop every enclosed cavity (negative-volume component) such as a hollow mouth interior
    left behind by a closed-mouth edit. Returns (manifold, cavities removed, their volume mm3)."""
    import manifold3d as m3
    parts = m.decompose()
    neg = [q for q in parts if q.volume() <= 0]
    if not neg:
        return m, 0, 0.0
    pos = [q for q in parts if q.volume() > 0]
    out = pos[0] if len(pos) == 1 else Manifold.batch_boolean(pos, m3.OpType.Add)
    return out, len(neg), float(sum(-q.volume() for q in neg))


def _union_bodies(m: Manifold):
    """Overlapping closed bodies (ZBrush subtools: eyes, teeth, a neck plug) are topologically fine but self-
    intersecting; a boolean union resolves them into one clean solid. Returns (manifold, merged?)."""
    import manifold3d as m3
    parts = [q for q in m.decompose() if q.volume() > 0]
    if len(parts) <= 1:
        return m, False
    u = Manifold.batch_boolean(parts, m3.OpType.Add)
    return u, u.volume() < sum(q.volume() for q in parts) - 1e-6


def _drop_tiny_bodies(m: Manifold) -> Manifold:
    """ZBrush exports carry stray subtools (eyes, teeth). Keep bodies above 0.5% of the largest (with their cavities)."""
    from .geom import keep_components
    return keep_components(m, min_fraction=0.005)
