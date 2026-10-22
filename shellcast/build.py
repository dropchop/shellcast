"""Mold geometry: conformal shell halves with flanges, bolts and keys; hollow core with collar and bolt tabs; sprue; vents.

All solids are manifold3d Manifolds in the head frame (+Z crown, neck at -Z). The cavity face is carved from the
full-resolution sculpt with an exact boolean; every other surface comes from a coarse signed-distance grid.

The sculpt is first extended below the neck plane with a prism of its neck outline ("neck extension"). Every
offset surface then continues naturally down that prism: the shell's outer skin becomes the neck sleeve, the
flange band follows it, and the core becomes the collar plug — no special-case geometry at the neck.
"""

from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy import ndimage

import manifold3d as m3

from .geom import (Grid, Manifold, cross_section_polys, cylinder_along, halfspace_keep, keep_components,
                   resample_polyline, slab, strip_slivers, tm_from_manifold, unit)
from .ingest import proxy_of
from .blank import place_blank
from .params import MoldParams

AXIS = {"x": np.array([1.0, 0, 0]), "y": np.array([0, 1.0, 0])}
UP = np.array([0, 0, 1.0])


class Build:
    """Holds every intermediate so the pipeline can report on, preview and validate each one."""

    def __init__(self, sculpt: Manifold, proxy: trimesh.Trimesh, p: MoldParams, log=None):
        self.S = sculpt
        self.proxy = proxy
        self.mouth_void = None
        self.closed = None            # second sculpt (closed mouth) in the same frame, for the mouth cover
        self.closed_proxy = None
        self.d_closed = None
        self.block = None
        self.mouth_wedge = None
        self.swept_void = None
        self.p = p
        self.log = log or (lambda s: None)
        self.n = AXIS[p.pull_axis]
        bmin, bmax = np.asarray(proxy.bounds[0]), np.asarray(proxy.bounds[1])
        self.center = (bmin + bmax) / 2
        self.plane_point = self.center + self.n * p.parting_offset
        self.z_neck = float(bmin[2] + p.neck_offset)
        self.z_bottom = self.z_neck - p.collar_height          # bottom of sleeve / collar
        self.grid: Grid | None = None
        self.parts: dict[str, Manifold] = {}
        self.vents: list[dict] = []
        self.bolts: list[dict] = []
        self.notes: list[str] = []
        self.sprue = None
        self.ring_holes = None
        self.needs_ring = p.mold_type == "mask" or p.puppet_cavity == "rod"
        # neck outline and the extended sculpt used for every offset. If the plane sits on a tiny footprint (a bust
        # resting on a small base, a rounded chin touching the bed) it is raised to the first real neck section.
        self.outline = self.S.slice(self.z_neck)
        auto = self._auto_neck(bmin[2], bmax[2])
        if auto is not None:
            self.z_neck = auto
            self.z_bottom = self.z_neck - p.collar_height
            self.outline = self.S.slice(self.z_neck)
            self.notes.append(f"Mask cut at the neck, {self.z_neck - bmin[2]:.0f} mm above the mesh bottom (narrowest section under the jaw). "
                              f"Set neck_offset to cut elsewhere.")
        if self.outline.is_empty() or p.collar_height <= 0:
            self.outline = None
            self.S_ext = self.S
            self.proxy_ext = proxy
            self.notes.append("No neck outline at the neck plane; no sleeve or collar was built (raise neck_offset so the plane cuts the mesh).")
        else:
            L = p.collar_height + p.shell_thickness + p.flange_width + 6
            prism = Manifold.extrude(self.outline, L + 0.5).translate((0, 0, self.z_neck - L))
            # everything below the neck plane (shoulders, a base) is discarded: the mask ends at the neck and only
            # the prism continues down, so the sleeve, ring and collar have free space around them
            if self.z_neck > bmin[2] + 0.5:
                self.S = halfspace_keep(self.S, UP, [0, 0, self.z_neck - 0.5])
            self.S_ext = self.S + prism
            self.proxy_ext = proxy_of(tm_from_manifold(self.S_ext), p.proxy_faces)

    def _auto_neck(self, z0, z1):
        """Where to cut the mask off. Returns a z or None (keep the requested plane).

        Busts arrive with shoulders or a base; a mask wants to end at the neck. Below 45 % of the height the
        narrowest horizontal section (the neck) is found; the plane goes there when the requested plane's
        section is either too small to be a neck (a rounded chin touching the bed) or clearly wider than the
        neck (shoulders / a base). A user-set neck_offset is honoured when it already looks like a neck."""
        p = self.p
        area0 = self.outline.area() if not self.outline.is_empty() else 0.0
        zs = np.linspace(z0 + 2.0, z0 + (z1 - z0) * 0.45, 60)
        areas = np.array([(lambda cs: cs.area() if not cs.is_empty() else 0.0)(self.S.slice(float(z))) for z in zs])
        big = areas.max() if len(areas) else 0.0
        if big <= 0:
            return None
        valid = areas >= max(2500.0, 0.10 * big)
        if not valid.any():
            return None
        # the neck is the narrowest section ABOVE the widest part of the lower body (shoulders / base)
        k_wide = int(np.argmax(areas[: max(2, len(areas) * 2 // 3)]))
        upper = np.where(valid, areas, np.inf).copy()
        upper[:k_wide] = np.inf
        k = int(np.argmin(upper))
        neck_z, neck_area = float(zs[k]), float(areas[k])
        wider_below = float(areas[k_wide]) > 1.15 * neck_area
        if p.neck_offset > 0 and area0 <= 1.3 * neck_area and area0 >= max(2500.0, 0.10 * big):
            return None                                     # the user's plane is already a neck-like section
        if area0 >= max(2500.0, 0.10 * big) and not wider_below:
            return None                                     # bottom is the neck already
        return neck_z + 1.0

    # ------------------------------------------------------------------ SDF
    def make_grid(self):
        p = self.p
        margin = p.shell_thickness + p.flange_width + 4 * p.voxel_pitch
        g = Grid.for_bounds(self.proxy_ext.bounds[0], self.proxy_ext.bounds[1], p.voxel_pitch, margin, max_cells=p.max_grid_cells)
        if g.pitch > p.voxel_pitch:
            self.notes.append(f"Voxel pitch raised from {p.voxel_pitch} to {g.pitch} mm to keep the SDF grid under {p.max_grid_cells / 1e6:.0f}M cells.")
        self.log(f"sdf grid {g.shape} = {g.n / 1e6:.1f}M cells at {g.pitch} mm on {len(self.proxy_ext.faces)} proxy tris")
        g.fill(self.proxy_ext.vertices, self.proxy_ext.faces, log=self.log)
        self.grid = g
        if self.closed_proxy is not None:
            self.log("sdf of the closed-mouth sculpt")
            self.d_closed = g.eval(self.closed_proxy.vertices, self.closed_proxy.faces, log=self.log)
        return g

    # ------------------------------------------------------------------ shell
    def make_shell(self):
        p, g = self.p, self.grid
        w, c, F, T = p.shell_thickness, p.cavity_clearance, p.flange_width, p.flange_thickness
        self.log("outer skin iso-surface")
        self.mouth_void = None
        if p.mouth_insert:
            self._mouth_void()
        if p.mouth_insert and self.mouth_void is not None:
            # around the mouth the skin follows the *closed* envelope (solid over the void, like a puppet shell) with a
            # smooth 6 mm boss so the insert's window can be stepped; elsewhere it is the normal thin skin
            near = self._smoothstep((35.0 - self.dV) / 15.0)                    # 1 within 20 mm of the void, 0 beyond 35
            base = near * self.closed_field + (1 - near) * g.d
            extra = 6.0 * self._smoothstep((22.0 - self.dV) / 8.0)
            self.skin_field = base - (c + w) - extra
            O = g.iso(0.0, field=self.skin_field, smooth_iters=p.smooth_iterations)
            OF = g.iso(c + w + F, field=base, smooth_iters=p.smooth_iterations)
            OM = g.iso(c + w + F / 2, field=base, smooth_iters=p.smooth_iterations)
        elif p.mold_type == "puppet":
            # a puppet mold casts the open mouth, teeth and all: make the shell solid inside such pockets instead of a
            # thin skin lining them (a closing of the outer skin; the cavity face is still the exact sculpt)
            skin = g.morph_field(0.0, close_radius=25.0, log=self.log, floor=c + w)
            O = g.iso(c + w, field=skin, smooth_iters=p.smooth_iterations)
            OF = g.iso(c + w + F, field=skin, smooth_iters=p.smooth_iterations)
            OM = g.iso(c + w + F / 2, field=skin, smooth_iters=p.smooth_iterations)
            self.notes.append("Puppet shells are solid inside the mouth and other pockets (outer skin closed over 25 mm).")
        else:
            O = g.iso(c + w, smooth_iters=p.smooth_iterations)
            OF = g.iso(c + w + F, smooth_iters=p.smooth_iterations)
            OM = g.iso(c + w + F / 2, smooth_iters=p.smooth_iterations)
        self.log("flange skin iso-surfaces")
        inner = self.S_ext if c == 0 else g.iso(c, smooth_iters=0)
        self.log(f"cavity boolean: skin {O.num_tri()} tris − sculpt {inner.num_tri()} tris")
        shell = O - inner
        shell = keep_components(halfspace_keep(shell, UP, [0, 0, self.z_bottom]), largest_only=True)
        self.O, self.OF, self.OM = O, OF, OM

        self.log("parting flange")
        flange = (OF - O) ^ slab(self.plane_point, self.n, 2 * T)
        flange = halfspace_keep(flange, UP, [0, 0, self.z_bottom])
        full = shell + flange
        if p.mouth_insert and self.mouth_void is not None:
            full = self._cut_mouth_window(full, inner)
        if self.outline is not None and self.needs_ring:
            # flat ring at the sleeve bottom for the core's collar tabs (or the puppet neck plug) to bolt against
            o_f = self.outline.offset(w + F, m3.JoinType.Round)
            ring = (Manifold.extrude(o_f, T) - Manifold.extrude(self.outline, T)).translate((0, 0, self.z_bottom))
            full = full + ring
        self.full_shell = full
        return full

    # ------------------------------------------------------------------ split, bolts, keys
    def split_shell(self):
        p = self.p
        n, pt = self.n, self.plane_point
        off = float(np.dot(n, pt))
        self.log("splitting shell halves")
        A, B = self.full_shell.split_by_plane(tuple(n), off)      # A: n·x >= off
        self.log("bolt line")
        line_pts = self._flange_centerline()
        loop_bolts = resample_polyline(line_pts, p.bolt_spacing, closed=False) if len(line_pts) > 1 else np.zeros((0, 3))
        holes, pins, sockets = [], [], []
        r_hole = (p.bolt_diameter + p.bolt_clearance) / 2
        L = 2 * p.flange_thickness + 4
        # a narrow flange cannot take the bolts: keep the keys, skip the holes, say so
        if p.flange_width < p.bolt_diameter + p.bolt_clearance + 4:
            self.notes.append(f"Flange is {p.flange_width:.0f} mm wide: too narrow for M{p.bolt_diameter:.0f} bolt holes, none were cut "
                              f"(use clamps or spring clips, or widen the flange to ≥ {p.bolt_diameter + p.bolt_clearance + 4:.0f} mm).")
            self.bolt_positions = loop_bolts
            loop_bolts = np.zeros((0, 3))
        else:
            self.bolt_positions = loop_bolts
        for b in loop_bolts:
            b = b - n * (np.dot(n, b) - off)             # project onto the parting plane
            holes.append(cylinder_along(b - n * L / 2, n, r_hole, r_hole, L, 24))
            self.bolts.append({"xyz": b.tolist(), "diameter": p.bolt_diameter})
        # conical keys between consecutive bolts: alternate pin on A / socket in B. A midpoint is used only if it
        # really sits on the flange band (the loop is curved, the chord between bolts is not).
        # registration keys: the cone must fit on the band between the two flange skins, so its radius shrinks
        # with the flange (a 4 mm key needs a 10 mm flange; a 6 mm flange gets 2 mm keys)
        r = float(min(p.key_radius, max(1.5, (p.flange_width - 2.0) / 2.0 - 0.5)))
        if len(self.bolt_positions) > 1:
            mids = resample_polyline(line_pts, p.bolt_spacing, closed=False, phase=1.0)   # along the curve, between bolts
            h = r * 0.8
            if r < p.key_radius - 0.01:
                self.notes.append(f"Registration keys shrunk to {r:.1f} mm radius to fit the {p.flange_width:.0f} mm flange.")
            band_lo = p.cavity_clearance + p.shell_thickness + r + 1.0
            band_hi = p.cavity_clearance + p.shell_thickness + p.flange_width - r - 1.0
            for i, mpt in enumerate(mids):
                mpt = mpt - n * (np.dot(n, mpt) - off)
                if not (band_lo <= self.grid.sample(mpt) <= band_hi):
                    continue
                d = -n if i % 2 == 0 else n               # even keys: pin on A into B; odd: pin on B into A
                # start the pin 0.5 mm inside its own half so the union overlaps a volume, not just a coplanar face
                pin = cylinder_along(mpt - d * 0.5, d, r, 0.35 * r, h + 0.5, 24)
                sock = cylinder_along(mpt - d * p.key_clearance, d, r + p.key_clearance, 0.35 * r + p.key_clearance, h + p.key_clearance, 24)
                pins.append((pin, "A" if i % 2 == 0 else "B"))
                sockets.append((sock, "B" if i % 2 == 0 else "A"))
        self.log(f"{len(holes)} bolts, {len(pins)} keys (r {r:.1f} mm)")
        self.n_keys = len(pins)
        if holes:
            H = Manifold.batch_boolean(holes, m3.OpType.Add)
            A, B = A - H, B - H
        for pin, side in pins:
            A, B = (A + pin, B) if side == "A" else (A, B + pin)
        for sock, side in sockets:
            A, B = (A - sock, B) if side == "A" else (A, B - sock)
        self.ring_holes = self._ring_bolt_holes()
        if self.ring_holes is not None:
            A, B = A - self.ring_holes, B - self.ring_holes
        self.parts["shell_A"], self.parts["shell_B"] = A, B
        return A, B

    def _flange_centerline(self) -> np.ndarray:
        """Polyline where the parting plane crosses the flange midline skin, kept clear of the neck ring."""
        t = tm_from_manifold(self.OM)
        sec = t.section(plane_origin=self.plane_point, plane_normal=self.n)
        if sec is None:
            return np.zeros((0, 3))
        loops = [np.asarray(d) for d in sec.discrete]
        loops.sort(key=lambda a: -len(a))
        pts = loops[0]
        z_lo = self.z_bottom + self.p.flange_thickness + self.p.bolt_diameter + 2
        keep = pts[:, 2] >= z_lo
        if not keep.any():
            return np.zeros((0, 3))
        if not keep.all():                                # rotate the closed loop so the dropped run is at the end
            i0 = int(np.argmax(~keep))
            pts = np.roll(pts, -i0, axis=0)
            keep = pts[:, 2] >= z_lo
            pts = pts[keep]
        return pts

    def _ring_bolt_holes(self):
        p = self.p
        if self.outline is None or not self.needs_ring:
            return None
        polys = cross_section_polys(self.outline.offset(p.shell_thickness + p.flange_width / 2, m3.JoinType.Round))
        if not polys:
            return None
        poly = max(polys, key=len)
        pts2 = resample_polyline(poly, p.bolt_spacing, closed=True)
        off = float(np.dot(self.n[:2], self.plane_point[:2]))
        pts2 = [q for q in pts2 if abs(q @ self.n[:2] - off) > p.flange_thickness + p.bolt_diameter]
        if not pts2:
            return None
        r = (p.bolt_diameter + p.bolt_clearance) / 2
        z0 = self.z_bottom - p.plate_thickness - 2
        L = p.flange_thickness + p.plate_thickness + 4
        cyls = [cylinder_along([q[0], q[1], z0], UP, r, r, L, 24) for q in pts2]
        for q in pts2:
            self.bolts.append({"xyz": [float(q[0]), float(q[1]), self.z_bottom], "diameter": p.bolt_diameter, "ring": True})
        return Manifold.batch_boolean(cyls, m3.OpType.Add)

    # ------------------------------------------------------------------ core
    def make_core(self):
        p, g = self.p, self.grid
        if p.mold_type == "puppet":
            return self.make_neck_plug()
        t, k = p.mask_thickness, p.core_wall
        self.log("core: inward offset + closing/opening")
        d_core = g.d if self.d_closed is None else np.maximum(g.d, self.d_closed)   # clear both versions of the lips
        if getattr(self, "swept_void", None) is not None:
            # wedge mode: the pocket is the wedge's pull-out shadow, so the blank ignores the open sculpt's inner mouth
            # detail (teeth cast solid) and clears the shadow by the mask thickness
            d_core = np.maximum(g.d, -self.swept_void)
        if p.core_shape == "head":
            # a generic human head form, clipped so the mask stays >= t thick everywhere
            target = getattr(self, "blank_circumference", None) or (self.proxy_hat_circumference() - 2 * np.pi * t)
            blank, binfo = place_blank(self.S, target, log=self.log)
            bt = tm_from_manifold(blank)
            # where the sculpt term decides max() anyway (outside, or less than two cells under, the sculpt offset)
            # the blank's distance is not needed: skip it, it is most of the grid
            floor_t = d_core + t
            field = np.maximum(g.eval(bt.vertices, bt.faces, log=self.log, only=floor_t <= 2 * g.pitch, known=floor_t), floor_t)
            self.blank_info = binfo
            self.notes.append(f"Core is a generic smoothed head form (hat line {binfo['blank_hat_circumference_mm']:.0f} mm), clipped to keep the mask ≥ {t:.0f} mm; "
                              f"the mask is solid wherever the sculpt is bigger than a head.")
        else:
            field = g.morph_field(-t, p.core_close_radius, p.core_open_radius, base=d_core)
        if p.core_smooth_mm > 0 and p.core_shape != "head":
            # "smooth inside": blur the core field so the inner face of the mask is a soft head shape, then clamp so the
            # core never gets closer to the sculpt than the mask thickness (the mask only ever thickens in the details)
            blurred = ndimage.gaussian_filter(field, sigma=p.core_smooth_mm / g.pitch)
            field = np.maximum(blurred, d_core + t)
            self.notes.append(f"Inside of the mask smoothed over {p.core_smooth_mm:.0f} mm; wall is {t:.0f} mm at the thinnest and thicker in folds and details.")
        outer = g.iso(0.0, field=field, smooth_iters=p.smooth_iterations, largest_only=True)
        self.core_field = field
        inner_field = field + k
        if p.core_roof_angle > 0 and (inner_field <= 0).any():
            inner_field = self._cathedral(inner_field, p.core_roof_angle)
        hollow = inner_field.min() < 0
        if hollow:
            inner = g.iso(0.0, field=inner_field, smooth_iters=p.smooth_iterations, largest_only=False, min_fraction=0.02)
            core = outer - inner
        else:
            core = outer
            self.notes.append("Core is too thin to hollow at this wall thickness; it was left solid.")
        core = halfspace_keep(core, UP, [0, 0, self.z_bottom])
        self.core_outer = outer
        if self.outline is not None:
            H = p.collar_height
            collar_cs = self.outline.offset(-p.collar_clearance, m3.JoinType.Round)
            collar = Manifold.extrude(collar_cs, H).translate((0, 0, self.z_bottom))          # fills the sleeve: the register
            core = core + collar + self._collar_plate()
            if hollow:
                hole_cs = inner.slice(self.z_bottom + 1.0)
                if not hole_cs.is_empty():
                    hole_cs = hole_cs.offset(-0.5, m3.JoinType.Round)
                if not hole_cs.is_empty():                                                  # keep the hollow open at the bottom
                    core = core - Manifold.extrude(hole_cs, H + p.plate_thickness + 2).translate((0, 0, self.z_bottom - p.plate_thickness - 1))
            if self.ring_holes is not None:
                core = core - self.ring_holes
            sprue = self._sprue()
            if sprue is not None:
                core = core - sprue
        core = keep_components(core, largest_only=True)
        self.parts["core"] = core
        return core

    def prepare_wedge(self):
        """Before the core: the open mouth's void V (from the registered open-mouth sculpt) and its shadow swept toward
        the face along the pull axis. The blank's mouth pocket is carved from that shadow, not from the void itself, so
        the rigid wedge can be pushed straight into the blank along the pegs (no interlocking with tooth cores) and the
        silicone teeth cast solid."""
        p, g = self.p, self.grid
        self.swept_void = None
        if self.closed is None or self.d_closed is None:
            self.notes.append("Mouth wedge skipped: no open-mouth sculpt was given (upload it as the second file).")
            return None
        self._mouth_void(sdf=self.d_closed, solid=self.closed)
        if self.mouth_void is None:
            return None
        # signed distance to the selected void only (dV, from _mouth_void); for each cell take its minimum over all
        # cells deeper into the head (+y): ≤ 0 wherever a point of the void lies straight behind it, i.e. the volume the
        # wedge sweeps when pulled out toward the face. The lip skin and the shell are in front of the void anyway.
        self.swept_void = np.minimum.accumulate(self.dV[:, ::-1, :], axis=1)[:, ::-1, :]
        return self.mouth_void

    def make_wedge(self):
        """Mouth wedge: the open-mouth sculpt's void, cut back to one mask thickness behind the working (closed-mouth)
        sculpt's lips, printed as a core. Cast with it in: sealed lips, hollow mouth with silicone teeth and tongue.
        Anchored to the head blank at the throat by a boss and two pegs."""
        p, g = self.p, self.grid
        t = p.mask_thickness
        V = self.mouth_void
        if V is None:
            return None
        inset = g.iso(-t, smooth_iters=1, largest_only=True)                 # the closed sculpt eroded by the mask thickness
        wedge = keep_components(V ^ inset, largest_only=True)
        if wedge.is_empty() or wedge.volume() < 1000:
            self.notes.append("Mouth wedge skipped: the open mouth's void lies entirely within the lip skin.")
            return None
        # throat anchor: the deepest part of the void (largest y = into the head) → a boss to the blank and two pegs into it
        tv = tm_from_manifold(V)
        y = tv.vertices[:, 1]
        deep = tv.vertices[y >= y.min() + 0.85 * (y.max() - y.min())]
        A = deep.mean(0)
        note_anchor = ""
        core_field = getattr(self, "core_field", None)
        if core_field is not None and "core" in self.parts:
            # march from A into the head (+y) until we are inside the blank
            steps = np.arange(0.0, 80.0, 0.5)
            pts = A[None, :] + steps[:, None] * np.array([0, 1.0, 0])
            idx = ((pts - g.origin) / g.pitch).T
            vals = ndimage.map_coordinates(core_field, idx, order=1, mode="nearest")
            hit = np.nonzero(vals <= 0)[0]
            if len(hit):
                gap = float(steps[hit[0]])
                yhat = np.array([0, 1.0, 0]); xhat = np.array([1.0, 0, 0])
                Rb, Rp = p.wedge_boss_diameter / 2, p.wedge_peg_diameter / 2
                boss = cylinder_along(A - 4 * yhat, yhat, Rb, Rb, gap + 8)
                core_grown = g.iso(0.3, field=core_field, smooth_iters=0, largest_only=True)
                wedge = ((wedge + boss) - core_grown) - self.parts["core"]      # exact: the smoothed blank surface can stray past +0.3 mm
                pegs = Manifold.batch_boolean([cylinder_along(A + sgn * 0.55 * Rb * xhat - 4 * yhat, yhat, Rp, Rp, gap + 4 + 12) for sgn in (-1, 1)], m3.OpType.Add)
                pegs_grown = Manifold.batch_boolean([cylinder_along(A + sgn * 0.55 * Rb * xhat - 4 * yhat, yhat, Rp + 0.25, Rp + 0.25, gap + 4 + 12.5) for sgn in (-1, 1)], m3.OpType.Add)
                wedge = wedge + pegs
                self.parts["core"] = self.parts["core"] - pegs_grown
                note_anchor = (f" Anchored at the throat: a {p.wedge_boss_diameter:.0f} mm boss meets the blank {gap:.0f} mm behind the void and two "
                               f"{p.wedge_peg_diameter:.0f} mm pegs go 12 mm into it (holes in the blank, 0.25 mm clearance).")
            else:
                note_anchor = " No blank surface found behind the throat within 80 mm: the wedge has no anchor; glue it to the blank."
        wedge = keep_components(wedge, largest_only=True)
        self.parts["mouth_wedge"] = wedge
        self.mouth_wedge = wedge
        self.notes.append(f"Mouth wedge: {wedge.volume() / 1000:.0f} cm³ textured core from the open-mouth sculpt's void, cut back {t:.0f} mm behind "
                          f"the closed lips. Cast with it in: sealed lips over a hollow mouth with silicone teeth and tongue; slit the lips to open it."
                          + note_anchor)
        return wedge

    def make_neck_plug(self):
        """Puppet molds have no head blank. The neck sleeve is the pour spout (solid pour or slush cast). With
        puppet_cavity=rod a third part plugs the neck: a plate that bolts to the ring, a collar that fills the
        sleeve, and a rod that leaves an armature/handle hole in the casting."""
        p = self.p
        self.core_outer = None
        if p.puppet_cavity != "rod" or self.outline is None:
            self.notes.append("Puppet mold: two shell halves, no head blank. Pour (or slush) through the neck sleeve.")
            return None
        H = p.collar_height
        collar = Manifold.extrude(self.outline.offset(-p.collar_clearance, m3.JoinType.Round), H).translate((0, 0, self.z_bottom))
        rod = cylinder_along([self.center[0], self.center[1], self.z_bottom], UP, p.rod_diameter / 2, p.rod_diameter / 2, H + p.rod_depth, 32)
        # the collar only fills the sleeve below the neck plane; above it the rod alone reaches into the head
        plug = self._collar_plate() + collar + rod
        if self.ring_holes is not None:
            plug = plug - self.ring_holes
        plug = keep_components(plug, largest_only=True)
        self.parts["neck_plug"] = plug
        self.notes.append(f"Puppet mold with a neck plug: {p.rod_diameter:.0f} mm rod, {p.rod_depth:.0f} mm into the head; three parts.")
        return plug

    @staticmethod
    def _smoothstep(x):
        x = np.clip(x, 0.0, 1.0)
        return x * x * (3 - 2 * x)

    def _mouth_void(self, sdf=None, solid=None):
        """V = the open mouth's void: what a closing of the sculpt over `mouth_radius` fills, in the mouth band, in front.
        Exact against the sculpt (teeth and mouth interior are real sculpt geometry). `sdf`/`solid` default to the
        working sculpt; the wedge mode passes the registered open-mouth sculpt instead."""
        p, g = self.p, self.grid
        sdf = g.d if sdf is None else sdf
        solid = self.S_ext if solid is None else solid
        R = max(p.mouth_radius, 12.0)
        closed = g.morph_field(0.0, close_radius=R, log=self.log, base=sdf, floor=0.0)
        self.closed_field = closed
        fill = np.maximum(closed, -sdf)                      # inside the closed envelope, outside the sculpt
        self.void_field = fill
        try:
            blobs = g.iso(0.0, field=fill, smooth_iters=0, largest_only=False, min_fraction=0.02)
        except ValueError:
            self.notes.append("No mouth void found for an insert (is the mouth open, and mouth_radius large enough?).")
            return
        b0, b1 = self.proxy.bounds
        h = b1[2] - b0[2]
        best, best_v = None, 0.0
        for q in blobs.decompose():
            if q.volume() <= 0:
                continue
            bb = q.bounding_box(); cen = [(bb[i] + bb[i + 3]) / 2 for i in range(3)]
            if cen[1] < (b0[1] + b1[1]) / 2 and b0[2] + 0.10 * h < cen[2] < b0[2] + 0.62 * h and q.volume() > best_v:
                best, best_v = q, q.volume()
        if best is None or best_v < 5000:
            self.notes.append("No mouth void found for an insert (is the mouth open, and mouth_radius large enough?).")
            return
        V = (best - solid)
        V = keep_components(V, largest_only=True)
        self.mouth_void = V
        tv = tm_from_manifold(V)
        self.dV = g.eval(tv.vertices, tv.faces, log=self.log)   # distance to the void, for the boss and the window
        self.log(f"mouth void: {V.volume() / 1000:.0f} cm³, {V.num_tri()} tris")
        if self.d_closed is not None:
            # the window must also enclose everything the closed-mouth sculpt changes (lips, maybe the jaw)
            # surface of one version that is not near the surface of the other (identical surfaces sit within a voxel of each
            # other on both fields; registration noise is well under the half-voxel margin)
            on_o, on_c = np.abs(g.d) < g.pitch, np.abs(self.d_closed) < g.pitch
            far_o, far_c = np.abs(g.d) > g.pitch + 0.5, np.abs(self.d_closed) > g.pitch + 0.5
            diff = ((on_o & far_c) | (on_c & far_o)) & (self.dV < 60.0)
            lab, n = ndimage.label(diff)
            if n:
                # keep only the edited regions that touch the mouth (lips, jaw); stray specks elsewhere are noise
                near = ndimage.minimum(self.dV, lab, index=np.arange(1, n + 1)) < 15.0
                diff = near[lab - 1] & (lab > 0)
            if diff.any():
                d_diff = ndimage.distance_transform_edt(~diff, sampling=g.pitch).astype(np.float32)
                self.dV = np.minimum(self.dV, d_diff)
                self.log(f"closed-mouth edit differs over {diff.sum() * g.pitch ** 3 / 1000:.0f} cm³ at the mouth; window widened to cover it")
            else:
                self.log("closed-mouth edit differs only where the void is")

    def _cut_mouth_window(self, full, inner):
        """Make the insert = void + a stepped plug of the shell around the lips, and cut its window out of the shell:
        inner part of the plug within 12 mm of the void through the whole skin, outer part within 16 mm only in the
        boss layer, so the plug seats on a step from the outside and cannot fall into the cavity."""
        p, g = self.p, self.grid
        w, c = p.shell_thickness, p.cavity_clearance
        depth_field = g.d - (c + w)                          # <= 0 within the normal skin thickness of the sculpt
        A = np.maximum.reduce([self.skin_field, -g.d, self.dV - 12.0])
        B = np.maximum.reduce([self.skin_field, -g.d, self.dV - 16.0, -depth_field])
        plug_field = np.minimum(A, B)
        # same field, no smoothing: the +clearance level set strictly contains the 0 level set, so plug and window nest
        plug = g.iso(0.0, field=plug_field, smooth_iters=0, largest_only=True)
        hole = g.iso(p.insert_clearance, field=plug_field, smooth_iters=0, largest_only=True)   # plug grown by the clearance
        insert = keep_components((plug + self.mouth_void) - inner, largest_only=True)
        # the grown window gives the clearance; subtracting the plug itself too makes shell and insert disjoint exactly
        # (two marching-cubes surfaces 0.25 mm apart can still cross by chord error at the step corners)
        full = ((full - hole) - plug) - self.mouth_void
        self.parts["mouth_insert"] = insert
        self.block = plug + self.mouth_void
        self.notes.append(f"Mouth insert: {insert.volume() / 1000:.0f} cm³ plug forming the open mouth; seats in a stepped window in the "
                          f"shell around the lips ({p.insert_clearance} mm clearance). Insert in = open mouth (puppet, or an open-mouth mask).")
        if self.closed is not None:
            cover = keep_components(self.block - self.closed, largest_only=True)
            self.notes.append(f"Mouth cover: {cover.volume() / 1000:.0f} cm³ plug for the same window whose cavity face is the closed-mouth "
                              f"sculpt. Cover in = closed-mouth mask.")
        else:
            cover = keep_components((plug - self.mouth_void) - inner, largest_only=True)
            self.notes.append(f"Mouth cover: {cover.volume() / 1000:.0f} cm³ plug for the same window that bridges the lips smoothly "
                              f"(no closed-mouth sculpt was given). Cover in = the mouth casts as a solid block.")
        self.parts["mouth_cover"] = cover
        return full

    def _collar_plate(self) -> Manifold:
        """What the collar bolts to the shell's neck ring with: bolt tabs at some ring bolts (cheap; tiny support when
        the core prints neck-up) tied together by a disc under the collar. plate_tabs = 0 gives a full plate."""
        p = self.p
        z0 = self.z_bottom - p.plate_thickness
        full = Manifold.extrude(self.outline.offset(p.shell_thickness + p.flange_width, m3.JoinType.Round), p.plate_thickness).translate((0, 0, z0))
        ring = [b for b in self.bolts if b.get("ring")]
        if p.plate_tabs <= 0 or not ring:
            return full
        step = max(1, len(ring) // p.plate_tabs)
        chosen = ring[::step][:p.plate_tabs]
        wdt = p.bolt_diameter + p.bolt_clearance + 10
        tabs = []
        for b in chosen:
            x, y = b["xyz"][0], b["xyz"][1]
            ang = np.degrees(np.arctan2(y - self.center[1], x - self.center[0]))
            L = np.hypot(x - self.center[0], y - self.center[1]) + p.flange_width
            bar = Manifold.cube((L, wdt, p.plate_thickness)).translate((0, -wdt / 2, 0)).rotate((0, 0, ang)).translate((self.center[0], self.center[1], z0))
            tabs.append(bar)
            b["tab"] = True
        tabs = Manifold.batch_boolean(tabs, m3.OpType.Add) ^ full
        disc = Manifold.extrude(self.outline.offset(p.shell_thickness, m3.JoinType.Round), p.plate_thickness).translate((0, 0, z0))
        return tabs + disc

    def _cathedral(self, inner_field, angle_deg):
        """Cap the hollow with a cone of the given slope so the core can print neck-down without internal support."""
        g = self.grid
        idx = np.argwhere(inner_field <= 0)
        z_top = idx[:, 2].max()
        top = idx[idx[:, 2] >= z_top - max(1, int(8 / g.pitch))]
        cx, cy = g.origin[0] + top[:, 0].mean() * g.pitch, g.origin[1] + top[:, 1].mean() * g.pitch
        z_apex = g.origin[2] + z_top * g.pitch
        ax = [g.origin[i] + g.pitch * np.arange(g.shape[i]) for i in range(3)]
        X, Y, Z = ax[0][:, None, None], ax[1][None, :, None], ax[2][None, None, :]
        cone = Z - (z_apex - np.hypot(X - cx, Y - cy) * np.tan(np.radians(angle_deg)))
        return np.maximum(inner_field, cone.astype(np.float32))

    def _sprue(self) -> Manifold | None:
        """Pour cup + gate. The cup is a funnel in the collar plate (head inverted, so it faces up when pouring); it feeds
        a thin gate slot that runs up through the collar centred in the silicone gap, so it can neither cut into the
        hollow core nor touch the sleeve. Placed at the back of the neck, away from the parting seam."""
        p = self.p
        t = p.mask_thickness
        polys = cross_section_polys(self.outline.offset(-t / 2 - 0.5, m3.JoinType.Round))   # gate centred in the gap, 0.5 mm toward the core
        if not polys:
            self.notes.append("Neck outline is too thin for a pour gate; no sprue was cut — add one in the slicer or raise neck_offset.")
            return None
        poly = max(polys, key=len)
        i = int(np.argmax(poly[:, 0])) if p.pull_axis == "y" else int(np.argmax(poly[:, 1]))
        q = poly[i]
        tangent = unit(poly[(i + 1) % len(poly)] - poly[i - 1])
        ang = np.degrees(np.arctan2(tangent[1], tangent[0]))
        z_cup = self.z_bottom - p.plate_thickness - 1
        cup_h = p.plate_thickness + 8
        cup = cylinder_along([q[0], q[1], z_cup], UP, p.sprue_top_diameter / 2, p.sprue_bottom_diameter / 2, cup_h + 1, 32)
        gate_len, gate_th = max(20.0, p.sprue_bottom_diameter * 2.5), max(1.5, t - 1.0)
        gate_h = (self.z_neck + 1.0) - (z_cup + cup_h - 2)
        gate = (Manifold.cube((gate_len, gate_th, gate_h), center=True).rotate((0, 0, ang))
                .translate((q[0], q[1], z_cup + cup_h - 2 + gate_h / 2)))
        self.sprue = {"xyz": [float(q[0]), float(q[1]), float(self.z_neck)], "top_diameter": p.sprue_top_diameter,
                      "bottom_diameter": p.sprue_bottom_diameter, "gate_mm": [gate_len, gate_th]}
        return cup + gate

    # ------------------------------------------------------------------ vents
    def make_vents(self, enabled=True):
        """Trapped-air points for a pour with the head inverted (neck up): local height maxima on ceiling faces."""
        p = self.p
        up = -UP                                   # world-up in the head frame during the pour
        shell_pts = self._trapped_air(self.proxy, sign=+1, up=up) if enabled else []
        core_pts = self._trapped_air(tm_from_manifold(self.core_outer), sign=-1, up=up) if (enabled and p.core_vents and self.core_outer is not None) else []
        if not enabled:
            self.notes.append("Vents skipped (make_vents off): drill your own, or turn it back on.")
        if not p.core_vents and self.core_outer is not None:
            self.notes.append("Head blank is unpierced: no core vents (its only opening is the neck). Air escapes through the shell vents.")
        r = p.vent_diameter / 2
        cyls_shell = [cylinder_along(q - nrm * 1.0, nrm, r, r, p.shell_thickness + p.flange_width + 6, 16) for q, nrm in shell_pts]
        cyls_core = [cylinder_along(q + nrm * 1.0, -nrm, r, r, p.core_wall + 4, 16) for q, nrm in core_pts]
        self.vents += [{"xyz": q.tolist(), "on": "shell"} for q, _ in shell_pts] + [{"xyz": q.tolist(), "on": "core"} for q, _ in core_pts]
        self.log(f"{len(cyls_shell)} shell vents, {len(cyls_core)} core vents")
        if cyls_shell:
            V = Manifold.batch_boolean(cyls_shell, m3.OpType.Add)
            self.parts["shell_A"] = self.parts["shell_A"] - V
            self.parts["shell_B"] = self.parts["shell_B"] - V
        if cyls_core and "core" in self.parts:
            self.parts["core"] = self.parts["core"] - Manifold.batch_boolean(cyls_core, m3.OpType.Add)
        if p.mold_type == "puppet" and self.outline is not None:
            # the sleeve is the pour spout: no cup, no gate
            self.sprue = {"xyz": [float(self.center[0]), float(self.center[1]), float(self.z_bottom)], "pour": "through the neck sleeve"}
        # every part must be exactly one printable body: drop anything that is not the main solid
        for name in list(self.parts):
            self.parts[name] = strip_slivers(keep_components(self.parts[name], largest_only=True))

    def _trapped_air(self, t: trimesh.Trimesh, sign: int, up: np.ndarray):
        """Vertices on faces whose silicone-side outward normal (sign * mesh normal) points up, that are local
        maxima of height within vent_radius_search. Skips a band near the neck (air escapes there anyway)."""
        p = self.p
        vn = t.vertex_normals * sign
        h = t.vertices @ up
        ceiling = (vn @ up) > 0.35
        near_neck = t.vertices[:, 2] < self.z_neck + 15
        cand = np.where(ceiling & ~near_neck)[0]
        if len(cand) == 0:
            return []
        tree = cKDTree(t.vertices[cand])
        picks = []
        order = cand[np.argsort(-h[cand])]
        taken = np.zeros(len(t.vertices), dtype=bool)
        for vi in order:
            if taken[vi]:
                continue
            nb = cand[tree.query_ball_point(t.vertices[vi], p.vent_radius_search)]
            if (h[nb] > h[vi] + 1e-6).any():
                taken[vi] = True
                continue
            picks.append(vi)
            taken[nb] = True
            if len(picks) >= p.max_vents:
                break
        return [(t.vertices[vi].copy(), unit(t.vertex_normals[vi])) for vi in picks]

    def proxy_hat_circumference(self):
        from .fit import hat_line
        return hat_line(self.S)[0]

    # ------------------------------------------------------------------ analysis helpers
    def cavity_mask(self, part: trimesh.Trimesh) -> np.ndarray:
        """Faces of a part that lie on the sculpt surface (the cavity face)."""
        import igl
        c = part.triangles_center
        d = igl.signed_distance(c, np.asarray(self.proxy.vertices, dtype=np.float64), np.asarray(self.proxy.faces, dtype=np.int64),
                                sign_type=igl.SIGNED_DISTANCE_TYPE_FAST_WINDING_NUMBER)[0]
        return np.abs(d) < max(0.6, self.p.voxel_pitch * 0.6)
