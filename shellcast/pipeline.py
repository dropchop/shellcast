"""End-to-end job: ingest → SDF → shell → split/bolts/keys → core → vents → orient/estimate → validate → export."""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import manifold3d as m3
import numpy as np
import trimesh

from . import __version__
from .build import AXIS, Build
from .estimate import best_orientation, estimate_part, fmt_hms
from .fit import plan as fit_plan
from .geom import Manifold, tm_from_manifold
from .ingest import ingest, proxy_of
from .mouth import close_mouth
from .render import render_views
from .params import MoldParams

PREVIEW_FACES = 120_000

# Platinum silicones commonly used for masks. Mixed specific gravity and mix ratio by weight, from the makers'
# technical bulletins (Smooth-On). Verify against the sheet of the batch you buy.
SILICONES = {
    "fx-pro": {"name": "Dragon Skin FX-Pro", "sg": 1.062, "ratio": (1, 1), "shore": "2A", "pot_life_min": 12, "cure_min": 40},
    "dragon-skin-10": {"name": "Dragon Skin 10 Medium", "sg": 1.07, "ratio": (1, 1), "shore": "10A", "pot_life_min": 20, "cure_min": 300},
    "ecoflex-30": {"name": "Ecoflex 00-30", "sg": 1.07, "ratio": (1, 1), "shore": "00-30", "pot_life_min": 45, "cure_min": 240},
}


class Job:
    def __init__(self, mesh_path: str | Path, out_dir: str | Path, params: MoldParams | None = None, log=None, closed_path=None, open_path=None):
        self.mesh_path = Path(mesh_path)
        self.closed_path = Path(closed_path) if closed_path else None
        self.open_path = Path(open_path) if open_path else None
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.p = params or MoldParams()
        self._log = log
        self.lines: list[str] = []
        self.report: dict = {"version": __version__, "params": self.p.to_dict(), "stages": {}, "parts": {}, "notes": [], "warnings": []}
        self.t0 = time.time()

    def log(self, s: str):
        line = f"[{time.time() - self.t0:6.1f}s] {s}"
        self.lines.append(line)
        if self._log:
            self._log(line)

    def stage(self, name):
        job = self

        class _S:
            def __enter__(s):
                s.t = time.time()
                job.log(f"── {name}")
                return s

            def __exit__(s, *a):
                job.report["stages"][name] = round(time.time() - s.t, 2)
        return _S()

    # ------------------------------------------------------------------
    def run(self) -> dict:
        p = self.p
        try:
            with self.stage("ingest"):
                ing = ingest(self.mesh_path, p.rewrap_pitch, p.proxy_faces, log=self.log, force_rewrap=p.force_rewrap,
                             cavity_max_faces=p.cavity_max_faces, max_grid_cells=p.max_grid_cells, orient=p.orient)
            self.report["input"] = ing["info"]
            if not p.fit_enabled:
                p.fit_target, p.scale = "none", 1.0
            with self.stage("fit"):
                self._fit(ing)
            if p.close_mouth and p.mouth_insert:
                self.report["warnings"].append("close_mouth and mouth_insert are exclusive: the mouth was left open for the insert")
                p.close_mouth = False
            if p.close_mouth:
                with self.stage("mouth"):
                    self._mouth(ing)
            elif ing["info"].get("interior_removed"):
                fp = self.out / "sculpt_edited.obj"
                ing["mesh"].export(fp)
                self.report["edited_sculpt"] = fp.name
                self.report["notes"].append(f"Interior geometry removed: {ing['info']['interior_removed']['count']} enclosed region(s), "
                                            f"{ing['info']['interior_removed']['volume_cm3']:.0f} cm³ made solid. Cleaned sculpt saved as sculpt_edited.obj.")
            closed = None
            if p.mouth_insert and p.mouth_wedge:
                self.report["warnings"].append("mouth_insert and mouth_wedge are exclusive: making the wedge")
                p.mouth_insert = False
            if p.mouth_insert:
                with self.stage("closed"):
                    closed = self._second(ing, self.closed_path, "closed-mouth", auto_close=True)
            elif p.mouth_wedge:
                with self.stage("open"):
                    closed = self._second(ing, self.open_path, "open-mouth", auto_close=False)
            b = Build(ing["manifold"], ing["proxy"], p, log=self.log)
            if closed is not None:
                b.closed = closed
                b.closed_proxy = proxy_of(tm_from_manifold(closed), p.proxy_faces)
            ft = self.report.get("fit") or {}
            if ft.get("wanted_inner_circumference_mm"):
                b.blank_circumference = float(ft["wanted_inner_circumference_mm"])     # the wearer's head, with ease
            with self.stage("sdf"):
                b.make_grid()
            with self.stage("shell"):
                b.make_shell()
            with self.stage("split"):
                b.split_shell()
            if p.mouth_wedge:
                with self.stage("void"):
                    b.prepare_wedge()
            if p.make_core:
                with self.stage("core"):
                    b.make_core()
            else:
                b.core_outer = None
                self.report["notes"].append("Head blank skipped (make_core off): shells only; no mask-thickness or silicone figures.")
            if p.mouth_wedge:
                with self.stage("wedge"):
                    b.make_wedge()
            with self.stage("vents"):
                b.make_vents(enabled=p.make_vents)
            self.report["notes"] += b.notes
            self.report["bolts"] = b.bolts
            self.report["vents"] = b.vents
            self.report["sprue"] = getattr(b, "sprue", None)
            if getattr(b, "blank_info", None):
                self.report["head_blank"] = b.blank_info
            if ing["info"].get("interior_removed"):
                self.report["interior_removed"] = ing["info"]["interior_removed"]
            self.report["frame"] = {"neck_z": b.z_neck, "parting_plane_point": b.plane_point.tolist(), "parting_normal": b.n.tolist()}
            with self.stage("validate"):
                self._validate(b)
            with self.stage("estimate+export"):
                self._orient_estimate_export(b)
            if not p.make_previews:
                self.report["notes"].append("3D previews skipped (make_previews off).")
            if p.make_previews:
                with self.stage("preview"):
                    self._previews(b, ing)
            self._block_comparison(b, ing)
            self.report["status"] = "done"
        except Exception as e:  # noqa: BLE001
            self.log("ERROR " + "".join(traceback.format_exception_only(type(e), e)).strip())
            self.report["status"] = "error"
            self.report["error"] = f"{type(e).__name__}: {e}"
            self.report["traceback"] = traceback.format_exc()
        self.report["log"] = self.lines
        self.report["elapsed_s"] = round(time.time() - self.t0, 1)
        (self.out / "report.json").write_text(json.dumps(self.report, indent=2, default=_json_default))
        return self.report

    # ------------------------------------------------------------------
    def _fit(self, ing):
        """Measure the hat line and, if asked, rescale sculpt/mesh/proxy uniformly so the mask fits the target head."""
        p = self.p
        rep = fit_plan(ing["manifold"], p.mask_thickness, p.fit_target, p.fit_circumference_mm, p.fit_ease_pct, p.scale)
        self.report["fit"] = rep
        f = rep["scale"]
        self.log(f"hat line: sculpt {rep['sculpt_hat_circumference_mm']:.0f} mm → mask inner {rep['mask_inner_circumference_mm']:.0f} mm"
                 + (f"; target {rep['target_circumference_mm']:.0f} mm ({rep['ease_pct']:+.0f} % ease) → scale ×{f:.3f}" if rep.get("target_circumference_mm") else
                    (f"; manual scale ×{f:.3f}" if abs(f - 1) > 1e-9 else "")))
        if abs(f - 1.0) < 1e-9:
            return
        ing["manifold"] = ing["manifold"].scale((f, f, f))
        ing["mesh"] = ing["mesh"].copy()
        ing["mesh"].apply_scale(f)
        ing["proxy"] = ing["proxy"].copy()
        ing["proxy"].apply_scale(f)
        ing["info"]["scaled_extents"] = ing["proxy"].extents.tolist()
        ing["info"]["volume_mm3"] = float(ing["manifold"].volume())

    def _second(self, ing, path, label, auto_close):
        """A second version of the same sculpt, in the working frame: the user's file registered onto the working
        sculpt (the closed-mouth edit for the mouth cover, or the open-mouth original for the mouth wedge); for the
        cover, the automatic lip closure when no file was given."""
        p = self.p
        if path is not None:
            from .register import register
            ic = ingest(path, p.rewrap_pitch, p.proxy_faces, log=self.log, force_rewrap=p.force_rewrap,
                        cavity_max_faces=p.cavity_max_faces, max_grid_cells=p.max_grid_cells, orient=p.orient)
            try:
                M, rms, scale = register(ic["proxy"], ing["proxy"], log=self.log)
            except ValueError as e:
                self.report["warnings"].append(f"{label} sculpt could not be registered ({e})" + ("; using the automatic closure for the cover" if auto_close else ""))
                self.log(f"registration failed: {e}")
            else:
                self.report["second_sculpt"] = {"file": path.name, "role": label, "registration_rms_mm": rms, "scale": scale,
                                                "triangles": int(len(ic["mesh"].faces))}
                if auto_close:
                    self.report["closed_mouth"] = self.report["second_sculpt"]
                if rms > 1.0:
                    self.report["warnings"].append(f"{label} sculpt registered with rms {rms:.2f} mm: is it the same head?")
                S = ic["manifold"].transform(np.ascontiguousarray(M[:3, :], dtype=np.float64))
                self.report["notes"].append(f"{label.capitalize()} sculpt {path.name} registered onto the working one (rms {rms:.2f} mm, scale ×{scale:.3f}).")
                return S
        if not auto_close:
            self.report["warnings"].append(f"no {label} sculpt was given: upload the open-mouth version as the second file for the wedge")
            return None
        try:
            S2, t2, rep = close_mouth(ing["manifold"], ing["mesh"], radius=p.mouth_radius, pitch=1.0, mode=p.mouth_mode, log=self.log)
            self.report["mouth"] = rep
            fp = self.out / "sculpt_closed.obj"
            t2.export(fp)
            self.report["closed_mouth"] = {"file": fp.name, "auto": True}
            self.report["notes"].append(f"Closed mouth for the cover made automatically ({rep['mode']}); saved as sculpt_closed.obj. "
                                        "Upload your own closed-mouth version for a sculpted result.")
            return S2
        except Exception as e:  # noqa: BLE001
            self.report["warnings"].append(f"automatic mouth closure failed ({e}); the cover bridges the lips smoothly instead")
            return None

    def _mouth(self, ing):
        p = self.p
        before = ing["proxy"]
        S2, t2, rep = close_mouth(ing["manifold"], ing["mesh"], radius=p.mouth_radius, pitch=1.0, mode=p.mouth_mode, log=self.log)
        ing["manifold"], ing["mesh"] = S2, t2
        ing["proxy"] = proxy_of(t2, p.proxy_faces)
        ing["info"]["volume_mm3"] = float(S2.volume())
        self.report["mouth"] = rep
        fp = self.out / "sculpt_edited.obj"
        t2.export(fp)
        self.report["edited_sculpt"] = fp.name
        prev = self.out / "preview"
        prev.mkdir(exist_ok=True)
        try:
            render_views(before, ing["proxy"], prev / "mouth.png", crop=rep["box"])
        except Exception as e:  # noqa: BLE001
            self.log(f"(preview render skipped: {e})")
        self.log(f"mouth closed ({rep['mode']}): {rep['width_mm']:.0f} mm wide, gap {rep['mean_gap_mm']:.0f} mm → sculpt_edited.obj")

    def _validate(self, b: Build):
        v = {}
        for name, m in b.parts.items():
            ok = m.status() == m3.Error.NoError and not m.is_empty() and m.volume() > 0
            v[name] = {"manifold": bool(ok), "volume_mm3": float(m.volume()), "triangles": int(m.num_tri()), "genus": int(m.genus()),
                       "bodies": sum(1 for c in m.decompose() if c.volume() > 1.0)}
            if not ok:
                self.report["warnings"].append(f"{name} failed manifold validation ({m.status()})")
        for nm in ("mouth_insert", "mouth_cover", "mouth_wedge"):
            if nm in b.parts:
                inter = (b.parts["shell_A"] + b.parts["shell_B"]) ^ b.parts[nm]
                v[f"{nm.split('_')[1]}_shell_interference_mm3"] = float(inter.volume())
                if inter.volume() > 5.0:
                    self.report["warnings"].append(f"{nm.replace('_', ' ')} intersects the shell by {inter.volume():.1f} mm³")
                if "core" in b.parts:
                    ic = b.parts["core"] ^ b.parts[nm]
                    v[f"{nm.split('_')[1]}_core_interference_mm3"] = float(ic.volume())
                    if ic.volume() > 5.0:
                        self.report["warnings"].append(f"{nm.replace('_', ' ')} intersects the head blank by {ic.volume():.1f} mm³")
        if self.p.mold_type == "puppet":
            cav = b.S.trim_by_plane((0, 0, 1), b.z_neck)
            spout = Manifold.extrude(b.outline, self.p.collar_height) if b.outline is not None else None
            vol = float(cav.volume()) + (float(spout.volume()) if spout is not None else 0.0)
            if "neck_plug" in b.parts:
                inter = (b.parts["shell_A"] + b.parts["shell_B"]) ^ b.parts["neck_plug"]
                v["plug_shell_interference_mm3"] = float(inter.volume())
                rod = np.pi * (self.p.rod_diameter / 2) ** 2 * (self.p.rod_depth + self.p.collar_height)
                vol -= min(vol, rod)
            v["silicone_volume_mm3"] = vol
            self._silicone(vol, b)
            skin = 6.0
            self.report["silicone"]["slush_skin_mm"] = skin
            self.report["silicone"]["slush_volume_ml"] = float(cav.surface_area() * skin / 1000.0)
            self.report["silicone"]["slush_mix_g"] = self.report["silicone"]["slush_volume_ml"] * self.report["silicone"]["sg"]
            v["silicone_mix_g"] = self.report["silicone"]["mix_g"]
        elif "core" in b.parts:
            inter = (b.parts["shell_A"] + b.parts["shell_B"]) ^ b.parts["core"]
            v["core_shell_interference_mm3"] = float(inter.volume())
            if inter.volume() > 5.0:
                self.report["warnings"].append(f"core intersects the shell by {inter.volume():.1f} mm³")
            cav = b.S.trim_by_plane((0, 0, 1), b.z_neck)
            if b.mouth_void is not None and b.block is not None:
                # cover in: the cavity is the open sculpt plus whatever of the window block the closed sculpt occupies
                extra = (b.closed ^ b.block) if b.closed is not None else b.mouth_void
                cav = cav + extra.trim_by_plane((0, 0, 1), b.z_neck)
            sil = cav - b.core_outer.trim_by_plane((0, 0, 1), b.z_neck)
            if b.mouth_wedge is not None:
                sil = sil - b.mouth_wedge                       # wedge in: the mouth interior is hollow
                v["mouth_hollow_mm3"] = float((b.mouth_wedge ^ cav).volume())
                # pull-out check: the rigid wedge must slide out of the blank's pocket toward the face (−y) freely
                worst = 0.0
                for s_mm in (2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 140.0):
                    hit = (b.mouth_wedge.translate((0, -s_mm, 0)) ^ b.parts["core"]).volume()
                    worst = max(worst, float(hit))
                v["wedge_pullout_interference_mm3"] = worst
                if worst > 5.0:
                    self.report["warnings"].append(f"the wedge cannot slide out of the blank along the pull axis (hits it by up to {worst:.0f} mm³)")
            v["silicone_volume_mm3"] = float(sil.volume())
            self._silicone(float(sil.volume()), b)
            v["silicone_mix_g"] = self.report["silicone"]["mix_g"]
            if b.mouth_void is not None and b.block is not None:
                # insert in, blank out, sleeve as the spout: the same shells cast a solid or slush puppet head
                pcav = b.S.trim_by_plane((0, 0, 1), b.z_neck)
                spout = float(Manifold.extrude(b.outline, self.p.collar_height).volume()) if b.outline is not None else 0.0
                sg = self.report["silicone"]["sg"]; skin = 6.0
                omask = pcav - b.core_outer.trim_by_plane((0, 0, 1), b.z_neck)
                self.report["silicone"]["open_mouth_mask"] = {"note": "insert in, head blank in", "volume_ml": float(omask.volume()) / 1000.0,
                                                              "mix_g": float(omask.volume()) * self.report["silicone"]["sg"] / 1000.0}
                self.report["silicone"]["puppet"] = {
                    "note": "insert in, head blank out, pour through the neck sleeve",
                    "solid_volume_ml": (float(pcav.volume()) + spout) / 1000.0,
                    "solid_mix_g": (float(pcav.volume()) + spout) * sg / 1000.0,
                    "slush_skin_mm": skin,
                    "slush_volume_ml": float(pcav.surface_area() * skin / 1000.0),
                    "slush_mix_g": float(pcav.surface_area() * skin / 1000.0) * sg,
                }
        self.report["validation"] = v

    def _silicone(self, mask_mm3: float, b: Build):
        if not self.p.estimate_silicone:
            self.report["silicone"] = {"skipped": True, "sg": 1.0, "mix_g": 0.0, "mask_volume_ml": mask_mm3 / 1000.0}
            return
        self._silicone_full(mask_mm3, b)

    def _silicone_full(self, mask_mm3: float, b: Build):
        """Silicone per mask: the cast itself, plus what the cup/gate hold, plus an overage for mixing losses and
        vent bleed. Reported as volume and as weight of mixed rubber split into parts A and B."""
        p = self.p
        spec = SILICONES.get(p.silicone) or {"name": f"custom (SG {p.silicone_sg})", "sg": p.silicone_sg, "ratio": (1, 1), "shore": "?", "pot_life_min": None, "cure_min": None}
        sg = spec["sg"]
        # cup + gate volume (what stays in the sprue after the pour)
        cup_h = p.plate_thickness + 8
        r1, r2 = p.sprue_top_diameter / 2, p.sprue_bottom_diameter / 2
        cup = np.pi * cup_h / 3 * (r1 * r1 + r1 * r2 + r2 * r2)
        gate = 0.0
        if b.sprue and b.sprue.get("gate_mm"):
            gl, gt = b.sprue["gate_mm"]
            gate = gl * gt * (p.collar_height + 2)
        sprue_mm3 = cup + gate if (b.sprue and b.sprue.get("gate_mm")) else 0.0
        pour_mm3 = mask_mm3 + sprue_mm3
        mix_g = pour_mm3 * sg / 1000.0
        order_g = mix_g * (1 + p.pour_overage_pct / 100.0)
        a, bb = spec["ratio"]
        self.report["silicone"] = {
            "product": spec["name"], "sg": sg, "shore": spec["shore"], "mix_ratio_by_weight": f"{a}A:{bb}B",
            "pot_life_min": spec["pot_life_min"], "cure_min": spec["cure_min"],
            "mask_volume_ml": mask_mm3 / 1000.0, "mask_weight_g": mask_mm3 * sg / 1000.0,
            "sprue_volume_ml": sprue_mm3 / 1000.0, "pour_volume_ml": pour_mm3 / 1000.0,
            "mix_g": mix_g, "part_a_g": mix_g * a / (a + bb), "part_b_g": mix_g * bb / (a + bb),
            "overage_pct": p.pour_overage_pct, "order_g": order_g,
            "order_part_a_g": order_g * a / (a + bb), "order_part_b_g": order_g * bb / (a + bb),
            "mask_thickness_mm": p.mask_thickness,
            "cavity_area_cm2": float(b.S.trim_by_plane((0, 0, 1), b.z_neck).surface_area() / 100.0),
        }

    def _orient_estimate_export(self, b: Build):
        p = self.p
        total_s, total_g = 0.0, 0.0
        for name, m in b.parts.items():
            t = tm_from_manifold(m)
            if p.estimate_print:
                cavity = b.cavity_mask(t) if name.startswith("shell") else None
                u, est = best_orientation(t, name, AXIS[p.pull_axis], p.printer, cavity)
            else:
                u = t
                ext = t.extents
                est = {"volume_mm3": float(t.volume), "filament_g": 0.0, "support_g": 0.0, "time_s_geom": 0.0, "orientation": "as built (estimate off)",
                       "transform": np.eye(4).tolist(),   # exported as built, so the viewer's assembled view needs no move
                       "cavity_support_area_mm2": 0.0, "fits_bed": True, "footprint_mm": [float(ext[0]), float(ext[1])], "height_mm": float(ext[2]), "layers": 0}
            files = {}
            if not (p.export_obj or p.export_stl or p.export_3mf):
                p.export_obj = True
            for fmt, on in (("obj", p.export_obj), ("stl", p.export_stl), ("3mf", p.export_3mf)):
                if on:
                    fp = self.out / f"{name}.{fmt}"
                    u.export(fp)
                    files[fmt] = fp.name
            est["files"] = files
            est["time_geom"] = fmt_hms(est["time_s_geom"])
            self.report["parts"][name] = est
            total_s += est["time_s_geom"]
            total_g += est["filament_g"]
            self.log(f"{name}: {est['orientation']}, {est['filament_g']:.0f} g, {est['time_geom']} (geom), cavity support {est['cavity_support_area_mm2']:.0f} mm²")
        self.report["totals"] = {"time_s_geom": total_s, "time_geom": fmt_hms(total_s), "filament_g": total_g}
        pr = p.printer
        self.report["print_profile"] = {"shell_thickness": p.shell_thickness, "perimeters": round(p.shell_thickness / pr.line_width, 1),
                                        "flange_width": p.flange_width, "flange_thickness": p.flange_thickness,
                                        "bolts": len([b for b in self.report["bolts"] if not b.get("ring")]),
                                        "keys": b.n_keys}

    def _previews(self, b: Build, ing):
        prev = self.out / "preview"
        prev.mkdir(exist_ok=True)
        for name, m in b.parts.items():
            t = tm_from_manifold(m)
            if len(t.faces) > PREVIEW_FACES:
                t = t.simplify_quadric_decimation(face_count=PREVIEW_FACES)
            t.export(prev / f"{name}.glb")
        s = ing["proxy"]
        if len(s.faces) > PREVIEW_FACES:
            s = s.simplify_quadric_decimation(face_count=PREVIEW_FACES)
        s.export(prev / "sculpt.glb")

    def _block_comparison(self, b: Build, ing):
        """What a conventional block mold of the same head would cost, for the report."""
        p = self.p
        ext = np.asarray(ing["proxy"].extents) + 2 * (p.shell_thickness + p.flange_width)
        block = float(np.prod(ext)) - ing["info"]["volume_mm3"]
        pr = p.printer
        printed = block * 0.20 + (2 * (ext[0] * ext[1] + ext[1] * ext[2] + ext[0] * ext[2])) * 3 * 0.8  # 20 % infill + ~3 walls
        self.report["block_mold_comparison"] = {"block_volume_mm3": block, "printed_mm3_est": printed,
                                                "filament_g_est": printed * pr.density_g_cm3 / 1000,
                                                "time_s_est": printed / pr.flow_mm3_s + (ext[2] / pr.layer_height) * pr.layer_overhead_s,
                                                # None with the print estimate off: there is no filament figure to compare against
                                                "saving_factor_filament": ((printed * pr.density_g_cm3 / 1000) / self.report["totals"]["filament_g"]
                                                                           if self.report["totals"]["filament_g"] > 0 else None)}

def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def run(mesh_path, out_dir, params: MoldParams | dict | None = None, log=print, closed_path=None, open_path=None) -> dict:
    if isinstance(params, dict):
        params = MoldParams.from_dict(params)
    return Job(mesh_path, out_dir, params, log=log, closed_path=closed_path, open_path=open_path).run()
