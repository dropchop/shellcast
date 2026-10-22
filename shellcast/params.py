"""Parameters for a ShellCast job. Everything is in millimetres, grams and seconds.

Head frame convention: +Z is the crown, the neck opening faces -Z, the face looks
toward -Y. The pull axis chooses the parting plane normal: "x" gives left/right
halves (seam over the crown and under the chin), "y" gives front/back halves.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields

_TRUE, _FALSE = ("1", "true", "yes", "on"), ("0", "false", "no", "off")


def _coerce(name, t, v):
    """v as type t (what a form, a JSON file or a -p key=value gives); ValueError when it cannot be read as one."""
    if t is bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, int) and v in (0, 1):
            return bool(v)
        if isinstance(v, str) and v.strip().lower() in _TRUE + _FALSE:
            return v.strip().lower() in _TRUE
        raise ValueError(f"{name} must be true or false (got {v!r})")
    if t is int:
        if isinstance(v, int) and not isinstance(v, bool):
            return v
        if isinstance(v, float) and v.is_integer():
            return int(v)
        if isinstance(v, str):
            try:
                return int(v.strip())
            except ValueError:
                pass
        raise ValueError(f"{name} must be a whole number (got {v!r})")
    if t is float:
        if isinstance(v, (int, float, str)) and not isinstance(v, bool):
            try:
                x = float(v)
            except ValueError:
                x = math.nan
            if math.isfinite(x):
                return x
        raise ValueError(f"{name} must be a number (got {v!r})")
    if t is str and isinstance(v, (str, int, float)) and not isinstance(v, bool):
        return str(v)
    raise ValueError(f"{name} must be text (got {v!r})")


def _coerce_fields(cls, d, prefix=""):
    """The fields of dataclass cls from d, each coerced to its default's type. An unknown key is an error, not dropped."""
    types = {f.name: type(f.default) for f in fields(cls) if f.name != "printer"}
    unknown = sorted(set(d) - set(types))
    if unknown:
        raise ValueError(f"unknown parameter {prefix}{unknown[0]} (after an update, restart the server)")
    return {k: _coerce(prefix + k, types[k], v) for k, v in d.items()}


@dataclass
class PrinterProfile:
    name: str = "large-format 0.6 nozzle"
    nozzle: float = 0.6
    line_width: float = 0.7            # extrusion width the slicer will use
    layer_height: float = 0.3
    flow_mm3_s: float = 15.0           # sustained volumetric flow you actually get (calibrate!)
    layer_overhead_s: float = 3.0      # z-hop, retract, wipe, min layer time per layer
    travel_mm_s: float = 200.0
    bed_x: float = 400.0
    bed_y: float = 400.0
    bed_z: float = 400.0
    density_g_cm3: float = 1.24        # PLA; PETG ≈ 1.27
    support_efficiency: float = 0.7    # support prints at this fraction of the part's flow (thin lines, fast)
    support_density: float = 0.10      # printed fraction of the support region (tree/organic supports are sparse)
    infill_pct: float = 12.0           # sparse infill inside thick regions (the slicer prints walls + this)
    wall_lines: int = 4                # perimeter count the slicer will use (wall = wall_lines × line_width)


@dataclass
class MoldParams:
    # --- what the mold is for ---
    mold_type: str = "mask"            # mask: shells + head blank (core) | puppet: shells only, solid pour or slush cast
    puppet_cavity: str = "none"        # none: solid casting | rod: a printed neck plug with a rod that leaves an armature hole
    rod_diameter: float = 12.0
    rod_depth: float = 120.0           # how far the rod reaches into the head above the neck plane
    mouth_insert: bool = False         # open-mouth sculpts: the mouth void becomes a removable insert (a stepped window
                                       # plug in the shell around the lips). In = open mouth; out = the mouth fills with silicone.
    insert_clearance: float = 0.25
    # --- compute switches: turn stages off to save time ---
    make_core: bool = True             # head blank (core). Off: shells only (no mask-thickness check, no silicone volume)
    fit_enabled: bool = True           # off: ignore fit_target (no rescale)
    estimate_silicone: bool = True     # off: no silicone block
    estimate_print: bool = True        # off: no orientation search / print-time estimate; parts export as built
    make_vents: bool = True            # off: no vent search or holes
    make_previews: bool = True         # off: no GLB previews for the viewer
    mouth_wedge: bool = False          # closed-mouth sculpt + open-mouth sculpt (--open): the open mouth's void becomes a textured
                                       # wedge core that sits one mask thickness behind the closed lips, pegged to the head blank.
                                       # The mask casts with sealed lips and a hollow mouth (silicone teeth/tongue); slit the lips to open it.
    wedge_peg_diameter: float = 8.0
    wedge_boss_diameter: float = 30.0
    # --- orientation (see orient.py): "auto", "keep", or "<up>,<front>" e.g. "+y,+z" ---
    orient: str = "auto"
    # --- sculpt edits (see mouth.py) ---
    close_mouth: bool = False
    mouth_radius: float = 15.0         # closing radius that must bridge the lip gap (fills openings up to 2×)
    mouth_mode: str = "auto"           # auto | pinch (parted lips) | hinge (open jaw rotates up)
    # --- fit (rescale the sculpt so the mask fits a real head; see fit.py) ---
    fit_target: str = "none"           # none | male | female | custom
    fit_circumference_mm: float = 0.0  # head circumference over the brows for fit_target=custom
    fit_ease_pct: float = -2.0         # mask inner circumference relative to the head: negative = snug
    scale: float = 1.0                 # extra manual scale factor, applied on top
    # --- parting ---
    pull_axis: str = "x"               # "x" (left/right halves) or "y" (front/back)
    parting_offset: float = 0.0        # shift of the parting plane along the pull axis from the mesh centre
    neck_offset: float = 0.0           # neck plane = z_min + neck_offset (cut everything below)
    # --- shell ---
    shell_thickness: float = 3.5       # w
    cavity_clearance: float = 0.0      # c (0 = the cavity face is the sculpt itself)
    flange_width: float = 20.0         # F
    flange_thickness: float = 6.0      # T per half
    bolt_diameter: float = 5.0         # M5
    bolt_clearance: float = 0.5
    bolt_spacing: float = 70.0
    key_radius: float = 4.0            # conical registration key base radius
    key_clearance: float = 0.3
    # --- core ---
    mask_thickness: float = 5.0        # t: silicone wall
    core_wall: float = 2.5             # k: hollow core wall
    core_shape: str = "sculpt"         # sculpt: the inside follows the sculpt (offset by t) | head: a generic smoothed
                                       # human head form, sized to the fit target — the mask is solid wherever the sculpt is bigger
    core_smooth_mm: float = 0.0        # >0: the inside of the mask (the core) is smoothed to a soft head shape over this radius;
                                       # the mask stays at least mask_thickness thick everywhere and gets thicker in the details
    core_close_radius: float = 6.0     # fill core slots/folds narrower than 2r (nostrils, ear canals, deep wrinkles)
    core_open_radius: float = 4.0      # remove core fins thinner than 2r (thin ears → solid silicone)
    core_roof_angle: float = 0.0       # >0: cap the hollow with a conical "cathedral" roof at this angle (costs volume; off = print neck-up)
    collar_height: float = 25.0        # neck sleeve / core collar height below the neck plane
    collar_clearance: float = 0.15
    plate_thickness: float = 6.0       # core collar plate / tabs (bolt to the shell's neck ring)
    plate_tabs: int = 4                # bolt tabs instead of a full plate (0 = full plate)
    # --- sprue & vents ---
    sprue_top_diameter: float = 25.0
    sprue_bottom_diameter: float = 12.0
    core_vents: bool = False           # bleeder holes through the head blank into its hollow (off: the blank is solid
                                       # skin, its only opening is the neck; air escapes through the shell vents)
    vent_diameter: float = 2.5
    vent_radius_search: float = 15.0   # local-maximum neighbourhood for trapped-air detection
    max_vents: int = 12
    # --- resolution ---
    voxel_pitch: float = 1.5           # SDF grid pitch for shell skin, flange and core
    rewrap_pitch: float = 0.75         # used only when the sculpt isn't manifold (or force_rewrap)
    force_rewrap: bool = False         # always rewrap (use for sculpts that self-intersect but pass the manifold gate)
    proxy_faces: int = 200_000         # decimated proxy used for SDF queries, vents and analysis
    cavity_max_faces: int = 1_000_000  # sculpt is decimated to this many triangles before anything else (memory + time bound)
    job_timeout_min: int = 30          # a job is killed after this long
    max_grid_cells: int = 40_000_000   # the SDF grid pitch is coarsened automatically to stay under this
    smooth_iterations: int = 3         # laplacian smoothing of iso-surfaces (fewer micro-features = faster print)
    # --- silicone estimate (see pipeline._silicone) ---
    silicone: str = "fx-pro"           # fx-pro | dragon-skin-10 | ecoflex-30 | custom
    silicone_sg: float = 1.062         # specific gravity used for "custom"
    pour_overage_pct: float = 15.0     # extra mixed to cover cup, mixing losses and bleed at the vents
    # --- export (plain geometry; all print settings are chosen in the slicer, e.g. OrcaSlicer) ---
    export_obj: bool = True
    export_stl: bool = False
    export_3mf: bool = False
    printer: PrinterProfile = field(default_factory=PrinterProfile)

    @classmethod
    def from_dict(cls, d: dict | None) -> "MoldParams":
        if d is not None and not isinstance(d, dict):
            raise ValueError("params must be a JSON object")
        d = dict(d or {})
        pr = d.pop("printer", None) or {}
        if not isinstance(pr, dict):
            raise ValueError("printer must be a JSON object")
        p = cls(**_coerce_fields(cls, d))
        p.printer = PrinterProfile(**_coerce_fields(PrinterProfile, pr, "printer."))
        p.validate()
        return p

    def validate(self):
        if self.pull_axis not in ("x", "y"):
            raise ValueError("pull_axis must be 'x' or 'y'")
        if self.silicone not in ("fx-pro", "dragon-skin-10", "ecoflex-30", "custom"):
            raise ValueError("silicone must be fx-pro, dragon-skin-10, ecoflex-30 or custom")
        if not (0.8 <= self.silicone_sg <= 1.6) or not (0 <= self.pour_overage_pct <= 100):
            raise ValueError("silicone_sg must be 0.8–1.6 and pour_overage_pct 0–100")
        if self.mouth_mode not in ("auto", "pinch", "hinge"):
            raise ValueError("mouth_mode must be auto, pinch or hinge")
        if not (3 <= self.mouth_radius <= 60):
            raise ValueError("mouth_radius must be 3–60 mm")
        if self.fit_target not in ("none", "male", "female", "custom"):
            raise ValueError("fit_target must be none, male, female or custom")
        if self.fit_target == "custom" and not (300 <= self.fit_circumference_mm <= 900):
            raise ValueError("fit_circumference_mm must be 300–900 mm for a custom fit")
        if self.mold_type not in ("mask", "puppet"):
            raise ValueError("mold_type must be mask or puppet")
        if self.puppet_cavity not in ("none", "rod"):
            raise ValueError("puppet_cavity must be none or rod")
        if self.core_shape not in ("sculpt", "head"):
            raise ValueError("core_shape must be sculpt or head")
        if not (0 <= self.core_smooth_mm <= 40):
            raise ValueError("core_smooth_mm must be 0–40 mm")
        if not (0.2 <= self.scale <= 5):
            raise ValueError("scale must be between 0.2 and 5")
        if not (-15 <= self.fit_ease_pct <= 15):
            raise ValueError("fit_ease_pct must be between -15 and 15")
        for name, lo, hi in (("shell_thickness", 1.5, 15), ("mask_thickness", 1.5, 20), ("flange_width", 4, 60), ("flange_thickness", 2, 15),
                             ("bolt_diameter", 2, 12), ("key_radius", 1.5, 10),
                             ("voxel_pitch", 0.4, 5), ("rewrap_pitch", 0.3, 3), ("core_wall", 1.2, 10),
                             ("bolt_spacing", 20, 300), ("collar_height", 0, 80)):
            v = getattr(self, name)
            if not (lo <= v <= hi):
                raise ValueError(f"{name} must be between {lo} and {hi} mm (got {v})")
        # every other number, so a request cannot ask for a negative gap, a zero hole or an unbounded job; the last
        # four are the resource limits a job runs under (memory and time), so their upper bounds are hard caps
        for name, lo, hi in (("rod_diameter", 3, 60), ("rod_depth", 10, 400), ("insert_clearance", 0, 3),
                             ("wedge_peg_diameter", 2, 30), ("wedge_boss_diameter", 5, 80), ("fit_circumference_mm", 0, 900),
                             ("parting_offset", -200, 200), ("neck_offset", 0, 500), ("cavity_clearance", 0, 5),
                             ("bolt_clearance", 0, 3), ("key_clearance", 0, 2), ("core_close_radius", 0, 30),
                             ("core_open_radius", 0, 30), ("core_roof_angle", 0, 80), ("collar_clearance", 0, 3),
                             ("plate_thickness", 2, 20), ("plate_tabs", 0, 12), ("sprue_top_diameter", 5, 80),
                             ("sprue_bottom_diameter", 3, 60), ("vent_diameter", 0.8, 10), ("vent_radius_search", 2, 100),
                             ("max_vents", 0, 100), ("smooth_iterations", 0, 20),
                             ("proxy_faces", 10_000, 2_000_000), ("cavity_max_faces", 10_000, 10_000_000),
                             ("max_grid_cells", 1_000_000, 150_000_000), ("job_timeout_min", 1, 240)):
            v = getattr(self, name)
            if not (lo <= v <= hi):
                raise ValueError(f"{name} must be between {lo} and {hi} (got {v})")
        pr = self.printer
        for name, lo, hi in (("nozzle", 0.1, 2), ("line_width", 0.1, 3), ("layer_height", 0.04, 1.5), ("flow_mm3_s", 0.5, 200),
                             ("layer_overhead_s", 0, 60), ("travel_mm_s", 10, 1000), ("bed_x", 50, 2000), ("bed_y", 50, 2000),
                             ("bed_z", 50, 2000), ("density_g_cm3", 0.5, 3), ("support_efficiency", 0.05, 1),
                             ("support_density", 0, 1), ("infill_pct", 0, 100), ("wall_lines", 1, 20)):
            v = getattr(pr, name)
            if not (lo <= v <= hi):
                raise ValueError(f"printer.{name} must be between {lo} and {hi} (got {v})")

    def to_dict(self) -> dict:
        return asdict(self)
