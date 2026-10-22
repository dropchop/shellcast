# ShellCast

Print-speed-first mold shell generator for silicone masks and prosthetics. Give it a sculpted head; get
back a printable mold set: two conformal shell halves with bolted flanges and registration keys, a hollow
core that sets the mask thickness, a pour cup with a gate, and bleeder vents — with a filament and print-time
estimate for each part and a comparison against a block mold.

Web UI on port 5070 (`./run.sh`, or the `shellcast` user service). Everything also runs headless from the
CLI, which is what the tests use.

## Install

Python 3.10 or newer on Linux (macOS should work). Only the systemd unit (`shellcast.service`) gives jobs a
memory ceiling; `./run.sh` and `shellcast serve` have none (see Hardware limits below).

```
git clone https://github.com/dropchop/shellcast.git
cd shellcast
pip install -e ".[dev]"
./run.sh            # web UI on http://localhost:5070
```

The web UI has no login. It listens on localhost only, unless you start it with `HOST=0.0.0.0 ./run.sh` or
`shellcast serve --host 0.0.0.0`. Then anyone on your network can upload sculpts, run jobs and delete them.

## Usage

```
python3 -m shellcast fixture head fixtures/head.stl          # analytic test head (nose, ears, chin, neck)
python3 -m shellcast info  sculpt.obj                         # units, size, manifold status
python3 -m shellcast run   sculpt.obj --out jobs/one \
        -p pull_axis=y -p shell_thickness=4 -p printer.flow_mm3_s=22 [-p export_stl=true]
python3 -m shellcast serve --port 5070
python3 -m pytest -q                                          # ~15 min on 8 cores
python3 -m pytest -q -m "not slow"                            # the fast subset: the core pipeline runs, no variants
```

## What comes out

`report.json` plus, per part, a plain **OBJ** (full precision, no embedded settings — choose nozzle, walls,
supports and speeds in OrcaSlicer), already rotated into the recommended print orientation and sitting on the
bed, plus GLB previews for the viewer. STL and 3MF are optional extra exports (`export_stl`, `export_3mf`).
FBX is not offered: nothing in the stack writes it, and slicers read OBJ directly.

| Part | What it is |
|---|---|
| `shell_A`, `shell_B` | 3.5 mm conformal skins whose cavity face *is* the sculpt (exact boolean), parting flanges on the pull-axis plane with M5 clearance holes every 70 mm and 45° conical keys between bolts, a neck sleeve, and a flat ring at the sleeve bottom |
| `core` | the sculpt offset inward by the mask thickness, hollowed to a 2.5 mm wall, with narrow folds filled (closing) and thin fins removed (opening); a collar plug that fills the neck sleeve to register it; bolt tabs that bolt to the shell's ring; the pour cup and gate cut through the collar; vents into the hollow |

Head frame: +Z crown, neck at −Z, face toward −Y. `pull_axis=x` gives left/right halves (seam over the crown),
`pull_axis=y` front/back halves.

## How it works (one paragraph per stage)

1. **Ingest.** trimesh loads OBJ/STL/PLY/3MF/GLB; units are guessed from the size (a head is 150–350 mm; cm, m and
   Nomad/ZBrush scene units a few hundredths across are scaled up, the latter to a nominal 250 mm — a fit target then sets the true size). The
   mesh is cleaned, then gated through manifold3d. Overlapping closed bodies (ZBrush subtools) are unioned. If
   it still isn't boolean-ready it is *rewrapped*: a signed-distance grid from fast winding numbers (works on
   soups) → iso-surface at 0, with a deviation report. `force_rewrap=1` for sculpts that pass the gate but
   self-intersect.
2. **Neck extension.** The sculpt is extended below the neck plane with a prism of its neck outline. Every offset
   surface then continues down that prism, so the shell's skin becomes the sleeve, the flange band follows it and
   the core becomes the collar — no special neck geometry.
3. **SDF.** One signed-distance grid (default 1.5 mm) on a decimated proxy (200k triangles), via libigl. Every
   non-visible surface (outer skin, flange skins, core) is an iso-surface of this grid, extracted with marching
   cubes and lightly smoothed. The cavity face is carved from the full-resolution sculpt with an exact boolean.
4. **Shell.** skin(w) − sculpt; flange = (skin(w+F) − skin(w)) ∩ a slab of 2·T about the parting plane; split by
   the plane; bolts along the flange midline (arc-length spaced); keys between bolts; ring bolts around the neck.
5. **Core.** iso(−t) after morphological closing (fills slots narrower than 2·6 mm) and opening (removes fins
   thinner than 2·4 mm), both done exactly (offset iso-surface → fresh SDF), hollowed at −k. Collar, tabs,
   cup + gate (a thin slot centred in the silicone gap so it can't break into the hollow), vents.
6. **Vents.** For a pour with the head inverted: vertices on "ceiling" faces of the silicone volume that are
   local height maxima within 15 mm; 2.5 mm bleeder holes through the shell. The head blank is not pierced unless
   `core_vents=1` (its only opening is the neck, plus the bolt holes in its base tabs).
7. **Orientation + estimate.** Candidates per part (flange up/down, ±20° tilts; core neck up/down) scored by a
   geometric time model plus a heavy penalty for support touching the cavity face. Time = volume / flow +
   layers × per-layer overhead + support; support is the Oh et al. proxy (overhang area × drop height).
   Calibrate `printer.flow_mm3_s` on your machine; expect ±20 % before calibration.
8. **Validate.** Each part must be manifold, one body, positive volume; core ∩ shells ≈ 0; silicone volume
   (cavity − core, above the neck) → Dragon Skin FX-Pro mix weight at SG 1.062.

## Orientation

Sculpts arrive in whatever frame the slicer left them in (one came in lying on its back). `orient=auto` finds
the crown (the flat neck/base cut faces down; otherwise the longest axis with the heavier side up) and the face
(in the band between chin and brows, the curvature-weighted centre of the surface is displaced toward the face,
because the back of a head is smooth) and rotates the mesh into the head frame. Override with `keep` or an
explicit `"<up>,<front>"` such as `+y,+z`. The report records what was detected.

## Closing the mouth

`close_mouth=1` brings the lips together and removes the teeth, tongue and inner mouth:

1. **locate** — a closing of radius `mouth_radius` on a signed-distance grid of the lower face fills every
   concavity narrower than twice the radius; the largest centred fill blob is the mouth. It yields the lip gap per
   column, the mid-line and the corners. The radius must be at least half the gap (20 mm for a wide-open jaw).
2. **deform** — `pinch` (gaps under 25 mm): every vertex near the cavity moves toward the mid-line, the lip edges
   overlapping by 1.5 mm; `hinge` (open jaws): the whole lower jaw rotates up about the jaw joint (mouth-line
   height, 62 % of the depth back) by the angle that presses the lips together, fading into the neck under the
   chin crease. Nothing is deleted: teeth and tongue ride along and end up inside the head.
3. **reseal** — the region is re-wrapped from a winding-number SDF of the deformed, self-overlapping mesh, so
   anything inside vanishes and the lips fuse along one line; the wrapped vertices are then snapped back onto the
   original surface wherever it is within 1 mm, restoring skin detail everywhere but the seam. The patch is
   swapped into the sculpt with booleans; the rest is untouched.

The edited sculpt is saved as `sculpt_edited.obj` next to the mold parts, with a before/after render in
`preview/mouth.png`. Results are a credible first pass, not a finish: expect to touch up the lip seam by hand for
a very wide grin.

## Puppet mold (`mold_type=puppet`)

For a puppet casting there is no head blank: two shell halves only, and the neck sleeve is the pour spout. Pour
solid, or slush-cast a skin and pull the casting out through the neck. The shells are made solid inside open
pockets such as a wide-open mouth (a 25 mm closing of the outer skin; the cavity face is still the exact sculpt),
so teeth and the inside of the mouth cast cleanly. `puppet_cavity=rod` adds a third part, a neck plug that bolts to
the ring with a rod (`rod_diameter`, `rod_depth`) that leaves an armature or handle hole. The silicone block
reports the solid pour and a 6 mm slush skin.

## Mouth insert (`mouth_insert=1`): one mold, two mouths

For an open-mouth sculpt the mouth void can become its own part instead of a second mold. The void (what a
closing of `mouth_radius` fills between the lips: teeth, tongue and the inside of the mouth, exact against the
sculpt) is fused to a stepped plug of the shell around the lips. Within 20 mm of the void the skin follows the
*closed* envelope (solid over the void, like a puppet shell) and carries a smooth 6 mm boss; the plug is that
skin within 12 mm of the void through its full thickness plus the outer layer within 16 mm, so it seats on a step
from the outside and cannot fall into the cavity (`insert_clearance` 0.25 mm all round; the shell's window is the
plug grown by that clearance, and the plug itself is subtracted too so the parts are disjoint exactly).

Two plugs share the window:

- **mouth_insert** — the void plus the plug. In: the mold casts the open mouth. With the head blank left out and
  the sleeve as the spout it is the puppet mold; with the blank in it is an open-mouth mask.
- **mouth_cover** — the plug alone, with the closed lips as its cavity face. In: the closed-mouth mask.
  The lips come from a second file (`--closed`, or the second upload) — your own closed-mouth edit of the same
  sculpt — registered onto the working sculpt by similarity ICP on the upper head (`register.py`; the window is
  widened to cover everything the edit changes, jaw included). Without one, the automatic lip closure is used, and
  if that fails the cover bridges the lips smoothly.

The head blank clears both versions of the lips. The silicone block reports the closed-mouth mask, the open-mouth
mask and the puppet (solid and 6 mm slush). Pick `mouth_radius` at least half the lip gap (25 mm for a wide grin).
`mouth_insert` and `close_mouth` are exclusive.

## Mouth wedge (`mouth_wedge=1`): sealed lips, hollow mouth, slit to open

The puppet route: the **closed-mouth** sculpt is the working file (normal shells, no window,
head blank) and the **open-mouth** original is the second file (`--open`, or the second upload), registered
onto it. The open mouth's void — teeth, tongue, palate, exact against the open sculpt — is cut back by one
`mask_thickness` behind the closed lips (the void intersected with the closed sculpt eroded by `mask_thickness`)
and becomes the `mouth_wedge` core. The head blank is clipped to clear both sculpts. Cast with the wedge in: the
mask comes out with sealed lips over a hollow mouth lined with silicone teeth and tongue; slit the lips to open
it. The wedge is anchored at the throat, where a hand goes anyway: a `wedge_boss_diameter` boss reaches the blank
and two `wedge_peg_diameter` pegs go 12 mm into it (matching holes in the blank, 0.25 mm clearance). The
silicone block reports the mask with the hollow. `mouth_wedge` and `mouth_insert` are exclusive.

## Compute switches

`make_core`, `fit_enabled`, `estimate_silicone`, `estimate_print`, `make_vents`, `make_previews` (all on by default)
skip whole stages. Off means skipped, not approximated: no blank → shells only (no mask-thickness check, no silicone
block); no print estimate → parts export as built and times read zero; fit off ignores `fit_target`. The toolkit
panel has them under "Compute", and a "Debug console" below the results with the engine status, every API call,
the params sent, the full job log and the report's stages, warnings and validation.

## Interior geometry is removed

A human head has to fit inside, so anything *inside* the sculpt is irrelevant: an inner mouth, teeth, a tongue, a
hollow behind a thin skin. On ingest every region of outside that cannot be reached from far away without passing
a slit narrower than 12 mm (a closed mouth's interior behind the lips) is made solid, and the cleaned sculpt is
saved as `sculpt_edited.obj`. Open crevices on the surface (wrinkles, ear folds, nostrils) are small and untouched.
The shells, the core and the silicone estimate all see the skin-only sculpt (`interior.py`).

## Head blank (`core_shape`)

`core_shape=head` (the panel default) makes the core a generic smoothed human head form instead of an offset of
the sculpt: a few ellipsoids and a neck in ANSUR-like proportions, softened with an 8 mm blur (`blank.py`). It is
scaled so its hat-line circumference equals the fit target (the wearer's head, with ease) and aligned with the
sculpt's hat line. The core field is `max(blank, sculpt + t)`: the head form, clipped wherever the sculpt is too
tight to keep the mask `t` thick, so it never breaks through; wherever the creature is bigger than a head the
mask is solid silicone. The head blank is never pierced (no core vents); its only opening is the neck, plus the
bolt holes in its base tabs. `core_shape=sculpt` restores the sculpt-following core.

## Smooth inside (`core_smooth_mm`)

By default the core follows the sculpt at a constant offset, so the inside of the mask carries every wrinkle and
pore. With `core_smooth_mm` > 0 (panel: Core & resolution → Inside of mask) the core field is blurred over that
radius and then clamped so the core never comes closer to the sculpt than the mask thickness. The inner face of
the mask becomes a soft head shape: the wall is `mask_thickness` at the thinnest and thicker in folds and details.
Side effects: more silicone (the report accounts for it), a simpler core that prints faster and cleaner, and no
inner-face detail. 8–14 mm suits a full head; 24 mm is a near-bald egg.

## Fitting the mask to a real head

The mask's inner surface is the core, so what has to match a wearer is the core's circumference at the hat
line (widest horizontal loop above the ears and brows), not the sculpt's outer size. `fit_target` measures the
sculpt's hat-line convex-hull perimeter, subtracts 2π × mask thickness, and applies one uniform scale so the
inner circumference equals the target × (1 + `fit_ease_pct`):

| `fit_target` | inner circumference target | source |
|---|---|---|
| `male` | 573 mm | ANSUR II 50th-percentile adult male head circumference |
| `female` | 549 mm | ANSUR II 50th-percentile adult female |
| `custom` | `fit_circumference_mm` | tape measure over the brows and the bump at the back of the head |
| `none` | no rescale (measurement still reported) | |

Default ease is −2 % (snug; Shore-2A silicone stretches). `scale` is an extra manual factor on top. The report's
`fit` block shows the measured and resulting circumferences, plus hat-line length and breadth, so you can spot
a measurement thrown off by horns, hair or huge ears (the band is 72–92 % of the mesh height).

## Print speed: walls and flanges

The three knobs that set print time are the shell wall, the flange width and the flange thickness. The toolkit
panel has them at the top with presets:

| Preset | Wall | Flange | Bolts | Keys |
|---|---|---|---|---|
| Fast | 2.5 mm | 10 × 4 mm | M4 every 60 mm | 3 mm |
| Standard | 3.5 mm | 20 × 6 mm | M5 every 70 mm | 4 mm |
| Sturdy | 5 mm | 25 × 8 mm | M6 every 60 mm | 5 mm |

Registration keys shrink automatically to fit a narrower flange (a key of radius r needs a flange of 2r + 2 mm).
Bolt holes are cut while the flange is at least bolt Ø + 4.5 mm wide; narrower than that they are skipped and the
report says so (use clamps or spring clips). Make the wall a whole number of extrusion widths (the panel shows the
perimeter count) so the slicer does not add slow gap fill. The report's `print_profile` block records what was
built.

## Parameters

See `shellcast/params.py` — every field is settable from the UI, `-p key=value`, or the JSON `params` form field
of `POST /api/jobs`. The ones that matter most: `pull_axis`, `neck_offset` (neck plane above the mesh bottom),
`shell_thickness`, `mask_thickness`, `flange_width`, `bolt_spacing`, `core_close_radius`, `core_open_radius`,
`voxel_pitch` (1.5 mm default; 1.0 for finer cores, slower), `printer.*`.

## Layout

```
shellcast/params.py     MoldParams + PrinterProfile
shellcast/geom.py       trimesh<->manifold3d, SDF grid, iso-surfaces, exact closing/opening, primitives
shellcast/ingest.py     load, units, clean, manifold gate, rewrap, proxy
shellcast/build.py      shell, flange, split, bolts, keys, core, collar, tabs, sprue, vents
shellcast/estimate.py   geometric time/filament model, orientation search
shellcast/pipeline.py   Job: stages, validation, export, report.json
shellcast/server.py     FastAPI: POST /api/jobs, GET /api/jobs/{id}, previews, files, zip
shellcast/cli.py        info / run / fixture / serve
shellcast/fixtures.py   analytic sphere + head proxy
shellcast/orient.py     put a sculpt into the head frame (+Z crown, face toward -Y)
shellcast/interior.py   strip inner geometry so the sculpt is its outer skin only
shellcast/mouth.py      close the mouth: bring lips together, remove teeth and tongue
shellcast/register.py   similarity ICP to align a second version of the same sculpt
shellcast/fit.py        rescale so the finished mask fits a real head
shellcast/blank.py      generic head form to use as the core
shellcast/render.py     matplotlib before/after previews
web/index.html          three.js viewer: assembled or print-oriented parts, section plane, bolts/vents/sprue
tests/                  volumes vs analytic answers, closing/opening, full pipeline, determinism, rewrap, union
```

## Hardware limits and what the engine does about them

Measured on an 8-core machine with 15 GB of RAM: a 2.3 M-triangle, 380 mm bust runs end to end in **5 min 40 s** with a
**5.5 GB** peak. The rules that keep it there:

- The sculpt is decimated to `cavity_max_faces` (1 M) **before** any repair or boolean; that is ~0.35 mm edges on
  a head, invisible on an FDM print, and every later stage scales with it. Turn it up only if you have RAM to spare.
- The SDF grid pitch is coarsened automatically to stay under `max_grid_cells` (40 M).
- 3MF files are read by a chunked parser (~1 GB for 2 M triangles). trimesh's reader needed >7 GB and, for
  OrcaSlicer/Bambu 3MFs, loaded every component's part file once per reference — tripling the mesh.
- Each job runs in its own `nice -n 10` process with a `job_timeout_min` budget (30); a job over it is stopped and
  reported as such.
- Memory: only the systemd unit (`shellcast.service`) sets a ceiling (`MemoryMax=10G`). Under it, a job that runs
  out is killed and reported, and the engine survives. Under `./run.sh` or `shellcast serve` there is no ceiling: a
  runaway job can use all the machine's memory until the kernel's OOM killer steps in.
- `voxel_pitch=2` roughly halves the time; the shape of the outer shell is unaffected, the core gets slightly smoother.

## Known limits (v1)

- The mask ends at the narrowest section under the jaw by default (busts with shoulders); set `neck_offset` to cut elsewhere.
- Planar parting only; ears with deep backs need `pull_axis=y` or a hand-chosen `parting_offset`. Silhouette
  parting and a third ear-cap piece are the next milestone.
- Vents are straight bleeder holes through the shell; they will weep a little silicone. Dog-legged vents routed
  above the sprue rim are not built yet.
- The estimator is uncalibrated until you compare it with OrcaSlicer's own estimate and adjust `flow_mm3_s`.
- Very large sculpts: 5–20 M triangles are fine for the cavity boolean but the first run will take minutes; the
  SDF works on the proxy so `voxel_pitch` is the main speed knob.
- No painted thickness map, no build-volume splitting, no ribs yet.

## License

MIT. See [LICENSE](LICENSE). The bundled three.js files under `web/vendor/three/` are © the three.js authors,
also MIT.
