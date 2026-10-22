# ShellCast build document — review and recommended changes

Design notes. The original build document is not part of this repo; section numbers (§) refer to it.

Reviewed against the v1 implementation in this repo (2026-09-23). Items are ordered by how much they change
the product, not by section. Each one says what the doc proposes, what happened when it was built, and the
recommendation.

## 1. The print-time objective needs a target and a measured baseline before any optimizer

The doc makes print time "priority #1" but never says what a good number is, and its illustrative
650 g / ~6–10 h for both halves omits the flanges, sleeve, ring and keys. On the 190 × 200 × 280 mm test
head the v1 output is ~750 g per half and ~770 g for the core; the flange band alone is ~25 % of a half.
The block-mold comparison in the report shows the approach saves 5–8× on filament, which is the real
argument — but the absolute numbers say the *support* term, not perimeters, dominates for a bowl printed on
its back. Recommendation: before building §7.3's orientation optimizer any further, print one half of the
fixture head and one core on the actual large-format printer, record grams and hours, and set
`printer.flow_mm3_s`, `layer_overhead_s` and `support_density` from that. Every later decision in the doc
(ribs vs thicker wall, flange-up vs tilted, cathedral roof vs support) is a ±30 % question that the
uncalibrated model cannot answer.

## 2. Morphological *closing*, not opening, does what §6.4 wants

§6.4 prescribes an opening (erode then dilate) "so the core does not follow ear canals, nostrils and deep
folds". Opening removes thin *protrusions*; it does not fill slots. Filling nostrils, ear canals and
wrinkles is a *closing*. Both are useful and they are different knobs: closing (`core_close_radius`, 6 mm)
fills folds narrower than 12 mm; opening (`core_open_radius`, 4 mm) removes core fins thinner than 8 mm
(thin ears become solid silicone, as the doc intends). v1 implements both. Also, doing morphology on voxel
masks gives a quantised field that marching cubes cannot mesh cleanly; do it exactly (offset iso-surface,
then a fresh signed distance to that surface). This is now in `Grid.morph_field`.

## 3. Hollow core: "cathedral roof" is the wrong default; print the core neck-up with bolt tabs

§7.3 says the core prints neck-down and the hollow needs either internal support or a ≥45° cathedral roof.
Measured on the fixture: a 45° conical roof adds ~400 cm³ of solid — nearly doubling the core — and is the
same cost as the internal support it replaces. The cheaper answer is to print the core the other way up
(crown on the bed): the exterior dome needs only a small support cap near the bed, and the hollow's roof is
now its floor. That only works if the collar plate does not overhang, so the full plate was replaced by four
bolt tabs tied to a disc under the collar (`plate_tabs`). The cathedral option is kept behind
`core_roof_angle` for people who must print neck-down.

## 4. Extend the sculpt below the neck instead of modelling the neck separately

§6.3–6.4 describe a sleeve, a collar with keys, and a neck flange as separate features. Building them
separately produced a collar that did not reach the core (the core's inward offset stops short of the neck
cap) and several coincident-face booleans. Extending the sculpt with a prism of its neck outline before any
offset makes the sleeve, flange band and collar fall out of the same iso-surfaces with no special geometry,
and the exact prism-in-sleeve fit (0.15 mm clearance) locates the core in X, Y and rotation; the tabs set Z.
Recommend rewriting §6.4's "core print/register" around this.

## 5. Sprue: the doc's 12 mm throat breaks into the hollow core

A 25→12 mm funnel entering "the neck gap" cannot fit into a 5 mm silicone gap; in v1 it cut through the
core wall into the hollow. Use a cup (in the collar plate) that feeds a thin *gate slot* the width of the
gap minus 1 mm and ~30 mm long, centred in the gap. §6.7's rule that vents must exit above the sprue rim is
correct and is *not* met by v1's straight bleeder holes — that is honest technical debt, listed below.

## 6. Vents: local maxima on a smooth surface fire in rows

The local-maximum test (§6.7) on a ridge such as the brow underside produces 4–5 vents in a line because
the ridge is nearly level. Add a second pass that merges maxima whose heights differ by less than ~2 mm
along a connected ceiling region, and expose "max vents" prominently. Also, air trapped against the *core*
is detected with the sign flipped (silicone's outward normal is −n on the core) — the doc's text only
mentions the shell.

## 7. Estimator: keep it, but stop quoting the literature's error bars

§7.1 cites 4–16 % errors from papers that measured different machines, materials and label types; the doc
itself notes they are not comparable. The useful number is the one from item 1. Two model changes made in
v1: overhang threshold 50° rather than 45° (a 45° roof must not count as overhang), and support printed at a
density (10 %) rather than as a fixed 15 % — both should become printer-profile fields the user can set from
slicer settings. Add the calibration store from §7.1 as soon as slicer runs exist.

## 8. Robustness items the doc under-weights

- Overlapping closed subtools pass the manifold gate but are self-intersecting; union them (v1 does).
- Sculpts with a big torn region cannot be hole-filled; the rewrap path works but at 0.75 mm it is the
  slowest stage — offer a coarser "preview rewrap" at 1.5 mm.
- STL is float32: every exported STL reloads non-watertight because vertices merge. Say so in the UI and
  make 3MF the only "exact" export (the doc says this in §5; the UI should too).
- manifold3d's `decompose()` splits by surface connectivity, so a hollow solid becomes an outer body plus
  a negative-volume cavity; "keep the largest body" silently fills every cavity. Any helper that touches
  components must be written with this in mind.
- Grid-aligned inputs (fixtures, CAD) put marching cubes exactly on the iso level; jitter the grid origin.

## 9. Roadmap re-ordering

Keep Phase 1 as built. In Phase 2 move the slicer adapter and calibration to the *front* (item 1), then
silhouette parting, then ribs. Add to Phase 2: dog-legged vents to the rim, ridge-merged vent placement,
a per-face undercut heatmap in the viewer (the analysis exists in the doc but v1 only computes the
support proxy), and the "flange-only stiffening" from §6.9 as a parameter (thicker flange, lip) since it is
free to implement. Push variable layer height hints to Phase 3; neither slicer exposes it headlessly.

## What v1 implements from the doc

Phase 0 and Phase 1 in full (M1–M6), plus from Phase 2: the web UI with viewer, section plane and estimate
table, discrete-candidate orientation with the cavity-quality term, and the block-mold comparison report.
Not yet: the slicer CLI adapter, silhouette parting, ribs, calibration store, undercut heatmap.
