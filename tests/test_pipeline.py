"""Acceptance tests from the build doc (§11/§12): analytic volumes, manifold parts, no core/shell interference, vents."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import trimesh

from shellcast.fixtures import head_proxy, sphere
from shellcast.geom import Grid, manifold_from_tm, tm_from_manifold
from shellcast.params import MoldParams
from shellcast.pipeline import run


@pytest.fixture(scope="session")
def head_stl(tmp_path_factory):
    p = tmp_path_factory.mktemp("fx") / "head.stl"
    tm_from_manifold(head_proxy()).export(p)
    return p


def test_sdf_iso_sphere_volume():
    """Outer skin of a sphere: iso(d = w) must be a sphere of radius R + w (marching cubes, ~1 % tolerance)."""
    R, w = 60.0, 4.0
    t = tm_from_manifold(sphere(R))
    g = Grid.for_bounds(t.bounds[0], t.bounds[1], 1.0, margin=w + 5).fill(t.vertices, t.faces)
    O = g.iso(w)
    expected = 4 / 3 * math.pi * (R + w) ** 3
    assert abs(O.volume() - expected) / expected < 0.015
    shell = O - manifold_from_tm(t)
    expected_shell = 4 / 3 * math.pi * ((R + w) ** 3 - R ** 3)
    assert abs(shell.volume() - expected_shell) / expected_shell < 0.03


def test_closing_fills_slot_and_opening_strips_fin():
    """Closing must fill a deep narrow slot; opening must remove a thin fin; neither may eat the bulk."""
    import manifold3d as m3
    cube = m3.Manifold.cube((80, 80, 80), center=True)
    slot = m3.Manifold.cube((4, 60, 50), center=True).translate((0, 0, 25))
    t = tm_from_manifold(cube - slot)
    g = Grid.for_bounds(t.bounds[0], t.bounds[1], 1.0, margin=6).fill(t.vertices, t.faces)
    plain = g.iso(0.0, field=g.morph_field(0.0))
    closed = g.iso(0.0, field=g.morph_field(0.0, close_radius=6.0))
    assert closed.volume() > plain.volume() + 0.7 * (4 * 60 * 50)
    assert closed.volume() < 80 ** 3 * 1.03
    fin = m3.Manifold.cube((3, 40, 30), center=True).translate((0, 0, 55))
    t2 = tm_from_manifold(cube + fin)
    g2 = Grid.for_bounds(t2.bounds[0], t2.bounds[1], 1.0, margin=6).fill(t2.vertices, t2.faces)
    opened = g2.iso(0.0, field=g2.morph_field(0.0, open_radius=4.0))
    assert abs(opened.volume() - 80 ** 3) < 0.06 * 80 ** 3       # fin gone, cube kept (edges get rounded by r)


def test_full_pipeline_head(head_stl, tmp_path):
    p = MoldParams(voxel_pitch=2.0, proxy_faces=60_000, export_obj=True, export_stl=True)
    rep = run(head_stl, tmp_path / "job", p, log=lambda s: None)
    assert rep["status"] == "done", rep.get("error")
    v = rep["validation"]
    for name in ("shell_A", "shell_B", "core"):
        assert v[name]["manifold"], name
        assert v[name]["volume_mm3"] > 0
        assert v[name]["bodies"] == 1, f"{name} split into {v[name]['bodies']} bodies"
    assert v["core_shell_interference_mm3"] < 5.0
    assert v["silicone_volume_mm3"] > 0
    si = rep["silicone"]
    assert abs(si["mask_volume_ml"] - v["silicone_volume_mm3"] / 1000) < 1e-6
    assert abs(si["part_a_g"] + si["part_b_g"] - si["mix_g"]) < 1e-6 and si["order_g"] > si["mix_g"] > si["mask_weight_g"]
    # a 5 mm skin over the cavity area: volume ≈ area × thickness within 30 % (curvature, folds filled by the core)
    assert 0.6 < si["mask_volume_ml"] / (si["cavity_area_cm2"] * 0.5) < 1.4
    assert rep["bolts"], "no bolts placed"
    assert rep["vents"], "no vents placed"
    assert any(vt["on"] == "shell" for vt in rep["vents"])
    assert not any(vt["on"] == "core" for vt in rep["vents"]), "head blank must not be pierced by default"
    assert rep["validation"]["core"]["genus"] <= 6, "core should have only the base openings (tab bolt holes)"
    # halves are roughly balanced for a symmetric head split on x
    va, vb = v["shell_A"]["volume_mm3"], v["shell_B"]["volume_mm3"]
    assert abs(va - vb) / max(va, vb) < 0.35
    # exported parts reload as manifold (OBJ keeps full precision; STL is float32 and lossy by design), oriented on the bed
    for name, e in rep["parts"].items():
        t = trimesh.load(tmp_path / "job" / e["files"]["obj"], force="mesh")
        assert t.is_watertight, name
        assert abs(t.bounds[0][2]) < 1e-3
        manifold_from_tm(t)
        assert len(trimesh.load(tmp_path / "job" / e["files"]["stl"]).faces) == len(t.faces)
        assert e["time_s_geom"] > 0 and e["filament_g"] > 0
    assert (tmp_path / "job" / "report.json").is_file()
    assert (tmp_path / "job" / "preview" / "shell_A.glb").is_file()


@pytest.mark.slow
def test_determinism(head_stl, tmp_path):
    p = MoldParams(voxel_pitch=2.5, proxy_faces=40_000, export_stl=False)
    r1 = run(head_stl, tmp_path / "a", p, log=lambda s: None)
    r2 = run(head_stl, tmp_path / "b", p, log=lambda s: None)
    assert r1["status"] == r2["status"] == "done"
    for n in r1["validation"]:
        if isinstance(r1["validation"][n], dict):
            assert r1["validation"][n]["volume_mm3"] == pytest.approx(r2["validation"][n]["volume_mm3"], rel=1e-6)


def test_rewrap_of_broken_mesh(tmp_path):
    """A sculpt with a hole and duplicate overlapping bodies must still produce a manifold via rewrap."""
    t = tm_from_manifold(head_proxy())
    keep = np.linalg.norm(t.triangles_center - [0, -60, 60], axis=1) > 18   # tear a 36 mm hole in the forehead
    broken = trimesh.Trimesh(t.vertices, t.faces[keep], process=False)
    extra = tm_from_manifold(sphere(30).translate((0, -60, 0)))     # overlapping body
    broken = trimesh.util.concatenate([broken, extra])
    p = tmp_path / "broken.stl"
    broken.export(p)
    from shellcast.ingest import ingest
    ing = ingest(p, rewrap_pitch=2.0, proxy_faces=30_000)
    assert ing["info"]["method"] == "rewrap"
    assert ing["manifold"].volume() > 0.9 * head_proxy().volume()
    assert ing["info"]["rewrap"]["deviation_p95"] < 2.5


def test_overlapping_bodies_are_unioned(tmp_path):
    t = trimesh.util.concatenate([tm_from_manifold(head_proxy()), tm_from_manifold(sphere(30).translate((0, -60, 0)))])
    p = tmp_path / "two.stl"
    t.export(p)
    from shellcast.ingest import ingest
    ing = ingest(p, rewrap_pitch=2.0, proxy_faces=30_000)
    assert ing["info"]["method"] == "manifold+union"
    assert len(ing["manifold"].decompose()) == 1


def test_params_from_dict_rejects_bad():
    with pytest.raises(ValueError):
        MoldParams.from_dict({"pull_axis": "z"})
    with pytest.raises(ValueError):
        MoldParams.from_dict({"shell_thickness": 0.2})
    with pytest.raises(ValueError):
        MoldParams.from_dict({"bogus": 1})
    p = MoldParams.from_dict({"shell_thickness": "4", "printer": {"flow_mm3_s": "22"}})
    assert p.shell_thickness == 4.0 and p.printer.flow_mm3_s == 22.0


@pytest.mark.slow
def test_fit_to_average_male(head_stl, tmp_path):
    from shellcast.fit import TARGETS, hat_line
    p = MoldParams(voxel_pitch=2.5, proxy_faces=40_000, export_stl=False, fit_target="male", fit_ease_pct=0.0)
    rep = run(head_stl, tmp_path / "fit", p, log=lambda s: None)
    assert rep["status"] == "done", rep.get("error")
    ft = rep["fit"]
    assert ft["target_circumference_mm"] == TARGETS["male"]["circumference"]
    assert abs(ft["mask_inner_circumference_after_mm"] - 573.0) < 1.0
    assert abs(ft["scale"] - (573.0 + 2 * math.pi * p.mask_thickness) / ft["sculpt_hat_circumference_mm"]) < 1e-6
    # the exported geometry really was scaled: outer hat line of the (scaled) sculpt preview matches
    circ_scaled = hat_line(head_proxy().scale((ft["scale"],) * 3))[0]
    assert abs(circ_scaled - ft["sculpt_hat_circumference_after_mm"]) < 2.0


def test_fit_validation():
    with pytest.raises(ValueError):
        MoldParams.from_dict({"fit_target": "custom", "fit_circumference_mm": 100})
    with pytest.raises(ValueError):
        MoldParams.from_dict({"fit_target": "giant"})
    assert MoldParams.from_dict({"fit_target": "custom", "fit_circumference_mm": 600}).fit_circumference_mm == 600


def test_3mf_streaming_reader_matches_core_spec(tmp_path):
    """A core-spec 3MF written by trimesh reloads through the chunked reader with the same geometry."""
    from shellcast.ingest import load_3mf_streaming, load_mesh
    src = tm_from_manifold(head_proxy())
    p = tmp_path / "head.3mf"
    src.export(p)
    t = load_3mf_streaming(p)
    assert len(t.faces) == len(src.faces)
    assert np.allclose(t.extents, src.extents, atol=0.01)
    assert len(load_mesh(p).faces) == len(src.faces)


def test_orientation_detection():
    from shellcast.orient import detect, resolve
    f = tm_from_manifold(head_proxy())
    d = detect(f)
    assert (d["up"], d["front"]) == ("+z", "-y")
    lying = f.copy()
    lying.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))   # crown → +y, face → +z
    d2 = detect(lying)
    assert (d2["up"], d2["front"]) == ("+y", "+z")
    R, _ = resolve("auto", lying)
    back = lying.vertices @ R.T
    assert np.allclose(np.ptp(back, axis=0), f.extents, atol=0.01)
    with pytest.raises(ValueError):
        resolve("+z,+z", f)


@pytest.mark.slow
def test_close_mouth_fills_parted_lips():
    """A head with a 12 mm slot cut across the front (parted lips) must come back with the slot closed and the
    interior gone, while the rest of the head is unchanged."""
    import manifold3d as m3
    from shellcast.mouth import close_mouth
    head = head_proxy()
    slot = m3.Manifold.cube((70, 40, 12), center=True).translate((0, -80, -30))       # mouth: 70 wide, 12 tall, 40 deep
    S = head - slot
    t = tm_from_manifold(S)
    S2, t2, rep = close_mouth(S, t, radius=10.0, pitch=1.0, mode="pinch", seal_pitch=0.8)
    assert rep["mode"] == "pinch" and 8 <= rep["max_gap_mm"] <= 16 and 55 <= rep["width_mm"] <= 85
    # the slot's mid-plane is solid everywhere the head exists: no cavity behind the closed lips
    xs, ys = np.linspace(-33, 33, 12), np.linspace(-98, -62, 10)
    P = np.array([[x, y, -30.0] for y in ys for x in xs])
    in_head = tm_from_manifold(head).contains(P)
    assert tm_from_manifold(S2).contains(P)[in_head].all()
    # the lips draw inward as they meet, so a little volume goes; the head is otherwise intact
    assert abs(S2.volume() - head.volume()) < 0.02 * head.volume()
    assert S2.status() == m3.Error.NoError and len([c for c in S2.decompose() if c.volume() > 0]) == 1
    top_before = t.vertices[:, 2].max(); top_after = t2.vertices[:, 2].max()
    assert abs(top_before - top_after) < 0.5


@pytest.mark.slow
def test_bust_with_shoulders_is_cut_at_the_neck(tmp_path):
    """A bust on shoulders must be molded from the neck up: plane at the narrowest section, no core/shell overlap."""
    from shellcast.fixtures import bust
    src = tmp_path / "bust.stl"
    tm_from_manifold(bust()).export(src)
    p = MoldParams(voxel_pitch=2.5, proxy_faces=40_000, export_stl=False)
    rep = run(src, tmp_path / "job", p, log=lambda s: None)
    assert rep["status"] == "done", rep.get("error")
    z_bottom = rep["input"]["bounds"][0][2]
    assert rep["frame"]["neck_z"] - z_bottom > 40           # well above the 45 mm shoulder slab
    assert rep["validation"]["core_shell_interference_mm3"] < 5.0
    assert all(v["bodies"] == 1 for v in rep["validation"].values() if isinstance(v, dict))
    assert any("neck" in n.lower() for n in rep["notes"])


@pytest.mark.slow
def test_fast_print_profile_keeps_keys_and_bolts(head_stl, tmp_path):
    """Thin walls + narrow flange: parts still valid, keys shrink to fit, bolts kept while they fit, skipped below."""
    p = MoldParams(voxel_pitch=2.5, proxy_faces=40_000, export_stl=False, shell_thickness=2.5, flange_width=10, flange_thickness=4,
                   bolt_diameter=4, key_radius=4)
    rep = run(head_stl, tmp_path / "fast", p, log=lambda s: None)
    assert rep["status"] == "done", rep.get("error")
    assert all(v["bodies"] == 1 and v["manifold"] for v in rep["validation"].values() if isinstance(v, dict))
    assert rep["print_profile"]["bolts"] > 0 and rep["print_profile"]["keys"] > 0
    assert any("keys shrunk" in n.lower() for n in rep["notes"])
    p2 = MoldParams(voxel_pitch=2.5, proxy_faces=40_000, export_stl=False, shell_thickness=2.5, flange_width=6, flange_thickness=4)
    rep2 = run(head_stl, tmp_path / "clamps", p2, log=lambda s: None)
    assert rep2["status"] == "done", rep2.get("error")
    assert rep2["print_profile"]["bolts"] == 0 and rep2["print_profile"]["keys"] > 0
    assert any("too narrow" in n for n in rep2["notes"])
    assert rep2["totals"]["filament_g"] < rep["totals"]["filament_g"]


def test_tiny_scene_units_are_normalised(tmp_path):
    """A Nomad/ZBrush export a few hundredths of a unit across must load as a ~250 mm head, not a 0.03 mm blob."""
    from shellcast.ingest import ingest
    t = tm_from_manifold(head_proxy())
    t.apply_scale(1e-4)                       # 280 mm head → 0.028 units
    p = tmp_path / "tiny.obj"
    t.export(p)
    ing = ingest(p, rewrap_pitch=2.0, proxy_faces=30_000)
    assert 200 <= max(ing["proxy"].extents) <= 300
    assert ing["info"]["scale_applied"] > 1000
    assert ing["info"]["method"] in ("manifold", "manifold+union")


@pytest.mark.slow
def test_smooth_inside_thickens_only(head_stl, tmp_path):
    """core_smooth_mm gives a smoother, simpler core; the mask never gets thinner than mask_thickness (no interference,
    more silicone), and the core has fewer triangles than the sculpt-following one."""
    base = MoldParams(voxel_pitch=2.5, proxy_faces=40_000, export_stl=False)
    r0 = run(head_stl, tmp_path / "follow", base, log=lambda s: None)
    sm = MoldParams(voxel_pitch=2.5, proxy_faces=40_000, export_stl=False, core_smooth_mm=10)
    r1 = run(head_stl, tmp_path / "smooth", sm, log=lambda s: None)
    assert r0["status"] == "done" and r1["status"] == "done", (r0.get("error"), r1.get("error"))
    assert r1["validation"]["core_shell_interference_mm3"] < 5.0
    assert r1["silicone"]["mask_volume_ml"] >= r0["silicone"]["mask_volume_ml"] * 0.98
    assert r1["validation"]["core"]["triangles"] < r0["validation"]["core"]["triangles"]
    assert any("smoothed" in n for n in r1["notes"])


@pytest.mark.slow
def test_generic_head_blank(head_stl, tmp_path):
    """core_shape=head: the core is the generic head form clipped by the sculpt; single body, no interference,
    no vents, silicone at least as much as when following the sculpt."""
    p = MoldParams(voxel_pitch=2.5, proxy_faces=40_000, export_stl=False, core_shape="head", fit_target="male")
    rep = run(head_stl, tmp_path / "blank", p, log=lambda s: None)
    assert rep["status"] == "done", rep.get("error")
    v = rep["validation"]
    assert v["core"]["bodies"] == 1 and v["core"]["manifold"] and v["core_shell_interference_mm3"] < 5.0
    assert rep["head_blank"]["blank_scale"] > 0 and abs(rep["head_blank"]["blank_hat_circumference_mm"] - 573 * 0.98) < 6
    assert not any(vt["on"] == "core" for vt in rep["vents"])
    assert any("generic smoothed head" in n for n in rep["notes"])


def test_internal_cavities_are_filled(tmp_path):
    """A hollow pocket inside the sculpt (e.g. a closed mouth's interior) must not survive ingest: either the void
    filler removes it, or normal repair turns it into a nested solid that the body union swallows. Either way
    the result is one solid with the head's volume, and the filler itself must detect a true void."""
    import manifold3d as m3
    from shellcast.ingest import fill_voids, ingest
    head = head_proxy()
    pocket = m3.Manifold.sphere(18, 32).translate((0, -60, -40))
    hollow = head - pocket                                # enclosed void, fully inside
    filled, n, vol = fill_voids(hollow)
    assert n == 1 and abs(vol - pocket.volume()) < 0.02 * pocket.volume()
    assert abs(filled.volume() - head.volume()) < 0.005 * head.volume()
    src = tmp_path / "hollow.stl"
    tm_from_manifold(hollow).export(src)
    ing = ingest(src, rewrap_pitch=2.0, proxy_faces=30_000)
    assert abs(ing["manifold"].volume() - head.volume()) < 0.005 * head.volume()
    assert len(ing["manifold"].decompose()) == 1


@pytest.mark.slow
def test_interior_geometry_is_removed(tmp_path):
    """A chamber inside the sculpt reachable only through a thin slit (a closed mouth's interior) is removed on
    ingest: the sculpt becomes skin-only, the core passes straight through, nothing overlaps the shell."""
    import manifold3d as m3
    head = head_proxy()
    chamber = m3.Manifold.sphere(1.0, 32).scale((40, 22, 14)).translate((0, -62, -40))      # 80 x 44 x 28 mm cavity
    slit = m3.Manifold.cube((60, 40, 2.0), center=True).translate((0, -84, -40))            # 2 mm slit to the outside
    sculpt = head - chamber - slit
    src = tmp_path / "pocket.stl"
    tm_from_manifold(sculpt).export(src)
    p = MoldParams(voxel_pitch=2.0, proxy_faces=60_000, export_stl=False, core_shape="sculpt")
    rep = run(src, tmp_path / "job", p, log=lambda s: None)
    assert rep["status"] == "done", rep.get("error")
    assert rep.get("interior_removed", {}).get("count", 0) >= 1
    assert rep["validation"]["core_shell_interference_mm3"] < 5.0 and rep["validation"]["core"]["bodies"] == 1
    core = trimesh.load(tmp_path / "job" / "core.obj", force="mesh")
    core.apply_transform(np.linalg.inv(np.array(rep["parts"]["core"]["transform"])))
    # the core wall (5–7.5 mm under the skin) runs straight across the former chamber, 6 mm under the surface at z=-48
    assert core.contains([[0, -79.5, -48]])[0]
    assert (tmp_path / "job" / "sculpt_edited.obj").is_file()


@pytest.mark.slow
def test_puppet_mold_is_two_or_three_parts(head_stl, tmp_path):
    p = MoldParams(voxel_pitch=2.5, proxy_faces=40_000, export_stl=False, mold_type="puppet")
    rep = run(head_stl, tmp_path / "puppet", p, log=lambda s: None)
    assert rep["status"] == "done", rep.get("error")
    assert set(rep["parts"]) == {"shell_A", "shell_B"}
    assert all(v["bodies"] == 1 and v["manifold"] for k, v in rep["validation"].items() if isinstance(v, dict))
    assert rep["silicone"]["mask_volume_ml"] > 2000 and rep["silicone"]["slush_volume_ml"] < rep["silicone"]["mask_volume_ml"]
    p2 = MoldParams(voxel_pitch=2.5, proxy_faces=40_000, export_stl=False, mold_type="puppet", puppet_cavity="rod")
    rep2 = run(head_stl, tmp_path / "rod", p2, log=lambda s: None)
    assert rep2["status"] == "done", rep2.get("error")
    assert set(rep2["parts"]) == {"shell_A", "shell_B", "neck_plug"}
    assert rep2["validation"]["plug_shell_interference_mm3"] < 5.0
    assert rep2["validation"]["neck_plug"]["bodies"] == 1


@pytest.mark.slow
def test_mouth_insert_plugs_the_shell(tmp_path):
    """An open-mouth head (a 24 mm slot across the front) with mouth_insert: the mouth void becomes one extra part
    that sits in a window of the shell without intersecting it, and whose volume is at least the slot's."""
    import manifold3d as m3
    head = head_proxy()
    slot = m3.Manifold.cube((70, 40, 24), center=True).translate((0, -80, -30))
    src = tmp_path / "open.stl"
    tm_from_manifold(head - slot).export(src)
    # the "closed-mouth version": the same head without the slot, deliberately 2 % bigger and turned 3° so the
    # registration has to earn its keep
    import trimesh.transformations as tf
    closed = tm_from_manifold(head)
    closed.apply_transform(tf.translation_matrix([40, -20, -300]) @ tf.rotation_matrix(np.radians(3.0), [0, 0, 1]) @ tf.scale_matrix(1.02))
    csrc = tmp_path / "closed.stl"
    closed.export(csrc)
    p = MoldParams(voxel_pitch=2.5, proxy_faces=40_000, export_stl=False, mouth_insert=True, mouth_radius=20.0)
    rep = run(src, tmp_path / "ins", p, log=lambda s: None, closed_path=csrc)
    assert rep["status"] == "done", rep.get("error")
    assert {"mouth_insert", "mouth_cover"} <= set(rep["parts"]), rep["notes"]
    assert rep["closed_mouth"]["registration_rms_mm"] < 0.4 and abs(rep["closed_mouth"]["scale"] - 1 / 1.02) < 0.01
    v = rep["validation"]
    for nm in ("mouth_insert", "mouth_cover"):
        assert v[nm]["bodies"] == 1 and v[nm]["manifold"], nm
    assert v["insert_shell_interference_mm3"] < 5.0 and v["cover_shell_interference_mm3"] < 5.0
    assert v["insert_core_interference_mm3"] < 5.0 and v["cover_core_interference_mm3"] < 5.0
    assert v["mouth_insert"]["volume_mm3"] > 0.8 * 70 * 40 * 24
    # the cover is the plug without the void, so it is the insert minus roughly the slot
    assert v["mouth_insert"]["volume_mm3"] - v["mouth_cover"]["volume_mm3"] > 0.5 * 70 * 40 * 24   # the slot pokes out of the head
    assert v["mouth_insert"]["volume_mm3"] < 150_000                                              # window not blown up by noise
    assert rep["silicone"]["puppet"]["solid_volume_ml"] > rep["silicone"]["mask_volume_ml"]


@pytest.mark.slow
def test_mouth_wedge_sits_behind_the_closed_lips(tmp_path):
    """Closed-mouth head as the working sculpt, the slotted (open) head as the second file: the wedge is the slot's
    void cut back one mask thickness behind the closed skin, pegged into the blank, touching neither shell nor blank."""
    import manifold3d as m3
    import trimesh.transformations as tf
    head = head_proxy()
    slot = m3.Manifold.cube((70, 40, 24), center=True).translate((0, -80, -30))
    closed_src, open_src = tmp_path / "closed.stl", tmp_path / "open.stl"
    tm_from_manifold(head).export(closed_src)
    op = tm_from_manifold(head - slot)
    op.apply_transform(tf.translation_matrix([-30, 10, 200]) @ tf.scale_matrix(0.98))
    op.export(open_src)
    p = MoldParams(voxel_pitch=2.5, proxy_faces=40_000, export_stl=False, mouth_wedge=True, mouth_radius=20.0, core_shape="head")
    rep = run(closed_src, tmp_path / "wedge", p, log=lambda s: None, open_path=open_src)
    assert rep["status"] == "done", rep.get("error")
    assert "mouth_wedge" in rep["parts"], (rep["notes"], rep["warnings"])
    assert rep["second_sculpt"]["registration_rms_mm"] < 0.4
    v = rep["validation"]
    assert v["mouth_wedge"]["bodies"] == 1 and v["mouth_wedge"]["manifold"]
    assert v["wedge_shell_interference_mm3"] < 5.0 and v["wedge_core_interference_mm3"] < 5.0
    assert v["mouth_wedge"]["volume_mm3"] > 0.3 * 70 * 40 * 24        # the slot void minus the lip skin, plus boss and pegs
    assert v["mouth_hollow_mm3"] > 0.3 * 70 * 40 * 24                  # the cast really has a hollow mouth
    assert v["wedge_pullout_interference_mm3"] < 5.0                    # the wedge slides straight out of the blank


def _lying(t):
    t = t.copy()
    t.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))   # crown → +y, face → +z
    return t


@pytest.mark.parametrize("kind", ["two_bodies", "hollow"])
@pytest.mark.parametrize("lying", [False, True])
def test_ingest_repairs_survive_orientation(tmp_path, kind, lying):
    """Body union and void fill must hold whether or not ingest rotates the sculpt, and the returned mesh and proxy
    must be the repaired solid, not the raw cleaned input."""
    import manifold3d as m3
    from shellcast.ingest import ingest
    head = head_proxy()
    if kind == "two_bodies":
        src_t = trimesh.util.concatenate([tm_from_manifold(head), tm_from_manifold(sphere(30).translate((0, -60, 0)))])
    else:
        src_t = tm_from_manifold(head - m3.Manifold.sphere(18, 32).translate((0, -60, -40)))
    if lying:
        src_t = _lying(src_t)
    src = tmp_path / "in.stl"
    src_t.export(src)
    ing = ingest(src, rewrap_pitch=2.0, proxy_faces=1_000_000)
    m, mesh, proxy = ing["manifold"], ing["mesh"], ing["proxy"]
    assert (ing["info"]["orientation"]["up"] != "+z") == lying
    assert len(m.decompose()) == 1
    assert mesh.body_count == 1
    assert proxy.body_count == 1
    assert m.volume() > 0.99 * head.volume() and m.volume() < 1.04 * head.volume()
    assert abs(mesh.volume - m.volume()) < 1e-3 * m.volume()
    assert abs(proxy.volume - m.volume()) < 1e-3 * m.volume()


def _render_result(rep):
    """Run the page's renderResult (web/index.html) on a report in node, with a stub DOM; returns node's result."""
    import shutil
    import subprocess
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    html = (Path(__file__).resolve().parent.parent / "web" / "index.html").read_text()
    fn = html[html.index("function renderResult(d) {"):]
    fn = fn[:fn.index("\n}\n") + 2]
    js = ("const out = {}; const $ = (id) => (out[id] ||= {}); const COLORS = {};\n"
          "const hms = (s) => `${Math.floor(s / 3600)}h`;\n" + fn +
          "\nconst d = JSON.parse(require('fs').readFileSync(0, 'utf8'));\n"
          "renderResult(d); for (const e of Object.values(d.report.parts)) e.transform.flat();\n"
          "process.stdout.write(out.result.innerHTML);\n")
    return subprocess.run([node, "-e", js], input=json.dumps({"id": "x", "report": rep}), capture_output=True, text=True)


def test_estimates_off_keep_the_report_shape(head_stl, tmp_path):
    """With Silicone and Print estimate off, the job's report still renders in the page (silicone line, part transforms),
    and the block-mold comparison gives no made-up saving factor (it would divide by 0 g of filament)."""
    p = MoldParams(voxel_pitch=2.5, proxy_faces=40_000, export_stl=False, estimate_print=False, estimate_silicone=False,
                   make_previews=False)
    rep = run(head_stl, tmp_path / "off", p, log=lambda s: None)
    assert rep["status"] == "done", rep.get("error")
    rep = json.loads((tmp_path / "off" / "report.json").read_text())
    for name, e in rep["parts"].items():
        assert np.allclose(e["transform"], np.eye(4)), name
    assert rep["block_mold_comparison"]["saving_factor_filament"] is None
    r = _render_result(rep)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "Total" in r.stdout and "NaN" not in r.stdout and "more filament" not in r.stdout
