"""Fast checks of the server, CLI and launcher contracts (no geometry is built)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from shellcast import cli

ROOT = Path(__file__).resolve().parent.parent


def test_serve_binds_loopback_by_default(monkeypatch):
    """The UI has no authentication: `shellcast serve` must not listen on the network unless asked to."""
    import uvicorn
    seen = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: seen.update(kw))
    assert cli.main(["serve", "--port", "5999"]) == 0
    assert seen["host"] == "127.0.0.1"


def test_run_sh_binds_loopback_by_default(tmp_path):
    """./run.sh (what the README tells people to run) must serve on localhost only; HOST opts in to the network."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "python3"
    fake.write_text(f'#!/bin/sh\necho "$@" >> "{tmp_path}/argv"\n')
    fake.chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}"}
    env.pop("HOST", None)
    subprocess.run(["bash", str(ROOT / "run.sh")], env=env, check=True, capture_output=True)
    serve = (tmp_path / "argv").read_text().splitlines()[-1].split()
    assert serve[:3] == ["-m", "shellcast", "serve"]
    assert serve[serve.index("--host") + 1] == "127.0.0.1"


@pytest.fixture
def api(tmp_path, monkeypatch):
    """The FastAPI app with JOBS_DIR at <tmp>/jobs/inner, and a decoy job.log in its parent folder."""
    monkeypatch.setenv("SHELLCAST_JOBS", str(tmp_path / "jobs" / "inner"))
    from fastapi.testclient import TestClient
    from shellcast import server
    inner = tmp_path / "jobs" / "inner"
    inner.mkdir(parents=True, exist_ok=True)
    (inner.parent / "job.log").write_text("DECOY outside the jobs folder\n")
    monkeypatch.setattr(server, "JOBS_DIR", inner)
    return TestClient(server.app, raise_server_exceptions=False)


def test_job_status_stays_inside_jobs_dir(api):
    r = api.get("/api/jobs/%2E%2E")
    assert r.status_code == 400
    assert "DECOY" not in r.text
    assert "DECOY" not in api.get("/api/jobs").text
    from shellcast import server
    assert server._state("..") is None and server._state("../inner") is None


@pytest.mark.parametrize("params", ["[1]", '{"printer": "fast"}', '"x"', '{"printer": [1]}'])
def test_malformed_params_are_a_400(api, params):
    files = {"file": ("head.stl", b"solid x\nendsolid x\n", "application/octet-stream")}
    r = api.post("/api/jobs", files=files, data={"params": params})
    assert r.status_code == 400, r.text
    assert r.json()["detail"].startswith("bad params")
    if "printer" in params:
        assert "printer must be a JSON object" in r.json()["detail"]


def test_cli_override_on_params_file_is_coerced_and_validated(tmp_path, monkeypatch):
    """`run --params-file f -p make_vents=false` must give False (bool("false") is True), and an override is validated."""
    from shellcast import pipeline
    seen = {}
    monkeypatch.setattr(pipeline, "run", lambda mesh, out, p, **kw: seen.update(p=p) or {"status": "done"})
    pf = tmp_path / "params.json"
    pf.write_text('{"make_vents": true, "shell_thickness": 4}')
    base = ["run", "x.stl", "--out", str(tmp_path / "o"), "--quiet", "--params-file", str(pf)]
    assert cli.main(base + ["-p", "make_vents=false", "-p", "printer.wall_lines=3"]) == 0
    assert seen["p"].make_vents is False and seen["p"].shell_thickness == 4.0 and seen["p"].printer.wall_lines == 3
    with pytest.raises(ValueError):
        cli.main(base + ["-p", "shell_thickness=99"])


@pytest.mark.parametrize("bad", [
    {"shell_thicknes": 9}, {"max_vents": "abc"}, {"plate_tabs": "2.5"}, {"max_vents": 2.5}, {"shell_thickness": "x"},
    {"make_vents": "flase"}, {"make_vents": 2}, {"printer": {"nozle": 0.4}}, {"printer": {"wall_lines": "four"}},
    {"pull_axis": None}, [1, 2], "x", {"printer": "fast"}, {"printer": [1]},
])
def test_from_dict_rejects_unknown_and_uncoercible(bad):
    from shellcast.params import MoldParams
    with pytest.raises(ValueError):
        MoldParams.from_dict(bad)


def test_from_dict_coerces_what_forms_and_the_cli_send():
    from shellcast.params import MoldParams
    p = MoldParams.from_dict({"make_vents": "false", "export_stl": "on", "max_vents": "8", "plate_tabs": 2.0,
                              "shell_thickness": "4", "printer": {"wall_lines": "3", "flow_mm3_s": "22", "name": 7}})
    assert p.make_vents is False and p.export_stl is True and p.max_vents == 8 and p.plate_tabs == 2
    assert p.shell_thickness == 4.0 and p.printer.wall_lines == 3 and p.printer.flow_mm3_s == 22.0 and p.printer.name == "7"
    assert isinstance(p.printer.wall_lines, int)


@pytest.mark.parametrize("bad", [
    {"job_timeout_min": 100000}, {"job_timeout_min": 0}, {"max_grid_cells": 10_000_000_000}, {"cavity_max_faces": 10**9},
    {"proxy_faces": 10**8}, {"vent_diameter": 0}, {"cavity_clearance": -5}, {"neck_offset": -100}, {"max_vents": -1},
    {"rod_diameter": 0}, {"smooth_iterations": 1000}, {"printer": {"flow_mm3_s": 0}}, {"printer": {"layer_height": -0.2}},
    {"printer": {"wall_lines": 0}}, {"printer": {"bed_x": 0}},
])
def test_validate_bounds_numeric_fields(bad):
    from shellcast.params import MoldParams
    with pytest.raises(ValueError):
        MoldParams.from_dict(bad)


def test_defaults_and_test_profiles_validate():
    """The defaults, every profile the pipeline tests use, and the web form's defaults stay inside the bounds."""
    from shellcast.params import MoldParams
    MoldParams().validate()
    for kw in ({"voxel_pitch": 2.5, "proxy_faces": 40_000, "shell_thickness": 2.5, "flange_width": 6, "flange_thickness": 4},
               {"voxel_pitch": 2.0, "proxy_faces": 60_000, "core_smooth_mm": 10, "mouth_radius": 20.0},
               {"mold_type": "puppet", "puppet_cavity": "rod"}):
        MoldParams(**kw).validate()


def test_every_numeric_field_is_bounded():
    """No number a request can set is unbounded, on either side."""
    from dataclasses import fields
    from shellcast.params import MoldParams, PrinterProfile
    for cls, wrap in ((MoldParams, lambda k, v: {k: v}), (PrinterProfile, lambda k, v: {"printer": {k: v}})):
        for f in fields(cls):
            if type(f.default) in (int, float):
                for v in (10**12, -10**12):
                    with pytest.raises(ValueError):
                        MoldParams.from_dict(wrap(f.name, v))


def test_wheel_install_serves_the_ui(tmp_path):
    """A plain (non-editable) install must carry web/: shellcast.server mounts web/vendor at import."""
    src = tmp_path / "src"
    files = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout.decode().split("\0")
    for f in filter(None, files):
        if (ROOT / f).is_file():
            (src / f).parent.mkdir(parents=True, exist_ok=True)
            (src / f).write_bytes((ROOT / f).read_bytes())
    subprocess.run([sys.executable, "-m", "pip", "wheel", "-q", "--no-deps", "--no-build-isolation", "-w", str(tmp_path / "whl"), str(src)],
                   check=True, capture_output=True)
    import zipfile
    site = tmp_path / "site"
    zipfile.ZipFile(next((tmp_path / "whl").glob("shellcast-*.whl"))).extractall(site)
    probe = ("from fastapi.testclient import TestClient; from shellcast import server; c = TestClient(server.app); "
             "assert str(server.WEB).startswith(%r), server.WEB; r = c.get('/'); assert r.status_code == 200 and 'three' in r.text; "
             "assert c.get('/vendor/three/three.module.js').status_code == 200; print('ok')") % str(site)
    env = {**os.environ, "PYTHONPATH": str(site), "SHELLCAST_JOBS": str(tmp_path / "jobs")}
    r = subprocess.run([sys.executable, "-c", probe], cwd=site, env=env, capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr[-2000:]
