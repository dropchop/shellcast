"""ShellCast command line.

    shellcast info  mesh.obj
    shellcast run   mesh.obj --out ./job1 [--param shell_thickness=4 --param pull_axis=y --param export_stl=true ...]
    shellcast fixture head|sphere out.stl
    shellcast serve [--port 5070]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .params import MoldParams


def _params(items, base=None):
    """-p key=value overrides on top of a params dict, coerced and validated by MoldParams.from_dict."""
    d = dict(base or {})
    if isinstance(d.get("printer"), dict):
        d["printer"] = dict(d["printer"])
    for kv in items or []:
        k, _, v = kv.partition("=")
        if k.startswith("printer."):
            d.setdefault("printer", {})[k[8:]] = v
        else:
            d[k] = v
    return MoldParams.from_dict(d)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="shellcast", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("info", help="load a mesh and report units, size, triangle count and manifold status")
    a.add_argument("mesh")
    a = sub.add_parser("run", help="generate a full mold set")
    a.add_argument("mesh")
    a.add_argument("--out", required=True)
    a.add_argument("--param", "-p", action="append", help="key=value (see shellcast/params.py); printer.key=value for the printer")
    a.add_argument("--params-file", help="JSON file with MoldParams (used by the server to run jobs out of process)")
    a.add_argument("--quiet", action="store_true", help="only print the log lines, not the summary")
    a.add_argument("--closed", help="closed-mouth version of the same sculpt (for the mouth cover when mouth_insert=1)")
    a.add_argument("--open", help="open-mouth version of the same sculpt (for the mouth wedge when mouth_wedge=1; the main mesh is then the closed-mouth one)")
    a = sub.add_parser("fixture", help="write an analytic test mesh")
    a.add_argument("name", choices=["head", "sphere"])
    a.add_argument("out")
    a = sub.add_parser("serve", help="start the web UI")
    a.add_argument("--port", type=int, default=5070)
    a.add_argument("--host", default="127.0.0.1", help="interface to listen on; the UI has no login, so 0.0.0.0 opens it to your whole network")
    args = ap.parse_args(argv)

    if args.cmd == "info":
        from .ingest import clean, guess_units, load_mesh
        from .geom import manifold_from_tm
        t = load_mesh(args.mesh)
        unit, scale = guess_units(t)
        print(f"file: {args.mesh}\ntriangles: {len(t.faces)}\nextents: {t.extents.round(1).tolist()} ({unit}, scale {scale})\nwatertight: {t.is_watertight}")
        try:
            m = manifold_from_tm(clean(t))
            print(f"manifold: yes  volume {m.volume() / 1000:.1f} cm³  genus {m.genus()}  bodies {len(m.decompose())}")
        except ValueError as e:
            print(f"manifold: NO ({e}) → will be rewrapped")
        return 0
    if args.cmd == "fixture":
        from .fixtures import write_fixture
        t = write_fixture(args.name, args.out)
        print(f"wrote {args.out}: {len(t.faces)} triangles, extents {t.extents.round(1).tolist()}")
        return 0
    if args.cmd == "run":
        from .pipeline import run
        p = _params(args.param, json.loads(Path(args.params_file).read_text()) if args.params_file else None)
        rep = run(args.mesh, args.out, p, log=lambda line: print(line, flush=True), closed_path=args.closed, open_path=args.open)
        if rep["status"] != "done":
            print("FAILED:", rep.get("error"), file=sys.stderr)
            return 1
        if args.quiet:
            return 0
        print(json.dumps({k: rep[k] for k in ("totals", "validation", "warnings", "notes") if k in rep}, indent=2))
        for name, e in rep["parts"].items():
            print(f"  {name:8s} {e['orientation']:22s} {e['filament_g']:7.0f} g  {e['time_geom']:>9s}  {'fits' if e['fits_bed'] else 'TOO BIG'}  {e['files']}")
        print(f"report: {Path(args.out) / 'report.json'}")
        return 0
    if args.cmd == "serve":
        import uvicorn
        if args.host not in ("127.0.0.1", "localhost", "::1"):
            print(f"warning: serving on {args.host}: anyone who can reach this port can upload, run and delete jobs (no login)", file=sys.stderr)
        uvicorn.run("shellcast.server:app", host=args.host, port=args.port, log_level="info")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
