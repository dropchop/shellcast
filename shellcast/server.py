"""FastAPI job server: upload a sculpt, run the pipeline in a worker thread, poll, download parts and previews."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .params import MoldParams

JOBS_DIR = Path(os.environ.get("SHELLCAST_JOBS", Path.home() / "shellcast_jobs"))
JOBS_DIR.mkdir(parents=True, exist_ok=True)
# installed from a wheel: shellcast/web; from a checkout or an editable install: web/ at the repo root
_HERE = Path(__file__).resolve().parent
WEB = next((p for p in (_HERE / "web", _HERE.parent / "web") if (p / "index.html").is_file()), _HERE.parent / "web")
MAX_UPLOAD = 1_500_000_000

app = FastAPI(title="ShellCast", version=__version__)
app.mount("/vendor", StaticFiles(directory=str(WEB / "vendor")), name="vendor")
_jobs: dict[str, dict] = {}
_lock = threading.Lock()
_sem = threading.Semaphore(1)     # one geometry job at a time; they use every core


OOM_HINT = ("The engine ran out of memory on this sculpt. Try a larger voxel pitch (2–2.5 mm), a lower "
            "cavity_max_faces (1,000,000), or force_rewrap, which uses the memory-bounded SDF path.")


def _read_log(d: Path, n=400):
    f = d / "job.log"
    if not f.is_file():
        return []
    try:
        return f.read_bytes()[-200_000:].decode("utf-8", errors="replace").splitlines()[-n:]
    except OSError:
        return []


def _state(jid):
    if not re.fullmatch(r"[\w\-]+", jid):     # a job id is one folder name inside JOBS_DIR, never a path out of it
        return None
    with _lock:
        j = _jobs.get(jid)
    d = JOBS_DIR / jid
    if j:
        return {**j, "log": _read_log(d)}
    if not d.is_dir():
        return None
    if (d / "report.json").is_file():
        try:
            rep = json.loads((d / "report.json").read_text())
        except Exception:  # noqa: BLE001
            rep = {"status": "error", "error": "report.json is unreadable"}
        return {"id": jid, "status": rep.get("status", "done"), "report": rep, "log": rep.get("log") or _read_log(d),
                "name": Path(rep.get("input", {}).get("file", "")).name}
    # a directory with an upload but no report: the engine died mid-job (e.g. OOM-killed) before this version
    src = next((f.name for f in d.iterdir() if f.suffix.lower() in (".obj", ".stl", ".ply", ".3mf", ".glb", ".gltf", ".off")), "")
    rep = {"status": "error", "error": "The engine was restarted while this job was running (most likely killed for using too much memory). " + OOM_HINT}
    return {"id": jid, "status": "error", "report": rep, "log": _read_log(d), "name": src}


def _run(jid, mesh_path, params, raw=None, closed_path=None, open_path=None):
    """Each job is a separate process: a crash or an OOM kill ends the job, not the engine, and gets reported.
    The raw parameter dict is handed over as sent (api_create has already checked it with from_dict), so the job
    reads exactly what the page sent, with this process's defaults filling the rest."""
    d = JOBS_DIR / jid
    merged = {**params.to_dict(), **(raw or {})}
    if raw and isinstance(raw.get("printer"), dict):
        merged["printer"] = {**params.to_dict()["printer"], **raw["printer"]}
    (d / "params.json").write_text(json.dumps(merged))
    with _sem:
        with _lock:
            _jobs[jid]["status"] = "running"
            _jobs[jid]["started"] = time.time()
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        with open(d / "job.log", "wb") as logf:
            # low priority so the machine stays responsive; a hard time budget so nothing runs away
            proc = subprocess.Popen(["nice", "-n", "10", sys.executable, "-m", "shellcast", "run", str(mesh_path), "--out", str(d),
                                     "--params-file", str(d / "params.json"), "--quiet"] + (["--closed", str(closed_path)] if closed_path else []) + (["--open", str(open_path)] if open_path else []),
                                    stdout=logf, stderr=subprocess.STDOUT, env=env, cwd=str(Path(__file__).resolve().parent.parent))
            with _lock:
                _jobs[jid]["pid"] = proc.pid
            try:
                rc = proc.wait(timeout=max(1, int(params.job_timeout_min)) * 60)
            except subprocess.TimeoutExpired:
                proc.kill()
                rc = -15
                logf.write(f"\n[timeout] job exceeded {params.job_timeout_min} min and was stopped\n".encode())
        rep = None
        if (d / "report.json").is_file():
            try:
                rep = json.loads((d / "report.json").read_text())
            except Exception:  # noqa: BLE001
                rep = None
        if rep is None:
            why = ("killed for using too much memory. " + OOM_HINT if rc in (-9, 137) else
                   f"it exceeded the {params.job_timeout_min}-minute budget. " + OOM_HINT if rc == -15 else
                   f"the job process exited with code {rc} before writing a report")
            rep = {"status": "error", "error": f"Job failed: {why}", "log": _read_log(d)}
            (d / "report.json").write_text(json.dumps(rep))
        with _lock:
            _jobs[jid].update(status=rep.get("status", "error"), report=rep, finished=time.time(), pid=None)


@app.get("/", response_class=HTMLResponse)
def index():
    return (WEB / "index.html").read_text()


def _engine_memory_mb():
    try:
        return int(Path("/sys/fs/cgroup" + Path("/proc/self/cgroup").read_text().split("::", 1)[1].strip() + "/memory.current").read_text()) // 1048576
    except Exception:  # noqa: BLE001
        return None


@app.get("/api/status")
def api_status():
    return {"version": __version__, "jobs_dir": str(JOBS_DIR), "defaults": MoldParams().to_dict()}


@app.post("/api/jobs")
async def api_create(file: UploadFile = File(...), params: str = Form("{}"), closed: UploadFile | None = File(None), open_mesh: UploadFile | None = File(None, alias="open")):
    name = re.sub(r"[^\w.\-]+", "_", file.filename or "mesh")[:80]
    if not re.search(r"\.(obj|stl|ply|3mf|glb|gltf|off)$", name, re.I):
        raise HTTPException(400, "Upload an OBJ, STL, PLY, 3MF or GLB")
    try:
        raw = json.loads(params or "{}")
        p = MoldParams.from_dict(raw)          # validation only; the job process re-reads the raw dict with the newest code
    except (ValueError, json.JSONDecodeError) as e:
        raise HTTPException(400, f"bad params: {e}")
    jid = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
    d = JOBS_DIR / jid
    d.mkdir(parents=True)
    dest = d / name
    size = 0
    with open(dest, "wb") as f:
        while chunk := await file.read(4 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD:
                f.close()
                shutil.rmtree(d, ignore_errors=True)
                raise HTTPException(413, "file too large")
            f.write(chunk)
    async def _save_second(up, sub):
        if up is None or not up.filename:
            return None
        cname = re.sub(r"[^\w.\-]+", "_", up.filename)[:80]
        if not re.search(r"\.(obj|stl|ply|3mf|glb|gltf|off)$", cname, re.I):
            shutil.rmtree(d, ignore_errors=True)
            raise HTTPException(400, f"{sub} file: upload an OBJ, STL, PLY, 3MF or GLB")
        (d / sub).mkdir(exist_ok=True)
        dest2 = d / sub / cname
        csize = 0
        with open(dest2, "wb") as f:
            while chunk := await up.read(4 << 20):
                csize += len(chunk)
                if csize > MAX_UPLOAD:
                    f.close()
                    shutil.rmtree(d, ignore_errors=True)
                    raise HTTPException(413, f"{sub} file too large")
                f.write(chunk)
        return dest2
    closed_dest = await _save_second(closed, "closed")
    open_dest = await _save_second(open_mesh, "open")
    with _lock:
        _jobs[jid] = {"id": jid, "status": "queued", "log": [], "report": None, "name": name, "created": time.time(), "params": p.to_dict()}
    threading.Thread(target=_run, args=(jid, dest, p, raw, closed_dest, open_dest), daemon=True, name=f"shellcast-{jid}").start()
    return {"id": jid}


@app.get("/api/jobs")
def api_jobs():
    out = []
    ids = set(_jobs) | {d.name for d in JOBS_DIR.iterdir() if d.is_dir()}
    for jid in sorted(ids, reverse=True)[:50]:
        s = _state(jid)
        if not s:
            continue
        rep = s.get("report") or {}
        out.append({"id": jid, "status": s["status"], "name": s.get("name") or Path(rep.get("input", {}).get("file", "")).name,
                    "totals": rep.get("totals"), "elapsed_s": rep.get("elapsed_s")})
    return out


@app.get("/api/jobs/{jid}")
def api_job(jid: str):
    if not re.fullmatch(r"[\w\-]+", jid):
        raise HTTPException(400)
    s = _state(jid)
    if not s:
        raise HTTPException(404)
    rep = s.get("report")
    return {"id": jid, "status": s["status"], "log": (s.get("log") or [])[-400:], "report": rep, "name": s.get("name"),
            "engine_memory_mb": _engine_memory_mb()}


@app.delete("/api/jobs/{jid}")
def api_job_delete(jid: str):
    if not re.fullmatch(r"[\w\-]+", jid):
        raise HTTPException(400)
    with _lock:
        j = _jobs.get(jid, {})
        if j.get("status") == "running":
            pid = j.get("pid")
            if pid:
                try:
                    os.kill(pid, 15)
                except ProcessLookupError:
                    pass
            j["status"] = "cancelled"
        _jobs.pop(jid, None)
    shutil.rmtree(JOBS_DIR / jid, ignore_errors=True)
    return {"ok": True}


@app.get("/api/jobs/{jid}/file/{name}")
def api_job_file(jid: str, name: str):
    if not re.fullmatch(r"[\w\-]+", jid) or not re.fullmatch(r"[\w.\-]+", name):
        raise HTTPException(400)
    p = JOBS_DIR / jid / name
    if not p.is_file():
        raise HTTPException(404)
    return FileResponse(p, filename=name)


@app.get("/api/jobs/{jid}/preview/{name}")
def api_job_preview(jid: str, name: str):
    if not re.fullmatch(r"[\w\-]+", jid) or not re.fullmatch(r"[\w.\-]+\.(glb|png)", name):
        raise HTTPException(400)
    p = JOBS_DIR / jid / "preview" / name
    if not p.is_file():
        raise HTTPException(404)
    return FileResponse(p, media_type="model/gltf-binary" if name.endswith(".glb") else "image/png")


@app.get("/api/jobs/{jid}/zip")
def api_job_zip(jid: str):
    if not re.fullmatch(r"[\w\-]+", jid):
        raise HTTPException(400)
    d = JOBS_DIR / jid
    if not d.is_dir():
        raise HTTPException(404)
    z = d / f"shellcast-{jid}.zip"
    if not z.is_file():
        import zipfile
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in d.iterdir():
                if f.suffix.lower() in (".obj", ".3mf", ".stl", ".json") and f != z:
                    zf.write(f, f.name)
    return FileResponse(z, filename=z.name)
