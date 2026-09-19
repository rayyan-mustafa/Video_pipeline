"""The Colab CPU render worker: a bundle in, an mp4 out. Nothing else.

Paste this whole file into ONE Colab cell and run it. It prints a URL; paste
that into the Studio's Timeline page, next to the Render button.

    Runtime -> Change runtime type -> CPU          <- CPU. Not T4, not L4.
    then run the cell, then paste the printed URL

WHY CPU, AND WHY THIS REFUSES TO RUN ON A GPU. Colab bills COMPUTE UNITS BY
WALL-CLOCK ON THE RUNTIME, not by how much the GPU is used, and an x264 encode
uses none of it. STUDIO_HANDOFF.md measured what that mistake costs:

    render on CPU runtime (free)      12 videos/month =  3.2 units
    render on GPU runtime (L4)        12 videos/month =  5.7-8.2 units

That is 2-2.5x the entire monthly bill, spent on an encode the GPU never
touched. So a GPU runtime is not a warning here, it is a refusal - the handoff
asks for exactly that ("refuse to run, or warn loudly, if nvidia-smi is
present"), because an operator who has to notice a warning is an operator who
will eventually not notice it. Set STUDIO_ALLOW_GPU=1 if you genuinely mean it.

WHY THIS FILE KNOWS NOTHING ABOUT THE PIPELINE. Same seam as
`colab_worker.py`: the laptop decided every editorial question - which move,
how many frames, what encode settings - and wrote the answers down as ffmpeg
argv inside the bundle. This worker unzips, runs them in order, and hands the
file back. There is no scene logic here, no pacing, no motion vocabulary, and
therefore nothing that can silently drift out of date with the repo. The
executor itself (`bundlerun.py`) is not even installed here - it arrives
INSIDE each bundle, so it always matches the manifest it is running.

WHAT IT WILL NOT RUN. Every command in a manifest must start with "ffmpeg".
See bundlerun.check().
"""


# =============================================================================
#  STUDIO RENDER WORKER - ffmpeg only, CPU only
# =============================================================================
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import uuid
import zipfile
from pathlib import Path

subprocess.run([sys.executable, "-m", "pip", "-q", "install",
                "fastapi", "uvicorn[standard]", "python-multipart"], check=True)

# cloudflared quick tunnel: no account, no login, new hostname each run.
if not shutil.which("cloudflared"):
    subprocess.run(
        "wget -q -O /usr/local/bin/cloudflared "
        "https://github.com/cloudflare/cloudflared/releases/latest/download/"
        "cloudflared-linux-amd64 && chmod +x /usr/local/bin/cloudflared",
        shell=True, check=True)

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
import uvicorn

# ---------------------------------------------------------------------------
#  The GPU refusal. See the header.
# ---------------------------------------------------------------------------
HAS_GPU = shutil.which("nvidia-smi") is not None and subprocess.run(
    ["nvidia-smi"], capture_output=True).returncode == 0
ALLOW_GPU = os.environ.get("STUDIO_ALLOW_GPU") == "1"

if HAS_GPU and not ALLOW_GPU:
    name = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True, text=True).stdout.strip() or "a GPU"
    raise SystemExit(
        f"\n!! REFUSING TO START: this is a {name} runtime.\n"
        f"   Rendering is 100% CPU work, and Colab bills a GPU runtime by\n"
        f"   wall-clock at GPU rates - roughly 2-2.5x the entire monthly bill\n"
        f"   for an encode the GPU never touches.\n\n"
        f"   Runtime -> Change runtime type -> CPU, then re-run this cell.\n"
        f"   (Set STUDIO_ALLOW_GPU=1 first if you really mean to do this.)\n")

if not shutil.which("ffmpeg"):
    subprocess.run("apt-get -qq update && apt-get -qq install -y ffmpeg",
                   shell=True, check=True)

FFMPEG_V = subprocess.run(["ffmpeg", "-version"], capture_output=True,
                          text=True).stdout.splitlines()[0]
CPUS = os.cpu_count()
print(f"CPU runtime, {CPUS} cores")
print(FFMPEG_V)

ROOT = Path("/content/studio_render")
ROOT.mkdir(parents=True, exist_ok=True)

app = FastAPI()
JOBS: dict[str, dict] = {}


@app.get("/health")
def health():
    return {"ok": True, "role": "render", "gpu": False,
            "cpus": CPUS, "ffmpeg": FFMPEG_V}


def _load_executor(bundle: Path):
    """Import the bundle's OWN bundlerun.py - never a copy installed here.

    The whole point (see the header) is that the executor and the manifest
    ship together, so a worker left running across a repo change cannot run
    an old executor against a new manifest.
    """
    src = bundle / "bundlerun.py"
    if not src.is_file():
        raise RuntimeError("bundle has no bundlerun.py - it was not written by "
                           "this repo's renderjob.build()")
    spec = importlib.util.spec_from_file_location(f"bundlerun_{bundle.name}", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _render(job_id: str) -> None:
    job = JOBS[job_id]
    bundle = Path(job["dir"])

    def log(line: str) -> None:
        job["log"].append(line)
        print(f"[{job_id}] {line}", flush=True)

    def run(argv, cwd, label):
        t0 = time.time()
        p = subprocess.run([str(a) for a in argv], cwd=str(cwd),
                           capture_output=True, text=True)
        if p.returncode != 0:
            # ffmpeg puts everything on stderr; the tail is the actual error.
            raise RuntimeError(f"{label} failed:\n{p.stderr[-2000:]}")
        log(f"  {label} in {time.time() - t0:.1f}s")

    try:
        manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
        run_mod = _load_executor(bundle)

        problems = run_mod.check(manifest)
        if problems:
            raise RuntimeError("; ".join(problems))
        bad = run_mod.verify_files(bundle, manifest)
        if bad:
            raise RuntimeError("bundle did not arrive intact: " + "; ".join(bad))

        job["total"] = len(manifest["steps"])
        log(f"{job['total']} slots, {manifest['total_duration']:.1f}s, "
            f"provider={manifest['provider']}")
        t0 = time.time()

        def progress(n, total, note):
            job["done"] = n
            if n % 10 == 0 or n == total:
                log(f"{n}/{total} slots")

        out = run_mod.execute(bundle, manifest, run, on_progress=progress)
        took = time.time() - t0
        job.update(status="done", result=str(out), seconds=round(took, 1))
        rate = manifest["total_duration"] / took if took else 0
        log(f"done in {took:.0f}s ({rate:.2f}x realtime) -> {out.name} "
            f"({out.stat().st_size / 1e6:.1f} MB)")
    except Exception as e:
        job.update(status="failed", error=str(e))
        log(f"FAILED: {e}")
        traceback.print_exc()


@app.post("/render")
async def render(bundle: UploadFile = File(...)):
    """Accept a zipped bundle and start rendering it. Returns a job id."""
    job_id = uuid.uuid4().hex[:12]
    d = ROOT / job_id
    d.mkdir(parents=True, exist_ok=True)

    zpath = d / "bundle.zip"
    with zpath.open("wb") as fh:
        while chunk := await bundle.read(1 << 22):
            fh.write(chunk)

    with zipfile.ZipFile(zpath) as z:
        # A zip from elsewhere can name entries "../.." and write outside the
        # folder. extractall does not check; this does.
        for n in z.namelist():
            p = (d / n).resolve()
            if not str(p).startswith(str(d.resolve())):
                raise HTTPException(400, f"bundle entry escapes its folder: {n}")
        z.extractall(d)
    zpath.unlink()

    JOBS[job_id] = {"id": job_id, "dir": str(d), "status": "running",
                    "log": [], "done": 0, "total": 0,
                    "started": time.time(), "result": None, "error": None}
    threading.Thread(target=_render, args=(job_id,), daemon=True).start()
    return {"id": job_id, "status": "running"}


@app.get("/jobs/{job_id}")
def job_status(job_id: str, since: int = 0):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    return {k: v for k, v in job.items() if k != "log"} | {
        "log": job["log"][since:], "log_len": len(job["log"])}


@app.get("/jobs/{job_id}/result")
def job_result(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    if job["status"] != "done":
        raise HTTPException(409, f"job is {job['status']}")
    return FileResponse(job["result"], media_type="video/mp4",
                        filename=Path(job["result"]).name)


@app.delete("/jobs/{job_id}")
def job_delete(job_id: str):
    """Free the runtime's disk. A bundle is ~150 MB and /content is not big."""
    job = JOBS.pop(job_id, None)
    if job is None:
        raise HTTPException(404, "no such job")
    shutil.rmtree(job["dir"], ignore_errors=True)
    return {"deleted": job_id}


threading.Thread(
    target=lambda: uvicorn.run(app, host="0.0.0.0", port=8898, log_level="warning"),
    daemon=True).start()
time.sleep(3)

proc = subprocess.Popen(
    ["cloudflared", "tunnel", "--url", "http://127.0.0.1:8898", "--no-autoupdate"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

url = None
for line in proc.stdout:
    m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
    if m:
        url = m.group(0)
        break

print("\n" + "=" * 62)
print("  RENDER WORKER URL - paste this into the Studio Timeline page:")
print(f"\n     {url}\n")
print("  Keep this cell running. The URL dies when the runtime does,")
print("  and a quick tunnel gets a NEW one every session.")
print("=" * 62)
