"""The Colab Blender render worker: a scene bundle in, an mp4 out. Nothing else.

Paste this whole file into ONE Colab cell and run it. It prints a URL; paste
that into the Studio's Timeline page (or pass it as --blender-colab-url).

    Runtime -> Change runtime type -> T4 GPU        <- GPU. Not CPU.
    then run the cell, then paste the printed URL

WHY GPU, AND WHY THIS REFUSES TO RUN ON CPU. This is the mirror image of
colab_render.py's refusal. THAT worker does pure ffmpeg encode work, which a
GPU never touches, so a GPU runtime there is 2-2.5x the bill for nothing.
THIS worker runs Cycles, which is the one workload on this whole project that
actually uses the GPU it's billed for - a CPU-only runtime would take an
order of magnitude longer per frame for the same compute-unit spend. So the
refusal direction flips: this file refuses a runtime with no CUDA device.

WHY THE ENCODE HAPPENS HERE TOO, NOT ON THE laptop OR THE CPU WORKER. An
8-minute render at 24fps is 11,520 PNG frames - handing those back raw would
be gigabytes over the tunnel for no reason. The runtime is already paying GPU
compute-unit rate for the render pass; the encode adds a few seconds of CPU
work on top of a session you're already paying for, which is a rounding
error next to shipping 11,520 files individually. Set
BLENDER_WORKER_KEEP_FRAMES=1 to skip the encode and download the PNG
sequence instead (e.g. if compositing happens on the laptop).

WHY THIS FILE KNOWS NOTHING ABOUT THE SCENE. Same seam as colab_render.py
and colab_worker.py: the laptop decides every editorial question - geometry,
materials, camera, keyframes - and writes the answer down as a `build_scene()`
call in a plain `scene.py`, shipped inside the bundle. This worker's only
jobs are: stand up Blender, force the render onto the GPU, run the scene
script, encode, hand the file back. There is no asset logic, no camera
grammar, no animation vocabulary here, and therefore nothing that can
silently drift out of date with the repo.

THE BUNDLE CONTRACT. A zip containing:
    scene.py          required. Must define build_scene(scene) -> None.
                       Builds geometry/materials/animation into the passed
                       bpy.types.Scene. Must NOT call bpy.ops.render.render()
                       itself, and must NOT touch scene.cycles.device - this
                       worker owns render execution, the bundle only owns
                       content. May set scene.frame_start/frame_end,
                       scene.render.resolution_x/y, scene.render.fps.
    assets/            optional. Anything scene.py loads by relative path
                       (textures, .blend linked libraries, reference images).
manifest.json      optional. {"engine": "CYCLES"|"EEVEE", "samples": int,
                       "resolution": [w, h], "fps": int}. Missing keys fall
                       back to the defaults below.

WHAT IT WILL NOT RUN. scene.py is exec'd inside Blender's own Python, not
sandboxed - same trust model as bundlerun.py in colab_render.py: the bundle
came from the laptop that built it, not from an untrusted third party. Don't
point this worker at a bundle you didn't build yourself.
"""


# =============================================================================
#  STUDIO BLENDER WORKER - headless Cycles render, GPU only
# =============================================================================
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
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
#  The CPU refusal. See the header - this is the opposite check from
#  colab_render.py, on purpose.
# ---------------------------------------------------------------------------
HAS_GPU = shutil.which("nvidia-smi") is not None and subprocess.run(
    ["nvidia-smi"], capture_output=True).returncode == 0
ALLOW_CPU = os.environ.get("BLENDER_WORKER_ALLOW_CPU") == "1"

if not HAS_GPU and not ALLOW_CPU:
    raise SystemExit(
        "\n!! REFUSING TO START: no CUDA GPU on this runtime.\n"
        "   Cycles rendering is the one workload here that actually uses\n"
        "   GPU compute you're already billed for - on CPU it is roughly\n"
        "   an order of magnitude slower per frame at the same spend.\n\n"
        "   Runtime -> Change runtime type -> T4 GPU, then re-run this cell.\n"
        "   (Set BLENDER_WORKER_ALLOW_CPU=1 first if you really mean to do\n"
        "   this - e.g. testing a scene script with 2 frames.)\n")

GPU_NAME = subprocess.run(
    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
    capture_output=True, text=True).stdout.strip() if HAS_GPU else None

# ---------------------------------------------------------------------------
#  Install Blender - the official portable tarball, not apt's stale package.
#  Pinned to match the version used for any local/interactive authoring so a
#  .blend saved on one side opens cleanly on the other.
# ---------------------------------------------------------------------------
BLENDER_VERSION = os.environ.get("BLENDER_VERSION", "5.2.1")
BLENDER_SERIES = ".".join(BLENDER_VERSION.split(".")[:2])
BLENDER_ROOT = Path(f"/opt/blender-{BLENDER_VERSION}")
BLENDER_BIN = BLENDER_ROOT / "blender"

if not BLENDER_BIN.is_file():
    url = (f"https://download.blender.org/release/Blender{BLENDER_SERIES}/"
           f"blender-{BLENDER_VERSION}-linux-x64.tar.xz")
    print(f"downloading Blender {BLENDER_VERSION} ...")
    t0 = time.time()
    subprocess.run(
        f"wget -q -O /tmp/blender.tar.xz '{url}' && "
        f"mkdir -p {BLENDER_ROOT} && "
        f"tar -xJf /tmp/blender.tar.xz -C {BLENDER_ROOT} --strip-components=1 && "
        f"rm /tmp/blender.tar.xz",
        shell=True, check=True)
    print(f"  installed in {time.time() - t0:.0f}s")

BLENDER_VER_STRING = subprocess.run(
    [str(BLENDER_BIN), "--version"], capture_output=True, text=True
).stdout.splitlines()[0]
print(f"GPU: {GPU_NAME or 'none (CPU override)'}")
print(BLENDER_VER_STRING)

ROOT = Path("/content/blender_render")
ROOT.mkdir(parents=True, exist_ok=True)

DEFAULTS = {"engine": "CYCLES", "samples": 128, "resolution": [1920, 1080], "fps": 24}

# ---------------------------------------------------------------------------
#  The driver script Blender actually runs. It imports the bundle's scene.py,
#  calls build_scene(), forces the render device, then renders the animation.
#  Written to a temp file per job so concurrent jobs never race on one path.
# ---------------------------------------------------------------------------
_DRIVER_TEMPLATE = textwrap.dedent("""
    import bpy, json, sys
    args = json.loads(sys.argv[sys.argv.index("--") + 1])

    sys.path.insert(0, args["bundle_dir"])
    import scene as user_scene  # the bundle's scene.py

    sc = bpy.context.scene
    user_scene.build_scene(sc)

    m = args["manifest"]
    sc.render.engine = m["engine"]
    sc.render.resolution_x, sc.render.resolution_y = m["resolution"]
    sc.render.fps = m["fps"]
    sc.frame_start = sc.frame_start or 1
    sc.render.image_settings.file_format = "PNG"
    sc.render.filepath = args["frames_dir"] + "/f_"

    if m["engine"] == "CYCLES":
        sc.cycles.samples = m["samples"]
        sc.cycles.device = "GPU"
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "CUDA"
        prefs.get_devices()
        for d in prefs.devices:
            d.use = True

    bpy.ops.render.render(animation=True)
""")


def _load_manifest(bundle: Path) -> dict:
    p = bundle / "manifest.json"
    m = dict(DEFAULTS)
    if p.is_file():
        m.update(json.loads(p.read_text(encoding="utf-8")))
    return m


app = FastAPI()
JOBS: dict[str, dict] = {}


@app.get("/health")
def health():
    return {"ok": True, "role": "blender-render", "gpu": GPU_NAME,
            "blender": BLENDER_VER_STRING}


def _render(job_id: str) -> None:
    job = JOBS[job_id]
    bundle = Path(job["dir"])
    frames_dir = bundle / "frames"
    frames_dir.mkdir(exist_ok=True)

    def log(line: str) -> None:
        job["log"].append(line)
        print(f"[{job_id}] {line}", flush=True)

    try:
        if not (bundle / "scene.py").is_file():
            raise RuntimeError("bundle has no scene.py")
        manifest = _load_manifest(bundle)
        log(f"engine={manifest['engine']} samples={manifest.get('samples')} "
            f"res={manifest['resolution']} fps={manifest['fps']}")

        driver = bundle / "_driver.py"
        driver.write_text(_DRIVER_TEMPLATE, encoding="utf-8")
        payload = json.dumps({"bundle_dir": str(bundle),
                              "frames_dir": str(frames_dir),
                              "manifest": manifest})

        t0 = time.time()
        p = subprocess.run(
            [str(BLENDER_BIN), "--background", "--factory-startup",
             "--python", str(driver), "--", payload],
            cwd=str(bundle), capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(f"blender failed:\n{p.stdout[-2000:]}\n{p.stderr[-1000:]}")
        n_frames = len(list(frames_dir.glob("f_*.png")))
        log(f"rendered {n_frames} frames in {time.time() - t0:.0f}s")
        if n_frames == 0:
            raise RuntimeError("blender exited 0 but wrote no frames")

        if os.environ.get("BLENDER_WORKER_KEEP_FRAMES") == "1":
            out = bundle / "frames.zip"
            with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
                for f in sorted(frames_dir.glob("f_*.png")):
                    z.write(f, f.name)
        else:
            out = bundle / "render.mp4"
            fps = manifest["fps"]
            t1 = time.time()
            enc = subprocess.run(
                ["ffmpeg", "-y", "-framerate", str(fps), "-i",
                 str(frames_dir / "f_%04d.png"), "-c:v", "libx264",
                 "-pix_fmt", "yuv420p", str(out)],
                capture_output=True, text=True)
            if enc.returncode != 0:
                raise RuntimeError(f"ffmpeg encode failed:\n{enc.stderr[-2000:]}")
            log(f"encoded in {time.time() - t1:.1f}s -> {out.name} "
                f"({out.stat().st_size / 1e6:.1f} MB)")

        job.update(status="done", result=str(out), seconds=round(time.time() - t0, 1))
    except Exception as e:
        job.update(status="failed", error=str(e))
        log(f"FAILED: {e}")
        traceback.print_exc()


@app.post("/render")
async def render(bundle: UploadFile = File(...)):
    """Accept a zipped scene bundle and start rendering it. Returns a job id."""
    job_id = uuid.uuid4().hex[:12]
    d = ROOT / job_id
    d.mkdir(parents=True, exist_ok=True)

    zpath = d / "bundle.zip"
    with zpath.open("wb") as fh:
        while chunk := await bundle.read(1 << 22):
            fh.write(chunk)

    with zipfile.ZipFile(zpath) as z:
        for n in z.namelist():
            p = (d / n).resolve()
            if not str(p).startswith(str(d.resolve())):
                raise HTTPException(400, f"bundle entry escapes its folder: {n}")
        z.extractall(d)
    zpath.unlink()

    JOBS[job_id] = {"id": job_id, "dir": str(d), "status": "running",
                    "log": [], "started": time.time(), "result": None, "error": None}
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
    media = "application/zip" if job["result"].endswith(".zip") else "video/mp4"
    return FileResponse(job["result"], media_type=media,
                        filename=Path(job["result"]).name)


@app.delete("/jobs/{job_id}")
def job_delete(job_id: str):
    """Free the runtime's disk. A frame set is large and /content is not big."""
    job = JOBS.pop(job_id, None)
    if job is None:
        raise HTTPException(404, "no such job")
    shutil.rmtree(job["dir"], ignore_errors=True)
    return {"deleted": job_id}


threading.Thread(
    target=lambda: uvicorn.run(app, host="0.0.0.0", port=8897, log_level="warning"),
    daemon=True).start()
time.sleep(3)

proc = subprocess.Popen(
    ["cloudflared", "tunnel", "--url", "http://127.0.0.1:8897", "--no-autoupdate"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

url = None
for line in proc.stdout:
    m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
    if m:
        url = m.group(0)
        break

print("\n" + "=" * 62)
print("  BLENDER WORKER URL - paste this into the Studio Timeline page:")
print(f"\n     {url}\n")
print("  Keep this cell running. The URL dies when the runtime does,")
print("  and a quick tunnel gets a NEW one every session.")
print("=" * 62)
