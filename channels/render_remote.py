"""Render this project on a remote worker instead of on this laptop.

    python render_remote.py --project wierdbiology/chin --worker https://x.trycloudflare.com

WHY A CLI AND NOT A FUNCTION IN THE BACKEND. The local render is already a
subprocess the Studio spawns as a job (`build_video.py --stage mux`), which is
what gives it a live log, a cancel button and a row on the Jobs page. Making
the remote render another CLI means it inherits all of that for free instead of
needing a second, parallel notion of "a long thing that is running". The two
render paths are then the same shape end to end: same job registry, same SSE
stream, same page.

WHAT THIS DOES NOT DO: decide anything. The bundle is built by
`pipeline.renderjob` from the same plan and the same providers the local render
uses, so choosing remote changes WHERE the ffmpeg runs and nothing else. If the
output differs from a local render of the same plan, that is a bug, not a
setting.

PRINTS PROGRESS TO STDOUT LINE BY LINE, unbuffered, because that is what the
job registry pumps into the browser.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import config, motionplan, renderjob          # noqa: E402
from pipeline.project import Project                         # noqa: E402


def _say(msg: str) -> None:
    print(msg, flush=True)


def _zip_bundle(src: Path, out: Path) -> Path:
    """Zip the bundle. STORED, not deflated, for the files that dominate it.

    The bundle is jpegs, mp4s and a wav - all already compressed. Deflate
    spends real seconds on them and saves almost nothing, and the upload is
    the slow part either way.
    """
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        for p in sorted(src.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(src).as_posix())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", required=True, help="<channel>/<topic>")
    ap.add_argument("--worker", required=True,
                    help="the render worker's URL (studio/colab_render.py prints it)")
    ap.add_argument("--motion", default="kenburns",
                    help="motion provider for still slots (default: kenburns)")
    ap.add_argument("--keep-bundle", action="store_true",
                    help="do not delete the staged bundle afterwards - for "
                         "inspecting exactly what was sent")
    ap.add_argument("--timeout", type=float, default=3600.0,
                    help="give up on the render after this many seconds")
    args = ap.parse_args()

    import httpx

    proj = Project.parse(args.project)
    worker = args.worker.rstrip("/")

    if not proj.render_plan.exists():
        raise SystemExit(f"no render plan at {proj.render_plan} - run the Plan "
                         f"stage first")
    if not proj.voice.exists():
        raise SystemExit(f"no mastered voiceover at {proj.voice} - run the Voice "
                         f"stage first")

    # ---- the worker has to be there BEFORE we spend minutes packing --------
    _say(f"Checking worker {worker} ...")
    try:
        h = httpx.get(f"{worker}/health", timeout=20.0)
        h.raise_for_status()
        info = h.json()
    except Exception as e:
        raise SystemExit(
            f"worker is not reachable: {type(e).__name__}: {e}\n"
            f"  Start studio/colab_render.py in a Colab CPU cell and paste its "
            f"CURRENT url - a cloudflared quick tunnel changes hostname every run.")
    if info.get("role") != "render":
        raise SystemExit(
            f"that worker says role={info.get('role')!r}. This is the RENDER "
            f"worker's URL, not the voice worker's - they are different "
            f"runtimes (voice wants a T4, render must be CPU).")
    _say(f"  ok - {info.get('cpus')} cores, {info.get('ffmpeg', '?')}")

    # ---- build the bundle from the same plan the local render would use ----
    plan = json.loads(proj.render_plan.read_text(encoding="utf-8"))
    script_plan = json.loads(proj.plan.read_text(encoding="utf-8"))
    ovr = {}
    if proj.overrides.exists():
        try:
            ovr = json.loads(proj.overrides.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            ovr = {}
    attach = motionplan.attach_kinds(plan, script_plan, proj,
                                     motion_overrides=ovr.get("motion"))
    for w in attach["warnings"]:
        _say(f"  [warn] {w}")
    _say(f"{attach['n_motion']} motion slot(s), {attach['n_ready']} with a clip on disk")

    stage = proj.build / "remote_bundle"
    if stage.exists():
        shutil.rmtree(stage)
    _say(f"\nPacking {len(plan)} slots into {stage} ...")

    def packing(n, total, _note):
        if n % 25 == 0 or n == total:
            _say(f"  packed {n}/{total}")

    manifest = renderjob.build(
        plan, proj.voice, proj.out.name, stage,
        provider_name=args.motion,
        width=config.FRAME_WIDTH, height=config.FRAME_HEIGHT,
        fps=config.OUTPUT_FPS, on_progress=packing)

    zpath = proj.build / "remote_bundle.zip"
    _zip_bundle(stage, zpath)
    mb = zpath.stat().st_size / 1e6
    _say(f"  bundle {mb:.1f} MB ({len(manifest['steps'])} steps)")

    # ---- upload, poll, download -------------------------------------------
    _say(f"\nUploading to {worker} ...")
    t0 = time.time()
    with zpath.open("rb") as fh:
        r = httpx.post(f"{worker}/render",
                       files={"bundle": (zpath.name, fh, "application/zip")},
                       timeout=httpx.Timeout(args.timeout, connect=30.0))
    if r.status_code != 200:
        raise SystemExit(f"worker refused the bundle ({r.status_code}): {r.text[:400]}")
    job_id = r.json()["id"]
    _say(f"  uploaded in {time.time() - t0:.0f}s, job {job_id}")

    _say("\nRendering on the worker ...")
    seen = 0
    deadline = time.time() + args.timeout
    status = "running"
    while time.time() < deadline:
        time.sleep(3.0)
        try:
            s = httpx.get(f"{worker}/jobs/{job_id}", params={"since": seen},
                          timeout=30.0).json()
        except Exception as e:
            # A tunnel hiccup is not a dead render - keep polling until the
            # deadline rather than throwing away a job that may still finish.
            _say(f"  [warn] poll failed ({type(e).__name__}) - retrying")
            continue
        for line in s.get("log", []):
            _say(f"  {line}")
        seen = s.get("log_len", seen)
        status = s.get("status", "running")
        if status in ("done", "failed"):
            break
    else:
        raise SystemExit(f"gave up after {args.timeout:.0f}s - the job may still "
                         f"be running on the worker as {job_id}")

    if status != "done":
        raise SystemExit(f"remote render failed: {s.get('error')}")

    _say(f"\nDownloading {proj.out.name} ...")
    proj.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = proj.out.with_suffix(".part")
    with httpx.stream("GET", f"{worker}/jobs/{job_id}/result",
                      timeout=httpx.Timeout(args.timeout, connect=30.0)) as resp:
        resp.raise_for_status()
        with tmp.open("wb") as fh:
            for chunk in resp.iter_bytes(1 << 20):
                fh.write(chunk)
    # Only replace a good previous render once the new one is fully on disk.
    tmp.replace(proj.out)

    try:
        httpx.delete(f"{worker}/jobs/{job_id}", timeout=30.0)
    except Exception:
        pass                      # the runtime dies soon enough anyway

    if not args.keep_bundle:
        shutil.rmtree(stage, ignore_errors=True)
        zpath.unlink(missing_ok=True)

    _say(f"\n  {proj.out}  ({proj.out.stat().st_size / 1e6:.1f} MB)")
    _say(f"  total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
