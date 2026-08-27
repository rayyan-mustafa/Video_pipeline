"""Video Studio backend - a local UI over the existing channels/ pipeline.

    ..\\vo_coach\\.venv\\Scripts\\python.exe -m uvicorn studio.backend.app:app --port 8770

WHAT THIS IS AND IS NOT. It is a thin driver: every decision about how a video is
built still lives in channels/. Short pure functions (imageplan, scenesplit) are
imported and called directly; long stages (voice, mux, script) are spawned as
subprocesses so their SystemExit cannot take the server down. Nothing about the
pipeline's behaviour changes because this exists - the CLI keeps working
untouched, and that is checked as a regression.

BINDS TO 127.0.0.1 ONLY. Same rule vo_coach/live/server.py follows: this serves
arbitrary local files by path and has no authentication, so it must never be
reachable off the machine.
"""

from __future__ import annotations

import mimetypes
from pathlib import Path

from fastapi import Body, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from . import assets, jobs, paths, planning, projects, scripts

app = FastAPI(title="Video Studio", version="0.1.0")

# The Vite dev server runs on another port, so the browser treats it as a
# different origin. Restricted to localhost - a wildcard here would let any page
# you happen to be browsing drive this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
#  Health
# ---------------------------------------------------------------------------
@app.get("/api/health")
def health() -> dict:
    return {
        "ok": not paths.check(),
        "problems": paths.check(),
        "repo": str(paths.REPO),
        "python": str(paths.PYTHON),
    }


# ---------------------------------------------------------------------------
#  Projects
# ---------------------------------------------------------------------------
@app.get("/api/projects")
def api_projects() -> list[dict]:
    return projects.list_projects()


@app.post("/api/projects")
def api_create_project(channel: str = Query(...), topic: str = Query(...)) -> dict:
    try:
        return projects.create(channel, topic)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


def _proj(channel: str, topic: str):
    """Resolve a project or 400. Every project route starts here, so a bad id
    fails the same way everywhere instead of raising a bare ValueError."""
    try:
        return projects.get(f"{channel}/{topic}")
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@app.get("/api/projects/{channel}/{topic}")
def api_project(channel: str, topic: str) -> dict:
    try:
        return projects.project_status(projects.get(f"{channel}/{topic}"))
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


# ---------------------------------------------------------------------------
#  Files
# ---------------------------------------------------------------------------
# Exactly the types the UI renders: pictures, the voiceover, the finished video,
# and the plan/report files. An ALLOWLIST rather than a denylist because the first
# version of this endpoint confined paths to the repo and happily served
# f:\wierd_bio\.env - the API keys are inside the repo, so "is it in the repo" was
# never the right question. Anything not on this list has no reason to reach a
# browser.
SERVABLE_EXTS = {
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif",
    ".wav", ".mp3", ".m4a", ".flac", ".ogg",
    ".mp4", ".webm",
    ".txt", ".md", ".json",
}

# Directories whose contents are never the UI's business, even for allowed types
# (.git holds objects; a venv holds thousands of .txt and .json files).
BLOCKED_PARTS = {".git", ".venv", "node_modules", "__pycache__"}


@app.get("/api/file")
def api_file(path: str = Query(..., description="absolute path inside the repo")):
    """Serve one file from disk, for <img>, <audio> and <video> tags.

    The browser needs real files - the waveform loads full_vo.wav, the editor
    shows 105 jpegs - and the plan JSON already carries absolute paths, so a
    by-path endpoint is the natural fit. Three guards make it safe: the resolved
    path must be inside the repo (symlinks resolved first), must not pass through
    a blocked directory, and must be a type the UI actually renders.
    """
    p = Path(path).resolve()
    try:
        rel = p.relative_to(paths.REPO)
    except ValueError:
        raise HTTPException(403, "path is outside the repository") from None

    if BLOCKED_PARTS.intersection(rel.parts):
        raise HTTPException(403, "path is in a blocked directory")
    if p.suffix.lower() not in SERVABLE_EXTS:
        raise HTTPException(403, f"{p.suffix or 'this file type'} is not servable")
    if not p.is_file():
        raise HTTPException(404, f"not found: {p}")

    media, _ = mimetypes.guess_type(p.name)
    return FileResponse(p, media_type=media or "application/octet-stream")


# ---------------------------------------------------------------------------
#  Script
# ---------------------------------------------------------------------------
@app.get("/api/projects/{channel}/{topic}/script")
def api_script(channel: str, topic: str, wpm: int = 150) -> dict:
    return scripts.analyse(_proj(channel, topic), wpm=wpm)


@app.put("/api/projects/{channel}/{topic}/script")
async def api_save_script(channel: str, topic: str, body: dict = Body(...)) -> dict:
    text = body.get("text")
    if not isinstance(text, str):
        raise HTTPException(400, "body must be {\"text\": \"...\"}")
    return scripts.write(_proj(channel, topic), text)


# ---------------------------------------------------------------------------
#  Plan
# ---------------------------------------------------------------------------
@app.get("/api/projects/{channel}/{topic}/plan")
def api_plan(channel: str, topic: str) -> dict:
    plan = planning.read(_proj(channel, topic))
    if plan is None:
        raise HTTPException(404, "no plan yet - run planning first")
    return plan


@app.post("/api/projects/{channel}/{topic}/plan")
async def api_run_plan(channel: str, topic: str, body: dict = Body(default={})) -> dict:
    proj = _proj(channel, topic)
    pid = f"{channel}/{topic}"
    if jobs.registry.running(pid, "plan"):
        raise HTTPException(409, "a plan is already running for this project")
    job = await planning.run(
        proj,
        split=body.get("split", "match"),
        seconds_per_image=body.get("seconds_per_image"),
        topic=body.get("topic"),
        allow_brackets=bool(body.get("allow_brackets")),
    )
    return job.as_dict()


# ---------------------------------------------------------------------------
#  Images
# ---------------------------------------------------------------------------
@app.get("/api/projects/{channel}/{topic}/images")
def api_images(channel: str, topic: str) -> dict:
    return assets.scan(_proj(channel, topic))


@app.get("/api/projects/{channel}/{topic}/images/pacing")
def api_image_pacing(channel: str, topic: str, wpm: int = 150) -> dict:
    return assets.pacing_table(_proj(channel, topic), wpm=wpm)


@app.post("/api/projects/{channel}/{topic}/images")
async def api_upload_images(channel: str, topic: str,
                            files: list[UploadFile] = File(...)) -> dict:
    return await assets.upload(_proj(channel, topic), files)


# ---------------------------------------------------------------------------
#  Jobs
# ---------------------------------------------------------------------------
@app.get("/api/jobs")
def api_jobs(project: str | None = None, limit: int = 50) -> list[dict]:
    return jobs.registry.list(project=project, limit=limit)


@app.get("/api/jobs/{job_id}")
def api_job(job_id: str) -> dict:
    job = jobs.registry.get(job_id)
    if job is None:
        raise HTTPException(404, "no such job")
    return job.as_dict()


@app.post("/api/jobs/{job_id}/cancel")
async def api_cancel(job_id: str) -> dict:
    ok = await jobs.registry.cancel(job_id)
    if not ok:
        raise HTTPException(409, "job is not running")
    return {"cancelled": True}


@app.get("/api/jobs/{job_id}/stream")
async def api_stream(job_id: str):
    """Server-sent events: one `data:` frame per line of pipeline output."""
    if jobs.registry.get(job_id) is None:
        raise HTTPException(404, "no such job")

    async def gen():
        async for line in jobs.registry.stream(job_id):
            # SSE frames are newline-delimited, so an embedded newline would
            # split one log line into two frames.
            yield f"data: {line}\n\n".encode("utf-8")
        job = jobs.registry.get(job_id)
        yield f"event: end\ndata: {job.status if job else 'gone'}\n\n".encode("utf-8")

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
