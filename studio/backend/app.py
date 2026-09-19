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

from . import (assets, jobs, motionstage, packager, paths, planning, projects,
               rendering, scripts, settings, timeline, voicestage, youtube)

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


# ---------------------------------------------------------------------------
#  Settings  (machine-level, not per project)
# ---------------------------------------------------------------------------
# The Colab worker URL lives here because it describes THIS MACHINE'S current
# link to a GPU, not anything about a video. A cloudflared quick tunnel gets a
# new hostname every session, so this gets retyped once per Colab run.
@app.get("/api/settings")
def api_settings() -> dict:
    return {**settings.load(), "colab_status": settings.check_colab()}


@app.put("/api/settings")
def api_save_settings(body: dict = Body(...)) -> dict:
    saved = settings.save(**body)
    return {**saved, "colab_status": settings.check_colab()}


@app.post("/api/settings/colab/check")
def api_check_colab(body: dict = Body(default={})) -> dict:
    return settings.check_colab(body.get("url"))


@app.post("/api/settings/render-worker/check")
def api_check_render_worker(body: dict = Body(default={})) -> dict:
    """The RENDER worker's health - a different runtime from the voice one."""
    return settings.check_render_worker(body.get("url"))


@app.get("/api/topics")
def api_topics() -> list[dict]:
    """The 15-topic bank, each flagged with whether a project exists for it."""
    return projects.topics()


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
# .youtube holds OAuth refresh tokens - .json, so servable by type, and each
# one is a password for a YouTube channel.
BLOCKED_PARTS = {".git", ".venv", "node_modules", "__pycache__", ".youtube"}


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


@app.post("/api/projects/{channel}/{topic}/script/generate")
async def api_generate_script(channel: str, topic: str) -> dict:
    proj = _proj(channel, topic)
    pid = f"{channel}/{topic}"
    if jobs.registry.running(pid, "script"):
        raise HTTPException(409, "a script generation is already running for this project")
    try:
        job = await scripts.generate(proj)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return job.as_dict()


@app.post("/api/projects/{channel}/{topic}/script/generate/import")
def api_import_script(channel: str, topic: str) -> dict:
    try:
        return scripts.import_generated(_proj(channel, topic))
    except FileNotFoundError as e:
        raise HTTPException(409, str(e)) from e


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
        motion_clips=body.get("motion_clips") or __import__("random").randint(12, 16),
    )
    return job.as_dict()


@app.post("/api/projects/{channel}/{topic}/prompts")
async def api_run_prompts(channel: str, topic: str) -> dict:
    proj = _proj(channel, topic)
    pid = f"{channel}/{topic}"
    if jobs.registry.running(pid, "prompts"):
        raise HTTPException(409, "prompts generation is already running for this project")
    job = await planning.run_prompts(proj)
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
#  Voice
# ---------------------------------------------------------------------------
# Naming is resolved as its own step, before any alignment runs. See
# voicestage.py: the proposal is shown with its evidence and the operator
# confirms, because a wrong mapping wastes a large-v3 pass.
@app.get("/api/projects/{channel}/{topic}/voice")
def api_voice(channel: str, topic: str) -> dict:
    return voicestage.status(_proj(channel, topic))


@app.post("/api/projects/{channel}/{topic}/voice/recordings")
async def api_upload_recordings(channel: str, topic: str,
                                files: list[UploadFile] = File(...)) -> dict:
    return await voicestage.upload(_proj(channel, topic), files)


@app.get("/api/projects/{channel}/{topic}/voice/naming")
def api_naming(channel: str, topic: str) -> dict:
    return voicestage.propose(_proj(channel, topic))


@app.post("/api/projects/{channel}/{topic}/voice/naming")
def api_apply_naming(channel: str, topic: str,
                     body: dict = Body(default={})) -> dict:
    try:
        return voicestage.apply(_proj(channel, topic),
                                force=bool(body.get("force")))
    except (ValueError, FileExistsError) as e:
        raise HTTPException(409, str(e)) from e


@app.delete("/api/projects/{channel}/{topic}/voice/naming")
def api_revert_naming(channel: str, topic: str) -> dict:
    try:
        return voicestage.revert(_proj(channel, topic))
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(409, str(e)) from e


@app.post("/api/projects/{channel}/{topic}/voice/run")
async def api_run_voice(channel: str, topic: str,
                        body: dict = Body(default={})) -> dict:
    proj = _proj(channel, topic)
    pid = f"{channel}/{topic}"
    if jobs.registry.running(pid, "voice"):
        raise HTTPException(409, "an alignment is already running for this project")

    state = voicestage.status(proj)
    if not state["ready"]:
        raise HTTPException(
            409,
            f"recordings are not resolved yet - section(s) "
            f"{state['missing_sections']} have no audio. Apply the naming "
            f"proposal first.")
    try:
        job = await voicestage.run(
            proj,
            whisper=body.get("whisper", voicestage.DEFAULT_MODEL),
            raw_voice=bool(body.get("raw_voice")),
            split=body.get("split"),
            voice_profile=body.get("voice_profile", "natural"),
            use_colab=bool(body.get("use_colab")),
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return job.as_dict()


# ---------------------------------------------------------------------------
#  Motion
# ---------------------------------------------------------------------------
# Selection needs MEASURED holds, so every route here is blocked until the voice
# and plan stages have run. See motionstage.status().
@app.get("/api/projects/{channel}/{topic}/motion")
def api_motion(channel: str, topic: str) -> dict:
    return motionstage.status(_proj(channel, topic))


@app.get("/api/projects/{channel}/{topic}/motion/propose")
def api_motion_propose(channel: str, topic: str) -> dict:
    try:
        return motionstage.propose(_proj(channel, topic))
    except ValueError as e:
        raise HTTPException(409, str(e)) from e


@app.post("/api/projects/{channel}/{topic}/motion")
def api_save_motion(channel: str, topic: str, body: dict = Body(default={})) -> dict:
    try:
        return motionstage.save(_proj(channel, topic), body.get("slots"))
    except ValueError as e:
        raise HTTPException(409, str(e)) from e


@app.post("/api/projects/{channel}/{topic}/motion/clips")
async def api_upload_clips(channel: str, topic: str,
                           files: list[UploadFile] = File(...)) -> dict:
    return await motionstage.upload(_proj(channel, topic), files)


# ---------------------------------------------------------------------------
#  Timeline  (Phase 5: roll edits, replace, reorder, motion toggle)
# ---------------------------------------------------------------------------
# Every route here recomputes render_plan.json through the SAME
# imageplan.plan_from_binding + motionplan.attach_kinds the CLI's --stage plan
# uses (see timeline.py's module docstring) - the browser can never see a
# timeline the real build would compute differently.
@app.get("/api/projects/{channel}/{topic}/timeline")
def api_timeline(channel: str, topic: str) -> dict:
    return timeline.read(_proj(channel, topic))


@app.post("/api/projects/{channel}/{topic}/timeline")
def api_save_timeline(channel: str, topic: str, body: dict = Body(default={})) -> dict:
    try:
        return timeline.save(_proj(channel, topic), body)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@app.delete("/api/projects/{channel}/{topic}/timeline")
def api_revert_timeline(channel: str, topic: str) -> dict:
    return timeline.revert(_proj(channel, topic))


# ---------------------------------------------------------------------------
#  Render  (the local half of Phase 6 - build a video and preview it here)
# ---------------------------------------------------------------------------
@app.get("/api/projects/{channel}/{topic}/render")
def api_render_status(channel: str, topic: str) -> dict:
    return rendering.status(_proj(channel, topic))


@app.post("/api/projects/{channel}/{topic}/render")
async def api_run_render(channel: str, topic: str, body: dict = Body(default={})) -> dict:
    proj = _proj(channel, topic)
    pid = f"{channel}/{topic}"
    # Both targets are a render of the same project, so either one running
    # blocks the other - two ffmpegs writing one output file is not a thing
    # the second one would survive.
    if jobs.registry.running(pid, "mux") or jobs.registry.running(pid, "mux-colab"):
        raise HTTPException(409, "a render is already running for this project")
    try:
        job = await rendering.run(
            proj,
            motion=body.get("motion", "kenburns"),
            only_section=body.get("only_section"),
            target=body.get("target"),
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return job.as_dict()


# ---------------------------------------------------------------------------
#  Package  (YouTube metadata + the thumbnail image prompt)
# ---------------------------------------------------------------------------
# The one stage whose output never reaches a render: it is what gets typed into
# the upload form. See packager.py for why the UI reads package.json rather than
# the two .txt files it is shown alongside.
@app.get("/api/projects/{channel}/{topic}/package")
def api_package(channel: str, topic: str) -> dict:
    return packager.status(_proj(channel, topic))


@app.post("/api/projects/{channel}/{topic}/package")
async def api_run_package(channel: str, topic: str) -> dict:
    proj = _proj(channel, topic)
    pid = f"{channel}/{topic}"
    if jobs.registry.running(pid, "package"):
        raise HTTPException(409, "a package run is already in flight for this project")

    state = packager.status(proj)
    if not state["can_run"]:
        raise HTTPException(409, state["why"])
    return (await packager.run(proj)).as_dict()


# ---------------------------------------------------------------------------
#  YouTube upload  (the package, the video and the thumbnail, sent)
# ---------------------------------------------------------------------------
# See youtube.py. The connection is per Studio channel, not per project - every
# wierdbiology topic uploads to the same YouTube channel.
@app.get("/api/projects/{channel}/{topic}/youtube")
def api_youtube(channel: str, topic: str) -> dict:
    return youtube.status(_proj(channel, topic))


@app.post("/api/projects/{channel}/{topic}/youtube/thumbnail")
async def api_youtube_thumbnail(channel: str, topic: str,
                                files: list[UploadFile] = File(...)) -> dict:
    if len(files) != 1:
        raise HTTPException(400, "send exactly one thumbnail image")
    try:
        return await youtube.save_thumbnail(_proj(channel, topic), files[0])
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@app.post("/api/youtube/client-secrets")
async def api_youtube_client_secrets(files: list[UploadFile] = File(...)) -> dict:
    try:
        return await youtube.save_client_secrets(files[0])
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@app.post("/api/youtube/{channel}/auth")
async def api_youtube_auth(channel: str) -> dict:
    if jobs.registry.running(f"youtube/{channel}", "youtube-auth"):
        raise HTTPException(409, "a connect is already waiting in your browser")
    if not youtube.CLIENT_SECRETS.is_file():
        raise HTTPException(409, "add the Google OAuth client JSON first")
    return (await youtube.run_auth(channel)).as_dict()


@app.post("/api/projects/{channel}/{topic}/youtube/upload")
async def api_youtube_upload(channel: str, topic: str,
                             body: dict = Body(default={})) -> dict:
    proj = _proj(channel, topic)
    pid = f"{channel}/{topic}"
    if jobs.registry.running(pid, "youtube-upload"):
        raise HTTPException(409, "an upload is already running for this project")

    state = youtube.status(proj)
    if not state["can_upload"]:
        raise HTTPException(409, "cannot upload yet: " + "; ".join(state["blockers"]))
    # The CLI refuses too, but only after spawning; saying it here keeps a
    # stray double-click from looking like a failed job.
    if state["record"] and not body.get("force"):
        raise HTTPException(409, f"already uploaded as {state['record'].get('url')}")
    try:
        job = await youtube.run_upload(
            proj,
            video=body.get("video") or "",
            title=body.get("title"),
            privacy=body.get("privacy", "private"),
            publish_at=body.get("publish_at"),
            category=str(body.get("category", "27")),
            synthetic=bool(body.get("synthetic")),
            force=bool(body.get("force")),
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return job.as_dict()


@app.post("/api/projects/{channel}/{topic}/youtube/thumbnail/retry")
async def api_youtube_thumb_retry(channel: str, topic: str) -> dict:
    proj = _proj(channel, topic)
    if not youtube.status(proj)["record"]:
        raise HTTPException(409, "nothing uploaded yet for this project")
    return (await youtube.run_thumbnail(proj)).as_dict()


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
