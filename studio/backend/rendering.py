"""The local render, and the preview once it lands.

THE COLAB CPU RENDER (STUDIO_HANDOFF.md's Phase 6) IS NOT HERE. That needs a
live Colab worker to build and verify against, the same reason the GPU-guarded
notebook (`studio/colab_render.ipynb`) is still a TODO - see the settings.py
Colab plumbing already built for Voice, which this would reuse once written.
This module is the half that needs nothing external: `build_video.py --stage
mux` already renders correctly on this machine, so wiring a button to it and
serving the result back for preview is real, working Phase 6, not a stub.

Spawned as a job, same as voice/plan - build_video.py is an argparse main()
that can SystemExit, so it runs out-of-process like every other long stage.
"""

from __future__ import annotations

from . import jobs, paths

paths.ensure_importable()
from pipeline import clips as clips_mod       # noqa: E402

MOTION_CHOICES = ("none", *sorted(clips_mod.PROVIDERS))


def status(proj) -> dict:
    """What a render needs, and what already exists to preview.

    `video_mtime` exists so the browser can be MADE to refetch: the output
    path is always the same file (`<topic>.mp4`), so a re-render after a
    timeline edit changes the bytes on disk but not the URL. A <video> tag
    (and the HTTP cache in front of it) has no reason to know that, and will
    happily keep playing the stale copy it already loaded. The frontend
    appends this to the URL as a cache-busting query param.
    """
    out = proj.out
    return {
        "ready": proj.render_plan.exists() and proj.voice.exists(),
        "has_render_plan": proj.render_plan.exists(),
        "has_voice": proj.voice.exists(),
        "motion_choices": list(MOTION_CHOICES),
        "has_video": out.exists(),
        "video_path": str(out) if out.exists() else None,
        "video_mb": round(out.stat().st_size / 1e6, 1) if out.exists() else 0,
        "video_mtime": out.stat().st_mtime if out.exists() else None,
    }


async def run(proj, motion: str = "kenburns", only_section: int | None = None):
    """Start `build_video.py --stage mux` as a job. Returns the Job.

    Needs render_plan.json (the Plan stage, with any timeline edits already
    baked in - see timeline.py) and full_vo.wav (the Voice stage) to already
    exist; mux reads both directly rather than recomputing anything, which is
    exactly why a roll edit has to write render_plan.json immediately instead
    of only living in overrides.json.
    """
    if motion not in MOTION_CHOICES:
        raise ValueError(f"unknown motion {motion!r} - choose one of {MOTION_CHOICES}")
    if not proj.render_plan.exists():
        raise ValueError("no render plan yet - run the Plan stage first")
    if not proj.voice.exists():
        raise ValueError("no mastered voiceover yet - run the Voice stage first")

    argv = [
        paths.PYTHON, paths.BUILD_VIDEO,
        "--project", f"{proj.channel}/{proj.topic}",
        "--stage", "mux",
        "--motion", motion,
    ]
    if only_section is not None:
        argv += ["--only-section", str(int(only_section))]

    return await jobs.registry.start(
        kind="mux", project=f"{proj.channel}/{proj.topic}",
        argv=argv, cwd=paths.CHANNELS,
    )
