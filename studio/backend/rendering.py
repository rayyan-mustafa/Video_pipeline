"""The render - here or on a Colab CPU runtime - and the preview once it lands.

TWO TARGETS, ONE SHAPE. Both are a CLI spawned as a job, so both get the same
live log, the same cancel, the same row on the Jobs page:

    local    build_video.py --stage mux      ffmpeg on this laptop
    colab    render_remote.py --worker URL   the same ffmpeg, on a free CPU
                                             runtime, driven by a bundle

WHY REMOTE IS FREE AND WHY IT MUST BE CPU. Colab bills compute units by
wall-clock on the runtime, not by GPU use, and an x264 encode uses no GPU at
all - so a GPU runtime costs 2-2.5x the entire monthly bill for exactly the
work a free CPU runtime does. `studio/colab_render.py` refuses to start on a
GPU runtime for that reason; this module never has to know about it.

CHOOSING REMOTE CHANGES WHERE THE FFMPEG RUNS AND NOTHING ELSE. The bundle
render_remote.py ships is built by `pipeline.renderjob` from the same plan and
the same provider objects `--stage mux` uses, so the two paths are the same
decisions executed in different places. A difference in the output is a bug,
not a setting.
"""

from __future__ import annotations

from . import jobs, paths, settings

paths.ensure_importable()
from pipeline import clips as clips_mod       # noqa: E402

MOTION_CHOICES = ("none", *sorted(clips_mod.PROVIDERS))
RENDER_TARGETS = ("local", "colab")


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
    cfg = settings.load()
    return {
        "ready": proj.render_plan.exists() and proj.voice.exists(),
        "has_render_plan": proj.render_plan.exists(),
        "has_voice": proj.voice.exists(),
        "targets": list(RENDER_TARGETS),
        "render_target": cfg.get("render_target", "local"),
        # Reported, not health-checked: `status` is called on every page load
        # and a dead tunnel takes 20s to time out. The Timeline page asks for
        # the live check separately, when the operator is looking at it.
        "render_worker_url": cfg.get("colab_render_url", ""),
        # Last known worker state, from whenever it was last probed - so the
        # page can say something without paying a 20s timeout on every load.
        "render_worker_last": cfg.get("render_last"),
        # The mastered voiceover, for the PRE-render preview: the timeline page
        # plays this against the slot images to show what the cut looks like
        # before ffmpeg is asked to spend minutes proving the same thing.
        "voice_path": str(proj.voice) if proj.voice.exists() else None,
        "motion_choices": list(MOTION_CHOICES),
        "has_video": out.exists(),
        "video_path": str(out) if out.exists() else None,
        "video_mb": round(out.stat().st_size / 1e6, 1) if out.exists() else 0,
        "video_mtime": out.stat().st_mtime if out.exists() else None,
    }


async def run(proj, motion: str = "kenburns", only_section: int | None = None,
              target: str | None = None):
    """Start the render as a job. Returns the Job.

    Needs render_plan.json (the Plan stage, with any timeline edits already
    baked in - see timeline.py) and full_vo.wav (the Voice stage) to already
    exist; mux reads both directly rather than recomputing anything, which is
    exactly why a roll edit has to write render_plan.json immediately instead
    of only living in overrides.json.

    `target` defaults to the saved setting, so the choice survives a reload
    rather than resetting to local every time the page mounts.
    """
    if motion not in MOTION_CHOICES:
        raise ValueError(f"unknown motion {motion!r} - choose one of {MOTION_CHOICES}")
    if not proj.render_plan.exists():
        raise ValueError("no render plan yet - run the Plan stage first")
    if not proj.voice.exists():
        raise ValueError("no mastered voiceover yet - run the Voice stage first")

    target = target or settings.load().get("render_target", "local")
    if target not in RENDER_TARGETS:
        raise ValueError(f"unknown render target {target!r} - choose one of "
                         f"{RENDER_TARGETS}")

    project = f"{proj.channel}/{proj.topic}"

    if target == "colab":
        url = settings.load().get("colab_render_url", "").strip()
        if not url:
            raise ValueError(
                "no render worker URL set - run studio/colab_render.py in a "
                "Colab CPU cell and paste the URL it prints, or switch the "
                "target back to local")
        argv = [
            paths.PYTHON, paths.RENDER_REMOTE,
            "--project", project,
            "--worker", url,
            "--motion", motion,
        ]
        # --only-section is a build_video concept; the bundle is always the
        # whole plan, so a section render stays local rather than silently
        # rendering more than was asked for.
        if only_section is not None:
            raise ValueError("a single-section render is local-only")
    else:
        argv = [
            paths.PYTHON, paths.BUILD_VIDEO,
            "--project", project,
            "--stage", "mux",
            "--motion", motion,
        ]
        if only_section is not None:
            argv += ["--only-section", str(int(only_section))]

    return await jobs.registry.start(
        kind="mux" if target == "local" else "mux-colab",
        project=project, argv=argv, cwd=paths.CHANNELS,
    )
