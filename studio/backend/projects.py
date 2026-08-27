"""Projects: discovery, creation, and the status the UI navigates by.

Everything here wraps `pipeline.project.Project` rather than reimplementing path
logic - that class is already the single source of truth for where a video's
files live, and duplicating it would guarantee the two drift.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import paths

paths.ensure_importable()
from pipeline.project import Project          # noqa: E402

AUDIO_EXTS = (".wav", ".flac", ".m4a", ".mp3")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

# A channel folder is any directory under channels/ that is not part of the
# machinery. Without this the UI would offer "pipeline" and "__pycache__" as
# channels to make videos in.
NOT_CHANNELS = {
    "pipeline", "__pycache__", ".venv", "_local_workdir", "_build", "_demo",
    "sections", "voice_recordings", "images", "build", "runs", "source_cache",
}


def _count(d: Path, exts: tuple[str, ...]) -> int:
    if not d.is_dir():
        return 0
    return sum(1 for p in d.iterdir() if p.suffix.lower() in exts)


def project_status(proj: Project) -> dict:
    """What exists on disk for this project, as the flags the UI gates on.

    Deliberately reports FACTS, not a computed "current step". Which stage to run
    next is a judgement the UI can make from these; baking it in here would mean
    two places disagree about what "ready" means.
    """
    plan = None
    if proj.plan.exists():
        try:
            plan = json.loads(proj.plan.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            plan = None                       # half-written; treated as absent

    timeline_n = 0
    if proj.timeline.exists():
        try:
            timeline_n = len(json.loads(proj.timeline.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            pass

    out = proj.out
    return {
        "channel": proj.channel,
        "topic": proj.topic,
        "id": f"{proj.channel}/{proj.topic}",
        "dir": str(proj.dir),
        "has_script": proj.script.exists(),
        "script_words": (len(proj.script.read_text(encoding="utf-8").split())
                         if proj.script.exists() else 0),
        "sections": _count(proj.sections, (".txt",)),
        "recordings": _count(proj.recordings, AUDIO_EXTS),
        "images": _count(proj.images, IMAGE_EXTS),
        "has_plan": plan is not None,
        "plan_split": (plan or {}).get("split"),
        "plan_scenes": len((plan or {}).get("scenes", [])),
        "plan_images": len((plan or {}).get("images", [])),
        "fingerprint": (plan or {}).get("fingerprint"),
        "has_timeline": timeline_n > 0,
        "timeline_scenes": timeline_n,
        "has_words": (proj.build / "words.json").exists(),
        "has_voice": proj.voice.exists(),
        "has_render_plan": proj.render_plan.exists(),
        "has_overrides": (proj.build / "overrides.json").exists(),
        "has_video": out.exists(),
        "video_mb": round(out.stat().st_size / 1e6, 1) if out.exists() else 0,
    }


def list_projects() -> list[dict]:
    """Every <channel>/<topic> under channels/ that looks like a project."""
    found: list[dict] = []
    if not paths.CHANNELS.is_dir():
        return found

    for channel in sorted(paths.CHANNELS.iterdir()):
        if not channel.is_dir() or channel.name in NOT_CHANNELS \
                or channel.name.startswith((".", "_")):
            continue
        for topic in sorted(channel.iterdir()):
            if not topic.is_dir() or topic.name.startswith((".", "_")):
                continue
            # A project is a folder that has at least one of the things a
            # project has. This keeps stray directories out of the list.
            markers = ("script.txt", "sections", "images", "build",
                       "voice_recordings")
            if not any((topic / m).exists() for m in markers):
                continue
            try:
                proj = Project(channel.name, topic.name, root=paths.CHANNELS)
            except ValueError:
                continue                      # unsafe folder name; skip quietly
            found.append(project_status(proj))
    return found


def get(project_id: str) -> Project:
    """'wierdbiology/chin' -> Project. Raises ValueError on a malformed id."""
    return Project.parse(project_id, root=paths.CHANNELS)


def create(channel: str, topic: str) -> dict:
    """Make the folder skeleton. Idempotent - make_dirs() uses exist_ok."""
    proj = Project(channel, topic, root=paths.CHANNELS)   # validates the names
    existed = proj.dir.exists()
    proj.make_dirs()
    status = project_status(proj)
    status["created"] = not existed
    return status
