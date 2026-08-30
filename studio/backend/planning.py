"""Running plan_script.py, and reading back what it produced.

The plan is the contract between the script and the pictures: it binds each
image to the exact sentences it illustrates and carries a fingerprint of those
sentence texts. Everything downstream trusts it, so this module never writes it
by hand - it always goes through the CLI, and only reads the result.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import jobs, paths

paths.ensure_importable()
from pipeline import pacing                       # noqa: E402


async def run(proj, split: str = "match",
              seconds_per_image: float | None = None,
              topic: str | None = None,
              allow_brackets: bool = False,
              motion_clips: int | None = None):
    """Start plan_script.py as a job. Returns the Job.

    `match` reads the images already on disk and divides the script into exactly
    that many scenes, giving a 1:1 binding. It is the right default when the
    pictures already exist, and the only mode that cannot be re-derived from the
    script alone.
    """
    argv = [
        paths.PYTHON, paths.PLAN_SCRIPT,
        "--project", f"{proj.channel}/{proj.topic}",
        "--split", split,
        "--seconds-per-image",
        str(seconds_per_image if seconds_per_image is not None
            else pacing.DEFAULT_SECONDS_PER_IMAGE),
    ]
    if topic:
        argv += ["--topic", topic]
    if allow_brackets:
        argv += ["--allow-brackets"]
    if motion_clips is not None:
        argv += ["--motion-clips", str(motion_clips)]

    return await jobs.registry.start(
        kind="plan", project=f"{proj.channel}/{proj.topic}",
        argv=argv, cwd=paths.CHANNELS,
    )


async def run_prompts(proj):
    """Start make_nb_prompts.py as a job to generate the beat sheet."""
    argv = [
        paths.PYTHON, paths.CHANNELS / "make_nb_prompts.py",
        "--project", f"{proj.channel}/{proj.topic}"
    ]
    return await jobs.registry.start(
        kind="prompts", project=f"{proj.channel}/{proj.topic}",
        argv=argv, cwd=paths.CHANNELS,
    )


def read(proj) -> dict | None:
    """script_plan.json, with paths made browser-safe.

    The file stores Windows relative paths like
    `wierdbiology\\chin\\images\\1section1_....jpeg`. The UI fetches pictures
    through /api/file, which wants an absolute path, so each image gains an
    `abs_path` while the original `path` is left untouched - rewriting it would
    make the JSON the UI sees differ from the JSON the pipeline reads.
    """
    if not proj.plan.exists():
        return None
    try:
        plan = json.loads(proj.plan.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    for img in plan.get("images", []):
        raw = Path(str(img.get("path", "")).replace("\\", "/"))
        img["abs_path"] = str((paths.CHANNELS / raw).resolve()
                              if not raw.is_absolute() else raw)

    # Estimated hold per picture from word counts alone - the Plan page needs a
    # pacing verdict before any audio exists.
    total_words = sum(i.get("words", 0) for i in plan.get("images", []))
    est_total = pacing.seconds_for_words(total_words, plan.get("wpm", 150))
    holds = [
        (i.get("words", 0) / total_words * est_total) if total_words else 0.0
        for i in plan.get("images", [])
    ]
    plan["pacing"] = pacing.verdict(holds)
    plan["est_seconds"] = round(est_total, 1)
    return plan


def markdown(proj) -> str:
    p = proj.build / "script_plan.md"
    return p.read_text(encoding="utf-8") if p.exists() else ""
