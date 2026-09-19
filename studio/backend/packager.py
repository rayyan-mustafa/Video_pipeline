"""The YouTube package: metadata, tags, and the thumbnail image prompt.

This is the last stage, and the only one whose output never enters a render. It
produces what gets typed into YouTube's upload form and pasted into an image
generator, so the UI's job here is not to display a report - it is to hand over
four exact strings: the title, the description, the comma-separated tags, and
the thumbnail prompt.

WHY THE JSON RATHER THAN THE .txt. `youtube_packager.py` writes metadata.txt and
thumbnail_prompt.txt for a human to read, and package.json with the same values
in copyable form. This module reads the JSON. Scraping the .txt would mean the
UI silently breaks whenever a divider or a heading in that file changes - and
the .txt exists precisely so it can be reformatted freely.

The two .txt files are still offered here as downloads, unparsed, because that
is the artefact the operator archives next to the video.
"""

from __future__ import annotations

import json

from . import jobs, paths

paths.ensure_importable()
from pipeline import config                    # noqa: E402

# YouTube's own limits, checked here rather than in the browser so the numbers
# live next to the code that knows what a package is. Exceeding any of these
# means the upload form silently truncates - worth a warning before the paste.
TITLE_MAX = 100
DESC_MAX = 5000
TAGS_MAX = 500                     # total characters across all tags, incl. commas

# The packager writes into <topic>/build/ and takes its sources from
# config.TOPICS, which only covers this channel's bank.
PACKAGER_CHANNEL = "wierdbiology"

# Where a finished thumbnail lands once the operator has generated it. Shown as
# a preview so the page can say whether this step is actually finished.
THUMB_NAMES = ("thumbnail.jpeg", "thumbnail.jpg", "thumbnail.png", "thumb.jpg")


def _file_info(path) -> dict:
    """A file the UI may download: existence, size, and an absolute path."""
    exists = path.is_file()
    return {
        "name": path.name,
        "abs_path": str(path),
        "exists": exists,
        "bytes": path.stat().st_size if exists else 0,
    }


def _thumbnail_image(proj) -> str | None:
    """The generated thumbnail, if one has been dropped into the project yet."""
    for name in THUMB_NAMES:
        for folder in (proj.dir, proj.build):
            p = folder / name
            if p.is_file():
                return str(p)
    return None


def _checks(pkg: dict) -> dict:
    """Whether each copyable field fits in the box it is going to be pasted in."""
    title = pkg.get("best_title", "") or ""
    desc = pkg.get("description", "") or ""
    tags = pkg.get("tags", "") or ""
    return {
        "title_chars": len(title), "title_max": TITLE_MAX,
        "title_ok": len(title) <= TITLE_MAX,
        "desc_chars": len(desc), "desc_max": DESC_MAX,
        "desc_ok": len(desc) <= DESC_MAX,
        "tags_chars": len(tags), "tags_max": TAGS_MAX,
        "tags_ok": len(tags) <= TAGS_MAX,
        "n_tags": len(pkg.get("tag_list", [])),
    }


def status(proj) -> dict:
    """What the packager has produced for this project, and whether it can run.

    Reports FACTS the same way projects.project_status does - `can_run` is about
    whether the CLI would accept this project at all, not about whether running
    it now is a good idea.
    """
    build = proj.build
    pkg_path = build / "package.json"

    pkg = None
    stale = False
    if pkg_path.is_file():
        try:
            pkg = json.loads(pkg_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pkg = None                       # half-written; treated as absent

    # A package produced before this JSON existed leaves the .txt files on disk
    # with no machine-readable twin. That is a re-run, not an error - say so
    # rather than showing an empty page.
    if pkg is None and (build / "metadata.txt").is_file():
        stale = True

    why = ""
    can_run = True
    if proj.channel != PACKAGER_CHANNEL:
        can_run = False
        why = (f"youtube_packager.py reads sources from config.TOPICS, which only "
               f"covers the {PACKAGER_CHANNEL} bank - {proj.channel} is not in it.")
    elif proj.topic not in config.TOPICS:
        can_run = False
        why = (f"{proj.topic!r} is not one of the {len(config.TOPICS)} topics in "
               f"config.TOPICS, so it has no curated sources to package from.")

    # The hook is what grounds the titles. Its absence is not a blocker - the
    # packager falls back to source excerpts - but it makes for weaker titles,
    # and the operator should know which of the two happened before running.
    has_hook = (proj.sections / "section_01.txt").is_file() or proj.script.exists()

    return {
        "can_run": can_run,
        "why": why,
        "has_package": pkg is not None,
        "stale": stale,
        "has_hook": has_hook,
        "package": pkg,
        "checks": _checks(pkg) if pkg else None,
        "thumbnail_image": _thumbnail_image(proj),
        "files": {
            "metadata": _file_info(build / "metadata.txt"),
            "thumbnail_prompt": _file_info(build / "thumbnail_prompt.txt"),
            "package_json": _file_info(pkg_path),
        },
    }


async def run(proj):
    """Start youtube_packager.py as a job. Returns the Job.

    Two model calls and a handful of source fetches - a minute or so, which is
    long enough that it belongs in the job registry with everything else rather
    than blocking a request.
    """
    argv = [
        paths.PYTHON, paths.YOUTUBE_PACKAGER,
        "--topic", proj.topic,
    ]
    return await jobs.registry.start(
        kind="package", project=f"{proj.channel}/{proj.topic}",
        argv=argv, cwd=paths.CHANNELS,
    )
