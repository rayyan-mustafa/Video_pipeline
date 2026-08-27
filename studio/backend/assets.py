"""The image folder: what is in it, what is wrong with it, and adding more.

Validation is `imageplan.scan_images` verbatim - the same parser the build uses.
A UI that accepted a filename the build later rejects would be worse than no UI,
because the error would surface after recording rather than before.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import UploadFile

from . import paths

paths.ensure_importable()
from pipeline import imageplan, pacing            # noqa: E402


def scan(proj) -> dict:
    """Every usable picture, plus whatever is wrong with the folder.

    Returns `ok: False` with a readable reason rather than raising, because a
    half-named folder is the normal state while someone is still copying files
    in - it is a thing to display, not an exception.
    """
    if not proj.images.is_dir():
        return {"ok": False, "error": "no images folder yet", "images": [],
                "alternates": [], "by_section": {}}

    try:
        items = imageplan.scan_images(proj.images)
    except imageplan.PlanError as e:
        return {"ok": False, "error": str(e), "images": [],
                "alternates": [], "by_section": {}}

    # scan_images stashes the skipped regenerations on the first entry, so the
    # UI can show "17 alternate takes not used" instead of silently hiding them.
    notes = items[0].pop("_notes", []) if items else []

    by_section: dict[int, int] = {}
    for it in items:
        by_section[it["section"]] = by_section.get(it["section"], 0) + 1

    on_disk = sorted(p for p in proj.images.iterdir()
                     if p.suffix.lower() in imageplan.IMAGE_EXTS)
    used = {Path(i["path"]).name for i in items}

    return {
        "ok": True,
        "error": None,
        "count": len(items),
        "on_disk": len(on_disk),
        "images": [{
            "name": Path(i["path"]).name,
            "abs_path": str(Path(i["path"]).resolve()),
            "section": i["section"],
            "order": i["order"],
            "variant": i.get("variant", 1),
        } for i in items],
        # Not used by the plan, but still available to swap in from the editor.
        "alternates": [p.name for p in on_disk if p.name not in used],
        "by_section": {str(k): v for k, v in sorted(by_section.items())},
        "notes": notes,
    }


def pacing_table(proj, wpm: int = 150) -> dict:
    """Per-section image counts against the pacing target, before any audio.

    Mirrors build_video.check_images, but returns data instead of printing, so
    the UI can colour rows rather than parse a table out of stdout.
    """
    from . import scripts

    analysis = scripts.analyse(proj, wpm=wpm)
    counts = scan(proj).get("by_section", {})

    rows, holds = [], []
    for sec in analysis["sections"]:
        n = int(counts.get(str(sec["n"]), 0))
        span = sec["est_seconds"]
        hold = (span / n) if n else 0.0
        if n:
            holds.extend([hold] * n)
        rows.append({
            "section": sec["n"],
            "words": sec["words"],
            "images": n,
            "est_span": span,
            "est_hold": round(hold, 2),
            "flag": (
                "no images" if n == 0
                else "flashes" if hold < pacing.MIN_HOLD
                else f"over {pacing.HARD_MAX:.0f}s" if hold > pacing.HARD_MAX
                else "very fast" if hold < pacing.TARGET_LOW
                else ""
            ),
        })

    return {"rows": rows, "verdict": pacing.verdict(holds) if holds else None}


async def upload(proj, files: list[UploadFile]) -> dict:
    """Save uploaded pictures, then re-validate the whole folder.

    Names are taken from the upload as-is - the build reads section and order
    straight out of the filename, so silently renaming here would silently
    re-time the video. A bad name is reported by the rescan instead.
    """
    proj.images.mkdir(parents=True, exist_ok=True)
    saved, rejected = [], []

    for f in files:
        name = Path(f.filename or "").name          # strip any directory part
        if not name:
            rejected.append({"name": f.filename, "why": "no filename"})
            continue
        if Path(name).suffix.lower() not in imageplan.IMAGE_EXTS:
            rejected.append({"name": name, "why": "not an image type"})
            continue
        dest = proj.images / name
        with dest.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        saved.append(name)

    result = scan(proj)
    result["saved"] = saved
    result["rejected"] = rejected
    return result
