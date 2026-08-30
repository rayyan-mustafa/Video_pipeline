"""The timeline editor: slots + words + pickers in one response, and the
endpoint that recomputes render_plan.json after an edit.

WHY THIS RECOMPUTES RATHER THAN TRUSTS THE BROWSER. Every other stage here
writes its output file and hands back what actually landed rather than
echoing the request - the plan on disk is the one description of the build.
A roll edit is no different: it goes through the exact same
imageplan.plan_from_binding + motionplan.attach_kinds the CLI's --stage plan
uses (see build_video.py), so the browser can never see a timeline the real
build would compute differently.

WHY require_images=False HERE, UNLIKE THE CLI'S DEFAULT. The editor exists to
be used mid-production - one section's pictures in, the rest still on order -
so demanding every picture in the whole video before a single cut can be
nudged would make the tool useless for exactly the workflow it is for. The
CLI's own --stage plan (no --only-section) still enforces require_images=True
at build time, unchanged; this module governs editing, not the build gate.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import assets as assets_mod
from . import overrides as overrides_mod
from . import paths

paths.ensure_importable()
from pipeline import imageplan, motionplan, pacing        # noqa: E402


def _load_inputs(proj):
    """script_plan + timeline + words, or None if a prerequisite is missing."""
    if not (proj.plan.exists() and proj.timeline.exists()):
        return None
    script_plan = json.loads(proj.plan.read_text(encoding="utf-8"))
    timeline = json.loads(proj.timeline.read_text(encoding="utf-8"))
    words_path = proj.build / "words.json"
    words = (json.loads(words_path.read_text(encoding="utf-8"))
             if words_path.exists() else None)
    return script_plan, timeline, words


def _recompute(proj, ovr: dict) -> tuple[list[dict], list[str]]:
    loaded = _load_inputs(proj)
    if loaded is None:
        raise ValueError("no plan/timeline yet - run the Plan stage, then Voice")
    script_plan, timeline, words = loaded

    entries, warnings = imageplan.plan_from_binding(
        script_plan, timeline, images_dir=proj.images, words=words,
        require_images=False, overrides=ovr)
    attach = motionplan.attach_kinds(entries, script_plan, proj,
                                     motion_overrides=ovr.get("motion"))
    return entries, warnings + attach["warnings"]


def _slot_view(e: dict) -> dict:
    p = Path(e["path"])
    out = dict(e)
    out["abs_path"] = str(p.resolve())
    out["has_file"] = p.exists()
    out["is_motion"] = e.get("kind") == "video"
    out["fits_motion"] = pacing.fits_motion(float(e["duration"]))
    if e.get("clip_path"):
        out["clip_abs_path"] = str(Path(e["clip_path"]).resolve())
    return out


def _empty(proj, why: str) -> dict:
    return {"ready": False, "why": why, "slots": [], "warnings": [],
           "words": [], "overrides": overrides_mod.read(proj),
           "available_images": [], "available_clips": [], "fingerprint": None}


def read(proj) -> dict:
    if not (proj.plan.exists() and proj.timeline.exists()):
        return _empty(proj, "No plan yet - run the Plan stage first.")

    ovr = overrides_mod.read(proj)
    try:
        entries, warnings = _recompute(proj, ovr)
    except imageplan.PlanError as e:
        return _empty(proj, str(e))

    words_path = proj.build / "words.json"
    words = (json.loads(words_path.read_text(encoding="utf-8"))
             if words_path.exists() else [])

    scan = assets_mod.scan(proj)
    available_images = sorted({im["name"] for im in scan.get("images", [])}
                              | set(scan.get("alternates", [])))
    if not scan.get("ok"):
        # Not fatal to the timeline itself - the plan above was already built
        # straight from script_plan.json's own filenames, a separate path from
        # scan_images. Only the REPLACE picker needs this list, so report the
        # reason and hand back an empty one rather than blocking the editor.
        # scan_images' own error can list every bad file in the folder (seen:
        # 100+ lines for a folder full of unrelated reference art) - keep only
        # the headline here, the full list is still in the Images page itself.
        reason = str(scan.get("error") or "").splitlines()[0]
        warnings = warnings + [f"image picker unavailable: {reason}"]
    available_clips = sorted(
        p.name for p in proj.motion.iterdir()
        if p.suffix.lower() in motionplan.VIDEO_EXTS
    ) if proj.motion.is_dir() else []

    return {
        "ready": True,
        "why": "",
        "slots": [_slot_view(e) for e in entries],
        "warnings": warnings,
        "words": words,
        "overrides": ovr,
        "available_images": available_images,
        "available_clips": available_clips,
        "fingerprint": ovr.get("fingerprint"),
    }


def _validate(proj, ovr: dict) -> None:
    for idx, name in ovr.get("assets", {}).items():
        if not (proj.images / name).is_file():
            raise ValueError(
                f"asset override for slot {idx}: {name!r} is not in {proj.images}")
    for idx, name in ovr.get("motion", {}).items():
        if isinstance(name, str) and name.lower() != "off" \
                and not (proj.motion / name).is_file():
            raise ValueError(
                f"motion override for slot {idx}: {name!r} is not in {proj.motion}")


def save(proj, patch: dict) -> dict:
    """Merge `patch` into overrides.json, recompute, write render_plan.json,
    return the fresh payload. Validated and written in that order - a bad
    reference never reaches disk, and disk always reflects what the response
    shows."""
    merged = overrides_mod.merge(proj, patch)
    _validate(proj, merged)

    entries, _warnings = _recompute(proj, merged)
    overrides_mod.write(proj, merged)
    proj.render_plan.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    return read(proj)


def revert(proj) -> dict:
    """Drop every override and recompute the plan clean."""
    overrides_mod.clear(proj)
    if proj.plan.exists() and proj.timeline.exists():
        entries, _ = _recompute(proj, overrides_mod.read(proj))
        proj.render_plan.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    return read(proj)
