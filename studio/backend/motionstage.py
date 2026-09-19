"""Motion slots: which ones get a Flow clip, and which clips have arrived.

Thin on purpose. `pipeline.motionplan` owns the policy and `pipeline.clips`
owns the ingest; this only reads a project's files and hands the UI a shape it
can render. Anything that looks like a decision about editorial belongs in
motionplan.py, so the CLI and the UI can never disagree about it.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import UploadFile

from . import paths

paths.ensure_importable()
from pipeline import motionplan, pacing        # noqa: E402

# Kept as one definition in pipeline.motionplan, shared with assetsort.py.
VIDEO_EXTS = motionplan.VIDEO_EXTS


def _plan(proj) -> list[dict] | None:
    """The render plan's measured holds - the only honest basis for selection."""
    if not proj.render_plan.exists():
        return None
    try:
        return json.loads(proj.render_plan.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def status(proj) -> dict:
    """Everything the Motion page renders.

    `blocked` is the one judgement, and it means "there are no measured holds
    yet". Estimated holds cannot answer which slots sit at the ceiling, so
    offering a selection from them would be a guess dressed as a plan.
    """
    plan = _plan(proj)
    if plan is None:
        return {
            "blocked": True,
            "why": ("No measured holds yet. Run the Voice stage, then the Plan "
                    "stage — which slots sit at the ceiling is only knowable "
                    "from real Whisper timings."),
            "slots": [], "planned": False,
        }

    saved = motionplan.read(proj)
    proposal = motionplan.select(plan)
    state = motionplan.status(proj, plan)

    files = sorted(p.name for p in proj.motion.iterdir()
                   if p.suffix.lower() in VIDEO_EXTS) if proj.motion.is_dir() else []

    return {
        "blocked": False,
        "why": "",
        "saved": saved is not None,
        "proposal": proposal.as_dict(),
        **state,
        "clips_on_disk": files,
        "motion_dir": str(proj.motion),
        "dose": pacing.motion_dose(sum(float(i.get("duration", 0.0)) for i in plan)),
    }


def propose(proj) -> dict:
    """Recompute the selection from measured holds, without saving it."""
    plan = _plan(proj)
    if plan is None:
        raise ValueError("no render_plan.json - run the Voice and Plan stages first")
    return motionplan.select(plan).as_dict()


def save(proj, slots: list[dict] | None = None) -> dict:
    """Write `build/motion_slots.json`.

    With no `slots` the planner's proposal is saved as-is. With `slots` the
    operator's own list wins — "which line deserves emphasis" is a taste call,
    and the planner proposes rather than imposes. Hand-picked slots are still
    checked against the window, because a clip physically cannot cover a hold
    outside it.
    """
    plan = _plan(proj)
    if plan is None:
        raise ValueError("no render_plan.json - run the Voice and Plan stages first")

    sel = motionplan.select(plan)
    if slots is not None:
        by_index = {int(i.get("index", -1)): i for i in plan}
        chosen = []
        for s in slots:
            idx = int(s["index"])
            item = by_index.get(idx)
            if item is None:
                raise ValueError(f"slot {idx} is not in the render plan")
            hold = float(item["duration"])
            if not pacing.fits_motion(hold):
                raise ValueError(
                    f"slot {idx} holds {hold:.2f}s, outside the "
                    f"{pacing.MOTION_SLOT_MIN}-{pacing.MOTION_CLIP_S}s window. "
                    f"A {pacing.MOTION_CLIP_S}s clip cannot cover it without "
                    f"slowing or freezing, and both are banned.")
            chosen.append(motionplan.Candidate(
                index=idx, hold=hold,
                section=int(item.get("section", 0)),
                order=int(item.get("order", 0)),
                filename=Path(str(item.get("path", ""))).name))
        chosen.sort(key=lambda c: c.index)
        sel.chosen = chosen
        sel.note = f"{len(chosen)} slot(s) chosen by hand."

    motionplan.write(proj, sel)
    return status(proj)


async def upload(proj, files: list[UploadFile]) -> dict:
    """Save Flow clips into `<project>/motion/`, named as given.

    Names are kept verbatim for the same reason images are: a slot resolves its
    clip by name (`flow_012.mp4`) or by the `<index>.mp4` convention, so
    renaming here would silently re-point a slot.
    """
    proj.motion.mkdir(parents=True, exist_ok=True)
    saved, skipped = [], []
    for f in files:
        name = Path(f.filename or "").name
        if not name or Path(name).suffix.lower() not in VIDEO_EXTS:
            skipped.append({"name": name or "(unnamed)",
                            "why": f"not one of {', '.join(VIDEO_EXTS)}"})
            continue
        (proj.motion / name).write_bytes(await f.read())
        saved.append(name)
    return {"saved": saved, "skipped": skipped, "status": status(proj)}
