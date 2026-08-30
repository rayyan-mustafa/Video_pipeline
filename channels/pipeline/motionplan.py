"""Which slots get an external Flow clip, and which stay stills.

This is the SELECTION half of motion. The INGEST half - conforming the mp4 onto
the slot - is `clips.VideoAssetProvider`. They are separate because selection is
arithmetic on measured holds and ingest is ffmpeg, and only the first one has an
opinion about editorial.

THE POLICY, ALL OF IT ALREADY DECIDED ELSEWHERE
    A motion clip is 4.0s, which is exactly `pacing.HARD_MAX`. So a clip is the
    longest hold the spec allows, and motion belongs on the slots nearest that
    ceiling - they are the ones most at risk of reading as slow, and motion is
    what makes them watchable.

    ACCENTS NEVER GET MOTION. A hold under 2s is emphasis; 4s of video does not
    fit in it. The [3.5, 4.0] window excludes them automatically.

    MOTION SLOTS ARE NOT EXTRA RUNTIME. A clip REPLACES a still, it never
    appends 4s. Nothing here changes a single timestamp.

    NEVER TWO IN A ROW. Two adjacent motion slots read as one eight-second shot
    with a cut buried in it, which erases the cut - the same reason
    `motion.ROTATION` exists for Ken Burns.

    PLACEMENT IS OPPORTUNISTIC. If no slot in a stretch fits the window, that
    stretch stays stills. The script is never bent to make room, and a hold is
    never stretched past HARD_MAX to fit a clip.

WHY THIS REFUSES TO FILL THE DOSE, AND WHY THAT IS THE POINT
    A slot only qualifies if its hold is ALREADY in [3.5, 4.0]. A 15-second hold
    is where motion is most wanted and is exactly where it cannot go: covering it
    would mean slowing the clip (the eye reads speed - a slow drift stretched
    over 15s reads as a still) or freezing it (which is worse). Both are banned.

    So on an under-planned video this function will find far fewer slots than the
    10-15% dose asks for, and it says so rather than quietly placing what it can.
    The fix for that is upstream: re-plan at the picture count the pacing verdict
    asks for, which pulls the long holds down INTO the window. Measured on chin -
    105 pictures over 594s - only 6 slots qualify against a dose of 14-22,
    because 65 of its holds are over the ceiling. Re-planned at ~198 pictures
    most of those land in the band and become candidates.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from . import pacing

# What Flow exports and ffmpeg reads back. Lives here, not in the backend, so
# a CLI tool (assetsort.py) and the studio backend (motionstage.py) share one
# definition instead of two that could drift.
VIDEO_EXTS = (".mp4", ".mov", ".webm", ".m4v")

# At least this many still slots between two motion slots. One is the minimum
# that stops them reading as a single long shot; the selector spreads them
# further than this whenever the candidates allow it.
MIN_GAP = 2


@dataclass
class Candidate:
    index: int
    hold: float
    section: int
    filename: str

    def as_dict(self) -> dict:
        return {"index": self.index, "hold": round(self.hold, 3),
                "section": self.section, "filename": self.filename}


@dataclass
class Selection:
    chosen: list[Candidate] = field(default_factory=list)
    candidates: int = 0
    want_low: int = 0
    want_high: int = 0
    runtime: float = 0.0
    note: str = ""

    @property
    def short(self) -> int:
        """How many clips below the dose's lower bound this selection lands."""
        return max(0, self.want_low - len(self.chosen))

    def as_dict(self) -> dict:
        return {
            "version": 1,
            "runtime": round(self.runtime, 2),
            "clip_seconds": pacing.MOTION_CLIP_S,
            "window": list(pacing.MOTION_SLOT_WINDOW),
            "want": [self.want_low, self.want_high],
            "candidates": self.candidates,
            "selected": len(self.chosen),
            "short_by": self.short,
            "note": self.note,
            # The operator edits THIS list. "which line deserves emphasis" is a
            # taste call, so the planner proposes and does not impose.
            "slots": [c.as_dict() for c in self.chosen],
        }


def candidates(plan: list[dict]) -> list[Candidate]:
    """Every slot whose measured hold could carry a 4s clip as it stands."""
    out = []
    for item in plan:
        hold = float(item.get("duration", 0.0))
        if pacing.fits_motion(hold):
            out.append(Candidate(
                index=int(item.get("index", len(out))),
                hold=hold,
                section=int(item.get("section", 0)),
                filename=Path(str(item.get("path", ""))).name,
            ))
    return out


def select(plan: list[dict], share: float | None = None,
           min_gap: int = MIN_GAP) -> Selection:
    """Choose the slots that get motion, from a render plan's measured holds.

    Ranking is by hold DESCENDING - closest to the 4.0s ceiling first, because
    those are the slots the ceiling rule is about. Ties break on index so a
    re-run is identical.
    """
    holds = [float(i.get("duration", 0.0)) for i in plan]
    runtime = sum(holds)
    dose = pacing.motion_dose(runtime)
    lo, hi = dose["clips_low"], dose["clips_high"]
    if share is not None:
        lo = hi = pacing.motion_slots_for(runtime, share)

    pool = candidates(plan)
    target = hi

    # Greedy, highest hold first, skipping anything too close to one already
    # taken. Greedy is right here rather than an optimiser: the candidates are
    # few, the objective (near the ceiling, spread out) is not in tension for
    # realistic inputs, and a deterministic result matters more than optimality.
    chosen: list[Candidate] = []
    for c in sorted(pool, key=lambda c: (-c.hold, c.index)):
        if len(chosen) >= target:
            break
        if any(abs(c.index - s.index) <= min_gap for s in chosen):
            continue
        chosen.append(c)
    chosen.sort(key=lambda c: c.index)

    sel = Selection(chosen=chosen, candidates=len(pool),
                    want_low=lo, want_high=hi, runtime=runtime)

    if not pool:
        sel.note = (
            f"No slot holds between {pacing.MOTION_SLOT_MIN} and "
            f"{pacing.MOTION_CLIP_S}s, so nothing can carry a clip without "
            f"slowing or freezing it. Both are banned.")
    elif sel.short:
        over = sum(1 for h in holds if h > pacing.HARD_MAX)
        sel.note = (
            f"Only {len(chosen)} slot(s) can take a clip, against a dose of "
            f"{lo}-{hi}. {over} hold(s) are over the {pacing.HARD_MAX}s ceiling "
            f"- that is where motion is most wanted and exactly where a 4s clip "
            f"does not fit. Re-plan at ~{pacing.images_for(runtime)} pictures to "
            f"pull those holds into the window.")
    else:
        sel.note = (
            f"{len(chosen)} slot(s) selected from {len(pool)} candidate(s), "
            f"inside the {lo}-{hi} dose "
            f"({len(chosen) * pacing.MOTION_CLIP_S / runtime * 100:.1f}% of runtime).")
    return sel


# ---------------------------------------------------------------------------
#  Persistence
# ---------------------------------------------------------------------------
def write(proj, sel: Selection) -> Path:
    """Save the proposal to `build/motion_slots.json` for the operator to edit."""
    path = proj.build / "motion_slots.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sel.as_dict(), indent=2), encoding="utf-8")
    return path


def read(proj) -> dict | None:
    """The saved selection, including any hand edits. None if never written."""
    path = proj.build / "motion_slots.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def clip_for(proj, index: int, saved: dict | None = None) -> Path | None:
    """The Flow mp4 assigned to a slot, if one has been dropped in.

    A slot may name its clip explicitly (`"clip": "flow_012.mp4"`); otherwise the
    convention is `motion/<index:03d>.mp4`. Returns None when the file is not
    there yet, so the UI can show which slots are still waiting on Flow.
    """
    saved = saved if saved is not None else read(proj)
    named = None
    for slot in (saved or {}).get("slots", []):
        if int(slot.get("index", -1)) == index:
            named = slot.get("clip")
            break

    for cand in ([named] if named else []) + [f"{index:03d}.mp4"]:
        p = proj.motion / cand
        if p.is_file():
            return p
    return None


def attach_kinds(entries: list[dict], script_plan: dict, proj,
                 motion_overrides: dict | None = None) -> dict:
    """Mutate `entries` in place: attach 'kind' ('image'|'video') and, for a
    motion slot, 'clip_path' if a Flow mp4 is actually present. Extracted from
    build_video.py's --stage plan so the CLI and the timeline editor's live
    recompute can never attach motion two different ways to the same plan.

    `motion_overrides` is the timeline editor's third override key, keyed by
    OUTPUT slot index same as `cuts`/`assets` (see Project.overrides):
        {"12": "flow_012.mp4"}   force this slot to video, using this exact
                                  file in proj.motion
        {"12": "off"}            force this slot to a still even though the
                                  plan marked it as motion
    Forcing a slot ON is refused, with a warning, when its hold does not fit
    `pacing.fits_motion` - a 4s clip cannot cover a hold outside [3.5, 4.0]
    without slowing or freezing it, and both are banned regardless of what the
    timeline editor was asked to do. The toggle can only choose AMONG slots
    the pacing spec already allows, never override the spec itself.
    """
    motion_overrides = motion_overrides or {}
    kinds = {(int(i["section"]), int(i["order"])): i.get("kind", "image")
             for i in script_plan.get("images", [])}

    n_ready = 0
    warnings: list[str] = []
    for e in entries:
        idx = int(e["index"])
        ov = motion_overrides.get(str(idx), motion_overrides.get(idx))
        base_kind = kinds.get((int(e["section"]), int(e["order"])), "image")
        forced_off = isinstance(ov, str) and ov.lower() == "off"

        if ov is None or forced_off:
            e["kind"] = "image" if forced_off else base_kind
        elif not pacing.fits_motion(float(e["duration"])):
            warnings.append(
                f"slot {idx}: motion override ignored - {e['duration']:.2f}s hold "
                f"is outside the {pacing.MOTION_SLOT_MIN}-{pacing.MOTION_CLIP_S}s "
                f"window a clip can cover"
            )
            e["kind"] = base_kind
        else:
            e["kind"] = "video"

        if e["kind"] != "video":
            e.pop("clip_path", None)
            continue

        clip = None
        if isinstance(ov, str) and not forced_off:
            named = proj.motion / ov
            if named.is_file():
                clip = named
            else:
                warnings.append(f"slot {idx}: override clip {ov!r} not found in "
                                f"{proj.motion}, falling back to the default lookup")
        if clip is None:
            clip = clip_for(proj, idx)

        if clip:
            e["clip_path"] = str(clip)
            n_ready += 1
        else:
            e.pop("clip_path", None)

    n_motion = sum(1 for e in entries if e["kind"] == "video")
    return {"n_motion": n_motion, "n_ready": n_ready, "warnings": warnings}


def status(proj, plan: list[dict] | None = None) -> dict:
    """What the UI needs: the proposal, and which clips have actually arrived."""
    saved = read(proj)
    if saved is None and plan:
        saved = select(plan).as_dict()
    if saved is None:
        return {"planned": False, "slots": [], "ready": 0, "waiting": 0}

    slots = []
    for s in saved.get("slots", []):
        clip = clip_for(proj, int(s["index"]), saved)
        slots.append({**s, "clip": clip.name if clip else None,
                      "clip_path": str(clip.resolve()) if clip else None,
                      "have": clip is not None})
    return {
        **saved,
        "planned": True,
        "slots": slots,
        "ready": sum(1 for s in slots if s["have"]),
        "waiting": sum(1 for s in slots if not s["have"]),
        "motion_dir": str(proj.motion),
    }
