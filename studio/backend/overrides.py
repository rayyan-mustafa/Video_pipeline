"""build/overrides.json - the timeline editor's only persisted output.

Pure storage: read, merge, validate shape, write. Recomputing the plan against
these overrides is timeline.py's job - this module never imports imageplan and
never touches render_plan.json, so "what the editor saved" and "what the plan
looks like" stay two separate, individually-testable questions.

    { "version": 1, "fingerprint": "cda70082ce6cc9d4",
      "cuts": {"7": 42.31}, "assets": {"12": "9section3_....jpeg"},
      "motion": {"12": "flow_012.mp4"} }

A save always re-stamps `fingerprint` from the CURRENT script_plan.json - it
describes "this edit was made against this binding", not a value the caller
can set. imageplan.plan_from_binding is what actually enforces staleness (warn
and ignore rather than apply), so this module's job is narrower: never persist
an edit without knowing which binding it was made against.
"""

from __future__ import annotations

import json

from . import paths

paths.ensure_importable()

EMPTY: dict = {"version": 1, "fingerprint": None, "cuts": {}, "assets": {}, "motion": {}}


def _current_fingerprint(proj) -> str | None:
    if not proj.plan.exists():
        return None
    try:
        return json.loads(proj.plan.read_text(encoding="utf-8")).get("fingerprint")
    except (json.JSONDecodeError, OSError):
        return None


def read(proj) -> dict:
    """The overrides on disk, normalised - never raises on a missing or
    half-written file, since 'no edits yet' is the normal starting state."""
    if not proj.overrides.exists():
        return dict(EMPTY, fingerprint=_current_fingerprint(proj))
    try:
        data = json.loads(proj.overrides.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return dict(EMPTY, fingerprint=_current_fingerprint(proj))
    return {
        "version": data.get("version", 1),
        "fingerprint": data.get("fingerprint"),
        "cuts": {str(k): float(v) for k, v in (data.get("cuts") or {}).items()},
        "assets": {str(k): str(v) for k, v in (data.get("assets") or {}).items()},
        "motion": {str(k): v for k, v in (data.get("motion") or {}).items()},
    }


def _merge_section(existing: dict, patch: dict | None) -> dict:
    """A key mapped to null in the patch REMOVES that override - the only way
    to delete one, short of hand-editing the file. Everything else is a plain
    upsert."""
    if patch is None:
        return dict(existing)
    out = dict(existing)
    for k, v in patch.items():
        if v is None:
            out.pop(str(k), None)
        else:
            out[str(k)] = v
    return out


def merge(proj, patch: dict) -> dict:
    """Merge a partial patch - any of {"cuts", "assets", "motion"} - into the
    stored overrides, re-stamped against the CURRENT plan fingerprint. Raises
    if there is no plan yet: an override with no binding to describe is not a
    valid state to save, not even an empty one."""
    fp = _current_fingerprint(proj)
    if fp is None:
        raise ValueError("no script_plan.json yet - run the Plan stage first")

    current = read(proj)
    return {
        "version": 1,
        "fingerprint": fp,
        "cuts": _merge_section(current["cuts"], patch.get("cuts")),
        "assets": _merge_section(current["assets"], patch.get("assets")),
        "motion": _merge_section(current["motion"], patch.get("motion")),
    }


def write(proj, overrides: dict) -> None:
    proj.build.mkdir(parents=True, exist_ok=True)
    proj.overrides.write_text(json.dumps(overrides, indent=2), encoding="utf-8")


def clear(proj) -> None:
    if proj.overrides.exists():
        proj.overrides.unlink()
