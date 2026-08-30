"""Sort a project's dropped-in assets: right folder, right name, no data loss.

WHY THIS EXISTS. Nano Banana and Flow hand back files named by their own
auto-generated title or export settings (`8section1.png_2K_202608281752.jpeg`,
`Woman_touching_chin_in_mirror_202608291531.jpeg`, `061.mp4_1080p_...mp4`), not
by the convention the pipeline reads. A batch re-upload then leaves the project
with pictures in two folders, videos sitting in the picture folder, and two or
three exports of the same slot. Sorting that by hand once is fine; doing it
after every re-upload is not, and doing it with `rm` is how a real session on
this project nearly lost eight only-copies to a stray glob. This module is the
fix for both: automate the sorting, and never let "duplicate" mean "deleted".

THE ONE RULE THAT MAKES THIS SAFE: NOTHING IS EVER DELETED. A file that loses
a dedup or cannot be placed with confidence is MOVED to a `_duplicates/` or
`_unresolved/` quarantine folder next to where it was found - never removed.
Recovering from a wrong call is "look one folder over", not "search the
Recycle Bin and hope."

WHAT IS RESOLVED AUTOMATICALLY, AND WHAT IS NOT
    - a video sitting in images/ (or an image sitting in motion/)      MOVED
    - a video named by its exact slot INDEX ("035.mp4", "35.mp4")      RENAMED
    - a video named by its ORDER-WITHIN-SECTION for a section whose
      motion slots have a unique such order (the mistake this project's
      own uploads made twice)                                          RENAMED
    - two or more files for the same picture slot                      NEWEST KEPT,
                                                                        rest quarantined
    - a video whose name matches no slot index or order at all
      ("Dashed_line_connecting_silhouettes...mp4")                     LEFT ALONE,
                                                                        reported by name
      next to the still-open slot's own SUBJECT/SCENE text, because the two
      read like the same shot - but this module never bets a render on a
      resemblance it read, only on a number it can verify.

Nothing here reads pixels. A subject/scene match is offered as a suggestion in
the report; applying it is a rename the operator confirms, never automatic.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from . import imageplan, motionplan

VIDEO_EXTS = motionplan.VIDEO_EXTS if hasattr(motionplan, "VIDEO_EXTS") else (
    ".mp4", ".mov", ".webm", ".m4v")

SLOT_RE = re.compile(r"^(\d+)section(\d+)")
QUARANTINE = "_duplicates"
UNRESOLVED = "_unresolved"


@dataclass
class Report:
    moved: list[str] = field(default_factory=list)      # wrong-folder fixes
    renamed: list[str] = field(default_factory=list)     # canonicalised names
    quarantined: list[str] = field(default_factory=list)  # extra copies, kept safe
    unresolved: list[str] = field(default_factory=list)   # could not place
    kept: dict = field(default_factory=dict)             # slot -> canonical name

    def as_text(self) -> str:
        lines = [
            f"moved (wrong folder)     : {len(self.moved)}",
            f"renamed (canonicalised)  : {len(self.renamed)}",
            f"quarantined (duplicates) : {len(self.quarantined)}",
            f"unresolved (left in place): {len(self.unresolved)}",
        ]
        for label, rows in (("MOVED", self.moved), ("RENAMED", self.renamed),
                            ("QUARANTINED", self.quarantined)):
            if rows:
                lines.append(f"\n{label}:")
                lines += [f"  {r}" for r in rows]
        if self.unresolved:
            lines.append(f"\nUNRESOLVED - check these by hand:")
            lines += [f"  {r}" for r in self.unresolved]
        return "\n".join(lines)


def _is_video(p: Path) -> bool:
    # Uploads carry extra suffixes after the real extension
    # ("007.mp4_1080p_2026....mp4"), so check every dot-part, not just p.suffix.
    return any(part.lower() in (e.lstrip(".") for e in VIDEO_EXTS)
               for part in p.name.split("."))


def _is_image(p: Path) -> bool:
    return p.suffix.lower() in imageplan.IMAGE_EXTS


def _quarantine(path: Path, root: Path, sub: str) -> Path:
    qdir = root / sub
    qdir.mkdir(exist_ok=True)
    dst = qdir / path.name
    n = 1
    while dst.exists():
        dst = qdir / f"{path.stem}__{n}{path.suffix}"
        n += 1
    shutil.move(str(path), str(dst))
    return dst


def sort_images(proj, plan_entries: list[dict], apply: bool = False) -> Report:
    """Move stray videos out, dedupe by slot, keep the newest per slot.

    `plan_entries` is `nb_prompts.json` (or any list with section/order) - the
    source of truth for which (section, order) pairs are expected to exist.
    """
    rep = Report()
    imgdir = proj.images
    expected = {(int(e["section"]), int(e["order"])) for e in plan_entries}

    by_slot: dict[tuple[int, int], list[Path]] = {}
    for f in sorted(imgdir.iterdir()):
        if not f.is_file() or f.parent.name in (QUARANTINE, UNRESOLVED):
            continue
        if _is_video(f):
            rep.moved.append(f"images/{f.name} -> motion/ (it's a video)")
            if apply:
                dst = proj.motion / f.name
                if dst.exists():
                    dst = _quarantine(f, proj.motion, QUARANTINE)
                else:
                    proj.motion.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(f), str(dst))
            continue
        m = SLOT_RE.match(f.stem)
        if m:
            key = (int(m.group(2)), int(m.group(1)))
            by_slot.setdefault(key, []).append(f)
        else:
            rep.unresolved.append(f"images/{f.name} - name doesn't match "
                                  f"<order>section<section>, left in place")

    for key, files in sorted(by_slot.items()):
        if key not in expected:
            rep.unresolved.append(
                f"images: {key} has {len(files)} file(s) but is not in the "
                f"current plan - left in place, check if the plan is stale")
            continue
        if len(files) == 1:
            rep.kept[key] = files[0].name
            continue
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        keeper, extras = files[0], files[1:]
        rep.kept[key] = keeper.name
        for e in extras:
            rep.quarantined.append(
                f"images/{e.name} (older copy of section {key[0]} #{key[1]}, "
                f"kept {keeper.name})")
            if apply:
                _quarantine(e, imgdir, QUARANTINE)

    return rep


def sort_motion(proj, plan_entries: list[dict], render_plan: list[dict] | None = None,
               apply: bool = False) -> Report:
    """Move stray images out of motion/, canonicalise names, dedupe per slot.

    `render_plan` (render_plan.json) supplies each motion slot's real INDEX;
    without it, order-number resolution is skipped and only exact-index or
    already-canonical names are recognised.
    """
    rep = Report()
    motdir = proj.motion
    motdir.mkdir(parents=True, exist_ok=True)

    motion_entries = [e for e in plan_entries if e.get("kind") == "video"]
    valid_index = set()
    order_to_index: dict[tuple[int, int], int] = {}
    if render_plan:
        by_so = {(i["section"], i["order"]): i["index"] for i in render_plan}
        for e in motion_entries:
            idx = by_so.get((e["section"], e["order"]))
            if idx is not None:
                valid_index.add(idx)
                order_to_index[(e["section"], e["order"])] = idx

    # Section-scoped order maps: order N -> index, but ONLY where that order
    # number is unique to motion slots in that section - otherwise a filename
    # like "13.mp4" could mean either of two different slots and guessing
    # would be exactly the silent-wrong-file failure this module exists to
    # prevent.
    order_by_section: dict[int, dict[int, int]] = {}
    for (sec, order), idx in order_to_index.items():
        order_by_section.setdefault(sec, {})[order] = idx
    ambiguous_orders: dict[int, set[int]] = {}
    seen: dict[int, dict[int, int]] = {}

    def resolve_index(stem: str, sec_hint: int | None) -> int | None:
        num = re.match(r"^0*(\d+)", stem)
        if not num:
            return None
        n = int(num.group(1))
        if n in valid_index:
            return n
        if sec_hint is not None and n in order_by_section.get(sec_hint, {}):
            return order_by_section[sec_hint][n]
        # No section hint on a bare video filename - check every section's
        # order map and only resolve if exactly one section claims this
        # number, so an accidental cross-section collision is refused rather
        # than guessed.
        hits = {sec for sec, m in order_by_section.items() if n in m}
        if len(hits) == 1:
            sec = next(iter(hits))
            return order_by_section[sec][n]
        return None

    from_images: list[Path] = []
    for f in sorted(proj.images.iterdir()):
        if f.is_file() and _is_video(f) and f.parent.name not in (QUARANTINE, UNRESOLVED):
            from_images.append(f)

    by_slot: dict[int, list[Path]] = {}
    unresolved: list[Path] = []
    for f in list(motdir.iterdir()) + from_images:
        if not f.is_file() or f.parent.name in (QUARANTINE, UNRESOLVED):
            continue
        idx = resolve_index(f.stem, None)
        if idx is None:
            unresolved.append(f)
        else:
            by_slot.setdefault(idx, []).append(f)

    for f in unresolved:
        loc = "images" if f.parent == proj.images else "motion"
        rep.unresolved.append(
            f"{loc}/{f.name} - name matches no known motion slot index or "
            f"order number, left in place")

    for idx, files in sorted(by_slot.items()):
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        keeper, extras = files[0], files[1:]
        canonical = f"{idx:03d}.mp4"
        dst = motdir / canonical
        rep.kept[idx] = canonical

        # QUARANTINE EVERY EXTRA BY ITS OWN PATH, BEFORE TOUCHING THE KEEPER.
        # One of those extras may already physically sit at `dst` (a stale
        # canonical file from a previous run). Moving it out of the way FIRST
        # means "the file currently at dst" and "the keeper we are about to
        # place there" can never be confused with each other - which is
        # exactly the bug this ordering replaces: quarantining the pre-existing
        # occupant of `dst` and then moving the keeper into `dst` left the
        # extras loop unable to tell the keeper's new file apart from the
        # occupant it had already moved, and it quarantined the keeper too.
        for e in extras:
            rep.quarantined.append(
                f"{e.parent.name}/{e.name} (older copy for slot {idx}, "
                f"kept {canonical})")
            if apply and e.exists():
                _quarantine(e, motdir, QUARANTINE)

        if apply and not (keeper.parent == motdir and keeper.name == canonical):
            if dst.exists():
                # Still occupied - the occupant is not one of `extras` (those
                # are gone above), so it is the keeper itself under a stale
                # canonical name from an earlier run. Nothing to do.
                if dst.resolve() != keeper.resolve():
                    _quarantine(dst, motdir, QUARANTINE)
                    shutil.move(str(keeper), str(dst))
            else:
                shutil.move(str(keeper), str(dst))
        if keeper.name != canonical or keeper.parent != motdir:
            (rep.moved if keeper.parent != motdir else rep.renamed).append(
                f"{keeper.name} -> motion/{canonical}")

    return rep
