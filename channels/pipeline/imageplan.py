"""Hand-supplied images -> a timed render plan.

THE PROBLEM THIS SOLVES. `scenes_timeline.json` says when every scene starts and
ends, anchored to real word timestamps from the recording. It does NOT say which
picture belongs to which scene, because on the generated path there was nothing
to decide - one frame was drawn per scene, so the mapping was the identity.

Supplying pictures by hand breaks that. A folder of images carries no timing and
no scene membership, so the only thing that can carry it is the filename.

THE CONVENTION
--------------
    <order>section<section>.<ext>          e.g.  1section1.png
                                                 2section1.png
                                                 1section2.png

`section` is which read-sheet section the picture belongs to. `order` is where it
sits inside that section.

WHY SORTING IS BY (section, order) AND NOT BY order ALONE. Two readings of the
convention are in circulation - `order` as a position within its section, and
`order` as a position across the whole video. Sorting by (section, order) gives
the identical answer under both, as long as sections play in ascending order,
which they do (they are read and concatenated 1..N). So the ambiguity never has
to be resolved: only the PAIR has to be unique, and that is what is enforced.

WHY CUTS SNAP TO SCENE BOUNDARIES. When a section has more scenes than pictures,
the obvious move is to divide the section's span evenly. That puts picture
changes mid-word. The scene boundaries are already sentence starts anchored to
real timestamps, so the cut points are chosen from THOSE instead - as close to
even as the available boundaries allow. The picture then changes when a sentence
changes, which is the only place a cut is invisible.
"""

from __future__ import annotations

import re
from pathlib import Path

# Common still formats. Kept explicit rather than "anything Pillow opens" so a
# stray .txt or .DS_Store in the folder is reported, not silently skipped.
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

# Shortest a picture may stay on screen. Below roughly this, a cut reads as a
# flash rather than as a change - and verify() rejects anything under 0.5s, so
# the planner must not emit what its own check would refuse.
MIN_HOLD = 0.5

# Canonical form plus the spellings that keep appearing by hand: a separator
# before the section word, and "sec"/"s" abbreviated. Anything else is rejected
# loudly - a filename this parser guesses at is a picture in the wrong place.
_STRICT_RE = re.compile(r"^(\d+)\s*[-_ ]?\s*(?:section|sec|s)\s*[-_ ]?\s*(\d+)$", re.I)

# The same pattern as a SUFFIX, so generator-assigned prefixes survive. A tool
# that names its output `chin_scene_047_3section2.png` is still telling us this
# is picture 3 of section 2, and rejecting it would mean bulk-renaming 105 files
# for no reason.
#
# The bare "s" alias is deliberately NOT accepted here. With a prefix in play it
# is ambiguous - `chin_scene_12` would parse as order 12 of section... nothing,
# or worse, match some earlier digit run. Requiring "sec"/"section" once there is
# a prefix keeps the match unambiguous, and the strict form above still allows
# `1s1` for hand-typed names.
_SUFFIX_RE = re.compile(r"^.*?[-_ ](\d+)\s*[-_ ]?\s*(?:section|sec)\s*[-_ ]?\s*(\d+)$",
                        re.I)

# The pattern LEADING, with anything after it - which is what real generators
# actually emit: `10section2_202608271333.jpeg` is order 10 of section 2 with a
# batch timestamp appended.
_PREFIX_RE = re.compile(
    r"^(\d+)\s*[-_ ]?\s*(?:section|sec)\s*[-_ ]?\s*(\d+)(?:[-_ ].*)?$", re.I)

# A trailing _2 / _3 on an otherwise identical name is a REGENERATION of the same
# slot, not an extra picture. Treating those as additional scenes would put 53
# images into a 36-sentence section and shift everything after them.
_VARIANT_RE = re.compile(r"^(?P<base>.*?)[-_ ](?P<n>[2-9])$")


class PlanError(ValueError):
    """A naming or coverage problem that must be fixed before rendering."""


def parse_name(stem: str) -> tuple[int, int]:
    """'2section1' -> (order=2, section=1). A prefix is allowed as long as the
    name ENDS in the pattern: 'chin_047_2section1' parses the same way.
    Raises PlanError on anything else."""
    stem = stem.strip()
    m = _STRICT_RE.match(stem) or _PREFIX_RE.match(stem) or _SUFFIX_RE.match(stem)
    if not m:
        raise PlanError(
            f"cannot read '{stem}' as <order>section<section>. "
            f"Expected the name to be, or to end in, 1section1 / 2section1 / "
            f"1section2 (a prefix before it is fine)."
        )
    order, section = int(m.group(1)), int(m.group(2))
    if section < 1:
        raise PlanError(f"'{stem}': section numbers start at 1")
    return order, section


def scan_images(folder: str | Path) -> list[dict]:
    """Read a folder of images into [{'path','order','section'}], sorted.

    Everything that is not a recognised image extension is ignored; every file
    that IS one must parse, because a picture that fails to parse would silently
    not appear in the video.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise PlanError(f"image folder not found: {folder}")

    items, bad = [], []
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        stem = p.stem
        variant = 1
        vm = _VARIANT_RE.match(stem)
        if vm:
            # Only a variant if the base still parses on its own - otherwise the
            # trailing digit is part of the name, not a regeneration marker.
            try:
                parse_name(vm.group("base"))
                stem, variant = vm.group("base"), int(vm.group("n"))
            except PlanError:
                pass
        try:
            order, section = parse_name(stem)
        except PlanError as e:
            bad.append(str(e))
            continue
        items.append({"path": str(p), "order": order, "section": section,
                      "variant": variant})

    if bad:
        raise PlanError("unparseable image filenames:\n  - " + "\n  - ".join(bad))
    if not items:
        raise PlanError(f"no images found in {folder}")

    # One picture per slot. Where a slot has regenerations, the lowest variant
    # wins and the rest are reported rather than silently dropped - a discarded
    # picture the user meant to use is exactly the kind of thing that must be
    # said out loud.
    chosen: dict[tuple[int, int], dict] = {}
    alternates: list[str] = []
    for it in sorted(items, key=lambda d: (d["section"], d["order"], d["variant"])):
        key = (it["section"], it["order"])
        if key not in chosen:
            chosen[key] = it
        elif it["variant"] != chosen[key]["variant"]:
            alternates.append(Path(it["path"]).name)
        else:
            raise PlanError(
                f"two images claim section {key[0]}, order {key[1]}: "
                f"{Path(chosen[key]['path']).name} and {Path(it['path']).name}"
            )

    out = sorted(chosen.values(), key=lambda d: (d["section"], d["order"]))
    if alternates:
        out[0].setdefault("_notes", []).append(
            f"{len(alternates)} alternate take(s) not used: "
            + ", ".join(alternates[:6]) + ("..." if len(alternates) > 6 else "")
        )
    return out


def _snap_cuts(span_start: float, span_end: float, boundaries: list[float],
               k: int) -> list[float]:
    """Pick k-1 cut points for k pictures, preferring real scene boundaries.

    `boundaries` are the interior scene starts inside this section. Each ideal
    even-division point claims the nearest unused boundary; the result is forced
    monotonic. With no boundaries to choose from (one scene, several pictures)
    it falls back to even division, which is the caller's warning to emit.
    """
    if k <= 1:
        return []
    ideal = [span_start + (span_end - span_start) * i / k for i in range(1, k)]
    if not boundaries:
        return ideal

    # SNAP ONLY WHEN THERE ARE ENOUGH BOUNDARIES TO GO ROUND. With k-1 cuts and
    # fewer boundaries than that, most cuts snap to the few available and the
    # leftovers pile up against them - measured at a 0.1s hold (three frames at
    # 30fps, a visible flash) on a 104-image plan over 68 scenes. Even division
    # gives a uniform hold instead, which is the better failure: every cut lands
    # mid-sentence either way, so there is nothing left to preserve by snapping.
    if len(boundaries) < k - 1:
        return ideal

    used, cuts = set(), []
    for want in ideal:
        free = [b for b in boundaries if b not in used]
        if not free:
            cuts.append(want)
            continue
        pick = min(free, key=lambda b: abs(b - want))
        used.add(pick)
        cuts.append(pick)

    cuts.sort()
    # Monotonic and strictly inside the span, so no zero-length picture can be
    # emitted - the ffmpeg concat demuxer silently DROPS a zero-duration entry.
    out, prev = [], span_start
    for c in cuts:
        c = min(max(c, prev + 0.05), span_end - 0.05)
        out.append(c)
        prev = c
    return out


def _word_axis(scenes: list[dict]) -> list[tuple[float, float, float, float]]:
    """[(word_start, word_end, t_start, t_end)] per scene, words cumulative."""
    axis, w = [], 0.0
    for s in scenes:
        n = max(1.0, float(len(str(s["text"]).split())))
        axis.append((w, w + n, float(s["start"]), float(s["end"])))
        w += n
    return axis


def _time_at_word(w: float, axis: list[tuple[float, float, float, float]]) -> float:
    """Convert a cumulative word position to a timestamp.

    Finds the scene that contains the position and interpolates linearly INSIDE
    that scene, between its two real Whisper timestamps. That is what makes the
    result track the speech: the interpolation error is bounded by one scene's
    length (a few seconds), not by the section's (three and a half minutes).
    """
    if not axis:
        return 0.0
    if w <= axis[0][0]:
        return axis[0][2]
    for w0, w1, t0, t1 in axis:
        if w0 <= w <= w1:
            frac = (w - w0) / max(w1 - w0, 1e-9)
            return t0 + (t1 - t0) * frac
    return axis[-1][3]


def _exact_word_cuts(words: list[dict], k: int,
                     span_start: float, span_end: float) -> list[tuple[float, float]]:
    """Split a section into k spans using REAL word timestamps. No interpolation.

    This is the whole point of running Whisper with word_timestamps=True. The
    section's spoken words are known individually, so the boundary between
    picture i and picture i+1 is simply "where word round(i*N/k) starts" - an
    exact timestamp Whisper measured, not a position estimated inside a scene.

    The cut is placed at the START of the boundary word, so that word is spoken
    under the new picture rather than being clipped by the outgoing one.
    """
    n = len(words)
    starts = [span_start]
    for i in range(1, k):
        idx = min(n - 1, max(0, round(n * i / k)))
        starts.append(float(words[idx]["s"]))
    starts = list(np_monotonic(starts, span_end))
    return list(zip(starts, starts[1:] + [span_end]))


def _word_cuts(scenes: list[dict], k: int) -> list[tuple[float, float]]:
    """Split a section into k spans of EQUAL WORD COUNT, then map to time.

    WHY NOT EQUAL TIME. Dividing a 212-second section into 49 equal 4.3s slices
    assumes the words are spread evenly through it, and they are not - a section
    holds short punchy lines and long explanatory ones, and Whisper's timestamps
    prove exactly where each falls. Cutting on equal time puts a picture change
    in the middle of whatever sentence happens to be running.

    Equal WORDS is what the pictures were actually made against: picture 7 of 20
    illustrates roughly the 7th twentieth of the words, because that is how the
    prompts were written. Mapping word position through the real timestamps then
    lands the cut where those words are actually spoken.

    Cuts still snap to a scene boundary when one is within half a scene, so a
    change lands on a sentence start wherever that costs nothing.
    """
    axis = _word_axis(scenes)
    total_w = axis[-1][1] if axis else 0.0

    raw = [_time_at_word(total_w * i / k, axis) for i in range(1, k)]

    # SNAP ONLY WHEN THERE ARE BOUNDARIES TO SPARE, AND NEVER TWICE TO THE SAME
    # ONE. Snapping every nearby cut onto the nearest scene start collapses
    # consecutive pictures onto one another - measured at 47 images under 0.5s
    # (visible flashes) on a 122-image plan over 68 scenes, because with more
    # pictures than sentences several cuts compete for the same boundary.
    #
    # So: with more pictures than scenes, the word positions ARE the answer and
    # nothing is snapped. With room to spare, each boundary can be claimed once.
    if k <= len(scenes):
        used: set[float] = set()
        snapped = []
        for t, (w0, w1, t0, t1) in zip(raw, axis[1:] + axis[-1:]):
            free = [b[2] for b in axis if b[2] not in used]
            if free:
                near = min(free, key=lambda b: abs(b - t))
                if abs(near - t) <= max(0.35, (t1 - t0) * 0.5):
                    used.add(near)
                    t = near
            snapped.append(t)
        raw = snapped

    starts = list(np_monotonic([float(scenes[0]["start"])] + raw,
                               float(scenes[-1]["end"])))
    ends = starts[1:] + [float(scenes[-1]["end"])]
    return list(zip(starts, ends))


def np_monotonic(vals: list[float], hard_end: float) -> list[float]:
    """Force non-decreasing with a minimum gap, without ever passing hard_end.

    The concat demuxer silently DROPS a zero-duration entry, so a degenerate span
    must be impossible rather than unlikely.
    """
    out, prev = [], None
    n = len(vals)
    for i, v in enumerate(vals):
        room = hard_end - v
        min_gap = min(0.05, room / max(n - i, 1)) if room > 0 else 0.0
        v = v if prev is None else max(v, prev + min_gap)
        out.append(min(v, hard_end))
        prev = out[-1]
    return out


def plan(images: list[dict], timeline: list[dict],
         words: list[dict] | None = None) -> tuple[list[dict], list[str]]:
    """Map images onto the aligned timeline. Returns (plan, warnings).

    Each plan entry is {'index','path','section','start','end','duration'} and
    the entries tile the audio exactly, in order, with no gaps.
    """
    warnings: list[str] = []

    by_section: dict[int, list[dict]] = {}
    for s in timeline:
        by_section.setdefault(int(s["section"]), []).append(s)
    for v in by_section.values():
        v.sort(key=lambda s: s["start"])

    img_sections = sorted({i["section"] for i in images})
    tl_sections = sorted(by_section)

    # A section with no picture would render as whatever the previous picture was,
    # holding silently over someone else's narration. A picture for a section that
    # does not exist is a typo that would otherwise just vanish. Both are fatal.
    uncovered = [s for s in tl_sections if s not in img_sections]
    if uncovered:
        raise PlanError(
            f"no images for section(s) {uncovered}. Every section in the timeline "
            f"needs at least one picture, or its narration plays over the previous "
            f"section's picture."
        )
    orphans = [s for s in img_sections if s not in tl_sections]
    if orphans:
        raise PlanError(
            f"images reference section(s) {orphans}, which the timeline does not "
            f"contain (it has {tl_sections}). Check the section numbers in the "
            f"filenames."
        )

    out: list[dict] = []
    for sec in tl_sections:
        scenes = by_section[sec]
        imgs = [i for i in images if i["section"] == sec]
        span_start = float(scenes[0]["start"])
        span_end = float(scenes[-1]["end"])
        k = len(imgs)

        if k == len(scenes):
            spans = [(float(s["start"]), float(s["end"])) for s in scenes]
        else:
            # Prefer REAL word timestamps when Whisper's are on disk; fall back
            # to interpolating within a scene only when they are not.
            sec_words = [w for w in (words or []) if int(w["section"]) == sec]
            if len(sec_words) >= k:
                spans = _exact_word_cuts(sec_words, k, span_start, span_end)
                how = "exact word timestamps"
            else:
                spans = _word_cuts(scenes, k)
                how = "word position interpolated within each scene"
            if k > len(scenes):
                warnings.append(
                    f"section {sec}: {k} images over {len(scenes)} scene(s), so "
                    f"{k - len(scenes)} cut(s) fall inside a sentence rather than "
                    f"on its edge. Cut using {how}, so they still track "
                    f"the speech."
                )
            else:
                warnings.append(
                    f"section {sec}: {k} image(s) over {len(scenes)} scene(s) "
                    f"across {span_end - span_start:.1f}s, cut using {how}."
                )

        for img, (a, b) in zip(imgs, spans):
            out.append({
                "index": len(out),
                "path": img["path"],
                "section": sec,
                "order": img["order"],
                "start": round(a, 3),
                "end": round(b, 3),
                "duration": round(b - a, 3),
            })

    # Absorb rounding into the last entry so the durations tile the audio exactly,
    # the same guarantee the timeline itself gives.
    if out:
        total = float(timeline[-1]["end"])
        out[-1]["duration"] = round(out[-1]["duration"] + (total - out[-1]["end"]), 3)
        out[-1]["end"] = round(total, 3)

    bad = [e["index"] for e in out if e["duration"] <= 0]
    if bad:
        raise PlanError(
            f"plan entries {bad} have zero or negative duration - the concat "
            f"demuxer drops those silently. Usually too many images in one section."
        )
    return out, warnings


def plan_from_binding(script_plan: dict, timeline: list[dict],
                      images_dir: str | Path | None = None
                      ) -> tuple[list[dict], list[str]]:
    """Join a script_plan.json against an aligned timeline. No division happens.

    This is the preferred path. plan() above has to GUESS how to split a section
    between its pictures, because a folder of images carries no binding. Here the
    binding was made deliberately from the script, so each image simply inherits
    the span of the scenes it already owns - first scene's start to last scene's
    end. Timing is derived, never guessed.
    """
    import hashlib

    warnings: list[str] = []

    # The join is positional, so a script edited after planning would shift every
    # picture by a sentence and produce a video that is subtly, entirely wrong.
    # Fingerprint first, fail loudly.
    plan_scenes = script_plan.get("scenes", [])
    if len(plan_scenes) != len(timeline):
        raise PlanError(
            f"the plan has {len(plan_scenes)} scenes but the timeline has "
            f"{len(timeline)}. The script changed after planning - re-run "
            f"plan_script.py, then re-record or re-align."
        )
    h = hashlib.sha256()
    for s in timeline:
        h.update(str(s["text"]).strip().encode("utf-8"))
        h.update(b"\x00")
    got = h.hexdigest()[:16]
    want = script_plan.get("fingerprint")
    if want and got != want:
        raise PlanError(
            f"scene-text fingerprint mismatch (plan {want}, timeline {got}).\n"
            f"The script was edited after the plan was made, so the image "
            f"bindings no longer line up. Re-run plan_script.py."
        )

    root = Path(images_dir or script_plan.get("images_dir") or "images")

    out: list[dict] = []
    missing: list[str] = []
    for img in script_plan.get("images", []):
        idxs = img["scene_indices"]
        start = float(timeline[idxs[0]]["start"])
        end = float(timeline[idxs[-1]]["end"])

        p = Path(img.get("path") or (root / img["filename"]))
        if not p.exists():
            # Accept any extension the generator happened to write.
            alts = [q for q in root.glob(Path(img["filename"]).stem + ".*")
                    if q.suffix.lower() in IMAGE_EXTS]
            if alts:
                p = alts[0]
            else:
                missing.append(img["filename"])
                continue

        out.append({
            "index": len(out),
            "path": str(p),
            "section": img["section"],
            "order": img["order"],
            "scene_indices": idxs,
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(end - start, 3),
        })

    if missing:
        raise PlanError(
            f"{len(missing)} image(s) named in the plan are not in {root}:\n  "
            + "\n  ".join(missing[:15])
            + ("\n  ..." if len(missing) > 15 else "")
        )

    if out:
        total = float(timeline[-1]["end"])

        # TILE BY CONSTRUCTION, DO NOT TRUST EACH SCENE'S OWN END.
        #
        # align() clamps every scene to at least 0.05s long (`ends = maximum(ends,
        # starts + 0.05)`), so two scenes that anchor within 50ms of each other -
        # a rushed "Not one." "Ever." - come back with scene i ending AFTER scene
        # i+1 starts. Taking each image's end from its own last scene then
        # produced overlapping pictures, a duration sum 0.1s longer than the
        # audio, and sub-frame holds. Measured on the chin video: 2 overlaps,
        # 2 flashes, 0.100s of phantom runtime.
        #
        # A picture's end is simply where the next one begins. That makes gaps and
        # overlaps unrepresentable rather than merely unlikely.
        starts = [e["start"] for e in out]

        # Then enforce a minimum hold, clamping so the tail always has room for
        # the pictures still to come. A cut moved by a few tens of milliseconds is
        # invisible; a 2-frame flash is not.
        n = len(starts)
        fixed = [0.0] * n
        prev = 0.0
        for i, s in enumerate(starts):
            ceiling = total - (n - i) * MIN_HOLD
            fixed[i] = min(max(s, prev), max(ceiling, prev))
            prev = fixed[i] + MIN_HOLD

        for i, e in enumerate(out):
            a = fixed[i]
            b = fixed[i + 1] if i + 1 < n else total
            e["start"] = round(a, 3)
            e["end"] = round(b, 3)
            e["duration"] = round(b - a, 3)

        moved = [i for i, (a, b) in enumerate(zip(starts, fixed)) if abs(a - b) > 0.001]
        if moved:
            worst = max(abs(starts[i] - fixed[i]) for i in moved)
            warnings.append(
                f"{len(moved)} cut(s) nudged to keep every picture at least "
                f"{MIN_HOLD:.2f}s on screen (largest move {worst:.2f}s). The "
                f"underlying scenes were read faster than that."
            )

    bad = [e["index"] for e in out if e["duration"] <= 0]
    if bad:
        raise PlanError(f"plan entries {bad} have zero or negative duration")

    short = [e for e in out if e["duration"] < 1.0]
    if short:
        warnings.append(
            f"{len(short)} image(s) hold under 1s - "
            f"{', '.join(Path(e['path']).name for e in short[:5])}. "
            f"Usually a rushed read of a short sentence."
        )
    return out, warnings


def verify(plan_entries: list[dict], timeline: list[dict],
           audio_seconds: float | None = None) -> tuple[list[str], list[str]]:
    """Prove the plan tiles the audio. Returns (checks, failures).

    Everything asserted here is cheap and would otherwise only surface as a
    picture sitting over the wrong sentence in the finished file.
    """
    checks, fails = [], []

    total = float(timeline[-1]["end"])
    covered = sum(e["duration"] for e in plan_entries)
    checks.append(f"timeline ends at        {total:.3f}s")
    checks.append(f"image durations sum to  {covered:.3f}s")
    if abs(covered - total) > 0.05:
        fails.append(f"durations sum to {covered:.3f}s but the timeline is "
                     f"{total:.3f}s ({abs(covered - total):.3f}s off)")

    if audio_seconds is not None:
        checks.append(f"audio file is           {audio_seconds:.3f}s")
        if abs(audio_seconds - total) > 0.05:
            fails.append(f"the audio file is {audio_seconds:.3f}s but the timeline "
                         f"ends at {total:.3f}s - they must be the same file")

    # First picture must open the video; last must close it. A gap at either end
    # shows as a frozen frame or a black hole in the render.
    if plan_entries:
        if plan_entries[0]["start"] > 0.001:
            fails.append(f"first image starts at {plan_entries[0]['start']:.3f}s, "
                         f"not 0 - the video would open on nothing")
        if abs(plan_entries[-1]["end"] - total) > 0.05:
            fails.append(f"last image ends at {plan_entries[-1]['end']:.3f}s, "
                         f"timeline ends at {total:.3f}s")

    gaps = [(a["index"], b["start"] - a["end"])
            for a, b in zip(plan_entries, plan_entries[1:])
            if abs(b["start"] - a["end"]) > 0.001]
    checks.append(f"gaps / overlaps         {len(gaps)}")
    if gaps:
        fails.append(f"{len(gaps)} gap(s) or overlap(s), first at image "
                     f"{gaps[0][0]} ({gaps[0][1]:+.3f}s)")

    short = [e for e in plan_entries if e["duration"] < 0.5]
    checks.append(f"images under 0.5s       {len(short)}")
    if short:
        fails.append(f"{len(short)} image(s) hold under 0.5s - visible as a flash")

    # Every picture must sit inside its own section's span, or it is illustrating
    # another section's words.
    span = {}
    for s in timeline:
        sec = int(s["section"])
        lo, hi = span.get(sec, (float("inf"), 0.0))
        span[sec] = (min(lo, float(s["start"])), max(hi, float(s["end"])))
    strays = [e["index"] for e in plan_entries
              if e["section"] in span
              and (e["start"] < span[e["section"]][0] - 0.05
                   or e["end"] > span[e["section"]][1] + 0.05)]
    checks.append(f"images outside section  {len(strays)}")
    if strays:
        fails.append(f"image(s) {strays[:8]} fall outside their own section's span")

    return checks, fails


def words_under(entry: dict, timeline: list[dict], limit: int = 90) -> str:
    """The narration actually playing while this image is on screen.

    Scene text is attributed by MIDPOINT, so a sentence straddling a cut belongs
    to exactly one picture and the report cannot double-count it.
    """
    out = []
    for s in timeline:
        mid = (float(s["start"]) + float(s["end"])) / 2.0
        if entry["start"] <= mid < entry["end"]:
            out.append(str(s["text"]))
    text = " ".join(out) or "(cut falls inside one sentence)"
    return text if len(text) <= limit else text[: limit - 1] + "…"


def render(plan_entries: list[dict], warnings: list[str]) -> str:
    lines = [f"  {'#':>3}  {'section':>7}  {'start':>8}  {'dur':>7}  image", ""]
    for e in plan_entries:
        lines.append(f"  {e['index']:>3}  {e['section']:>7}  {e['start']:>8.2f}  "
                     f"{e['duration']:>7.2f}  {Path(e['path']).name}")
    total = plan_entries[-1]["end"] if plan_entries else 0.0
    lines += ["", f"  {len(plan_entries)} images over {total:.2f}s"]
    if warnings:
        lines += ["", "  notes:"] + [f"    - {w}" for w in warnings]
    return "\n".join(lines)
