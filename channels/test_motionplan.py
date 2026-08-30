"""Does motion land on the right slots, and refuse the wrong ones?

Pure arithmetic on synthetic render plans - no ffmpeg, no clips, no project.
The ingest half (VideoAssetProvider) is exercised separately because it needs
real video files.

    ..\\vo_coach\\.venv\\Scripts\\python.exe test_motionplan.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pipeline import motionplan, pacing  # noqa: E402

PASS = FAIL = 0


def check(label: str, got, want) -> None:
    global PASS, FAIL
    if got == want:
        PASS += 1
        print(f"  ok    {label}")
    else:
        FAIL += 1
        print(f"  FAIL  {label}\n          got  {got}\n          want {want}")


def plan(holds: list[float]) -> list[dict]:
    return [{"index": i, "duration": h, "section": 1, "path": f"img_{i}.jpg"}
            for i, h in enumerate(holds)]


def main() -> int:
    # -- 1. ONLY HOLDS INSIDE THE WINDOW ARE CANDIDATES -------------------
    p = plan([0.4, 1.0, 2.5, 3.4, 3.5, 3.75, 4.0, 4.1, 9.0])
    got = [c.index for c in motionplan.candidates(p)]
    print("the [3.5, 4.0] window")
    check("takes 3.5, 3.75 and 4.0 only", got, [4, 5, 6])
    check("a 1.0s accent is never a candidate", 1 in got, False)
    check("a 4.1s hold is not a candidate", 7 in got, False)
    check("a 9.0s hold is not a candidate either", 8 in got, False)

    # -- 2. NEVER TWO IN A ROW --------------------------------------------
    # Ten adjacent qualifying slots must not all be chosen: adjacent motion
    # reads as one long shot and erases the cut between them.
    sel = motionplan.select(plan([3.8] * 10 + [3.0] * 200))
    idx = [c.index for c in sel.chosen]
    gaps = [b - a for a, b in zip(idx, idx[1:])]
    print("\nspacing")
    check("no two chosen slots are adjacent", all(g > motionplan.MIN_GAP for g in gaps), True)

    # -- 3. RANKS TOWARD THE CEILING --------------------------------------
    # Within the window, the slots nearest 4.0s are the ones the ceiling rule
    # is about, so they win over slots at the bottom of the window.
    p = plan([3.5, 3.0, 3.0, 4.0, 3.0, 3.0, 3.55])
    sel = motionplan.select(p, share=0.5)
    print("\nranking")
    check("the 4.0s slot is chosen first", 3 in [c.index for c in sel.chosen], True)

    # -- 4. DETERMINISTIC --------------------------------------------------
    a = motionplan.select(plan([3.6, 3.9, 3.0, 3.7] * 30))
    b = motionplan.select(plan([3.6, 3.9, 3.0, 3.7] * 30))
    print("\ndeterminism")
    check("same input, same selection",
          [c.index for c in a.chosen], [c.index for c in b.chosen])

    # -- 5. THE UNDER-PLANNED CASE, WHICH IS CHIN'S ------------------------
    # Long holds are where motion is most wanted and exactly where a 4s clip
    # cannot go. The planner must report the shortfall, not fill it by bending
    # a rule.
    # The two qualifying slots are kept well apart, so this measures the
    # window rule alone - adjacency is already covered above.
    holds = [9.0] * 60
    holds[10] = holds[40] = 3.8
    sel = motionplan.select(plan(holds))
    print("\nunder-planned video (long holds, few candidates)")
    check("finds only the two that fit", len(sel.chosen), 2)
    check("reports being short", sel.short > 0, True)
    check("says why", "over the 4.0s ceiling" in sel.note, True)
    check("points at re-planning", "Re-plan at" in sel.note, True)

    # -- 6. NOTHING FITS AT ALL -------------------------------------------
    sel = motionplan.select(plan([9.0] * 40))
    print("\nno candidate anywhere")
    check("chooses nothing", len(sel.chosen), 0)
    check("does not claim to be fine", sel.short > 0, True)
    check("explains the ban", "banned" in sel.note, True)

    # -- 7. A HEALTHY VIDEO FILLS ITS DOSE --------------------------------
    # ~160 slots at 3s, every fourth one sized for a clip. The dose should be
    # reachable and the result should sit inside the 10-15% band.
    holds = [3.8 if i % 4 == 0 else 2.9 for i in range(160)]
    sel = motionplan.select(plan(holds))
    share = len(sel.chosen) * pacing.MOTION_CLIP_S / sum(holds)
    print("\nwell-planned video")
    check("fills the dose", sel.short, 0)
    check("lands inside 10-15% of runtime",
          pacing.MOTION_SHARE_LOW <= share <= pacing.MOTION_SHARE_HIGH + 0.005, True)
    check("never on an accent",
          all(h >= pacing.ACCENT_HIGH for h in (c.hold for c in sel.chosen)), True)
    print(f"    {len(sel.chosen)} clips over {sum(holds):.0f}s = {share * 100:.1f}%")

    # -- 8. MOTION ADDS NO RUNTIME ----------------------------------------
    # A clip REPLACES a still. Selecting slots must not change a timestamp.
    before = plan(holds)
    total_before = sum(i["duration"] for i in before)
    motionplan.select(before)
    print("\nmotion replaces, never appends")
    check("the plan is untouched",
          sum(i["duration"] for i in before), total_before)

    # -- 9. attach_kinds: the timeline editor's motion toggle --------------
    import tempfile
    from pathlib import Path

    class _FakeProj:
        def __init__(self, root: Path):
            self.motion = root / "motion"
            self.build = root / "build"
            self.motion.mkdir()
            self.build.mkdir()

    with tempfile.TemporaryDirectory() as td:
        proj = _FakeProj(Path(td))
        (proj.motion / "000.mp4").write_bytes(b"\x00")     # slot 0's BY-CONVENTION clip
        (proj.motion / "named.mp4").write_bytes(b"\x00")

        entries = [
            {"index": 0, "section": 1, "order": 1, "duration": 3.8, "path": "a.jpg"},
            {"index": 1, "section": 1, "order": 2, "duration": 2.0, "path": "b.jpg"},
            {"index": 7, "section": 1, "order": 3, "duration": 3.9, "path": "h.jpg"},
        ]
        script_plan_images = [
            {"section": 1, "order": 1, "kind": "video"},   # planner marked slot 0 motion
            {"section": 1, "order": 2, "kind": "image"},
            {"section": 1, "order": 3, "kind": "image"},
        ]
        sp = {"images": script_plan_images}

        print("\nattach_kinds: the timeline editor's motion toggle")

        # plain planner marks, no overrides: slot 0 finds its clip by convention
        e = [dict(x) for x in entries]
        r = motionplan.attach_kinds(e, sp, proj)
        check("no override: slot 0 stays video (planner's own mark)", e[0]["kind"], "video")
        check("no override: slot 0 finds 000.mp4 by convention", e[0].get("clip_path"),
              str(proj.motion / "000.mp4"))
        check("no override: slot 1 stays a still", e[1]["kind"], "image")

        # force OFF a slot the planner marked motion
        e = [dict(x) for x in entries]
        r = motionplan.attach_kinds(e, sp, proj, motion_overrides={"0": "off"})
        check("forced off: slot 0 becomes a still", e[0]["kind"], "image")
        check("forced off: slot 0 has no clip_path", "clip_path" in e[0], False)

        # force ON a still slot that FITS the motion window (slot 7, 3.9s), naming a clip
        e = [dict(x) for x in entries]
        r = motionplan.attach_kinds(e, sp, proj, motion_overrides={"7": "named.mp4"})
        check("forced on (fits): slot 7 becomes video", e[2]["kind"], "video")
        check("forced on (fits): uses the named clip", e[2].get("clip_path"),
              str(proj.motion / "named.mp4"))
        check("forced on (fits): no warning raised", r["warnings"], [])

        # force ON a still slot OUTSIDE the window (slot 1, 2.0s = an accent) - refused
        e = [dict(x) for x in entries]
        r = motionplan.attach_kinds(e, sp, proj, motion_overrides={"1": "named.mp4"})
        check("forced on (does not fit): stays a still", e[1]["kind"], "image")
        check("forced on (does not fit): a warning is raised", len(r["warnings"]) > 0, True)

        # named override clip that does not exist on disk falls back to lookup
        e = [dict(x) for x in entries]
        r = motionplan.attach_kinds(e, sp, proj, motion_overrides={"0": "missing.mp4"})
        check("missing named clip: falls back to convention lookup",
              e[0].get("clip_path"), str(proj.motion / "000.mp4"))
        check("missing named clip: a warning names the file",
              any("missing.mp4" in w for w in r["warnings"]), True)

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
