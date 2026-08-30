"""Does plan_from_binding apply the timeline editor's overrides correctly -
and refuse to when it shouldn't?

Synthetic script_plan + timeline + a temp images folder. No audio, no ffmpeg.

    ..\\vo_coach\\.venv\\Scripts\\python.exe test_imageplan.py
"""

from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pipeline import imageplan  # noqa: E402

PASS = FAIL = 0


def check(label: str, got, want) -> None:
    global PASS, FAIL
    if got == want:
        PASS += 1
        print(f"  ok    {label}")
    else:
        FAIL += 1
        print(f"  FAIL  {label}\n          got  {got}\n          want {want}")


def check_true(label: str, cond: bool) -> None:
    check(label, bool(cond), True)


def make_project(tmp: Path, n: int = 4) -> tuple[dict, list[dict], Path]:
    """n scenes, n images, one picture per scene - the simplest binding."""
    images_dir = tmp / "images"
    images_dir.mkdir()
    timeline = []
    images = []
    for i in range(n):
        timeline.append({"text": f"Sentence {i}.", "start": i * 2.0, "end": i * 2.0 + 1.8})
        fname = f"{i + 1}section1.png"
        (images_dir / fname).write_bytes(b"\x89PNG\r\n")   # content is irrelevant, only exists() matters
        images.append({"section": 1, "order": i + 1, "scene_indices": [i], "filename": fname})

    h = hashlib.sha256()
    for t in timeline:
        h.update(str(t["text"]).strip().encode("utf-8"))
        h.update(b"\x00")
    fingerprint = h.hexdigest()[:16]

    script_plan = {
        "scenes": [{"text": t["text"]} for t in timeline],
        "images": images,
        "fingerprint": fingerprint,
        "images_dir": str(images_dir),
    }
    return script_plan, timeline, images_dir


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        # -- 1. BASELINE: no overrides, plan_from_binding works as before ----
        script_plan, timeline, images_dir = make_project(tmp)
        entries, warnings = imageplan.plan_from_binding(script_plan, timeline, images_dir=images_dir)
        check("no overrides: 4 slots produced", len(entries), 4)
        check("no overrides: no warnings", warnings, [])

        fp = script_plan["fingerprint"]

        # -- 2. CUTS: a roll edit moves a slot's start, tiling stays sane ----
        overrides = {"fingerprint": fp, "cuts": {"2": 3.5}, "assets": {}}
        entries2, warnings2 = imageplan.plan_from_binding(
            script_plan, timeline, images_dir=images_dir, overrides=overrides)
        check("cuts override: slot 2 starts at the override time", entries2[2]["start"], 3.5)
        check("cuts override: slot 1 still ends where slot 2 starts",
              entries2[1]["end"], entries2[2]["start"])
        check_true("cuts override: durations still all positive",
                   all(e["duration"] > 0 for e in entries2))
        check_true("cuts override: total runtime unchanged",
                   abs(entries2[-1]["end"] - entries[-1]["end"]) < 1e-9)

        # -- 3. CUTS clamp: an override that would crush a later slot is tamed
        # by the SAME min-hold pass as an unedited cut - never a raw override.
        overrides_extreme = {"fingerprint": fp, "cuts": {"1": 7.9}}
        entries3, _ = imageplan.plan_from_binding(
            script_plan, timeline, images_dir=images_dir, overrides=overrides_extreme)
        check_true("extreme cuts override: still monotonic, no negative hold",
                   all(entries3[i]["end"] <= entries3[i + 1]["start"] + 1e-9
                       for i in range(len(entries3) - 1)))
        check_true("extreme cuts override: no zero/negative duration slot",
                   all(e["duration"] > 0 for e in entries3))

        # -- 4. ASSETS: swap which file a slot renders ------------------------
        (images_dir / "swapped.png").write_bytes(b"\x89PNG\r\n")
        overrides_asset = {"fingerprint": fp, "assets": {"1": "swapped.png"}}
        entries4, warnings4 = imageplan.plan_from_binding(
            script_plan, timeline, images_dir=images_dir, overrides=overrides_asset)
        check("assets override: slot 1 now points at the swapped file",
              Path(entries4[1]["path"]).name, "swapped.png")
        check("assets override: slot 0 untouched",
              Path(entries4[0]["path"]).name, "1section1.png")
        check("assets override: no warnings on a clean swap", warnings4, [])

        # -- 5. ASSETS override on a missing file still raises, doesn't silently
        # fall back to the original picture - a wrong filename must fail loudly.
        overrides_bad = {"fingerprint": fp, "assets": {"1": "does_not_exist.png"}}
        try:
            imageplan.plan_from_binding(script_plan, timeline, images_dir=images_dir,
                                        overrides=overrides_bad)
            check("bad asset override raises PlanError", "no exception", "PlanError")
        except imageplan.PlanError:
            check("bad asset override raises PlanError", True, True)

        # -- 6. STALE FINGERPRINT: overrides are ignored, not half-applied ----
        overrides_stale = {"fingerprint": "OLD_FINGERPRINT", "cuts": {"2": 3.5},
                           "assets": {"1": "swapped.png"}}
        entries5, warnings5 = imageplan.plan_from_binding(
            script_plan, timeline, images_dir=images_dir, overrides=overrides_stale)
        check("stale fingerprint: cuts NOT applied", entries5[2]["start"], entries[2]["start"])
        check("stale fingerprint: assets NOT applied",
              Path(entries5[1]["path"]).name, "2section1.png")
        check_true("stale fingerprint: a warning is raised",
                   any("IGNORED" in w for w in warnings5))

        # -- 7. No overrides at all behaves identically to overrides={} -------
        entries6, _ = imageplan.plan_from_binding(script_plan, timeline, images_dir=images_dir,
                                                  overrides={})
        check("empty overrides dict: identical to no overrides", entries6, entries)

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
