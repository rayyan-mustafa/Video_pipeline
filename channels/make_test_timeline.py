"""
Fake scenes_timeline.json, for testing the VISUALS half in isolation.

    python make_test_timeline.py

Writes scenes_timeline.json from script_tagged.json alone -- no recording, no
Whisper, no notebook Step 17 at all. Every scene gets a duration from its word
count at SPOKEN_WPM instead of a real timestamp.

=============================================================================
 WHY THIS IS SAFE TO USE FOR TESTING THE VISUALS, AND NOT FOR A REAL VIDEO
=============================================================================
Every consumer downstream of scenes_timeline.json only cares about the shape:
`{scene_index, start, end, duration, text, section}` (`pipeline/scenes.py:
load_timeline()` checks exactly those keys, nothing else). Nothing about
run_scenes.py, the character step, or FLUX drawing reads real timing -- they
read TEXT and turn it into an image. The only thing genuinely missing here is
the accuracy of `duration`, which matters for muxing the final video against
your real voice, not for whether the scene planner and FLUX agree on what a
line should look like.

So this is the right tool for "does the story board make sense and do the
frames look like one character" -- exactly the visuals question -- and the
wrong tool for a real upload. A real video's timeline MUST come from Step 17
aligning against your actual voice, because that is what keeps the pictures
synced to what you actually said, at the pace you actually said it.

Chunking is the SAME build_chunks() the notebook's aligner and
pipeline/readsheet.py use, so the scenes you get here are the same scenes a
real recording would produce -- just with estimated timing instead of measured.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import config, paths, readsheet  # noqa: E402


def build_fake_timeline(tagged: list[dict], wpm: float) -> list[dict]:
    scenes = []
    cursor = 0.0
    for i, c in enumerate(readsheet.build_chunks(tagged)):
        words = len(c["text"].split())
        dur = round(max(words / wpm * 60.0, 0.6), 3)   # floor: no frame under 0.6s
        scenes.append({
            "scene_index": i,
            "start": round(cursor, 3),
            "end": round(cursor + dur, 3),
            "duration": dur,
            "text": c["text"],
            "section": c["section"],
        })
        cursor += dur
    return scenes


def main() -> int:
    paths.force_utf8_stdout()
    ap = argparse.ArgumentParser(
        description="Fake scenes_timeline.json from script_tagged.json, for testing "
                    "the visuals without recording or aligning real audio.")
    ap.add_argument("--workdir", help="the shared Drive folder (default: VIDEO_PIPELINE_WORKDIR)")
    ap.add_argument("--wpm", type=float, default=config.SPOKEN_WPM,
                    help=f"assumed speaking rate (default: {config.SPOKEN_WPM}, "
                         "the same estimate run_script.py prints)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing scenes_timeline.json")
    args = ap.parse_args()

    if args.workdir:
        paths.set_workdir(args.workdir)

    src = paths.p("script_tagged.json")
    if not src.exists():
        print(f"No script_tagged.json in {paths.workdir()}.")
        print("  Run:  python run_script.py --topic <topic>")
        return 1

    out = paths.p("scenes_timeline.json")
    if out.exists() and not args.force:
        print(f"{out} already exists.")
        print("  If it is real (from notebook Step 17), do not overwrite it - "
              "re-run with --force only if you mean to replace it with fake timing.")
        return 1

    tagged = json.loads(src.read_text(encoding="utf-8"))
    scenes = build_fake_timeline(tagged, args.wpm)

    out.write_text(json.dumps(scenes, indent=2), encoding="utf-8")
    total = scenes[-1]["end"] if scenes else 0.0

    print(f"FAKE scenes_timeline.json written: {len(scenes)} scenes, "
          f"~{total/60:.1f} min at {args.wpm:.0f} wpm.")
    print(f"  -> {out}")
    print()
    print("This is for testing the VISUALS ONLY. It has no relationship to your")
    print("actual voice and must never be muxed into a real video - only Step 17,")
    print("aligning against a real recording, produces a timeline safe to ship.")
    print()
    print("Next: python run_scenes.py                    (plans the pictures, ~few cents)")
    print("      then open the notebook and start at Step 20 (skip 16-19 entirely)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
