"""
Step 22, locally: decide what every frame shows.

    python run_scenes.py --shots         the rotation only. No key, no cost.
    python run_scenes.py --preflight
    python run_scenes.py
    python run_scenes.py --workers 4 --dry-run

Reads  scenes_timeline.json   (written by the notebook's Step 17)
Writes scenes_planned.json    (read by the notebook's Steps 23-25)

Two decisions per frame, and they are made by different things. The SHOT -- what
kind of picture it is -- is assigned locally and deterministically by
pipeline/shots.py before any call goes out. The SUBJECT is written by the model,
for the shot it was given. Run --shots first: it is free, and it is how you tune
the rotation without paying to find out.

This is the last CPU stage that was still living in the notebook. It is one
cheap model call per scene -- on a 150-scene script, 150 network round trips
run eight at a time, with no torch anywhere. Running it in Colab bills GPU
session time for pure network wait, and a disconnect halfway loses all of it.

=============================================================================
 RUN THIS BETWEEN THE TWO NOTEBOOK HALVES
=============================================================================
    notebook Step 17-19   voice: align, master, check     (GPU-ish, Whisper)
    >>> run_scenes.py     plan the pictures               (local, this file)
    notebook Step 20-26   FLUX: character, frames, mux    (GPU)

Because it sits in the middle, it is the stage most exposed to Drive sync lag:
it reads a file the notebook just wrote. `--preflight` checks that the timeline
is fully synced before spending anything on it.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import config, engine, paths, scenes as scenes_mod  # noqa: E402


def load_env() -> None:
    """One key for everything -- Claude and any other model both route through
    OpenRouter. Same resolution order as run_script.py."""
    try:
        from dotenv import load_dotenv
        for candidate in (Path(__file__).resolve().parent.parent / ".env",
                          Path(__file__).resolve().parent / ".env"):
            if candidate.exists():
                load_dotenv(candidate, override=False)
    except ImportError:
        pass

    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit(
            "No OPENROUTER_API_KEY.\n"
            "  Get one at openrouter.ai/keys, then add it to f:\\wierd_bio\\.env:\n"
            "      OPENROUTER_API_KEY=sk-or-v1-...\n"
            "  or set it in this shell:\n"
            "      $env:OPENROUTER_API_KEY = 'sk-or-v1-...'"
        )


def main() -> int:
    paths.force_utf8_stdout()

    ap = argparse.ArgumentParser(description="Step 22 of the stickman pipeline, locally.")
    ap.add_argument("--workdir", help="the shared Drive folder (default: VIDEO_PIPELINE_WORKDIR)")
    ap.add_argument("--workers", type=int, default=scenes_mod.MAX_WORKERS,
                    help=f"scenes planned in parallel (default {scenes_mod.MAX_WORKERS})")
    ap.add_argument("--models", default="claude", choices=sorted(config.MODEL_PRESETS),
                    help="which models to use. 'cheap' runs everything on one small "
                         "model to prove the wiring for about a cent - never judge a "
                         "script written under it")
    ap.add_argument("--preflight", action="store_true",
                    help="check the key, the model slugs and the timeline, then exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="show what would be planned and what it will cost, without calling")
    ap.add_argument("--limit", type=int,
                    help="plan only the first N scenes - for trying the prompt cheaply")
    ap.add_argument("--shots", action="store_true",
                    help="assign the shot rotation and print it, then exit. No key, no "
                         "calls, no cost - this is how you tune shots.py")
    ap.add_argument("--strict", action="store_true",
                    help="refuse to finish if the rotation validator complains")
    args = ap.parse_args()

    if args.workdir:
        paths.set_workdir(args.workdir)

    print(f"Working folder: {paths.workdir()}")

    # The timeline crosses the machine boundary, so check it is all here before
    # anything else. A half-synced file would otherwise plan frames against
    # truncated text and the error would only surface at mux time.
    try:
        paths.require("voice")
    except FileNotFoundError:
        # The notebook does not write stage markers yet, so its absence is
        # expected rather than an error. The shape check below still runs.
        pass

    timeline = scenes_mod.load_timeline()
    if args.limit:
        timeline = timeline[:args.limit]
    print(f"Timeline: {len(timeline)} scenes, "
          f"{sum(s['duration'] for s in timeline):.1f}s of voiceover")

    # Before the key, before the preset, before anything billable: the rotation
    # is pure local scheduling, so tuning it must cost nothing.
    if args.shots:
        from pipeline import look as look_mod, shots as shots_mod

        shots_mod.assign(timeline)
        look_mod.assign(timeline)         # after the shot: it decides what is legal
        print(f"\nSHOT ROTATION + EXPRESSIONS ONLY - no model was called.\n")
        print(shots_mod.summary(timeline))
        print()
        print(look_mod.summary(timeline))
        print("\n  first twelve frames:")
        for s in timeline[:12]:
            print(f"    S{s['section']} #{s['scene_index']:>3}  e={s['energy']:.2f} "
                  f"{s['shot']:<18} {s['expression'] or '-'}")
        problems = [p for p in shots_mod.validate(timeline) + look_mod.validate(timeline)
                    if "carry text" not in p]      # captions are not planned yet
        if problems:
            print("\n  WARNINGS:")
            for p in problems:
                print(f"    - {p}")
            return 1 if args.strict else 0
        print("\n  Rotation is clean: no adjacent repeats, every shot inside its cap.")
        return 0

    if args.dry_run:
        config.apply_preset(args.models)
        # ~350 prompt + ~40 completion tokens per scene, measured on the
        # notebook's own prompt. Rough, but the right order of magnitude for
        # deciding whether to run it.
        pin, pout = engine.PRICE_PER_MTOK.get(config.MODEL_SCENES, (0.0, 0.0))
        est = len(timeline) * (350 / 1e6 * pin + 40 / 1e6 * pout)
        print(f"\nDRY RUN - nothing was called.")
        print(f"  model    {config.MODEL_SCENES}")
        print(f"  calls    {len(timeline)} (at {args.workers} concurrent)")
        print(f"  estimate ~${est:.4f}")
        print("\n  first three lines that would be planned:")
        for s in timeline[:3]:
            print(f"    S{s['section']} #{s['scene_index']}: {s['text'][:70]}")
        return 0

    load_env()
    _preset = config.apply_preset(args.models)
    if args.models != "claude":
        print(f"Models: {args.models} preset -> "
              + ", ".join(f"{r}={m}" for r, m in _preset.items()))
        print("  This preset is for checking the pipeline runs, not the writing.\n")
    engine.preflight()
    print()

    if args.preflight:
        print(f"Timeline is present and well-formed ({len(timeline)} scenes).")
        print("Ready to run.")
        return 0

    # Ship the prompt constants alongside the outputs, so the notebook's
    # Redraw button builds the same prompt this run baked into the frames.
    config.export_style()

    scenes_mod.run(timeline, max_workers=args.workers, strict=args.strict)
    engine.print_usage()

    print("\n" + "=" * 70)
    print("NEXT: back to the notebook for the pictures.")
    print("=" * 70)
    print("  Step 20  lock the look")
    print("  Step 21  pick your character (once per channel)")
    print("  Step 23  draw the frames  <- reads scenes_planned.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
