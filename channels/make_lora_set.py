"""
Build the LOCKED training set for the character LoRA.

    python make_lora_set.py                 # 30 prompts -> lora_set.json
    python make_lora_set.py --n 40
    python make_lora_set.py --workdir PATH

=============================================================================
 WHY A PURPOSE-BUILT SET, AND NOT JUST FRAMES FROM THE VIDEO
=============================================================================
The obvious idea is to train on frames already drawn for a script. It is the
wrong idea twice over:

  1. A LoRA learns whatever is CONSTANT across its training images. Video frames
     carry props, charts, giant doors and empty typographic panels, so a LoRA
     trained on them learns "sometimes there is a bar chart" -- noise, not the
     character.

  2. Video frames are drawn from whatever the script happened to say, so the
     pose coverage is accidental. Here it is deliberate: every pose the channel
     actually needs, each one paired with a different face.

So this set holds EVERYTHING constant except pose and expression, which is
exactly the axis a character LoRA should generalise over. No props at all --
props are what shots.py is for, and a LoRA that learned them would fight it.

=============================================================================
 THE SET IS LOCKED
=============================================================================
Seeded and written to disk as a file you can read. That matters more here than
anywhere else in the pipeline: a LoRA is trained ONCE and then silently steers
every frame after it. If the training set is not reproducible, a model that
starts behaving oddly six weeks from now cannot be traced back to what taught
it. `lora_set.json` is that record.

=============================================================================
 WHAT HAPPENS NEXT (read this before you train)
=============================================================================
    1. this script          -> lora_set.json          (free, local, now)
    2. notebook Step 21d    -> lora_raw/*.png         (GPU, ~10 min)
    3. YOU, by eye          -> keep the good ones     (this is the whole trick)
    4. notebook Step 21e    -> character_lora.safetensors

Step 3 is not a formality. A LoRA trained on 30 frames where the character
drifts will learn the drift and reproduce it forever. Keep only the frames
where the figure looks like ONE character -- if fewer than 20 survive, draw the
set again with a different seed rather than training on the rejects.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import config, look, paths  # noqa: E402

# Deliberately not shots.SHOT_SEED or look.LOOK_SEED: retuning the video's
# rotation must never silently change what the LoRA was trained on.
LORA_SEED = 20260824

# Every pose the channel actually needs, written in the CHARACTER_BIBLE's own
# vocabulary -- plain line arms and legs, no hands, no feet in detail. A pose
# this figure cannot hold ("hands on hips", "fingers spread") would teach the
# LoRA to invent anatomy, which is the exact failure the bible exists to stop.
POSES = [
    "standing still facing forward, arms hanging at its sides",
    "standing facing forward, one arm raised straight up",
    "walking to the left, mid-stride",
    "walking to the right, mid-stride",
    "sitting on a plain flat-coloured chair, arms resting on its legs",
    "pointing off to the right with one straight arm",
    "pointing off to the left with one straight arm",
    "pointing down at the floor with one arm",
    "touching its own arm with its other arm",
    "both arms raised straight above its head",
    "arms folded across its chest",
    "leaning forward slightly, both arms hanging down",
    "turning to look back over its shoulder",
    "crouching low with both arms down",
    "one arm raised in a still wave",
    "standing with its back to the camera, arms at its sides",
    "jumping with both arms up and both legs bent",
    "lying flat on its back on the floor line",
    "kneeling on one knee, one arm on the raised knee",
    "standing side-on in profile, arms at its sides",
]

# The scale the figure is drawn at. Held to three values, not left free: a LoRA
# trained on one framing only reproduces that framing, and the video needs the
# character to work close AND small.
SCALES = [
    "medium shot, the figure filling most of the frame height",
    "wide shot, the figure small and centred with empty space around it",
    "close shot, the figure's head and upper body only",
]


def build(n: int = 30, seed: int = LORA_SEED) -> list[dict]:
    """Pose x expression x scale, spread evenly and then locked.

    Round-robin rather than random sampling: with 20 poses, 6 expressions and 3
    scales there are 360 combinations, and a random draw of 30 would leave some
    poses unused and double others. Even coverage is the point -- the LoRA
    should see every pose at least once."""
    rng = random.Random(seed)
    expressions = sorted(look.EXPRESSIONS)
    rows = []
    for i in range(n):
        pose = POSES[i % len(POSES)]
        expression = expressions[i % len(expressions)]
        scale = SCALES[i % len(SCALES)]
        rows.append({
            "index": i,
            "pose": pose,
            "expression": expression,
            "scale": scale,
            # SAME ORDER AS pipeline/scenes.visual_prompt, and the order matters
            # more here than anywhere else. CLIP has a hard 77-token cap, so the
            # bible must come BEFORE the room description or the training images
            # are generated from a prompt whose character CLIP never saw -- which
            # would teach the LoRA the drift instead of the character.
            "prompt": (
                f"{config.STYLE_PREFIX}"
                f"{config.CHARACTER_BIBLE}, "
                f"{look.EXPRESSIONS[expression]['clause']}, "
                f"{pose}, "
                f"one figure alone, {scale}, "
                f"pure white background wall, one horizontal floor line with a flat "
                f"tan floor band along the bottom, large areas of empty white space"
                f"{getattr(config, 'STYLE_TAIL', '')}"
            ),
            # A different seed per image, unlike the video. Here variety is the
            # goal: 30 frames from one seed would be 30 near-copies and teach
            # the LoRA nothing about pose.
            "seed": rng.randint(1, 2**31 - 1),
        })
    return rows


def main() -> int:
    paths.force_utf8_stdout()
    ap = argparse.ArgumentParser(description="Lock the character-LoRA training set.")
    ap.add_argument("--n", type=int, default=30,
                    help="how many training images (default 30; keep >= 20 after review)")
    ap.add_argument("--seed", type=int, default=LORA_SEED)
    ap.add_argument("--workdir", help="the shared Drive folder")
    args = ap.parse_args()

    if args.workdir:
        paths.set_workdir(args.workdir)

    rows = build(args.n, args.seed)
    out = paths.p("lora_set.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(f"Locked {len(rows)} LoRA training prompts -> {out}")
    print(f"  seed         {args.seed}")
    print(f"  poses        {len(POSES)} ({len(set(r['pose'] for r in rows))} used)")
    print(f"  expressions  {len(look.EXPRESSIONS)} "
          f"({len(set(r['expression'] for r in rows))} used)")
    print(f"  scales       {len(SCALES)} ({len(set(r['scale'] for r in rows))} used)")
    print("\n  first three:")
    for r in rows[:3]:
        print(f"    #{r['index']:>2} [{r['expression']}] {r['pose'][:58]}")
    print("\nNext: notebook Step 21d draws these into lora_raw/, then YOU keep the")
    print("frames where the character looks right. Train on those, never on all 30.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
