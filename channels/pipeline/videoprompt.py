"""One filled Google Flow prompt per motion slot.

The sibling of `make_nb_prompts.py`'s image half, and it follows the same rule:
**assignment is not the model's job.** CAMERA is chosen here in Python,
deterministically and seeded, before any call goes out - so two consecutive
clips never get the same move, for exactly the reason `motion.ROTATION` exists
for Ken Burns. The model is only asked for the things that require reading the
narration: what moves, how the character moves, and how the biological element
behaves across the four seconds.

WHY A MOTION SLOT STILL NEEDS ITS PICTURE. Flow is driven image-to-video and the
uploaded still is the exact first frame, so a motion slot is not a picture the
plan saves - it is a picture that then gets animated. The sheet therefore emits
BOTH prompts for a video slot, in that order, and says so.

WHY THE TIMING BEATS ARE ASKED FOR AS THREE SPANS. The brief's own template
splits the clip 0-1.0 / 1.0-2.5 / 2.5-4.0, which is a beginning, a middle and a
result. Asking the model for one paragraph of "movement" instead reliably
produced a single continuous drift with no shape - the three spans are what make
the clip read as a thought rather than a wobble.
"""

from __future__ import annotations

import random

# The brief's own camera vocabulary. STATIC is weighted heavily on purpose: this
# is a hand-drawn explainer, and a moving camera on every shot reads as a slide
# deck with Ken Burns on it rather than as animation.
CAMERAS = (
    "STATIC",
    "STATIC",
    "VERY SLOW PUSH-IN",
    "VERY SLOW PAN",
    "SLIGHT TRACKING",
)

CAMERA_SEED = 7919          # same constant shots.py uses, for one seeded world


def assign_cameras(slots: list[dict], seed: int = CAMERA_SEED) -> None:
    """Fill `camera` on every motion slot, in place. Never twice in a row."""
    rng = random.Random(seed)
    prev = None
    for s in slots:
        cands = [c for c in CAMERAS if c != prev] or list(CAMERAS)
        cam = rng.choice(cands)
        prev = cam
        s["camera"] = cam


SYSTEM = """You are the animation director for a hand-drawn editorial channel
about human biology. The drawing style is fixed and handled elsewhere - never
mention style, linework, colour, or the camera.

You are describing a FOUR SECOND animated shot that starts from an existing
still image. The still is the exact first frame and must not be redesigned.

For ONE span of voiceover, return ONLY this JSON, no fences, no commentary:

{"movement": "...", "character": "...", "biological": "...",
 "beat1": "...", "beat2": "...", "beat3": "..."}

"movement": the single overall movement in the shot, at most 15 words. One idea,
  not a list. If almost nothing should move, say so plainly.
"character": what the figure does across the four seconds - a small pose or
  expression change. At most 12 words. If there is no character, say "none".
"biological": how the biological element behaves - what grows, flows, contracts,
  travels or reveals. At most 15 words. This is the shot's actual content.
"beat1": what happens from 0.0 to 1.0 seconds. The starting state settling.
"beat2": what happens from 1.0 to 2.5 seconds. The main action.
"beat3": what happens from 2.5 to 4.0 seconds. The result or reaction.

HARD RULES for every field:
- Small, limited, hand-drawn motion. No realistic physics, no 3D, no morphing.
- Nothing enters or leaves the frame. No new characters or objects appear.
- The last frame must still look like the first frame, only advanced.
- Never describe text, labels, letters or captions.
Each beat is at most 14 words."""


BLANK = {
    "movement": "[DESCRIBE EXACT MOVEMENT]",
    "character": "[DESCRIBE CHARACTER MOVEMENT]",
    "biological": "[DESCRIBE BIOLOGICAL MOVEMENT]",
    "beat1": "[INITIAL ACTION]",
    "beat2": "[MAIN ACTION]",
    "beat3": "[REACTION / RESULT]",
}


def build_prompt(entry: dict, topic: str = "") -> str:
    """The user prompt for one motion slot. Mirrors make_nb_prompts.ask_model."""
    head = f"THIS ENTIRE VIDEO IS ABOUT: {topic}.\n" if topic else ""
    return (
        f"{head}"
        f"The still image this animates already shows: {entry.get('scene', '')}\n"
        f"The mechanism it explains: {entry.get('mechanism', '')}\n"
        f"The visual metaphor already in the frame: {entry.get('metaphor', '')}\n\n"
        f"Voiceover spoken over these four seconds:\n"
        f"{' '.join(entry['narration']) or '(bridging span, no narration)'}"
    )


# The template, verbatim from the brief, with the slots filled.
TEMPLATE = """Create a 4-second 16:9 animated shot using the uploaded image as the EXACT visual starting frame.

The uploaded image is the source of truth for:
- character design
- character proportions
- facial features
- linework
- colors
- background
- props
- biological illustration style
- composition

DO NOT redesign, redraw, reinterpret, replace, or restyle any character or object.

VISUAL STYLE:
Preserve the established ORIGINAL Human Biology Visual Bible exactly.

Keep:
- minimalist 2D hand-drawn editorial science-cartoon aesthetic
- charcoal imperfect ink outlines
- flat muted colors
- simplified shapes
- minimal shading
- clean negative space
- original character construction
- original background language

ANIMATION:
Use subtle, purposeful 2D movement only.

The animation should feel like a hand-drawn educational cartoon coming gently to life.

MOVEMENT:
{movement}

CHARACTER:
{character}

BIOLOGICAL ELEMENT:
{biological}

CAMERA:
{camera}

TIMING:
0.0-1.0 seconds:
{beat1}

1.0-2.5 seconds:
{beat2}

2.5-4.0 seconds:
{beat3}

MOTION QUALITY:
- smooth but intentionally limited animation
- small natural movements
- simple hand-drawn motion
- subtle secondary motion
- readable silhouette throughout

IMPORTANT:
The scene must remain visually stable.

Do NOT introduce:
- new characters
- new objects
- changing clothing
- changing facial structure
- realistic physics
- 3D movement
- camera shake
- cinematic lens effects
- photorealism
- realistic human motion
- exaggerated squash and stretch
- morphing
- object duplication
- background changes
- style changes
- flickering outlines
- unstable anatomy

The final frame should remain visually consistent with the starting frame.

16:9.
4 seconds."""
