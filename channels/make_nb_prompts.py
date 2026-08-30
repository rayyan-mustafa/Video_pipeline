"""script_plan.json -> one filled Nano Banana prompt per image, as a text file.

    python make_nb_prompts.py --project wierdbiology/chin
    python make_nb_prompts.py --project wierdbiology/chin --no-llm

WHAT IS DECIDED HERE VS ASKED OF THE MODEL
------------------------------------------
This follows the rule pipeline/shots.py already sets for this project: assignment
is not the model's job. COMPOSITION, CHARACTER PLACEMENT and BACKGROUND are
chosen in Python, deterministically, seeded, before any call goes out. The model
is only asked for the things that require reading the narration - what the
subject is, what mechanism it explains, and how to show something invisible.

WHY BACKGROUND IS NAMED RATHER THAN LEFT TO THE MODEL. The brief's own wording
was "choose the most appropriate established background colour rather than
defaulting to the same colour used in previous scenes". A text-to-image model
cannot honour that: it has NO memory of your previous generations, so every
image is scene one as far as it is concerned. Non-repetition is only achievable
by naming a specific colour per image, which is what happens here - rotated from
config.PALETTE, never twice in a row. Use --background-mode ask to restore the
original wording instead.

COMPOSITION comes from the shot rotation, not from a fresh coin flip, so the
"no two consecutive frames are the same kind of picture" property that shots.py
was built to guarantee survives into this workflow.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pipeline import config, scenes as scenes_mod, shots, videoprompt
from pipeline.project import Project


def load_env() -> None:
    """Same resolution order as run_scenes.py / run_script.py: one OpenRouter key,
    read from the repo root .env or from this folder's."""
    try:
        from dotenv import load_dotenv
        for candidate in (Path(__file__).resolve().parent.parent / ".env",
                          Path(__file__).resolve().parent / ".env"):
            if candidate.exists():
                load_dotenv(candidate, override=False)
    except ImportError:
        pass

# --------------------------------------------------------------------------
#  Deterministic slots
# --------------------------------------------------------------------------
# Shot type -> the brief's own composition vocabulary. object-focus splits: in a
# mechanism section a cutaway explains more than a close-up of the outside does,
# and SECTION_DIRECTION[4] already asks for "a simple diagram or cutaway".
COMPOSITION = {
    "character-moment": "MEDIUM",
    "two-character": "WIDE",
    "symbolic-metaphor": "WIDE",
    "object-focus": "CLOSE-UP",
    "icon-grid": "COMPARISON",
    "data-chart": "COMPARISON",
    "typographic": "EXTREME CLOSE-UP",
}
CUTAWAY_SECTIONS = {4}          # "process and machinery" - see SECTION_DIRECTION

PLACEMENTS = ("LEFT", "CENTER", "RIGHT")

# Backgrounds are drawn from PALETTE, but not all of it: near-black is the
# outline colour and tan is the floor band, so neither can be a background
# without colliding with something the style depends on.
BACKGROUND_CHOICES = [
    ("white", (255, 255, 255)),
    ("warm beige", (226, 202, 160)),
    ("teal", (74, 148, 148)),
    ("navy", (44, 74, 124)),
    ("slate blue-grey", (112, 128, 148)),
    ("forest green", (58, 128, 62)),
    ("olive", (140, 152, 74)),
]

NO_CHARACTER = "None - this is an abstract frame with no character in it."


def deterministic_slots(entries: list[dict], seed: int = shots.SHOT_SEED) -> None:
    """Fill shot, composition, placement and background on every entry, in place."""
    import random

    pseudo = [{"text": " ".join(e["narration"]), "section": e["section"]} for e in entries]
    shots.assign(pseudo, seed=seed)

    rng = random.Random(seed)
    prev_place, prev_bg = None, None

    for e, p in zip(entries, pseudo):
        shot = p["shot"]
        spec = shots.SHOTS[shot]
        e["shot"] = shot
        e["has_character"] = bool(spec.get("character"))

        comp = COMPOSITION.get(shot, "MEDIUM")
        if shot == "object-focus" and e["section"] in CUTAWAY_SECTIONS:
            comp = "CUTAWAY"
        e["composition"] = comp

        if e["has_character"]:
            cands = [x for x in PLACEMENTS if x != prev_place] or list(PLACEMENTS)
            place = rng.choice(cands)
            prev_place = place
            e["placement"] = place
        else:
            e["placement"] = NO_CHARACTER

        cands = [c for c in BACKGROUND_CHOICES if c[0] != prev_bg] or BACKGROUND_CHOICES
        name, rgb = rng.choice(cands)
        prev_bg = name
        e["background"] = f"{name} rgb{rgb}"


# --------------------------------------------------------------------------
#  The model's half
# --------------------------------------------------------------------------
SYSTEM = """You are the visual director for a hand-drawn editorial channel about
human biology. The drawing style is fixed and handled elsewhere - never mention
style, linework, colour or camera.

For ONE span of voiceover, return ONLY this JSON, no fences, no commentary:

{"subject": "...", "scene": "...", "character_action": "...",
 "mechanism": "...", "metaphor": "...", "location": "..."}

"subject": the exact biological topic this frame is about. A noun phrase, at most
  8 words. Not a sentence.
"scene": what is physically happening in the picture, 8-20 words. Concrete and
  drawable. No abstract nouns.
"character_action": what the figure is doing - pose, movement, expression. At
  most 12 words. Say what the WHOLE FIGURE does.
"mechanism": the exact biological mechanism being explained, one plain sentence a
  15-year-old follows. If the line explains no mechanism, say what it establishes.
"metaphor": how to SHOW the invisible part of that mechanism as a picture -
  arrows, scale, cross-section, before/after, a physical analogy. At most 20
  words. This is the most important field: if the process is invisible, the
  frame fails without it.
"location": the environment. 2-6 words. Plain and simple.

Never name text, letters, labels, signs or lettering - captions are added
separately. Never describe hands or fingers in close detail."""


def ask_model(entry: dict, topic: str = "") -> dict:
    direction = scenes_mod.SECTION_DIRECTION.get(entry["section"], "")
    kind = ("This frame CONTAINS the character."
            if entry["has_character"] else
            "This frame contains NO character - it is an abstract or object frame. "
            "Return \"character_action\": \"none\".")
    # WHY THE TOPIC IS REPEATED ON EVERY CALL. Each span is planned in isolation,
    # so a sentence like "the most identifiable feature on your entire face" names
    # no organ and the model fills the gap with whatever fits - measured at 10 of
    # 66 frames drifting onto ears, tongues, ribcages and DNA on the chin script.
    # The topic line costs a dozen tokens and removes the whole failure class.
    head = f"THIS ENTIRE VIDEO IS ABOUT: {topic}.\n" if topic else ""
    prompt = (
        f"{head}"
        f"Every frame must stay on that subject. If the line below does not name "
        f"it explicitly, it is still about that subject - never substitute a "
        f"different organ, body part or process.\n\n"
        f"Shot type: {entry['shot']} ({entry['composition']} composition). {kind}\n"
        f"Section direction: {direction}\n\n"
        f"Voiceover for this frame:\n{' '.join(entry['narration']) or '(bridging span, no narration)'}"
    )
    raw = scenes_mod.call_model(
        config.MODEL_SCENES, prompt, system=SYSTEM,
        max_tokens=600, temperature=0.4, retries=2,
    )
    d = scenes_mod.parse_json_response(raw)
    if not isinstance(d, dict):
        raise ValueError(f"not a dict: {raw[:120]!r}")
    return d


BLANK = {
    "subject": "[EXACT BIOLOGICAL TOPIC]",
    "scene": "[WHAT IS HAPPENING]",
    "character_action": "[POSE / MOVEMENT / EXPRESSION]",
    "mechanism": "[EXACT MECHANISM BEING EXPLAINED]",
    "metaphor": "[HOW THE INVISIBLE BIOLOGICAL PROCESS SHOULD BE SHOWN]",
    "location": "[ENVIRONMENT]",
}


def fill_all(entries: list[dict], workers: int = 6, topic: str = "") -> int:
    """Ask the model for every entry. A failure leaves the blank slot, never kills
    the run - a 62-frame brief must not die on frame 57."""
    failures = 0

    def one(e):
        nonlocal failures
        try:
            d = ask_model(e, topic)
            for k, v in BLANK.items():
                got = str(d.get(k, "") or "").strip()
                e[k] = got if got else v
            if not e["has_character"] or e.get("character_action", "").lower() == "none":
                e["character_action"] = NO_CHARACTER
        except Exception as ex:
            failures += 1
            print(f"  [warn] {e['filename']}: {ex}")
            e.update(BLANK)
            if not e["has_character"]:
                e["character_action"] = NO_CHARACTER

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, entries))
    return failures


def ask_video(entry: dict, topic: str = "") -> dict:
    """The animation half, asked only for slots the plan marked as motion."""
    raw = scenes_mod.call_model(
        config.MODEL_SCENES, videoprompt.build_prompt(entry, topic),
        system=videoprompt.SYSTEM, max_tokens=600, temperature=0.4, retries=2,
    )
    d = scenes_mod.parse_json_response(raw)
    if not isinstance(d, dict):
        raise ValueError(f"not a dict: {raw[:120]!r}")
    return d


def fill_video(entries: list[dict], workers: int = 6, topic: str = "") -> int:
    """Fill the animation fields on every motion slot. Same failure policy as
    the image half: a dropped call leaves blank [SLOTS], it never kills the run.

    Runs AFTER the image pass on purpose - the animation prompt is built from
    the still's own scene, mechanism and metaphor, so it needs those filled
    first or it would be describing movement in a frame nobody has described.
    """
    motion = [e for e in entries if e.get("kind") == "video"]
    if not motion:
        return 0
    failures = 0

    def one(e):
        nonlocal failures
        try:
            d = ask_video(e, topic)
            for k, v in videoprompt.BLANK.items():
                got = str(d.get(k, "") or "").strip()
                e[k] = got if got else v
        except Exception as ex:
            failures += 1
            print(f"  [warn] video {e['filename']}: {ex}")
            e.update(videoprompt.BLANK)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(one, motion))
    return failures


# --------------------------------------------------------------------------
#  The template  (verbatim from the brief, with the slots filled)
# --------------------------------------------------------------------------
TEMPLATE = """Create a 16:9 scene for a YouTube video about human biology.

SUBJECT:
{subject}

SCENE:
{scene}

CHARACTERS:
Use the uploaded character references as the source of truth for character design.

CHARACTER ACTION:
{character_action}

BIOLOGICAL MECHANISM:
{mechanism}

VISUAL METAPHOR:
{metaphor}

LOCATION:
{location}

COMPOSITION:
{composition}

CHARACTER PLACEMENT:
{placement}

BACKGROUND:
{background}

STYLE:
Follow the uploaded ORIGINAL Human Biology Visual Bible exactly.

Preserve:

* character design
* line quality
* facial system
* color language
* biological diagram language
* level of detail
* simplified editorial aesthetic

Allow the scene itself to be visually different from previous scenes.

The image should communicate the biological idea immediately without requiring text.

IMPORTANT:
Do not introduce a different art style.
Do not copy any external reference.
Do not use photorealism.
Do not use 3D.
Do not use anime aesthetics.
Do not use glossy rendering.
Do not create realistic medical artwork.
Do not unnecessarily repeat compositions from previous scenes.

16:9 cinematic editorial illustration."""

BACKGROUND_ASK = ("Choose the most appropriate established background color from "
                  "the channel palette rather than defaulting to the same color "
                  "used in previous scenes.")


def mmss(t: float) -> str:
    return f"{int(t) // 60}:{int(t) % 60:02d}"


def render_txt(entries: list[dict], section: int | None = None) -> str:
    """The shot sheet: every slot in order, with the prompt(s) it needs.

    A MOTION SLOT CARRIES TWO PROMPTS, in order: the still first, then the
    animation that uses it as its exact first frame. Flow is driven
    image-to-video, so the picture is not optional for a video slot - it is
    step one of two. Emitting them adjacent, under one header, is what stops
    the pair being separated when this is worked through by hand.
    """
    rows = [e for e in entries if section is None or e["section"] == section]
    motion = [e for e in rows if e.get("kind") == "video"]
    timed = [e for e in rows if e.get("hold") is not None]

    head = [
        "SHOT SHEET",
        "=" * 78,
        f"{len(rows)} slots"
        + (f" in section {section}" if section is not None else "")
        + f"  -  {len(rows) - len(motion)} still, {len(motion)} motion",
    ]
    if timed:
        total = sum(e["hold"] for e in timed)
        head.append(f"{total:.1f}s of screen time, timings measured by Whisper")
    if motion:
        head.append(f"{len(motion)} x 4.0s Flow clip = {len(motion) * 4.0:.1f}s of motion")
    head += [
        "",
        "Attach your character reference sheet and the Visual Bible to EVERY",
        "image generation. Save each result under the filename in its header -",
        "the build reads section and order out of that name and nothing else.",
        "",
        "A MOTION slot needs BOTH prompts below it: generate the still first,",
        "then upload that still to Flow with the animation prompt. Save the clip",
        "as <slot index>.mp4 in the project's motion/ folder.",
        "",
    ]

    out = list(head)
    for e in rows:
        is_video = e.get("kind") == "video"
        when = ""
        if e.get("hold") is not None:
            when = f"   |   {mmss(e['start'])}-{mmss(e['end'])}  ({e['hold']:.2f}s)"
        idx = f"slot {e['index']}   |   " if e.get("index") is not None else ""
        out += [
            "",
            "=" * 78,
            f" {'[MOTION] ' if is_video else ''}{e['filename']}   |   {idx}"
            f"section {e['section']}   |   {e['shot']}{when}",
            "=" * 78,
            f" narration: {' '.join(e['narration'])}",
            "-" * 78,
            "",
        ]
        if is_video:
            out += ["--- 1 of 2: the still (this becomes Flow's first frame) ---", ""]
        out += [TEMPLATE.format(**e), ""]
        if is_video:
            out += [
                "",
                "--- 2 of 2: the animation (upload the still above) ---",
                "",
                videoprompt.TEMPLATE.format(**e),
                "",
            ]
    return "\n".join(out)


def attach_timings(entries: list[dict], proj) -> int:
    """Copy measured start/end/hold onto each slot from render_plan.json.

    Timing is DERIVED, never stored in the plan - see plan_script's header - so
    the sheet reads it back from the render plan when one exists and simply
    omits the column when it does not. Matching is by (section, order), which is
    the pair the filename encodes and the only stable identity a slot has.
    """
    plan = None
    if proj.render_plan.exists():
        try:
            plan = json.loads(proj.render_plan.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            plan = None

    # DERIVE THE TIMING WHEN THERE IS NO RENDER PLAN - which is the normal case
    # here, because the render plan needs the pictures to exist and this sheet is
    # what you generate them FROM. The aligned timeline and words.json are enough:
    # a slot's span is where its own words were spoken.
    if plan is None or len(plan) != len(entries):
        if not (proj.timeline.exists() and proj.plan.exists()):
            return 0
        try:
            from pipeline import imageplan
            script_plan = json.loads(proj.plan.read_text(encoding="utf-8"))
            timeline = json.loads(proj.timeline.read_text(encoding="utf-8"))
            wf = proj.build / "words.json"
            words = json.loads(wf.read_text(encoding="utf-8")) if wf.exists() else None
            plan, _ = imageplan.plan_from_binding(
                script_plan, timeline, require_images=False, words=words)
        except Exception as e:
            print(f"  [warn] could not derive timings: {e}")
            return 0
    # STALE TIMINGS ARE WORSE THAN NO TIMINGS. render_plan.json carries no
    # fingerprint of its own, so a plan re-run at a different picture count
    # leaves a render plan whose (section, order) pairs still partially match -
    # 105 of them did on chin, silently pinning the new 181-slot sheet to the
    # old plan's holds. Slot count is the cheap check that catches it.
    if len(plan) != len(entries):
        print(f"  [warn] timing source has {len(plan)} slots but the plan has "
              f"{len(entries)} - stale. No timings attached.")
        return 0

    by_key = {(int(i["section"]), int(i["order"])): i for i in plan}
    n = 0
    for e in entries:
        item = by_key.get((int(e["section"]), int(e["order"])))
        if item:
            e["index"] = item["index"]
            e["start"] = item["start"]
            e["end"] = item["end"]
            e["hold"] = item["duration"]
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", required=True,
                    help="<channel>/<topic>, e.g. wierdbiology/chin")
    ap.add_argument("--out", type=Path, default=None,
                    help="defaults to <project>/build/nb_prompts.txt")
    ap.add_argument("--no-llm", action="store_true",
                    help="emit the template with blank [SLOTS], no API calls")
    ap.add_argument("--background-mode", choices=("name", "ask"), default="name",
                    help="'name' picks a palette colour per image (default); "
                         "'ask' restores the original 'choose a colour' wording, "
                         "which the model cannot honour across images")
    ap.add_argument("--topic", default=None,
                    help="override the plan's topic line")
    ap.add_argument("--repair", action="store_true",
                    help="re-ask ONLY the prompts still holding blank [SLOTS] in "
                         "an existing nb_prompts.json, and leave every filled one "
                         "byte-for-byte alone. A 66-call run drops one now and "
                         "then; this fixes it without rewriting the rest.")
    ap.add_argument("--section", type=int, default=None,
                    help="write the sheet for ONE section only. The plan is still "
                         "read whole, so motion slots stay where the whole-video "
                         "budget put them.")
    ap.add_argument("--export-only", action="store_true",
                    help="write ONLY the .txt sheet from the nb_prompts.json "
                         "already on disk - no API calls, and nb_prompts.json is "
                         "not touched. Use this for a --section sheet once the "
                         "full run has already filled the prompts: passing "
                         "--no-llm here instead would overwrite every filled "
                         "prompt with blanks, because main() always rebuilds "
                         "nb_prompts.json from the plan.")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=shots.SHOT_SEED)
    args = ap.parse_args()

    proj = Project.parse(args.project)
    args.work = proj.build

    if args.export_only:
        prev_path = args.work / "nb_prompts.json"
        if not prev_path.exists():
            raise SystemExit(f"--export-only needs an existing {prev_path} - "
                             f"run a full fill first")
        entries = json.loads(prev_path.read_text(encoding="utf-8"))
        suffix = f"_section{args.section}" if args.section else ""
        out_txt = args.out or (args.work / f"shot_sheet{suffix}.txt")
        out_txt.write_text(render_txt(entries, args.section), encoding="utf-8")
        shown = [e for e in entries
                if args.section is None or e["section"] == args.section]
        print(f"  {len(shown)} slots exported from the existing nb_prompts.json "
              f"(unchanged) -> {out_txt}")
        return

    plan_path = proj.plan
    if not plan_path.exists():
        raise SystemExit(f"{plan_path} not found - run plan_script.py first")
    _plan = json.loads(plan_path.read_text(encoding="utf-8"))
    entries = _plan["images"]
    topic = args.topic or _plan.get("topic", "")

    deterministic_slots(entries, seed=args.seed)
    # Cameras are assigned across ALL motion slots, not just the ones being
    # written, so the "never the same move twice running" property holds for the
    # finished video rather than for whichever section was exported.
    videoprompt.assign_cameras([e for e in entries if e.get("kind") == "video"],
                               seed=args.seed)
    n_timed = attach_timings(entries, proj)
    if args.background_mode == "ask":
        for e in entries:
            e["background"] = BACKGROUND_ASK

    if not args.no_llm:
        load_env()
        if not os.environ.get("OPENROUTER_API_KEY"):
            raise SystemExit(
                "No OPENROUTER_API_KEY.\n"
                "  Add it to f:\\wierd_bio\\.env, or run with --no-llm to emit the\n"
                "  template with blank [SLOTS] and fill them by hand."
            )

    if args.no_llm:
        for e in entries:
            e.update(BLANK)
            if not e["has_character"]:
                e["character_action"] = NO_CHARACTER
            if e.get("kind") == "video":
                e.update(videoprompt.BLANK)
        failures = 0
    elif args.repair:
        prev_path = args.work / "nb_prompts.json"
        if not prev_path.exists():
            raise SystemExit(f"--repair needs an existing {prev_path}")
        prev = {e["filename"]: e for e in
                json.loads(prev_path.read_text(encoding="utf-8"))}
        blanks = []
        for e in entries:
            old = prev.get(e["filename"])
            if old and old.get("subject") != BLANK["subject"]:
                e.update({k: old[k] for k in BLANK if k in old})
            else:
                blanks.append(e)
        for e in entries:
            if e.get("kind") == "video" and "movement" not in e:
                e.update(videoprompt.BLANK)
        if not blanks:
            print("  nothing to repair - every prompt is filled")
            failures = 0
        else:
            print(f"Repairing {len(blanks)} blank prompt(s) via {config.MODEL_SCENES}...")
            failures = fill_all(blanks, workers=args.workers, topic=topic)
    else:
        print(f"Filling {len(entries)} prompts via {config.MODEL_SCENES}...")
        failures = fill_all(entries, workers=args.workers, topic=topic)
        n_motion = sum(1 for e in entries if e.get("kind") == "video")
        if n_motion:
            print(f"Filling {n_motion} animation prompt(s)...")
            failures += fill_video(entries, workers=args.workers, topic=topic)

    suffix = f"_section{args.section}" if args.section else ""
    out_txt = args.out or (args.work / f"shot_sheet{suffix}.txt")
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text(render_txt(entries, args.section), encoding="utf-8")
    (out_txt.parent / "nb_prompts.json").write_text(
        json.dumps(entries, indent=2), encoding="utf-8")

    from collections import Counter
    shown = [e for e in entries
             if args.section is None or e["section"] == args.section]
    n_motion = sum(1 for e in shown if e.get("kind") == "video")
    print(f"\n  {len(shown)} slots written"
          f"  ({len(shown) - n_motion} still, {n_motion} motion)")
    if n_timed:
        print(f"  timings attached for {n_timed} slot(s), measured by Whisper")
    else:
        print("  no render_plan.json yet - sheet has no timings")
    print(f"  shots:       {dict(Counter(e['shot'] for e in shown))}")
    print(f"  backgrounds: {dict(Counter(e['background'].split(' rgb')[0] for e in shown))}")
    if n_motion:
        print(f"  cameras:     {dict(Counter(e['camera'] for e in shown if e.get('kind') == 'video'))}")
    if failures:
        print(f"  {failures} prompt(s) fell back to blank slots - fill them by hand")
    print(f"\n  {out_txt}")
    print(f"  {out_txt.parent / 'nb_prompts.json'}")


if __name__ == "__main__":
    main()
