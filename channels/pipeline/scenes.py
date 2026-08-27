"""
Step 22 -- decide what each frame shows.

For each spoken line, two decisions:

  action   what the character is physically doing        -> goes to FLUX
  caption  the words hand-lettered onto the frame        -> drawn by PIL, never FLUX

The caption is the reference channel's signature move: a short instruction or
punchline sitting in the white space next to the drawing. It is not on every
frame -- mostly on directives, questions, and hard short statements -- so the
model returns "" when a frame should be picture-only.

=============================================================================
 WHY THIS IS LOCAL AND NOT IN THE NOTEBOOK
=============================================================================
It looks like it belongs with the pictures, and it does not. There is no torch
here and no GPU: it is one cheap model call per scene, and on a 150-scene script
that is 150 network round trips run eight at a time. Doing that inside a Colab
session bills GPU session time for something that is pure network wait, and a
disconnect halfway loses the lot.

The FLUX prompt is assembled here too, but only as string concatenation --
STYLE_PREFIX and CHARACTER_BIBLE are constants in config.py, so building the
prompt needs no model weights. Step 23 reads scenes_planned.json and draws.

=============================================================================
 WHAT IS DELIBERATELY *NOT* THE MODEL'S JOB
=============================================================================
Four things make 150 frames read as one artist. Only two of them are FLUX's:

  1. the character never changes   -> CHARACTER_BIBLE, repeated verbatim
  2. the drawing style never changes -> STYLE_PREFIX, repeated verbatim
  3. the colours never change      -> PALETTE, enforced in code after generation
  4. on-screen text is always the same font in the same place -> drawn by code

The visual director prompt is therefore forbidden from mentioning style at all.
If it starts describing the look, it is competing with STYLE_PREFIX and the
frames drift apart.

=============================================================================
 AND ONE THING THAT MUST *NOT* BE THE SAME
=============================================================================
The list above holds the linework constant, which is right, and used to hold the
COMPOSITION constant too, which was wrong: every frame was one figure on a white
wall above a tan floor line. See pipeline/shots.py -- the shot type is assigned
there, deterministically, before any call goes out, and this module writes the
subject for whichever shot a scene was given. A scene's shot also decides whether
CHARACTER_BIBLE enters its prompt at all; a bar chart must not have a stick
figure glued to it.
"""

from __future__ import annotations

import json
import re as _re
import time
from concurrent.futures import ThreadPoolExecutor

from . import config, look, paths, props, shots
from .engine import call_model, parse_json_response

MAX_RETRIES = 3
MAX_WORKERS = 8

# The reply is one short JSON object, ~25 tokens. The cap is far above that on
# purpose: max_tokens is a ceiling, not a charge -- you pay for what is
# generated -- and a reasoning model spends invisible tokens BEFORE the first
# visible character.
#
# The notebook's 150 was fine for Claude and silently fatal for anything that
# reasons. Measured: Ling 3.0 Flash burns 167-232 tokens thinking on this exact
# prompt, so every one of 160 scenes would have returned empty, been caught by
# the fallback below, and produced 160 frames of "standing still facing
# forward" -- a whole video quietly ruined, with only a warning line per scene
# to show for it.
MAX_OUTPUT_TOKENS = 700

SECTION_DIRECTION = {
    1: "tight and physical, close on the body part. The character is doing the thing the viewer was just told to do.",
    2: "wider and more contextual. A comparison, a timeline, or an object being examined.",
    3: "the evidence made visible - a measurement, a comparison, an experiment in progress.",
    4: "process and machinery. Show the mechanism as a simple diagram or cutaway the character interacts with.",
    5: "sparse and wide. Landscape, weather, distance, one small figure. Room to breathe.",
    6: "return to the opening composition, quieter. The character alone, still.",
}

VISUAL_PROMPT_SYSTEM = """You are the visual director for a minimalist hand-drawn
cartoon channel. The drawing style is fixed and handled elsewhere - never mention it.

For ONE line of voiceover, return ONLY this JSON, no fences, no commentary:

{"action": "...", "caption": "...", "prop_key": "..."}

"action": 5-12 words naming ONLY the concrete things in the frame and what they are
  doing. It must be drawable by someone who can draw simple outlined shapes.
  No abstract nouns. No style words. No camera words. No words, letters, labels,
  signs or written text of any kind -- text is drawn separately, never by naming
  it here, and a FLUX model asked to draw "a sign that says X" will attempt to
  paint its own lettering, uncontrolled and never uppercase like the real caption.
  NEVER make a hand, fingers, a palm or a foot the subject. The character is a
  stick figure with no hands or feet drawn in detail, so "a hand presses two
  fingers" asks for something the character does not have, and what comes back is
  a photorealistic hand stuck on a stick body. Say what the WHOLE FIGURE does:
  "the figure touches its own arm", never "a hand presses on an arm".

"prop_key": a SINGLE lowercase word naming the main object in the frame, or "" if
  the frame contains no object at all (a figure on its own needs no prop).
  - lowercase, singular, underscores only, at most three words: "lung", "mri_machine",
    "coffee_mug". Never a sentence, never an adjective, never a plural.
  - If the object is in the KNOWN PROPS list you are shown, you MUST reuse that exact
    key. Reusing is the point: the same key means the same picture in every video,
    and a near-miss like "lungs" or "human_lung" creates a second, different lung.
  - Only invent a key when the object genuinely is not in the list.
  - The key names the OBJECT, never the action: "door", not "figure_opens_door".

"caption": the words to hand-letter onto the frame, or "" for no text.
  - Use a caption when the line is a direct instruction to the viewer, a question, or
    a blunt short statement that lands harder in writing.
  - Use "" when the line is explanatory, descriptive, or long. Most frames get "".
  - Never more than 5 words. Take the words from the voiceover line itself, shortened.
  - Never invent a fact, number or name that is not in the line."""

# What the director is told on top of the system prompt, per shot. The shot is
# already chosen (shots.assign) -- this only says what to write for it.
#
# The caption line is a nudge, not a guarantee. 160 independent nudges do not add
# up to a cap, which is why shots.enforce_caption_cap runs afterwards.
_CAPTION_STEER = {
    "require": "This frame's TEXT IS THE PICTURE. A caption is mandatory - never return \"\".",
    "prefer":  "This shot usually carries a caption. Write one unless the line is long.",
    "neutral": "Caption only if the line genuinely lands harder in writing.",
    "avoid":   "This shot reads on its own. Return \"\" for the caption unless the line "
               "is a direct instruction to the viewer.",
}


# Said only for the lines that need it. A line like "press two fingers against
# the back of your arm" pulls the model into echoing its own words, and "two
# fingers" is the one thing the character does not have -- so without this the
# action is rejected, retried, rejected again, and falls back to a default
# subject. Measured: 3 of 10 scenes fell back this way before this line existed.
_BODY_ACTION_STEER = (
    "This line describes touching a body part, and the words it uses (fingers, "
    "hand, palm) are things this character DOES NOT HAVE - it has plain line "
    "arms. Do not echo them. Write it as the whole figure acting on itself: "
    "\"the figure touches its own arm\", \"the figure holds its own wrist\". "
    "Name the body part being touched, never the part doing the touching.")


def shot_instruction(shot: str, body_action: bool = False) -> str:
    spec = shots.SHOTS.get(shot)
    if not spec:
        return ""
    steer = _CAPTION_STEER.get(spec["caption"], _CAPTION_STEER["neutral"])
    people = ("There are NO PEOPLE in this frame at all - do not mention a person, "
              "a figure, a character or a stickman."
              if not spec["character"] else
              "There ARE people in this frame.")
    out = (f"Shot type: {shot}\n"
           f"What to write: {spec['ask']}\n"
           f"{people}\n"
           f"Captions: {steer}")
    if body_action and spec["character"]:
        out += f"\n{_BODY_ACTION_STEER}"
    return out


# A character-free shot whose action names a person is the one failure worth
# catching here: it survives all the way to a bar chart with a stickman on it,
# and the review step is where you would find out.
#
# Matched on WORD BOUNDARIES, not substring. A naive `w in text` check on short
# words is a trap: "he " is a substring of "the ", so "sits in the corner" was
# being rejected for containing a person, on every action that used the word
# "the" -- which is most of them. Measured on a real run: 24/80 scenes (30%)
# fell back to a default subject, almost all of them character-free shots whose
# only fault was an ordinary sentence. \b anchors both sides so "the", "another"
# and "human" (itself intentionally banned, as a whole word) all behave.
_PERSON_WORDS = ("person", "people", "figure", "character", "stickman", "stick figure",
                 "stick-figure", "man", "woman", "someone", "he", "she", "they",
                 "human", "viewer", "boy", "girl", "child", "kid")
_PERSON_RE = _re.compile(
    r"\b(" + "|".join(_re.escape(w) for w in _PERSON_WORDS) + r")\b", _re.IGNORECASE
)

# A quoted fragment in the action ('a sign that says "CLOSE"', "a beaker
# labeled 'sample'") means the model is naming exact words for FLUX to paint,
# which is the one route by which lettering reaches FLUX outside the single
# controlled caption path -- draw_caption() always uppercases, so a viewer-
# visible text in a case draw_caption never produces did not come from it.
# Deliberately narrow (quote marks only, not a word list like "label" or
# "sign"): those words are also ordinary and legitimate here -- "two bars
# labeled X and Y" is ordinary chart description, not spelled-out lettering --
# and a word-list ban on them would recreate the exact false-positive-reject
# bug the person-word check above already had (see the regression test below).
# CHARACTER_BIBLE says "no hands or feet drawn in detail" and an action that makes
# a hand the subject says the opposite, in the same prompt. FLUX resolves that by
# drawing a photorealistic hand on a stick body -- which reads as an AI tell and
# breaks the character's identity against the 159 frames with plain line limbs.
#
# Only enforced on shots that HAVE a character, because that is where the
# contradiction actually is: on a character-free shot there is no bible in the
# prompt to contradict.
#
# This bans ANY mention of these words, not just a hand used as the grammatical
# subject, and that is broader than it first looks. "a figure presses two fingers
# against its own arm" has the figure as its subject and is still rejected --
# because the character HAS no fingers, so naming them at all invites FLUX to
# draw the detailed hand this exists to prevent. "the figure touches its own arm"
# is the phrasing that works, and it passes.
#
# The cost is real: a script about arms and fingers will hit this, and a rejected
# action falls back to a default subject. That is the trade, taken knowingly --
# a generic frame beats a photorealistic hand welded to a stick body.
# Whole-word matching, for the reason the person check documents.
_ANATOMY_WORDS = ("hand", "hands", "finger", "fingers", "fingertip", "fingertips",
                  "palm", "palms", "thumb", "thumbs", "knuckle", "knuckles",
                  "foot", "feet", "toe", "toes")
_ANATOMY_RE = _re.compile(
    r"\b(" + "|".join(_re.escape(w) for w in _ANATOMY_WORDS) + r")\b", _re.IGNORECASE
)

# A symbolic-metaphor frame ASSERTS "one large symbolic object that dwarfs it" in
# its clause. If the action never names that object, FLUX is told a huge symbol is
# present and given nothing to draw -- and it fills the blank with the most
# generic symbol it knows. Measured: a scene whose line was "press two fingers
# against the back of your arm" came back as a figure beside a giant red heart.
#
# The backstop is crude on purpose: strip the words that cannot BE the object --
# grammar, people, body parts, and the stance verbs a figure does on its own --
# and require something to be left. It does not need to identify the object, only
# to notice that there isn't one.
#
# It errs STRICT, and that asymmetry is deliberate: a wrongly rejected action
# retries and then falls back to _FALLBACK_SUBJECT, which is "standing still
# beside one large plain closed door" -- a real symbolic object, and a fine
# frame. A wrongly ACCEPTED one is the giant red heart contradicting the
# voiceover. Cheap failure on one side, wrong picture on the other.
#
# It is still a word list, so it is not a parser: an action made only of
# adjectives would slip through. It catches the case that actually happened.
_STOPWORDS = frozenset("""
a an the and or of in on at to for with against its it his her their they them
is are was were be been being do does did doing this that these those there here
one two three some any no not up down back front side own into onto from by as
stand stands standing sit sits sitting wait waits waiting walk walks walking
look looks looking watch watches watching stare stares staring point points
pointing touch touches touching press presses pressing hold holds holding
reach reaches reaching turn turns turning move moves moving lean leans leaning
raise raises lower lowers lift lifts face faces facing
still quietly slowly alone motionless again already just
""".split())


def _names_an_object(action: str) -> bool:
    words = [w for w in _re.findall(r"[a-z']+", action.lower())
             if w not in _STOPWORDS]
    return any(not _PERSON_RE.fullmatch(w) and not _ANATOMY_RE.fullmatch(w)
               for w in words)


def _valid(d, shot: str = "") -> bool:
    if not isinstance(d, dict):
        return False
    a = (d.get("action") or "").strip()
    c = (d.get("caption") or "").strip()
    if not a or len(a.split()) > 16:
        return False
    if len(c.split()) > 5:
        return False
    if a.lower().startswith(("here is", "here's", "i cannot", "sure,", "note:")):
        return False
    # Straight or curly double quotes only -- a bare apostrophe is ordinary
    # English (a possessive, a contraction) and banning it would be exactly the
    # kind of over-broad reject the person-word regression test above exists
    # to prevent.
    if any(ch in a for ch in ('"', "“", "”")):
        return False
    spec = shots.SHOTS.get(shot)
    if spec and not spec["character"] and _PERSON_RE.search(a):
        return False
    if spec and spec["character"] and _ANATOMY_RE.search(a):
        return False
    if shot == "symbolic-metaphor" and not _names_an_object(a):
        return False
    return True


def _known_props_block(keys: list[str]) -> str:
    """Show the director the library so it reuses keys instead of inventing them.

    This is the whole mechanism that makes the prop cache work. Normalising a
    freely-invented key afterwards (props.slug) is only a safety net -- it can
    fix "Lungs" into "lung", but nothing downstream can know that "belly" and
    "stomach" were meant to be one picture. Choosing from a list can.
    """
    if not keys:
        return "KNOWN PROPS: (library is empty - invent keys as needed)"
    header = "KNOWN PROPS - reuse one of these EXACTLY if the object is among them:"
    return header + chr(10) + ", ".join(keys)


def plan_scene(scene: dict, known: list[str] | None = None) -> dict:
    """One scene -> {'action', 'caption', 'prop_key'}. Never raises: a scene that
    cannot be planned falls back to a default subject rather than killing a
    150-scene run."""
    direction = SECTION_DIRECTION.get(scene.get("section", 1), "")
    shot = scene.get("shot") or shots.DEFAULT_SHOT
    known = known if known is not None else []
    last = None
    for attempt in range(MAX_RETRIES):
        try:
            raw = call_model(
                config.MODEL_SCENES,
                f"{shot_instruction(shot, shots.is_body_action(scene['text']))}\n\n"
                f"Section direction: {direction}\n\n"
                f"Voiceover line: {scene['text']}",
                system=VISUAL_PROMPT_SYSTEM, max_tokens=MAX_OUTPUT_TOKENS,
                temperature=0.3, retries=1,
            )
            d = parse_json_response(raw)
            # Trim the caption BEFORE validating, never reject over it. The
            # caption is ours -- it is drawn by PIL from this string, so an
            # over-long one costs nothing to shorten. Rejecting instead threw
            # away a perfectly good action because the model returned a six-word
            # caption on a five-word cap, and after three retries the frame fell
            # back to a default subject. Measured: one scene in ten.
            if isinstance(d, dict) and d.get("caption"):
                words = str(d["caption"]).split()
                if len(words) > 5:
                    d["caption"] = " ".join(words[:5])
            if _valid(d, shot):
                caption = (d.get("caption") or "").strip().strip('"')
                # A typographic frame with no text is a blank frame. Borrow from
                # the line rather than ship one.
                if not caption and shots.SHOTS.get(shot, {}).get("caption") == "require":
                    caption = " ".join(scene["text"].split()[:4]).strip(".,;:")
                # Normalise whatever came back. A model that ignored the list
                # and wrote "A pair of Lungs" still lands on "lung"; one that
                # wrote a sentence is dropped to "" rather than poisoning the
                # library with an un-collidable key.
                raw_key = str(d.get("prop_key") or "").strip()
                key = props.slug(raw_key) if raw_key else ""
                if key and not props.VALID_KEY.match(key):
                    key = ""
                return {"action": d["action"].strip().strip('"'),
                        "caption": caption, "prop_key": key}
            last = f"invalid: {raw[:80]!r}"
        except Exception as e:
            last = str(e)
        time.sleep(1.5 * (attempt + 1))
    print(f"  [warn] fallback for: {scene['text'][:55]!r} ({last})")
    return {"action": "", "caption": "", "prop_key": ""}


# The fallback subject per shot. "standing still facing forward" is right for a
# figure and nonsense for a bar chart, and a fallback that contradicts its shot
# clause produces a worse frame than no fallback at all.
_FALLBACK_SUBJECT = {
    "character-moment":  "standing still facing forward",
    "two-character":     "two figures standing still facing each other",
    "symbolic-metaphor": "standing still beside one large plain closed door",
    "object-focus":      "one plain cardboard box",
    "icon-grid":         "three plain simple circles spaced apart",
    "data-chart":        "three plain bars of different heights",
    "typographic":       "one small plain circle low in the corner",
}


def visual_prompt(action: str, shot: str | None = None,
                  expression: str | None = None) -> str:
    """The full FLUX prompt for one frame.

    STYLE_PREFIX is repeated verbatim on every single frame - that repetition IS
    the consistency mechanism, so it must never be paraphrased per scene.

    CHARACTER_BIBLE is repeated verbatim on every frame that HAS a character, and
    omitted entirely from the ones that do not. That distinction is the point of
    the shot rotation: concatenating the bible unconditionally is what put a
    stick figure on the bar charts. The EXPRESSION clause rides with the bible for
    exactly the same reason - a bar chart has no face to put an expression on.
    config.LORA_TRIGGER rides with them too, and leads the subject: without it in
    the prompt the character LoRA is inert and every checkpoint renders identically.

    An unknown or missing shot falls back to the default one rather than dropping
    the clause: STYLE_PREFIX no longer carries the background, so a prompt built
    with no clause at all would not say what room the frame is in.

    The notebook's Step 24 Redraw button rebuilds this string from style.json, and
    test_pipeline.py asserts the two agree for every shot in the catalog.
    """
    name = shot if shot in shots.SHOTS else shots.DEFAULT_SHOT
    spec = shots.SHOTS[name]
    body = action if action else _FALLBACK_SUBJECT.get(name, "")
    if spec["character"]:
        ex = expression if expression in look.EXPRESSIONS else look.DEFAULT_EXPRESSION
        if not getattr(config, "EXPRESSIONS_ON", True):
            ex = look.DEFAULT_EXPRESSION
        # The LoRA trigger leads the subject on character frames only. A frame
        # with no character must not summon the character LoRA's concept -- that
        # is the same reasoning that keeps CHARACTER_BIBLE off the bar charts.
        trigger = getattr(config, "LORA_TRIGGER", "")
        lead = f"{trigger}, " if trigger else ""
        subject = f"{lead}{config.CHARACTER_BIBLE}, {look.EXPRESSIONS[ex]['clause']}, {body}"
    else:
        subject = body

    # THE SUBJECT COMES BEFORE THE SHOT CLAUSE, AND THE ORDER IS THE FIX.
    #
    # It used to be STYLE_PREFIX + clause + subject. The clause carries the room,
    # the floor and the composition -- ~50 tokens of it -- so by the time the
    # character bible arrived, CLIP's 77-token window was already spent and the
    # bible only ever reached T5. Three of six character frames then drew a
    # different character.
    #
    # Now: short style lead, then WHO/WHAT is in the frame, then the composition,
    # then the long tail of negations that only T5 needs. Identity lands inside
    # CLIP's window; content that spills past 77 is still fully seen by T5, which
    # is the encoder that handles content well anyway.
    return (f"{config.STYLE_PREFIX}{subject}, {spec['clause']}"
            f"{getattr(config, 'STYLE_TAIL', '')}")


def run(scenes: list[dict], max_workers: int = MAX_WORKERS,
        strict: bool = False) -> list[dict]:
    """Plan every scene and write scenes_planned.json.

    The shot rotation is decided FIRST, locally and for free, because the shot is
    an input to the prompt. Then the captions are capped and the whole rotation
    is validated -- also locally and for free, before a compute unit is spent."""
    scenes.sort(key=lambda s: s["scene_index"])

    shots.assign(scenes)
    print(f"Shot rotation assigned for {len(scenes)} scenes "
          f"({len(set(s['shot'] for s in scenes))} of {len(shots.SHOTS)} shot types used).")

    # AFTER the shot, never before: the shot decides whether the frame has a
    # face to put an expression on at all.
    look.assign(scenes)
    print(f"Expressions assigned: "
          f"{len(set(s['expression'] for s in scenes if s['expression']))} distinct "
          f"faces across {sum(1 for s in scenes if s['expression'])} character frames.")

    known = props.known_keys()
    print(f"Planning visuals for {len(scenes)} scenes ({max_workers} at a time)"
          f" against {len(known)} known prop(s)...")
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        plans = list(pool.map(lambda sc: plan_scene(sc, known), scenes))

    for scene, plan in zip(scenes, plans):
        scene["action_clause"] = plan["action"]
        scene["caption"] = plan["caption"]
        scene["prop_key"] = plan.get("prop_key", "")

    dropped = shots.enforce_caption_cap(scenes)

    # After the cap, not before: a stripped caption changes nothing in the FLUX
    # prompt (captions are drawn by PIL) but the prompt is built here anyway so
    # there is exactly one place that does it.
    for scene in scenes:
        scene["visual_prompt"] = visual_prompt(
            scene["action_clause"], scene["shot"], scene.get("expression"))

    with open(paths.p("scenes_planned.json"), "w", encoding="utf-8") as f:
        json.dump(scenes, f, indent=2)

    n_fb = sum(1 for s in scenes if not s["action_clause"])
    n_cap = sum(1 for s in scenes if s["caption"])
    print(f"\nDone. {len(scenes) - n_fb}/{len(scenes)} scenes planned"
          f"{f', {n_fb} fell back to a default subject' if n_fb else ''}.")
    print(f"{n_cap} frames will carry on-screen text"
          f"{f' ({dropped} caption(s) dropped to stay under the cap)' if dropped else ''}.\n")

    print("Shot rotation:")
    print(shots.summary(scenes))
    print()
    print("Expressions:")
    print(look.summary(scenes))
    print()
    for s in scenes[:6]:
        print(f'  #{s["scene_index"]}  [{s["shot"]}'
              f'{"/" + s["expression"] if s.get("expression") else ""}]  '
              f'{s["action_clause"]}')
        print(f'       caption: {s["caption"] or "(none)"}')

    problems = shots.validate(scenes) + look.validate(scenes)
    if problems:
        print("\n" + "-" * 70)
        print("ROTATION WARNINGS - these are the repetition the rotation exists to stop:")
        for p in problems:
            print(f"  - {p}")
        print("-" * 70)
        if strict:
            raise SystemExit(
                "\nRefusing to hand this to the GPU with --strict set.\n"
                "  Nothing has been spent yet. Adjust pipeline/shots.py (weights,\n"
                "  max_share, SECTION_SHOTS) and re-run -- the rotation is seeded,\n"
                "  so the same timeline replans identically."
            )
    else:
        print("\nRotation is clean: no adjacent repeats, every shot inside its cap.")

    paths.mark_done("scenes", ["scenes_planned.json"])
    return scenes


def load_timeline() -> list[dict]:
    """Read scenes_timeline.json, the notebook's Step 17 output.

    Checked rather than assumed: this file crosses the machine boundary, so a
    partial Drive sync or a stale hand-edit shows up here as a clear error
    instead of as 150 frames planned against the wrong text."""
    f = paths.p("scenes_timeline.json")
    if not f.exists():
        raise FileNotFoundError(
            f"{f} not found.\n"
            "  It is written by the notebook's Step 17 (voice alignment), alongside\n"
            "  vo_raw.wav. Run that first, and let Drive finish syncing."
        )
    with open(f, "r", encoding="utf-8") as fh:
        scenes = json.load(fh)

    if not isinstance(scenes, list) or not scenes:
        raise ValueError(f"{f} is empty or not a list of scenes.")

    required = {"scene_index", "text", "section", "duration"}
    missing = required - set(scenes[0])
    if missing:
        raise ValueError(
            f"{f} scenes are missing {sorted(missing)}. Expected the Step 17 shape: "
            f"scene_index, start, end, duration, text, section."
        )
    return scenes
