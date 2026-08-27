"""
Expression -- what the character's face is doing on this frame.

=============================================================================
 WHAT THIS MODULE IS, AND WHAT IT DELIBERATELY IS NOT ANY MORE
=============================================================================
This started out larger. A background axis (saturated cyan / yellow) and a
framing axis (extreme close-up ... very wide) were built here alongside the
expression, as an attempt at the reference channel's look, and both were
REVERTED after seeing real frames: the palette rebrand was not the channel's
look and nobody asked for it. The warm white-wall/tan-floor room in config.py
and the composition rotation in shots.py are the look.

What survived is the one part that was actually an improvement: THE FACE.

CHARACTER_BIBLE used to end "...two short curved black lines for eyes, one
small curved line for the mouth" -- one locked neutral expression on all 160
frames. The construction of the figure still never changes (circle head, white
fill, thin line limbs, no colour), but the eyes-and-mouth clause is chosen per
scene, so a shocking beat gets a shocked face instead of the same blank one.

=============================================================================
 CHOSEN FROM THE LINE, NOT ROTATED, AND NOT ASKED FOR
=============================================================================
The expression is decided here, in Python, before any call goes out -- same
rule as shots.assign() and the palette lock. A rule you can only ask for is not
a rule, and a face on a rotation would be a face that is enraged because rage
was "due", not because the line was.

So the driver is an ENERGY SCORE of the scene's own text: lexical, free, and
deliberately dumb. No model call, no GPU -- on the pipeline's highest-volume
step, a sentiment model would cost a call per scene to steer a choice between
six faces. Being dumb also makes it reproducible: the same line scores the same
every run, so a replan gives the same face and a redraw matches its neighbours.

=============================================================================
 THE ONE RISK, NAMED
=============================================================================
PLAN.md defers per-scene visual variance because prompt variance drives
character drift, and drift is the thing there is no fix for yet. An expression
DOES change the face, which is exactly the thing that has to stay recognisable
across 160 frames -- so this is the one axis that genuinely spends some of that
budget. It is on because a single frozen face across a whole video is the
templated look in miniature, and config.EXPRESSIONS_ON = False turns it off in
one edit.
"""

from __future__ import annotations

import random
import re
from collections import Counter

from . import config, shots

# Separate from shots.SHOT_SEED so retuning one does not reshuffle the other.
# Same contract: the same timeline must look identical on a replan.
LOOK_SEED = 4271


# =============================================================================
#  THE ENERGY SCORE
# =============================================================================
#  Every signal below is something you can see in the raw text without knowing
#  what it means. Weights are deliberately coarse -- the output is one of three
#  bands, so precision past that is wasted.
# =============================================================================

# Words that mark a line as alarming, visceral or absolute. Kept short on
# purpose: a long list drifts toward matching every line, and a score that
# fires on everything is the same as one that fires on nothing.
_HOT_WORDS = frozenset("""
kill kills killed killing dead death die dies dying corpse blood bleeding
pain agony scream screaming horror terror terrifying panic shock shocking
brutal violent savage disaster catastrophe collapse explode exploded
trapped suffocate drown burned burning torn crushed rip ripped
never always nobody everyone impossible insane crazy wrong lie lied liar
worst deadly fatal doomed vanish vanished destroy destroyed
""".split())

# Second person, and the imperative openers that go with it. A line aimed
# straight at the viewer is a line the character should be reacting to.
_DIRECT_WORDS = frozenset("""
you your yours imagine picture consider try press touch close look listen
watch feel notice stop think remember count hold
""".split())

# Scale words. "two million years" is contemplative, not urgent.
_SCALE_WORDS = frozenset("""
million millions billion billions thousand thousands century centuries
millennia era epoch universe galaxy planet ocean continent everywhere
generations ancestors evolution extinct
""".split())

_WORD_RE = re.compile(r"[a-z']+")

HIGH_BAND = 0.55
MID_BAND = 0.28


def _forms(word: str):
    """The word, plus the obvious inflections stripped back.

    Without this the lists above have to spell out every form of every verb, and
    the one that gets forgotten is the one that costs a frame: "scream" was in
    the list and "screamed" was not, so a line about someone screaming scored
    below a line about a measurement."""
    yield word
    if len(word) > 3 and word.endswith("s"):
        yield word[:-1]
    if len(word) > 4 and word.endswith("ed"):
        yield word[:-2]
    if len(word) > 5 and word.endswith("ing"):
        yield word[:-3]


def _hits(words: list[str], vocab: frozenset) -> int:
    return sum(1 for w in words if any(f in vocab for f in _forms(w)))


def energy(text: str) -> float:
    """A 0.0-1.0 temperature for one voiceover line. Deterministic and free.

    Not calibrated against anything -- there is nothing to calibrate against.
    The only property that matters is ORDERING: a line about someone screaming
    must out-score a line about a measurement, and it does."""
    if not text or not text.strip():
        return 0.0
    lowered = text.lower()
    words = _WORD_RE.findall(lowered)
    if not words:
        return 0.0

    score = 0.0
    score += 0.25 * min(2, lowered.count("!"))
    score += 0.20 * min(2, lowered.count("?"))

    score += min(0.36, 0.18 * _hits(words, _HOT_WORDS))
    score += min(0.20, 0.10 * _hits(words, _DIRECT_WORDS))

    # A short line is a punch; a long one is explanation.
    #
    # 10, not 8. At 8 a direct instruction to the viewer -- "Press two fingers
    # against the back of your arm", nine words -- scored 0.20 and landed in the
    # LOW band. Ordinary narration runs 10-20 words, so 10 still marks a short
    # line without catching explanation.
    if len(words) <= 10:
        score += 0.15
    elif len(words) >= 22:
        score -= 0.10

    # Scale pulls the other way: contemplative, not urgent.
    if _hits(words, _SCALE_WORDS):
        score -= 0.12

    return max(0.0, min(1.0, score))


def band(score: float) -> str:
    """high / mid / low. Three bands because six faces split three ways."""
    if score >= HIGH_BAND:
        return "high"
    if score >= MID_BAND:
        return "mid"
    return "low"


# =============================================================================
#  THE EXPRESSION CATALOG
# =============================================================================
#  Each clause replaces the eyes-and-mouth sentence that used to be frozen into
#  CHARACTER_BIBLE. Everything else about the figure is still the bible's, word
#  for word, so the character is the same character wearing a different face.
#
#  bands      which energy bands this face is eligible for.
#  weight     relative frequency, as a ratio (used_count/weight decides).
#  max_share  ceiling on this face's share of the CHARACTER frames, so one
#             expression cannot quietly become the new frozen default.
# =============================================================================
# THE FACES WERE TOO SMALL TO READ.
#
# Side-by-side with the reference channel, the single biggest expressive gap was
# scale. Theirs draws LARGE eyes with visible dark pupils and thick brows that
# fill much of the head, so the emotion lands at thumbnail size -- which is the
# size a YouTube viewer actually sees first. Ours said "two small dots" and "two
# short curved lines", and the face read as blank at any size below full frame.
#
# Every clause below now states the eyes are LARGE and fill much of the head.
# That is a deliberate spend of drift budget: the face is the thing that has to
# stay recognisable across 160 frames, and a bigger feature is a feature with
# more room to vary. It is worth it, because an unreadable face is not
# consistent -- it is just uniformly blank.
EXPRESSIONS: dict[str, dict] = {
    "deadpan": {
        "clause": "two big round white eyes with large black pupils, one straight "
                  "line mouth, a blank deadpan stare",
        "bands": ("low", "mid"),
        "weight": 3.0,
        "max_share": 0.34,
    },
    "rage": {
        "clause": "two big round white eyes with large black pupils under thick black "
                  "eyebrows angled down into a hard V, mouth wide open shouting",
        "bands": ("high",),
        "weight": 2.0,
        "max_share": 0.20,
    },
    "shock": {
        "clause": "two huge round white eyes with large black pupils, mouth a small "
                  "round open O of shock",
        "bands": ("high",),
        "weight": 2.0,
        "max_share": 0.20,
    },
    "smug": {
        "clause": "two big white eyes with large black pupils half covered by heavy "
                  "lids, one thick black eyebrow raised high, mouth a smug curve",
        "bands": ("mid", "high"),
        "weight": 1.5,
        "max_share": 0.16,
    },
    "worried": {
        "clause": "two big round white eyes with large black pupils under thick black "
                  "eyebrows tilted up in the middle, mouth a small worried curve",
        "bands": ("low", "mid"),
        "weight": 1.5,
        "max_share": 0.16,
    },
    "delight": {
        "clause": "two eyes squeezed shut into thick black upward arcs, mouth open in "
                  "a wide grin",
        "bands": ("mid", "high"),
        "weight": 1.2,
        "max_share": 0.14,
    },
}

DEFAULT_EXPRESSION = "deadpan"


def _pick(cands: list[str], used: Counter, rng: random.Random) -> str:
    """Lowest used/weight wins, ties broken by the seeded rng. Same rule as
    shots._pick -- `weight` behaves as a ratio, not a quota."""
    best = min(cands, key=lambda n: used[n] / EXPRESSIONS[n]["weight"])
    ref = used[best] / EXPRESSIONS[best]["weight"]
    tied = [n for n in cands
            if abs(used[n] / EXPRESSIONS[n]["weight"] - ref) < 1e-9]
    return rng.choice(sorted(tied))


def assign(scenes: list[dict], seed: int = LOOK_SEED) -> list[dict]:
    """Set scene["expression"] on every scene. Mutates and returns the list.

    Must run AFTER shots.assign(): the shot decides whether there is a face in
    the frame at all. Four of the seven shots have none, and an expression on a
    bar chart is the CHARACTER_BIBLE bug wearing a hat.

    Like shots.assign(), this can never fail -- constraints relax rather than
    leaving a scene unassigned."""
    rng = random.Random(seed)
    n = len(scenes) or 1
    caps = {k: max(1, round(n * s["max_share"])) for k, s in EXPRESSIONS.items()}
    used: Counter = Counter()
    prev: str | None = None

    for scene in scenes:
        score = energy(scene.get("text", ""))
        tone = band(score)
        scene["energy"] = round(score, 3)
        scene["tone"] = tone

        shot = scene.get("shot") or shots.DEFAULT_SHOT
        spec = shots.SHOTS.get(shot, shots.SHOTS[shots.DEFAULT_SHOT])

        if not spec["character"]:
            scene["expression"] = ""          # nothing with a face in this frame
            continue
        if not getattr(config, "EXPRESSIONS_ON", True):
            scene["expression"] = DEFAULT_EXPRESSION
            continue

        by_band = [e for e, s in EXPRESSIONS.items() if tone in s["bands"]] \
            or [DEFAULT_EXPRESSION]
        # The band is a PREFERENCE (this face suits this energy); max_share is a
        # CEILING that stops one face becoming the new frozen default. So the
        # band is dropped before the ceiling is.
        cands = ([e for e in by_band if e != prev and used[e] < caps[e]]
                 or [e for e in EXPRESSIONS if e != prev and used[e] < caps[e]]
                 or [e for e in by_band if e != prev]
                 or by_band)
        expression = _pick(cands, used, rng)
        used[expression] += 1
        prev = expression
        scene["expression"] = expression

    return scenes


def validate(scenes: list[dict]) -> list[str]:
    """Complaints about the expressions -- empty means clean. Free and local,
    run before a compute unit is spent, same as shots.validate()."""
    problems: list[str] = []
    n = len(scenes)
    if not n:
        return ["no scenes to validate"]

    # A face on a frame with nobody in it is the CHARACTER_BIBLE bug in a new
    # place, and it is the one failure here worth catching hard.
    faced = [s.get("scene_index") for s in scenes
             if s.get("expression")
             and not shots.SHOTS.get(s.get("shot", ""), {}).get("character", False)]
    if faced:
        problems.append(f"{len(faced)} character-free frame(s) were given an "
                        f"expression: {faced[:8]}")

    char = [s for s in scenes
            if shots.SHOTS.get(s.get("shot", ""), {}).get("character", False)]
    if not char:
        return problems

    missing = [s.get("scene_index") for s in char if not s.get("expression")]
    if missing:
        problems.append(f"{len(missing)} character frame(s) have no expression: "
                        f"{missing[:8]}")

    counts = Counter(s.get("expression") for s in char if s.get("expression"))
    for name, count in counts.items():
        cap = max(1, round(n * EXPRESSIONS[name]["max_share"]))
        if count > cap:
            problems.append(f"expression {name!r} is on {count}/{len(char)} character "
                            f"frames, ceiling is {cap}")

    return problems


def summary(scenes: list[dict]) -> str:
    """A one-block readout, printed next to shots.summary() by run_scenes.py."""
    n = len(scenes) or 1
    out = []
    ex = Counter(s.get("expression") for s in scenes if s.get("expression"))
    if ex:
        out.append("  expression (character frames only):")
        for name, count in ex.most_common():
            out.append(f"    {name:<18} {count:>4}  {count / n:>4.0%}")
    tones = Counter(s.get("tone") or "?" for s in scenes)
    out.append("  energy: " + "  ".join(f"{k}={tones.get(k, 0)}"
                                        for k in ("high", "mid", "low")))
    return "\n".join(out)
