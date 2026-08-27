"""
Shot-type rotation -- the part of the look that must NOT repeat.

=============================================================================
 WHY THIS MODULE EXISTS
=============================================================================
config.py's "THE LOOK" block lists four things that make 160 frames read as one
artist. Three of them are still right. One of them was quietly making every
frame the same picture:

    the character never changes        -> right, keep
    the drawing style never changes   -> right, keep
    the colours never change          -> WRONG at the scene level (see below)
    the text is always in one place    -> WRONG at the scene level (see below)

The reference channel holds the *linework and the character* constant and
changes the *composition* on every beat. Frames alternate between narrative
shots (a figure doing something) and abstract ones (a bar chart, an icon grid,
a machine with nobody in it, one big hand-lettered word). No two consecutive
frames are the same kind of picture.

That distinction is the whole point. Repeated LINEWORK reads as a style.
Repeated COMPOSITION reads as a template -- to a viewer, and to the platform's
repetitious-content checks.

Before this module there was exactly one variation lever, SECTION_DIRECTION in
scenes.py, at six-bucket granularity: every scene inside section 4 got the same
direction, so a 30-scene section produced 30 near-identical compositions.

=============================================================================
 WHAT IS THE MODEL'S JOB AND WHAT IS NOT
=============================================================================
Assignment is NOT the model's job. Which shot a scene gets is decided here, in
Python, deterministically, before any call goes out -- for the same reason the
palette lock is not the model's job. A rule you can only ask for is not a rule.
The model is told which shot to write for and writes the subject; it never picks.

The rotation is also seeded, so re-running run_scenes.py on the same timeline
plans the same shots. A rotation that reshuffles on every run makes a redraw
non-reproducible.
"""

from __future__ import annotations

import random
import re as _re
from collections import Counter

# Same rotation for the same timeline, every run. See the module note above.
SHOT_SEED = 7919

# No more than this many narrative frames in a row before an abstract one has to
# break the run. Three is the reference channel's own rhythm: a couple of story
# beats, then a chart or a diagram, then back.
MAX_NARRATIVE_RUN = 3

# Captions are emphasis. Past roughly a third of frames they stop reading as
# emphasis and start reading as a lower third, which is the templated look this
# module exists to avoid.
CAPTION_MAX_SHARE = 0.35


# =============================================================================
#  THE SHOT CATALOG
# =============================================================================
#  family    "narrative" puts a person on screen, "abstract" does not. The
#            alternation between the two is what MAX_NARRATIVE_RUN enforces.
#
#  character whether CHARACTER_BIBLE goes into the FLUX prompt at all. This is
#            the field that matters most: with it always True, a bar-chart frame
#            came out with a stick figure glued to it, because the bible was
#            concatenated unconditionally.
#
#  clause    a fixed composition fragment appended to STYLE_PREFIX for this shot.
#            Fixed, not model-written, so all the data-chart frames in a video
#            are built the same way even though they show different data.
#
#            THE BACKGROUND AND THE FLOOR LIVE HERE, not in STYLE_PREFIX. They
#            used to be in the prefix, which put every frame in the same room.
#            The colours are unchanged -- the narrative shots still say white wall
#            and tan floor band in the prefix's exact old words, so the palette
#            lock still has the same six colours to snap to. What changed is that
#            an abstract shot can now say it has no floor, instead of being handed
#            a prompt that asserts both.
#
#  ask       what the visual director is told to return for this shot.
#
#  caption   "require" | "prefer" | "neutral" | "avoid"  -- how hard to push for
#            on-screen text. A typographic frame IS its text; a two-character
#            scene almost never needs any.
#
#  weight    relative frequency. Higher shows up more. Picking is by
#            used_count/weight, so these are ratios, not percentages.
#
#  max_share hard ceiling on this shot's fraction of the video, enforced during
#            assignment. This is what stops the rotation collapsing back onto
#            one favourite composition.
# =============================================================================

# The floor-and-wall sentence that used to sit in STYLE_PREFIX, verbatim. Kept as
# one constant so the shots that DO have a room all describe it identically -- and
# so the palette lock is provably looking at the same six colours it always was.
# THE FLOOR WAS RENDERING AS A THIN LINE, NOT A BAND.
#
# "one horizontal floor line with a flat tan floor band along the bottom" names
# the LINE first and the BAND second, as a modifier. FLUX drew the line and
# treated the band as optional -- our frames came back with a scratchy horizontal
# stroke and white below it, sometimes with a visible GAP in the middle. The
# reference channel fills the bottom quarter of the frame with solid tan, which
# is a large part of why theirs reads as a finished drawing and ours as a sketch.
#
# So the band is now the subject, its extent is stated as a fraction of the
# frame, and "unbroken" is explicit because the gap was a real observed defect.
# ROUND 2. "the bottom quarter of the frame filled with a solid flat tan floor"
# came back as ORANGE SCRIBBLE -- loose diagonal crayon hatching across the
# lower frame, not a band. "filled" and "solid" were not enough against a model
# that had been told the whole picture was "hand-drawn".
#
# Naming the SHAPE is what fixes this: a rectangle is a thing FLUX can draw
# flatly. "one solid rectangle of flat tan colour" leaves no room for texture,
# and the anti-hatching negations now also live in STYLE_TAIL.
# ROUND 3. Round 2's "lower fifth ... one solid rectangle of flat tan colour"
# made the floor WORSE, not better: it came back as a faint speckled smudge, or
# vanished entirely on 3 of 9 frames.
#
# Cause: the anti-texture negations added to STYLE_TAIL in round 2 ("no scribble
# fill, no pencil texture") were applied by FLUX to the tan area itself, so it
# stopped filling it at all. The same negation trap the character bible fell
# into originally -- the model does not process "no X" as absence, it just
# suppresses X wherever X might have been.
#
# Fix: describe the floor as a THING IN THE SCENE the figure stands on, which is
# how the reference channel's frames actually read, and let the fill follow from
# "solid". The texture negations that killed it are removed from STYLE_TAIL.
# ROUND 4 -- THE FLOOR IS NO LONGER ASKED FOR AT ALL.
#
# Three rounds of wording ("band", "rectangle", "lower fifth", "fills the
# bottom", "solid flat") produced a thin line, an orange scribble, or nothing,
# and differently on different frames. It is a rectangle; pipeline/finish.py
# paints it deterministically, the way the palette lock and the captions are
# already done. See the long note in config.py.
#
# Asking for it here as well would be actively harmful: FLUX would draw its own
# horizon line at its own height, which finish.py would then have to bury. So
# the room is now the WALL ONLY, and the frame is composed with empty space at
# the bottom for the band to land in.
_ROOM = ("pure white background wall, large areas of empty white space, "
         "the figure standing with empty space below it")

# No room at all. The abstract shots say this instead, so the prompt never asserts
# a floor and a lack of floor at the same time.
_VOID = "pure white background, no floor line, no wall, large areas of empty white space"

# THE PROPS ARE COLOURED. THE CHARACTER IS NOT. Both halves matter.
#
# CHARACTER_BIBLE ends "no colour anywhere on the character" and sits in the
# prompt immediately after the object description, so a shot that never said the
# prop WAS coloured left that ban as the frame's only colour statement, and FLUX
# generalised it. The reference channel is the opposite: a white-and-black
# figure next to one boldly filled prop.
#
# AND THE COLOURS ARE NAMED, which is the other half.
#
# It used to say only "one flat solid colour" -- which one was never stated
# anywhere in the prompt. The only colours named at all were white (the wall),
# black (the linework) and tan (the floor), so FLUX chose prop colours freely
# and produced pink brains, blue microscopes and green circles.
#
# That matters because config.PALETTE is enforced AFTER generation, by snapping
# every pixel to its nearest of six colours. Run the real lock_palette() on what
# FLUX actually produced and the props do not survive it:
#
#     pink brain    (232,168,200)  ->  (255,255,255)  white   -- invisible
#     blue eyeball  (140,195,235)  ->  (255,255,255)  white   -- invisible
#     green clock   ( 90,165, 95)  ->  (120, 72, 40)  brown
#
# A palette enforced in code but never requested in the prompt is not a lock, it
# is a demolition. So the three prop colours the palette actually contains are
# named here, and FLUX now aims at colours that survive the snap unchanged.
#     rust red   #B6462D   the "one bold prop" colour
#     wood brown #C48C50   furniture, tables, doors
#     dark brown #784828   depth and shadow on wood
# ROUND 2. The giant brain came back photo-realistic: modelled lobes, cast
# shadow, specular highlights. "one flat solid colour" was already here and lost
# to the sheer detail FLUX packs into a known object drawn large.
#
# "simple flat shape" is the fix -- it constrains the SILHOUETTE, not just the
# fill, which is what stops the model reaching for its realistic prior.
# ROUND 3. "no interior detail" WAS THE THING BLOCKING THE REFERENCE LOOK.
#
# Held side by side with the reference channel, the gap was never that our model
# could not draw complex objects -- the round-2 note above records it drawing a
# brain with modelled lobes and specular highlights, i.e. drawing one far too
# well. The gap was that this clause FORBADE the detail that makes the
# reference's objects readable at all:
#
#     an MRI machine  needs the bore, the bed, the control panel
#     a mosquito      needs wings, legs, a proboscis
#     a sofa          needs the cushion seams
#
# Every one of those is "interior detail", and every one was banned here. So the
# pipeline was asking for LESS complexity than the channel it was chasing, and
# then the six-colour palette lock removed the colour from what was left.
#
# The distinction that actually matters is not detail vs no detail. It is FLAT
# vs SHADED. The reference is full of interior lines; it has no gradients. So
# the ban moves off detail and onto rendering: interior lines are allowed and
# encouraged, shading and gradients are not, and every enclosed area is one flat
# colour. That is a constraint the palette lock can enforce, which "no interior
# detail" never was.
#
# The colour list widens with config.PALETTE for the same reason -- see the note
# there. Naming colours the lock actually contains is what keeps props alive
# through the snap.
_PROP = ("drawn as a bold flat illustration: thick black outlines, interior "
         "detail drawn as more black outlines, every enclosed area filled with "
         "one flat solid colour from rust red, warm wood brown, dark brown, "
         "teal, navy, forest green, olive, warm beige or yellow, with no "
         "shading, no gradients and no highlights")

SHOTS: dict[str, dict] = {
    "character-moment": {
        "family": "narrative",
        "character": True,
        # NO PROP CLAUSE HERE, deliberately. "any prop filled with one flat solid
        # colour" was added to make props coloured, and FLUX read it as an
        # assertion that a prop EXISTS -- a frame whose action was only "the
        # figure touches its own forearm" came back with an invented blue mug
        # standing on the floor. A prop is optional on this shot, so the shot
        # must not claim one. STYLE_PREFIX's "flat solid colour fills" already
        # colours whatever the action actually names.
        "clause": f"one figure alone, medium shot, {_ROOM}",
        "ask": "ONE figure and at most one prop. Say what the figure is physically doing.",
        "caption": "neutral",
        "weight": 5.0,
        "max_share": 0.34,
    },
    "two-character": {
        "family": "narrative",
        "character": True,
        # Same reason as character-moment above: the shared object is optional
        # here, so naming it in the clause invents one.
        "clause": f"two figures facing each other on the same floor line, "
                  f"medium-wide shot, empty space between them, {_ROOM}",
        "ask": "TWO figures and what passes between them -- one gesture, one shared object. "
               "Say what each is doing.",
        "caption": "avoid",
        "weight": 2.5,
        "max_share": 0.20,
    },
    "symbolic-metaphor": {
        "family": "narrative",
        "character": True,
        "clause": f"one figure and one large symbolic object that dwarfs it, wide shot, "
                  f"the object placed off to one side and {_PROP}, {_ROOM}",
        "ask": "ONE figure plus ONE oversized concrete object standing in for the idea "
               "(a door, a mirror, a wall, a scale). No abstract nouns -- NAME THE "
               "OBJECT, always. The frame asserts a huge symbol is present, so an "
               "action that names none leaves the model to invent one, and it "
               "reaches for a cliche (a giant heart) that contradicts the line.",
        "caption": "prefer",
        "weight": 2.0,
        "max_share": 0.18,
    },
    "object-focus": {
        "family": "abstract",
        "character": False,
        # "centred" IS GONE, and that word was the bug. It sat next to "resting
        # on a stand or shelf" and the two fought: once the subject moved ahead
        # of this clause, "centred" won and pulled the object to the middle of
        # the frame while the surface stayed at the bottom. Measured: a jar that
        # had been sitting on its table in the previous render came back
        # levitating above it. Grounding beats centring -- an object floating in
        # mid-air is the single most obvious AI tell this pipeline produces.
        "clause": f"one large object alone resting directly on a simple flat surface "
                  f"drawn with the same thick black outline, {_PROP}, "
                  f"no people anywhere in the frame, {_ROOM}",
        # "nothing anatomical" USED TO EXCLUDE THE WHOLE ANIMAL KINGDOM.
        #
        # The rule exists to stop a real defect -- a photorealistic detached
        # forearm on a wooden slab, from a body-action line that landed on this
        # character-free shot. That defect is about DETACHED HUMAN BODY PARTS,
        # and the wording overshot into banning any organic subject at all.
        #
        # A biology channel cannot draw a mosquito, a neuron, a deer or a lung
        # under that ban, and those are exactly the frames the reference channel
        # is built from. So the ban is narrowed to what actually broke: a human
        # body part shown on its own, and organic/manufactured hybrids.
        "ask": "ONE recognizable real-world object or whole living creature -- an "
               "animal, an insect, a plant and a machine are all fine, and drawing "
               "its real distinguishing features is wanted, not avoided. No people. "
               "No DETACHED human body part on its own, and no hybrid of a body "
               "part with an object. Name the subject, the features that make it "
               "recognizable, and what it is sitting on or held by.",
        "caption": "neutral",
        "weight": 3.0,
        "max_share": 0.22,
    },
    "icon-grid": {
        "family": "abstract",
        "character": False,
        "clause": f"three to five small separate objects spaced apart, each resting on "
                  f"its own small flat base drawn with the same thick black outline, "
                  f"each {_PROP}, no people anywhere in "
                  f"the frame, {_VOID}",
        # Narrowed for the same reason as object-focus above: the ban was meant
        # for detached human body parts and was excluding every animal, insect
        # and organ a biology channel needs to draw.
        "ask": "THREE to FIVE small separate, recognizable real-world things -- "
               "objects, animals, insects or plants. No people, no DETACHED human "
               "body part on its own, no hybrid of a body part with an object. "
               "List them plainly.",
        "caption": "prefer",
        "weight": 2.0,
        "max_share": 0.16,
    },
    "data-chart": {
        "family": "abstract",
        "character": False,
        "clause": f"a simple hand-drawn bar chart, the bars {_PROP}, "
                  f"no people anywhere in the frame, {_VOID}",
        "ask": "A comparison drawn as bars: what is being compared, and which bar is the "
               "outlier. No people. Only quantities that are actually in the line.",
        "caption": "prefer",
        "weight": 1.5,
        "max_share": 0.12,
    },
    "typographic": {
        "family": "abstract",
        "character": False,
        # THE WORD "FRAME" IS GONE FROM HERE, deliberately. This clause opened
        # with "a nearly empty frame" -- the most heavily weighted position in
        # the prompt -- and FLUX read "frame" as a PICTURE FRAME and drew a
        # hand-drawn black border around all four edges of the image. Every
        # other shot says "in the frame" mid-sentence and is fine; it was
        # leading with it that did it. "picture" everywhere here instead.
        "clause": f"almost the entire picture left blank and empty, one small simple "
                  f"shape low in the corner {_PROP}, nothing at all in the "
                  f"middle, no people anywhere in the picture, no border and no box "
                  f"drawn around the edges, {_VOID}",
        "ask": "Almost nothing: name ONE tiny simple shape for the corner. The frame is "
               "mostly empty because the words are the picture.",
        "caption": "require",
        "weight": 1.5,
        "max_share": 0.12,
    },
}

# Which shots suit which section. Read alongside SECTION_DIRECTION in scenes.py:
# that sets the tone and the width, this sets what kind of picture it is.
#
# Section 1 opens on a body and section 6 closes on the same figure, so both are
# narrative-leaning. The evidence sections in the middle are where the charts and
# the grids belong -- which is also where a long unbroken run of talking-head
# frames used to happen.
SECTION_SHOTS: dict[int, tuple[str, ...]] = {
    1: ("character-moment", "symbolic-metaphor", "typographic", "object-focus"),
    2: ("two-character", "object-focus", "data-chart", "icon-grid", "character-moment"),
    3: ("data-chart", "object-focus", "icon-grid", "character-moment", "typographic"),
    4: ("object-focus", "icon-grid", "character-moment", "typographic", "two-character"),
    5: ("symbolic-metaphor", "icon-grid", "character-moment", "object-focus"),
    6: ("character-moment", "symbolic-metaphor", "typographic"),
}

ALL_SHOTS = tuple(SHOTS)
ABSTRACT_SHOTS = tuple(n for n, s in SHOTS.items() if s["family"] == "abstract")

DEFAULT_SHOT = "character-moment"


# =============================================================================
#  THE ONE PLACE THE ROTATION LOOKS AT WHAT THE LINE SAYS
# =============================================================================
#  The rotation is content-blind on purpose -- that indifference is what stops
#  160 frames reading as a template. But indifference has one failure it cannot
#  absorb: a line that is an intrinsically TIGHT PHYSICAL ACTION on the body,
#  handed to a shot that demands an oversized symbolic object.
#
#  Measured, on a real frame: the line "Press two fingers against the back of
#  your arm" drew symbolic-metaphor. There is no giant symbol for touching your
#  own arm, so the model returned a body action instead, the clause still
#  asserted "one large symbolic object that dwarfs it", and FLUX filled the
#  blank with the most generic symbol it knows -- a giant red heart, next to a
#  voiceover about goosebumps. No prompt wording fixes that; the shot and the
#  line simply cannot both be satisfied.
#
#  Section 1's own direction already says "tight and physical, close on the body
#  part", so the rotation was fighting the section direction as well.
#
#  This is deliberately the NARROWEST possible content rule: it removes exactly
#  one shot from the pool for exactly these lines, and changes nothing else. It
#  is not a general content-aware assignment -- see PLAN.md before making it one.
# =============================================================================
_BODY_VERBS = frozenset("""
press presses pressing pressed push pushes pushing pushed
touch touches touching touched feel feels feeling felt
rub rubs rubbing rubbed scratch scratches scratching scratched
squeeze squeezes squeezing pinch pinches pinching
tap taps tapping brush brushes brushing stroke strokes
""".split())

# Only the OUTSIDE of the body -- the parts a viewer can touch on themselves
# right now. Organs are deliberately absent: "a figure holding a brain" is a
# perfectly good symbolic-metaphor frame, and one of the best this pipeline has
# produced, so "brain" must not trip this rule.
_BODY_WORDS = frozenset("""
arm arms forearm forearms hand hands finger fingers fingertip fingertips
thumb wrist wrists elbow elbows palm palms knuckle knuckles
skin neck nape scalp hair face cheek cheeks jaw chin lip lips tongue
eye eyes ear ears nose shoulder shoulders chest stomach belly
leg legs knee knees ankle foot feet toe toes
""".split())

# A body-action line has to be drawn as THE CHARACTER doing it. Nothing else
# works, and the failure mode is worse than it sounds.
#
# The first version of this rule only banned symbolic-metaphor. The line then
# drew object-focus instead -- a character-free shot -- and the model wrote "two
# fingers press against the back of an arm" as the OBJECT. What came back was a
# photorealistic peach-coloured forearm on a wooden slab: no stick figure, no
# channel style, nothing recognisable. That is a bigger break than the giant
# heart it replaced, because at least the heart was drawn in the right style.
#
# ONE shot, not two. two-character was in this list and had to come out: a body
# action is intrinsically SOLO -- "touch your own arm" is one person -- but the
# two-character clause asserts "two figures facing each other ... empty space
# between them". The model correctly wrote "THE figure touches its own arm",
# singular, and the render put a second stick figure in the frame with nothing
# to do, standing there idle. A spare character is not as ugly as a
# photorealistic hand, but it is still a frame with an unexplained person in it.
BODY_ACTION_SHOTS = ("character-moment",)


def is_body_action(text: str) -> bool:
    """Does this line ask the viewer to do something to their own body?

    Both halves required -- a touching verb AND an outside-the-body part. Either
    alone is far too common: "hold" appears in any sentence, and "skin" appears
    in every other line of a script about skin."""
    words = set(_re.findall(r"[a-z']+", (text or "").lower()))
    return bool(words & _BODY_VERBS) and bool(words & _BODY_WORDS)


def _pool(section: int) -> tuple[str, ...]:
    return SECTION_SHOTS.get(section) or ALL_SHOTS


def _pick(cands: list[str], used: Counter, rng: random.Random) -> str:
    """Lowest used/weight ratio wins, ties broken by the seeded rng.

    Dividing by weight rather than comparing raw counts is what makes `weight`
    behave like a ratio: character-moment at 5.0 gets picked roughly twice as
    often as object-focus at 3.0 without any explicit quota."""
    best = min(cands, key=lambda n: used[n] / SHOTS[n]["weight"])
    tied = [n for n in cands
            if abs(used[n] / SHOTS[n]["weight"]
                   - used[best] / SHOTS[best]["weight"]) < 1e-9]
    return rng.choice(sorted(tied))


def assign(scenes: list[dict], seed: int = SHOT_SEED) -> list[dict]:
    """Set scene["shot"] on every scene. Mutates and returns the list.

    Constraints are applied hardest-first and relaxed one at a time only when a
    scene would otherwise have no candidate at all. Assignment can therefore
    never fail -- a 160-scene run must not die because a rule painted itself
    into a corner on scene 157.

    Relaxation order, weakest constraint dropped first:
        1. the per-shot ceiling      (max_share)
        2. no repeat of the previous shot
        3. the section's shot pool
    The narrative-run break is NOT relaxed: it looks past the section pool
    instead, because breaking a long run of talking heads matters more than
    section affinity does.
    """
    rng = random.Random(seed)
    n = len(scenes)
    caps = {name: max(1, round(n * spec["max_share"])) for name, spec in SHOTS.items()}
    used: Counter = Counter()
    prev: str | None = None
    narrative_run = 0

    for i, scene in enumerate(scenes):
        pool = list(_pool(int(scene.get("section", 1) or 1)))

        # The one content rule -- see is_body_action() above.
        body = is_body_action(scene.get("text", ""))

        # The first frame is pinned, not rotated. Every shot ties at zero usage on
        # scene 0, so the seeded rng was opening the video on whatever came up --
        # once on a typographic frame, which is a nearly empty picture. Two things
        # depend on the opening being a figure: SECTION_DIRECTION[1] ("tight and
        # physical, close on the body part"), and section 6's "return to the
        # opening composition", which needs an opening composition to return to.
        if i == 0:
            scene["shot"] = DEFAULT_SHOT
            used[DEFAULT_SHOT] += 1
            narrative_run = 1 if SHOTS[DEFAULT_SHOT]["family"] == "narrative" else 0
            prev = DEFAULT_SHOT
            continue

        if body:
            # Body actions outrank BOTH the section pool and the narrative-run
            # break. Extending a run of talking heads by one frame is a mild
            # repetition problem; a detached photorealistic forearm is not a
            # frame from this channel at all. validate() still reports the long
            # run afterwards, so the trade stays visible rather than silent.
            cands = ([s for s in BODY_ACTION_SHOTS
                      if s != prev and used[s] < caps[s]]
                     or [s for s in BODY_ACTION_SHOTS if s != prev]
                     or list(BODY_ACTION_SHOTS))
        elif narrative_run >= MAX_NARRATIVE_RUN:
            # Force the break. Prefer an abstract shot from this section, but
            # take any abstract shot rather than extend the run.
            forced = [s for s in pool if SHOTS[s]["family"] == "abstract"] \
                     or list(ABSTRACT_SHOTS)
            cands = [s for s in forced if s != prev and used[s] < caps[s]] \
                    or [s for s in forced if s != prev] \
                    or forced
        else:
            cands = [s for s in pool if s != prev and used[s] < caps[s]] \
                    or [s for s in pool if s != prev] \
                    or [s for s in ALL_SHOTS if s != prev] \
                    or list(pool)

        shot = _pick(cands, used, rng)
        scene["shot"] = shot
        used[shot] += 1
        narrative_run = narrative_run + 1 if SHOTS[shot]["family"] == "narrative" else 0
        prev = shot

    return scenes


def enforce_caption_cap(scenes: list[dict], max_share: float = CAPTION_MAX_SHARE) -> int:
    """Drop captions until at most `max_share` of frames carry text. Returns the
    number dropped.

    Deterministic, and done here rather than asked for in the prompt, because
    the caption bias in the prompt is a nudge and 160 independent nudges do not
    add up to a guarantee. Frames whose shot is "require" (typographic -- the
    text IS the picture) are never stripped; the rest go in bias order, worst
    first, so the captions that survive are the ones that were most wanted."""
    order = {"avoid": 0, "neutral": 1, "prefer": 2, "require": 3}
    captioned = [s for s in scenes if (s.get("caption") or "").strip()]
    allowed = int(len(scenes) * max_share)
    if len(captioned) <= allowed:
        return 0

    droppable = sorted(
        (s for s in captioned
         if SHOTS.get(s.get("shot", ""), {}).get("caption") != "require"),
        key=lambda s: (order.get(SHOTS.get(s.get("shot", ""), {}).get("caption", "neutral"), 1),
                       s.get("scene_index", 0)),
    )
    dropped = 0
    for scene in droppable:
        if len(captioned) - dropped <= allowed:
            break
        scene["caption"] = ""
        dropped += 1
    return dropped


def validate(scenes: list[dict]) -> list[str]:
    """Return a list of complaints -- empty means the rotation is clean.

    This is the check that this whole module is for. It is cheap, it needs no
    model and no GPU, and it runs on the laptop before a single compute unit is
    spent, which is the same reason run_qc.py exists for the audio."""
    problems: list[str] = []
    n = len(scenes)
    if not n:
        return ["no scenes to validate"]

    shots = [s.get("shot") for s in scenes]

    missing = [s.get("scene_index") for s in scenes if not s.get("shot")]
    if missing:
        problems.append(f"{len(missing)} scene(s) have no shot assigned: {missing[:8]}")

    repeats = [(scenes[i].get("scene_index"), shots[i])
               for i in range(1, n) if shots[i] and shots[i] == shots[i - 1]]
    if repeats:
        problems.append(
            f"{len(repeats)} adjacent frame(s) repeat the same shot: "
            + ", ".join(f"#{i} {s}" for i, s in repeats[:6]))

    # Checked against the same integer ceiling assign() enforced, not against the
    # raw fraction. At small n a single frame can exceed any percentage -- one
    # scene out of seven is 14% -- and a validator that fires on arithmetic
    # rather than on repetition is one you learn to ignore.
    counts = Counter(s for s in shots if s)
    caps = {name: max(1, round(n * spec["max_share"])) for name, spec in SHOTS.items()}
    for name, count in counts.most_common():
        cap = caps.get(name)
        if cap is not None and count > cap:
            problems.append(f"shot {name!r} is on {count}/{n} frames "
                            f"({count / n:.0%}), ceiling is {cap}")

    run, worst = 0, 0
    for shot in shots:
        run = run + 1 if SHOTS.get(shot, {}).get("family") == "narrative" else 0
        worst = max(worst, run)
    if worst > MAX_NARRATIVE_RUN:
        problems.append(f"{worst} narrative frames in a row (limit {MAX_NARRATIVE_RUN}) "
                        f"- that stretch will read as one long talking head")

    n_cap = sum(1 for s in scenes if (s.get("caption") or "").strip())
    if n_cap / n > CAPTION_MAX_SHARE + 0.02:
        problems.append(f"{n_cap}/{n} frames carry text ({n_cap / n:.0%}, "
                        f"cap {CAPTION_MAX_SHARE:.0%}) - captions stop reading as "
                        f"emphasis and start reading as a lower third")

    # The relaxation chain in assign() can fall back to ALL_SHOTS in a corner
    # case, which could put a body-action line back on symbolic-metaphor. Rare,
    # and never silent: it is the frame that produced the giant red heart.
    mismatched = [sc.get("scene_index") for sc in scenes
                  if sc.get("shot") not in BODY_ACTION_SHOTS
                  and is_body_action(sc.get("text", ""))]
    if mismatched:
        problems.append(
            f"{len(mismatched)} body-action line(s) landed on a shot that cannot "
            f"show a figure doing it: {mismatched[:6]} - these render as detached "
            f"photorealistic body parts, not as this channel's character")

    unused = [name for name in SHOTS if not counts.get(name)]
    if len(unused) > 2:
        problems.append(f"{len(unused)} shot types never used: {', '.join(sorted(unused))}")

    return problems


def summary(scenes: list[dict]) -> str:
    """A one-block readout of the rotation, for the end of run_scenes.py."""
    n = len(scenes) or 1
    counts = Counter(s.get("shot") or "(none)" for s in scenes)
    lines = []
    for name, count in counts.most_common():
        spec = SHOTS.get(name, {})
        fam = spec.get("family", "?")[:4]
        caps = sum(1 for s in scenes
                   if s.get("shot") == name and (s.get("caption") or "").strip())
        lines.append(f"    {name:<18} {count:>4}  {count / n:>4.0%}  {fam:<5} "
                     f"{caps} with text")
    return "\n".join(lines)
