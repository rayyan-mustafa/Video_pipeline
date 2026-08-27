"""
Every constant the pipeline is steered by. No API calls, no I/O at import time.

=============================================================================
 WHAT CHANGED COMING OUT OF THE NOTEBOOK, AND WHY
=============================================================================
Three notebook globals mutated themselves after import. That is harmless when
cells run once, top to bottom, and quietly wrong the moment anything runs twice
in one process -- which a script does constantly.

  1. BEAT_SHEET[3] was ASSIGNED at runtime from the evidence profile, and
     BANNED_PHRASES was REBOUND to itself plus the learnings file. Running two
     topics in one process would carry topic A's Section 3 into topic B.
     -> build_beat_sheet(evidence_type) now returns a fresh dict every call.
     -> banned_phrases(learnings) returns a fresh list every call.

  2. LEARNINGS was loaded at import from a bare relative path, so it depended
     on os.chdir() having already happened.
     -> load_learnings() reads through paths.p() and is called explicitly.

  3. TOPIC_KEY was a module-level constant edited by hand.
     -> now a --topic argument. The bank below is unchanged.

Everything else is a verbatim lift. The comments are the notebook's own and
document real failures; they are worth more than the lines they sit above.
"""

from __future__ import annotations

import json

from . import paths

# =============================================================================
#  WHICH MODEL DOES WHICH JOB
# =============================================================================
#  These are OPENROUTER SLUGS, not Anthropic API model ids. Everything routes
#  through one OPENROUTER_API_KEY -- see engine.py.
#
#  THE NOTEBOOK'S CHECKER SLUG WAS WRONG AND WOULD HAVE HARD-FAILED. It used
#  "claude-haiku-4-5-20251001", which is the Anthropic API's dated id. OpenRouter
#  does not route that; the slug is "anthropic/claude-haiku-4.5". With
#  allow_fallbacks off, the first call on the checker path would have died --
#  and the checker is the FIRST paid call in the run (the evidence profiler),
#  so nothing would have worked at all.
#
#  Verified against openrouter.ai/api/v1/models/<model>/endpoints on 2026-08-23.
#  Run `python run_script.py --preflight` to re-verify before a run.
#
#  Prices, per million tokens (input/output):
#    anthropic/claude-sonnet-5     $2.000 / $10.000   (cheaper than $3/$15 direct)
#    anthropic/claude-haiku-4.5    $1.000 / $ 5.000
#    deepseek/deepseek-v4-flash    $0.049 / $ 0.098
#    inclusionai/ling-3.0-flash    $0.042 / $ 0.126
#
# =============================================================================
#  FIVE ROLES, NOT THREE. The split is by WHAT THE OUTPUT IS, not by how hard
#  the task feels:
#
#    prose a viewer reads or hears  -> Sonnet. No substitutions.
#    a judgment call                -> Sonnet. A weak judge fails INVISIBLY,
#                                      by passing everything (vo_coach's
#                                      `clean_take` case, in a different hat).
#    machinery nobody reads         -> the cheapest model that does the job.
#
MODEL_WRITER    = "anthropic/claude-sonnet-5"    # script, line fixes, section rebuilds, hooks, packaging
MODEL_JUDGE     = "anthropic/claude-sonnet-5"    # retention gate + hook ranking

# ONE call per video, and it decides everything downstream: `hard_numbers` is
# the writer's entire statistics allowance, and `evidence_type` picks the whole
# shape of Section 3. A number this call misses is a fact the script can never
# use, silently. Upgraded from Haiku for about $0.01 a video - the cheapest
# quality insurance in the pipeline.
MODEL_EXTRACTOR = "anthropic/claude-sonnet-5"    # the evidence profile

# Mechanical matching: does this sentence follow from the source it cites, and
# is this [0]-tagged line smuggling a claim? Yes/no against text that is right
# there in the prompt. Deliberately NOT downgraded further - this is the gate
# that stops a false claim reaching a viewer, and the savings would be ~$0.05.
MODEL_CHECKER   = "anthropic/claude-haiku-4.5"   # fact-check + smuggled-claim audit

# ~160 calls per video, by far the highest volume, and the most constrained
# task in the pipeline: one sentence -> 5-12 drawable words + an optional
# 5-word caption. Measured on real scene prompts, DeepSeek V4 Flash with
# reasoning off gives the same answer quality as Haiku for 1/37th the price
# ($0.0024 vs $0.088 per video), and Step 24 lets you redraw anything odd.
MODEL_SCENES    = "deepseek/deepseek-v4-flash"   # visual direction, 160x per video


# =============================================================================
#  WHICH UPSTREAM HOST SERVES EACH MODEL
# =============================================================================
#  Pinned PER MODEL, not globally: the right host is a property of the model,
#  and a global pin makes it impossible to mix providers. Every tag below was
#  read from openrouter.ai/api/v1/models/<model>/endpoints on 2026-08-23 --
#  never off the providers tab, which is how vo_coach lost a call to a slug
#  that did not exist.
#
#  A model with no entry here is REFUSED rather than routed freely. Unpinned
#  routing is how a run silently costs several times what the last one did.
# =============================================================================
PROVIDER_PINS = {
    # Claude. Several tags serve these at the SAME price, so the second entry is
    # a free failover rather than a compromise: if Anthropic 529s mid-run, Vertex
    # picks it up at $2.00/$10.00 too. The REGIONAL tags are the ones to avoid --
    # google-vertex/us is $2.20/$11.00 for the identical model.
    "anthropic/claude-sonnet-5":   ["anthropic", "google-vertex/global"],
    "anthropic/claude-haiku-4.5":  ["anthropic", "google-vertex/global"],
    "anthropic/claude-opus-5":     ["anthropic", "google-vertex/global"],

    # DeepSeek V4 Flash, for the 160 scene-direction calls. Two hosts tie at the
    # cheapest price, so both are pinned - primary and in-pin backup.
    #     baidu/fp8       $0.049 / $0.098
    #     streamlake/fp8  $0.049 / $0.098
    #     gmicloud/fp8    $0.059 / $0.118
    #     deepinfra/fp8   $0.090 / $0.180   (~2x, for the same weights)
    "deepseek/deepseek-v4-flash":  ["baidu/fp8", "streamlake/fp8"],

    # Ling 3.0 Flash, for cheap end-to-end test runs. THREE hosts serve it and
    # the spread is the point:
    #     novita          $0.021 / $0.063   cheapest
    #     novita/fast     $0.042 / $0.126   2x the price
    #     deepinfra/bf16  $0.060 / $0.180
    # `novita/fast` is pinned rather than the cheapest. vo_coach measured plain
    # Novita at roughly a TENTH of Fast's speed, and a full script run emits
    # ~15k output tokens: about six minutes of waiting to save half a cent.
    # Cheapest is the wrong optimisation when the absolute cost is already
    # ~$0.01 for the whole pipeline.
    "inclusionai/ling-3.0-flash":  ["novita/fast"],
}

# =============================================================================
#  REASONING, PER MODEL  --  measured, not guessed
# =============================================================================
#  A reasoning model bills and TIMES its chain-of-thought as output tokens that
#  never appear in the reply. Left alone that makes the cost and the latency
#  budget fiction, and it can exhaust max_tokens before a single visible
#  character is produced.
#
#  Ling 3.0 Flash reasons, hard. Measured on a real scene-planning prompt
#  (2026-08-23), max_tokens=600:
#
#      setting                     out    reasoning   visible   $/call
#      (unset)                     183       167        16      0.0000355
#      effort: low                 197       172        25      0.0000373
#      effort: medium              243       232        11      0.0000431
#      effort: high                239       221        18      0.0000425
#      reasoning: {exclude: true}  169       164         5      0.0000335
#      reasoning: {enabled: false}  23         0        23      0.0000151
#
#  Two things that only measurement would have told you:
#
#  1. EFFORT DOES NOT HELP. "low" thinks slightly MORE than unset. Pinning it,
#     the way vo_coach pins it for gpt-oss-120b, would have achieved nothing
#     here -- the setting is model-specific, not a general lever.
#  2. `exclude: true` IS A TRAP. It hides the reasoning from the response and
#     still bills all 164 tokens of it. Only `enabled: false` actually stops it.
#
#  Turning it off makes the cheap preset 2.4x cheaper and the answer is just as
#  good -- this is scene captions and fact-matching, not analysis.
#
#  Value is passed straight through as OpenRouter's `reasoning` field. Unset
#  means the field is not sent at all, which is right for Claude.
# =============================================================================
#  DeepSeek V4 Flash reasons too, and just as expensively. Measured on a real
#  scene prompt: 243 reasoning tokens ON vs 0 OFF, $0.0000386 -> $0.0000148, and
#  the reasoning-off answer was the BETTER of the two ("Stick figure points past
#  a bright star" against "Figure points at the farthest bright star"). Thinking
#  is not free and is not always an improvement.
#
#  Claude is absent from this dict on purpose: omitting the field entirely is
#  the right default there.
REASONING: dict[str, dict] = {
    "inclusionai/ling-3.0-flash":  {"enabled": False},
    "deepseek/deepseek-v4-flash":  {"enabled": False},
}


# =============================================================================
#  MODEL PRESETS
# =============================================================================
#  `--models cheap` swaps every role onto one small model so the whole pipeline
#  can be exercised end to end for about a cent. It is for proving the WIRING,
#  not the writing: a small model will pass gates a strong one fails, so never
#  judge a script produced under this preset.
# =============================================================================
MODEL_PRESETS = {
    # The one to ship with. Sonnet where a viewer sees the output or a judgment
    # is being made; cheaper models only on machinery.
    "claude": {
        "writer":    "anthropic/claude-sonnet-5",
        "judge":     "anthropic/claude-sonnet-5",
        "extractor": "anthropic/claude-sonnet-5",
        "checker":   "anthropic/claude-haiku-4.5",
        "scenes":    "deepseek/deepseek-v4-flash",
    },
    # Everything on one small model, ~$0.01 a run. Proves the WIRING only.
    "cheap": {
        "writer":    "inclusionai/ling-3.0-flash",
        "judge":     "inclusionai/ling-3.0-flash",
        "extractor": "inclusionai/ling-3.0-flash",
        "checker":   "inclusionai/ling-3.0-flash",
        "scenes":    "inclusionai/ling-3.0-flash",
    },
    # Sonnet everywhere, including the machinery. A control run: if `claude`
    # output ever looks wrong, this tells you whether a cheap role caused it.
    "all-sonnet": {
        "writer":    "anthropic/claude-sonnet-5",
        "judge":     "anthropic/claude-sonnet-5",
        "extractor": "anthropic/claude-sonnet-5",
        "checker":   "anthropic/claude-sonnet-5",
        "scenes":    "anthropic/claude-sonnet-5",
    },
}

_ROLE_GLOBALS = {
    "writer": "MODEL_WRITER", "judge": "MODEL_JUDGE", "extractor": "MODEL_EXTRACTOR",
    "checker": "MODEL_CHECKER", "scenes": "MODEL_SCENES",
}


def apply_preset(name: str) -> dict:
    """Point every role at a preset's models. Returns what was applied."""
    if name not in MODEL_PRESETS:
        raise ValueError(
            f"Unknown model preset {name!r}. Known: {', '.join(sorted(MODEL_PRESETS))}"
        )
    preset = MODEL_PRESETS[name]
    for role, slug in preset.items():
        globals()[_ROLE_GLOBALS[role]] = slug
    return dict(preset)


def active_models() -> dict:
    """The slug behind each role right now."""
    return {role: globals()[g] for role, g in _ROLE_GLOBALS.items()}

# --- Narration ---------------------------------------------------------------
# This notebook is your-voice-only. You record one file per section; the narration
# step aligns your words against the script to recover scene boundaries, and writes
# vo_raw.wav + scenes_timeline.json. There is no TTS path and no voice conversion.
NARRATION = "human"

# =============================================================================
#  THE LOOK  —  read this before you change anything below
# =============================================================================
#  The reference style is: black-outlined hand-drawn shapes, flat solid colour
#  fills, and the script line hand-lettered in the empty space. Four things make
#  it read as ONE artist across 150 frames:
#
#    1. the character never changes            -> CHARACTER_BIBLE, repeated verbatim
#                                                 on every frame THAT HAS ONE
#    2. the linework never changes             -> STYLE_PREFIX, repeated verbatim
#    3. the colours never change               -> PALETTE, enforced in code after
#                                                 generation (see Step 19)
#    4. the on-screen text is always the same
#       font in the same place                 -> drawn by code, never by FLUX
#
#  Only 1 and 2 are FLUX's job, and FLUX is unreliable at them. 3 and 4 are done
#  deterministically in Python, which is why they are the parts you can trust.
#
#  ---------------------------------------------------------------------------
#  AND THE FIFTH THING, WHICH IS THE OPPOSITE OF THE OTHER FOUR
#  ---------------------------------------------------------------------------
#  The four rules above hold the LOOK constant. Read too broadly, they held the
#  PICTURE constant too: STYLE_PREFIX used to carry the wall and the floor, so
#  all 160 frames were a figure standing in the same room, and the only variation
#  lever was SECTION_DIRECTION at six-bucket granularity.
#
#    repeated LINEWORK    reads as a style
#    repeated COMPOSITION reads as a template
#
#  So the composition rotates: see pipeline/shots.py. Seven shot types, no two
#  consecutive frames alike, an abstract frame breaking every run of three
#  narrative ones, and four of the seven with no character in them at all --
#  which is why rule 1 above now says "that has one". The background and the
#  floor moved OUT of STYLE_PREFIX and into the shot clause for the same reason.
#
#  The palette has NOT moved, and a cyan/yellow rebrand of it was tried and
#  reverted -- the warm white-wall/tan-floor room is the channel's look. What
#  survived that experiment is ONE thing: the FACE.
#
#  The character's construction is still constant (circle head, white fill,
#  line limbs, no colour), but the eyes-and-mouth clause is now per scene, so a
#  shocking beat gets a shocked face instead of the same neutral one 160 times.
#  See pipeline/look.py -- assigned in Python from a free lexical energy score
#  of the line, never asked for in the prompt.
#
#  Per-scene colour stays deferred, for the original reason: prompt variance
#  drives character drift, and drift has no fix yet. Expressions do raise that
#  variance, which is why EXPRESSIONS_ON exists to switch them off in one edit.
# =============================================================================

# THE FACE IS NO LONGER IN HERE, AND THAT IS DELIBERATE.
#
# This used to end "...two short curved black lines for eyes, one small curved
# line for the mouth..." -- a single locked neutral face on all 160 frames. The
# reference channel does the opposite: the construction of the figure never
# changes, and the EXPRESSION is the whole performance (exaggerated rage, blank
# stares, smugness -- meme-native, not corporate).
#
# So everything here is the invariant construction, repeated verbatim on every
# frame that has a character, and the eyes-and-mouth clause comes from
# pipeline/look.py per scene. Everything else about the figure -- including
# "no colour anywhere on the character" -- is unchanged and still verbatim on
# every frame that has one. See THE LOOK above.
# WRITTEN IN THE POSITIVE, AND THAT IS THE WHOLE POINT.
#
# This used to be six negations: "no nose, no ears, no hair, no hands or feet
# drawn in detail, no clothes, no colour". Diffusion models do not process "no
# X" as absence -- the token is still in the prompt, and it frequently SUMMONS
# the thing instead. Measured on the first real render: 3 of 6 character frames
# came back with a nose, ears, detailed fingered hands and shoes, i.e. exactly
# the six things the bible forbade, while the frames whose prompt happened to
# land differently drew a clean stick figure. Three different character designs
# across six frames.
#
# So every clause below states what IS there. The only way to get "no ears" out
# of a diffusion model is to describe a head so completely that ears have no
# room left, never to ask for their absence.
# WHY THIS WORDING CHANGED AGAIN (the oval-body and finger defects)
#
# Side-by-side against the reference channel, our figure came back with an OVAL
# FILLED TORSO and small fingered hands -- the two defects this bible has been
# trying to prevent since the first render. The previous wording, "a body of five
# thin straight lines - one spine, two arms, two legs - blunt tips at the wrists
# and ankles", failed for a structural reason:
#
#   "a BODY of five thin straight lines" leads with the noun "body", and a body
#   is a solid thing. FLUX drew a body and then drew lines near it. The count
#   was honoured; the geometry was not.
#
# So the torso is now named as a SINGLE STRAIGHT LINE, first, as its own clause,
# before any word that could denote a solid mass. "Stick" is repeated because it
# is the one token that carries the whole geometry on its own.
#
# The hands are handled the same way: "blunt tips" described an absence of
# detail, which is a negation wearing positive clothes. It is now a positive
# shape -- the limb simply ENDS.
# ROUND 2. The previous wording still produced an OVAL TORSO, FINGERS and SHOES
# on 6 of 9 test frames. Three separate causes, all fixable in the wording:
#
#   1. "drawn as single THIN lines" directly contradicted STYLE_PREFIX's "very
#      thick heavy marker outline" -- in the SAME prompt. FLUX resolved the
#      conflict toward thin. The word "thin" is gone.
#
#   2. "one straight vertical line for the whole TORSO" still names a torso, and
#      a torso is a body part with volume. FLUX drew the volume and put a line
#      near it. The noun is now "stroke", which has no mass.
#
#   3. "each limb ending in a plain rounded stub" mentions limb ENDINGS at all,
#      which invites the model to draw something there -- it drew hands and
#      shoes. The reference channel's figure has no hands and no feet: the
#      strokes simply stop. So endings are no longer mentioned. The only thing
#      said about the limbs is that they are straight strokes, and that every
#      stroke is the same thickness (which is also what makes the linework read
#      as deliberate rather than sketchy).
#
# Shorter is also better here: this sits inside CLIP's 77-token window, and
# every token spent on the figure is a token the shot clause cannot have.
# ROUND 3. The oval torso is FIXED by round 2's wording and stays. Two residual
# defects remain, both additive rather than structural:
#
#   HANDS. Fingers and thumbs still appear, driven by the ACTION clause -- an
#   action like "points at" summons a pointing hand. The counter is to state
#   positively what the arm IS along its whole length: a bare line. "bare"
#   describes the stroke, it does not ask for the absence of a hand.
#
#   HAIR. One frame came back with a full head of hair. "plain circle" was not
#   enough on its own, so the head is now explicitly bald and empty -- again
#   stated as what it IS.
CHARACTER_BIBLE = (
    "one stick figure: a plain bald circle for the head, one single straight "
    "vertical stroke for the body, four bare straight strokes for the two arms "
    "and two legs, every stroke the same even thickness"
)

# THE LORA TRIGGER, AND WHY IT HAS TO BE IN THE PROMPT AT ALL.
#
# The character LoRA was trained with --instance_prompt "stkmn style", so the
# whole learned character lives behind that phrase. If it never appears at
# inference time the LoRA loads, fuses, reports no error -- and changes nothing.
# Measured: base, checkpoint-250 and checkpoint-500 rendered pixel-for-pixel
# indistinguishable across all 10 test frames, because not one prompt contained
# the trigger. A LoRA comparison run that way can only ever conclude "no
# difference", no matter how well the training went.
#
# It goes at the FRONT of the subject, ahead of CHARACTER_BIBLE, for the same
# reason the subject goes ahead of the shot clause: CLIP's 77-token window is
# the scarce resource, and identity has to land inside it.
#
# Set to "" to draw with the base model wording (the A/B control) without
# touching any other part of the prompt.
#
# TURNED OFF, because the LoRA it triggers is abandoned.
#
# The trained LoRA learned a degenerate direction, not a character: fused at 8x
# it ERASES the drawing, at 15x it renders pure noise. It is not undertrained,
# it is wrong, and no number of extra steps fixes a bad gradient. Meanwhile the
# base model with the current prompts draws the figure correctly.
#
# With no LoRA loaded, "stkmn style" is two tokens FLUX has never seen, sitting
# in the most expensive real estate in the prompt -- the front of CLIP's 77-token
# window, ahead of the character description. It is not neutral either: adding it
# was measured to move 2.36% of pixels, i.e. it is acting as noise.
#
# Restore it ONLY alongside a LoRA that has been shown to work.
LORA_TRIGGER = ""

# WHAT BELONGS IN HERE, AND WHAT DOES NOT
#
# Only the LINEWORK -- the part that is identical in every frame of the channel.
# The background wall and the floor band used to live here too, which meant every
# one of 160 frames was a figure on a white wall above a tan strip. That is the
# templated look, baked in at the prompt level.
#
# The background and the floor now belong to the shot clause (pipeline/shots.py),
# because they are the thing that must vary. A frame that is a bar chart has no
# floor; a frame that is a figure in a room does. Leaving the floor here and
# putting "no floor line" in the clause hands FLUX a prompt that contradicts
# itself, which is how this split was found.
#
# The colours have NOT changed: the narrative shots still say white wall and tan
# floor, in these exact words, so the palette lock is untouched.
# SPLIT IN TWO, BECAUSE CLIP ONLY READS THE FIRST 77 TOKENS.
#
# FLUX has two text encoders: T5-XXL reads up to 512 tokens and does the
# semantic work, and CLIP has a HARD 77-token cap and produces the pooled vector
# that steers global style. Our prompts run ~191 tokens, so CLIP was seeing only
# the old STYLE_PREFIX and part of the room description -- it never once
# received the character bible, the expression, or the action. Colab logged the
# truncation on every single frame.
#
# STYLE_PREFIX is now the short front of the prompt, so the linework AND the
# character identity both fit inside CLIP's window. STYLE_TAIL carries the rest,
# which only T5 needs to see: the anti-text negations exist to stop FLUX
# scribbling lettering, and T5 is the encoder that actually honours them.
# THE WORDS THAT WERE MAKING US LOOK AMATEUR
#
# Compared side-by-side with the reference channel, our frames read as a thin
# scratchy PENCIL SKETCH while theirs reads as a confident BOLD MARKER drawing.
# Three tokens were doing that damage, and all three were ours, not FLUX's:
#
#   "crude"                       -> asked for roughness. We got roughness.
#   "slightly wobbly hand-drawn"  -> asked for a shaky line. We got a shaky line.
#   "simple childlike drawing"    -> asked for a child's drawing. We got one.
#
# None of those were describing the target. The reference look is *clean and
# deliberate*: uniform stroke width, closed confident curves, bold fills. It is
# simple, but it is not crude and it is not childlike -- that distinction is the
# whole difference between "a style" and "a bad drawing".
#
# So the prefix now asks for the weight of line we actually want, and the tail
# keeps ONLY the anti-lettering negations (which T5 does honour and which stop
# FLUX scribbling fake text).
# ROUND 2. "bold clean hand-drawn cartoon" still produced thin wobbly pen lines,
# and the props came back with real shading -- the giant brain rendered with
# highlights and cast shadow, which the palette lock then has to demolish.
#
# "hand-drawn" was part of the problem: it invites pen-sketch texture. The
# target is closer to a flat vector illustration that HAPPENS to be simple.
# "even width" is stated because uniform stroke weight is the single strongest
# signal separating a deliberate style from a doodle.
# ROUND 3. Two changes, both from the side-by-side with the reference channel:
#
#   1. LINE WEIGHT. "thick black strokes" still rendered thin. Naming the TOOL
#      rather than the result is a stronger signal to a diffusion model -- it has
#      seen far more images captioned "marker pen drawing" than ones captioned
#      "strokes of even width".
#
#   2. FIGURE SIZE. This is the one that actually fixes the faces. Face quality
#      tracked FIGURE SIZE, not the expression wording: on medium shots where the
#      figure was large the eyes came out big and expressive, and on wide shots
#      with a small figure the identical expression clause rendered as two dots.
#      The reference channel's figure fills roughly 70% of the frame height; ours
#      was landing near 40%. So the framing is now stated up front, where CLIP
#      will see it.
STYLE_PREFIX = (
    "bold minimal stick-figure drawing made with a thick black marker pen, "
    "heavy black outlines, flat solid colour fills, plain white background, "
    "the figure drawn large and filling most of the height of the picture, "
)

# The negations that remain are the ones T5 demonstrably honours: lettering, and
# now texture. "no shading, no gradients" was already here and the brain ignored
# it, so the positive form leads: flat areas of ONE colour.
# ROUND 3. "no scribble fill, no pencil texture" is REMOVED -- those two were
# what erased the tan floor (see the _ROOM note in shots.py). The positive form
# "every filled area is one flat block of colour" does the same job without
# telling the model to suppress fill.
#
# The anti-lettering negations stay and are strengthened: round 2 leaked
# gibberish text into a frame ("Sinn dozin" on a diagram), which the palette
# lock cannot remove and a human reviewer has to catch.
STYLE_TAIL = (
    ", every filled area is one flat block of colour, no shading, no gradients, "
    "clean minimal composition, no writing anywhere, no text, no letters, "
    "no words, no numbers, no labels, no captions, no watermark, no signature"
)

# --- The locked colour palette. Every finished frame is snapped to exactly these. ---
#     This is what guarantees frame 3 and frame 140 look like the same channel.
#     Add a colour only if you really need it — a small palette is the whole trick.
# WIDENED FROM SIX TO THIRTEEN, AND THE OLD SIX ARE UNCHANGED AND STILL FIRST.
#
# The six-colour version was specified for a white-wall/tan-floor room, and it
# does that job perfectly. The reference channel is not that room: it is
# full-bleed colour -- navy behind a sleeping figure, beige behind a sofa, flat
# green behind trees, teal machines. Measured against the actual reference
# frames, the six-colour lock does not preserve any of it:
#
#     navy   #2C4A7C  ->  near-black    the whole frame goes black
#     teal   #5A9696  ->  wood brown    the MRI machine becomes brown
#     green  #3C823C  ->  dark brown    the forest becomes brown
#     yellow #F0D65C  ->  tan           washes out
#
# So the lock was not protecting the look, it was demolishing it -- and every
# hour spent on prompt wording or a LoRA to produce colour the lock then removed
# was spent downstream of that. See START_HERE.md's standing lesson.
#
# ORDER MATTERS AND MUST NOT BE REARRANGED. finish.py indexes PALETTE[1] as the
# outline colour, and FLOOR_COLOR is PALETTE[2] by value. New colours are
# APPENDED for that reason.
#
# NO NEUTRAL MID-GREY, deliberately. An anti-aliased edge pixel around black
# linework sits near (150,150,150); adding a grey that close to it would snap
# those edges to grey and ring every stroke with a halo. The one grey here is
# blue-tinted hard enough to be a scene colour rather than an edge colour.
PALETTE = [
    (255, 255, 255),   # white — background wall, and the figure's own fill
    (26,  26,  26),    # near-black — every outline
    (216, 164, 86),    # tan — the floor band
    (182, 70,  45),    # rust red — the "one bold prop" colour
    (196, 140, 80),    # wood brown — furniture
    (120, 72,  40),    # dark brown — shadows / depth on wood
    # --- scene colours, added for full-bleed backgrounds and coloured objects ---
    (44,  74,  124),   # navy — night / interior backgrounds
    (74,  148, 148),   # teal — machines, upholstery, equipment
    (58,  128, 62),    # forest green — foliage, flat outdoor fields
    (140, 152, 74),    # olive — grass, ground planes that are not the tan floor
    (226, 202, 160),   # warm beige — lit interior walls
    (240, 214, 92),    # yellow — the second bold accent
    (112, 128, 148),   # slate blue-grey — insects, metal, cool props
]
PALETTE_LOCK = True     # snap every frame to PALETTE. Turn off to see raw FLUX output.
WHITE_SNAP   = 236      # anything brighter than this becomes pure white

# =============================================================================
#  THE FLOOR AND THE LINE WEIGHT ARE PYTHON'S JOB NOW, NOT FLUX'S
# =============================================================================
#  Three rounds of prompt work moved the character (the oval torso became a real
#  line) and the faces (dots became big eyes with pupils) -- but did NOT move
#  two things, across five different phrasings each:
#
#     the tan floor      "band" / "rectangle" / "lower fifth" / "fills the
#                        bottom" / "solid flat" all rendered as a thin line, an
#                        orange scribble, or nothing at all, inconsistently
#                        between frames.
#     the line weight    "thick" / "very thick heavy" / "even width" / "heavy
#                        outlines" / "made with a thick black marker pen" all
#                        still came out thin.
#
#  That is the signal to stop asking. PLAN.md already states the principle this
#  project runs on, and it was written about exactly this situation:
#
#     "Solid, because it is deterministic and done in Python, not by the model:
#      the colour palette lock, the caption lettering ... Do not touch the
#      palette or caption code. Those are the parts that are working BECAUSE
#      they are not the model's job."
#
#  The floor is a rectangle. The captions look right in all 160 frames precisely
#  because PIL draws them and FLUX never touches them. The floor gets the same
#  treatment, and line weight becomes a morphological dilation -- a standard,
#  deterministic image operation.
#
#  What this buys beyond quality: the floor becomes IDENTICAL in every frame
#  (the inconsistency was itself a drift source), it costs zero GPU, and it is
#  tunable without redrawing anything.
#
#  The floor is correspondingly REMOVED from the shot clause, so FLUX stops
#  drawing a competing horizon line underneath ours.
# =============================================================================

FLOOR_BAND_ON = True     # draw the tan floor in code instead of asking FLUX
FLOOR_TOP     = 0.82     # FALLBACK only, for frames with no detectable horizon.
                         # Normally the band is anchored to the figure's feet --
                         # a fixed height clipped them, because feet land at
                         # different heights on a wide shot than a medium one.
FLOOR_COLOR   = (216, 164, 86)   # must be a PALETTE colour, or the lock moves it
FLOOR_EDGE_PX = 5        # black rule along the top edge; 0 for none

# How far below the detected horizon to keep looking for the feet, as a
# fraction of frame height, and how much clearance to leave under them.
# The window is bounded so a caption or a low prop cannot drag the band down.
FLOOR_FOOT_WINDOW    = 0.06
FLOOR_FOOT_MARGIN_PX = 3

# Dilation grows dark pixels outward by this many pixels in each direction.
#
# TUNED BY LOOKING, at 1280x720, against the reference channel side by side:
#
#     px=0   1.19% ink   thin and sketchy -- what we had, clearly amateur
#     px=1   1.76% ink   barely better. This was the first guess and it was
#                        far too timid.
#     px=2   2.40% ink   noticeably bolder, everything still legible
#     px=3   3.03% ink   CLOSEST to the reference weight. Eyes and mouth still
#                        read. <- chosen
#     px=4   3.63% ink   too far: the whites of the eyes start closing up and
#                        a raised hand becomes a blob
#
# Boosting before vs after the palette lock is visually a wash (1.76 vs 1.85 at
# px=1), so it stays before the lock, where the line's own anti-aliased edge
# still exists.
#
# HONEST LIMIT: dilation fixes stroke WEIGHT, not stroke SMOOTHNESS. The
# reference channel's lines are thick AND clean; ours become thick and still
# slightly wobbly, because the wobble is in what FLUX drew. Closing that last
# gap is a generation problem, not a post-processing one.
#
# ^ THAT LAST PARAGRAPH IS WRONG, AND RESTROKE BELOW IS WHY.
LINE_BOOST_ON = True
LINE_BOOST_PX = 3
LINE_DARK_MAX = 110      # a pixel counts as "line" if its luminance is under this

# =============================================================================
#  RESTROKE  --  the smoothness half, which dilation could never do
# =============================================================================
#  The note above concludes that stroke SMOOTHNESS is "a generation problem, not
#  a post-processing one". That conclusion was premature. Wobble is
#  high-frequency noise on a contour; blurring a binary mask and re-thresholding
#  it is the textbook way to remove high-frequency noise from a shape. The
#  large-scale form survives the blur, the jitter averages out.
#
#  And thickness comes free from the same pass, because where you threshold the
#  blurred mask decides how far the boundary moves:
#
#      RESTROKE_THRESH < 0.5   expands  (thicker)
#      RESTROKE_THRESH = 0.5   area-preserving, smoothing only
#      RESTROKE_THRESH > 0.5   contracts (thinner)
#
#  So one operation replaces BOTH the dilation and five failed prompt phrasings.
#  It runs INSTEAD of boost_lines(), never after it -- see finish.apply_all().
#
#  ⚠️ NOT YET TUNED ON A REAL FRAME. These defaults were set against a 640x360
#  JPEG panel, which is the only render on this laptop, and JPEG ringing at that
#  size is the same order as the wobble being measured. Re-tune at 1280x720
#  before trusting them -- sigma scales with resolution, so a value tuned at
#  640 wide is roughly HALF what 1280 needs.
#  THICKNESS COMES FROM THE DILATION, NOT FROM THE THRESHOLD. The first version
#  used the threshold for both jobs and it erased thin strokes silently -- a 1px
#  line blurred at sigma=1.6 peaks near 0.25, under any usable threshold, so it
#  vanished completely. See the note in finish.restroke(). Now:
#
#      RESTROKE_WIDTH_PX   how much thicker, in pixels (same units as
#                          LINE_BOOST_PX, which it replaces)
#      RESTROKE_SIGMA      how much wobble is removed
#      RESTROKE_THRESH     stays at 0.5, i.e. the blur only moves the edge
#  ⚠️ DO NOT COPY LINE_BOOST_PX INTO RESTROKE_WIDTH_PX. That was the first
#  guess and it is visibly wrong: restroke dilates AND blurs, so it lands
#  heavier than the same number of pixels of plain dilation. Tuned by cropping
#  the face and looking at it, which is the only way this has ever worked:
#
#      w=1   eyes, pupils and brows all still separate and readable  <- chosen
#      w=2   brows starting to merge into the eyes, whites shrinking
#      w=3   eyes and brows fused into one black mass. Face unreadable.
#
#  This is the same cliff config's own px=4 note describes ("the whites of the
#  eyes start closing up"), just reached two steps earlier.
#
#  ⚠️ AND IT IS NOT VALIDATED. The only render on this laptop is a 640x360 JPEG
#  strip upscaled to 1280x720, whose strokes are already soft and wide. A real
#  1280x720 PNG has crisper, thinner strokes and will probably want a HIGHER
#  number. Re-run the sweep on real output before trusting these.
#
#  THE TENSION TO WATCH: restroke thickens every line, including the interior
#  lines that make a machine read as a machine. shots._PROP was just changed to
#  ALLOW interior detail; set this too high and this code destroys what that
#  prompt change buys. Judge them together, on an object frame, not on a figure.
RESTROKE_ON       = True
RESTROKE_WIDTH_PX = 1      # see the sweep above. NOT the same scale as LINE_BOOST_PX
RESTROKE_SIGMA    = 1.2    # blur radius in px. Higher = smoother, loses detail
RESTROKE_THRESH   = 0.5    # 0.5 = area-preserving. Lower thickens, but prefer
                           # RESTROKE_WIDTH_PX for that -- it cannot erase

# =============================================================================
#  THE BACKGROUND IS CODE NOW, FOR THE FLOOR BAND'S EXACT ARGUMENT
# =============================================================================
#  The reference channel's frames are full-bleed colour. Ours are white, because
#  _ROOM in shots.py asks for "pure white background wall".
#
#  Asking FLUX for a coloured wall instead brings back precisely the per-frame
#  inconsistency the floor was moved into code to kill: a navy that is a
#  different navy every frame, landing wherever the palette lock throws it.
#  A flat field is a rectangle. Rectangles are Python's job here.
#
#  finish.paint_background() fills ONLY the white region connected to the frame
#  edge, which is what leaves the character's own white fill -- head, eyes --
#  intact. That white-figure-on-colour-field contrast IS the reference look.
#
#  Set BACKGROUND_ON = False to go back to white walls in one edit.
BACKGROUND_ON    = True
BACKGROUND_COLOR = (226, 202, 160)   # warm beige. MUST be a PALETTE colour

# --- On-screen text (the "CLOSE YOUR EYES" lettering) ---
CAPTIONS_ON     = True          # draw the script line onto frames that call for it
CAPTION_FONT    = "PermanentMarker-Regular.ttf"
CAPTION_COLOR   = (26, 26, 26)
CAPTION_MAX_W   = 0.42          # caption box: 42% of frame width, top-left
CAPTION_MAX_H   = 0.26
CAPTION_X       = 0.06
CAPTION_Y       = 0.07

# Expressions change the face per scene -- the ONE thing kept from the palette
# experiment that was reverted. False pins every face to the neutral eyes-and-
# mouth clause, i.e. the behaviour before expressions existed.
EXPRESSIONS_ON = True

# --- Video geometry (16:9, YouTube) ---
FRAME_WIDTH  = 1280
FRAME_HEIGHT = 720
OUTPUT_FPS   = 30
FLUX_SEED    = 42     # base seed. Honest note: this does NOT give you character
                      # consistency on its own — see the note in Step 19.
FLUX_STEPS   = 4      # schnell is distilled for 4 steps; more does not help


SPOKEN_WPM = 150  # Chatterbox lands around here; used only for the length estimate printout


# =============================================================================
#  THE BEAT SHEET — single source of truth
# =============================================================================
# This dict is rendered into BOTH the writer's prompt (Cell 4) and the retention
# judge's prompt (Cell 8). That is the whole point: the judge holds the writer to
# the exact same spec, rather than to a second copy that can drift out of sync.
#
# Section 3 is deliberately NOT defined here — it is chosen at runtime in Cell 3.5
# based on what kind of evidence the fetched sources actually contain. The old
# version of this notebook hardcoded a "volunteers performed a slippery task,
# here is the percentage difference between groups" shape, which forced the model
# to manufacture an experiment for topics that never had one.
# =============================================================================

BEAT_SHEET = {
    1: {
        "name": "The Cold Hook",
        "words": (90, 110),
        "rhythm": "Rapid, staccato, somatic. Short sharp lines. No sentence over 14 words.",
        "beats": [
            "Direct Somatic Directive - one short sentence (3-6 words) telling the viewer to look at, touch, or notice a specific part of their own body right now.",
            "Rhetorical Diagnostic Question - one sentence asking why this universal, mundane, slightly odd thing happens there.",
            "The Common-Sense Explanation - one sentence giving the widely-accepted schoolbook answer, stated plainly and confidently, as if it settles the matter.",
            "The Absolute Negation ('The Snap') - one blunt sentence of 2-4 words stating that answer is wrong. It must be 2-4 words. Not five. Not a clause with a comma.",
            "The Escalating Mystery - one or two sentences revealing the specific, concrete counter-evidence that breaks the schoolbook answer. Must be drawn from the sources, not asserted.",
        ],
        "hard_rules": [
            "Beat 4 must be 2-4 words total.",
            "No sentence in this section may exceed 14 words.",
            "Do not name the topic in a textbook way ('Today we are looking at the human chin'). The viewer should not know they are watching an explainer yet.",
        ],
    },
    2: {
        "name": "Context & Setup",
        # 170-210 was sized for four beats. The Human Stake beat is a fifth, so the
        # range moves with it - otherwise judge_section() flags every script for
        # overrun against a budget that no longer matches the spec it renders.
        "words": (200, 245),
        "rhythm": "Moderate. Explanatory swells of 10-16 words alternating with short 5-8 word landings.",
        "beats": [
            "The Historical Anchor - one sentence placing the question in time: who first asked it, when it was first noticed, or how long it has gone unanswered. Use a named person or year ONLY if the sources give you one; otherwise anchor it in whatever framing the sources do support.",
            "The Status Quo - one or two sentences on how science classified this: useless, vestigial, accidental, passive, a side effect.",
            "The Crack - one or two sentences on the observation or anomaly that made the status quo untenable.",
            "The Human Stake - one or two sentences naming WHO this is actually happening to: a group the sources identify who live with this, work around it, or are affected by it without it being noticed. A real population, never a hypothetical person and never a bare statistic. If the sources support no such group, skip this beat entirely rather than inventing one.",
            "The Stakes Question - one open sentence that names what is actually at stake in the answer, and hands off into the evidence section.",
        ],
        "hard_rules": [
            "Do not invent a scientist's name, an institution, or a year. If the sources do not contain one, write the beat without one.",
            "The Human Stake beat names real people the sources support, or it does not appear. An invented sufferer is worse than no stake at all - it will be cut by the fact-checker and it breaks trust on a health topic.",
            "Do not label the stake ('and this matters because...'). Put the people in the sentence and let the weight land on its own.",
        ],
    },
    4: {
        "name": "The Re-Hook & Mechanism",
        "words": (200, 250),
        "rhythm": "Opens with a hard pivot, then builds. Mechanism explained in physical, tangible terms.",
        "beats": [
            "The Re-Hook - one short sentence (4-8 words) that reopens the mystery just as the viewer thinks it is solved. This is the single most important retention beat in the script; it sits at roughly the 4-minute mark where viewers leave.",
            "The Mechanism Breakdown - two to four sentences on how this physically works at a smaller scale: what tissue, what signal, what process, what timing. Concrete nouns and verbs only.",
            "The Core Metaphor - one or two sentences translating the mechanism into a familiar non-biological object or system. It must be an object the viewer has physically handled.",
            "The Micro-Impact - one short sentence on what this means for the viewer's own body, today.",
        ],
        "hard_rules": [
            "The metaphor must be concrete and handleable (a zipper, a wet sponge, a door wedge). Not abstract (a symphony, a dance, a conversation).",
            "Do not restate a statistic already given in Section 3.",
        ],
    },
    5: {
        "name": "The Dilemma",
        "words": (280, 330),
        "rhythm": "Slower, heavier, sweeping. Longer sentences with room to breathe. This is the only section allowed to be reflective.",
        "beats": [
            "The Trap Closing - one reflective sentence turning the finding on the viewer's own comfortable modern life.",
            "The Evolutionary Flashback - two to four vivid sentences putting the viewer in a specific ancestral scene where this mattered. Show one concrete situation, not a montage of humanity.",
            "The Trade-off - one or two sentences on the physiological cost: why the body does not simply keep this switched on, or what it gave up to have this.",
            "The Uncomfortable Truth - one or two sentences on the thing about this that does not resolve cleanly.",
        ],
        "hard_rules": [
            "This is the section most prone to over-writing. Show, do not label. You may not tell the viewer how to feel about a fact - no 'unsettling', 'haunting', 'strangely beautiful', 'profound', 'remarkable'. Describe the thing precisely enough that the feeling arrives on its own.",
            "Maximum ONE adjective per noun. No stacked adjectives ('a quiet, ancient, forgotten reflex').",
            "The ancestral flashback must be one specific scene with one specific body doing one specific thing - not 'our ancestors' as a collective.",
        ],
    },
    6: {
        "name": "Impactful Outro",
        "words": (150, 185),
        "rhythm": "Meditative, slow, intimate. Then one hard stop.",
        "beats": [
            "The Somatic Return - one sentence looping back to the exact body part and action from the Cold Hook, using some of the same words so the loop is audible.",
            "The Re-framing - one or two sentences contrasting the sterile modern setting the viewer is sitting in with the machinery still running inside them.",
            "The Realization - one or two slow sentences on what this quirk actually is, stated plainly.",
            "The Closing Question ('The Snap') - one final open-ended question of 3-8 words. It must be genuinely unanswerable and must invite the viewer to answer in the comments implicitly, without ever saying 'comment below'.",
        ],
        "hard_rules": [
            "No new factual claims in this section at all. Every sentence must be tagged [0].",
            "The final line must be a real question the viewer could argue about - not a statement wearing a question mark ('Isn't the body amazing?').",
            "No call to action, no 'subscribe', no 'thanks for watching'.",
        ],
    },
}




# the second, which is exactly the "532 skulls, 32 landmarks, 46 distances" pacing
# problem. The beat sheet caused it. Adding a retention judge on top would just fight
# the prompt forever. So: pick the section shape that matches the evidence.
# =============================================================================

SECTION_3_TEMPLATES = {
    "controlled_experiment": {
        "name": "The Experiment",
        "words": (250, 300),
        "rhythm": "Alternating: a short sharp finding, then a longer vivid image, then a short consequence. Never three data sentences in a row.",
        "beats": [
            "The Setup - one sentence naming who ran the study and what they wanted to find out. Include institution/year ONLY if it is in named_entities.",
            "The Task - one or two sentences on the specific physical thing subjects were asked to do. Make it visualisable.",
            "The Result - one sharp sentence with the single strongest number from the allowed list.",
            "The Translation - one or two sentences making that number mean something in a scale a person can feel.",
            "The Caveat - one sentence on the limit of the finding, if the sources hedge it.",
        ],
        "hard_rules": [
            "Maximum TWO sentences carrying numbers in this entire section.",
            "Never place two number-carrying sentences next to each other.",
        ],
    },
    "observational_study": {
        "name": "The Measurement",
        "words": (250, 300),
        "rhythm": "Alternating: one measured fact, then an image that makes it physical, then a consequence.",
        "beats": [
            "The Question - one sentence on what researchers set out to measure and why nobody had.",
            "The Method - one or two sentences on what they actually looked at, in concrete visual terms (what was on the table in front of them).",
            "The Pattern - one sharp sentence with the strongest single number or comparison from the allowed list.",
            "The Translation - one or two sentences turning that pattern into something the viewer can picture at human scale.",
            "The Limit - one sentence on what this kind of measurement can and cannot prove.",
        ],
        "hard_rules": [
            "Maximum TWO sentences carrying numbers in this entire section.",
            "Do NOT list methodology counts (sample sizes, number of measurements taken, number of landmarks). Those are how the work was done, not what was found. A viewer does not care that 46 distances were measured; they care what the measurement showed.",
        ],
    },
    "comparative_anatomy": {
        "name": "The Comparison",
        "words": (250, 300),
        "rhythm": "Two things held side by side. Short comparison, long image, short consequence.",
        "beats": [
            "The Two Things - one or two sentences setting up the comparison: us versus a relative, now versus then, this population versus that one.",
            "The Difference - one sharp sentence naming the single most striking difference, with a number only if one is in the allowed list.",
            "The Image - one or two sentences putting the viewer physically in front of the two specimens, seeing the difference.",
            "The Inference - one or two sentences on what that difference implies about how we got here.",
            "The Wrinkle - one sentence on the part of the comparison that does not fit the neat story.",
        ],
        "hard_rules": [
            "Maximum ONE sentence carrying numbers in this entire section.",
            "Do NOT recite methodology (how many specimens, how many measurements, what software). Describe what was seen, not what was counted.",
        ],
    },
    "clinical_case": {
        "name": "The Patient",
        "words": (250, 300),
        "rhythm": "Narrative. One person, one problem, one revealing detail.",
        "beats": [
            "The Case - one or two sentences on the specific patient or condition, in scene, not in summary.",
            "The Anomaly - one sharp sentence on the thing about them that should not have been possible.",
            "The Inference - one or two sentences on what their case proved about how this works in everyone.",
            "The Generalisation - one or two sentences moving from the one case out to the viewer's own body.",
            "The Limit - one sentence on what a single case can and cannot establish, if the sources hedge it.",
        ],
        "hard_rules": [
            "Maximum ONE sentence carrying numbers in this entire section.",
            "Do not sensationalise a patient. Describe clinically; the strangeness carries itself.",
        ],
    },
    "competing_hypotheses": {
        "name": "The Argument",
        "words": (250, 300),
        "rhythm": "Two positions, fairly stated, then the collision. Short claim, long support, short doubt.",
        "beats": [
            "Hypothesis One - one or two sentences stating the leading explanation and its single best piece of support.",
            "The Problem With It - one sharp sentence on what it fails to account for.",
            "Hypothesis Two - one or two sentences on the rival explanation and what it gets right that the first does not.",
            "The Problem With That One Too - one sentence on its weakness.",
            "The Honest Position - one or two sentences stating plainly that this is unresolved, and what would settle it.",
        ],
        "hard_rules": [
            "Maximum ONE sentence carrying numbers in this entire section.",
            "Both hypotheses must be stated in their strongest form. Do not set up a straw man to knock down.",
            "You must not resolve the debate. The unresolvedness IS the content here, and it is what makes the outro land.",
        ],
    },
    "mechanism_only": {
        "name": "The Machinery",
        "words": (250, 300),
        "rhythm": "A process unfolding in order. Each step short, each consequence vivid.",
        "beats": [
            "The Trigger - one sentence on what starts the process, in physical terms.",
            "The Chain - two or three sentences walking the process forward one step at a time. Concrete nouns: tissue, nerve, muscle, chemical, timing.",
            "The Speed or Scale - one sharp sentence anchoring how fast, how small, or how often, using a number only if one is in the allowed list.",
            "The Visible Result - one or two sentences connecting the invisible chain to the thing the viewer can actually see or feel on their own body.",
            "The Odd Detail - one sentence on the part of the machinery that seems badly designed.",
        ],
        "hard_rules": [
            "Maximum ONE sentence carrying numbers in this entire section.",
            "Every step must be something happening to a physical thing. No abstractions like 'signals are processed'.",
        ],
    },
}


# =============================================================================
#  BANNED LEXICON — deterministic linter input (Cell 5), no API call needed
# =============================================================================
BANNED_PHRASES = [
    # AI-essay tells
    "delve", "tapestry", "testament to", "in a world where", "little did",
    "unlock the secret", "the truth is stranger", "buckle up", "let that sink in",
    "it turns out that", "here's the kicker", "plot twist", "mind-blowing",
    "nothing short of", "a fascinating glimpse", "sheds light on", "paves the way",
    "at the end of the day", "when it comes to", "needless to say",
    # emotion-labelling (the Section 5 failure mode)
    "unsettling", "strangely beautiful", "hauntingly", "profoundly",
    "remarkably", "astonishingly", "incredibly", "utterly", "truly amazing",
    "beautifully", "eerily", "oddly beautiful", "deeply moving",
    # filler
    "basically", "essentially", "literally", "actually just", "simply put",
    "in other words", "to put it simply", "believe it or not",
    # YouTube-slop CTA
    "subscribe", "smash that", "comment below", "let me know in the comments",
    "thanks for watching", "don't forget to",
]

HEDGE_WORDS = ["perhaps", "maybe", "possibly", "arguably", "somewhat", "rather", "quite", "fairly"]

# =============================================================================
#  AI-TELL LEXICON — the strings that survived BANNED_PHRASES above
# =============================================================================
#  These are not clichés, which is exactly why the existing list misses them.
#  They are the connective tissue a language model reaches for when it is about
#  to reveal something: the phrase that sets up an em-dash, the abstract-poetic
#  closer, the false-profundity hinge. Measured on a finished chin script that
#  had already passed both gates.
#
#  The SHAPE tells — em-dash density, tricolons, "not X but Y", anaphora — are
#  not strings and cannot live here. They are quotas, and they live in
#  pipeline/humanize.py with the code that counts them.
# =============================================================================
AI_TELL_PHRASES = [
    # set-ups for an em-dash reveal
    "something radical", "something strange", "something stranger",
    "something nobody expected", "something remarkable", "here's the thing",
    "here's the part", "here's what", "the real question is", "the real story",
    "what if i told you", "that's not the whole story", "but that's not all",
    # abstract-poetic closers
    "waiting for someone to", "the tightrope", "in ways nobody",
    "for the rest of your life", "over an unimaginable", "never agreed to",
    # false-profundity hinges
    "and yet", "which is to say", "make no mistake", "as it happens",
    "the answer, it turns out",
]



# =============================================================================
#  TOPIC BANK
# =============================================================================
# =============================================================================
#  THE OPTIONAL "stake" KEY  --  who the topic is happening TO
# =============================================================================
#  Every title in this bank is a mechanism question about the listener's own
#  body: "why do WE...", "why can't YOU...". That is what earns the click. It is
#  not what earns a share, and it is the difference between this channel and the
#  reference channel, whose titles are about PEOPLE ("They Do Everything. Nobody
#  Says Thank You.").
#
#  So BEAT_SHEET[2] now carries a Human Stake beat. That beat may only name a
#  population the SOURCES support -- it refuses to invent a sufferer, because on
#  a health topic a fabricated patient is both a factcheck failure and a trust
#  failure.
#
#  Which creates the trap this key exists to close:
#
#      sources.run() fetches ONLY the urls listed here. Every original url in
#      this bank is mechanism-focused. So a topic with no stake-bearing source
#      gives the writer nothing to build the beat from, and the beat correctly
#      skips itself -- silently, on every script, forever.
#
#  A topic therefore needs BOTH:
#      "stake"    one line naming the real population, for the reader of this file
#      a source   in the list below that actually covers those people
#
#  Add the source or do not add the stake. A "stake" line with no matching source
#  is worse than no stake line at all: it reads as done and produces nothing.
#
#  GOOD SOURCE DOMAINS for this channel, roughly in order of how well they
#  extract and how well they carry a human stake:
#      sapiens.org         anthropology, Wenner-Gren Foundation. Excellent on
#                          evolution topics and unusually people-centred, which
#                          is exactly what the stake beat needs.
#      americananthro.org  American Anthropological Association. Use OPEN
#                          ANTHROPOLOGY (their digital-only public journal) and
#                          Anthropology News - both free. NOT American
#                          Anthropologist on Wiley, which is paywalled.
#      therai.org.uk       Royal Anthropological Institute - the oldest such
#                          body, covers biological anthropology. Its journal
#                          (JRAI) lives on Wiley and is paywalled: use the RAI's
#                          own public pages and Anthropology Today features, not
#                          rai.onlinelibrary.wiley.com.
#      pmc.ncbi.nlm.nih.gov / ncbi.nlm.nih.gov/books   full text, free
#      Cleveland Clinic, Johns Hopkins, MedlinePlus    patient-facing, plain
#      nhs.uk              patient-facing, plain English, extracts cleanly
#      Smithsonian, National Geographic, Scientific American, The Conversation
#      bbc.com/future, knowablemagazine.org, quantamagazine.org, wellcomecollection.org
#                          long-form science writing, free. Wellcome in particular
#                          writes health as lived experience, which is the stake beat.
#      University press offices (Duke, Manchester, CU Boulder)
#
#  RARE DISEASE - most of this bank's stakes are rare conditions, so these carry
#  more weight here than they would on a general science channel:
#      rarediseases.info.nih.gov (GARD)   NIH's rare disease centre
#      medlineplus.gov/genetics           NIH, plain English, per-condition
#      rarediseases.org (NORD)            patient-advocacy, authoritative, free
#      orpha.net (Orphanet)               European reference, prevalence figures
#  Prefer these over a news article when the claim is about the condition itself
#  - prevalence, inheritance, what patients actually experience.
#
#  USE WITH CARE, already in the bank:
#      forbes.com          contributor platform, quality varies by author
#      acsh.org            industry-funded advocacy; fine on chemistry, treat any
#                          health-risk framing as a position, not a finding
#      iflscience.com, sciencenotes.org   popularisers, fine for orientation,
#                          weak as the only support for a claim
#      en.wikipedia.org    orientation and terminology only, never the sole source
#
#  AVOID: ScienceDirect, Wiley, JSTOR and other paywalled abstract pages, and
#  PDFs. Extraction is thin and looks_like_junk() in sources.py will reject them,
#  so the source silently does not exist as far as the writer is concerned. This
#  is why the free arm of a good publisher beats the flagship journal here.
#
#  Topics without a "stake" key still work. They publish as pure curiosity pieces.
#  Prefer the ones that have it -- see run_script.py --list.
#
# =============================================================================
#  ORDER OF OPERATIONS: CURIOSITY LEADS. THE STAKE DEEPENS.
# =============================================================================
#  This is a WEIRD BIOLOGY channel. The strangeness is the brand and the title
#  is where it has to live. The stake is not a replacement for that - it is what
#  the video turns out to be about once the viewer is already in.
#
#      title  -> the plain curiosity question. "Why Fingers Prune in Water".
#      stake  -> arrives around Section 2, once curiosity has bought the time.
#
#  KEEP THE ESTABLISHED TITLE FORMULA. Every title in this bank is either
#  "Why <ordinary thing>?" or "The <Ordinary Thing>". That consistency is the
#  channel's shelf identity - do not rewrite them into stake-forward headlines
#  ("The Bone That Decides If a Baby Can Breathe"). Those read as medical
#  documentary, buy a smaller and different audience, and cost the channel the
#  weird-biology brand.
#
#  These are INTERNAL working titles anyway: the published YouTube titles are
#  generated per video by packaging.py against TITLE_FORMULA, which produces 8
#  candidates. The stake belongs in those candidates, not here.
#
# =============================================================================
#  "cluster" -- the anti-template axis, one level up from shots.py
# =============================================================================
#  shots.py already encodes the lesson: repeated LINEWORK reads as a style,
#  repeated COMPOSITION reads as a template. The same is true of subject matter.
#  Repeated NICHE reads as an identity; repeated ANGLE reads as a formula.
#
#  So topics carry a cluster. Weird biology is cluster one and gets built out
#  first, deliberately, because a new channel needs a recognisable identity
#  before it can afford range. Later clusters (psychology, and others) rotate in
#  so the channel widens without ever reading as templated.
#
#  Keep the SHAPE constant across clusters - ordinary thing, broken assumption,
#  real people - and vary the DOMAIN. That is what makes the widening feel like
#  the same channel growing rather than a different channel wearing its skin.
#
# =============================================================================
#  WHO THIS IS FOR  --  and why "audience" exists
# =============================================================================
#  TARGET: 25-44, both genders. Not 18-24.
#
#  Age. The largest YouTube bracket is 25-34, and education content earns
#  roughly $4-6.50 RPM against $10-25 CPM, versus $2-8 CPM for entertainment.
#  The money and the attention are in the same place. 13-17 is close to
#  worthless to monetise and 18-24 is thin, so chasing either with teen-coded
#  language or meme framing costs revenue twice: lower RPM per view AND a
#  younger skew that suppresses it further. Write for an adult.
#
#  Gender - and this is the part worth understanding. There is a documented gap
#  in who engages with science content on YouTube ("Are Women a Missing Audience
#  for Science on YouTube?", Frontiers in Communication, 2021). The gap is not
#  in who uses YouTube; it is in who engages with science framing specifically.
#
#  Mechanism-only content - "here is how the reflex works" - is what selects for
#  the narrow, male-skewed science audience. The Human Stake beat is therefore
#  not only a retention device. It is the single biggest lever this channel has
#  on WHO SHOWS UP, because a video about people has a far wider door than a
#  video about a mechanism. The stake work and the audience work are the same
#  work.
#
#  So "audience" tags the rough skew of each topic (mixed / mixed-f / mixed-m)
#  and exists for one purpose: do not publish four male-leaning topics in a row.
#  Rotate it the way shots.py rotates composition. The bank is currently 4
#  female-leaning, 4 male-leaning, 7 neutral - keep it near that.
# =============================================================================

TOPICS = {
    "chin": {
        "title": "Why Do We Have a Chin?",
        # We are the only animal with a chin; even Neanderthals lacked one. The
        # stake is what happens when the mandible does not develop: in Pierre
        # Robin sequence the small jaw lets the tongue fall back and block the
        # airway, so the baby cannot breathe or feed. 1 in 2,000-30,000 births.
        "stake": "Babies born with micrognathia, whose undersized jaw blocks their own airway",
        "audience": "mixed-f",
        "sources": [
            {"title": "Why Do Humans Have Chins? (evolutionary spandrel research)",
             "url": "https://www.smithsonianmag.com/smart-news/why-do-humans-have-chins-they-might-be-an-evolutionary-accident-new-research-suggests-180988205/"},
            {"title": "Why Do Humans Have Chins? (Smithsonian, mechanical-stress hypothesis)",
             "url": "https://www.smithsonianmag.com/science-nature/why-do-humans-have-chins-15140492/"},
        ],
    },
    "photic_sneeze": {
        "title": "The Photic Sneeze Reflex",
        # 18-35% of people have this and most do not know it has a name. The
        # military studied it because a combat pilot who sneezes coming out of
        # cloud is blind at the worst possible moment. Same for a driver leaving
        # a tunnel. There is no treatment - only avoiding the light transition.
        "stake": "Pilots and drivers, for whom a sneeze at a light transition is a real hazard",
        "audience": "mixed-m",
        "sources": [
            {"title": "When the Sun Prickles Your Nose: EEG study of photic sneezing (PLOS ONE, via PMC)",
             "url": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2821404/"},
            {"title": "Photic sneeze reflex: definition, cause, symptoms",
             "url": "https://www.medicalnewstoday.com/articles/photic-sneeze-reflex"},
            {"title": "ACHOO Syndrome, incl. combat-pilot risk research (NCBI Medical Genetics Summaries)",
             "url": "https://www.ncbi.nlm.nih.gov/books/NBK109193/"},
            {"title": "Photic Sneeze Reflex - ACHOO Syndrome (Disabled World)",
             "url": "https://www.disabled-world.com/health/respiratory/achoo.php"},
        ],
    },
    "finger_pruning": {
        "title": "Why Fingers Prune in Water",
        # The fact is genuinely inverted: pruning is not skin soaking up water,
        # it is active nerve-driven vasoconstriction. Fingers that DON'T wrinkle
        # mean the nerve signal is not arriving - which is why doctors have used
        # it as a bedside test of nerve function.
        #
        # NARROWED after Phase-1 source verification: the cystic-fibrosis half of
        # the original stake (fibrosis patients wrinkle abnormally fast) had no
        # extractable source once the PMC/Wikipedia links both failed
        # looks_like_junk(). Rather than keep an unsupported claim in the config,
        # the stake now covers only what the working sources actually carry -
        # nerve damage, not CF. Re-add CF if a real source is found.
        "stake": "People with nerve damage, whose fingers stop wrinkling because the signal never arrives",
        "audience": "mixed",
        "sources": [
            {"title": "Why Do Our Fingers and Toes Wrinkle During a Bath? (Scientific American)",
             "url": "https://www.scientificamerican.com/article/why-do-our-fingers-and-toes-wrinkle-during-a-bath/"},
            {"title": "Debate: did wrinkled fingers evolve for better grip? (National Geographic)",
             "url": "https://www.nationalgeographic.com/science/article/debate-did-wrinkled-fingers-evolve-for-better-grip"},
        ],
    },
    "emotional_tears": {
        "title": "Why We Cry Emotional Tears",
        # Familial dysautonomia (Riley-Day) and Triple A syndrome both produce
        # alacrima - children who cry with no tears at all. The clinical sign is
        # literally recorded as "crying without tears". It is not cosmetic: tears
        # protect the cornea, and 27% of FD patients lose vision to the damage.
        #
        # WRITE THIS ONE WITH CARE. These are children with a serious rare
        # disease, not a curiosity. Report what the literature says, keep them as
        # people rather than an example, and never imply that a viewer whose eyes
        # are dry has this. Dry eye is common; familial dysautonomia is not.
        "stake": "Children with familial dysautonomia, who cry without tears and can lose their sight to it",
        "audience": "mixed-f",
        "sources": [
            {"title": "All About Emotional Tears (American Academy of Ophthalmology)",
             "url": "https://www.aao.org/eye-health/tips-prevention/all-about-emotional-tears"},
            {"title": "8 benefits of crying: why do we cry, and when to seek support",
             "url": "https://www.medicalnewstoday.com/articles/319631"},
            {"title": "Familial dysautonomia: a disease with hidden tears (PMC)",
             "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC5468537/"},
            {"title": "Familial dysautonomia (MedlinePlus Genetics, NIH)",
             "url": "https://medlineplus.gov/genetics/condition/familial-dysautonomia/"},
        ],
    },
    "appendix": {
        "title": "The Mystery of the Appendix",
        # Parker and Bollinger at Duke, 2007: the appendix is a bacterial safe
        # house. When cholera or dysentery strips the gut, the appendix
        # repopulates it. Which means the organ Western medicine calls useless
        # is most useful exactly where diarrhoeal disease still kills people.
        "stake": "People in places where cholera and dysentery still strip the gut - where the appendix matters most",
        "audience": "mixed",
        "sources": [
            {"title": "Appendix Isn't Useless at All: It's a Safe House for Bacteria (Duke Health)",
             "url": "https://corporate.dukehealth.org/news/appendix-isnt-useless-all-its-safe-house-bacteria"},
            {"title": "The appendix plays a hidden role in gut health (NPR)",
             "url": "https://www.npr.org/sections/health-shots/2024/02/02/1228474984/appendix-function-appendicitis-gut-health"},
            {"title": "Appendix Isn't Useless At All: It's A Safe House For Good Bacteria (ScienceDaily)",
             "url": "https://www.sciencedaily.com/releases/2007/10/071008102334.htm"},
            {"title": "Appendix not useless after all: scientists (CBC News)",
             "url": "https://www.cbc.ca/news/science/appendix-not-useless-after-all-scientists-1.648482"},
        ],
    },
    "crowded_teeth": {
        "title": "Why Our Teeth Are So Crowded",
        # Skulls from 10,000+ years ago show almost no crowding and almost no
        # impacted wisdom teeth. Jaws grow in response to chewing; soft processed
        # food means smaller jaws and the same number of teeth. Crowding runs at
        # ~66% in the US. Braces treat a condition our ancestors did not have.
        "stake": "The ~66% of people with crowded teeth - a modern soft-diet condition, not bad luck",
        "audience": "mixed-f",
        "sources": [
            {"title": "Humans used to have straighter teeth - what changed? (National Geographic)",
             "url": "https://www.nationalgeographic.com/health/article/crooked-teeth-human-evolution-jaw-size"},
            {"title": "How Come Ancient Skulls Often Have Straight Teeth? (IFLScience)",
             "url": "https://www.iflscience.com/how-come-ancient-skulls-often-have-straight-teeth-70078"},
            {"title": "Before Agriculture, Human Jaws Were a Perfect Fit for Human Teeth (Smithsonian)",
             "url": "https://www.smithsonianmag.com/smart-news/how-dawn-farming-changed-our-mouths-worst-180954167/"},
            {"title": "Dental Crowding and Impaction (CARTA, UC San Diego)",
             "url": "https://carta.anthropogeny.org/moca/topics/dental-crowding-and-impaction"},
            {"title": "Why do humans have crooked teeth when our ancestors did not? (The Conversation)",
             "url": "https://theconversation.com/why-do-humans-have-crooked-teeth-when-our-ancestors-did-not-181542"},
        ],
    },
    "contagious_yawn": {
        "title": "The Contagious Yawn",
        # The thermoregulatory account says a yawn cools the brain. The test case
        # is multiple sclerosis, a documented thermoregulatory disorder: MS
        # patients yawn atypically, and more than 1 in 3 report that yawning
        # gives temporary relief from their symptoms.
        #
        # NOTE: I checked the "reduced contagious yawning in dementia" angle and
        # the literature does NOT support it - empathy loss in frontotemporal
        # dementia is real, but the yawning link is not established, and
        # Alzheimer's patients show HEIGHTENED emotional contagion. Do not use it.
        "stake": "People with MS - over 1 in 3 say yawning briefly relieves their symptoms",
        "audience": "mixed",
        "sources": [
            {"title": "Brain size and neuron numbers drive differences in yawn duration (Nature, Communications Biology)",
             "url": "https://www.nature.com/articles/s42003-021-02019-y"},
            {"title": "What is MS? - on fatigue and thermoregulation (MS Society UK)",
             "url": "https://www.mssociety.org.uk/about-ms/what-is-ms"},
        ],
    },
    "motion_sickness": {
        "title": "Why We Get Motion Sickness",
        # The people selected hardest for physical resilience on Earth are the
        # ones it hits worst: 70-90% of crew get Space Adaptation Syndrome, and
        # veteran test pilots are not spared. NASA has a standing medical brief
        # on it. A good counter to the idea that motion sickness is weakness.
        "stake": "Astronauts - 70-90% are sick in their first days in orbit, veteran pilots included",
        "audience": "mixed-m",
        "sources": [
            {"title": "Motion sickness: it all started 550 million years ago (The Conversation)",
             "url": "https://theconversation.com/motion-sickness-it-all-started-550-million-years-ago-121789"},
            {"title": "Why Do Humans Get Dizzy? An Evolutionary Biologist Explains (Forbes)",
             "url": "https://www.forbes.com/sites/scotttravers/2026/06/07/why-do-humans-get-dizzy-an-evolutionary-biologist-explains/"},
            {"title": "With space travel comes motion sickness (University of Colorado Boulder)",
             "url": "https://www.colorado.edu/today/2024/02/29/space-travel-comes-motion-sickness-these-engineers-want-help"},
            {"title": "Space adaptation syndrome (Wikipedia, NASA-sourced)",
             "url": "https://en.wikipedia.org/wiki/Space_adaptation_syndrome"},
        ],
    },
    "goosebumps": {
        "title": "Why We Get Goosebumps",
        # The reflex fluffs up fur we no longer have - but the same circuit fires
        # for music. Sachs et al. found people who get chills from music have
        # measurably denser white-matter connections between auditory cortex and
        # the emotional/reward areas. Between a half and two-thirds of people get
        # frisson; the rest simply never do, with a perfectly healthy reward
        # system. A trait difference, not a disorder - write it that way.
        "stake": "The third to half of people who never get chills from music, and the brain wiring behind it",
        "audience": "mixed",
        "sources": [
            {"title": "Why Do We Get Goosebumps? The Answer Lies with the Stem Cells (NY Stem Cell Foundation)",
             "url": "https://www.nyscf.org/resources/why-do-we-get-goosebumps-the-answer-lies-with-the-stem-cells-that-grow-hair/"},
            {"title": "Why Do We Get Goosebumps? An Evolutionary Biologist Explains (Forbes)",
             "url": "https://www.forbes.com/sites/scotttravers/2026/04/07/why-do-we-get-goosebumps-an-evolutionary-biologist-explains/"},
            {"title": "Goosebumps: When the body remembers (Hektoen International, medical humanities journal)",
             "url": "https://hekint.org/2026/07/23/goosebumps-when-the-body-remembers/"},
        ],
    },
    "hypnic_jerk": {
        "title": "The Hypnic Jerk",
        # Present intermittently in ~70% of people, daily in ~10%. Intensified by
        # caffeine, stress, anxiety and sleep deprivation - which makes it
        # self-feeding: the jerk wakes you, the lost sleep makes the next one
        # more likely. Shift workers sit at the sharp end, drinking more caffeine
        # to stay awake and carrying higher rates of anxiety and depression.
        "stake": "Shift workers and insomniacs caught in the loop - less sleep means more jerks means less sleep",
        "audience": "mixed-m",
        "sources": [
            {"title": "Hypnic Jerks (Sleep Foundation)",
             "url": "https://www.sleepfoundation.org/parasomnias/hypnic-jerks"},
            {"title": "Hypnic jerk: why you twitch before falling asleep",
             "url": "https://www.medicalnewstoday.com/articles/324666"},
        ],
    },
    "brain_freeze": {
        "title": "Brain Freeze Explained",
        # Brain freeze fires the sphenopalatine ganglion - the same nerve cluster
        # implicated in migraine and cluster headache. Migraine sufferers are
        # more prone to cold-stimulus headache, and SPG blocks are a real
        # treatment being trialled for chronic migraine. So the ice-cream
        # headache is a safe, self-inflicted window onto a disabling condition.
        "stake": "Migraine and cluster-headache patients, whose pain runs through the same nerve",
        "audience": "mixed-f",
        "sources": [
            {"title": "What causes brain freeze? (Harvard Health)",
             "url": "https://www.health.harvard.edu/diseases-and-conditions/what-causes-brain-freeze"},
            {"title": "What causes brain freeze? - sphenopalatine ganglion research (ScienceDaily)",
             "url": "https://www.sciencedaily.com/releases/2017/07/170703201454.htm"},
            {"title": "Sphenopalatine ganglion blockade for migraine treatment (American Journal of Interventional Radiology)",
             "url": "https://americanjir.com/how-we-do-it-sphenopalatine-ganglion-blockade-for-migraine-treatment/"},
        ],
    },
    "smell_of_rain": {
        "title": "Why We Can Smell Rain Better Than Sharks Can Smell Blood",
        # The flip side of an extraordinary sensitivity is what its absence costs.
        # Depressive symptoms accompany olfactory loss in 36-76% of patients, and
        # Johns Hopkins found poor smell predicts later depression in older
        # adults. COVID made this a mass experience - and sudden loss is harder
        # to adapt to than being born without it.
        #
        # Depression is named here. State what the studies found, do not give
        # advice, and do not imply causation the sources do not support.
        "stake": "People who have lost their sense of smell - depressive symptoms follow in 36-76% of cases",
        "audience": "mixed",
        "sources": [
            {"title": "Petrichor and Geosmin - The Smell of Rain",
             "url": "https://sciencenotes.org/petrichor-and-geosmin-the-smell-of-rain/"},
            {"title": "Why Does Rain Smell So Good? - the geosmin mechanism (IFLScience)",
             "url": "https://www.iflscience.com/why-does-rain-smell-so-good-70560"},
            {"title": "Geosmin: Why We Like The Smell Of Air After A Storm (ACSH)",
             "url": "https://www.acsh.org/news/2018/07/28/geosmin-why-we-smell-air-after-storm-13240"},
            {"title": "A Poor Sense of Smell Is Linked with Depression in Seniors (Psychology Today, on the Johns Hopkins study)",
             "url": "https://www.psychologytoday.com/us/blog/the-sensory-revolution/202312/a-poor-sense-of-smell-is-linked-with-depression-in-seniors"},
            {"title": "Depression, Olfaction, and Quality of Life: A Mutual Relationship (PMC)",
             "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC5977071/"},
        ],
    },
    "self_tickle": {
        "title": "Why Can't You Tickle Yourself?",
        # Some people CAN - and the overlap is the point. Patients experiencing
        # hallucinations and loss of agency can self-tickle, because the same
        # prediction machinery that cancels the sensation is what tells you an
        # action was yours. The trick and the symptom are one mechanism.
        # HEALTH-TOPIC CARE: this is schizophrenia. The script describes what the
        # research found; it never diagnoses, and never implies a reader who can
        # self-tickle is ill. Schizotypy is a normal-population trait dimension.
        "stake": "People with schizophrenia and hallucinations of lost agency, who can self-tickle",
        "audience": "mixed",
        "sources": [
            {"title": "Why can't a person tickle himself? (Scientific American)",
             "url": "https://www.scientificamerican.com/article/why-cant-a-person-tickle/"},
            {"title": "Why Can't You Tickle Yourself? (Yale Scientific Magazine)",
             "url": "https://www.yalescientific.org/2015/05/why-cant-you-tickle-yourself/"},
            {"title": "People with schizophrenia-like traits can tickle themselves (British Psychological Society)",
             "url": "https://www.bps.org.uk/research-digest/people-schizophrenia-traits-can-tickle-themselves-whereas-most-people-cant"},
        ],
    },
    "fingerprints": {
        "title": "Why Do We Have Fingerprints?",
        # People with adermatoglyphia are born with none. The condition's own
        # clinical nickname is "immigration delay disease" - they are stopped at
        # borders because the regulations assume every human has ridges.
        "stake": "People born without fingerprints (adermatoglyphia), detained at borders",
        "audience": "mixed",
        "sources": [
            {"title": "Fingerprints and Friction (Smithsonian Magazine)",
             "url": "https://www.smithsonianmag.com/science-nature/fingerprints-and-friction-12860851/"},
            {"title": "Get a grip! Blistering new evidence on why we have fingerprints (University of Manchester)",
             "url": "https://www.manchester.ac.uk/about/news/get-a-grip-blistering-new-evidence-on-why-we-have-fingerprints/"},
            {"title": "Adermatoglyphia: The Genetic Disorder Of People Born Without Fingerprints (Smithsonian)",
             "url": "https://www.smithsonianmag.com/science-nature/adermatoglyphia-genetic-disorder-people-born-without-fingerprints-180949338/"},
            {"title": "Tip Off: Solving the Curious Case of the Missing Fingerprints (Scientific American)",
             "url": "https://www.scientificamerican.com/article/missing-fingerprints/"},
        ],
    },
    "naked_ape": {
        "title": "Why Are We the Only \"Naked Ape\"?",
        # We did not just lose fur, we traded it for 2.6 million sweat glands -
        # the cooling system that made endurance running possible. The stake is
        # the people whose system does not work: in anhidrosis the body cannot
        # cool itself at all, turning ordinary heat into a heatstroke risk.
        #
        # NOT hypertrichosis. That framing is circus-history and this channel
        # will not do it. Anhidrosis is the mechanism's own failure mode, which
        # makes it both the more respectful angle and the more relevant one.
        "stake": "People with anhidrosis, who cannot sweat and overheat in ordinary weather",
        "audience": "mixed-m",
        "sources": [
            {"title": "Why Humans Lost Their Hair and Became Naked and Sweaty (Discover Magazine)",
             "url": "https://www.discovermagazine.com/why-humans-lost-their-hair-and-became-naked-and-sweaty-307"},
            {"title": "Human hairlessness - overheating avoidance and eccrine sweating (Wikipedia)",
             "url": "https://en.wikipedia.org/wiki/Human_hairlessness"},
            {"title": "Anhidrosis (Lack of Sweat): Symptoms, Causes & Treatment (Cleveland Clinic)",
             "url": "https://my.clevelandclinic.org/health/diseases/15891-anhidrosis-lack-of-sweat"},
            {"title": "Anhidrosis - No Sweating (International Hyperhidrosis Society)",
             "url": "https://www.sweathelp.org/home/anhidrosis-no-sweating.html"},
        ],
    },
}


# question about a category. This formula is a question about a scene.
# =============================================================================
TITLE_FORMULA = {
    "pattern": (
        "[What/Why/How] + [did/do] + [a named, concrete subject — Ancient Humans, "
        "Early Humans, Your Body, a specific body part] + [a short, mundane, everyday "
        "verb phrase — at night, before mirrors, in the cold, every morning]?"
    ),
    "reference_examples": [
        "What Did Ancient Humans Do at Night?",
    ],
    "rules": [
        "The subject must be named and concrete — never an abstract collective noun on its own.",
        "The verb phrase must describe an ordinary, everyday situation — the mundane framing IS the "
        "curiosity gap. It is never the twist itself and never hints at the answer.",
        "Title Case, ends in a question mark, under 60 characters.",
        "Never answer the question in the title.",
    ],

    # =========================================================================
    #  THE STAKE-LED VARIANT  --  now the majority, but NOT all 8
    # =========================================================================
    #  The formula above is a curiosity formula and it is the channel's shape.
    #  The reference channel's biggest titles are not curiosity questions at all
    #  - they are about PEOPLE ("They Do Everything. Nobody Says Thank You.").
    #  The working assumption is that stake-led titles win CTR. Probably true.
    #
    #  So stake leads the mix (see TITLE_MIX below). But it does not take all 8,
    #  for three reasons that survive even if the assumption is right:
    #
    #    1. YOU ONLY PUBLISH ONE TITLE. Eight candidates of one shape is not
    #       eight options, it is one option written eight ways. The other shape
    #       is what makes the eight a choice.
    #
    #    2. NO A/B, NO LEARNING. There is no CTR data yet - zero videos, empty
    #       learnings.json. Going all-stake converts an untested hypothesis into
    #       a permanent decision, and removes the only mechanism that could ever
    #       correct it.
    #
    #    3. ALL-STAKE IS ITSELF A TEMPLATE. Every video titled "The People
    #       Who..." is the formula problem again, one level up - and this channel
    #       rotates topic clusters specifically to avoid reading as templated.
    #
    #  THE TRAP TO WATCH: a stake title beats a curiosity title on CTR almost by
    #  construction, because a person is more arresting than a mechanism. CTR
    #  alone will therefore always say "use more stake". Only raise the ratio if
    #  CTR goes up AND average-view-duration holds. If CTR rises while retention
    #  falls, the titles are baiting - and on a health channel that is a straight
    #  line to the inauthentic-content problem, not away from it.
    # =========================================================================
    "stake_pattern": (
        "[The/A] + [a real, named group the video is actually about] + [the ordinary "
        "thing that is quietly true of them]. A flat statement, not a question."
    ),
    "stake_examples": [
        "The People Born Without Fingerprints",
        "The Children Who Cry Without Tears",
    ],
    "stake_rules": [
        "The group must be the one the script actually names in its Human Stake beat, "
        "supported by the sources. If the script names no group, produce 8 curiosity "
        "titles instead and do not invent a population to fill the quota.",
        "Flat statement, Title Case, under 60 characters, no question mark.",
        "Never pity, never spectacle. 'The People Born Without Fingerprints' is a fact; "
        "'The Woman With No Fingerprints Trapped at the Border' is a tabloid.",
        "No condition name in the title if a plain-English description works - say "
        "'Born Without Fingerprints', not 'Adermatoglyphia'.",
    ],
}


# =============================================================================
#  TITLE_MIX  --  the one number to move when real data arrives
# =============================================================================
#  How the 8 candidates split by shape. Stake leads, because the working
#  assumption is that it wins CTR. Curiosity keeps a real share so the A/B
#  survives and the channel keeps its weird-biology shelf identity.
#
#  Change this ONLY against published numbers from pull_analytics.py, and only
#  on the two-part test in the comment above stake_pattern: CTR up AND
#  average-view-duration holding. Never on CTR alone.
#
#  If a topic's script names no real group, stake titles are impossible and the
#  whole 8 come back curiosity-led. That is correct - see stake_titles_possible
#  in packaging.py. Do not "fix" it by loosening the sourcing rules.
# =============================================================================

TITLE_MIX = {
    "stake": 5,       # flat statements about real people
    "curiosity": 3,   # of which at least 2 follow TITLE_FORMULA["pattern"] exactly
}
assert sum(TITLE_MIX.values()) == 8, "TITLE_MIX must total the 8 candidates packaging asks for"


# =============================================================================
#  THE 2AM TEST  --  the gate every title passes, whatever its shape
# =============================================================================
#  TITLE_MIX decides the SHAPE of a title. This decides whether it is any good.
#  Both shapes face it; a title that fails it is cut regardless of shape.
#
#  THE TEST: it is 2am. Someone is in bed, tired, phone in hand, and they have
#  already told themselves they are going to sleep. Does this title get clicked
#  ANYWAY?
#
#  That is a much harder bar than "is this interesting", and it is the right one,
#  because it is the actual competitive situation. Nobody is looking for your
#  video. They are tired and they are about to stop.
#
#  What passes at 2am:
#    - A question the viewer has ALREADY WONDERED, alone, and never looked up.
#      Recognition, not novelty. "Why do my fingers do that in the bath" beats
#      any fact they have never once thought about.
#    - Something true of THEIR OWN BODY, right now, tonight. It is in the room.
#    - Slightly unsettling. Not frightening - unresolved. The kind of thing that
#      gets worse if you leave it alone in the dark.
#    - Something never said out loud in daylight. Nobody discusses hypnic jerks
#      at lunch. Everyone has had one at 2am.
#    - Unresolvable by thinking. They cannot lie there and reason it out, so the
#      only way to close the loop is to watch.
#
#  What fails at 2am, however clever:
#    - Anything requiring prior interest in biology. At 2am nobody has a hobby.
#    - Trivia about someone else's body with no route back to their own.
#    - A title that ANSWERS itself. Loop closed, phone down, sleep.
#    - Cleverness that needs a second read. Tired people do not re-read.
#    - Any word doing work the viewer has to decode - clinical names especially.
#
#  Why this matches the human stake rather than competing with it: 2am is when
#  people feel alone with something. A title about people quietly living with
#  the thing you are lying there wondering about is the strongest version of
#  both ideas at once. That is the target.
# =============================================================================

# =============================================================================
#  TITLE RESEARCH SOURCES  --  and the firewall around them
# =============================================================================
#  THESE ARE NOT FACT SOURCES. Nothing here may ever appear in a topic's
#  "sources" list, be fetched by sources.py, or support a claim in a script.
#  They are for finding out HOW PEOPLE PHRASE THE QUESTION, nothing else.
#
#  The firewall matters because this is a health channel. A claim sourced from a
#  forum post is a factcheck failure and a YMYL trust failure at the same time.
#  Facts come from the domains listed above TOPICS. Phrasings can come from here.
#
#  Reddit is the best of them for one specific reason: r/NoStupidQuestions,
#  r/explainlikeimfive and r/Showerthoughts are literally archives of people
#  asking the thing they wondered alone and never looked up. That is the 2am
#  test as a dataset. Read how the question gets asked when nobody is performing
#  expertise - the upvoted phrasing is usually plainer than anything a writer
#  would produce, and plainer wins at 2am.
#
#  Use them to answer: what words do people actually use for this? What do they
#  assume? What is the version of this question that already has 4,000 upvotes?
#  Then write the title in that language and verify every fact elsewhere.
# =============================================================================

TITLE_RESEARCH_SOURCES = {
    "reddit": [
        "r/NoStupidQuestions", "r/explainlikeimfive", "r/Showerthoughts",
        "r/AskDocs",   # phrasing only - never a medical claim, it is strangers online
    ],
    "notes": [
        "PHRASING ONLY. Never a source for a fact, a number, a mechanism or a diagnosis.",
        "The upvoted wording is the research output - it is how a tired person says it.",
        "Condition subreddits show how people describe living with something, which is "
        "useful for tone in the stake beat. It is never evidence, and never a quote.",
        "If something found here looks like a fact, go and find it in a real source or "
        "drop it. It does not become true by being popular.",
    ],
}


TWO_AM_TEST = {
    "question": (
        "It is 2am. The viewer is tired, in bed, and has already decided to sleep. "
        "Do they click this ANYWAY?"
    ),
    "passes": [
        "A question they have already wondered on their own and never looked up - "
        "recognition beats novelty.",
        "True of their own body, tonight, in the room with them.",
        "Unsettling in an unresolved way, not a frightening one.",
        "Something nobody says out loud in daylight.",
        "Impossible to resolve by lying there thinking - watching is the only way to close it.",
    ],
    "fails": [
        "Needs an existing interest in biology. At 2am nobody has a hobby.",
        "Trivia about someone else's body with no route back to the viewer's own.",
        "Answers itself in the title - the loop closes and they sleep.",
        "Cleverness that needs a second read.",
        "Clinical or technical words the viewer has to decode.",
    ],
}


# =============================================================================
#  BEAT SHEET ASSEMBLY
# =============================================================================
#  Section 3 is chosen from the evidence, not hardcoded -- see the long comment
#  above SECTION_3_TEMPLATES for why that matters. In the notebook this was done
#  by assigning into the module-level BEAT_SHEET. Here it builds a copy, so the
#  template dict itself is never touched and two topics in one process cannot
#  contaminate each other.
# =============================================================================

def build_beat_sheet(evidence_type: str) -> tuple[dict, str]:
    """-> (beat sheet with Section 3 filled in, the evidence type actually used)."""
    if evidence_type not in SECTION_3_TEMPLATES:
        print(f"  (unrecognised evidence_type {evidence_type!r} - falling back to mechanism_only)")
        evidence_type = "mechanism_only"

    sheet = {k: dict(v) for k, v in BEAT_SHEET.items()}
    sheet[3] = dict(SECTION_3_TEMPLATES[evidence_type])
    return {k: sheet[k] for k in sorted(sheet)}, evidence_type


# =============================================================================
#  RETENTION MEMORY  --  persisted learnings from real published videos
# =============================================================================
#  The analytics loop that used to WRITE this file is not part of the pipeline.
#  Empty defaults are the normal case: the judge and packaging prompts simply run
#  without prior notes. Drop a hand-written learnings.json in the working folder
#  to feed them notes yourself.
#
#  Deliberately a flat, bounded, human-readable list of NOTES rather than weights
#  or scores, because anything feeding a judge's prompt should stay legible to a
#  human reviewing it instead of being a black-box adjustment.
# =============================================================================
LEARNINGS_PATH = "learnings.json"

_EMPTY_LEARNINGS = {
    "retention_notes": [],        # feeds the retention judge
    "title_notes": [],            # feeds the packaging prompt
    "thumbnail_notes": [],        # feeds packaging + the thumbnail cell
    "banned_phrase_additions": [],
    "history": [],                # [{date, video_id, source: "analytics"|"competitor"}]
}


def load_learnings() -> dict:
    f = paths.p(LEARNINGS_PATH)
    if f.exists():
        with open(f, "r", encoding="utf-8") as fh:
            loaded = json.load(fh)
        return {**{k: list(v) for k, v in _EMPTY_LEARNINGS.items()}, **loaded}
    return {k: list(v) for k, v in _EMPTY_LEARNINGS.items()}


def learnings_block(learnings: dict, keys: list[str], n: int = 10) -> str:
    """Render the most recent n notes per category as a prompt-ready bullet list.
    Empty on the first videos on this channel -- that is correct, not a bug."""
    lines = []
    for k in keys:
        lines.extend(f"- {note}" for note in learnings.get(k, [])[-n:])
    return "\n".join(lines) if lines else "(no data yet - first videos on this channel, ignore this section)"


def banned_phrases(learnings: dict) -> list[str]:
    """The deterministic linter's word list, plus anything the learnings file adds.
    Returns a new list; BANNED_PHRASES above is never mutated.

    AI_TELL_PHRASES is folded in here rather than kept for the humanizer gate
    alone, so the grounding linter kills them on round one -- before the retention
    fixer has built three good sentences around one."""
    return (list(BANNED_PHRASES) + list(AI_TELL_PHRASES)
            + list(learnings.get("banned_phrase_additions", [])))


# =============================================================================
#  STYLE EXPORT  --  closing a real drift hazard
# =============================================================================
#  CHARACTER_BIBLE and STYLE_PREFIX exist in two places: here, and in the
#  notebook's Step 3. That is not duplication for convenience, it is a live bug
#  waiting to happen:
#
#    - run_scenes.py bakes them into every scene's `visual_prompt`, and the
#      notebook's Step 23 draws from that field. So the PACKAGE copy paints all
#      150 frames.
#    - But Step 24's Redraw button rebuilds the prompt from the NOTEBOOK copy.
#
#  Edit one and not the other and a redrawn frame stops matching the other 149 --
#  silently, because both prompts look perfectly reasonable on their own.
#
#  The fix follows the pattern beat_sheet.json already set: the authoritative
#  values cross the machine boundary as DATA. The notebook reads style.json and
#  overrides its own constants when it is present. Data, not code, because
#  shipping the package to Colab means solving sys.path there, and every other
#  stage boundary in this pipeline is already a file.
# =============================================================================

def export_style() -> None:
    """Write the prompt constants the package owns, for the notebook to read.

    The shot catalog crosses the boundary for the same reason the two prompt
    constants do, and more urgently: a shot decides whether CHARACTER_BIBLE is in
    a frame's prompt at all, so a notebook redrawing without it puts a stick
    figure back onto a bar chart. Only the two fields the notebook needs to
    rebuild a prompt are exported -- weights and caps are laptop-side scheduling
    and have no meaning in Colab."""
    from . import look, shots      # local: keeps the config <- shots/look edge one-way

    with open(paths.p("style.json"), "w", encoding="utf-8") as f:
        json.dump({
            "_note": ("Written by run_script.py / run_scenes.py from "
                      "pipeline/config.py. The notebook reads this so a redrawn "
                      "frame uses the same prompt as the original render. "
                      "Edit pipeline/config.py, not this file."),
            "character_bible": CHARACTER_BIBLE,
            "style_prefix": STYLE_PREFIX,
            "style_tail": STYLE_TAIL,
            # Crosses for the same reason the bible does: a redraw that loses the
            # trigger draws the base character while its 159 neighbours carry the
            # LoRA's, and nothing in either prompt looks wrong on its own.
            "lora_trigger": LORA_TRIGGER,
            "shots": {name: {"character": spec["character"], "clause": spec["clause"]}
                      for name, spec in shots.SHOTS.items()},
            "fallback_subject": dict(_scene_fallbacks()),

            # The expression catalog crosses for the same reason the shots do:
            # Step 24's Redraw button rebuilds a prompt by hand, and a redraw
            # that loses the face comes back not matching the 159 around it.
            "expressions": {name: spec["clause"]
                            for name, spec in look.EXPRESSIONS.items()},
            "default_expression": look.DEFAULT_EXPRESSION,
            "expressions_on": EXPRESSIONS_ON,
        }, f, indent=2)


def _scene_fallbacks() -> dict:
    from . import scenes
    return scenes._FALLBACK_SUBJECT


def estimate_runtime(beat_sheet: dict) -> str:
    lo = sum(v["words"][0] for v in beat_sheet.values())
    hi = sum(v["words"][1] for v in beat_sheet.values())
    return (f"{lo}-{hi} words, ~{lo/SPOKEN_WPM:.1f}-{hi/SPOKEN_WPM:.1f} min "
            f"at {SPOKEN_WPM} wpm")
