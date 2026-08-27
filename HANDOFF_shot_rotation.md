# Handoff: shot rotation, session of 2026-08-24

Companion to `PLAN.md` (the whole-pipeline doc) and `channels/README.md`. This
file is scoped to one thing: the anti-template shot rotation built this session,
what it proved, and what is still open. Read `PLAN.md`'s "The visuals" section
first for how this fits the rest of the pipeline.

---

## Why this exists

The competitor channel (`@Zenn0009`) doesn't read as templated because it holds
**linework** constant and varies **composition** every frame — a story beat, a
bar chart, an icon grid, a near-empty typographic frame, never the same shot
type twice in a row. Our system held composition constant too: every one of 160
frames was one figure standing on a white wall above a tan floor band, because
`STYLE_PREFIX` baked the room into every prompt and `CHARACTER_BIBLE` was
concatenated unconditionally.

```
repeated LINEWORK    reads as a style   (right, keep)
repeated COMPOSITION reads as a template (was wrong, now fixed)
```

---

## What was built

**New file: `channels/pipeline/shots.py`**
Seven shot types (`character-moment`, `two-character`, `symbolic-metaphor`,
`object-focus`, `icon-grid`, `data-chart`, `typographic`), each with a `family`
(narrative/abstract), whether it carries `CHARACTER_BIBLE`, a fixed composition
`clause`, a director `ask`, a caption bias, a `weight`, and a `max_share` ceiling.

`assign()` picks a shot per scene deterministically (seeded, so a timeline
replans identically), enforces no-adjacent-repeat, breaks any run of 3+
narrative frames with an abstract one, and pins frame 0 to `character-moment`
(the opening needs to be a composition, since section 6 explicitly "returns" to
it). `validate()` checks the result and never lets assignment fail outright.
`enforce_caption_cap()` deterministically caps captioned frames at 35% —
biggest cut first (`avoid` → `neutral` → `prefer`), `typographic` frames
(caption is mandatory there) are never touched.

**Changed: `channels/pipeline/scenes.py`**
- `visual_prompt(action, shot)` now includes `CHARACTER_BIBLE` only when the
  shot has a character — this was the actual bug that put a stick figure on
  bar-chart frames before.
- `_valid()` rejects a character-free action that names a person.
- `shot_instruction()` builds the per-shot addition to the director prompt.

**Changed: `channels/pipeline/config.py`**
- `STYLE_PREFIX` is linework-only now. The floor/wall sentence moved into the
  shot clauses (`_ROOM` for shots with a room, `_VOID` for those without) —
  it used to contradict a chart shot's "no floor line."
- `export_style()` now ships the shot catalog into `style.json` for the
  notebook, not just the two prompt constants.

**Changed: `channels/stickman.ipynb`** (Step 3 / cell 17, Step 24 / cell 43)
- Step 3 loads the shot catalog from `style.json`.
- Step 24's Redraw button calls a `build_visual_prompt()` that mirrors the
  package instead of hardcoding `f"{STYLE_PREFIX}{CHARACTER_BIBLE}, {action}"`.
- The review widget shows each frame's shot name.
- Backup of the pre-patch notebook: ask for it if needed, it was written to
  the session scratchpad, not the repo.

**Changed: `channels/run_scenes.py`**
- `--shots`: assigns and prints the rotation only. No key, no API call, no
  cost. This is how to tune `shots.py` for free.
- `--strict`: refuse to finish if the validator complains.

**Changed: `channels/test_pipeline.py`** — went from 37 checks to 48. New
coverage: the shot catalog is well-formed, a 160-scene script rotates clean,
no adjacent repeats, narrative runs break at 3, every shot appears under its
cap, the rotation is seeded (reproducible), assignment never fails on odd
inputs (1 scene, one huge section, an unknown section), the caption cap holds
even in the worst case, the validator actually catches what it claims to, and
the person-word regex is whole-word (see bug below). The notebook drift test
now checks all 7 shots agree between package and notebook, not just one.

**Docs updated:** `PLAN.md` ("The visuals" section), `channels/README.md`
(`run_scenes.py` section + the file-contract table).

---

## The bug found and fixed (this matters, read it)

`_PERSON_WORDS` used **plain substring matching** to reject a character-free
shot's action if it named a person. `"he "` is a substring of `"the "`. Since
"the" is one of the most common words in English, almost every ordinary
sentence on a character-free shot (`object-focus`, `icon-grid`, `data-chart`,
`typographic`) was being rejected as a false person-mention.

**Measured on a real run** (before the fix, `cheap` preset): 24/80 scenes (30%)
fell back to a generic default subject instead of the model's real answer.

**Fix**: switched to a word-boundary regex (`\bhe\b`, `\bman\b`, etc.) in
`scenes.py`. Verified against the exact false-positive strings from that run —
all now pass — while genuine person-mentions still correctly reject.
Regression test added (`test_pipeline.py`, "person-word check is whole-word").

**After the fix**, re-run on the same 80-scene timeline with the actual
production scenes model (DeepSeek V4 Flash, not the `cheap` test preset):
**80/80 scenes planned, zero fallbacks, cost $0.0018.** The remaining ~20%
fallback rate seen with the `cheap` preset (Ling 3.0 Flash) was a *different*,
pre-existing thing — that model writes captions over the 5-word cap — and is
expected: `cheap` is documented as "for checking the pipeline runs, not the
writing," never a quality signal.

---

## What real generated frames showed (this session's visual QA)

You ran the updated pipeline through the notebook and shared 5 real frames.
Findings — **all three now fixed**, see below for what changed and why.

1. **Grounding — fixed.** `object-focus`'s clause said the object is
   "centred" but never said what it rests on — an ear frame came out floating
   unsupported above the floor band. `SHOTS["object-focus"]["clause"]` in
   `pipeline/shots.py` now adds "resting on a simple flat stand or shelf drawn
   with the same thick black outline"; `icon-grid`'s clause got the matching
   "each resting on its own small flat base." The `ask` for both now also asks
   the model to say what the object is "sitting on or held by."
2. **Icon-grid subject drift — fixed.** One frame showed eyeballs growing on
   stalks out of lab beakers — anatomically surreal, reads as an AI tell
   rather than a human illustrator's deliberate choice. `object-focus` and
   `icon-grid`'s `ask` now say "ordinary, recognizable, real-world" and
   explicitly rule out "anatomical or hybrid (no body parts fused onto
   objects, no object growing organic parts)."
3. **Caption casing — mitigated, root cause not fully provable without a real
   render.** `draw_caption()` always does `text.upper()` and it's the only
   render path (`finish_frame()`), so a frame that came out mixed-case did not
   go through it. The one route by which lettering can reach FLUX outside
   that single controlled path is the model's `"action"` field naming exact
   words to paint (e.g. `a sign that says "ONE SPECIES"`) — `visual_prompt()`
   puts `action` straight into the FLUX prompt, uncontrolled and never
   uppercased. Fixed two ways in `pipeline/scenes.py`: the director system
   prompt now explicitly forbids naming words/letters/signs in the action, and
   `_valid()` rejects any action containing a quote mark (straight or curly)
   as a hard backstop if the model ignores that. Deliberately narrow — no
   word-list ban on "label"/"sign"/etc., since those are ordinary legitimate
   words in a chart or object description and a word-list ban would repeat the
   exact substring-match bug fixed above (see `test_pipeline.py`, "an action
   that quotes literal text is rejected, ordinary words are not"). **If this
   recurs**, capture the redrawn frame's `scenes_planned.json` entry (the
   `action_clause` and `visual_prompt` fields) at the time it was generated —
   that's the one piece of evidence this session didn't have.
4. Two of five frames matched Zenn's quality closely (the held-brain frame
   especially — grounded, single prop, matches `CHARACTER_BIBLE` exactly).

`style.json` in `_local_workdir/` has been regenerated (`config.export_style()`)
so a notebook Step 3 load picks up the new clauses immediately — no need to
re-run `run_script.py` just to see them.

`test_pipeline.py` grew 3 more checks for these fixes (48 → 51): the quote
guard rejects literal quoted text but not ordinary words, and the
object-focus/icon-grid clause+ask changes are asserted present.

---

## Also found — fixed

`run_scenes.py --dry-run`'s printed model and cost estimate used
`config.MODEL_CHECKER` for pricing instead of `config.MODEL_SCENES` — so the
dry-run display didn't match what a real run actually uses or costs. Now prices
off `MODEL_SCENES`, which is what `scenes.py` actually calls; confirmed the
dry-run now prints `deepseek/deepseek-v4-flash` and an estimate (~$0.0017 on
the 80-scene fake timeline) in line with the $0.0018 measured on a real run
earlier in this doc. Regression test added (`test_pipeline.py`, "run_scenes:
--dry-run prices off MODEL_SCENES, not MODEL_CHECKER").

---

## State of `_local_workdir/`

`script_tagged.json` and `scenes_timeline.json` in there currently hold
**synthetic test data** (160 generated sentences / an 80-scene fake timeline
built via `make_test_timeline.py`), not a real script. You confirmed there was
no real content lost when this got overwritten mid-session, but flagging it
here so a new session doesn't mistake it for real material. Before a real
video: `python run_script.py --topic <topic>` regenerates it for real.

---

---

## Session 2: the Zenn palette experiment — TRIED AND REVERTED

An attempt to implement the *Zenn Channel Analysis & Anti-Template Strategy
Guide* wholesale. **Most of it was reverted the same session, after seeing real
frames.** Recorded here so nobody rebuilds it by accident.

### What was reverted, and why

A saturated **cyan `#38B6FF` / yellow `#FFDE59` palette**, a **framing axis**
(7 camera distances chosen by line energy), and **per-scene caption placement**.

The verdict on a real frame was blunt and correct: *that is not our look*. The
warm white-wall/tan-floor room **is** the channel's identity, and the rebrand
threw it away to chase a different channel's. The crude "MS-Paint style"
`STYLE_PREFIX` rewrite went with it — the previous linework was already right.

Reverted to the previous state:

- `config.PALETTE` — back to the six warm colours (white, near-black, tan,
  rust, wood brown, dark brown)
- `config.STYLE_PREFIX` — back to the original wording, verbatim
- `config.CAPTION_X/Y` — back to the fixed top-left corner
- `shots.py` clauses — `_ROOM` / `_VOID` and the per-shot distance words
  ("medium shot", "wide shot") are back in the clauses where they belong
- `stickman.ipynb` — restored from a pre-change backup, so the notebook's
  palette and style constants are untouched originals

`test_pipeline.py` has a check that **keeps it reverted**: it asserts the
palette is the warm six, that the shot clauses still say "white background
wall" and "tan", and that `STYLE_PREFIX` contains no `38B6FF`, `FFDE59`,
`MS-Paint` or `sky blue`.

### What was KEPT: the expression axis

One thing from that session was an improvement and survived: **the face**.

`CHARACTER_BIBLE` used to end "…two short curved black lines for eyes, one
small curved line for the mouth" — one frozen neutral expression across all 160
frames. The construction of the figure is still constant and still verbatim
(circle head, pure white fill, thin line limbs, no colour); only the
eyes-and-mouth clause varies now, from six: `deadpan`, `rage`, `shock`, `smug`,
`worried`, `delight`.

`pipeline/look.py` is now scoped to exactly this. It picks the face from a free
**lexical energy score** of the line — punctuation, sentence length, a small
word list — with no model call on the pipeline's highest-volume step. Seeded,
capped per expression, validated, and assigned before any call goes out, same
contract as the shot rotation. An expression only ever reaches a shot that HAS
a character; a face on a bar chart is the `CHARACTER_BIBLE` bug wearing a hat,
and `look.validate()` catches it.

`config.EXPRESSIONS_ON = False` pins every face to deadpan in one edit — worth
knowing, because expression is the one axis that genuinely raises character
variance, which is the drift risk PLAN.md has no fix for yet.

### Two real bugs found and fixed along the way

1. **The energy word list needed stemming.** `"scream"` was in the list and
   `"screamed"` was not, so *"Blood everywhere. He screamed!"* scored below an
   analytical line. Light suffix stripping (`-s`, `-ed`, `-ing`) rather than
   spelling out every inflection forever.
2. **Expression relaxation dropped the ceiling before the preference.** The
   energy band is a preference; `max_share` is a ceiling that stops one face
   becoming the new frozen default. The band is dropped first now.

Also calibrated: the short-line bonus moved 8→10 words, because *"Press two
fingers against the back of your arm"* (9 words) landed in the **low** band —
and section 1's direction is "tight and physical, close on the body part."

**Verified with a real run** (`run_scenes.py`, DeepSeek V4 Flash, 80 scenes):
**80/80 planned, 0 fallbacks, $0.0031.** `scenes_planned.json` in
`_local_workdir/` holds real prompts on the **restored** look, with expressions.

Tests: 51 → 57 checks. Also fixed this session: `run_scenes.py --dry-run` priced
its estimate off `MODEL_CHECKER` instead of `MODEL_SCENES`.

### From the Zenn doc, still not implemented

- **TTS pacing / pitch (Kokoro, Musa)** — not applicable. `NARRATION = "human"`;
  there is no TTS path in this pipeline, you record the voiceover yourself.
- **Topic hopping / story-first hooks / philosophical framing** — script and
  content strategy, already the writer model's territory via `TOPIC_BANK` and
  the hook tournament. Nothing to change in the visual layer.
- **The palette and framing ideas** — deliberately rejected, see above.

### The standing lesson

Both sessions now point the same way: **generate a small real batch through FLUX
before changing anything that touches the look.** The shot rotation survived
contact with real frames; the palette rebrand did not, and it cost a full
build-and-revert cycle to find that out.

## Next steps, in order

1. **Generate a fresh end-to-end test batch through the notebook** (Steps
   20–26) on the 80-scene fake timeline in `_local_workdir/scenes_planned.json`
   — the shot clauses and the QA-item fixes below are code-only so far, none
   of them have been checked against a real FLUX render yet. While reviewing:
   - confirm `object-focus`/`icon-grid` props now read as grounded (resting on
     a stand/base/shelf) instead of floating.
   - confirm no more anatomical/hybrid subjects (eyeballs-on-stalks and
     similar) on those two shots.
   - **specifically watch for the caption-casing bug recurring.** It's
     mitigated (system-prompt instruction + a quote-mark reject in
     `scenes.py`'s `_valid()`), not proven fixed — the actual mechanism was
     never confirmed against a real frame. If a mixed-case frame turns up
     again, save that scene's `action_clause` and `visual_prompt` from
     `scenes_planned.json` — that's the evidence this session lacked.
   - count the Step 24 redraw rate — that number is what PLAN.md says should
     decide whether per-scene palette (still deferred) is worth doing next.
2. **Run `python run_scenes.py --shots`** on a real (or synthetic) timeline
   any time you touch `shots.py` — it's free and shows the rotation instantly.
3. **Per-scene colour and varied caption placement remain deferred**, per the
   original plan — they raise prompt variance and therefore character drift,
   and there's no fix for drift yet beyond the LoRA idea in `PLAN.md`. Wait for
   a real redraw-rate measurement before touching `PALETTE`.
4. When ready for an actual video: `python run_script.py --topic <topic>`
   first, to replace the synthetic test data in `_local_workdir/` with real
   content, then proceed through the normal pipeline seam-by-seam per
   `PLAN.md`.

---

## Quick command reference

```powershell
# Free, no key needed — tune the rotation
python run_scenes.py --workdir _local_workdir --shots

# Cost estimate before spending anything (note: mispriced, see bug above)
python run_scenes.py --workdir _local_workdir --dry-run

# Real run, cheap test model (wiring check only, not quality)
python run_scenes.py --workdir _local_workdir --models cheap

# Real run, production model (DeepSeek V4 Flash for scenes)
python run_scenes.py --workdir _local_workdir --models claude

# Rebuild a fake timeline for visuals-only testing, from any script_tagged.json
python make_test_timeline.py --workdir _local_workdir --force

# Test suites
python test_pipeline.py     # 57 checks, no API calls
python test_notebook.py     # 13 checks, no API calls
```

All commands run from `channels/`, using `.venv\Scripts\python.exe` (or an
activated venv).
