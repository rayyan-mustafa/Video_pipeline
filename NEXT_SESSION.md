# NEXT_SESSION — handoff for a fresh chat

Written 2026-08-26, end of session. Supersedes nothing — read alongside
`STATUS.md`, which still has the architecture rationale and the pre-session
history. This file is what happened *since* STATUS.md and what to do next.

---

## The one-paragraph version

This session ran the actual step 11 test STATUS.md called for (base SDXL on
10 props), found the answer was invalid because of a prompt bug, fixed it,
found base SDXL still isn't on-style, added a free HuggingFace vector-art
LoRA, hit and fixed two library bugs to load it, and got a working end-to-end
render: **SDXL prop → PIL character (draw.py) → compositor merge**, all real,
all local files you can open. Along the way found and fixed a silent
cutout-on-coloured-background bug that would have corrupted the prop library,
and confirmed (0 of 8 seeds) that generating a whole scene with the character
in it still fails exactly the way STATUS.md said it would. Session ended on
an open strategic question: differentiate the visual style from Zenn, or
close the gap to it. Not resolved — see "Open decision" below.

---

## THE STEPS — updated

```text
  ✅ DONE   1-9.  (unchanged, see STATUS.md)
  ✅ DONE  10. colab-mcp reconnected. .mcp.json timeout raised 30000->600000
              (needs a Claude Code RESTART to take effect -- not yet done)
  ✅ DONE  11. base SDXL prop test run -- see "What was found" below.
              ANSWER: base SDXL alone -> NO. +vector LoRA -> YES, with caveats.
  ✅ DONE  11b. whole-scene-with-character test, 8 seeds. ANSWER: NO, 0/8.
              Confirms the FLUX-era "two-character renders one figure" defect
              reproduces on SDXL. The compositor's reason to exist still holds.
  ⏳ NOW   12. cutout() fixed for coloured backgrounds (was silently broken,
              see below). props.stage() gate now REJECTS a bad cutout instead
              of filing it. One real merged frame produced and reviewed.
  ⬜ NEXT  13. layout templates -- STILL the actual gap. Nothing decides
              x=0.68. This blocks going from 1 hand-tuned frame to 160.
  ⬜ NEXT  14. seat the character on the sofa properly (figure_seated +
              SEATED_ARMS["drape"] written this session, NOT yet re-rendered
              -- see "Unfinished, in-repo" below)
  ⬜ OPEN  15. style decision: match Zenn or differentiate? See below.
  ⬜ OPEN  16. SDXL backgrounds-as-base-layer (new idea, not started)
```

---

## What was found, in order (the actual evidence, not a summary)

### Round 1 — base SDXL, the STYLE prompt as originally written: FAILED, but the test was invalid
Ran the 10 props from `lora_object_prompts.md` exactly as specced. Got a
circuit-panel instead of lungs, a car instead of a mosquito, a dashboard
instead of a bar chart. Looked like "SDXL can't do this style" — **it wasn't
that.** Measured it: `STYLE` block alone was **70 of CLIP's 77-token budget**,
leaving ~7 tokens for the subject, so every colour clause got silently
truncated off the end. The two prompts that happened to fit (bulb,
microscope) were the only two with any colour in the output. Not a
coincidence — the mechanism.

**Lesson, in case it needs repeating for a future SDXL prompt:** put the
subject FIRST, keep the style block short, and never put "no X" phrases in
the positive prompt (CLIP has no negation — "no text" injects the concept
"text"; put negations in the actual negative prompt).

### Round 2 — same 10, prompt bug fixed, still no LoRA: right object, right colour, wrong style
Subject-first, style trimmed to ~14 tokens, negations moved to NEG. Every
object and colour landed correctly (lungs are pink, MRI is teal, bulb is
yellow). But: **backgrounds were flat coloured fields, not white** (pink,
teal, slate grey — never white), several were still shaded/realistic (skull,
microscope), and the chair rendered as a tiled pattern of ~15 chairs. Base
SDXL knows "flat vector" as a genre-adjacent look but not this exact style.

### Round 3 — added DoctorDiffusion/doctor-diffusion-s-controllable-vector-art-xl-lora: closed the style gap
Free, HuggingFace, exact SDXL-1.0 base match, trigger word `vector`. Hit two
library bugs getting it to load (both fixed, see "Bugs fixed" below). Once
loaded: **flat fill and thick uniform outline finally appeared** — MRI, sofa,
bulb, bar chart, skull all came back genuinely on-style. Two defects
remained: (1) backgrounds still coloured, not white (2) chair still tiled
into multiple objects.

### Round 4 — props for your ACTUAL reference frame (sofa, table, brain, camera): confirmed the LoRA works on real targets, backgrounds still not white
Same LoRA, same fixed prompt shape, targeting frame 1's actual props. All
four generated as recognisable, on-style objects. All four still on a flat
coloured field, never white. This directly caused a real bug — see next.

### The cutout bug this surfaced, and the fix
`compositor.cutout()` only ever tested "is this pixel bright" (`>= 240`).
Every SDXL+LoRA prop has a coloured background, so that test removed
**nothing**, and the prop pasted as a visible rectangle over the scene —
**silently**. No exception, alpha all 255, looked fine in the file listing.
Would have corrupted every cached prop the same way, invisibly, until someone
happened to look at a composited frame.

Fixed in `channels/pipeline/compositor.py`:
- `cutout(img, tolerance=N)` — when `tolerance` is given, matches the
  **median of the four corner pixels** instead of testing brightness. Same
  flood-fill-from-border algorithm, different predicate, so the
  hollow-face/hollow-interior guard still holds (verified).
- `erode_alpha(rgba, px=2)` — trims the 1-3px boundary fringe that survives
  the tolerance test and would otherwise paste as a white/coloured halo.
  Erosion, not a wider tolerance — a wider tolerance was what ate the
  table's legs in one test (background colour too close to the wood fill).
- `cutout_report(rgba, original)` — measures `kept` fraction and `halo`
  fraction, returns a verdict: `ok`, `FAILED: removed almost nothing`,
  `FAILED: removed almost everything`, or `HALO: ...`.

Fixed in `channels/pipeline/props.py`:
- `stage(key, image, ..., tolerance=None, erode=2)` now runs
  `cutout_report()` and **raises** on a FAILED verdict instead of filing a
  broken prop. A silently-corrupted prop can no longer reach the library.
  Un-eroded halo also raises if `erode<=0`.

All verified with unit-style checks (white bg still works, coloured bg
correctly flagged before fix / correctly cut after, hollow-interior survives,
erosion shrinks and never inverts). Not run through the actual pytest suite
this session (no pytest installed in this shell — `pip install pytest` or
run via whatever the project normally uses).

### The real merged frame
`channels/_local_workdir/merge_frame1.py` (script, kept) produced
`channels/_local_workdir/merge_frame1.png`: SDXL sofa + SDXL table (tolerance
auto-searched per prop against `cutout_report`, 2px eroded) + `draw.py`
character, composited via `compositor.composite()`, captioned with PIL. This
is the real pipeline, actually running, not a mockup. Character is
**standing beside** the sofa, not sitting in it (see step 14).

### Whole-scene-with-character test: confirms the compositor's reason to exist
User asked directly: can SDXL draw the WHOLE apple-stall reference frame
(two figures + stall + crates), skipping PIL/compositor entirely? Ran it
properly — 4 seeds with a flawed negative prompt (accidentally suppressed
people), caught the flaw, reran 4 more seeds with it fixed. **0 of 8 usable.**
Best case (seed 33): asked for two figures, got one — reproducing the exact
FLUX-era defect (`two-character` shots rendering one figure, ~18/160 frames)
that justified building the compositor in the first place. Other seeds: no
faces ever (blank white ovals), fabricated text/signage leaking in despite
being negative-prompted out, zero consistency between seeds.
**Conclusion: the SDXL-prop + PIL-character + compositor architecture is
confirmed correct, not just assumed. The character must never come from a
model.**

---

## Bugs fixed this session (infrastructure, not the pipeline itself)

1. **`.mcp.json` timeout 30000 → 600000.** Colab-mcp's own reconnect
   handshake needs up to 60s (`UI_CONNECTION_TIMEOUT`); 30000 was too tight
   and caused every tool call to "abort" client-side even though the kernel
   kept running server-side (confirmed via the server log). **Needs a Claude
   Code restart to take effect — not yet restarted this session.**

2. **peft/torchao incompatibility blocking LoRA load.** Colab preinstalls
   `torchao 0.10.0`; `peft 0.20.0`'s `is_torchao_available()` *raises*
   instead of returning `False` on an old version, so the LoRA dispatcher
   crashed before reaching the working plain-Linear path. Fixed by
   uninstalling torchao (nothing in this pipeline uses it — it's a
   quantization lib, SDXL here is plain fp16).

3. **Kohya LoRA text-encoder keys don't map in diffusers.** After the
   torchao fix, `load_lora_weights()` still crashed
   (`IndexError: list index out of range` in `get_peft_kwargs`) because the
   LoRA's text-encoder keys produce an empty rank dict. Fixed by loading only
   the UNet half of the state dict (2166 of 2958 keys) — the right call
   anyway, since this is a style LoRA and style lives in the UNet
   (`lora_object_prompts.md` already specifies text-encoder OFF for the same
   reason).

4. **The MCP WebSocket 1MB frame cap actually got hit and killed the
   connection.** Server log: `CloseCode.MESSAGE_TOO_BIG`, 1,117,482 bytes
   sent against a 1,048,576 cap — caused by emitting 4 base64 PNGs in one
   cell output. STATUS.md's rule about this existed and got violated anyway
   because nothing *measured* the payload before printing. Fixed with an
   `emit(path, tag, max_px)` helper (now in the notebook, cell near the end)
   that shrinks an image until its base64 is under a 400,000-char budget and
   prints exactly one image per call. Used successfully afterward with zero
   drops.

---

## Unfinished, in-repo — do not lose this

`channels/pipeline/draw.py` has an **untested** change:
- `SEATED_ARMS` dict added (`rest`, `drape`, `lap`) — `drape` stretches the
  near arm along a sofa's backrest top, matching the reference frame where
  the arm rests ON the furniture rather than hanging by the hip.
- `figure_seated(..., arms="rest", back_top=None)` — new params, `drape`
  snaps the hand onto `back_top` if given.
- This was written in direct response to a measured gap (rendered the
  reference frame with old code, arms were invisible/wrong) but **the
  session pivoted to the SDXL-prop path before re-rendering with `drape`**.
  First thing to do next session if returning to `draw.py` work: render
  `figure_seated(..., arms="drape", back_top=sofa_info["back_top"])` against
  the sofa and look at it.

`channels/_local_workdir/repro_ref1.py` and `repro_ab.py` — scripts that
reproduce reference frame 1 with current code, useful for before/after
comparison, never fully looked at after the `drape` edit (repro_ab.py was
written to A/B rest vs drape but its output was never reviewed — the pivot
happened first).

---

## Files touched this session

| file | what changed |
| --- | --- |
| `channels/pipeline/compositor.py` | `cutout(tolerance=)`, `erode_alpha()`, `cutout_report()` — all new |
| `channels/pipeline/props.py` | `stage(tolerance=, erode=)` — gate now rejects bad cutouts |
| `channels/pipeline/draw.py` | `SEATED_ARMS`, `figure_seated(arms=, back_top=)` — **untested, see above** |
| `.mcp.json` | timeout 30000 → 600000 — **needs Claude Code restart** |
| `channels/_local_workdir/*.py`, `*.png` | scratch scripts + real output images, all kept, not garbage |
| `STEP11_RESUME.md` | written mid-session, now superseded by this file |

Not a git repo (`git status` → fatal, no `.git`). Nothing is committed
anywhere; everything above is just files on disk.

---

## Open decision — resolve this first next session

Raised late this session, not settled: **should the visual style match Zenn
(STATUS.md's original framing: "quality vs Zenn — not parity, closing the
gap") or deliberately differentiate from it?** These are different projects.

If differentiating, evaluated three proposed styles against the existing
architecture (which depends on flat fill / no shading / no perspective / no
hands drawn — that's *why* cutout and compositing are cheap):

| style | compatible with current architecture? | cost |
| --- | --- | --- |
| **Blueprint** (white line on blue) | yes, fully — same properties, palette inverted | low, ~a weekend |
| **Chalkboard** (chalk on slate) | partially — needs a grain/texture layer, background still flat | medium |
| **Vellum ink-blot** (bleeding ink) | no — breaks flat-fill and no-perspective, would mean full-frame generation, which the whole-scene test just showed fails 8/8 | effectively a different project |

A separate list of hosted tools (NightCafe, Canva Magic Media, Flick.art,
Adobe Animate, Runway Gen-3, VideoScribe, Kling AI, CapCut) was also
proposed and **not recommended** — nearly all reopen the character
-consistency and per-generation-cost problems this pipeline was specifically
built to eliminate.

## My actual recommendation, unfiled elsewhere

1. Confirm match-Zenn vs. differentiate before doing more style work — it
   changes what "next" means.
2. **Revised per user direction (2026-08-26): don't pick by architectural
   convenience — run an unbiased A/B at production level, minimal effort,
   and let the winner decide, not sunk cost in either direction.**
   Concretely:
   - Generate ONE small batch (the existing frame-1 props + the merged
     frame already on disk) and derive BOTH variants from it cheaply:
     as-is (closing-gap-to-Zenn) and a Blueprint palette/line-invert. This
     is a post-process branch on output already produced, not a second
     generation pass or new LoRA — that's what keeps it minimal-effort.
   - Test both against the real target, not internal preference: either
     blind outside viewers who don't know which is "ours"/which is meant
     to differentiate, or (if there's a live channel/audience) post both
     as matched-length clips and read actual retention/CTR/comments.
   - Decision rule: whichever wins the external signal ships, full stop.
     No tie-break toward the version more effort was already sunk into.
   - Chalkboard/Vellum stay out of this round (cost/architecture-fit
     already rules them out per the table above) — this A/B is
     specifically match-Zenn vs. Blueprint.
3. **Do step 13 (layout templates) regardless of the style decision.** It's
   the actual bottleneck between "1 hand-tuned frame" and "160 frames," and
   every style option needs it.
4. **New idea, not yet started, worth doing regardless of style:** use SDXL
   for full BACKGROUND scenes (market street, hospital room), not just
   foreground props. A background is the bottom of the z-order — it never
   needs `cutout()`, never touches the character-consistency problem (no
   character is ever in it), and directly targets the biggest visible gap
   vs. the reference frames (yours are visually bare; a painted background
   costs nothing extra to composite under existing props/character). Same
   review-gate-and-cache pattern as `props.py`, applied to backgrounds
   instead of objects.
5. **Motion is still untouched.** The video build is a plain ffmpeg concat
   of static frames — no `zoompan`, no transitions. This was flagged as the
   single highest perceived-quality win available for the least effort, and
   nothing has been done about it yet.

---

## Colab state at end of session

Runtime: L4, still connected as of the last successful cell run, base SDXL +
DD-vector-v2 LoRA resident in `pipe`. **Not verified still connected** — if
resuming, check first before assuming GPU state carries over; `/content` is
wiped on any disconnect and the ~7GB SDXL download + LoRA load would need to
repeat (~5 min, unavoidable, budget for it).

⚠️ Still outstanding, untouched, unrelated to this session's work: HF token
hardcoded in notebook cells 4, 12, 33, 36 (`hf_***ROTATE-ME***`),
also in `stickman.ipynb.bak`. Cell 13 already does `load_dotenv` correctly.
Rotate the token, then point those four cells at the dotenv path.
