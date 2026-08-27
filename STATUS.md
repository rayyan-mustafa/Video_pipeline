# STATUS — where the project is, and what happens next

Written 2026-08-26. Supersedes `START_HERE.md` on the **visual** side; everything
that file says about the script pipeline, costs, and Colab traps still stands.
`HANDOFF_NEXT.md` remains the long history.

---

## The one-paragraph version

The visual strategy changed. The character is no longer drawn by a diffusion
model at all — it is **drawn in Python** (`pipeline/draw.py`), which makes
consistency 100% instead of 96%, costs nothing, needs no GPU, and permanently
kills the "two-character renders one figure" defect. The planned **character
LoRA v2 is cancelled**; the 100 Nano Banana images are not needed. What a model
is still needed for is **complex objects** (an MRI machine, a mosquito, a lung)
— those become a **cached prop library**, generated once, reviewed once, reused
forever. Whether that needs a trained LoRA is **still an open question, and the
next thing to test.**

---

## THE STEPS — where we are

```text
  ✅ DONE   1. pipeline works end to end
  ✅ DONE   2. character consistency SOLVED -- by code, not by a model
  ✅ DONE   3. LoRA v1 killed (learned noise, proven by amplification)
  ✅ DONE   4. floor + line weight moved out of the prompt into code
  ❌ CUT    5. character LoRA v2 / the 100 Nano Banana images -- NOT NEEDED
  ✅ DONE   6. draw.py: figure, 6 faces, 6 poses, character flags, furniture
  ✅ DONE   7. props.py: prop store + review gate
  ✅ DONE   8. prop_key in the scene planner
  ✅ DONE   9. 100 object prompts written
  ────────────────────────────────────────────────────────────────────────
  ⏳ NOW   10. reconnect colab-mcp        <-- blocked, see below
  ⏳ NOW   11. generate ~10 props on BASE SDXL and LOOK at them
              this answers "do we need to train at all" for ~0.2 units
  ────────────────────────────────────────────────────────────────────────
  ⬜ NEXT  12. if good -> generate the other 90, review, admit. NO TRAINING
           13. per-shot-type layout templates (the known gap)
           14. generate-on-miss runner
           15. train an object LoRA ONLY if 11 shows a nameable failure
```

---

## What changed this session, and why

### The character is code now

`pipeline/draw.py` draws the figure. A stick figure is a circle and five
strokes — the single most drawable object there is — and putting it in code
made every attribute that kept failing free and exact:

| attribute | was | now |
| --- | --- | --- |
| uniform stroke width | failed on 5 phrasings, 3 rounds | `width=LW`, uniform **by construction** |
| flat fills | palette lock had to repair them | a literal RGB tuple |
| consistency | ~96% (2-4% drift) | **100%** — it is the same function call |
| two figures in a frame | ~18 of 160 rendered ONE | two calls. The defect cannot occur |
| character variety | impossible | flags: `hair`, `beard`, `glasses`, `coat`, `child` |
| cost | a training run + GPU per frame | **zero** |

**A "female doctor stickman" is not an asset.** It is
`figure(..., hair="long", coat=True, glasses=True)`. Cached as a PNG she would
be locked into one pose and one expression; as parameters she composes with all
six expressions and every pose. Props are stored, characters are always drawn.

The six expression names match `look.py` exactly, so the existing lexical-energy
expression assignment keeps working with **no change at all** — the clause just
stops being sent to a model and starts being drawn.

### The rule that decides who draws a frame

Already in the codebase, no new mechanism needed:

```python
shots.SHOTS[shot]["character"]      # True -> draw.py.  False -> prop library
```

**Never generate a scene that contains the character.** The moment a model draws
the figure it is not *our* figure, and `look.py`'s expression assignment stops
meaning anything. Restage to avoid contact instead — `two-character`'s clause
already says "empty space between them", which is the same constraint arrived at
from the other direction.

### Props are a library, not a per-frame generation

The obvious reason is cost. The real reason is **consistency**: generate a lung
fresh each time and video 3's lung differs from video 40's — the drift problem
returning through the props. A cached lung is the same lung forever. Same
argument as the palette lock and the PIL captions.

Review cost amortises too: look at a prop once, not at 160 frames per video.

**The review gate is not optional.** Caching amplifies whatever it stores; one
bad MRI admitted silently becomes forty bad videos. `props.lookup()` refuses to
return anything with `reviewed: false`.

---

## Files

| file | what it is |
| --- | --- |
| `channels/pipeline/draw.py` | **new.** The figure, 6 faces, 6 poses, character flags, chair/table/sofa |
| `channels/pipeline/props.py` | **new.** Prop store, review gate, `coverage()` |
| `channels/pipeline/compositor.py` | cutout, anchors, scale, composite (already existed) |
| `channels/make_style_sheet.py` | **new.** Every face/pose/prop on one page. Run after ANY draw.py change |
| `channels/lora_object_prompts.md` | **new.** 100 object prompts + rules + SDXL settings |
| `channels/lora_prompts_full.txt` | **new.** The same 100, style block pre-pasted, one per line |
| `channels/pipeline/scenes.py` | modified — emits `prop_key` |
| `channels/lora_prompts.md` | ⚠️ **OBSOLETE** — the cancelled character LoRA |

Tests: **97 passing** (was 67).

---

## Step 11 — the next real action

**Do not train anything yet.** Generate ten props on **base SDXL**, no LoRA,
using the style block and the negative prompt from `lora_object_prompts.md`.
Suggested: prompts 1 (lungs), 23 (MRI), 39 (sofa), 84 (mosquito), 94 (light
bulb), plus five of your choosing.

Then look at them:

```text
consistent and on-style  ->  generate the other 90. NO TRAINING NEEDED.
inconsistent             ->  you now have a CURATED set of the good ones,
                             which is better training data than anything
                             planned blind.
```

Either outcome is a win, and it costs ~0.2 units instead of committing to a
training run up front. **"flat vector illustration, thick black outline, flat
colours" is a style SDXL knows extremely well** — and existing flat-vector SDXL
LoRAs can be downloaded rather than trained.

### Two things that will bite

- **`add_watermarker=False`** on `StableDiffusionXLPipeline.from_pretrained()`.
  SDXL applies an **invisible watermark by default** when `invisible-watermark`
  is installed. Miss it and you ship watermarked props believing the local path
  is clean. This is the one real gotcha in moving off FLUX.
- **No people, no stick figures, in any generated prop.** `stick figure` and
  `person` are in the negative prompt for a reason: a prop that arrives with a
  figure attached cannot be placed by a single ground anchor, and
  `props.stage()` refuses it at the door.

---

## Known defects, honestly

| | |
| --- | --- |
| **quality vs Zenn** | **not parity.** Faces, poses and the table are close. Figures read stiff and mechanical against Zenn's; staging is a row of figures rather than a staged scene |
| seated proportions | torso short, knee bend subtle. Constants, fixable |
| `up` pose | raised arm overlaps the head, figure reads one-armed |
| chair | seat surface reads weakly; backrest and legs dominate |
| **layout templates** | **missing.** Nothing decides `x=0.68`. Hand-written per demo today — step 13 |
| `test_notebook.py` | 1 failure, **environmental not ours**: it asserts the notebook raises when CPU deps are missing, but PIL/numpy/soundfile/ffmpeg are all installed here |
| ⚠️ **Defender** | **two source files went to 100% null bytes mid-session.** Add `f:\wierd_bio` to Windows Defender exclusions |

---

## colab-mcp — read before retrying

`/mcp` reports **0 of 5 connected**. Diagnosed this session:

- `.mcp.json` config intact, `uvx.exe` present, package fetches and runs fine
  (104 packages, 4.9s, `--help` works)
- **no log directory exists** → the server has never started under this session
- all five servers down → session-level, not colab-specific

From the source (`session.py`): `open_colab_browser_connection` is an *injected*
tool that only exists while the server runs. It calls
`webbrowser.open_new(...#mcpProxyToken=…&mcpProxyPort=…)` — token and port are
generated **inside the running process**, so the URL cannot be built from
outside. It then waits `UI_CONNECTION_TIMEOUT = 60.0` seconds for the browser
handshake and fires `send_tool_list_changed()` on success.

- **Raise the timeout.** `.mcp.json` has `30000`; the handshake alone waits 60s
  on top of a git fetch and a 104-package install. Use `120000`.
- It opens `/notebooks/empty.ipynb`, a scratch notebook — **not** `stickman.ipynb`.
- Reconnect via `/mcp` in the terminal; no restart needed, the tool reappears
  via `tools/list_changed`.

---

## Do not redo these

1. **Do not train a character LoRA.** `draw.py` replaced it. v1 learned a
   degenerate direction (erased the drawing at 8x, pure noise at 15x).
2. **Do not generate the 100 Nano Banana character images.** Cancelled.
3. **Do not generate a scene containing the character.** See the routing rule.
4. **Never `display()` an image through colab-mcp.** The WebSocket caps a frame
   at 1,048,576 bytes and it silently eats `update_cell` writes.
5. **Do not use the Drive model cache** — download direct (31.4 GB in 2.3 min).
6. **A redraw must change the seed or the prompt.** FLUX at a fixed seed is
   deterministic; `draw.py` is too, deliberately.

---

## The standing lesson, reconfirmed twice this session

**Never judge a visual change from code or summary statistics. Draw a batch and
look at it.** `make_style_sheet.py` exists because the engine was being tuned
two figures at a time, and defects obvious across the whole set survived several
rounds each — eyebrows drawn *outside* the head on every face, the "worried"
mouth curving *upward* (a smile), smug's eyelid drawn as a bar struck through
the eye, feet missing from every pose.

And its corollary, also earned twice: **when a shape looks wrong, compute where
it actually lands before nudging constants.** The eyebrow and the smiling-frown
bugs were both settled by arithmetic in seconds after surviving rounds of
by-eye tuning.
