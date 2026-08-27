# Handoff: the LoRA is NOT NEEDED — base passes the drift test at ~2-4%

Last updated 2026-08-25 (second sitting). Paste this whole file into a new chat.

> **READ THIS FIRST — the LoRA question is CLOSED. Do not train it further.**
>
> Two measurements, both now done, both free to re-check:
>
> 1. **The trained LoRA learned noise.** Its delta reaches the model fine and
>    scales with `lora_scale`, but amplifying it **erases the drawing at 8x
>    and produces pure orange static at 15x**. A LoRA that learned a character
>    would draw a *stronger* character when amplified. More steps along that
>    gradient buys nothing — the old "+1000-1500 steps" plan is **refuted, not
>    deferred**.
> 2. **Base does not need fixing.** 45 character frames across 44 distinct
>    action clauses, drawn on base with the LoRA stripped: **~2-4% drift**,
>    inside PLAN.md's "under 5%: the current approach is fine" band. The
>    prompt fixes already solved the problem the LoRA was commissioned for.
>
> The premise that justified training — *"the figure settled on FLUX's
> oval-body design"* — was measured **before** those prompt fixes landed and
> no longer holds.
>
> **The budget now belongs to two other things:** the `two-character` shots
> that render only one figure (~18 of 160 frames in a real video, the largest
> known visual defect), and the content issues in item 6.
>
> Also fixed this session and worth reading before you touch Colab: **delete
> the Drive model cache** (direct download is 7x faster and the cache is what
> stalls), and **never `display()` an image through colab-mcp** (a 1 MB
> WebSocket frame cap was the cause of every "random disconnect").

Companion to `PLAN.md`, `channels/README.md`, `channels/TESTING_VISUALS.md`,
`HANDOFF_shot_rotation.md`.

---

## TL;DR

**The pipeline works end-to-end and has been run for real** (10 frames, L4,
palette-locked, captions burned in — see *Verified by real renders* below).

**The character LoRA is trained, loads correctly, and has been A/B/C compared
against base on the live L4 via colab-mcp — and it does almost nothing.**
Root cause is measured, not guessed: `||B||` (the part of a LoRA that actually
learns) moved 71% between checkpoint-250 and checkpoint-500, meaning training
had not begun to converge when it stopped. **Jump straight to *THE COMPARISON
HAS NOW BEEN RUN* below** — that section is the current truth; the *LoRA —
state as of this handoff* section right after this TL;DR is the history of
how training got justified and run, kept for context but superseded on the
two points marked below.

---

## Verified by real renders (not code review)

| | |
| --- | --- |
| Frames drawn | **10/10**, 1280x720, 0 fallbacks, 0 blank retries |
| Speed | **7.9 s/frame**, FLUX resident on L4 |
| Palette lock | **10/10 palette-clean**, 0 off-palette pixels |
| Captions | burned in, uppercase, correct slots |
| Tests | `test_pipeline.py` **61**, `test_notebook.py` **13**, all pass |

---

## Bugs found ONLY by looking at rendered frames

Every one of these passed code review and passed tests. They were found by
looking at pictures. **That is the standing lesson of this project.**

1. **Phantom black in the palette.** `_palette_image()` padded 250 unused
   palette slots with `[0]*n` = pure `(0,0,0)`, while our near-black is
   `(26,26,26)`. `quantize()` snapped linework to a colour **not in PALETTE**.
   Worst frame 5.37% off-palette, drift exactly 45.0 = `|(0,0,0)-(26,26,26)|`.
   Scaled with linework density, so sparse frames looked fine.
   **Fixed:** pad with a repeat of a real palette colour.
2. **Captions drawn after the palette lock.** PIL anti-aliases glyph edges,
   adding ~230 colours per captioned frame. **Fixed:** re-snap after drawing.
3. **CLIP truncation.** FLUX has two text encoders — T5-XXL (512 tokens) and
   **CLIP (hard 77-token cap)**. Our prompts run ~191 tokens, so CLIP never saw
   the character bible, the expression, or the action on any frame.
   **Fixed:** `STYLE_PREFIX` split into a short front + `STYLE_TAIL`, and
   `visual_prompt()` now puts the SUBJECT BEFORE the shot clause.
4. **Negation inversion in `CHARACTER_BIBLE`.** Six negations ("no nose, no
   ears, no hands drawn in detail"). Diffusion keeps the token and often
   *summons* the thing — 3 of 6 character frames had a nose, ears, fingered
   hands and shoes. **Fixed:** bible is positive-only, and the torso is now
   countable: *"a body of five thin straight lines - one spine, two arms, two
   legs"*.
5. **`centred` vs grounding.** `object-focus` said "one large object alone,
   **centred**, resting on a stand or shelf". After the reorder, `centred` won
   and a jar that had been sitting on its table came back **levitating**.
   **Fixed:** `centred` removed; "resting directly on a simple flat surface".

---

## Hardware — measured, and it matters

**Compute units are billed by WALL-CLOCK, not GPU use.** The notebook's Step 0
comment says so; it drives every decision below.

| GPU | verdict |
| --- | --- |
| **T4** | ❌ **Cannot run this pipeline.** 16 GB forces CPU offload, which fills the 12.7 GB system RAM and crashes. Happened once, reproducibly. |
| **L4** | ✅ **Use for image generation.** 24 GB keeps FLUX resident. ~3.1 units/video. |
| **A100** | Use **only** for LoRA training. ~5.2 units/video for generation — **68% worse than L4**, because ~11 min of fixed overhead (pip, Drive, FLUX load) doesn't speed up but does cost more. |
| TPU | ❌ Cannot run this at all — 4-bit NF4 via bitsandbytes is CUDA-only. |

**Per video on L4 (160 frames, 8 min audio): ~3.1 units → ~31 videos/month.**
Phase 2 (review, encode, upload) is free on the CPU runtime.

### Approximate hourly rates — and where units actually go

> **CORRECTED 2026-08-25.** The L4 figure below was ~3x too high. Colab's own
> session readout says **1.54 units/hr on an L4**, not 4.8. Verified while an
> L4 was attached and Step 0 was reporting `NVIDIA L4`. At 1.54/hr a 77-unit
> balance is **~50 hours** of GPU runtime, not ~16.
>
> This matters because several decisions in this document were made under the
> wrong number — the panic about idle runtime especially. Ten idle minutes is
> **~0.26 units**, not ~0.8. Idle time is still waste and still worth avoiding,
> but it is not the emergency the older sections imply. Re-read any cost
> argument below with the 1.54 figure in mind.
>
> The A100 and T4 rows have NOT been re-verified and may be wrong by a similar
> factor. Measure before trusting them.

| GPU | units/hr | 77 units buys |
| --- | --- | --- |
| A100 | ~15 (unverified) | ~5 hours |
| **L4** | **1.54 (measured)** | **~50 hours** |
| T4 | ~1.8 (unverified) | (cannot run this pipeline) |

**Failed launches are NOT what drains the budget** — each died in 60-90 s,
~0.3 units. What drains it is the runtime sitting **connected and idle** while
someone reads a traceback and edits a cell: ten idle A100 minutes = 2.5 units.
Edit cells with the runtime disconnected; connect to run, disconnect after.

**Training on A100 is the cheaper choice, counterintuitively.** L4 is 1/3 the
rate but FLUX is 12B params — bf16 weights alone are ~24 GB, L4's whole card.
Training there needs 8-bit Adam plus text-encoder offload, runs 3-4x slower,
and can still OOM. Total cost is roughly a wash with none of the risk.
**A100 for training only; L4 for the 30 generation runs a month.**

**High-RAM is unnecessary** once FLUX is resident, and it raises the hourly rate.
Use Standard.

**The biggest remaining waste:** the ~6 min FLUX load from the Drive cache costs
~0.5 units *every video* (~15 units over 30 videos). Local-disk caching would
likely be faster AND avoid the crashes below. Worth measuring.

### Google Drive is fragile for the model cache

- `OSError: I/O error (os error 5)` during `snapshot_download`, twice.
- Cause: HF's **Xet** transfer does many small random writes; FUSE-mounted Drive
  fails on that pattern.
- **Fix that worked:** `HF_HUB_DISABLE_XET=1` (patch `huggingface_hub.constants`
  at runtime — the env var alone is frozen at import).
- A crash also **lost an un-flushed 4.99 GB shard**, leaving the cache 58.2 GB
  for a 31.4 GB model. Clean `.incomplete` files before retrying.
- **Never walk the whole Drive cache with `os.walk`** — that almost certainly
  caused the second crash.

---

## LoRA — state as of this handoff

**Decision path:** metrics said "no LoRA needed"; that was **wrong** — ink
coverage cannot see a different character. Then prompt fixes were pushed twice
(positive bible, CLIP window, countable torso). Result: consistency improved but
the figure settled on FLUX's own **oval-body** design, not the line-torso
stick figure, with frame 8 the only true stick figure. Prompting is exhausted →
LoRA justified.

**Training set is NOT FLUX output.** The user generated **23 images with Google
Nano Banana** (their own generation — no copyright issue). Prepared as:

- `lora_data/` — **23 PNG at 768x768 + 23 caption files**
- trigger word **`stkmn style`**
- cleaned with the notebook's own `lock_palette()`: off-palette **62.4% → 1.44%**
- **padded** to square, not cropped (4 of 23 clipped the figure on centre crop;
  the edge-to-edge ink in the other 19 was the *floor line*, not the figure)

**Training command** (Step 21e), running on A100-SXM4-80GB:

Actual command run (cell `O7IVBdeNbv8W`, after 3 pre-flights — see below):

```
accelerate launch train_dreambooth_lora_flux.py \
  --pretrained_model_name_or_path black-forest-labs/FLUX.1-schnell \
  --instance_data_dir lora_data --output_dir lora_out \
  --instance_prompt "stkmn style" --resolution 768 \
  --train_batch_size 1 --gradient_accumulation_steps 4 \
  --learning_rate 1e-4 --lr_scheduler constant --lr_warmup_steps 0 \
  --max_train_steps 500 --checkpointing_steps 250 --rank 16 --seed 0 \
  --mixed_precision bf16 --gradient_checkpointing
```

**TRAINING COMPLETE.** Three launch failures along the way, all dependency/data
issues, all now fixed as pre-flights in cell `O7IVBdeNbv8W` that run BEFORE any
GPU work:

1. **diffusers gate.** `main` branch script calls
   `check_min_version("0.40.0.dev0")` but Step 1a pins `diffusers>=0.35,<0.40`
   (0.39.0 installed). **Do not upgrade diffusers** — the FLUX loader depends
   on that range. **Fix: fetch the script from the matching release tag.**
2. **Colab's stale `torchao` 0.10.0.** Died during peft's LoRA injection:
   `dispatch_torchao` -> `is_torchao_available()`, which **raises** on an old
   copy instead of returning False. peft runs that dispatcher for *every*
   module it wraps, so one unused preinstalled package blocks all LoRA
   injection. **We do not use torchao** — quantization here is bitsandbytes
   NF4. **Fix: uninstall it, don't upgrade it** (torchao>=0.16 would pull a
   newer torch, which the working 4-bit FLUX loader depends on staying put).
3. **Caption `.txt` files inside `lora_data/`.** v0.39.0's `DreamBoothDataset`
   is `[Image.open(p) for p in Path(instance_data_root).iterdir()]` — no
   extension filter, so a caption file crashes it with
   `UnidentifiedImageError`. This script version also doesn't read per-image
   captions with `--instance_data_dir` anyway — it applies one
   `--instance_prompt` to every image, which is what we want for a
   single-character LoRA. **Fix: pre-flight 3 moves every non-image file out
   to `lora_captions/`** before each launch.

**Lesson, and it is a money lesson:** all three failures were dependency/data
issues that cost a partial A100 model load apiece. Version and data checks now
run as a 5-second CPU probe before touching the GPU.

**Final run:** 500 steps (cut from the planned 1000 — 23 images at
20-40 steps/image lands near 500; going to 1000 risks overfitting a small set).
`--checkpointing_steps 250` was set precisely so an early stop was possible
without losing the run — in the end the interrupt didn't take effect in time
and it ran to completion anyway, which was fine: both checkpoints came out
real and different (see below), so nothing was wasted.

**Verified on disk before disconnecting** (all in the Drive-mounted workdir,
survives disconnect):
```
character_lora_step250.safetensors   89,747,208 bytes   md5 f5656b4c...
character_lora_step500.safetensors   89,747,208 bytes   md5 b342eac6...
character_lora.safetensors           (currently = step500)
```
`step250 != step500` confirms real learning happened between the two — not a
stalled/duplicate write. `step500`'s hash matches `lora_out/pytorch_lora_weights.safetensors`
exactly, which is *expected*: training stopped right on a checkpoint boundary,
so the "final" save and the "checkpoint-500" save captured the same state.

**A resume cell exists:** `PoehYu0BlDJP`, right after the training cell. It
resumes from a checkpoint for `EXTRA_STEPS` more — NOT a retrain, so it only
costs the extra steps plus one model load.

> **SUPERSEDED by the measurement below.** `EXTRA_STEPS` defaults to 100 and
> the reasoning above ("1000 steps is generous", "usable LoRA exists at
> step 500") was a rule of thumb that turned out wrong for this set: `||B||`
> moved 71% between step 250 and step 500, i.e. nowhere near converged at
> either checkpoint. Resume from `checkpoint-500`, not 250, and plan for
> **+1000-1500 steps**, not 100. See *THE COMPARISON HAS NOW BEEN RUN* below.

**Why schnell, not dev:** FLUX.1-schnell is Apache-2.0. **FLUX.1-dev is
non-commercial licence only** — wrong for a monetized channel.

**LoRA loading is already wired** into `get_flux()` (Step 1c):
`LORA_PATH = "character_lora.safetensors"`, `LORA_SCALE = 0.85`,
`USE_LORA = True`. Absent file ⇒ base model, pipeline still runs.
Step 21d forces `USE_LORA=False` — never train a LoRA on a LoRA's output.

---

## Colab cell IDs (this notebook instance)

`cell002` 1a · `cell004` 1b · `cell006` 1c · `cell008` 2 · `cell011` 3 ·
`cell045` 20 (style lock) · `cell056` 23 (draw) · `o2pvZHPgP5Lh` 21d

---

## THE COMPARISON HAS NOW BEEN RUN — the LoRA is UNDERTRAINED

Session of 2026-08-25, on an L4, via colab-mcp. 30 frames drawn (base /
step250 / step500), twice: once with the prompts as they were, once after
fixing the trigger. Measured, not eyeballed.

### Two separate findings, in the order they were found

**1. The trigger word was missing from every prompt.** Training used
`--instance_prompt "stkmn style"`, but `visual_prompt()` never emitted that
phrase, so the LoRA loaded, fused, reported no error and was never invoked.
**Fixed** — `config.LORA_TRIGGER`, added to the subject on character frames
only, at ~token 18 so it lands inside CLIP's 77-token window. Mirrored into the
notebook's `build_visual_prompt` and into `style.json` (`lora_trigger`), so a
Step 24 redraw cannot drift. `rebuild_prompts.py` re-bakes `visual_prompt` from
the stored action/shot/expression with **no API calls** — a prompt-constant
change must never cost budget or re-roll the actions being compared.

**2. Fixing the trigger did NOT make the LoRA work.** This is the part that
matters, and it is why the first finding was not the answer:

| % pixels changed, character frames | base→250 | base→500 | 250→500 |
| --- | --- | --- | --- |
| **with** trigger | 0.85% | 0.96% | 0.82% |
| **without** trigger | 0.85% | 1.04% | 0.97% |

Identical within noise. **Adding two words to the prompt moved the picture
2.36% (max 5.06%) — more than the entire LoRA does at scale 0.85.**

### Why: it trained, but nowhere near far enough

LoRA inits `lora_A` random and **`lora_B` at zero**; the applied delta is
`B@A`, so `||B||` is the only number that means "this learned". Total norm is
mostly A's random init and proves nothing.

```text
                 ||A||      ||B||
step250         295.99     10.34
step500         296.02     13.78
||B(500) - B(250)|| = 7.38   =  71% relative movement in the last 250 steps
```

684 tensors, none still zero — so training *is* working. But **71% movement
over the final 250 steps means it had not begun to converge at step 500.** A
converged LoRA moves a few percent, not 71%. The earlier reasoning that "23
images at 20-40 steps/image lands near 500" was a rule of thumb, and the
weights say it was wrong for this set.

**So the standing lesson holds twice over:** the metric (`step250 != step500`,
different md5s) said "real learning happened" and was technically true and
practically useless — it could not distinguish "learned" from "learned 15% of
the way". And the first render comparison looked like a clean null result when
it was actually measuring a disconnected LoRA.

## colab-mcp — operational notes, hard-won this session

The `colab-mcp` MCP server (`.mcp.json`, `uvx git+https://github.com/googlecolab/colab-mcp`)
drives a **live Colab notebook** from Claude directly — no copy-pasting cells.
Docs are thin; this is what actually worked:

- **Connecting is two steps.** `open_colab_browser_connection` (no args) opens
  a **separate automation-controlled browser window** — not your normal
  Chrome tab. You then have to manually open/connect the real notebook
  *inside that specific window* (File → Open, or paste the token it shows
  into your already-open tab — either way, the automation window and the
  content it can see must be the same document). First connect to a brand
  new session usually lands on a blank throwaway notebook — that's normal,
  not a failure; `get_cells` returning one empty cell means "wrong tab", not
  "broken".
- **`run_code_cell` has a 30s client-side timeout that is NOT the cell
  timing out.** A real FLUX load or a 3-variant draw takes minutes; the tool
  call errors at 30s while the cell keeps running for real on Colab's side.
  Recovery: do NOT re-run the same cell blind — poll `get_cells` (by cell
  **id**, not index — see below) until you see real output, and only re-run
  if `execution_count` is still `None` with no output at all (meaning the
  click never landed).
- **tqdm/ipywidgets progress bars don't update in a `get_cells` snapshot** —
  only new `print()` stdout lines do. A widget frozen at "0%" across several
  polls is not proof of a hang; it kept reporting `0/3` for ~9 minutes here
  while the transformer genuinely finished loading underneath it. Don't
  interrupt on widget-percentage evidence alone — wait for an actual absence
  of new stdout, or the "done. X GB free" print line.
- **Cell **index** drifts every time you insert a cell** (`add_code_cell` /
  `add_text_cell` shift everything after it) — always target by cell **id**
  for `run_code_cell`, never assume a previously-seen index is still that
  cell. `get_cells` with an index range is only safe to re-use immediately
  after a fresh full listing.
- **To actually SEE a rendered frame** (the whole point, per the standing
  lesson): build a contact-sheet PNG in a code cell with PIL, `display()` it
  via `IPython.display.Image`, run the cell, then pull the `image/png`
  base64 out of the `display_data` output and decode it to a local file —
  `run_code_cell`'s own result often exceeds the tool's token cap and gets
  saved to a `tool-results/*.txt` file instead; read *that* file with Python
  to extract the base64, don't assume the inline result has it.
- **A `KeyboardInterrupt` mid-cell does not lose kernel state.** Top-level
  variables (`pipe`, partially-fused LoRA state, frames already saved to
  disk) survive — write a resume cell that checks what's actually on disk
  and in `globals()` rather than re-running the whole thing from scratch.
- **"Transport endpoint is not connected" from Drive-mounted file ops** is
  the FUSE mount dropping mid-session (same family as the Xet bug already in
  this doc) — re-run the Step 2 Drive-mount cell to reconnect, then retry;
  don't assume the file is actually missing until the retry also fails.
- Kernel state resets completely after Step 1a's restart — Step 0
  (`HAS_GPU`) and everything after it must be re-run from scratch, every
  time the runtime restarts, not just the first time.

---

## THE SCALE DIAGNOSTIC HAS NOW BEEN RUN — and it reframes the problem

Session of 2026-08-25 (second sitting), on an L4 via colab-mcp. Next-steps
item 1 below is **done**. 24 frames drawn: the 6 character frames at
`LORA_SCALE` 0.85 / 2.0 / 3.0 (plus 8.0 / 15.0 drawn and saved to Drive but
not yet read back — the colab-mcp connection dropped repeatedly and the
runtime was disconnected rather than idle-burn units to re-read PNGs that
already exist).

### What the contact sheet shows

Rows = scale, columns = the 6 character scene indices, judged **by eye** per
the standing lesson (`diagnostic_contact_sheet.png` in the workdir, and a
local copy at `channels/_local_workdir/`):

- **The character does not change at any scale.** 0.85, 2.0 and 3.0 all render
  the same circle head, the same single-line torso, the same line limbs, the
  same pose. Not weakly-toward-the-training-set; not degraded into mush. The
  third outcome the plan below did not anticipate: *nothing*.
- **The props drift slightly as scale rises**, and only the props. Frame 4's
  eye goes from a heavy black outline to thinner and more golden by 3.0;
  frame 8's door goes from outlined-with-grain to more solidly filled;
  frame 0's lung shape shifts a little. So the LoRA is *doing something* —
  it is simply not doing anything to the character.

### The observation that matters most, and it is not about the LoRA

**The BASE row already draws a correct stick figure.** Circle head, one line
for the torso, thin line limbs, no oval body — on all six frames, with no LoRA
fused at all.

That directly contradicts the premise recorded in *LoRA — state as of this
handoff* above, which justified training in the first place: *"the figure
settled on FLUX's own oval-body design, not the line-torso stick figure, with
frame 8 the only true stick figure."* On these renders that is no longer true.

The most likely explanation is that **the prompt fixes already solved it** —
the positive-only bible, the CLIP-window reorder, and the countable
"five thin straight lines" torso (bugs 3, 4 and 5 in *Bugs found ONLY by
looking at rendered frames*). Those landed *after* the renders that motivated
the LoRA.

**So the live hypothesis is no longer "the LoRA is undertrained". It is "the
LoRA has nothing left to correct".** A LoRA cannot move a character that is
already drawn the way the LoRA was trained to draw it.

### The 8x / 15x renders settle it: the learned direction is DEGENERATE

Measured on the CPU phase, free. `pct` = share of pixels changed vs base,
character frames only (`base` vs `base` = 0.00% is the sanity check that the
metric itself is sound):

```text
base        mean  0.00%   max  0.00%
s500@0.85   mean  0.96%   max  2.44%
s500@2      mean  0.88%   max  2.26%
s500@3      mean  1.93%   max  4.56%
s500@8      mean  4.49%   max 11.59%
s500@15     mean 36.62%   max 38.33%
```

**The LoRA is wired correctly and does reach the model.** The delta scales
monotonically with `lora_scale` (the 0.85→2 dip is noise at that magnitude).
So the structural hypothesis is **dead**: NF4 `fuse_lora()` rounding is *not*
swallowing the delta, and schnell's 4 steps are not blocking it either.

But look at what the amplification actually draws (`sweep_sheet_allscales.png`
on Drive; a 3-row base/8x/15x crop at
`channels/_local_workdir/sheet_8x_15x.jpg`):

- **8x** — the drawing is **erased**. Near-blank white frames with faint
  speckles. The only thing still visible is the caption, and that survives
  *because PIL paints it in Python after generation* — it was never FLUX's.
- **15x** — **pure orange noise**, edge to edge. Total destruction.

**That kills the "undertrained, train more" plan.** If the learned direction
were "draw the character the way the training set draws it", amplifying it
would exaggerate that style — a heavier line, a more insistent stick figure.
Instead it deletes the picture and then fills the frame with noise. The
weights have moved off zero into **noise, not into a coherent concept** —
which is exactly what 500 steps on 23 images produces.

So the 71% `||B||` movement was real and was movement **toward nothing
useful**. More steps only help when the gradient is heading somewhere good.

### THE DRIFT COUNT HAS NOW BEEN RUN — base passes, the LoRA is not needed

The question the LoRA was commissioned for, finally answered directly: **do
many frames look like ONE character?** Not "does the LoRA change pixels".

Method (2026-08-25, L4 then CPU):

- Re-planned the full 80-scene timeline locally, $0.0036, no GPU.
- **45 of those 80 carry a character** (character-moment 25, symbolic-metaphor
  11, two-character 9) across **44 distinct action clauses** — the old batch
  had 4, which is why it could never answer this.
- Drew all 45 on **base, LoRA explicitly stripped**, at the pipeline's real
  setting (`SEED_PER_FRAME = False`, seed 42 every frame), 7.8 s/frame.
- 45 frames means 1 frame = 2.2%, fine enough to land on PLAN.md's bands.

**Verdict by eye (`drift_contact_sheet.png` on Drive, local copy at
`channels/_local_workdir/drift_sheet.jpg`): 1-2 frames of 45 deviate at all,
= ~2-4%.** Every frame carries the same white oval head with black outline,
the same two-dot eyes, the same single-line torso, the same thin line limbs,
the same line weight and proportions.

**That is inside PLAN.md's "under ~5%: the current approach is fine" band.
The LoRA chapter closes.**

#### The metric lied first, again — read this before trusting any drift number

The first automated pass reported **20% outliers** and would have said
"blocking". It flagged frames 4, 8, 61, 63, 65, 69, 71, 75, 79 — and **every
single one is a symbolic-metaphor shot**, whose prompt *demands* "one large
symbolic object that dwarfs it, filled with one flat solid colour".

The ink-coverage metric was measuring **the giant object, which is the
design**, not a drifting character. Comparing *within* shot type, where
composition is held constant, the number fell to 11.1% — and looking at the
pictures put it at 2-4%.

So: **never compare ink coverage across shot types.** A drift detector has to
be per-shot, and even then it flags statistical outliers, not wrong
characters. The eye is still the instrument.

#### A separate defect the drift question does NOT cover

**`two-character` shots frequently render only ONE figure.** Frames 16, 43,
45, 47, 53 all specify *"two figures facing each other on the same floor line
... empty space between them"* and mostly show one. That is
**prompt adherence**, not character identity — 9 such frames in this batch,
which scales to ~18 of 160 in a real video. Cheaper to fix than anything
LoRA-shaped, and it is now the largest known visual defect. See item 6.

### What this does to the +1000-1500 step plan

**Do not spend those units. The plan is refuted, not deferred.** It rested on
"undertrained → train more", and "undertrained" only implies "train more" if
(a) there is a visible deficit for more training to close and (b) the current
direction is the right one. Both fail:

- **(a) fails** — base already draws the correct line-torso stick figure.
- **(b) fails** — amplifying what it learned destroys the image.

The one check still worth doing is free and is **not about the LoRA at all**:

**Re-answer the question the LoRA was meant to answer.** That question is not
"does the LoRA change the picture" but **"do 160 frames look like one
character?"** Draw a proper batch on base alone with the current prompts and
count the drifters against the thresholds already in `PLAN.md` (under ~5%
fine, 10-20% worth fixing, over 20% blocking). On the 6 frames seen so far
base looks strong. If a full batch agrees, **the LoRA is not needed** and the
budget belongs to the content-level issues in item 6 below.

If a full batch *does* show real drift, the retry is a different training
run, not a longer one: more images than 23, and a lower LR or a smaller rank
so the weights settle into a concept instead of into noise.

### Checkpoint-250 was deliberately NOT tested

`||B(250)|| = 10.34` against `||B(500)|| = 13.78` — 250 is strictly less
trained. If 500 moves the character zero at 3x, 250 can only move it zero or
less. Testing it would have cost a ~6 min FLUX reload to confirm a foregone
conclusion. Skip it unless the reason for testing changes.

---

## Next steps

> **⚠️ THIS SECTION IS HISTORY. The live step list lives in
> [START_HERE.md](START_HERE.md) — read that instead.**
>
> Items 1-5 below are done or superseded. Item 2 in particular ("resume
> training for +1000-1500 steps") is **refuted, not deferred** — do not act on
> it. Items 6-9 are still open and were carried forward into START_HERE.md's
> *other open work* section.
>
> Kept only because the reasoning shows how each conclusion was reached.

1. ~~**Cheap diagnostic first, before spending training units.** Redraw the 6
   character frames at `LORA_SCALE` 2.0 and 3.0 (no retraining, one FLUX load).~~
   **DONE — see *THE SCALE DIAGNOSTIC HAS NOW BEEN RUN* above.** The answer was
   neither of the two predicted outcomes: the character does not move at all,
   and the base model already draws it correctly. Item 2 below is **on hold**
   pending the two free checks in that section.
2. **Then resume training, for far more than +100.** ⚠️ **ON HOLD** — see
   above; the deficit this was meant to close may no longer exist. The staged
   resume cell
   `PoehYu0BlDJP` defaults to `EXTRA_STEPS = 100`; the 71% figure says that is
   nowhere near enough. Resume from `checkpoint-500` (not 250 — 500 is strictly
   further along) and plan for **+1000-1500**, or raise the LR above `1e-4`.
   Overfitting was the wrong thing to guard against here.
3. Re-run the comparison after that — the cells are all in the notebook now
   (`q5JxYptq-rXz` draws the three variants; `0Oau0FdWKN5j` measures them;
   `iw394uc9KW7_` / `ATBXCaqKKg7s` read the weight norms; `6A-IXyeqwLs1` is
   the 2.0/3.0 scale sweep and `9m58JWY70mMp` the 8.0/15.0 one; `rjbZi82j4kUT`
   builds the 6-row contact sheet across every scale and prints pct-changed
   against base — that last one is drawn but was never read back, run it first
   on a **CPU** runtime, it touches no GPU).
4. **The old trigger-less renders are kept** as `frames_notrigger_*` — they are
   the control for finding 1, do not delete them.
5. **Disconnect the runtime** whenever GPU work finishes
   (`from google.colab import runtime; runtime.unassign()`). Standing practice
   — and the reason a launch failure now costs 5 seconds of CPU instead of a
   partial A100 load.
6. **The largest known visual defect is now `two-character`, not the character.**
   Those shots specify *"two figures facing each other on the same floor line,
   medium-wide shot, empty space between them"* and frequently render **one**
   figure. Seen on frames 16, 43, 45, 47, 53 of the drift batch — 9 such shots
   in 80 scenes, so ~18 of 160 in a real video. Fix the clause before anything
   else: it is prompt-level, costs nothing to iterate with
   `run_scenes.py --shots`, and is worth more than any remaining character work.
   Other content-level issues: frame 0's lung reads as wings; "a single muscle
   fiber" renders as a blob; `object-focus`'s anti-anatomical `ask` is a nudge,
   not enforced.
7. **The review bottleneck is REVIEWING, not fixing.** At 31 videos/month x 160
   frames that is ~5,000 frames to look at, and a human must scan all of them
   to find the few bad ones even at a 2% fix rate. The unbuilt lever with the
   best return is **automated flagging** — per-shot ink coverage, figure
   height, connected components, off-palette pixels — surfacing only suspicious
   frames for human eyes. Turns "review 160" into "review the 12 flagged".
   Note the trap found this session: such a detector **must compare within
   shot type**, or it flags every symbolic-metaphor frame for having the giant
   object its own prompt demanded.
8. **A redraw must change the seed or the prompt.** FLUX at a fixed seed is
   deterministic, so re-drawing a bad frame unchanged returns the identical bad
   frame. `SEED_PER_FRAME = False` in Step 23 means every frame shares seed 42.
9. **If a LoRA is ever revisited**, it needs a different training run, not a
   longer one: 60-100 varied images of the SAME character (not 23), lower LR or
   smaller rank. **Do not duplicate the 23** — a duplicate gives an identical
   gradient, so it is mathematically the same as training longer on 23, which is
   what produced noise. Augmentation (flips, small rotations) adds a little real
   invariance; duplication adds nothing. **Do not import Kaggle/open-source
   stick-figure sets** — they teach "stick figures in general", which FLUX
   already knows, diluting the one-specific-character signal, and they
   reintroduce the licensing risk the Nano Banana set was chosen to avoid.

---

## Standing lesson

Three times this session a change looked right in JSON and was wrong in the
render — the flesh hands, the giant heart, the floating jar. Twice a *metric*
declared success while the picture disagreed. **Never judge a visual change from
prompt text or summary statistics. Draw a small batch and look at it.** Ten
frames cost $0.0004 in planning and ~90 seconds of GPU.

**A fourth time, and it cost the most.** The whole LoRA — the Nano Banana set,
the three failed A100 launches, the 500 steps, the A/B/C comparison, the
71%-movement analysis, the plan to spend 1000-1500 more steps — rests on one
sentence written from renders made *before* the prompt fixes landed: *"the
figure settled on FLUX's own oval-body design."* Nobody redrew the base model
after those fixes to check whether the premise still held. It did not: base
now draws the line-torso figure correctly on 6 of 6 frames.

So the lesson generalises past "look at the render". **Re-check the PREMISE
after you change anything upstream of it.** A justification measured before a
fix is not evidence about the world after the fix, and every hour spent
downstream of a stale premise is spent solving a problem you may no longer
have.

## VISUAL QUALITY: three prompt rounds, then a move to code

Triggered by a side-by-side against the reference channel, which was plainly
more professional than ours. The gap was **quality**, not consistency — and the
earlier drift test had only measured consistency. Two frames can be perfectly
consistent with each other and both look amateur.

### What prompt work DID fix (rounds 1-3, ~0.15 units each)

| defect | cause found | fix |
| --- | --- | --- |
| **Oval torso** | `"one straight vertical line for the whole TORSO"` — a torso is a body part with volume, so FLUX drew the volume and put a line near it | `"one single straight vertical STROKE for the body"` |
| **Thin lines fighting the style** | `CHARACTER_BIBLE` said `"drawn as single THIN lines"` while `STYLE_PREFIX` said `"very thick heavy marker"` — **a contradiction inside one prompt** | removed `"thin"` |
| **Fingers / shoes** | `"each limb ending in a plain rounded stub"` *mentions limb endings*, inviting something to be drawn there | endings never mentioned; `"four bare straight strokes"` |
| **Dot faces** | not the wording — **face quality tracked FIGURE SIZE**. Identical expression clauses gave big eyes on medium shots and dots on wide shots | `"the figure drawn large and filling most of the height"` |
| **Amateur linework** | we were literally asking for it: `"crude"`, `"slightly wobbly"`, `"simple childlike drawing"` | `"bold minimal"`, `"thick black marker pen"` |
| **Realistic shaded brain** | trailing modifiers lost to FLUX's object prior | `"one simple flat shape ... no interior detail"` |
| Hair, gibberish text | one-off leaks | `"plain bald circle"`, `"no writing anywhere, no labels"` |

### What prompt work COULD NOT fix, after five phrasings each

- **the tan floor** — "band", "rectangle", "lower fifth", "fills the bottom",
  "solid flat" all gave a thin line, an orange scribble, or nothing, and
  *differently on different frames*.
- **the line weight** — "thick", "very thick heavy", "even width", "heavy
  outlines", "made with a thick black marker pen" all still rendered thin.

**Two failures in a row on the same attribute is the signal to stop asking.**

### So they moved to code — `pipeline/finish.py`

This is PLAN.md's own principle, which was written about exactly this case:

> *"Solid, because it is deterministic and done in Python, not by the model:
> the colour palette lock, the caption lettering … Do not touch the palette or
> caption code. Those are the parts that are working BECAUSE they are not the
> model's job."*

The captions look right in all 160 frames precisely because PIL draws them.
The floor is a rectangle; line weight is a morphological dilation. Both are
now code, both verified by eye on real frames:

```text
tan floor coverage   0.58%  ->  17.64%
dark (ink) pixels    1.19%  ->   1.77%
```

**The floor is anchored to the FEET, not to a fixed height.** The first version
used `FLOOR_TOP = 0.82` and painted over the figure's shoes, because feet land
lower on a wide shot than a medium one. `find_ground_line()` locates the rule
FLUX still draws, then walks *down* from it to the lowest ink within a bounded
window and puts the band under that. Measured band tops across three frames:
0.842 / 0.861 / 0.882 — i.e. genuinely per-frame.

**The floor was also REMOVED from the shot clause.** Leaving it in makes FLUX
draw a competing horizon at its own height underneath ours.

Benefits beyond looks: the floor is now **identical on every frame** (its
inconsistency was itself a drift source), it costs zero GPU, and it is tunable
without redrawing anything.

### A test that passed for the wrong reason

`test_pipeline.py` asserted `"tan" in SHOTS["character-moment"]["clause"]`, and
kept passing after the floor left the clause — because **"tan" is a substring of
"sTANding"**, which the new clause happens to contain. Replaced with six real
checks on `finish.py`. A test that passes on an accident is worse than none.

## colab-mcp disconnects: DIAGNOSED — it is the 1 MB WebSocket frame limit

Symptom: `get_cells` starts returning `Unknown tool`,
`open_colab_browser_connection` times out, everything recovers ~60 s later,
then dies again. Looks like a flaky network or a wedged server. **It is
neither, and it is not random.**

The server's own log gives the exact cause
(`%TEMP%\colab-mcp-logs-*\colab-mcp.*.log` — one dir per process, take the
one whose mtime is current):

```text
INFO:connection closed
ERROR:Task exception was never retrieved
  ColabWebSocketServer._read_from_socket ...
  ConnectionClosedError: sent 1009 (message too big)
  frame after reading 930553 bytes exceeds limit of 1048576 bytes
```

**colab-mcp's WebSocket caps a frame at 1,048,576 bytes.** A cell that
`display()`s an image puts that image into the cell output as base64. A
270 KB PNG becomes ~360 KB of base64, and with the cell source plus JSON
overhead the frame crosses 1 MB. The server then closes the socket with
code 1009 the moment it tries to read it.

Three disconnects this session, all the same, all immediately after touching
an image-bearing cell: 930,553 / 947,008 / 1,018,326 bytes. Deterministic.

**It also silently loses writes.** An `update_cell` that appeared not to
persist (the contact sheet kept running its old 4-row source) was a write
landing on a socket that was being torn down by an oversized frame.

### Rules

1. **Never `display()` an image in a cell you drive through colab-mcp.**
   Save it to a file instead.
2. **Move pictures over Drive, not the WebSocket.** Open the saved PNG in
   Drive directly, or copy it down.
3. **Print numbers, not pictures.** A pct-diff table is a few hundred bytes
   and always survives.
4. If you must pull an image through, keep the whole cell output under ~700 KB
   — a small thumbnail sheet, downscaled hard.

### Separately: disconnect the GPU before reading results

Fighting the above **while an L4 is connected costs ~4.8 units/hr (~0.08/min)**.
The reconnect saga this session plausibly cost more than the 24 frames it was
trying to read. The moment GPU work finishes, disconnect — before doing
anything else, including looking at output. Renders live on Drive and a
contact sheet over them is free on a CPU runtime.

### DELETE THE DRIVE MODEL CACHE — measured, it is strictly worse

Three paths to get FLUX's weights onto the GPU, all measured on the same L4 in
one session on 2026-08-25:

| path | result |
| --- | --- |
| **`snapshot_download` straight to `/content`** | **31.4 GB in 2.3 min = 238 MB/s** ✅ |
| `cp -a` Drive cache → `/content` | 16+ min at ~45 MB/s, **aborted unfinished** |
| `from_pretrained` reading the Drive cache | **stalled 22+ min, zero output, never finished** |

**Direct download is ~7x faster than copying off Drive, and it is the only one
of the three that completed.**

The Drive cache exists because Step 2's comment says it *"stops FLUX
re-downloading (~33 GB) after every disconnect"* — an assumption that
downloading is slow. **Nobody ever measured it.** It is not slow: it is the
fastest option by a wide margin, because Colab and Hugging Face are both
well-connected datacenters while Drive is a FUSE mount over the network.

So the cache costs 31.4 GB of Drive, adds the FUSE stall that burned ~3 units
this session, and saves nothing.

**Do this:**

- Set `HF_HUB_CACHE` / `HF_HOME` to `/content/hf_local` (local disk), never Drive.
- Let Step 1c's `prefetch_flux()` download there on first use, every session.
- Reclaim the 31.4 GB on Drive.
- Keep `HF_HUB_DISABLE_XET = 1` — cheap, and the Xet bug was a FUSE
  interaction that local disk should not hit anyway.

Cost is ~2.3 min per session and it never stalls. That is *better* than the
~6 min "cached" load it replaces.

**Downloading to the user's laptop instead does not help** and is much worse:
Colab cannot read the laptop's disk, so the weights would have to be uploaded
23+ GB over a home connection (single-digit MB/s, hours) to reach the VM.
Local weights only matter if FLUX runs on local hardware, which needs ~24 GB
of VRAM.

### And: Drive FUSE stalls on many small reads

Reading 36 PNGs one at a time off the FUSE mount stalled indefinitely
mid-loop (same family as the Xet bug above). `shutil.copyfile` them to
`/content/` first, then do all the reading on local disk.

### And: Pillow is broken on the CPU phase

Step 1a's `-U Pillow` lands on top of Colab's preinstalled copy and leaves
mixed files: `from PIL import ImageDraw` dies with
`cannot import name '_Ink' from 'PIL._typing'` (a new `ImageText.py` against
an old `_typing.py`). `PIL.Image` itself still works, so image *reading* and
*pasting* are fine — only `ImageDraw`/`ImageFont` are lost, which costs you
text labels on a contact sheet. Force-reinstall Pillow if labels are needed.
