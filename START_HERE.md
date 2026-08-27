# START HERE — paste this whole file into a new chat

Written 2026-08-25. This is the **current state and the current plan**.
`HANDOFF_NEXT.md` is the long history — read it only when you need the *why*
behind something below.

---

## The one-paragraph version

The pipeline works end to end. The character consistency problem is **solved**
(~2-4% drift on base, inside PLAN.md's acceptable band) and the first LoRA is
**abandoned** (it learned noise, proven by amplification). Three rounds of
prompt work fixed the oval torso, the dot eyes, the hair and the text leaks;
the floor band and the line weight were moved out of the prompt into
deterministic code (`pipeline/finish.py`). What remains is a **quality gap
against the reference channel** that FLUX.1-schnell cannot close on prompting
alone. The plan is a **second LoRA, trained properly** on 100 Nano Banana
images — the prompt list is written and waiting at
[channels/lora_prompts.md](channels/lora_prompts.md).

---

## THE STEPS — where we are, and what happens next

The whole project reduced to one sequence. Everything else in this file is
detail hanging off these steps.

```text
  ✅ DONE      1. pipeline works end to end
  ✅ DONE      2. character consistency solved (~2-4% drift)
  ✅ DONE      3. LoRA v1 killed (learned noise, proven by amplification)
  ✅ DONE      4. floor + line weight moved out of the prompt into code
  ✅ DONE      5. 100 training prompts written -> channels/lora_prompts.md
  ────────────────────────────────────────────────────────────────────────
  ⏳ NOW       6. USER generates the 100 Nano Banana images   <-- blocked on user
  ────────────────────────────────────────────────────────────────────────
  ⬜ NEXT      7. mirror finish.py into the notebook  (can run in parallel)
  ⬜ NEXT      8. train LoRA v2 on A100, 1500 steps
  ⬜ NEXT      9. check ||B|| convergence BEFORE drawing anything
  ⬜ NEXT     10. 9-frame test batch, look at it, compare to reference
  ⬜ THEN     11. fix two-character shots rendering one figure
  ⬜ THEN     12. build automated frame flagging
```

### Step 6 — NOW (user's action, not the agent's)

Generate 100 images from [channels/lora_prompts.md](channels/lora_prompts.md).
Square 1024×1024. Paste the `STYLE` block verbatim in front of every prompt.
Discard and regenerate anything that looks wrong — a bad image in the set
teaches the wrong figure.

**Nothing downstream can start until these exist.** Steps 8-10 are all
blocked on it.

### Step 7 — NEXT, and it does NOT need the images

The only work available in parallel while the user generates. `pipeline/finish.py`
exists and is tested, but the **notebook's drawing path never calls it** — so
the floor band and the line boost are not actually in any rendered frame yet.

- Mirror `finish.py`'s `boost_lines()` / `find_ground_line()` / `draw_floor()`
  into the notebook's `finish_frame()`, Step 20, cell `cell045`.
- Redraw the 9-frame test batch and check the band lands correctly on **every
  shot type**, especially wide and two-character.
- ~2 min of L4 time, ~0.05 units.

⚠️ The floor is currently validated on **3 frames upscaled from a JPEG strip**.
By this project's own standard that is not validated. Step 7 is what makes it
real.

### Steps 8-10 — after the images land

Full detail in *THE ACTIVE TASK: LoRA v2* below. The short form:

1. `lora_data/` ← the 100 images; Step 21d pads + palette-locks them.
2. Move caption `.txt` files out (pre-flight 3, cell `O7IVBdeNbv8W`).
3. Train on A100, `--max_train_steps 1500`, rank 16, LR 1e-4.
4. **`||B||` movement under ~10% before you draw anything** (cell `ATBXCaqKKg7s`).
   v1 was still moving 71% and that was the tell.
5. `config.LORA_TRIGGER = "stkmn style"`, `rebuild_prompts.py`, 9-frame batch,
   **look at it**.

**Decision point at step 10:** if v2's linework still doesn't reach the
reference, the problem is the base model, not the training — and the next move
is a different base, not a third LoRA. Do not spend a third training run
chasing the same gap.

### Steps 11-12 — after the LoRA question closes

Both are prompt/code work, no training. Detail in *The other open work* below.

---

## Where things stand

| area | status |
| --- | --- |
| Pipeline end-to-end | ✅ works. 10/10 frames, 1280x720, 0 fallbacks, 7.9 s/frame on L4 |
| Palette lock | ✅ 10/10 palette-clean, code-owned, never touch it |
| Captions | ✅ burned in by PIL, correct in all frames, code-owned |
| Character consistency | ✅ **~2-4% drift, measured on 45 frames / 44 action clauses.** Not a problem any more |
| LoRA v1 | ❌ **abandoned.** Learned a degenerate direction. `LORA_TRIGGER = ""` |
| Line weight | ✅ fixed in code — dilation, `LINE_BOOST_PX = 3` |
| Tan floor band | ✅ fixed in code — foot-anchored, `finish.py:find_ground_line()` |
| Line *smoothness* | ❌ **open.** Dilation makes lines thick, not clean. Generation problem |
| `two-character` shots | ❌ **open.** Often renders ONE figure. ~18 of 160 frames |
| Tests | ✅ `test_pipeline.py` 67 pass, `test_notebook.py` 13 pass |

---

## The three things a new chat must NOT redo

These cost real money and are settled. Do not re-litigate them.

### 1. Do not train LoRA v1 further

It is not undertrained — it is **wrong**. Fused at 8x it *erases* the drawing;
at 15x it renders pure orange noise. A LoRA that learned a character would draw
a *stronger* character when amplified. `||B||` moved 71% in the last 250 steps,
so it never converged, but more steps along a bad gradient buys nothing.

```text
base        mean  0.00%   max  0.00%   <- sanity check
s500@0.85   mean  0.96%   max  2.44%
s500@3      mean  1.93%   max  4.56%
s500@8      mean  4.49%   max 11.59%   <- drawing erased
s500@15     mean 36.62%   max 38.33%   <- pure noise
```

### 2. Do not use the Drive model cache — download direct

Measured on one L4, same session:

| path | result |
| --- | --- |
| `snapshot_download` → `/content` | **31.4 GB in 2.3 min = 238 MB/s** ✅ |
| `cp -a` Drive cache → `/content` | 16+ min @ ~45 MB/s, aborted unfinished |
| `from_pretrained` off the Drive cache | **stalled 22+ min, never finished** |

Set `HF_HUB_CACHE` / `HF_HOME` to `/content/hf_local`. Keep
`HF_HUB_DISABLE_XET = 1`. Reclaim the 31.4 GB on Drive.

Downloading weights to the laptop does **not** help — Colab can't read the
laptop's disk, and uploading 31 GB over a home connection takes hours.

### 3. Never `display()` an image through colab-mcp

Every "random disconnect" was this. colab-mcp's WebSocket caps a frame at
**1,048,576 bytes**; a `display()`d image blows it and the server closes with
code 1009. Three disconnects, all deterministic: 930,553 / 947,008 / 1,018,326
bytes. **It also silently eats `update_cell` writes.**

Rules: save pictures to a file, move them over Drive, print numbers not
pictures. If you must pull an image through, keep the cell output under ~700 KB.

---

## Cost model — corrected, use these numbers

**Compute units bill by WALL-CLOCK, not GPU use.**

| GPU | units/hr | verdict |
| --- | --- | --- |
| **L4** | **1.54 (measured on the session readout)** | ✅ use for all generation. 24 GB keeps FLUX resident |
| A100 | ~15 (unverified) | training only. 68% worse than L4 for generation — fixed overhead doesn't speed up but does cost more |
| T4 | ~1.8 (unverified) | ❌ **cannot run this pipeline.** 16 GB forces CPU offload → fills the 12.7 GB RAM → crash |
| TPU | — | ❌ 4-bit NF4 via bitsandbytes is CUDA-only |

An older version of this doc said 4.8/hr for L4 — **3x too high**, and several
panicky decisions were made under that wrong number. At 1.54/hr, ten idle
minutes is ~0.26 units. Still waste, not an emergency.

**Per video on L4: ~3.1 units → ~31 videos/month.** Phase 2 (review, encode,
upload) is free on the CPU runtime.

### Standing operating pattern (the user's, and it is right)

**Draw → disconnect the GPU → diagnose offline.** The moment GPU work
finishes: `from google.colab import runtime; runtime.unassign()`. Renders live
on Drive; contact sheets and diffs are free on a CPU runtime. Edit cells
disconnected; connect only to run.

---

## THE ACTIVE TASK: LoRA v2

### Why a LoRA at all, when consistency is solved

Consistency and **quality** are different problems, and the drift test only
measured the first. Side by side with the reference channel, ours reads as a
thin scratchy sketch and theirs reads as a confident bold marker drawing. Two
frames can be perfectly consistent with each other and both look amateur.

FLUX.1-schnell will not close that gap by prompting. It is 4-step distilled and
guidance-distilled (no working CFG), with a photographic prior. Five phrasings
each failed on line weight and on the floor. The reference frames were made with
Nano Banana, and the way to get Nano Banana's linework out of FLUX is to
**train on Nano Banana output** — which is the user's own generation, so no
licensing problem.

### What the user is doing now

Generating **100 images** from [channels/lora_prompts.md](channels/lora_prompts.md).
That file is complete and self-contained: a verbatim `STYLE` block to prefix
every prompt, the rules, 100 prompts in 5 groups mirroring the real shot
distribution, and the post-generation training settings.

| group | count | mirrors |
| --- | --- | --- |
| A. one figure, body actions | 30 | `character-moment`, 56% of real frames |
| B. close on the face | 15 | fixes dot-eyes on wide shots |
| C. one giant symbolic object | 20 | `symbolic-metaphor`, 24% |
| D. two figures | 20 | `two-character`, 20% — and the defect |
| E. one small prop | 15 | props |

**Output square 1024×1024, not 16:9.** Training runs at `--resolution 768`
(square) so anything else is padded anyway; a padded 16:9 source is ~44% blank
bars; square gives the linework more pixels, which is the exact problem being
solved. The aspect mismatch is harmless — a character LoRA learns what the
figure *looks like*, not where it sits in frame.

### What changes from the run that failed

| | v1 (failed) | v2 |
| --- | --- | --- |
| images | 23 | **100** |
| variety | 4 distinct poses | 100 varied poses/props/framings |
| steps | 500 | **1500–2500** |
| rank | 16 | 16, drop to 8 if it overfits |
| LR | 1e-4 | 1e-4 (fine at this volume) |
| convergence check | none | **norm-B movement under ~10% before drawing anything** |

**Do NOT duplicate the 23** — a duplicate gives an identical gradient, so it is
mathematically the same as training longer on 23, which is what produced noise.
**Do NOT import Kaggle/open-source stick-figure sets** — they teach "stick
figures in general", which FLUX already knows, and reintroduce the licensing
risk the Nano Banana set was chosen to avoid.

### The training run, once the 100 images exist

1. Put them in `lora_data/`. Step 21d's prep cell pads to square and runs
   `lock_palette()` over them.
2. **Move caption `.txt` files out before training.** diffusers 0.39.0's
   `DreamBoothDataset` iterates the directory with no extension filter and a
   caption file crashes it with `UnidentifiedImageError`. Pre-flight 3 in cell
   `O7IVBdeNbv8W` does this.
3. Launch on **A100** (cell `O7IVBdeNbv8W`), `--max_train_steps 1500`.
4. **Check `||B||` movement between checkpoints before drawing anything.**
   Cell `ATBXCaqKKg7s` prints it.
5. Set `config.LORA_TRIGGER = "stkmn style"` (it is `""` now), re-run
   `rebuild_prompts.py`, draw the 9-frame test batch, **look at it**.

Three launch failures in v1, all dependency/data, all now pre-flights that run
as a 5-second CPU probe *before* touching the GPU:

- **diffusers gate** — the `main` script demands 0.40.0.dev0; we pin
  `>=0.35,<0.40` because the FLUX loader needs it. Fetch the script from the
  matching release tag; do **not** upgrade diffusers.
- **Colab's stale `torchao` 0.10.0** — peft's dispatcher *raises* on an old
  copy instead of returning False, blocking all LoRA injection. **Uninstall
  it, don't upgrade it** (torchao>=0.16 pulls a newer torch).
- **Caption files in `lora_data/`** — see 2 above.

---

## The other open work, in priority order

1. **Mirror `pipeline/finish.py` into the notebook's `finish_frame()`**
   (Step 20, cell `cell045`). The post-process exists in the package but the
   notebook's drawing path does not call it yet. ~2 min, ~0.05 units.
2. **Validate the floor band on real full-res frames** and on wide /
   two-character shots. Currently validated on 3 frames upscaled from a JPEG
   strip — which is not good enough by this project's own standard.
3. **Fix `two-character` rendering one figure.** The clause says *"two figures
   facing each other on the same floor line ... empty space between them"* and
   mostly shows one. Seen on frames 16, 43, 45, 47, 53 of the drift batch;
   9 such shots in 80 scenes → ~18 of 160 in a real video. Prompt-level, costs
   nothing to iterate with `run_scenes.py --shots`. Largest known visual defect.
4. **Build automated frame flagging.** At 31 videos/month × 160 frames that is
   ~5,000 frames a human has to scan to find the few bad ones. Per-shot ink
   coverage, figure height, connected components, off-palette pixels — surface
   only the suspicious ones. Turns "review 160" into "review the 12 flagged".
   ⚠️ **It must compare within shot type** — see the trap below.
5. Content-level: frame 0's lung reads as wings; "a single muscle fiber"
   renders as a blob; `object-focus`'s anti-anatomical `ask` is a nudge, not
   enforced.

---

## Traps that have already cost money

**A redraw must change the seed or the prompt.** FLUX at a fixed seed is
deterministic. `SEED_PER_FRAME = False`, so every frame is seed 42 — redrawing
a bad frame unchanged returns the identical bad frame.

**Never compare ink coverage across shot types.** The first automated drift
pass reported 20% outliers and would have said "blocking". Every flagged frame
was `symbolic-metaphor`, whose prompt *demands* "one large symbolic object that
dwarfs it". The metric was measuring the design. Within shot type: 11.1%. By
eye: 2-4%.

**Drive FUSE stalls on many small reads.** Reading 36 PNGs one at a time off
the mount stalled indefinitely. `shutil.copyfile` to `/content/` first, then
read locally.

**Pillow is broken on the CPU phase.** Step 1a's `-U Pillow` leaves mixed
files: `from PIL import ImageDraw` dies with `cannot import name '_Ink' from
'PIL._typing'`. `PIL.Image` still works, so reading and pasting are fine — you
just lose text labels on a contact sheet. Force-reinstall if you need them.

**A test can pass for the wrong reason.** `assert "tan" in
SHOTS["character-moment"]["clause"]` kept passing after the floor left the
clause, because **"tan" is a substring of "sTANding"**. Replaced with 6 real
`finish.py` checks. A test that passes on an accident is worse than none.

**colab-mcp's `run_code_cell` has a 30s client-side timeout that is NOT the
cell timing out.** A FLUX load takes minutes; the tool call errors while the
cell keeps running. Do not re-run blind — poll `get_cells` **by cell id, not
index** (index drifts when you insert cells) until real output appears.
tqdm widgets don't update in a snapshot; a bar frozen at 0% is not proof of a
hang — wait for an absence of new stdout.

**Why schnell, not dev:** FLUX.1-schnell is Apache-2.0. **FLUX.1-dev is
non-commercial only** — wrong for a monetized channel.

---

## Key files

| file | what it is |
| --- | --- |
| [channels/lora_prompts.md](channels/lora_prompts.md) | **the active task** — 100 training prompts + settings |
| [channels/pipeline/config.py](channels/pipeline/config.py) | every constant, with the reasoning for each in comments |
| [channels/pipeline/finish.py](channels/pipeline/finish.py) | floor band + line boost, numpy/PIL only, no torch |
| [channels/pipeline/shots.py](channels/pipeline/shots.py) | 7 shot types, composition rotation |
| [channels/pipeline/look.py](channels/pipeline/look.py) | 6 expressions, chosen by lexical energy score |
| [channels/test_pipeline.py](channels/test_pipeline.py) | 67 tests |
| [HANDOFF_NEXT.md](HANDOFF_NEXT.md) | the long history and every measurement |
| `PLAN.md`, `channels/README.md`, `channels/TESTING_VISUALS.md` | the originals |

**Colab cell ids (this notebook instance):**
`cell002` 1a · `cell004` 1b · `cell006` 1c · `cell008` 2 · `cell011` 3 ·
`cell045` 20 (style lock) · `cell056` 23 (draw) · `o2pvZHPgP5Lh` 21d ·
`O7IVBdeNbv8W` 21e (train) · `PoehYu0BlDJP` (resume) · `ATBXCaqKKg7s` (‖B‖)

---

## The standing lesson

**Never judge a visual change from prompt text or summary statistics. Draw a
small batch and look at it.** Ten frames cost $0.0004 in planning and ~90
seconds of GPU. Five times this project a change looked right in JSON and was
wrong in the render; twice a *metric* declared success while the picture
disagreed; once a whole LoRA was trained on a premise that had already been
fixed upstream and nobody redrew the base model to check.

So the lesson generalises: **re-check the PREMISE after you change anything
upstream of it.** A justification measured before a fix is not evidence about
the world after the fix, and every hour downstream of a stale premise solves a
problem you may no longer have.

And the corollary that keeps paying: **the parts of this pipeline that look
right in all 160 frames are the parts FLUX does not touch.** The palette lock
is code. The captions are code. The floor and the line weight are now code.
Everything left to the prompt varies. When two phrasings in a row fail on the
same attribute, stop asking and write the Python.
