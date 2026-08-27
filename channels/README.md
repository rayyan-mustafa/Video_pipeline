# The split: what runs where

`stickman.ipynb` used to do everything. Steps 6–15 now run locally as plain
Python, and the notebook keeps the GPU work. They talk through **files in one
shared folder**, which is how the notebook's stages already talked to each other.

```text
LOCAL (laptop)              COLAB PHASE 1 (T4)        COLAB PHASE 2 (CPU, free)
  run_script.py  6-15          0 1a 1b 1c setup          0 1a 1b  light setup
  vo_coach       record        2 3 4 5    load script    2 3 5    load script
  run_scenes.py  22            16-19  voice + Whisper    24   review frames
                               20 21  look + character   25   x264 encode
                               23     draw 160 frames    27   archive
                               26     thumbnail          28   upload
                               24b    redraw the queue
```

Nothing in `pipeline/` imports torch, diffusers or faster-whisper. That is a hard
rule — the moment one module does, the local half stops being installable on the
laptop and the split is gone.

## Setup, once

```powershell
cd f:\wierd_bio\channels
.venv\Scripts\python.exe -m pip install -r requirements-cpu.txt   # already done
```

### The key

**Everything routes through OpenRouter on one key** — Claude and anything else
you swap in.

```ini
OPENROUTER_API_KEY=sk-or-v1-...
```

Confirm the key and every model slug for free before spending anything:

```powershell
.venv\Scripts\python.exe run_script.py --preflight
```

### Measured cost (2026-08-23, real runs)

| | calls | tokens | cost |
| --- | --- | --- | --- |
| `run_script.py --models cheap` | 59 | 154k in / 22k out | **$0.0093** |
| `run_scenes.py --models cheap` (29 scenes) | 32 | 10k in / 0.7k out | **$0.0005** |
| **one video, cheap** | | | **~$0.010** |
| one video, Claude preset (projected) | | | ~$0.80 |
| 100 videos/month, Claude | | | ~$85 |

Set the key's credit limit to at least **$5**. A limit of $0.01 does not cover a
single run; $0.00001 was exhausted by four calls of the word "OK".

Point both halves at the same folder. Install **Google Drive for Desktop**, then:

```powershell
setx VIDEO_PIPELINE_WORKDIR "G:\My Drive\videos\video_pipeline"
```

That is the same folder the notebook's Step 2 already uses
(`/content/drive/MyDrive/videos/video_pipeline`). No zipping, no manual upload.

## Run it

```powershell
.venv\Scripts\python.exe run_script.py --preflight   # key + model slugs, free
.venv\Scripts\python.exe run_script.py --list        # the 15 topics
.venv\Scripts\python.exe run_script.py --topic goosebumps
.venv\Scripts\python.exe run_script.py --status      # which stages have run
.venv\Scripts\python.exe test_pipeline.py            # 57 checks, no API calls
.venv\Scripts\python.exe test_notebook.py            # 13 checks on the notebook cells
```

Then record, one section at a time, and check it before you pay for a runtime:

```powershell
# what to read is in sections/README.txt, written by run_script.py
python -m vo_coach mark-script sections/section_01.txt --niche zenn
python -m vo_coach live --script sections/section_01_marked.txt --niche zenn
.venv\Scripts\python.exe collect_takes.py --section 1   # take -> section_01.wav
.venv\Scripts\python.exe run_qc.py                      # before you open Colab
```

`--preflight` also runs automatically at the start of a real run, so a bad slug
or a dead key costs nothing instead of dying after the source fetch.

Useful flags: `--no-cache` (refetch sources), `--no-hook-tournament`,
`--hooks N`, `--workdir PATH`.

Cost is a few cents of Claude calls and zero GPU time. Source fetches are cached
per-URL on disk, so iterating on a script is free after the first run.

## The file contract

This is the whole interface between the two halves. Each row is written by one
side and read by the other.

| file | written by | read by |
| --- | --- | --- |
| `source_pack.json` | local, step 6 | local steps 7–15 |
| `evidence_profile.json` | local, step 7 | local steps 8–15 |
| `script_tagged.json` | local, steps 8–14 | **notebook step 17** (scene boundaries) |
| `vo_raw.txt` | local, steps 8–14 | you, as one continuous read |
| `sections/section_NN.txt` | local, step 15b | **you and `vo_coach mark-script`** |
| `sections/manifest.json` | local, step 15b | `run_qc.py`, `collect_takes.py` |
| `voice_recordings/_takes.json` | **local, `collect_takes.py`** | you — which take is which section |
| `packaging.json` | local, step 15 | **notebook steps 26, 28** |
| `youtube_description.txt` `youtube_tags.txt` `pinned_comment.txt` | local, step 15 | you, at upload |
| `voice_recordings/section_NN.wav` | vo_coach, via `collect_takes.py` | `run_qc.py`, then **notebook step 17** |
| `vo_raw.wav` `scenes_timeline.json` | notebook step 17 | notebook steps 18, 22 |
| `full_vo.wav` | notebook step 18 | notebook step 25 |
| `beat_sheet.json` | local, step 7 | **notebook step 5** (Section 3 name + word targets) |
| `style.json` | local, steps 7 / 22 | **notebook step 3** (keeps Redraw in sync — prompt constants, the shot catalog **and the expressions**) |
| `scenes_planned.json` | **local, `run_scenes.py`** | notebook steps 23–25 |
| `frames/` | notebook step 23 | notebook step 25 |
| `final_video.mp4` | notebook step 25 | you |

`_stages.json` sits alongside them and records the size of every output.
`paths.require("script")` checks those sizes back, because **Google Drive sync is
not atomic** — a half-uploaded file still passes `os.path.exists()`, and that
failure surfaces much later as a drift error at mux time, after the frames are
already paid for.

## The two Colab phases

Colab bills compute units by **wall-clock on the runtime, not by GPU use**. A
minute of ffmpeg on a T4 costs exactly what a minute of FLUX costs. Two steps
never touch the card:

| | wall-clock | units / 60 videos |
| --- | --- | --- |
| Step 24 frame review | ~15 min of human clicking | **27.3** |
| Step 25 x264 encode | ~12 min of pure CPU | **21.8** |

At 60 videos the whole run was **~135 units against a 100-unit budget**. Moving
those two to a free CPU runtime brings it to **~85** — under budget, with room
for Whisper `large-v3`.

**Step 0** detects the runtime (via `nvidia-smi`, not torch — torch is never
installed on the CPU phase) and prints the checklist for whichever one you are
on. Work order:

```text
Phase 1  T4 GPU   0 1a 1b 1c · 2 3 4 5 · 16-19 voice · 20 21 · 23 draw · 26 thumbnail
                  → Runtime → Change runtime type → CPU
Phase 2  CPU      0 1a 1b · 2 3 5 · 24 review · 25 encode · 27 28
                  → only if the review queued redraws: one short GPU session for 24b
```

Four things make that work:

- **Step 1a installs a light stack on CPU** — Pillow, numpy, soundfile, ffmpeg,
  ipywidgets. No torch, no diffusers, no bitsandbytes. ~1 minute instead of ~4,
  and no runtime restart, because the restart only exists to fix transformers
  caching "is bitsandbytes here?" at import.
- **Step 1c defines stubs on CPU** so the cells that call `release_flux()`
  defensively keep working, and `get_flux()` explains itself instead of throwing
  `NameError`.
- **Step 24's Redraw queues instead of drawing.** Flagged frames go to
  `redraw_queue.json`; **Step 24b** draws them all in one GPU session. That is
  cheaper than the old behaviour even ignoring the phase split — FLUX loads once
  for the batch rather than once per frame you happen to dislike.
- **Thumbnail (26) moved before encode (25)**, so the GPU work is contiguous and
  the phases cut at exactly one point.

`test_notebook.py` executes the real cell sources with `HAS_GPU` forced both
ways — 13 checks, so a phase guard cannot rot silently.

## ⚠️ The ordering constraint

From `vo_coach/STATUS.md`:

> The stickman notebook's Step 18 locks its length so the pictures stay in sync.
> Anything that changes duration — comping, `fix-pauses` — **must run before that
> step**, or the images desync.

Splitting the pipeline puts more distance between those two points, so treat this
as a real hazard, not a note.

## The recording seam

Three files sit between the script and the notebook. They exist because that
seam used to be you, moving and renaming things by hand.

**`sections/section_NN.txt` — what to read.** Written by `run_script.py` (step
15b) from `pipeline/readsheet.py`. The text is grouped by the notebook's own
`build_chunks()`, **not** by the `[S<n>|…]` tags, and that distinction is the
whole point: `build_chunks` merges a short sentence into the chunk that is
already open, so a three-word line at the top of section 4 can belong to a chunk
the aligner files under section 3. Group by tag and you would read it in the
wrong section, the difflib match rate in both would drop, and their scene
boundaries would quietly become interpolated guesses.

The chunker is therefore a verbatim copy, and `test_notebook.py` executes the
real Step 17 cell and asserts the two agree. A copy that drifts is worse than no
copy at all.

**`collect_takes.py` — which take is which section.** `vo_coach live` writes
`live_TIMESTAMP_comped.wav`; the notebook wants `section_NN.wav`. Doing that
rename by hand six times is one typo away from section 4 holding section 5's
audio, and **nothing downstream raises** — Whisper transcribes it happily,
difflib matches almost nothing, and the first symptom is drift in a finished
video. So it refuses the two silent mistakes (overwriting a section that already
has audio, using one take for two sections) and records provenance in
`voice_recordings/_takes.json`.

**`run_qc.py` — is this audio worth a T4 session?** Peak level, runs of
full-scale samples, noise floor and SNR, duration against the read sheet's word
count, and the cross-section checks (missing files, mismatched rates, level
jumps). No model, no GPU, no network — seconds to run.

It exists because the first thing that currently notices bad audio is Whisper,
in Phase 1, on a paid runtime, and it does not *report* it. Clipping is
permanent, and it propagates:

```text
clipped audio -> worse transcript -> difflib matches fewer words
             -> scene boundaries land wrong -> frames drift against your voice
```

The thresholds are vo_coach's, not new ones: clipping is **runs** of full-scale
samples (≥ 0.99, three or more consecutive — one sample at full scale is not
clipping, and treating it as such once stopped a reader on line one), and the
noise floor excludes digital silence, because a gated file otherwise reports a
~175 dB SNR and reads as flawless. `pipeline/qc.py` parses the RIFF chunks
itself rather than importing soundfile: the standard library's `wave` refuses
float WAVs, which is exactly what vo_coach writes at 32-bit, and the local half
stays installable with numpy as its only numeric dependency.

Pace is checked only widely enough to catch the wrong file (70–260 wpm fails,
110–200 warns). Judging delivery is vo_coach's job and needs the calibration
takes, which do not exist yet.

## The notebook surgery

**Done.** `stickman.ipynb` went from **71 cells to 50**. Removed:

- cells 18–39 — Steps 5–15 (engine, sources, evidence, writer, lint, fact-check,
  grounding loop, retention, hook tournament, pacing, packaging)
- cells 57–58 — Step 22 (plan the pictures)

A backup sits at `stickman.ipynb.bak`.

Three surviving cells had read variables the deleted cells left in the kernel,
and were rewired to read files instead:

| cell | needed | now |
| --- | --- | --- |
| Step 17 voice | `tagged_sentences` | loaded at Step 5 |
| Step 19 QA | `master_script` | loaded at Step 5 |
| Step 27 archive | `profile`, `BEAT_SHEET[3]`, `source_pack`, issue counts, `USAGE` | read from disk; counts from the last `validation_round_*.json` |

**New Step 5 — "Load the script"** reads `script_tagged.json`, `vo_raw.txt`,
`evidence_profile.json`, `source_pack.json` and `beat_sheet.json`, and rebuilds
every name the deleted cells used to leave behind. It is strictly better than
what it replaced: after a Colab reconnect you run one cell and carry on, instead
of re-running fifteen.

It also asserts `script_tagged.json` and `vo_raw.txt` are **byte-identical**.
`run_script.py` writes the second as exactly `" ".join(sentences)` from the
first, so any difference means either Drive is mid-sync or one was hand-edited —
and a hand-edited `vo_raw.txt` desyncs the pictures, because Step 17 aligns your
voice against the *tagged* file while you read from the plain one.

Also removed from the notebook: `anthropic` and `trafilatura` from the Step 1a
pip list (nothing there calls a text model or fetches a page now), and the three
`MODEL_*` constants from Step 3 — dead *and* wrong, since they held Anthropic
API ids rather than OpenRouter slugs.

### The style drift hazard — closed

`CHARACTER_BIBLE` and `STYLE_PREFIX` live on both sides, and that was a live bug,
not merely duplication:

- `run_scenes.py` bakes them into every scene's `visual_prompt`, and Step 23
  draws from that field — so the **package** copy paints all 150 frames.
- But **Step 24's Redraw button rebuilt the prompt from the notebook's copy.**

Edit one and not the other and a redrawn frame silently stops matching the other
149, because both prompts look perfectly reasonable on their own.

Fixed the same way `beat_sheet.json` was: the authoritative values cross as
**data**. `run_script.py` and `run_scenes.py` write `style.json`; notebook Step 3
defers to it when present and says which copy is in use:

```text
Style: using style.json from the laptop (identical to this cell's copy)
Style: using style.json from the laptop (differs from this cell's copy - the laptop wins)
Style: no style.json yet - using this cell's own constants.
```

Data rather than code, because shipping the package to Colab means solving
`sys.path` there, and every other stage boundary here is already a file. The
notebook's own definitions stay as the fallback so it still runs standalone.

`test_pipeline.py` now parses the real Step 3 and Step 24 cells out of the
notebook and asserts the Redraw prompt is **byte-identical** to what
`run_scenes.py` bakes in — so this cannot quietly reopen.

## What changed coming out of the notebook

Three notebook globals mutated themselves after import — harmless when cells run
once top-to-bottom, quietly wrong when anything runs twice in one process:

1. `BEAT_SHEET[3]` was assigned at runtime from the evidence profile, and
   `BANNED_PHRASES` was rebound to itself plus the learnings file. Two topics in
   one process would carry topic A's Section 3 into topic B.
   → `config.build_beat_sheet()` and `config.banned_phrases()` return fresh
   objects. A test asserts the module-level originals are never touched.
2. `LEARNINGS` loaded at import from a bare relative path, depending on
   `os.chdir()` having already happened. → `config.load_learnings()`, explicit.
3. Six prompt fragments (`sources_block`, `numbers_block`, `entities_block`,
   `banned_block`, `profile`, `BEAT_SHEET`) were set in one cell and read in four
   others. → all bundled into `Brief`, so a function's inputs are its arguments.

Two things were added that the notebook did not need:

- `paths.force_utf8_stdout()`. A Windows console is cp1252 and raises
  `UnicodeEncodeError` on the first curly quote a model returns — *after* the
  paid call has succeeded. Colab's stdout is UTF-8, so this bug cannot appear
  there. `vo_coach` hit the same thing and fixed it the same way.
- `engine.client()` is lazy. The notebook did `client = anthropic.Anthropic()` at
  module level, which raises on import if no key is set.

Everything else is a verbatim lift, comments included. Those comments document
real failures and are worth more than the lines they sit above.

## Routing through OpenRouter

`engine.py` is the `openai` SDK pointed at `openrouter.ai/api/v1` — the same
shape `vo_coach/src/vo_coach/mentor/providers.py` already uses, so both halves of
the project share one key, one bill and one set of habits. Three rules carried
over from `vo_coach`'s STATUS.md:

**Host pinning is mandatory, not tuning.** `provider.order = ["anthropic"]` with
`allow_fallbacks: false`. The regional tags cost 10% more for the identical
model (`$2.20/$11.00` on `google-vertex/us` vs `$2.00/$10.00` on `anthropic`),
and unpinned routing makes a run's cost unreproducible for no benefit.

**Verify every slug against the endpoints API**, never the providers tab:

```text
https://openrouter.ai/api/v1/models/<model>/endpoints
```

That rule caught a real bug here. The notebook's `MODEL_CHECKER` was
`claude-haiku-4-5-20251001` — the Anthropic API's dated id, which **OpenRouter
does not route**. The slug is `anthropic/claude-haiku-4.5`. With fallbacks off
this would have hard-failed on the evidence profiler, which is the *first* paid
call in the run, so nothing would have worked at all. `--preflight` now checks
every slug and pin before a run spends anything.

**Retry only what a retry can fix.** The notebook retried every exception four
times with exponential backoff. That is right for a 429 or a 529 overload and
wrong for anything permanent — a bad key or a bad slug fails identically every
time, so the backoff just burns 45 seconds per call before reporting the same
error. Measured: a 401 took 45s to surface, now 5s.

Model prices via OpenRouter, per million tokens (input/output):

| slug | role | price |
| --- | --- | --- |
| `anthropic/claude-sonnet-5` | writer, judge | $2 / $10 (vs $3/$15 direct) |
| `anthropic/claude-haiku-4.5` | checker | $1 / $5 |
| `anthropic/claude-opus-5` | — | $5 / $25 |

Cost accounting prefers OpenRouter's **actual billed cost** (returned per call
via `usage: {include: true}`) and falls back to the local price table only when
that is absent — the table drifts the moment pricing changes.

**Swapping in other models** is a slug edit plus a pin. Pins are **per model**,
in `config.PROVIDER_PINS` — a model with no entry is *refused*, not routed
freely, because unpinned routing is how a run silently costs several times what
the last one did.

### Five roles, and why each got the model it did

The split is by **what the output is**, not by how hard the task feels.

| role | model | calls | why |
| --- | --- | --- | --- |
| `writer` | **Sonnet 5** | ~9 | prose a viewer hears. No substitutions. |
| `judge` | **Sonnet 5** | ~19 | judgment. A weak judge fails *invisibly*, by passing everything. |
| `extractor` | **Sonnet 5** | 1 | ← upgraded from Haiku |
| `checker` | **Haiku 4.5** | ~8 | mechanical matching against text in the prompt |
| `scenes` | **DeepSeek V4 Flash** | ~160 | ← downgraded from Haiku |

**`extractor` went up.** One call per video, and it decides everything
downstream: `hard_numbers` is the writer's entire statistics allowance, and
`evidence_type` picks the whole shape of Section 3. A number it misses is a fact
the script can never use, and nothing reports the loss. ~$0.01 a video — the
cheapest quality insurance here.

**`scenes` went down.** Highest volume by 8x and the most constrained task in the
pipeline: one sentence → 5–12 drawable words plus an optional 5-word caption.
Measured on real scene prompts, DeepSeek V4 Flash with reasoning off answers as
well as Haiku for **1/37th** the price ($0.0024 vs $0.088 per video), and Step 24
lets you redraw anything odd.

**`checker` deliberately stayed on Haiku.** Dropping it to DeepSeek would save
another ~$0.05, but this is the gate that stops a false claim reaching a viewer.
Not the place to save five cents.

**`judge` and `writer` are not negotiable.** `vo_coach`'s golden-set work is the
argument: a cheap model on a judgment task does not fail loudly, it approves
everything. Whether a cheaper judge is *actually* good enough is measurable —
run the same takes through both and compare — but it is not assumable.

### Projected cost per video

| role | model | cost |
| --- | --- | --- |
| writer | Sonnet 5 | $0.226 |
| judge | Sonnet 5 | $0.130 |
| checker | Haiku 4.5 | $0.054 |
| extractor | Sonnet 5 | $0.023 |
| scenes | DeepSeek V4 Flash | $0.003 |
| **total** | | **~$0.44** |

Against **~$0.63** with every role on Sonnet — a **30% saving with Sonnet kept
on everything a viewer experiences**. At 100 videos/month: ~$44 vs ~$63.

Projected by scaling the measured `--models cheap` token profile up to a
full-length script; not yet measured on Claude.

`--models all-sonnet` runs the control: if a `claude` run ever looks wrong, that
tells you whether a cheap role caused it.

### The bigger lever still on the table

`apply_fixes` re-sends the beat sheet, the full source block, the numbers and
the entities on **every** call — ~7k tokens, 5–8 times per run. Anthropic prompt
caching through OpenRouter would cut ~90% off that, worth roughly **$0.09 a
video** — more than the scene-model change. Not implemented.

### Model presets

```powershell
.venv\Scripts\python.exe run_script.py --topic goosebumps --models cheap
.venv\Scripts\python.exe run_scenes.py --models cheap
```

`cheap` puts all three roles on **`inclusionai/ling-3.0-flash`**, so the whole
pipeline can be exercised end to end for roughly a cent. It proves the *wiring*,
not the writing — a small model will sail through gates a strong one fails, so
never judge a script produced under it.

Three hosts serve Ling, and the pin is deliberate:

| tag | price / Mtok | |
| --- | --- | --- |
| `novita` | $0.021 / $0.063 | cheapest |
| **`novita/fast`** | **$0.042 / $0.126** | **pinned** |
| `deepinfra/bf16` | $0.060 / $0.180 | |

Not the cheapest, on purpose. `vo_coach`'s own notes measure plain Novita at
about a tenth of Fast's speed, and a full script run emits ~22k output tokens —
roughly six minutes of extra waiting to save half a cent. Cheapest is the wrong
optimisation when the whole run already costs about $0.01.

### Ling reasons, and that had to be measured

`inclusionai/ling-3.0-flash` is a **reasoning** model, which is not obvious from
its name and is not what `vo_coach`'s notes assume. On a real scene-planning
prompt at `max_tokens=600`:

| setting | out | reasoning | visible | $/call |
| --- | --- | --- | --- | --- |
| (unset) | 183 | 167 | 16 | 0.0000355 |
| `effort: low` | 197 | 172 | 25 | 0.0000373 |
| `effort: high` | 239 | 221 | 18 | 0.0000425 |
| `reasoning: {exclude: true}` | 169 | 164 | 5 | 0.0000335 |
| **`reasoning: {enabled: false}`** | **23** | **0** | **23** | **0.0000151** |

Two things only measurement would have told you:

1. **Effort does not help.** `low` thinks slightly *more* than unset. Pinning it
   the way `vo_coach` pins it for `gpt-oss-120b` would have achieved nothing —
   the lever is model-specific, not general.
2. **`exclude: true` is a trap.** It hides the reasoning from the response and
   still bills all 164 tokens of it. Only `enabled: false` actually stops it.

Turning it off is 2.4x cheaper and the answer is just as good — this is scene
captions and fact-matching, not analysis. Set per model in `config.REASONING`.

## Layout

``````text
channels/
  stickman.ipynb          GPU half (unchanged so far)
  run_script.py           steps 6-15 driver
  run_qc.py               the QC gate - measure the recordings before Colab
  collect_takes.py        vo_coach take -> voice_recordings/section_NN.wav
  requirements-cpu.txt    no torch, deliberately
  pipeline/
    paths.py              workdir resolution + the sync guard
    config.py             models, look, beat sheet, topic bank, title formula
    engine.py             the one OpenRouter call, pinned, retried and costed
    brief.py              everything the prompts need, in one object
    sources.py            step 6
    evidence.py           step 7
    script.py             steps 8/11/14 - writer, fixer, section rebuild
    lint.py               step 9  - deterministic, no API call
    factcheck.py          step 10 - claims + smuggled claims
    retention.py          steps 12/13 - the gate and the hook tournament
    packaging.py          step 15
    readsheet.py          step 15b - per-section read sheets, the aligner's own
                                     chunking (parity-tested against Step 17)
    qc.py                 the measurements behind run_qc.py - no model, no GPU
    scenes.py             step 22 - plan the pictures
    shots.py              step 22 - what is in each frame
    look.py               step 22 - which face the character wears
  run_scenes.py           step 22 driver
  stickman.ipynb.bak      the 71-cell original
```

## run_scenes.py

Runs **between the two notebook halves** — after Step 19 (voice), before Step 20
(FLUX). It reads `scenes_timeline.json` and writes `scenes_planned.json`.

```powershell
.venv\Scripts\python.exe run_scenes.py --shots       # the rotation only, free
.venv\Scripts\python.exe run_scenes.py --dry-run     # cost + first 3 lines, free
.venv\Scripts\python.exe run_scenes.py
.venv\Scripts\python.exe run_scenes.py --limit 5     # try the prompt cheaply
.venv\Scripts\python.exe run_scenes.py --strict      # fail on rotation warnings
```

150 scenes is ~$0.08 on `claude-haiku-4.5`, eight calls at a time. It is the
stage most exposed to Drive lag, since it reads a file the notebook just wrote —
`load_timeline()` checks the file exists, is non-empty, and has the Step 17
shape (`scene_index`, `start`, `end`, `duration`, `text`, `section`) before
spending anything.

A scene the model cannot plan falls back to a default subject rather than killing
a 150-scene run, and output is always sorted by `scene_index` regardless of the
order the thread pool finishes in.

### Three decisions per frame, made by different things

Only one of them is the model's. The rest are assigned locally and
deterministically before any call goes out — for the same reason the palette lock
is not the model's job: a rule you can only ask for is not a rule.

| decision | what it sets | who decides | driven by |
| --- | --- | --- | --- |
| **shot** | what is in the frame | `pipeline/shots.py` | a seeded rotation |
| **expression** | what the face is doing | `pipeline/look.py` | the line's energy |
| **subject** | what the thing actually is | the model | the line |

Seven shot types, four of which have no character in them at all (`object-focus`,
`icon-grid`, `data-chart`, `typographic`). A scene's shot decides whether
`CHARACTER_BIBLE` **and its expression clause** enter the FLUX prompt —
concatenating the bible unconditionally is what used to put a stick figure on the
bar charts, and an expression on a bar chart is the same bug wearing a hat. The
shot also carries the background and floor clause, which is why `STYLE_PREFIX`
no longer does.

**The expression** replaces the frozen eyes-and-mouth sentence that used to live
inside `CHARACTER_BIBLE`. Six faces, chosen from `look.energy()` — a lexical,
free, deliberately dumb score of the line (punctuation, length, a small word
list, no model call). It only has to order things: a visceral line above an
analytical one. `EXPRESSIONS_ON = False` in `config.py` pins every face to
deadpan.

> A cyan/yellow palette rebrand and a framing axis were built here and
> **reverted** — that is not this channel's look. The warm white-wall/tan-floor
> room is. Only the expression survived.

`--shots` prints the rotation and the expressions and exits, with no key and no
cost. That is how you tune `shots.py` and `look.py`; both are seeded, so the same
timeline replans identically and a redraw stays reproducible. `--strict` refuses
to hand a rotation the validator complained about to the GPU. See PLAN.md, *The
visuals*, for what the rotation guarantees and why.
