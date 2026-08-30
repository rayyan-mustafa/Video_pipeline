# WHERE WE ARE — chin video, section-by-section build

**Written 2026-08-28.** Paste this whole file into a new chat to resume instantly.
Read alongside `STUDIO_HANDOFF.md` (architecture) — this file is the *current
job*: getting the chin video made, one section at a time.

---

## In one paragraph

**Section 1 is built, rendered and approved.** 13 pictures + 1 Google Flow clip
over 35.96s, on the reader's raw voice, pacing on target. The plan for the whole
video exists — **217 slots, 11 motion clips** — and the shot sheet with every
prompt already filled is on disk. What remains is generating the other 206
pictures and 10 clips, section by section. **One decision is open and it must be
settled BEFORE those pictures are generated: which Whisper model the timings come
from.** See "THE OPEN DECISION" below — it is the only thing here that can waste
real money.

---

## THE COLAB SEAM IS BUILT — the GPU align is a button now

`large-v3` on a Colab GPU is no longer a manual errand. The Voice page has a
worker box: paste a URL, tick "run on Colab GPU", press Run.

```text
  studio/colab_worker.py       paste into ONE Colab cell on a T4; prints a URL
  studio/backend/settings.py   stores that URL, health-checks it
  channels/pipeline/transcribe.py   the seam: LocalWhisper | RemoteWhisper
  build_video.py --colab-url   the same thing from the CLI
```

**THE SEAM IS ONE STEP WIDE, and that is the whole design.** Mastering stays
local and alignment stays local; only the transcribe call moves. So the worker
needs *no pipeline code*, cannot drift out of step with this repo, and the audio
it receives is already mastered — which means the timestamps it returns describe
exactly the audio the render will use.

Proved, not assumed: `test_transcribe_seam.py` runs the identical contract
locally and compares remote against local on real audio —
**same words, worst timestamp delta 0.00e+00.**

**Use a T4, not an L4.** Measured on this job: the align is 95s on a T4, and the
session is dominated by pip install and model load, not inference. An L4 is
~2.7x the hourly rate and saves perhaps 30s.

```text
  T4   ~0.15 units per video     <- use this
  L4   ~0.32 units per video
```

That replaces the old unmeasured 0.27 u/video estimate with a real number.

A cloudflared quick tunnel gets a **new hostname every run**, so the URL is
retyped once per Colab session. The Voice page shows whether the saved one is
still alive rather than letting a job fail four minutes in.

---

## ⚠️ THE OPEN DECISION — settle this before generating 206 images

**Every cut in this video sits on a Whisper word timestamp.** The current
timings came from **`base`, on CPU, int8** — not `large-v3`, which
`STUDIO_HANDOFF.md` says is what should ship.

```text
  model used     base, cpu/int8
  word match     78.4% - 93.5% per section
  anchored       114/114 scenes (100%)  <- nothing is broken
```

Nothing is broken: every scene anchored to a real timestamp. But `large-v3`
places word boundaries more precisely, and word boundaries are exactly what the
slot cuts and the 3.5-4.0s motion window depend on.

**Why `base` was used:** it runs in ~7 minutes locally, where `large-v3` on this
CPU is roughly 10x that — and **Colab never connected** (`colab-mcp` timed out
with `CONNECT_TIMEOUT` on both attempts), so no GPU was available.

**THE TRAP:** if the video is re-aligned on `large-v3` *after* the pictures are
made, the timestamps shift, the slots get re-cut, and **the narration on each
slot changes — which changes its prompt.** Every affected picture would have been
generated against words it no longer illustrates. This is the same failure the
raw-voice switch caused, only bigger.

```text
  DECIDE THE MODEL BEFORE GENERATING, NOT AFTER.

  want large-v3?  ->  connect Colab, re-align on GPU, re-cut ONCE,
                      regenerate the sheet, THEN generate pictures.
  happy with base? ->  generate now; the timings will not move again.
```

---

## What is DONE

### Section 1 — finished and approved

```text
  f:\wierd_bio\channels\wierdbiology\chin\chin_section1.mp4
  35.96s   6.4 MB   13 slots   12 stills (Ken Burns) + 1 Flow clip
  pacing: body average 2.9s, one 1.68s accent, on target
  drift 0.027s  <- frame quantisation, under one frame. NOT a defect,
                   and NOT fixable by regenerating pictures.
```

Verified by frame, not just exit code: the Flow clip is on screen at 19.5s with
its own timecode reading 1.16s into the clip, matching the slot start of 17.90s.

### The voice — RAW, and the reader approved it

New `--voice {clip,natural,raw}` flag on `build_video.py`. **The build uses
`natural`.**

```text
  clip     full mastering chain incl. silence trim      (the old default)
  natural  silence trim + level ONLY. No denoise, EQ,   <- WHAT WE USE
           de-esser, compressor, or 85Hz high-pass.
  raw      level only, every pause kept
```

`natural` exists because the reader wanted the pauses tightened but the tone
untouched — no existing profile did that combination. Delivered at -14.1 LUFS
(YouTube applies no gain change at that level).

If the take ever sounds rumbly, put back **`highpass_hz` alone** rather than
reaching for `clip`.

### The plan — 217 slots, 11 motion clips

```text
  217 slots over 597.78s      pacing "on target", body average 2.9s
  206 stills                  2-4s band, 1s accents where the narration wants one
   11 motion clips            4.0s each; Flow EXPORTS 44.0s, 41.3s lands on screen
                              (the tail is trimmed - cuts sit on word boundaries)
```

### The shot sheets — every prompt already filled

```text
  f:\wierd_bio\channels\wierdbiology\chin\build\shot_sheet.txt            all 217
  f:\wierd_bio\channels\wierdbiology\chin\build\shot_sheet_section1.txt   the 13 (done)
```

A **motion slot carries BOTH prompts**, still first: Flow is image-to-video, so
the picture is step one of two, not an alternative.

---

## What is NOT done

### 5 of the 11 motion slots no longer fit — do not generate clips for these

Switching to the raw voice kept more silence, which stretched five motion slots
past the 4.0s ceiling. **A 4s Flow clip cannot cover them.**

```text
  BROKEN (hold > 4.0s, cannot take a clip)
    slot  54  s3  4.26s        slot 123  s3  4.36s
    slot  77  s3  4.02s        slot 162  s4  4.48s
                               slot 204  s5  4.24s

  STILL FINE
    slot   7  s1  3.92s  <- DONE      slot 100  s3  3.58s
    slot  22  s2  3.56s               slot 144  s4  3.90s
    slot  36  s2  3.66s               slot 180  s4  3.54s
```

The sheet still marks all 11 as MOTION because the plan was deliberately **not**
re-cut — re-cutting would have invalidated the section 1 work that was already
paid for.

**The fix, when section 3 is reached:** re-cut *only* sections 3-5 with sections
1-2 pinned. That restores 11/11 and changes prompts only in sections not yet
started. Needs a small change to `pipeline/timeplan.py` (a "pin these sections"
argument).

### Assets still to make

```text
  pictures   206 of 217   (section 1's 13 are done)
  clips       10 of 11    (section 1's is done; 5 of the 10 are blocked above)
```

---

## How to carry on — the exact loop, per section

**1. Generate that section's pictures** from `shot_sheet.txt`. Save each under
the filename in its header, into:

```text
  f:\wierd_bio\channels\wierdbiology\chin\images\
```

Extension does not matter — the build globs `1section2.*`, so `.png`, `.jpg`,
`.jpeg`, `.webp` all work, and the generator's own suffix
(`1section2.png_2K_2026....jpeg`) is fine. **Do not zero-pad**: `1section2`, not
`01section2`.

**2. Generate that section's Flow clips** (only the ones not in the broken list).
Save as the **slot index, zero-padded to 3**, into:

```text
  f:\wierd_bio\channels\wierdbiology\chin\motion\    e.g. 022.mp4
```

> The image filename and the clip filename use DIFFERENT numbers. The sheet
> header shows both: `[MOTION] 23section2.png | slot 22`. The picture is
> `23section2`, the clip is `022.mp4`.

**3. Build it.**

```powershell
cd f:\wierd_bio\channels
$py = "..\vo_coach\.venv\Scripts\python.exe"
& $py build_video.py --project wierdbiology/chin --stage plan --only-section 2
& $py build_video.py --project wierdbiology/chin --stage mux  --only-section 2 --motion kenburns
```

Output lands at `chin_section2.mp4`. `--only-section` scopes both the picture
check and the sync check to that section, so a problem in section 3 cannot block
section 2.

---

## Traps already paid for in THIS session — do not rediscover

**Whisper on CPU, `base` — see THE OPEN DECISION.**

**The 0.027s drift is frame rounding, not an error.** Each slot must be a whole
number of frames, so a 2.68s hold becomes 80 frames = 2.667s. Those roundings sum
to 27ms across 13 slots — under one frame at 30fps. **Regenerating pictures
changes nothing**, because the holds are identical. Ignore it.

**A picture's end may run past its own section's last word.** Two recordings
joined leave a pause between them, and the tiling rule hands it to the outgoing
section's last picture — correct editorially, and nothing is spoken there.
`imageplan.verify` was flagging this as a failure; it now bounds by the *next*
section's first word instead.

**Word counts cannot place a motion slot — only measured time can.** Slots built
at 9 words (4.0s at the section's average pace) came out between **2.82s and
5.44s**, because delivery speed varies far more *within* a section than between
sections. Only 3 of 11 landed in the window. `--refine-from-words` re-cuts
against real timestamps and took it to 11/11.

**Several pictures sharing one scene cannot be timed by scene spans.** Once three
pictures split one sentence they all report the same `scene_index`, inherit the
same span, and the tiling pass crushes two onto the 0.5s floor — measured at **81
of 181 slots flattened to 0.50s**. `imageplan._spans_from_words` times each
picture by its own words instead. **Whisper supplies the clock; the script
supplies the words** — never the transcript, which is 78-94% accurate here and
would quietly rewrite the prompts.

**The uploaded clip arrived in the wrong folder as `007.mp4.mp4`**, and slot 7's
seed still was missing. The still was recovered as frame 1 of the clip (Flow's
own prompt guarantees the uploaded image IS the first frame).

**The original 122 chin frames were lost** when `images/` was replaced during the
section 1 upload. They belonged to the superseded 105-slot plan, so nothing
current needs them — recorded only so nobody looks for them.

**Colab MCP never connected.** `colab-mcp` timed out (`CONNECT_TIMEOUT`, 30s) on
both attempts. Everything this session was local CPU. The GPU is needed only for
`large-v3` alignment; the ffmpeg render is CPU and **must never run on a GPU
runtime**.

**Disk is tight.** F: sits around 1.6 GB free. The preflight is now sized to the
job (`~0.3 GB for 36s`) instead of a flat 2 GB, so short test renders work — but
a **full 594s render needs ~1.7 GB** and will refuse until space is freed.

---

## Code written this session — none of it committed

```text
  NEW   channels/pipeline/recordings.py    which take is which section, by measurement
        channels/pipeline/motionplan.py    which slots get a clip, and why not more
        channels/pipeline/timeplan.py      re-cut slots against measured timestamps
        channels/pipeline/videoprompt.py   the Flow prompt + deterministic cameras
        channels/test_recordings.py        23 checks
        channels/test_motionplan.py        18 checks
        studio/backend/voicestage.py       naming + alignment + results
        studio/backend/motionstage.py      motion slots + clip ingest
        studio/frontend/                   Vite + React + TS, 6 pages

  CHANGED  channels/build_video.py      --voice, --only-section, MixedProvider wiring,
                                        job-sized disk preflight
           channels/plan_script.py      splits long scenes, --motion-seconds,
                                        --refine-from-words
           channels/pipeline/imageplan.py  word-based spans, require_images,
                                           section-scoped verify
           channels/pipeline/clips.py      VideoAssetProvider, MixedProvider
           channels/pipeline/voice.py      'natural' profile, blocked limiter
           channels/pipeline/pacing.py     motion constants
           channels/pipeline/project.py    .motion property
```

Tests: **23 + 18 + 97, zero failures.**

**Nothing is committed.** Branch `studio-ui`, last commit `bc4ff7c`. Untracked
logs `channels/_align.log`, `_natural.log`, `_plan.log` and the scratch backup
`channels/wierdbiology/chin/build_match_backup/` should be deleted or ignored
before committing.

---

## The immediate next action

1. **Answer the open decision: `base` or `large-v3`?**
2. If `large-v3` — connect Colab first, then re-align, re-cut, regenerate the sheet.
3. If `base` — generate section 2's **29 pictures + 2 clips** from
   `shot_sheet.txt`, drop them in, and run the two build commands above.

```text
  section 2   29 slots   1section2 .. 29section2
              2 motion   slot 22 -> 022.mp4      slot 36 -> 036.mp4
                         (both still need their seed picture: the motion
                          slots are 2 OF the 29, not extra)
```
