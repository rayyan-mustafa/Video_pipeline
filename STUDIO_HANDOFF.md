# HANDOFF — Video Studio (local React UI over the pipeline)

Written 2026-08-27. **Updated 2026-08-28** — production model changed:
pictures and motion are now generated externally. Paste this whole file into
a new chat to continue.

Supersedes nothing — read alongside `channels/HANDOFF_VIDEO_BUILD.md`, which
covers the CLI pipeline this UI drives. `STATUS.md` / `NEXT_SESSION.md` cover the
older *visual* architecture (draw.py, props, LoRA) and are not needed for this work.

---

## One-paragraph state

The chin video builds end to end from the CLI. We are wrapping that pipeline in a
local React UI so it can be driven without terminal commands, and adding the one
thing it genuinely lacks: **a timeline editor to fix what Whisper got wrong**
rather than re-recording.

**The production model changed on 2026-08-28.** Pictures are now generated
*outside* this pipeline (Nano Banana) and motion clips are generated *outside* it
too (Google Flow). The pipeline's job is no longer to *make* assets — it is to
**time them, place them and cut them**. That is a simplification, not a loss: it
removes FLUX and every GPU generator from Colab, which now does exactly two
things — Whisper `large-v3` alignment on a **GPU** runtime, and the ffmpeg render
on a **free CPU** runtime. Keeping those two on separate runtimes is the whole
reason the monthly bill is ~3 units instead of ~8.

**Phases 0–2 are done and committed. Phase 2.5 (motion) is built but UNCOMMITTED.**
Backend runs on `127.0.0.1:8770`. No frontend exists yet.

---

## ⚠️ Do this first

**Rotate both API keys.** During Phase 1 testing a `curl` printed `.env` into a
chat transcript, so `HF_TOKEN` and `OPENROUTER_API_KEY` are burned.
- openrouter.ai/keys → revoke, reissue
- huggingface.co/settings/tokens → revoke, reissue
- update `f:\wierd_bio\.env`

The HF token also sits hardcoded in 4 cells of `channels/stickman.ipynb` and its
`.bak`. Both notebooks are **gitignored** for exactly that reason — once the
token is rotated and those cells read from `.env` (cell 13 already does), delete
the two ignore lines so the notebooks get tracked.

---

## THE PACING SPEC — this drives everything

Fast-cut editorial. This is the current, authoritative version:

```text
  BODY     cut to a new scene every  2 - 4 seconds     <- the normal rhythm
  ACCENT   quick changes at ~1 second (0.5-2.0s)       <- deliberate emphasis
  FLOOR    below 0.5s is a flash, never intentional
  CEILING  above 4s is too slow
  MOTION   a Flow clip is 4.0s = the CEILING           <- so it lands on the
           slot accepted only if hold is 3.5 - 4.0s         slowest legal hold
           10-15% of runtime, never on an accent
```

A one-second hold is **not a defect** — it is how a specific point in the
narration gets emphasised. Accents should stay under ~25% of frames, or they stop
reading as emphasis.

The 2-4s band is not folklore: it matches the documented fast-cut convention for
this format (high-performing short-form averages a cut every 2-4s). Note it is
*faster* than generic long-form advice, which says change something visually every
5-7s — that is deliberate, this channel is a fast-cut cartoon explainer, not a
talking head.

Lives in **`channels/pipeline/pacing.py`** — change constants there, never at
call sites. `--seconds-per-image` now defaults to **3.0** (band midpoint); the old
8.0 was never a decision, just the first number that looked plausible.

### How pacing is scored

`pacing.verdict(holds)` judges the **body** (everything ≥ 2s, no upper bound),
not the raw mean — accents would otherwise drag the mean down and a correctly
paced video would score "too fast".

> Bug already fixed here, do not reintroduce: the body set originally capped at
> `HARD_MAX`, which excluded over-long holds from their own average. Chin scored
> "on target" on a 2.9s body mean while 67 of its 105 pictures sat above 4s.

### The chin video fails this spec

```text
105 images over 602s
  body average    6.69s        (band is 2-4s)
  over 4s         67 of 105    (64%)
  worst hold      15.6s
  needs           ~201 images, has 105
```

Not fixable by re-planning — it needs ~100 more pictures, or motion.

**Ken Burns is now built** and makes a 6.7s hold watchable, but it does not change
a single number above: the verdict scores *holds*, and a slow zoom does not
shorten one. **More pictures is still the only fix for the verdict itself** — and
under the new model those pictures come from Nano Banana, so the ~100 missing ones
are now cheap to produce. Re-plan chin at ~201 slots before treating its pacing as
a hard problem.

### Scene count is NOT the cap on cut speed — words are

Two binding strategies exist:

| strategy | when | cuts land |
|---|---|---|
| **1:1** | scenes ≈ pictures | on sentence edges (cleanest) |
| **word-cuts** | more pictures than scenes | inside sentences, at exact Whisper word timestamps |

The chin script splits to at most 114 scenes but wants ~201 pictures — that is
**word-cuts at ~6.3 words per picture**, entirely supported by
`imageplan._exact_word_cuts`. An earlier version of `_recommend()` called this
"unreachable" and would have sent the user off to rewrite a fine script.

---

## THE PRODUCTION MODEL — read this before touching the planner

```text
  PICTURES   Nano Banana, external            -> dropped into <project>/images/
  MOTION     Google Flow, external, 4s min    -> dropped into <project>/motion/
  TIMING     Whisper large-v3, Colab GPU      -> the only thing that needs a GPU
  CUTTING    this pipeline, Colab CPU runtime -> ffmpeg, FREE (see below)
```

Nothing in this pipeline generates a pixel any more. It *places* what you give it.

### The hybrid: what an 8-minute video is made of

```text
  480s total, body pace 3.0s          ->  ~160 slots
  motion   48-55s   =  12-14 clips at 4s each   (10.0 - 11.5% of runtime)
  stills   425-432s =  146-148 slots            (2.91s average, inside the band)
```

For chin's real 602s the same *percentage* is 60-69s = **15-17 clips**; the same
*absolute* 48-55s is only 8-9%. Prefer the percentage — motion should scale with
runtime, not be a fixed dose.

**Why 4 seconds is the number that matters.** Flow's minimum clip is 4s and
`pacing.HARD_MAX` is 4.0s. Those being equal is a coincidence, and a lucky one:
**a motion clip is exactly the longest hold the spec allows.** So the rule writes
itself —

```text
  MOTION GOES ON THE SLOTS THAT WOULD OTHERWISE SIT AT THE CEILING.
  A 4s hold is the one most at risk of reading as slow, and motion is what
  makes it watchable. That is the logic pacing.py already states in prose.
```

### DECIDED 2026-08-28: the planner sizes motion slots, not the provider

**The planner picks sentences that fit a 4-second clip.** A motion slot is chosen,
not adapted to afterwards. Constants live in **`pacing.py`** next to the band they
belong to:

```text
  MOTION_CLIP_S      4.0    what Flow emits, and == HARD_MAX
  MOTION_SLOT_MIN    3.5    shortest hold still allowed to carry a motion clip
```

**But "exactly 4.000s" is not reachable, and the doc must not pretend it is.**
Cuts land on real Whisper word timestamps — that is the whole basis of the
timeline editor's roll edits and must not be broken to hit a round number. At
150 wpm a word is ~0.4s, so the achievable holds near the target are quantised:

```text
   8 words = 3.20s      <- too short, reject
   9 words = 3.60s      <- accept
  10 words = 4.00s      <- ideal
  11 words = 4.40s      <- over HARD_MAX, reject
  worst-case miss from 4.000s = +/- 0.2s (half a word gap)
```

So the rule the planner actually implements:

```text
  1. TARGET 4.0s. From the slot's start word, choose the cut word whose
     timestamp is nearest to start + 4.0 WITHOUT EXCEEDING IT. Never
     produce a hold over HARD_MAX to fit a clip.
  2. ACCEPT the slot only if the resulting hold lands in [3.5, 4.0].
     Outside that window, this sentence does not get motion - pick another.
  3. Motion placement is OPPORTUNISTIC. If no sentence in a stretch fits
     the window, that stretch stays stills. Never bend the script.
```

**The residual (0 - 0.5s) is absorbed by trimming the Flow clip's TAIL.**
Not by slowing it: `motion.py`'s founding rule is that the eye reads speed, and
these clips are already a slow ambient drift — stretching 4.0s over 3.6s of screen
time would make them read as static. The tail is also the least informative part
of an inbetween, so it is the cheapest half-second to lose. **Never freeze or
extend** a clip to fill a longer hold; that pushes past `HARD_MAX` by definition.

Corollaries, all falling straight out of the existing spec:

- **Accents never get motion.** A hold under 2s is emphasis; 4s of video cannot
  fit in it, and `motion.py` already refuses to move anything under `ACCENT_HIGH`.
- **Motion slots are not extra runtime.** A motion clip *replaces* a still slot,
  it does not append 4s. The planner must place them, not add them.
- **Each motion slot still needs a seed picture** if Flow is driven
  image-to-video, so the picture count does not drop — 12-14 of the ~160 stills
  become video seeds rather than final frames.
- **10-15% is the editorial target, not more.** Retention guidance for animation
  explainers is consistent that pacing carries the video and over-animating costs
  comprehension: if everything moves, nothing is emphasised — the same failure
  mode as `pacing.verdict`'s "too fast".

### Colab cost is now a rounding error

```text
  Whisper large-v3 align + env, per video   ~0.27 units   (ESTIMATED, not measured)
  ffmpeg render, per video                   0.00 units   (CPU runtime is free)
  12 videos/month                            ~3.2 units of 100
  36 videos/month                            ~9.7 units of 100
```

### The runtime split is the whole reason rendering is free

`stickman.ipynb` Step 0 already worked this out and split itself in two over it:

> *Colab bills COMPUTE UNITS BY WALL-CLOCK ON THE RUNTIME, not by how much the
> GPU is used. A minute of ffmpeg on a T4 costs exactly what a minute of FLUX
> costs.*

Rendering is 100% CPU. Put it on a GPU runtime and you pay GPU rates for zero GPU
work. Measured local rate this session was **1.30x realtime** (51s of render for a
39.2s window), so an 8-minute video is ~10 min on this laptop and ~3-5 min on
Colab's faster CPU:

```text
  render on CPU runtime (free)      12 videos/month =  3.2 units   <- align only
  render on GPU runtime (L4)        12 videos/month =  5.7-8.2 units
```

That is **2-2.5x the entire monthly Colab bill, spent on an x264 encode.** The
render job must therefore:

- **refuse to run, or warn loudly, if `nvidia-smi` is present.** Step 0 already
  computes `HAS_GPU`; reuse it as a guard rather than trusting the operator to
  remember which runtime they are on.
- keep the per-slot clip cache on **`/content`, not Drive**. It is ~160 small mp4s
  and Drive's per-file overhead would dominate; a disconnect loses the cache, but
  a full render is only 3-5 minutes, so that is the cheaper trade.
- write only the **final mp4** back to Drive.

Assets reach Colab the way `chin_full_bundle.zip` already does — through the
mounted Drive folder. Remember `force_remount=True`: the mount caches its listing,
so anything uploaded after mounting is invisible without it.

**Measure this once and replace the estimate.** The old 1.42 u/video figure was a
real actual (85 units / 60 videos) and included FLUX; 0.27 is derived from the T4
rate times a rough align time and has never been clocked. Flow and Nano Banana
bill separately and are **not** in this budget — a second ledger nobody has sized.

---

## MOTION — Ken Burns is built; external clips are not

`--motion kenburns` on `build_video.py`. Default is still `none`, so nothing about
the existing build changed.

Three rules, all in **`channels/pipeline/motion.py`** — constants there, never at
call sites, same as `pacing.py`:

```text
  MOTION IS A RATE, NOT A TOTAL   amount = per-second * hold, then clamped.
                                  A fixed 12% zoom looks right at 3s and looks
                                  static at 15s, because the eye reads speed.
  ACCENTS DO NOT MOVE             a hold under 2s renders static. A zoom that
                                  starts and stops inside 1s is a twitch and it
                                  destroys the emphasis the short cut exists for.
  THE MOVE ROTATES BY SLOT INDEX  two adjacent push-ins read as one push with a
                                  cut in it, which erases the cut. Deterministic,
                                  so a re-render is identical.
```

Zoom anchors at its **lower** end, never the midpoint. Anchoring at the midpoint
(the first version) turned a 1.00->1.22 push into 1.09->1.13, so every body cut
opened already cropped by 9% and never showed the whole picture. Do not
reintroduce.

`SUPERSAMPLE = 3` is the expensive constant and it is load-bearing: zoompan
quantises zoom to integer source pixels, so on flat line art the strokes crawl
frame to frame. Measure before lowering it.

Ken Burns is now the treatment for **every still slot** — roughly 146-148 of ~160.
It is free, local, and needs no units.

### The provider seam — the actual architectural point

`pipeline/clips.py` renders **one clip per slot** and joins them with
`concat -c copy` (no second encode). A *provider* answers one question: given a
slot and a hold, produce that many seconds of video.

```text
  StillProvider       what the pipeline did before, expressed as a clip   BUILT
  KenBurnsProvider    zoompan, ~1.3x realtime, local, free                BUILT
  VideoAssetProvider  conform an EXTERNAL Flow clip onto the slot         TO BUILD
```

That third row is the whole of Phase 4. It does not generate — it **ingests**:
scale/pad to 1280x720, conform fps, trim to the slot's hold, and hand back a clip
that concatenates byte-identically with the others. Far smaller than a generator
integration, because all the hard part now happens in Flow.

Why per-slot and not one `filter_complex` with 160 zoompans: Windows caps a
command line at 32767 chars and 200 slots blows through it as a truncated-arg
error that looks unrelated to length; one bad jpeg at slot 190 would throw away
the whole render; and nothing would be reusable.

Clips are **cached** in `build/clips/<provider>/`, keyed on the source bytes'
size+mtime, the hold, the move and the output format — so a regenerated picture
invalidates its clip instead of silently serving the old one.

### A/B harness

```powershell
cd f:\wierd_bio\channels
..\vo_coach\.venv\Scripts\python.exe ab_motion.py --project wierdbiology/chin `
    --from 120 --seconds 30
```

Renders the same window through each provider at production settings, plus a
side-by-side. **Needs no voice stage**: holds come from `render_plan.json` if it
exists, else from the word counts already in `script_plan.json` at the planner's
wpm. `--dry-run` prints the move table and renders nothing.

Measured on chin: still 32s, kenburns 51s for a 39.2s window — so a full 602s
build is roughly 13 minutes, once, then cached.

---

## TOONCRAFTER — evaluated and REJECTED, do not retry

Fully ported and working on Colab 2026 (L4). **The port was never the problem; the
model is.** Recorded so nobody spends the units again. Cost to learn: **6 units.**

Measured on the chin caveman frame, 50 steps, L4:

```text
  same frame at both ends     every frame within 1.5% of the input
                              -> a wobble, not motion
  two different scenes        holds near A for 13 of 16 frames, then SNAPS
                              to B in the last two
                              -> a hold-and-cut, not an inbetween
```

**Why: it is an inbetweener, not an animator.** It wants two drawings of *the same
shot* with a small pose change. Every picture this channel produces is a different
composition, so there is nothing to interpolate between.

The `_2` alternates are **not** keyframe pairs — measured, not assumed:

```text
  alternate vs its own base   mean 44.0 / 255   (range 16.0 - 73.9)
  two UNRELATED scenes             58.6 / 255
```

An independent re-roll moves the pose, the background and the framing. Three
alternates differ from their own base *more* than two random scenes differ from
each other. Real pairs would need img2img or fixed-seed variation — a change to
the image stage, which is external now anyway.

Other facts worth keeping: 102s per 2-second clip on L4; `temporal_length` is 16
frames baked into the checkpoint; **25 sampling steps produces pure black** (a
sampler indexing bug, not gradual quality loss), so there is no cheap mode.

Four porting fixes if it is ever revisited: `torch.load` needs
`weights_only=False`; pin `open_clip_torch==2.22.0` (newer is batch-first and
inverts the attn_mask); shim `xformers` onto torch SDPA **with the backend forced
to EFFICIENT/FLASH** (the default `math` backend materialises the full matrix,
6.25 GiB, instant OOM); `torchvision.io.write_video` is gone, use an ffmpeg pipe.
Also evict the UNet before VAE decode or 16 frames will not fit beside it.

---

## Phase status

```text
  OK   Phase 0    git init, .gitignore, secrets kept out of history
  OK   Phase 1    backend skeleton: projects, jobs+SSE, guarded file serving
  OK   Phase 2    script / plan / images endpoints + pacing as first-class
  WARN Phase 2.5  motion: rate-based Ken Burns, provider seam, A/B harness
                  BUILT AND VERIFIED, NOT COMMITTED
  --   Phase 3    voice stage - Whisper large-v3 timings      <- NEXT, and it
                  blocks every phase below it                    is the blocker
  --   Phase 4    motion slots + VideoAssetProvider (Flow ingest)
  --   Phase 5    the timeline editor
  --   Phase 6    render + preview — Colab CPU runtime, local as fallback
  --   Phase 7    AI script generation
  --   (unscheduled) the React frontend - nothing exists yet
```

Commits on branch **`studio-ui`**:
```text
a46c534  studio: script, plan and image endpoints; pacing as first-class target
24c8f3f  studio: backend skeleton - projects, jobs, guarded file serving
a9460c9  Initial commit: script pipeline, video builder, vo_coach
```

---

## Architecture

```text
  EXTERNAL                    THIS PIPELINE                     EXTERNAL
  Nano Banana  --images-->                          <--motion--  Google Flow
                              React (Vite, :5173)
                                     |  HTTP/SSE
                              FastAPI (:8770)  --import-->  channels/pipeline/*
                                     |  HTTPS
                              Colab worker (cloudflared URL)
                                |
                                +-- GPU runtime : large-v3 align  (billed)
                                +-- CPU runtime : ffmpeg render   (FREE)
```

- **Short pure functions are imported** (`imageplan`, `scenesplit`, `pacing`,
  `Project`) — the UI can never disagree with the CLI because it runs the same code.
Assets arrive as files; the pipeline never calls an image or video API.

- **Long stages are subprocesses** (`build_voice`, `mux`, `run_script`) — they are
  `argparse main()` + `print()` + `SystemExit`, so in-process calls would take the
  server down. `jobs.py` streams their stdout over SSE.

### Decisions already made

| | |
|---|---|
| Pictures | **Nano Banana, external.** Pipeline never generates one. |
| Motion | **Google Flow, external, 4s minimum clip.** Ingested, not generated. |
| Motion dose | **10-15% of runtime**, 12-14 clips per 8 min. Never on accents. |
| Motion slot size | **planner targets 4.0s, accepts [3.5, 4.0]**, cut stays on a word boundary |
| Clip residual | **trim the tail.** Never slow, never freeze, never exceed `HARD_MAX`. |
| Flow clips | **validated on ingest**, never trusted. Short clip = hard fail. |
| Colab link | tunnel — Colab runs FastAPI + cloudflared, URL pasted into UI per session |
| Colab scope | **Whisper `large-v3` alignment only.** No FLUX, no generators. |
| Render | **Colab CPU runtime** — free, and faster than the laptop. Local ffmpeg stays as fallback. |
| Runtime split | **ffmpeg NEVER on a GPU runtime.** Pure CPU work billed at GPU rates. |
| Voice input | upload `.wav` per section |
| AI scripts | picker over the existing 15 `config.TOPICS` only |
| Editor scope | roll edits, replace, reorder — **not** split/merge (see Phase 5) |

---

## Run it

```powershell
cd f:\wierd_bio
.\vo_coach\.venv\Scripts\python.exe -m uvicorn studio.backend.app:app --host 127.0.0.1 --port 8770
```

**Always `vo_coach\.venv`** (Python 3.12.10). `channels\.venv` lacks
faster-whisper, noisereduce and pyloudnorm — the voice stage dies in it.
Already added: `python-multipart`, `httpx`.

`pkill` silently does nothing on this machine. To stop the server:
```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -like '*uvicorn*studio.backend*' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

### Endpoints that work today

```text
GET  /api/health
GET  /api/projects                                  list + per-project status
POST /api/projects?channel=&topic=                  create skeleton
GET  /api/projects/{ch}/{tp}
GET  /api/file?path=<abs>                           allowlisted static serving
GET  /api/projects/{ch}/{tp}/script                 text + brackets + splits + pacing
PUT  /api/projects/{ch}/{tp}/script                 {"text": "..."}
GET  /api/projects/{ch}/{tp}/plan                   script_plan.json + abs_path + pacing
POST /api/projects/{ch}/{tp}/plan                   {split, seconds_per_image, topic}
GET  /api/projects/{ch}/{tp}/images                 scan + alternates + by_section
GET  /api/projects/{ch}/{tp}/images/pacing          per-section hold table
POST /api/projects/{ch}/{tp}/images                 multipart upload
GET  /api/jobs  ·  /api/jobs/{id}  ·  /api/jobs/{id}/stream  ·  POST .../cancel
```

Verified against the chin project:
```text
SCRIPT  1314 words, 8.76 min, brackets_ok=True, splits {chunk:68, sentence:97, dash:114}
IMAGES  105 used, 122 on disk, 17 alternates excluded
PLAN    match, 105 scenes / 105 images, fingerprint b5da03f977a7c0d4
```

---

## Files

### New (2026-08-27 / 28)
```text
channels/pipeline/pacing.py        the pacing spec, shared by planner + checker
channels/pipeline/motion.py        the motion spec: rate not total, accents static
channels/pipeline/clips.py         per-slot clip render + provider seam + cache
channels/ab_motion.py              A/B a window across providers, needs no voice stage
studio/backend/paths.py            interpreter + repo layout, one place
studio/backend/jobs.py             subprocess registry + SSE, keeps a line backlog
studio/backend/projects.py         wraps pipeline.project.Project
studio/backend/scripts.py          script CRUD, brackets, splits, recommendation
studio/backend/planning.py         runs plan_script, reads back with abs_path
studio/backend/assets.py           image scan/upload/pacing
studio/backend/app.py              the FastAPI app
STUDIO_HANDOFF.md                  this file
```

### Modified
```text
channels/plan_script.py     --seconds-per-image default 8.0 -> pacing default 3.0
channels/build_video.py     check_images scores against the band, not 1.5s/15s
                            + --motion {none,kenburns,still}, default none
.gitignore                  media, venvs, notebooks, build dirs
NEXT_SESSION.md             redacted a verbatim HF token
STEP11_RESUME.md            redacted a verbatim HF token
```

### Still to build
```text
channels/pipeline/project.py    ADD a `motion` property + make_dirs entry.
                                <project>/motion/ is referenced all through this
                                doc and does NOT exist in Project yet.
channels/pipeline/motionplan.py which slots get a Flow clip (Phase 4)
studio/backend/settings.py      colab_url, render target
studio/backend/colab.py         ColabClient over the tunnel
studio/backend/voicestage.py    transcribe: local or Colab
studio/backend/timeline.py      editor data source
studio/backend/overrides.py     build/overrides.json
studio/backend/rendering.py     mux: local or Colab CPU (never GPU)
studio/colab_render.ipynb       CPU-runtime render job, HAS_GPU guarded
studio/colab_worker.ipynb       FastAPI + cloudflared worker
studio/frontend/                Vite + React + TypeScript — nothing exists yet
```

---

## Next: Phase 3 — the voice stage, and why it blocks everything

**This is the next thing to build, and the reason is stronger than it was.**

Motion slots cannot be chosen without real timings. The rule is "put the 12-14
motion clips on the slots that would otherwise sit at the ceiling" — and which
slots those *are* is only knowable from measured Whisper holds, not from word-count
estimates. Phase 4 is unbuildable until Phase 3 lands.

Right now the chin project has **no `render_plan.json` and no `full_vo.wav`** — the
voice stage was run on Colab in an earlier session and never came back. There is
also a naming mismatch blocking a local run: recordings are `1.wav`..`6.wav` but
`find_recording()` wants `section_01.wav` / `section_1.wav`, and there are 6 files
for 5 sections. **Resolve that mapping first** — it is the single cheapest unblock
in the whole plan.

1. Fix the recording naming so `--stage voice` runs locally at all.
2. `studio/colab_worker.ipynb`: install deps, preload `large-v3` **once**, expose
   `/health` and `/transcribe` (multipart zip in, zip out), open a cloudflared
   quick tunnel (no account needed) and print the URL. Nothing else goes in it.
3. `settings.py` + `colab.py`: store the pasted URL, health-check it, POST bundles.
4. `voicestage.py`: local subprocess path *and* Colab path behind one interface.
5. Upload UI for per-section `.wav`.
6. Results must show **word-match %** and **anchored N/M** as different things —
   see the trap below.
7. **Measure the units one align actually costs** and replace the 0.27 estimate.

### Phase 4 — motion slots + Flow ingest

Two halves, and the first is pure arithmetic on data Phase 3 produces:

**Selection.** Given measured holds, pick the 12-14 slots that get motion. The
policy is already decided: highest holds first (those nearest/over `HARD_MAX`),
never a slot under `ACCENT_HIGH`, and spread them out — two adjacent motion slots
read as one 8s shot and erase the cut between them, exactly the reason
`motion.ROTATION` exists. Emit a `motion_slots.json` the operator can hand-edit,
because "which line deserves emphasis" is a taste call the planner should propose
and not impose.

**Ingest.** `VideoAssetProvider` in `clips.py`: read the Flow mp4 for a slot from
`<project>/motion/`, scale/pad to 1280x720, conform fps, trim to the slot's hold,
and emit a clip that concatenates byte-identically with the Ken Burns ones. Same
`signature()` / `render()` contract as the other two providers, so the cache, the
concat and the drift check all work unchanged.

**Slot sizing is DECIDED** — see "the planner sizes motion slots" above. The
planner targets 4.0s, accepts only holds in [3.5, 4.0], and the provider trims the
clip's tail to the exact hold. No slowing, no freezing, no cut placed off a word
boundary.

**Ingest validation is DECIDED: the provider checks, it does not trust.** A Flow
clip is an external artefact that nothing in this repo produced, so
`VideoAssetProvider.render()` must verify before it encodes:

```text
  ffprobe reads it at all      -> else fail loudly, naming the file
  duration >= the slot hold    -> HARD FAIL. A clip shorter than its slot cannot
                                  be covered without slowing, which is banned.
  dimensions                   -> report actual; conform by scale + pad to
                                  1280x720, same as every other provider
  fps                          -> conform to the build fps
  output frame count           -> EXACTLY motion.frames_for(hold, fps), because
                                  concat -c copy has no way to fix drift later
```

Fail loudly and name the file. A silently wrong clip at slot 140 is discovered at
the end of a 5-minute render, which is the expensive moment to find it — the same
reasoning behind `mux()`'s existing PIL-verify preflight over every image.

### Phase 5 — the timeline editor (the centrepiece)

Included: **roll edits** (drag a cut, snapped to real Whisper word timestamps from
`build/words.json`), **replace** (swap the file on a slot), **reorder** (drag
between slots = two replaces), and now **toggle a slot between still and motion**.

Excluded deliberately: **split/merge slots** — that changes the image-to-sentence
binding the fingerprint protects. It belongs on the Plan page, where split modes
live.

Persistence — `<project>/build/overrides.json`, the editor's only output:
```json
{ "version": 1, "fingerprint": "b5da03f977a7c0d4",
  "cuts": {"7": 42.31}, "assets": {"12": "9section3_....jpeg"},
  "motion": {"12": "flow_012.mp4"} }
```
Apply in `imageplan.plan_from_binding()` via a new `overrides` param, to the
`starts` list **before** the existing tiling/min-hold pass, so gaps, overlaps and
flashes stay structurally impossible. Stale fingerprint -> warn, do not apply.

---

## Traps already paid for — do not rediscover

**Word-match % is not sync accuracy.** A scene anchors on its *first matched
word* and then carries that word's exact timestamp. Chin section 1 scored 73.9%
word-match yet anchored 9/9. Report `anchored N/M` as the real number.
100% word-match is impossible on a script with 27 proper nouns and 2 years.

**`.env` lives inside the repo**, so "is the path inside the repo?" was never a
sufficient guard for file serving. `/api/file` uses an **allowlist** of renderable
extensions plus a blocked-directory check. Verified: `.env`, `.git/config`, venv
files and `.py` sources all 403.

**A picture's end is where the next one begins.** `align()` floors every scene at
0.05s, so two scenes anchoring within 50ms produce scene *i* ending after scene
*i+1* starts. Taking each image's end from its own scene gave 2 overlaps, 2 flash
frames and 0.1s of phantom runtime. Never reintroduce per-scene ends.

**`_2` suffixed files are regenerations of the same slot**, not extra pictures.
`scan_images` keeps the lowest variant and reports the rest.

**Filename pattern may lead or trail**: `10section4_2026...jpeg` and
`chin_047_2section1.jpeg` both parse. Sorting is by `(section, order)`.

**`match` split cannot be re-derived from the script** — its scenes come from the
image folder, so they must travel in `script_plan.json["scenes"]`.
`build_video.py --split` only accepts `chunk|sentence|dash`.

**Editing `script.txt` after planning invalidates every image binding.** The
fingerprint catches it and `plan_from_binding` refuses. The UI must surface this.

**Two disconnected path universes.** `run_script.py` writes through
`pipeline/paths.py`'s global workdir; everything else uses `Project`. They meet
only in Phase 6, which is why it is last.

**Colab specifics**: the Drive mount caches its listing — a file uploaded after
mounting needs `force_remount=True`. `runtime.unassign()` frees the GPU but the
runtime is then recycled, so pip installs must be redone. Never `display()` an
image through colab-mcp (1,048,576-byte frame cap).

---

## Known gaps

- **No frontend at all.** No `package.json`, no React. Node 24 / npm 11 available.
- **Chin fails the pacing spec** (6.7s body average vs 2-4s band). Now cheap to
  fix: it needs ~100 more pictures and pictures are external/Nano Banana.
- **Transitions are deliberately absent.** Motion exists now, but only *inside*
  a frame. A crossfade softens a cut and this channel's whole rhythm is that the
  cut is hard.
- **No captions** in this path.
- **`vo_coach/.venv` has install drift** — `librosa`, `numba`, `torch`,
  `silero-vad` are pinned but missing (Defender zeroed files once). None are
  needed for Studio; all 9 packages we do need import cleanly.
- **`run_script.py` topics are a fixed bank of 15**, each needing curated source
  URLs. Free-text topics are not supported.
- **Phase 2.5 is uncommitted.** `motion.py`, `clips.py`, `ab_motion.py` and the
  `build_video.py` changes are verified but not in git.
- **The chin voice stage has never run locally** — recordings are named
  `1.wav`..`6.wav` against a `section_NN.wav` matcher, 6 files for 5 sections.
  No `render_plan.json`, so no measured holds, so no motion-slot selection.
- **The 0.27 u/video Colab estimate is unmeasured.** See Phase 3 step 7.
- **Flow and Nano Banana costs are unsized.** A second budget nobody has modelled.
- **`<project>/motion/` does not exist yet.** `Project` has no `motion`
  property and `make_dirs()` does not create it. Every reference to that folder
  in this doc is a plan, not a fact.
- **Flow clip validation is specced but unbuilt.** The checks are written down
  under Phase 4; `VideoAssetProvider` does not exist yet.
- **`pacing.py` does not yet have `MOTION_CLIP_S` / `MOTION_SLOT_MIN`.** The
  policy is decided and documented; the constants are not in the code.
- **Colab CPU render speed is unmeasured.** The 3-5 min figure is the laptop's
  measured 1.30x realtime divided by an assumed 2-4x CPU advantage. Clock it once.
- **No guard exists yet against rendering on a GPU runtime.** Until
  `colab_render.ipynb` has the `HAS_GPU` check, this is an operator-discipline
  problem that costs real units when it goes wrong.
