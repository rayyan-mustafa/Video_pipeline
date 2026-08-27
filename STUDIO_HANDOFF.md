# HANDOFF — Video Studio (local React UI over the pipeline)

Written 2026-08-27. Paste this whole file into a new chat to continue.

Supersedes nothing — read alongside `channels/HANDOFF_VIDEO_BUILD.md`, which
covers the CLI pipeline this UI drives. `STATUS.md` / `NEXT_SESSION.md` cover the
older *visual* architecture (draw.py, props, LoRA) and are not needed for this work.

---

## One-paragraph state

The chin video builds end to end from the CLI. We are now wrapping that pipeline
in a local React UI so it can be driven without terminal commands, and adding the
one thing it genuinely lacks: **a timeline editor to fix what Whisper got wrong**
rather than re-recording. **Phases 0–2 are done and committed.** Backend runs on
`127.0.0.1:8770`. No frontend exists yet — that is the next thing to build.

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
```

A one-second hold is **not a defect** — it is how a specific point in the
narration gets emphasised. Accents should stay under ~25% of frames, or they stop
reading as emphasis.

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

Not fixable by re-planning — it needs ~100 more pictures, or motion (Ken Burns),
which `NEXT_SESSION.md` already flags as the highest quality-per-effort win and
which is still unbuilt.

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

## Phase status

```text
  ✅ Phase 0  git init, .gitignore, secrets kept out of history
  ✅ Phase 1  backend skeleton: projects, jobs+SSE, guarded file serving
  ✅ Phase 2  script / plan / images endpoints + pacing as first-class
  ⬜ Phase 3  Colab worker (tunnel) + voice stage      <- NEXT
  ⬜ Phase 4  the timeline editor                      <- the centrepiece
  ⬜ Phase 5  render + preview
  ⬜ Phase 6  AI script generation
  ⬜ (unscheduled) the React frontend — nothing exists yet
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
React (Vite, :5173) ──HTTP/SSE──▶ FastAPI (:8770) ──import──▶ channels/pipeline/*
                                        │
                                        └──HTTPS──▶ Colab worker (cloudflared URL)
                                                    large-v3 transcribe, optional mux
```

- **Short pure functions are imported** (`imageplan`, `scenesplit`, `pacing`,
  `Project`) — the UI can never disagree with the CLI because it runs the same code.
- **Long stages are subprocesses** (`build_voice`, `mux`, `run_script`) — they are
  `argparse main()` + `print()` + `SystemExit`, so in-process calls would take the
  server down. `jobs.py` streams their stdout over SSE.

### Decisions already made

| | |
|---|---|
| Colab link | tunnel — Colab runs FastAPI + cloudflared, URL pasted into UI per session |
| Whisper | `large-v3` on GPU, never tiny/base for real runs |
| Render | local ffmpeg default, Colab optional |
| Voice input | upload `.wav` per section |
| AI scripts | picker over the existing 15 `config.TOPICS` only |
| Editor scope | roll edits, replace, reorder — **not** split/merge (see Phase 4) |

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

### New this session
```text
channels/pipeline/pacing.py        the pacing spec, shared by planner + checker
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
.gitignore                  media, venvs, notebooks, build dirs
NEXT_SESSION.md             redacted a verbatim HF token
STEP11_RESUME.md            redacted a verbatim HF token
```

### Still to build
```text
studio/backend/settings.py      colab_url, render target
studio/backend/colab.py         ColabClient over the tunnel
studio/backend/voicestage.py    transcribe: local or Colab
studio/backend/timeline.py      editor data source
studio/backend/overrides.py     build/overrides.json
studio/backend/rendering.py     mux: local or Colab
studio/colab_worker.ipynb       FastAPI + cloudflared worker
studio/frontend/                Vite + React + TypeScript — nothing exists yet
```

---

## Next: Phase 3 — Colab worker + voice

1. `studio/colab_worker.ipynb`: install deps, preload `large-v3` **once**, expose
   `/health` and `/transcribe` (multipart zip in, zip out), open a cloudflared
   quick tunnel (no account needed) and print the URL.
2. `settings.py` + `colab.py`: store the pasted URL, health-check it, POST bundles.
3. `voicestage.py`: local subprocess path *and* Colab path behind one interface.
4. Upload UI for per-section `.wav`.
5. Results must show **word-match %** and **anchored N/M** as different things —
   see the trap below.

### Phase 4 — the timeline editor (the centrepiece)

Included: **roll edits** (drag a cut, snapped to real Whisper word timestamps from
`build/words.json`), **replace** (swap the file on a slot, including the 17 unused
`_2` alternates), **reorder** (drag between slots = two replaces).

Excluded deliberately: **split/merge slots** — that changes the image↔sentence
binding the fingerprint protects. It belongs on the Plan page, where split modes
live.

Persistence — `<project>/build/overrides.json`, the editor's only output:
```json
{ "version": 1, "fingerprint": "b5da03f977a7c0d4",
  "cuts": {"7": 42.31}, "assets": {"12": "9section3_....jpeg"} }
```
Apply in `imageplan.plan_from_binding()` via a new `overrides` param, to the
`starts` list **before** the existing tiling/min-hold pass, so gaps, overlaps and
flashes stay structurally impossible. Stale fingerprint → warn, do not apply.

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
- **Chin fails the pacing spec** (6.7s body average vs 2-4s band).
- **No motion** — mux is a plain concat of stills, no zoompan, no transitions.
- **No captions** in this path.
- **`vo_coach/.venv` has install drift** — `librosa`, `numba`, `torch`,
  `silero-vad` are pinned but missing (Defender zeroed files once). None are
  needed for Studio; all 9 packages we do need import cleanly.
- **`run_script.py` topics are a fixed bank of 15**, each needing curated source
  URLs. Free-text topics are not supported.
