# HANDOFF — the local video build (chin video, and every one after it)

Written 2026-08-27. Paste this whole file into a new chat.

This covers the **local image + audio → mp4 pipeline** built this session. It is
separate from, and currently replaces, the notebook's generated-frames path.
`STATUS.md` and `NEXT_SESSION.md` still hold the visual-architecture history —
read those only for the *why* behind the older decisions.

---

## The one-paragraph version

The chin video is fully planned and one step from rendering. 105 hand-made images
are bound **1:1 to the script's 105 scenes**, verified by opening the pictures and
matching them against the sentences. The only thing standing between here and a
finished mp4 is the **voice stage**: the six recordings need renaming to the
expected scheme, then Whisper (`large-v3`, on Colab Pro) aligns them and the
remaining two commands run locally in seconds.

---

## Where we are

```text
  ✅ DONE   1. project structure: channels/wierdbiology/<topic>/
  ✅ DONE   2. script checked - zero brackets, 5 sections, 1314 words, ~8.8 min
  ✅ DONE   3. 105 images in place, named <order>section<N>_<timestamp>.jpeg
  ✅ DONE   4. 17 "_2" regenerations detected and excluded (kept on disk, unused)
  ✅ DONE   5. script split to match the images EXACTLY - 105 scenes, 1:1
  ✅ DONE   6. binding verified by eye: image 1 = "Touch your chin." and it is
  ✅ DONE   7. --stage check passes: even pacing, ~5.0s per image
  ────────────────────────────────────────────────────────────────────────
  ⏳ NOW    8. rename the 6 recordings          <-- BLOCKED, see below
            9. run the voice stage on Colab with large-v3
  ────────────────────────────────────────────────────────────────────────
  ⬜ NEXT  10. --stage plan   (seconds, local)
           11. read build/sync_report.txt and eyeball the pairing
           12. --stage mux    (a few minutes, local)
```

---

## STEP 8 — the one open question, resolve it first

`voice_recordings/` holds **six** files but the script has **five** sections:

```text
1.wav   36.07s      4.wav   87.73s
2.wav   81.40s      5.wav  153.34s
3.wav  165.62s      6.wav   78.17s        total 602.3s = 10:02
```

Arithmetic says section 3 was recorded in two takes — `3.wav` + `4.wav`:

| section | words | file(s) | implied pace |
| --- | --- | --- | --- |
| 1 | 89 | 1.wav | 148 wpm |
| 2 | 180 | 2.wav | 133 wpm |
| 3 | 520 | 3.wav + 4.wav | 123 wpm |
| 4 | 354 | 5.wav | 139 wpm |
| 5 | 171 | 6.wav | 131 wpm |

Every section lands in 123–148 wpm, a consistent human read. No other pairing
works — mapping `2.wav` to section 1 implies 66 wpm, which is impossible.

**This is a hypothesis, not a verified fact.** It was never confirmed against the
audio. Confirm it (transcribe 15 seconds off the front of `4.wav` and see whether
it continues section 3 or starts section 4), then rename:

```powershell
cd wierdbiology\chin\voice_recordings
ren 1.wav section_01.wav
ren 2.wav section_02.wav
ren 3.wav section_03_a.wav      # multi-part: joined automatically, in name order
ren 4.wav section_03_b.wav
ren 5.wav section_04.wav
ren 6.wav section_05.wav
```

Multi-part sections are supported: `section_03_a` + `section_03_b` are
concatenated in filename order and treated as one section. Do **not** splice them
by hand — splicing changes duration, and every timestamp downstream is derived
from the duration this stage measures.

---

## STEP 9 — the voice stage on Colab

Full instructions in **[COLAB_VOICE.md](COLAB_VOICE.md)**. The short form:

```powershell
..\vo_coach\.venv\Scripts\python.exe make_colab_bundle.py --project wierdbiology/chin
```

That writes `wierdbiology\chin\chin_voice_bundle.zip` (34.5 MB) containing the
read sheets, the recordings, five pipeline modules, and the split mode. Upload it
to Colab, run the five cells in COLAB_VOICE.md, download `voice_out.zip`, extract
into `wierdbiology\chin\build\`.

**Only the voice stage goes to Colab.** Planning is arithmetic and muxing is
libx264 on CPU — Colab bills wall-clock, so encoding there spends compute units
on work the laptop does identically. The notebook's own Step 25 says the same.

**The mastering must run in Colab too, not locally first.** The `clip` profile
trims silence, which moves every timestamp, so alignment has to see the audio
that actually ships.

What comes back is exactly three files, and they are the whole contract with the
next stage: `scenes_timeline.json`, `words.json`, `full_vo.wav`.

**Watch the word-match % per section.** Below 75% means Whisper could not follow
the script there and those scene boundaries are guesses — re-record rather than
ship it.

### Running it locally instead

Viable — faster-whisper is installed in `vo_coach\.venv`:

```powershell
..\vo_coach\.venv\Scripts\python.exe build_video.py --project wierdbiology/chin --stage voice --whisper base
```

`base` on CPU is ~8–12 min for 10 min of audio; `large-v3` on CPU is roughly 10x
that. Device is auto-detected (CUDA+float16 when available, else CPU+int8;
pre-Volta falls back to int8 because float16 is invalid there).

---

## STEPS 10–12 — finish locally

```powershell
$py = "..\vo_coach\.venv\Scripts\python.exe"
&$py build_video.py --project wierdbiology/chin --stage plan
#   read wierdbiology\chin\build\sync_report.txt
&$py build_video.py --project wierdbiology/chin --stage mux
```

`--stage plan` re-measures `full_vo.wav` and asserts timeline == image durations
== audio length, zero gaps, no sub-0.5s flashes, no image outside its section. It
**exits 1** on failure, before any encoding is paid for.

`sync_report.txt` lists every image against the words playing under it. Read it.
It is the last chance to catch a mismatch before the encode.

Output lands at `wierdbiology\chin\chin.mp4`.

**Do NOT pass `--ignore-plan`.** That flag divides sections proportionally and is
only for pictures with no binding. The chin images are bound 1:1 — using it would
throw that away.

---

## How the sync actually works

Three layers, all verified this session:

1. **Section boundaries — exact.** Each section is its own recording, so its span
   is its real measured length.
2. **Image → words — bound, not guessed.** `script_plan.json` maps each image to
   the exact sentence it illustrates. At build time each image inherits that
   scene's span: first owned scene's start → last owned scene's end.
3. **Cut placement — exact word timestamps.** Whisper's per-word timing is kept in
   `words.json` and used directly.

Measured on a variable-pace read, three methods compared:

```text
  equal TIME division      median  1.34s   worst  7.23s
  interpolated in scene    median  0.29s   worst  2.36s
  exact word timestamp     median  0.00s   worst  0.00s
```

The first two are fallbacks for unbound images. The chin video uses neither.

---

## The project layout

```text
channels/wierdbiology/chin/
├── script.txt              the words
├── sections/               section_01.txt … section_05.txt  <- read sheets
├── voice_recordings/       section_01.wav …                 <- NEEDS RENAMING
├── images/                 105 in use + 17 unused "_2"
├── build/                  derived; safe to delete and regenerate
│   ├── script_plan.json    the binding + fingerprint
│   ├── script_plan.md      readable shot list
│   ├── scenes_timeline.json  } from the voice stage
│   ├── words.json            }
│   ├── full_vo.wav           }
│   ├── render_plan.json    from --stage plan
│   └── sync_report.txt     image → narration, for eyeballing
└── chin.mp4                the output
```

Everything derives from `--project wierdbiology/chin`. A second topic is
`wierdbiology/knuckles` — because the project name is in every path, two topics
cannot cross-contaminate.

---

## The commands

| command | what it does |
| --- | --- |
| `plan_script.py --project P --split match` | script + images → 1:1 binding |
| `make_nb_prompts.py --project P` | plan → one Nano Banana prompt per image |
| `make_colab_bundle.py --project P` | zip the voice stage for Colab |
| `build_video.py --project P --stage check` | verify image naming + pacing, no audio needed |
| `build_video.py --project P --stage voice` | master → Whisper → align |
| `build_video.py --project P --stage plan` | bind images to time + sync checks |
| `build_video.py --project P --stage mux` | ffmpeg → mp4 |

Always use `..\vo_coach\.venv\Scripts\python.exe` — that venv has faster-whisper,
noisereduce and pyloudnorm. `channels\.venv` does not.

---

## The image naming convention

```text
<order>section<section>[_anything].<ext>      10section4_202608271333.jpeg
```

Pattern may **lead** (with a trailing timestamp) or **trail** (after a prefix).
Sorting is by `(section, order)`; only that pair must be unique. Numbers compare
numerically, so `10section3` follows `9section3` — no zero-padding.

A trailing `_2`…`_9` marks a **regeneration of the same slot**, not an extra
picture. The lowest variant wins and the rest are reported, never silently
dropped. That is what excludes the 17 section-4 alternates.

---

## Traps already paid for

**The `_2` files are not extra scenes.** Treating all 122 as distinct would put 53
images into section 4's 36 sentences and shift everything after them. Verified by
opening `10section4` and `10section4_2` — a caveman and a modern man, same slot.

**Never divide a section by equal time.** Sentences are not evenly spaced; on a
212-second section that put cuts up to 10.7s from their narration.

**Snapping cuts to sentence boundaries collapses them when images outnumber
scenes.** Several cuts compete for one boundary — measured at 47 images under
0.5s (visible flashes). Boundaries are now claimed once each, and when images
outnumber scenes nothing snaps.

**A script edited after planning breaks every binding.** `script_plan.json`
carries a fingerprint of its scene texts; the build refuses to stitch against a
mismatched timeline rather than shifting every picture by one sentence.

**Bracketed text in a script is fatal.** It gets read aloud, and Whisper then
aligns real audio against words the script claims are spoken. `plan_script.py`
refuses to plan such a script.

**colab-mcp cannot be opened from here.** `open_colab_browser_connection` is an
*injected* tool that only exists while the server runs, and its URL carries a
token and port generated inside that process. Connect with `/mcp` in the
terminal. `.mcp.json` has `timeout: 600000` but that only applies from a Claude
Code restart. None of COLAB_VOICE.md needs colab-mcp — the upload/download cells
work in a plain browser tab.

**Never `display()` an image through colab-mcp.** The WebSocket caps a frame at
1,048,576 bytes and silently eats `update_cell` writes. (From `STATUS.md`; still
true.)

---

## For the next script

```powershell
$py = "..\vo_coach\.venv\Scripts\python.exe"
$P  = "wierdbiology/knuckles"
```

1. Write `channels/wierdbiology/knuckles/script.txt`. Blank line between
   sections. No brackets.
2. `&$py plan_script.py --project $P --topic "why knuckles crack" --split dash`
   — prints scene counts under every mode so you can pick.
3. **Decide where the prompts come from.** This is the real fork:
   - `&$py make_nb_prompts.py --project $P` gives one prompt per planned image,
     each bound to exact sentences. Generate from those and the binding is exact
     with no reconciliation.
   - Using an outside agent means the binding is lost and has to be recovered
     afterwards — which is what `--split match` did for the chin video, and it
     only worked because the images happened to be one-per-sentence.
4. Generate into `knuckles/images/`.
5. `&$py build_video.py --project $P --stage check` — 2 seconds, catches naming
   and pacing errors before recording.
6. Record into `knuckles/voice_recordings/` as `section_01.wav …`.
7. voice → plan → mux.

**Record one section first and take it all the way to mp4** before generating the
rest. The chain is tested end to end on synthetic data, but has never run on real
recorded audio.

---

## Known gaps

- **The voice stage has never run on real audio.** Every test used synthetic
  timelines. The first real run is the chin video; expect surprises in the
  word-match percentages.
- **No motion.** The mux is a plain concat of stills — no `zoompan`, no
  transitions. `NEXT_SESSION.md` flags this as the highest perceived-quality win
  for the least effort, and it is still not done.
- **No captions.** The notebook burned them in with PIL; this path does not.
- **The 6-vs-5 recording mapping is unverified** — see STEP 8.
- `make_nb_prompts.py` was written for the 66-image plan and has not been re-run
  since the switch to `--split match`.
