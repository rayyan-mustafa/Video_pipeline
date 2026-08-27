# Pipeline plan and handoff

Last updated: 2026-08-23. Companion to `vo_coach/STATUS.md`, not a replacement.

**STATUS.md** documents the voice coach: what it does, what it measured, what it
refuses to do. **This file** documents the *whole video pipeline* end to end —
where the seams are, which ones are automated, which ones are still you moving
files by hand, and what to build next.

Read `The flow, seam by seam` first. It is the only section that answers
"is this crystal clear or not".

---

## Where things stand in one paragraph

The script half is **built, run for real, and costed**. The notebook is **split
into a local half and two Colab phases**, all verified. The visuals half is
**built but never run end to end on a real script**. The voice half is now
**connected**: the script writes per-section read sheets, `collect_takes.py`
moves a finished take into place, and `run_qc.py` measures it before any GPU is
paid for. **One thing still blocks a first real video: the input gain is still
clipping**, and that is a Windows setting, not code.

---

## The flow, seam by seam

Every arrow is a handoff. `AUTO` means a file appears where the next step looks
for it. `MANUAL` means you move or rename something. `GAP` means it does not
exist yet.

```
LAPTOP
  1  run_script.py --topic X                                    ~$0.44, ~15 min
       -> source_pack.json  evidence_profile.json  beat_sheet.json
          script_tagged.json  vo_raw.txt  packaging.json  style.json
          sections/section_NN.txt  sections/README.txt  manifest.json
                                                                        AUTO
  2  the read sheets are written by step 1                              AUTO
       One file per section, grouped by the ALIGNER's chunking, not by the
       section tags - see gap 1 below for why that distinction matters.
       sections/README.txt lists the exact commands for steps 3-5.

  3  vo_coach mark-script sections/section_NN.txt --niche zenn
       -> section_NN_marked.txt  (pause targets, emphasis, one note/line)  AUTO
          the path is in sections/README.txt; nothing to paste

  4  vo_coach live --script sections/section_NN_marked.txt --niche zenn
       -> live_TS.wav (raw)  live_TS_comped.wav (the keeper)             AUTO

  5  collect_takes.py --section NN                                       AUTO
       newest unused comped take -> voice_recordings/section_NN.wav, with
       a one-line QC readout and a provenance record. Refuses to overwrite
       a section, and refuses to use one take for two sections.
       Do NOT run `vo_coach master` first. See "Who masters what".

  5b run_qc.py                                          seconds, no GPU  AUTO
       Clipping, level, SNR, duration vs the read sheet, missing files.
       Exit 1 means re-record now, while it is still free.

COLAB PHASE 1  (T4 GPU)
  6  Step 0 1a 1b 1c 2 3 4 5                                            AUTO
       Step 5 reads what run_script.py wrote, and refuses if Drive is
       still syncing.
  7  Step 16 audio tools                                                AUTO
  8  Step 17 align   Whisper large-v3, word timestamps
       -> vo_raw.wav  scenes_timeline.json                              AUTO
  9  Step 18 master  timeline profile, LENGTH LOCKED
       -> full_vo.wav                                                   AUTO
 10  Step 19 QA (optional)                                              AUTO
 11  Step 20 21 look + character  -> character_ref.png                  AUTO

LAPTOP  (can run in parallel with 11)
 12  run_scenes.py  -> scenes_planned.json               ~$0.003, ~2 min AUTO

COLAB PHASE 1  (still T4)
 13  Step 23 draw   -> frames/                            ~35 min, 1.07 units
 14  Step 26 thumbnail -> thumbnails/                                   AUTO
       then: Runtime -> Change runtime type -> CPU

COLAB PHASE 2  (CPU, free)
 15  Step 24 review -> redraw_queue.json                                AUTO
       if the queue is non-empty: back to a T4 for Step 24b, then return
 16  Step 25 encode -> final_video.mp4                                  AUTO
 17  Step 27 archive -> runs/<topic>_<stamp>/                           AUTO
 18  Step 28 upload                                                     AUTO
```

**Verdict: every arrow is now AUTO.** The only human steps left are the ones
that have to be human — reading the script into a microphone, and clicking
through the frame review. What used to be ragged (steps 2, 3 and 5) is three
commands copied out of `sections/README.txt`.

---

## Who masters what, and why it matters

This confused things once and will again.

| stage | what it does | profile |
| --- | --- | --- |
| vo_coach comp | joins keeper takes with real room tone | **none** |
| notebook Step 17 | masters each section before aligning | `clip` |
| notebook Step 18 | masters the concatenation, length-locked | `timeline` |

**Hand the notebook the COMPED, UNMASTERED file.** The notebook masters twice by
design — per section before alignment (so the timestamps describe the audio you
ship), then once over the whole thing. Running `vo_coach master` first makes it
three times, and a limiter applied three times sounds squashed.

**`full_vo.wav` is the final audio.** vo_coach never produces the deliverable; it
produces good raw material.

### ⚠️ The ordering constraint, restated

From STATUS.md: anything that changes duration — comping, `fix-pauses` — **must
happen before notebook Step 18**, which locks the length so the pictures stay in
sync. In the flow above, that means all of it happens on the laptop, in steps
3–5. Never after step 9.

---

## What is built and verified

| | state |
| --- | --- |
| `run_script.py` steps 6–15 | **run for real**, 59 calls, $0.0093 on the cheap preset |
| `run_script.py` step 15b | read sheets, parity-tested against the notebook's chunker |
| `run_qc.py` + `pipeline/qc.py` | the QC gate: clipping, level, SNR, length, the set |
| `collect_takes.py` | take → `section_NN.wav`, with a ledger and two refusals |
| `run_scenes.py` step 22 | **run for real**, 29/29 scenes, $0.0005 |
| OpenRouter engine | one key, per-model provider pins, real billed cost |
| Five model roles | writer/judge/extractor = Sonnet 5, checker = Haiku, scenes = DeepSeek V4 Flash |
| Notebook split | 54 cells, Phase 1 (T4) + Phase 2 (free CPU) |
| Whisper | `large-v3`, one setting, both call sites |
| Tests | `test_pipeline.py` 37 checks, `test_notebook.py` 13 checks, no API calls |

Costs, measured or projected:

```
script (Claude preset, projected)   ~$0.44 / video
script (cheap preset, measured)     ~$0.010 / video
compute units at 60 videos/month    ~85 of 100
```

---

## The gaps, in the order they will bite you

### 1. Nothing tells you what to read — **BUILT**

`run_script.py` now writes `sections/section_NN.txt`, one per section, plus
`README.txt` with the exact commands and `manifest.json` with the word counts
the QC gate checks against. `pipeline/readsheet.py`.

**One thing changed from the plan above.** It does *not* group tagged sentences
by their `section` field. `build_chunks()` merges a short sentence into the
chunk that is already open, and that chunk keeps the section of the sentence
that *opened* it — so a three-word line at the top of section 4 belongs to a
chunk the aligner files under section 3. Grouping by tag would have put it in
the wrong read sheet, dropped the word-match rate in both sections, and turned
their scene boundaries into interpolated guesses. That is exactly the failure
this was meant to prevent, arriving by a different door.

So the chunker is a verbatim copy of the notebook's, and `test_notebook.py`
executes the real Step 17 cell and asserts the two agree. A copy that drifts is
worse than no copy.

### 2. The input gain is still too hot — **the only thing left blocking a video**

STATUS.md blocker #2, unchanged: a measured 225 s session held **541 runs of
full-scale samples**, roughly a fifth of all four-second windows clipping. It is
a Windows input-volume setting, not code.

Clipping destroys waveform information permanently. No model recovers it. It
flows straight into the pictures:

```
clipped audio -> worse transcript -> difflib matches fewer words
             -> scene boundaries land wrong -> frames drift against your voice
```

`large-v3` is *more* robust to it than `base` was, and will spend longer on bad
windows because of Whisper's temperature-fallback retries (~6–10 min instead of
~4 on an 18-minute script). That is a symptom, not a new problem.

### 3. No QC gate before GPU spend — **BUILT**

`run_qc.py`, on the laptop, over `voice_recordings/*.wav`. No model, no GPU, no
network — seconds. Per section: peak level, **runs of full-scale samples**, noise
floor and SNR, duration against the read sheet's word count. Across sections:
missing files, mismatched sample rates and channel counts, and level jumps big
enough to be audible after Step 18 masters the whole thing as one piece.

`PASS` / `WARN` / `FAIL` per section, with the fix in words. Exit 1 on any FAIL.
`--section N` spot-checks one while you are still recording; `--strict` fails on
warnings; `--json` writes the report.

The measurements are vo_coach's, ported rather than imported — `pipeline/` stays
installable with numpy as its only numeric dependency, and `pipeline/qc.py`
parses the RIFF chunks itself because the standard library's `wave` refuses the
float WAVs vo_coach writes at 32-bit. Ported code drifts, so the thresholds and
the algorithms are pinned by tests: one full-scale sample is *not* clipping,
three consecutive are; a gated file reports *no* SNR rather than a fabricated
~175 dB.

Pace is checked only widely enough to catch the wrong file (70–260 wpm fails,
110–200 warns). Judging delivery stays vo_coach's job — and needs gap 5.

### 4. vo_coach output → `voice_recordings/` is a manual rename — **BUILT**

`collect_takes.py --section N` copies the newest unused comped take into place,
prints a one-line QC readout while the mic is still in front of you, and writes
`voice_recordings/_takes.json` so "which read is this?" has an answer later.

It refuses the two mistakes that are otherwise silent: overwriting a section
that already has audio, and using one take for two sections. `--force` overrides
both, `--take` names one explicitly, `--from` takes any file, `--list` shows the
comped takes, `--status` shows what is recorded and where each came from.

### 5. Calibration takes were never recorded

STATUS.md: the personal baseline is built and tested, but **the three natural
takes do not exist**, so pacing and pronunciation feedback are still judged
against US-English norms. Until then, do not trust those two metrics.

Blocked on the gain (a baseline built from clipped audio bakes the fault in).

### 6. Prompt caching — the biggest remaining cost lever

`apply_fixes` re-sends the beat sheet, the full source block, the numbers and
entities on **every** call: ~7k tokens, 5–8 times per run. Anthropic prompt
caching through OpenRouter would cut ~90% off that.

Worth roughly **$0.09 a video** — more than the scene-model change was worth, and
about 20% of the total. Not implemented.

---

## Next steps, in order

Gaps 1, 3 and 4 are built and tested. What is left is not code:

1. **Fix the input gain.** Windows input volume, peaks near −9 dBFS. Nothing
   downstream is trustworthy until this is done, and it is not a code task.
   `run_qc.py` now tells you when it is right: record thirty seconds, drop it in
   as any section, and run it. Clipping at zero and SNR over 30 means done.
2. **Record the three calibration takes**, then `vo_coach calibrate` and set
   `baseline.enabled: true`. Blocked on 1 — a baseline built from clipped audio
   bakes the fault in.
3. **Run one real video end to end** on `--models claude`. Budget ~$0.50 and a
   T4 session. This is the first time the whole chain runs on real material —
   expect it to surface things this document does not predict. In particular:
   - the read sheets have never been read aloud, so the first real number to
     write down is Step 17's **worst-section word match** (under 75% means the
     text and the audio disagree somewhere)
   - count the frames you queue for redraw in Step 24; see *The visuals*
4. **Then optimise.** Prompt caching (gap 6) is worth ~$0.09 a video, but every
   number past this point is projected and one real run replaces all of them
   with measurements.

---

## Super-next: after the first video works

- **Prompt caching** (gap 6). ~20% off the API bill.
- **LLM-marked emphasis at `mark-script` time.** STATUS.md calls this the
  highest-value thing to build once a key exists. One call per script, zero
  latency while reading, replaces the offline heuristic that gets BEHAVING wrong.
- **Golden-set regression tests for the coaching**, then LLM-as-judge — but only
  once real takes exist to judge. Judging synthetic takes measures nothing.
- **Clause-level comping across separate takes.** `comp.py` assembles within one
  session only.
- **A learnings loop.** `learnings.json` is read by the retention judge and the
  packaging prompt and is empty. Hand-write notes after the first few videos
  publish; that is worth more than any model upgrade.

---

## The visuals: what we do next

The picture half is the least-tested part of the pipeline. What is solid, and
what is not:

**Solid, because it is deterministic and done in Python, not by the model:**

- the colour palette lock (every frame snapped to 6 colours)
- the caption lettering (PIL, same font, same place, never FLUX)
- the prompt assembly — `STYLE_PREFIX` verbatim on every frame, `CHARACTER_BIBLE`
  verbatim on every frame *that has a character*, and both shared with the
  notebook through `style.json` so a redrawn frame cannot drift from the other 159
- **the shot rotation** — `pipeline/shots.py`, new. Seven shot types assigned
  locally before any call goes out, seeded so a timeline replans identically

### The shot rotation, and why it was needed

The four consistency rules in `config.py` held the *look* constant, and read too
broadly they held the *picture* constant too. `STYLE_PREFIX` carried the wall and
the floor, so every one of 160 frames was a figure standing in the same room, and
the only variation lever was `SECTION_DIRECTION` at six-bucket granularity — a
30-scene section produced 30 near-identical compositions.

```
repeated LINEWORK    reads as a style
repeated COMPOSITION reads as a template
```

The reference channel holds the linework and the character constant and changes
the composition on every beat, alternating narrative frames with abstract ones
(a bar chart, an icon grid, a machine with nobody in it, one big word). That is
the distinction the rotation implements.

What it guarantees, in code, asserted by 10 checks in `test_pipeline.py`:

| | |
| --- | --- |
| no two consecutive frames get the same shot | enforced in `assign()` |
| an abstract frame breaks every run of 3 narrative ones | `MAX_NARRATIVE_RUN` |
| no shot exceeds its own ceiling | `max_share`, per shot |
| every shot type appears | validated, warns if >2 unused |
| at most 35% of frames carry text | `enforce_caption_cap()` |
| `CHARACTER_BIBLE` is in a prompt iff the shot has a character | 4 of 7 do not |
| the same timeline replans identically | seeded, so a redraw is reproducible |

That last row of the previous behaviour is worth naming: `visual_prompt()`
concatenated `CHARACTER_BIBLE` unconditionally, so a bar-chart frame came out
with a stick figure glued to it. The background split fell out of the same bug —
leaving the floor in `STYLE_PREFIX` while a chart's clause says "no floor line"
hands FLUX a prompt that contradicts itself.

Assignment is **not** the model's job, for the same reason the palette lock is
not: a rule you can only ask for is not a rule. The model is told which shot to
write for and writes the subject; it never picks. Same for the caption cap — a
bias in the prompt is a nudge, and 160 independent nudges do not add up to a cap.

```
python run_scenes.py --shots      the rotation only: no key, no calls, no cost
python run_scenes.py --strict     refuse to finish if the validator complains
```

`--shots` is how you tune `shots.py` without paying to find out. Measured on a
realistic 160-scene, 6-section script: 0 adjacent repeats, longest narrative run
exactly 3, all 7 shots used, character-moment 31%, data-chart 6%.

### Expressions: the one thing kept from the Zenn palette experiment

A larger change was tried and **reverted**: a saturated cyan/yellow palette, a
framing axis and per-scene caption placement, taken from the reference channel's
look. Real frames showed it was not this channel's look — the warm
white-wall/tan-floor room is. `config.PALETTE`, `STYLE_PREFIX` and the shot
clauses are all back to what they were.

What survived is the **face**, in `pipeline/look.py`. `CHARACTER_BIBLE` used to
end "…two short curved black lines for eyes, one small curved line for the
mouth" — one frozen neutral expression on all 160 frames. The construction of
the figure is still constant (circle head, white fill, thin line limbs, no
colour), but the eyes-and-mouth clause is now chosen per scene from six.

It is chosen from a free **lexical energy score** of the line — punctuation,
sentence length and a small word list, no model call on the pipeline's
highest-volume step. A visceral beat gets a shocked or enraged face; a
contemplative one gets a deadpan. Assigned in Python before any call goes out,
seeded, capped per expression, and validated — same contract as the shot
rotation, and for the same reason: a rule you can only ask for is not a rule.

**This is the one axis that genuinely spends drift budget**, since it changes the
face, which is the thing that has to stay recognisable across 160 frames. It is
on because a single frozen face is the templated look in miniature.
`EXPRESSIONS_ON = False` turns it off in one edit.

**Still deferred, unchanged:** per-scene colour and varied caption placement.
Both still need a redraw-rate measurement from one real run first (see step 1 of
*The plan*).

**Not solid: character consistency.** The notebook says so itself — `FLUX_SEED`
does not give character consistency on its own. Repeating the bible verbatim
helps and does not guarantee. Across 160 frames some will not look like the same
person.

**The plan, in order:**

1. **Measure it first.** On the first real run, count how many frames you queue
   for redraw in Step 24. That number decides everything:
   - under ~5%: the current approach is fine, redraw and move on
   - 10–20%: worth a real fix
   - over 20%: the review becomes the bottleneck and must be fixed before scaling
2. **If it needs fixing**, in increasing order of effort:
   - tighten `CHARACTER_BIBLE` — fewer, more mechanical descriptors beat more
     adjectives
   - a **character LoRA** trained on 15–20 accepted frames of your own stickman.
     This is the real answer to consistency and costs one short training run.
   - IP-Adapter / reference conditioning off `character_ref.png` — more moving
     parts, needs the diffusion stack extended
3. **Batch redraw is already the right shape** for whichever fix wins: Step 24
   queues, Step 24b draws the queue in one GPU session with FLUX loaded once.

Do not touch the palette or caption code. Those are the parts that are working
*because* they are not the model's job.

---

## Decisions not worth re-litigating

- **The split is by what the output is, not by how hard the task feels.** Prose a
  viewer hears and judgment calls get Sonnet. Machinery gets the cheapest model
  that does the job.
- **Provider pinning is mandatory.** A model with no pin is refused, not routed.
- **Reasoning is off for the cheap models, measured not assumed.** `effort: low`
  made Ling think *more*; `exclude: true` hides the thinking and still bills it.
- **Stage boundaries are files.** Not variables, not a shipped package. Every
  handoff between machines is a file in the Drive folder with a size recorded in
  `_stages.json`.
- **Colab bills wall-clock, not GPU use.** That is the whole reason for the
  two-phase split, and the reason the frame review moved off the T4.
- **Nothing in `pipeline/` imports torch.** The moment one module does, the local
  half stops being installable and the split is gone.

---

## Files worth knowing

```
channels/
  README.md            the split, the file contract, the recording seam
  run_script.py        steps 6-15        run_scenes.py   step 22
  run_qc.py            the QC gate      collect_takes.py  the take handoff
  test_pipeline.py     37 checks         test_notebook.py  13 checks
  stickman.ipynb       54 cells, two phases
  stickman.ipynb.bak   the 71-cell original, before any of this
  pipeline/            16 modules, no torch anywhere
    shots.py           the shot rotation: the anti-repetition guarantees

vo_coach/
  STATUS.md            the voice coach: what it measured and what it refuses to do

PLAN.md                this file
```
