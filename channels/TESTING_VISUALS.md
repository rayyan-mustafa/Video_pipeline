# Testing the visuals in isolation — no recording required

Written 2026-08-24. Companion to `README.md` (the split) and `f:\wierd_bio\PLAN.md`
(the whole pipeline, seam by seam). This file exists so a fresh chat can pick up
**just the visuals question** — do the frames match the narration, and do 150+
frames look like one consistent character — without re-deriving any of this.

---

## The question this answers

> I want to test the FLUX/visuals part on its own: do the scenes get planned
> correctly, do the images match the narration line they're for, and are the
> images consistent (same character, same style) across frames?

You do **not** need to record real audio, run Whisper, or open a T4 for
alignment to test this. The whole visuals chain only reads `scenes_timeline.json`
in the shape `{scene_index, start, end, duration, text, section}` — verified by
reading `pipeline/scenes.py: load_timeline()` directly, which checks exactly
those keys and nothing else. Nothing past that point cares whether `duration`
came from a real recording or an estimate; it only cares about the `text`.

So a **fake timeline**, built from word count instead of a real recording, lets
you test the entire scene-planning → FLUX chain for a fraction of a cent and
zero recording time.

**The one thing a fake timeline is NOT for:** shipping a real video. Its
`duration`/`start`/`end` values are estimates, not measurements of your actual
voice, so muxing them against real audio would desync the pictures. Use it only
to look at frames, never to render `final_video.mp4` for real.

---

## The tool: `make_test_timeline.py`

New, lives at `channels/make_test_timeline.py`. Reads `script_tagged.json`
(written by `run_script.py`), chunks it with the **same `build_chunks()`** the
notebook's real aligner and `pipeline/readsheet.py` use (so the scenes you get
here are the same scenes a real recording would produce — just with estimated
timing instead of measured), and writes `scenes_timeline.json` straight from
word count ÷ wpm.

```powershell
python make_test_timeline.py                     # uses SPOKEN_WPM (150)
python make_test_timeline.py --wpm 140            # if you read slower
python make_test_timeline.py --workdir PATH       # a specific folder
python make_test_timeline.py --force              # overwrite an existing one
```

It refuses to overwrite a real `scenes_timeline.json` (one written by notebook
Step 17) unless you pass `--force` — that guard exists so a real one is never
silently clobbered by a test run.

---

## The full walkthrough

### 1. Get a script

Either a real one:

```powershell
python run_script.py --topic <topic>              # ~$0.44 Claude, or --models cheap for ~$0.01
```

or reuse whatever `script_tagged.json` is already sitting in your working
folder from a previous run — this step only needs that one file plus
`beat_sheet.json`, `vo_raw.txt`, `evidence_profile.json`, `source_pack.json`
(all written together by `run_script.py`, and all required by the notebook's
Step 5 loader — checked directly in the notebook cell).

### 2. Fake the timeline

```powershell
python make_test_timeline.py
```

Writes `scenes_timeline.json`. No recording, no Whisper, no GPU, instant.

### 3. Plan the scenes for real

```powershell
python run_scenes.py --preflight     # confirms the timeline is well-formed, free
python run_scenes.py                 # real API calls, a fraction of a cent to ~$0.01
```

Writes `scenes_planned.json`: one entry per scene with `action_clause` (what
the character does), `caption` (on-screen text, or none), and `visual_prompt`
(the full FLUX prompt — `STYLE_PREFIX` + `CHARACTER_BIBLE`, verbatim, plus the
action).

**This is already the first real check.** Read `scenes_planned.json` and
confirm each `action_clause` actually matches its `text` line — verified
working on a real run:

```
scene 0: "Press two fingers against the back of your arm. Feel that?..."
  action: Character presses two fingers against back of arm
scene 4: "You have almost none of that fur left..."
  action: points at exposed wires on a toy      <- a metaphor for "no fur left", not literal
```

And confirm every `visual_prompt` starts with the identical style/character
block — that repetition, not anything FLUX decides on its own, is what keeps
150 frames looking like one artist and one character. If a `visual_prompt`
starts describing the LOOK instead of the ACTION, that's a bug worth catching
here, before spending GPU time on it — `pipeline/scenes.py`'s own docstring
calls this out explicitly.

### 4. Get the files onto Drive

Colab only sees the shared Drive folder. Copy (or `--workdir` a local run
straight into the Drive-synced path so this step is automatic):

```
scenes_planned.json
scenes_timeline.json   (the fake one — harmless, nothing in Step 20+ reads it directly)
style.json
script_tagged.json, vo_raw.txt, evidence_profile.json, source_pack.json, beat_sheet.json
```

### 5. In the notebook — **skip the voice steps entirely**

```
Step 0            detect runtime
Step 1a 1b 1c     setup (needs a T4 for FLUX)
Step 2 3 4        folder, settings, control panel
Step 5            load the script (reads the files listed above)

  --- SKIP 16, 17, 18, 19 entirely. No recording, no Whisper, no mastering. ---

Step 20 21        lock the look, pick your character (once per channel)
Step 23           draw the frames  <- reads scenes_planned.json, this is the real test
Step 26           thumbnail, if you want it
```

Step 23 is the actual FLUX generation — this is the part that costs GPU time
(~35 min / ~1.07 compute units for a full script's worth of frames; a 7-scene
test batch like the one used to verify this walkthrough costs a small fraction
of that).

### 6. What to actually look at once frames exist

- **Does each frame match its line?** Open `frames/frame_0000.png` etc.
  alongside `scenes_planned.json`'s `text` for that `scene_index`.
- **Character consistency across frames** — this is the real unknown per
  `PLAN.md`'s own visuals section: `FLUX_SEED` does not guarantee character
  consistency on its own; repeating `CHARACTER_BIBLE` verbatim helps but does
  not guarantee it. On this test run, **count how many frames look like a
  different person/pose than the rest.**
  - under ~5%: current approach is fine
  - 10–20%: worth a real fix (tighter `CHARACTER_BIBLE`, then a character LoRA)
  - over 20%: this becomes the bottleneck before scaling to real videos
- **Palette and captions** should be perfect every time — those are enforced
  in code after generation, not left to FLUX, so any drift there is a bug in
  `pipeline/config.py`'s palette lock or the caption drawing, not a FLUX
  inconsistency.

---

## Cost and time, measured

| step | cost/time | GPU? |
| --- | --- | --- |
| `make_test_timeline.py` | free, instant | no |
| `run_scenes.py` (7-scene test) | $0.0001, seconds | no |
| `run_scenes.py` (full ~150-scene script) | ~$0.0005–0.01 | no |
| notebook Steps 0–5, 20–21 | a few minutes | T4 |
| notebook Step 23 (full script) | ~35 min, ~1.07 units | T4 |

Nothing here touches the recording/Whisper cost or time at all — that is the
entire point of testing visuals this way.

---

## What NOT to do with this

- Do not run `run_scenes.py` output built from a fake timeline through Step 25
  (encode) expecting a real, watchable video. The audio does not exist and the
  timing is invented.
- Do not delete or hand-edit a REAL `scenes_timeline.json` (one Step 17 wrote
  from an actual recording) to "test" something — `make_test_timeline.py`
  refuses to overwrite one without `--force` for exactly this reason.
- If a real recording's `scenes_timeline.json` already exists in your working
  folder and you want to test visuals separately, use a **different** working
  folder (`--workdir`) rather than risk mixing the two.
