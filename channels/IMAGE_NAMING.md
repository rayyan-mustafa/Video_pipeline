# The image workflow — naming, planning, stitching

## The rule

```text
<order>section<section>.<ext>          1section1.png, 2section1.png, 1section2.png
```

`section` is which read-sheet section it belongs to. `order` is its position
inside that section. One flat folder — no subfolders, the section is in the name.

Sorting is by `(section, order)`, so the only thing that must be true is **no two
files share the same (section, order) pair**. Numbers compare numerically, so
`10section3` lands after `9section3`. No zero-padding needed.

Tolerated spellings: `1_section_1`, `1 sec 1`, `1s1`.

## Why the plan has no timestamps in it

`script_plan.json` binds each image to **the exact sentences it illustrates** and
stores no seconds at all. At build time Whisper timestamps those same sentences
and each image inherits the span of the scenes it owns:

```text
image start = first owned scene's start
image end   = last owned scene's end
```

So timing is *derived*, never stored. Three consequences worth knowing:

- You can generate pictures **before or during** recording — the plan does not
  wait on audio.
- Re-record a section as fast or as slow as you like; the bindings still hold.
- Nothing has to guess how to divide a section between its pictures, because the
  binding already says.

The join is positional (scene N in the plan is scene N in the timeline), so
`script_plan.json` carries a **fingerprint** of its scene texts. Edit the script
after planning and the build refuses to stitch rather than shifting every picture
by one sentence.

## The sequence

```powershell
$py = "..\vo_coach\.venv\Scripts\python.exe"

# 1. plan from the script -> read sheets + bindings + prompts
&$py plan_script.py ..\vo_coach\data\scripts\chin.txt --work _build `
     --topic "the human chin" --seconds-per-image 8
&$py make_nb_prompts.py --work _build
&$py make_nb_prompts.py --work _build --repair    # only if one came back blank

# 2. record _build\sections\section_01.txt ... into voice_recordings\
#    generate the images from _build\nb_prompts.txt into images\

# 3. build
&$py build_video.py --stage voice --work _build
&$py build_video.py --stage plan  --work _build --images images
&$py build_video.py --stage mux   --work _build --out final_video.mp4
```

`--stage voice` is the slow one (Whisper). It caches, so re-running `plan` and
`mux` after replacing an image costs seconds.

### How many images

`--seconds-per-image` sets it. On the 1314-word chin script:

| setting | images |
| --- | --- |
| 8s | 66 |
| 12s | 57 |
| 16s | 46 |
| 20s | 34 |

A slot never spans a section boundary — sections are separate recordings, so a
picture straddling one would be cut across two takes.

## Scripts must contain no brackets

`plan_script.py` refuses to plan a script containing `[...]`, `(...)`, `{...}`
or `<...>`. Bracketed stage directions get read aloud, and Whisper then aligns
real audio against words the script claims are spoken. `--allow-brackets`
overrides, but fix the script instead.

## What the audio does on its own

Hand over `voice_recordings/section_01.wav …` **unmastered**:

1. each section gets the `clip` master — high-pass, denoise, EQ, de-ess,
   compress, **trim silence**, normalise;
2. Whisper transcribes the mastered section and aligns what it heard against the
   script, so every scene start is a real timestamp;
3. sections concatenate into `vo_raw.wav`;
4. the whole thing gets the `timeline` master — same chain **minus the trim**,
   length-locked by assert — at −14 LUFS for YouTube.

Trimming moves every timestamp, which is why mastering runs *before* alignment.
Do not master these files yourself: two passes is the design, three sounds
squashed.

## Files

| file | what it is |
| --- | --- |
| `plan_script.py` | script → read sheets, bindings, `script_plan.json` |
| `make_nb_prompts.py` | `script_plan.json` → `nb_prompts.txt`, one per image |
| `build_video.py` | voice → align → bind → mux |
| `pipeline/imageplan.py` | the binding and its guards |
| `pipeline/voice.py` | the mastering chain (ported from the notebook) |
| `pipeline/align.py` | scene splitting + word alignment (ported) |

`scenes_planned.json` is **not** used in this workflow — that belongs to the
generated-frames path (`scenes.py` → notebook Step 23). `nb_prompts.txt` replaces
it. `scenes_timeline.json` is still the spine and is unchanged in shape.
