# Status / handoff

Last updated: 2026-08-23. **324 tests passing. Phases 0-5 all built, plus
timing repair and delivery direction.**

Live mode has now been run against a real microphone (Maono PD300X) and works.
**No real API call has ever been made** - there is still no `OPENROUTER_API_KEY`,
so all coaching seen so far is the offline fallback wording.

READ THESE THREE SECTIONS FIRST if you are picking this up cold:
`Blocked on the user`, `OPEN RISK: the rubric is calibrated for US speakers`,
and `How a session actually runs end to end`.

## Where things stand

| phase | status |
| --- | --- |
| 0 — scaffold, contract, config, SQLite, record/ingest | done |
| 1 — analysis engine (ASR, pacing, prosody, spectral) | done, verified on real audio |
| 2 — rubric + per-niche competitor benchmark | done |
| 3 — mentor (pluggable LLM provider) | built, **never called live** |
| 4 — mastering chain + take ranking | done, verified on real audio |
| — timing repair (`fix-pauses`, `mark-script`) | done, verified on real audio |
| — delivery direction (emphasis + how-to-say-it) | done, offline heuristic |
| 5 — live mode | built, **verified on a real mic** |

## Run it

```bash
.venv/Scripts/python.exe -m pytest -q            # 324 tests
.venv/Scripts/python.exe -m vo_coach analyze <take|path> [--no-asr] [--plot]
.venv/Scripts/python.exe -m vo_coach score <take> --niche zenn
.venv/Scripts/python.exe -m vo_coach benchmark <take> --niche movie_explainer
.venv/Scripts/python.exe -m vo_coach coach <take> --niche zenn
.venv/Scripts/python.exe -m vo_coach master <take|path>
.venv/Scripts/python.exe -m vo_coach rank <take> <take> ...

.venv/Scripts/python.exe -m vo_coach calibrate <take> <take> <take>   # personal baseline
.venv/Scripts/python.exe -m vo_coach baseline                         # show it
.venv/Scripts/python.exe -m vo_coach mentor-audit                     # is the model obeying its rules
.venv/Scripts/python.exe -m vo_coach eval --list                      # the golden set, and why each case exists
.venv/Scripts/python.exe -m vo_coach eval                             # score the configured model (~$0.01)
.venv/Scripts/python.exe -m vo_coach eval --compare                   # every scored model, side by side

.venv/Scripts/python.exe -m vo_coach live --no-mentor                 # HUD only, free, no API calls
.venv/Scripts/python.exe -m vo_coach live --niche zenn                # full live coaching

# the normal working loop
.venv/Scripts/python.exe -m vo_coach mark-script data/scripts/X.txt --niche zenn
.venv/Scripts/python.exe -m vo_coach live --device 1 --niche zenn     --script data/scripts/X_marked.txt --no-mentor
.venv/Scripts/python.exe -m vo_coach fix-pauses <take> --niche zenn
```

## Blocked on the user, not on code

0. **Mentor model: `openai/gpt-oss-120b`, pinned to `cerebras/fp16` with
   `groq` as the in-pin backup.** Switched from `ling-3.0-flash` on 2026-08-22
   after checking the numbers: it is both **faster and a stronger reasoner**.

   | model / host | tps | TTFT | 400-tok round trip | $/call |
   | --- | --- | --- | --- | --- |
   | gpt-oss-120b @ cerebras/fp16 | 970 | 0.22s | **0.63s** | $0.0011 |
   | gpt-oss-120b @ groq | 330 | 0.23s | 1.44s | $0.00059 |
   | ling-3.0-flash @ novita/fast | 391 | 0.67s | 1.69s | $0.00015 |

   Ling was chosen when the assumption was that speed had to be bought with
   reasoning. Cerebras breaks that trade - the hardware is the speed, so a real
   reasoning model arrives faster than a fast chat model does. ~7x the cost,
   which is cents per month at this volume.

   **Reasoning effort is pinned to `low`** (`mentor.openrouter_reasoning_effort`).
   A reasoning model bills and TIMES its chain-of-thought as output tokens that
   never appear in the reply, so an unpinned effort setting makes the latency
   budget fiction. `reasoning_tokens` is now reported in the call meta and
   printed by `coach` - that number is what to check if a call feels slow.

   **A REAL BUG WAS FOUND DOING THIS.** The Ling pin was `novita/fp8`, guessed
   from OpenRouter's display name. The actual routing slug is **`novita/fast`**.
   With `allow_fallbacks: false`, the first real API call would have hard-failed
   on an unroutable provider. Verify every slug against
   `openrouter.ai/api/v1/models/<model>/endpoints`, which returns exact tags -
   never read them off the providers tab. All current slugs are verified there.

   **The Phase 5 pin was re-measured, not assumed.** Live replies are ~40
   tokens, where time-to-first-token dominates rather than throughput, so the
   Phase 3 pin was not taken on trust. Round trips at 40 tokens:
   `cerebras/fp16` 0.26s, `baseten/fp4` 0.33s, `groq` 0.35s. Cerebras wins at
   this length too - its throughput is high enough that its slightly worse
   first-token time never catches up - so live uses `mentor.*` unchanged and
   there is no separate live pin. **If the model ever changes, redo this
   arithmetic**; the answer is model-specific, not a general rule.

1. **`OPENROUTER_API_KEY` is not set.** Phase 3 has never made a real call. The
   request shape is mock-verified for both providers, but that is not the same
   as "worked". First real call costs ~$0.0002.
2. **INPUT GAIN IS TOO HOT.** The mic is a **Maono PD300X** (not the Fifine
   A6V that earlier notes assumed) and it works - `devices` shows it, and live
   mode captures from it. But a measured 225 s session held **541 runs of
   full-scale samples**, so roughly a fifth of all four-second windows contain
   real clipping. Peaks want to sit near **-9 dBFS**; they are hitting 0.
   Nothing downstream can be trusted until that is fixed, and it is a Windows
   input-volume setting, not a code problem.
3. **No real labeled fixtures.** WPM, word stress and mispronunciation have no
   ground truth - synthetic audio cannot provide it. See
   `tests/fixtures/manifest.json` -> `_needs_real_recordings`.

## Hardware, and what it forces

**Intel i5-5200U (2015 dual-core), 8 GB RAM, no GPU.** This is not a footnote;
it decided several designs.

| measured | consequence |
| --- | --- |
| `base.en` transcribes a 2 s clause in **~3.9 s** | live ASR can never keep up. It is off the critical path and allowed to drop clauses. |
| the same clip took **35 s** when the audio was near-silent | Whisper burns enormous time on temperature-fallback retries at low SNR. **Bad gain makes transcription slower, not just wronger.** |
| 18-minute script | ~35-40 min of local ASR, so work SECTION BY SECTION |

**The split that follows from this:** Flux image generation goes to Colab GPU;
script, recording, analysis, mastering and timing repair stay local. **Video
encoding should also go to Colab** - on Colab's free CPU runtime, which costs
zero GPU compute units. Encoding 18 minutes of 1080p on this laptop would take
longer than generating the images did.

Compute-unit budget: Colab Pro's 100 units is roughly 55 hours of T4. Flux
Schnell at ~150 frames is about **1 unit per video**. Whisper on a GPU would
cost ~0.03 units - so units are not the reason to keep ASR local; uploading
~100 MB of WAV per script is.

## OPEN RISK: the rubric is calibrated for US speakers, the user is not

The user is a Pakistani English speaker. The spec's stated goal is "good for a
US audience", and several defaults encode US-English norms that do not
transfer. This is not hypothetical - each item below is a concrete way the
current build can penalise a perfectly clear read.

| what | why it misfires |
| --- | --- |
| `pacing.target_wpm_min/max` = 140-170 | a US-English VO norm. Different syllable timing can be perfectly intelligible at 130 and get flagged as dragging |
| both niche references are American voices | zenn (Zenn) and movie_explainer (New Rockstars). Every benchmark delta is measured against American delivery |
| `pronunciation.min_word_confidence` = 0.55 | **the sharpest edge.** Whisper's word confidence is lower on non-American accents, so the ASR being less certain about the user's ACCENT would fire false mispronunciation flags - the rubric penalising how they sound, which the spec explicitly forbids |

Partial protection already exists and should not be removed:

- `rubric.NEVER_MUST_FIX` = {pitch_range, pitch_mean, accent} is a hard
  guarantee, not a convention - these can never escalate to must_fix.
- The mentor system prompt rule 4 forbids telling the speaker to change their
  voice; pitch, accent and timbre are identity, not delivery.
- `movie_explainer` already derives its pacing band from its reference rather
  than the global band (`wpm_from_reference: true`).

**BUILT: the calibration baseline** - `scoring/baseline.py`, 24 tests. It
scores relative to the user's own natural range instead of absolute US norms.

```
python -m vo_coach calibrate <take_id> <take_id> <take_id>   # 3 natural takes
python -m vo_coach baseline                                  # show it
# then set baseline.enabled: true in config/thresholds.yaml
```

Design points that should not be undone:

- **It refuses bad takes.** Clipping, SNR below the must-fix floor, under 20 s,
  under 40 words, or mixed analyzer versions -> `CalibrationError`. A baseline
  built from a bad recording makes the fault the norm, and every take after it
  is judged against that.
- **Median and MAD, not mean and stdev.** With three takes a single odd read
  drags a mean anywhere. A baseline describes the usual case.
- **Word confidence is medianed within a take before across takes**, so a take
  with three times the words does not dominate just for being longer.
- **`min_word_confidence` can only ever be LOOSENED, never tightened.** The
  config value stays the ceiling. A speaker who happens to transcribe cleanly
  must not end up with a stricter mispronunciation threshold than the rubric's.
- **The blend is the point.** `baseline.weight` 0.0 is the pure US rubric, 1.0
  normalises away real faults ("you always rush, so rushing is fine"). Never
  set it to 1.0. Default 0.3.
- `wpm_band()` and `min_word_confidence()` return the unchanged rubric value
  when no baseline exists, so everything works before calibration and improves
  after it.

**Still true until the user actually records the calibration takes: do not
trust pacing or pronunciation feedback.** The code is there; the takes are not.

One loose end: `baseline.min_word_confidence()` is ready but nothing calls it
yet, because no pronunciation findings are implemented in the rubric. Wire it
in at the same time as that check - not before, and definitely not after.

## RESOLVED: the mentor is now checked against its own rules

The system prompt states hard rules - at most 3 fixes, never invent a
measurement, never coach voice identity, treat excluded metrics as unknown.
The prompt instructed; nothing enforced. `mentor/validate.py` now does, on
every call, with no LLM and no cost.

| rule | check | severity |
| --- | --- | --- |
| 1: at most `max_fixes` | count bolded fix headings (bullet/numbered prefixes tolerated) | hard |
| 5: never invent a measurement | every unit-bearing number in the reply must appear in the payload | hard |
| 4: never coach identity | instruction patterns for changing accent, pitch or timbre | hard |
| commented on pacing when `timings_reliable=False` | the flag is already in the payload | hard |
| "do not go looking for something to say" | fixes raised when the rubric flagged nothing | hard |
| treated an excluded benchmark metric as measured | `BenchmarkResult.excluded` | soft |
| promoted a stylistic finding into a fault | must-language in the same paragraph as a stylistic metric | soft |
| rule 3: no listener rationale / praise preamble | keyword scan | soft |

**hard vs soft is deliberate.** Hard = a stated rule with a yes/no answer;
these count toward the failure rate. Soft = a heuristic on tone, which can fire
on a reply that is actually fine; reported, never counted. A validator that
fires on good replies would poison the exact measurement it exists to produce.

Two false-positive guards worth keeping:

- **Rounding is not invention.** The payload says 14.4 dB, the reply says
  14 dB - that is good coaching. Tolerance is half a unit or 2%.
- **Bare numbers are not checked.** Only numbers with a unit attached are
  claims about the audio; "step 2" is not.
- **Reassurance about accent is not a violation.** "Your accent is not the
  issue" must pass; only instructions to change it fail.

### This is the mechanical answer to "is the cheap model good enough"

```
python -m vo_coach mentor-audit
```

Prints per-model: calls, calls with a hard violation, the rate, and which rules
are failing. Read the RATE, not any single reply - one bad reply is noise, a
rule failing repeatedly is a capability problem. Below ~10% is normal model
noise; a pattern is the signal to switch to the DeepSeek block in
`config/thresholds.yaml` and compare on the same takes.

The log (`data/mentor_audit.jsonl`) counts CALLS, not violations - a reply with
three violations is one failed call. Cached replies are checked but not logged:
counting a replayed answer would move the rate with no new evidence behind it.
`tests/conftest.py` redirects the log for every test, so a fake provider can
never write into the real evidence base.

### What this still does not tell you

It checks obedience, not judgment. "Was this the most useful of the three
findings to raise" has no regex. That needs, in order:

1. **Golden-set regression tests** - fixed `AnalysisResult` inputs, assert
   properties of the coaching. Free, catches drift when the model or prompt
   changes. Buildable now.
2. **LLM-as-judge** - see below. Hold until real takes exist to judge.

### What an LLM judge is, and when it runs

A second, stronger model (Claude Sonnet, or DeepSeek V4) is shown the same
`AnalysisResult` the cheap model saw, plus the cheap model's coaching, and
grades it on questions a regex cannot answer:

- Did it pick the three findings that most affect the listener, or the three
  easiest to write about?
- Does each fix say what the listener experiences, or just restate a number?
- Did it respect the stylistic/must-fix split in substance, not just wording?
- Would a real VO coach say this?

**It is not a monitor.** It never runs in the coaching path - that would double
the cost and the latency of the thing whose whole selling point is 1.7 s and
$0.00015. It is a batch tool, run deliberately, at three moments:

1. **When deciding whether to switch models.** Run 10-20 stored takes through
   both models, judge both, compare. That is the decision, made on evidence.
2. **After changing the system prompt.** Catches a wording change that quietly
   made the coaching worse.
3. **Once, when real takes first exist**, to establish that the cheap model is
   acceptable at all.

Cost is ~$0.01 per judged take, so a full 20-take comparison is about 40 cents.
Build it after the golden set, and not before real recordings exist - judging
synthetic takes measures nothing.

Standing caveat: the judge is also a model. It is better evidence than an
impression and worse than the user's own ear. Where the two disagree on a real
take, the ear wins.

## BUILT: the golden set - `vo-coach eval`

`mentor-audit` measures rule-obedience on whatever takes happened to exist.
That is a real signal but not a COMPARISON: two models never see the same
inputs, so a gap between their rates could just be a gap between the takes.
The golden set fixes the inputs. Nine cases, same for every model.

| case | what it catches |
| --- | --- |
| `clean_take` | **the most important one.** Can the coach say NOTHING? A model that always produces three fixes is filling a template, and a user who learns that stops reading the output |
| `clipping_only` | one real fault - does it pad with two invented ones |
| `crowded_take` | 4 findings, cap of 3: must drop the STYLISTIC one, not a must-fix |
| `stylistic_only` | pitch range alone - does it promote a stylistic finding into a fault |
| `accent_bait` | **the one that matters for this user.** Low ASR confidence (what a non-US accent looks like to Whisper) plus a real noise problem. Does the model reach for "work on your accent"? Nothing in the payload names an accent, so any mention is the model inferring one |
| `timings_unreliable` | take IS slow, payload says pacing was never measured. Does it comment anyway |
| `sustained_slow` | the mirror: same slow read, timings reliable. Stops a model scoring well on the case above by never mentioning pacing at all |
| `excluded_benchmark` | does it compare against a metric marked `not_measured` |
| `hesitant_pauses` | the one finding whose advice is genuinely non-obvious |

Design points:

- **The real rubric runs over each fixture** - findings are produced by
  `scoring.rubric`, not hand-written, so a case tests the whole chain.
- **`expect_rubric` is a staleness check that runs BEFORE any API call.** If a
  threshold change emptied a payload, the case fails as stale instead of
  spending money judging a reply against a take that no longer represents
  anything. `test_no_case_is_stale` enforces it at test time too.
- **A failing case never reaches the provider**, and one broken case does not
  abort the run - otherwise a single transport error costs the other eight
  calls.
- Cases are Python, not JSON: each exists because of a specific failure mode,
  and that reason has to sit next to the case or it is gone in a month.

**A bug this already caught:** the accent case's `take_id` was `eval_accent`,
and `take_id` is sent in the payload - handing the model the exact word the
case exists to see whether it reaches for unprompted. Renamed to
`eval_lowconf`; a test now guards it.

Cost: ~$0.01 for a full run; the mentor cache makes repeats free. Workflow is
`eval` -> switch `mentor.model` -> `eval` -> `eval --compare`.

**Read `clean_take` and `accent_bait` above the headline rate.** A model that
fails the first coaches takes with nothing wrong; one that fails the second
coaches identity.

## BUILT: Phase 5 live mode - `vo-coach live`

Opens a HUD at `http://127.0.0.1:8765`. Capture starts when the HUD connects.

```
live/ring.py      fixed-capacity ring buffer, one rate (48 kHz)
live/clause.py    clause-boundary detection from energy alone
live/monitor.py   live metrics + session state (what the HUD shows)
live/coach.py     the one-line live mentor, and the three throttles
live/session.py   the thread wiring: callback -> worker -> state
live/server.py    FastAPI, WebSocket, static HUD
live/static/      the HUD page, entirely self-contained
```

**THE AUDIO IS CAPTURED BY THIS PROCESS, NOT THE BROWSER.** That is a departure
from "stream the mic over the WebSocket", made for one reason that outweighs
the rest: the browser would capture through WebAudio at whatever rate it likes
and resample in JavaScript, so live and offline metrics would be measured on
two different signals - and when they disagreed there would be no way to tell
which was right. Capturing here means live and `record()` are the same path.
It also removes the mic-permission prompt, binary WS framing, and JS
resampling. The HUD is a viewer; the only thing it sends is `stop`.

**Clause detection is the centre of the phase.** A note that lands mid-sentence
derails the read. Detection runs on 20 ms RMS frames against an *adaptive*
noise floor - a fixed dBFS gate works in one room at one gain and silently
fails in every other, and this coach actively tells the user to change mic
distance, which moves the floor. Two minimums, two different failure modes:
`min_speech` stops a cough reading as a clause, `min_silence` stops every
inter-word gap being one.

**ASR transcribes one clause, not a sliding window.** A sliding window has to
solve words split across window edges - transcribed twice, or lost. Clauses are
bounded by silence by construction, so the cut is exactly where a word is not.
This is the main reason boundary detection runs on energy rather than waiting
for a transcript.

**Three throttles, three failure modes** (`live.*` in config):
`coach_cooldown_sec` stops it talking over itself; `coach_budget` bounds spend
for a session; `repeat_after_sec` stops the same note twice. **The third is the
one that matters** - without it the other two just slow the repetition down.
Only `must_fix` findings are eligible live: a speaker cannot act on "consider
more pitch variation" mid-take, and it costs them the line. Stylistic notes
wait for the post-take `coach` pass.

**Pacing and pause coaching are suppressed live** (`timings_reliable=False`).
Over one clause they measure a single sentence, and the rubric's own design
says one fast sentence is human.

**Degrades instead of stopping.** No API key, budget spent, or a dead endpoint
falls back to the rubric's own sentence (`kind: "rubric"` on the HUD). A live
coach that goes silent because of a billing limit is worse than one that says
something plain.

**Sessions are saved as normal takes**, written incrementally, and registered
in the database on stop - so `analyze` / `master` / `rank` all work on them
afterwards, and a crash still leaves a playable wav.

**Verified on this machine (2026-08-23):** the HUD connects over the
WebSocket, live metrics stream from the Realtek built-in mic (peak -41.9 dBFS,
noise floor -72.3, SNR 11.6 in a quiet room), and clause detection fires. The
full callback -> clause -> rubric -> note path also runs on synthetic audio.
**Not verified:** the Fifine A6V (still not plugged in - `devices` shows only
Realtek), and any real API call.

### Two bugs found here, both of which looked like a dead microphone

**FastAPI could not resolve the WebSocket parameter type.** `server.py` uses
`from __future__ import annotations`, so annotations are strings at runtime and
FastAPI resolves them against the MODULE globals. `WebSocket` was imported
inside `create_app`, making it a local - invisible to that lookup. FastAPI read
the socket parameter as a query param, failed validation, and closed before
accepting: a bare **403 on the upgrade, no traceback anywhere**, and a HUD
frozen at 0:00 that was indistinguishable from a silent mic. `/api/state` kept
working throughout, which pointed the diagnosis at audio rather than routing.

Two things made it hard: asserting the route exists does NOT catch it (the
route was registered and correct), and the HUD's reconnect-on-close loop hid
the rejection. Both are now covered - `test_the_websocket_actually_accepts_a_
connection` opens a real connection, and a second test asserts `WebSocket` is a
module global so the condition is caught on a machine with no audio device.

**uvicorn 0.34 could not drive websockets>=14**, which returns the same 403 by
a different route. Requirements now floor uvicorn at 0.52.4 rather than pinning
websockets backwards.

### Bug found building this

`Ring.write` dropped time when handed a block larger than the whole buffer: it
truncated to the tail but advanced the sample clock by the truncated length.
`_write` is the clock every absolute timestamp is measured against, so every
boundary after such a block would have shifted earlier, and ASR would have been
handed the wrong span with nothing to indicate it.

### First run, in order

1. `vo-coach devices` - confirm the A6V is there and note its index.
2. `vo-coach live --no-mentor` - free, no API calls. Watch the HUD: does the
   clause list fill as you speak, and does it stop when you stop? Tune
   `live.min_silence_sec` / `live.min_speech_sec` until it does.
3. Only then set `OPENROUTER_API_KEY` and run `vo-coach live`.

Doing it in that order means clause detection is tuned before any money is
spent, and the two possible faults - "boundaries are wrong" and "the coaching
is wrong" - stay separable.

## How a session actually runs end to end

This is the loop the whole project exists to serve. Everything else is support.

```
1  write the script                       plain .txt
2  mark-script --niche zenn               -> X_marked.txt
                                             pause targets, which word carries
                                             each line, one plain note per line
3  live --script X_marked.txt             read it. Teleprompter holds you on a
                                             line until a take of it is accepted
4  Stop & save                            -> live_TIMESTAMP.wav          (raw)
                                             live_TIMESTAMP_comped.wav   (keepers)
5  copy the comped file to the notebook   voice_recordings/section_NN.wav
```

**The comped file is the deliverable.** It contains the last passing attempt at
each line, joined with the script's own pause targets and real room tone in the
gaps - so pause correction happens at assembly rather than as a separate pass.
`fix-pauses` is only needed when you recorded WITHOUT a script.

**The raw file is never overwritten.** A comp built from a misfiring rubric has
to be recoverable, and these thresholds are still being tuned.

**⚠️ ORDERING CONSTRAINT.** The stickman notebook's Step 18 "locks its length so
the pictures stay in sync". Anything that changes duration - comping,
`fix-pauses` - **must run before that step**, or the images desync. The notebook
also requires one file per section (`section_01.wav`, ...), which is why the
whole design is section-by-section rather than one long read.

## BUILT: timing repair - `fix-pauses`, `mark-script`

Stage 3 of a studio pipeline (capture -> select -> **repair** -> master -> QC ->
deliver) was the missing link. Both are self-contained and need no API key.

**`fix-pauses`** sets performed pauses to their target length, silently. It
reports nothing by design - the user asked for a fixer, not a critic.

- **It cannot stretch a pause that was never performed.** Below
  `gaps.MIN_REAL_GAP` (0.25 s) it leaves the gap alone. When two sentences are
  run together the first one's intonation never closed, so padding room tone in
  behind it produces the exact "this was edited" sound the tool exists to avoid.
  **Correction, not creation - that is the line.**
- Gaps are filled with **room tone sampled from the same recording**, tiled with
  alternating direction and equal-power crossfades. Digital silence drops the
  noise floor to true zero mid-pause, which is the clearest tell of amateur
  editing. A plain loop repeats any faint hum at exactly the sample length.
- Edits run **back to front**, so earlier gap positions stay valid.
- No ASR anywhere - energy only, which is what makes it usable on this CPU.

**`mark-script`** derives a niche's pause rhythm and marks a script with it.
From the zenn reference: sentence 0.81 s / clause 0.31 s / paragraph 1.22 s -
within a few hundredths of hand-marked values, which is the sanity check.

## BUILT: delivery direction - the coach that teaches

The coach used to only know about microphone technique. It now also knows
**how to say the line**: which word carries it, and one plain instruction.

```
Somewhere past the last bright star you can see, ⬚ space stops BEHAVING ...
  Lean on BEHAVING. Keep it easy, and let the last word fall away.
```

**Computed at mark time, never during the read.** The shape of a sentence does
not depend on how it was performed, so it can be worked out before any audio
exists and shown instantly. That is also the latency fix: no network call in the
moment that matters.

Different sentence shapes get different direction - a list steps down, a long
line takes two breaths at the comma, the last line slows and sits.

**What it is NOT:** it never touches pronunciation or accent. It marks which
word carries meaning, which is identical in every accent. Tests assert that.

Honest limit: emphasis picking is a **free offline heuristic** (content words,
length, landing position). It gets STRETCHES and CONSEQUENCE right and BEHAVING
over BRIGHT STAR wrong. **The natural first upgrade once an API key exists is
one LLM call per script at mark time** - still zero latency while reading.

## The coach speaks plain English, and that is enforced

Rewritten so a nine-year-old could act on every line.

| before | after |
| --- | --- |
| "Spectral tilt averages -9.8 dB/octave" | "You are very close and it is making you sound boomy, like talking into a box. Back off a little." |
| "Signal-to-noise is 17 dB." | "The room is a bit louder than your voice - get closer to the mic, about a hand's width away." |

Correction SIZE is given in words, not units: **"try about half of what it is
now"**, not "about 9 dB". A decibel is not a thing anyone can picture, and the
size of a fix is the most useful thing a coach adds.

`tests/test_live.py::TestCoachSpeaksPlainly` runs a jargon list and a digit
check over every offline line, and asserts the prompt still contains
"not trying to sound American" and "never use a technical term". **These are the
commitments easiest to erode by accident, so they are tested, not trusted.**

## Line approval: what stops a take, and what does not

With a script loaded the teleprompter holds on a line until a take is accepted.

**Only `live.blocking_metrics` (default `["clipping"]`) can fail a line.**

This started as "any must_fix blocks" and that was wrong in a way only real
audio revealed: a measured session ran **51 sibilants/min against a 30/min
threshold**, so sibilance fired on nearly every line and the reader could not
leave line one - for something **the de-esser removes two steps later**. A coach
that stops the take for a fault the pipeline repairs automatically is worse than
one that misses it.

Keep the list short and keep it to things mastering CANNOT fix. Clipping
qualifies; sibilance, proximity and pacing do not. They are still coached and
still shown - they just do not block.

## Not built

- **LLM-as-judge** - grades judgment, which the validators cannot. Batch tool,
  never in the coaching path. Hold until real takes exist to judge; see the
  section above for what it is and when it runs.
- **Pronunciation findings** - the rubric has no pronunciation check at all
  yet, which is why `baseline.min_word_confidence()` is written but unused.
- **LLM-marked emphasis** - one call per script at `mark-script` time would
  replace the offline heuristic and cost nothing in latency. **Highest-value
  thing to build once the API key exists.**
- **QC gate** - BUILT, but on the pipeline side, not here:
  `channels/run_qc.py` says PASS/WARN/FAIL per section over
  `voice_recordings/*.wav` before any Colab runtime is paid for. It ports this
  project's own measurements (clipping as runs of full-scale samples, the
  noise-floor percentile with digital silence excluded) rather than importing
  them, because `channels/pipeline/` may not depend on soundfile or scipy. It
  would have caught today's clipping before a 225 s read was wasted. A vo_coach
  `qc` subcommand judging a single take against a delivery spec is still
  unbuilt, and would be the natural place for the metrics the rubric knows
  about that arithmetic alone cannot see.
- **Clause-level comping across SEPARATE takes** - `comp.py` assembles within
  one session only. Comping take 1 line 3 with take 2 line 7 is the next step.
- **Voice-model pickups** - the genuine "hybrid" technique: a model trained on
  the user's own voice generates one-word fixes instead of re-recording. Needs
  consistent captures first, so it is blocked on the gain problem above.
- **Open decision #2**: live coaching via spoken TTS or text-only. Currently
  TEXT-ONLY - notes appear on the HUD. Spoken coaching would need a TTS voice
  that does not bleed into the mic, which on an open condenser like the A6V
  means headphones are mandatory, not optional. Decide after using the text HUD
  on a real take.
- **Live pacing coaching** - suppressed today because one clause is one
  sentence. It becomes possible once the personal baseline exists: a rolling
  session rate compared against the user's own band, rather than against a US
  norm over three seconds.

## Bugs found this build, and where the reasoning now lives

Every one of these produced a *plausible wrong answer* rather than an error,
which is why each fix is documented at its site rather than only here.

| bug | where it is explained |
| --- | --- |
| ASR fed 48 kHz to a 16 kHz API -> every timestamp 3x too long, fluent hallucinated transcript, 40 wpm | `analysis/asr.py` |
| SNR read ~175 dB on any file containing digital silence | `analysis/spectral.py` |
| Sibilance fired on 19 frames of a fixture containing none; adding hiss *reduced* the count | `analysis/spectral.py` |
| pedalboard `Limiter` applies makeup gain - boosts a -26 dBFS tone to -21.3 | `mastering/chain.py` |
| pedalboard `Compressor` too slow for transients; limiter is hand-written with lookahead | `mastering/chain.py` |
| Compressor never engaged - threshold above the post-denoise signal level | `mastering/chain.py` |
| Stale analyses reused forever after a metric fix | `ANALYZER_VERSION` in `analysis/pipeline.py` |
| Bengali word timestamps non-deterministic (33-94 words, 34-89 wpm, same clip) | `README.md`, config `niches:` |
| cp1252 console killed a finished Bengali ASR result at the print | `cli.py` `_force_utf8_stdout` |
| **FastAPI could not resolve the WebSocket type**, so it closed before accepting: bare 403, no traceback, HUD frozen at 0:00 looking exactly like a dead mic | `live/server.py` header |
| **uvicorn 0.34 cannot drive websockets>=14** - same 403, different cause | `requirements.txt` |
| **`localhost` resolves to IPv6 `::1`; uvicorn binds IPv4 only** - page unreachable in a normal browser, fine in VS Code's | `live/server.py` `serve()` |
| **`Ring.write` lost time** on a block larger than the buffer: truncated the audio but advanced the sample clock by the truncated length, shifting every later timestamp | `live/ring.py` |
| **Coaching waited on ASR** - notes arrived ~4 s after the moment they were about | `live/session.py` `_handle_boundary` |
| **Live clipping fired on ONE sample** over a 4 s window, and that flag failed the line | `live/monitor.py` |
| **Every must_fix blocked a line**, so sibilance at 51/min made the script unfinishable | `config` `live.blocking_metrics` |
| **The marked-script legend was parsed as a pause target**, shifting every pause onto the next one's target - silently wrong phrasing, first gap set to 0.00 s | `timing/marking.py` `targets()` |
| **Gap detection used a 20th-percentile floor**, which lands inside speech when a take is >20% silence - the whole file read as one pause | `timing/gaps.py` `find()` |
| **Emphasis picked duplicates** ("lean on PRESSING" twice) and long empty words ("SOMETHING") | `timing/marking.py` |
| **Tests opened the real microphone**; PortAudio BLOCKS rather than failing when the device is busy, so the suite hung indefinitely whenever a live session was running | `tests/conftest.py` |

## Judgment calls worth not re-litigating

- **No composite score.** Per-metric deltas only - a single number lets good
  pacing cancel out a proximity problem.
- **Stylistic can never escalate to must-fix.** Pitch range and accent are
  identity, not delivery. Enforced in `rubric.NEVER_MUST_FIX`, not by convention.
- **Excluded benchmark metrics are reported with reasons**, so a missing row
  never reads as "checked, and fine".
- **OpenRouter host pinning is mandatory**, not tuning: 30 hosts serve DeepSeek
  at up to 6.7x price spread. `allow_fallbacks: false` so a bad slug fails
  loudly instead of silently routing expensive.
- **Delivery quality is not scored** in ranking. Nothing measures whether the
  read lands, and inventing a proxy would rank something never measured.
- **The browser never touches the microphone.** Capture is in the Python
  process, same path as `record()`. If the browser captured via WebAudio at its
  own rate, live and offline metrics would measure two different signals and
  there would be no way to tell which was right. The HUD is a viewer.
- **Live ASR is best-effort and may drop clauses.** It is slower than speech on
  this CPU, so a deep queue would not lag - it would diverge. Dropped clauses
  are marked; the saved take gets a proper transcript from `analyze`, where
  there is no clock.
- **`fix-pauses` says nothing.** Deliberate, at the user's request. The one
  thing it will not do silently is stretch a gap that was never performed.
