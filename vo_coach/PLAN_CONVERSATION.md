# The conversational coach — not built, scoped here

Written 2026-08-24, out of a design conversation, before any code. Companion to
`STATUS.md` (what exists today) and `f:\wierd_bio\PLAN.md` (the whole pipeline).
**Do not start this until one real video has gone through the pipeline
end-to-end** — see that file's "Next steps". Tuning a coach before you've seen
a finished video risks building it for a workflow the real run reshapes anyway.

---

## What exists today, stated plainly so this isn't confused with it

`vo_coach live` already reacts to your reading in real time, but it is:

- **Text only.** Every note is a string printed to the browser HUD
  (`live/coach.py: speak()` / `note_local()`). There is no speech synthesis
  anywhere in the codebase — no `pyttsx3`, no browser `speechSynthesis`, no
  `<audio>` element playing a generated voice. Confirmed by reading the source,
  not assumed.
- **One-directional.** The coach fires a note when a metric trips. You cannot
  ask it something and get an answer back, live. `vo_coach coach <take_id>` is
  the closest thing to a conversation and it is a single call-and-print after
  the fact, not a dialogue.
- **Barely blocking.** Only `clipping` can fail a line
  (`live.blocking_metrics`, default `["clipping"]`). Everything else — pacing,
  sibilance, proximity — is shown and coached but the take is **accepted
  anyway**. This was a deliberate fix, not an oversight: STATUS.md records a
  real session where sibilance fired on nearly every line against a threshold
  the mastering chain's de-esser removes automatically, and a coach that stops
  the take for something the pipeline repairs on its own is worse than one that
  lets it through.
- **No pronunciation check exists at all.** `baseline.py` has a comment
  referencing a "mispronunciation floor," but nothing computes pronunciation
  today. STATUS.md lists it under *Not built* for a specific reason (see below).

What this document adds, per the conversation that produced it, is three
things: real spoken output and spoken input (not the printed HUD),
syllable-level pronunciation guidance, and a hard gate — the sentence does not
advance until it is right.

---

## The three asks, and what each actually requires

### 1. Real human conversation — spoken, both directions

Today: text note appears on screen when a metric trips.
Wanted: the coach speaks the note aloud, and you can respond — ask "what do
you mean by that", or push back — and it answers, out loud, mid-session.

This needs, roughly:

- **TTS on the coach's output.** The note text already exists
  (`coach.speak()` returns `(text, meta)`); the missing piece is a voice
  synthesis call between that and the speaker. Two shapes:
  - a **local** TTS voice (fast, free, works offline, sounds like a TTS voice)
  - an **API** TTS voice (better quality, costs money per call, adds a network
    round trip in exactly the moment `speak()`'s own docstring already worries
    about latency)

  Given the hardware constraint already on record — i5-5200U, no GPU, and
  `live` ASR is already "off the critical path and allowed to drop clauses"
  because local Whisper can't keep up — **do not add local neural TTS**. Either
  a lightweight local TTS (`pyttsx3`-class, instant, machine-sounding) with the
  API path as an upgrade later, or accept the API latency and design around it
  the same way ASR already is: never on the strict path of deciding whether a
  line passed.

- **Speech input for the "talk back" half.** The mic is already open and
  streaming (`live/session.py`'s `sd.InputStream`); what's missing is a mode
  where the system recognizes you're *talking to the coach* rather than
  *reading the script* — a push-to-talk key, or a wake phrase, or a pause in
  reading long enough to imply a question. Whichever is chosen, it needs a
  real decision: **the script clause detector and the "talking to the coach"
  detector cannot both be listening to the same audio stream without stepping
  on each other.**

- **An actual dialogue loop**, not a single request/response. This means
  conversation history gets threaded into the mentor prompt per session
  (`mentor/prompt.py` already builds one-shot payloads; this needs a running
  transcript, not a single finding), and a real decision about how much of that
  history is sent on every turn — cost and latency both scale with it.

**This is the most expensive of the three pieces**, both in engineering effort
and in ongoing API cost per session if the API path is chosen for either
direction.

### 2. Syllable-level pronunciation guidance

STATUS.md is explicit about why this doesn't exist yet, and that reasoning
does not go away just because it's wanted now:

> **OPEN RISK: the rubric is calibrated for US speakers, the user is not.**
> Whisper's word confidence is lower on non-American accents, so the ASR being
> less certain about the user's ACCENT would fire false mispronunciation flags
> — the rubric penalising how they sound, which the spec explicitly forbids.

A syllable-level check is the same risk at finer resolution — a bigger risk if
anything, since syllable-level ASR confidence is noisier than word-level. So
this cannot be built the naive way (flag any syllable Whisper is unsure about).
It needs, at minimum:

- **The calibration baseline recorded first** (`vo_coach calibrate`, still
  blocked on the gain per `PLAN.md`). Pronunciation flags must be judged
  against the user's own natural baseline, the same principle already applied
  to pacing and never to be applied only loosely here.
- **A real phoneme/syllable reference**, not just word-level Whisper
  confidence — likely a forced-alignment tool (e.g. a phoneme-level aligner)
  rather than stretching Whisper past what it's good at.
- **`rubric.NEVER_MUST_FIX` = {pitch_range, pitch_mean, accent}** stays a hard
  guarantee. A syllable check must be checking pronunciation of a WORD (did you
  say the right sounds in the right order), never accent (how those sounds are
  colored). That line is easy to blur at the syllable level and must not be.

Practically: this is a real ASR/alignment feature, not a threshold tweak, and
it is explicitly gated on the calibration takes existing.

### 3. A hard gate — the sentence doesn't advance until it's right

Today only clipping blocks. Wanted: the reader cannot move to the next line
until pronunciation (and presumably other flagged issues) are resolved.

This is a **design reversal**, and needs to be a deliberate one, because the
reason blocking was narrowed to clipping alone is on record and real: broad
must-fix blocking measurably stopped a reader on line one for something the
mastering chain fixes automatically. Widening the block list again means:

- **Decide, metric by metric, which faults mastering genuinely cannot fix.**
  Clipping already qualifies (permanent, no recovery). Pronunciation likely
  qualifies too (mastering doesn't fix what word you said). Sibilance,
  proximity, pacing do NOT — re-litigating those back onto the blocking list
  would be repeating a mistake already made and fixed once.
- **A per-metric decision, not a single toggle.** `live.blocking_metrics` is
  already a set; this is "add `pronunciation` to it once it exists and is
  trustworthy" — not "make everything blocking."
- **A path forward when the coach is wrong.** Any blocking gate needs an
  override — the reader must be able to say "no, that was right, move on" —
  or a single false positive traps the session the same way sibilance once did.

---

## Suggested build order, when this is picked up

1. **Calibration takes recorded, baseline enabled.** Blocks #2 entirely and
   makes #3's pronunciation-blocking meaningfully safe rather than guesswork.
2. **TTS on the existing note pipeline first, API input still text.** Smallest
   slice that delivers "the coach talks to me" — no new detection, no new
   blocking, just voice on what `speak()` already returns. Proves the latency
   and cost are tolerable before anything harder is built on top.
3. **Pronunciation/syllable checking**, gated on the calibration baseline,
   added to the rubric as a new metric — NOT yet blocking.
4. **Add `pronunciation` to `live.blocking_metrics`** once its false-positive
   rate has actually been measured against real takes, not assumed.
5. **Two-way spoken conversation** (mic input recognized as talking-to-coach,
   dialogue history threaded through the mentor prompt) last, because it is
   the most architecturally invasive piece — it changes what the live session
   loop is doing while you read, not just what it prints.

Do not build 5 before 3 and 4 are solid: a coach that can chat but still gives
bad pronunciation calls is a worse experience than the current one, because
now it argues back.

---

## Cost note

Both TTS-out and speech-recognized-in, if done through the API rather than
locally, add a per-session cost on top of the existing mentor-call cost
(`mentor.*`, already OpenRouter-routed and pinned per STATUS.md). This has not
been estimated yet — do that before committing to the API path for either
direction, the same way every other model choice in this project was measured,
not assumed.
