# VO Coach

AI voiceover coach & mastering agent. Listens to a read, coaches toward
"good for a US audience" (not studio-perfect), then masters the winning take.

Built from `VO_COACH_BUILD_SPEC.md`.

---

## Setup

**Python 3.12 is required** — not 3.13 or 3.14. `parselmouth`, `pedalboard` and
the CTranslate2 backend behind `faster-whisper` ship compiled wheels that lag new
Python releases; on 3.14 pip falls back to a source build and fails on a missing
C++ toolchain. That error looks like a code problem but isn't.

```bash
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m pip install -e .
```

## Try it

```bash
.venv/Scripts/python.exe -m vo_coach devices          # find your mic
.venv/Scripts/python.exe -m vo_coach record --seconds 30 --device 11
.venv/Scripts/python.exe -m vo_coach takes            # take history
.venv/Scripts/python.exe -m vo_coach config           # resolved thresholds
```

```bash
.venv/Scripts/python.exe scripts/make_fixtures.py     # regenerate test fixtures
.venv/Scripts/python.exe -m pytest -q                 # run tests
```

---

## Layout

```text
vo_coach/
├── config/thresholds.yaml     ALL tunable numbers. Never hard-code these.
├── src/vo_coach/
│   ├── schemas.py             THE DATA CONTRACT — read this first.
│   ├── config.py              only module that reads the YAML
│   ├── record.py              Phase 0: capture / ingest takes
│   ├── cli.py                 CLI-first, UI-last
│   ├── analysis/              Phase 1: ASR, pacing, prosody, spectral, VAD
│   ├── scoring/               Phase 2: rubric + competitor benchmark
│   ├── mentor/                Phase 3: Claude coaching conversation
│   ├── mastering/             Phase 4: pedalboard chain + LUFS
│   ├── storage/db.py          SQLite take history
│   └── viz/                   debug plots (before the real HUD)
├── scripts/make_fixtures.py   synthetic labeled test signals
├── tests/fixtures/            generated WAVs + manifest.json
└── data/                      takes, references, cache (gitignored)
```

### The contract

`AnalysisResult` in `schemas.py` is the single object every phase after Phase 1
consumes — scoring reads it, the benchmark diffs two, the mentor prompt is built
from it, the plots draw it. Nothing downstream touches raw audio. Change it
deliberately.

### The two thresholds

Every metric in `thresholds.yaml` has a **silence** threshold (say nothing below
this) and a **must-fix** threshold (always flag above this). The gap between them
is the coach's whole personality — widen for calmer, narrow for stricter. A test
enforces `silence < must_fix` on every metric, since inverting them silently
makes the coach flag everything or nothing.

---

## Build order

Phases 1–4 are offline batch processing and much easier to debug. Live mode
reuses all of that logic, just windowed and streamed — **don't start Phase 5
until offline scoring is solid and the rubric feels right.**

- [x] **Phase 0** — scaffold, contract, config, SQLite, record/ingest, fixtures
- [x] **Phase 1** — offline analysis → `AnalysisResult` *(ASR path unverified, see below)*
- [x] **Phase 2** — scoring rubric + competitor benchmark
- [ ] **Phase 3** — Claude mentor conversation
- [ ] **Phase 4** — take ranking + pedalboard mastering chain
- [ ] **Phase 5** — live mode (WebSocket, streaming ASR, clause-boundary coaching)

---

## Phase 1 — what's verified and what isn't

**Verified against fixtures** (32 tests passing):

| metric | status |
| --- | --- |
| SNR / noise floor | 11.5 dB measured vs 12.0 dB constructed |
| sibilance | finds exactly the 4 constructed bursts, 0 false positives on all other fixtures |
| spectral tilt | proximity fixture tilts measurably below baseline |
| clipping | detected on the clipped fixture only |
| pacing / pauses | unit-tested on synthetic word lists incl. ASR-insertion desync |
| pitch | 150 Hz tone reads 150 Hz (octave-error guard) |

**NOT yet verified: the ASR path.** `analysis/asr.py` is written and imports
cleanly, but has never run against real speech — the fixtures are synthetic
noise, which would return garbage, and no real recording exists yet. WPM,
transcript and punctuation-aware pause labelling are therefore untested end to
end. First real take on the A6V will exercise them:

```bash
python -m vo_coach record --seconds 30 --script myscript.txt
python -m vo_coach analyze <take_id> --script myscript.txt --plot
```

Three bugs the fixtures caught during Phase 1, worth knowing about because each
would have been invisible on real audio:

1. **SNR of 175 dB on any gated file.** A percentile over true-zero samples
   floors to -200 dB, and the rubric would have read that as a flawless take.
   Now returns `None` when there is no room tone to measure.
2. **Sibilance fired on 19 frames of a fixture containing no sibilance.**
   Absolute band energy vs its own median just tracks broadband noise. Now
   measured as the 5-9 kHz : 300-4000 Hz ratio, gated on speech activity, and
   required to be prominent in absolute terms too — so hiss reads as a
   mastering problem, not as a delivery note.
3. **A test that passed at 19 vs 20 peaks.** `assert clean < harsh` gave false
   confidence; asserting the exact expected count is what exposed it.

## Phase 2 — niches, rubric, benchmark

Two separate channels, each with its own reference voice. A take is only ever
compared against **its own niche** — cross-comparing an English explainer to a
Bengali one would be meaningless.

```bash
python -m vo_coach refs build                    # extract + cache both references
python -m vo_coach refs list
python -m vo_coach score <take_id> --niche zenn
python -m vo_coach benchmark <take_id> --niche zenn
```

### Reference voices (both 60s English, both lossy YouTube rips)

| | zenn | movie_explainer |
| --- | --- | --- |
| source | *What Did Ancient Humans Do at Night?* @194s | New Rockstars, *Spider-Man NWH Breakdown* @378.5s |
| pacing target | global 140–170 band | derived from reference |
| timing metrics | ✅ | ✅ |

### Why a Bengali reference was dropped

The second niche originally used a Bengali movie-explainer. Whisper's word
timestamps turned out to be **non-deterministic** for Bengali — the same 60s
clip, same model, repeated runs:

| run | Bengali words | WPM | | English words | WPM |
| --- | --- | --- | --- | --- | --- |
| 0 | 33 | 34.3 | | 188 | 188.1 |
| 1 | 47 | 47.3 | | 188 | 188.1 |
| 2 | 61 | 89.4 | | — | — |
| (earlier) | 94 | ~96 | | | |

**A 2.6× spread in speaking rate from identical input**, where English returns
the same answer bit for bit. Since WPM, pause rate and pause duration are all
derived from those timestamps, all three were *meaningless* — and the pipeline
had already produced a confident "derived pacing band: 71–90 wpm" from them.

The machinery built for that case is still in place and tested
(`asr_word_timestamps_reliable: false` suppresses timing metrics per-niche in
both rubric and benchmark, printing the reason). It is currently unused because
both niches are English. **If a non-English reference is added later, verify it
first:** run the same clip 3× and confirm the word counts agree.

### Three deliberate exclusions

Recorded in `excluded_from_benchmark` so a missing row never reads as "checked,
and it was fine":

- **noise floor / SNR** — the zenn reference measures 14.4 dB SNR, *below* this
  config's own 20 dB must-fix threshold. Benchmarking your hiss against a lossy
  rip would coach you toward a worse recording.
- **pitch mean** — vocal identity, not delivery. Matching another speaker's
  pitch means changing your voice.
- **cross-language WPM** — flagged with a caveat rather than silently compared.

### Design rules the tests enforce

- **No composite score.** Per-metric deltas only — a single number lets good
  pacing cancel out a proximity problem.
- **Stylistic can never escalate.** `NEVER_MUST_FIX` is a hard guarantee, not a
  convention: pitch range stays stylistic at any magnitude.
- **Direction matters.** A pitch range *wider* than the reference is not a
  fault; `|delta|` alone would flag a more expressive read as wrong.
- **Rates, not counts.** Everything is per-minute so a 3-minute take doesn't
  score worse than a 1-minute one for being longer.
- **Silence is a valid answer.** A clean take produces zero findings; a rubric
  that always finds something trains you to ignore it.

## The sample-rate bug (worth reading)

`faster-whisper`'s `transcribe()` takes a bare float array with **no
sample-rate argument** — it assumes 16 kHz. This pipeline runs at 48 kHz, so
audio was being heard at ⅓ speed.

Nothing about the failure looked like a failure. No exception, no warning, a
fluent transcript, well-formed word timestamps. Only the arithmetic gave it
away — **words ending at 179s in a 60s file** (60 × 48000/16000 = 180 ✓).

| | before | after |
| --- | --- | --- |
| last word ends | 179.5s | 59.98s |
| WPM | 40 | 188 |
| transcript | *"falling short from over a spooker"* | *"prayer books from the 1400s, court records, diaries…"* |

Two things worth noting: a **bigger model made it worse** (`small.en` gave 40
wpm vs `base.en`'s 43) — model size can't fix a rate mismatch. And this would
have silently scaled *every* pacing number in the project, including Phase 5's
live coaching. There is now a hard runtime check in `asr.py` that raises rather
than letting a >5% timestamp overrun through, plus regression tests in
`test_asr.py`.

## Known gaps

**Fixtures can't cover everything.** Synthetic signals validate spectral tilt,
sibilance, SNR, pauses and clipping — real DSP measurements with known answers.
They *cannot* cover WPM, word stress or mispronunciation, which need real speech
and a real ASR pass. `tests/fixtures/manifest.json` has a
`_needs_real_recordings` block listing the takes to record and hand-label before
trusting pacing or pronunciation scoring. Don't let that gap go invisible.

**`whisperx` is deliberately unpinned.** It pulls its own torch build and often
conflicts with the `faster-whisper`/CTranslate2 pairing. Phase 1 works without it
— `faster-whisper` already returns word timestamps. Add it when phoneme-level
scoring actually starts, in a separate venv if the resolver fights.

## Open decisions

From the spec — these need your answer before the phase that consumes them:

1. **Target loudness** for final export — `mastering.target_lufs` currently
   `-16` (podcast/VO norm); `-14` is the streaming norm. *Needed by Phase 4.*
2. **Live coaching delivery** — spoken TTS or text-only at clause breaks.
   *Needed by Phase 5.*
3. **Competitor reference files** — which two, and stored long-term or
   per-project. *Needed by Phase 2.*
4. **WPM band** — currently `140–170` from general VO norms. Should drift toward
   your own best takes; see `baseline.*` in the config. *Needed by Phase 2.*
