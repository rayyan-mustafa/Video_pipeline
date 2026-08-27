"""Live signal metrics and session state.

WHAT IS MEASURED LIVE AND WHAT IS NOT. Only metrics that are honest over a few
seconds of audio appear on the HUD:

  measured live   peak / clipping, RMS level, SNR, spectral tilt, sibilance
                  rate, and - once a clause has been transcribed - speaking
                  rate.
  NOT measured    anything needing the whole take. Pause-alignment needs the
                  script; pitch range over three seconds describes one phrase,
                  not a delivery. Showing those would put a number on the HUD
                  that means something different from the same number after
                  `analyze`, and the user would have no way to know which to
                  believe.

REUSES THE OFFLINE FUNCTIONS. `spectral_tilt`, `noise_floor_and_snr` and
`sibilance_peaks` are called here exactly as `analysis/pipeline.py` calls them,
over an `Audio` built from the ring. Re-implementing a faster live version is
how live and offline start disagreeing.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

import numpy as np

from ..analysis.audio import Audio
from ..config import Config


@dataclass
class LiveMetrics:
    """One HUD frame. Every field may be None - the HUD renders "--" for
    those, which is the honest display for "not measurable yet"."""

    t: float = 0.0
    rms_db: float | None = None
    peak_db: float | None = None
    clipping: bool = False
    snr_db: float | None = None
    tilt_db_per_oct: float | None = None
    sibilance_per_min: float | None = None
    wpm: float | None = None
    speaking: bool = False
    noise_floor_db: float | None = None
    # How many runs of full-scale samples are in the window. `clipping` is the
    # judgement; this is the evidence behind it.
    clip_runs: int = 0
    # Peak touched full scale at least once. Worth showing - it is the warning
    # before real clipping - but NOT enough to fail a take on.
    peak_hot: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def _db(x: float) -> float:
    return 20.0 * np.log10(max(float(x), 1e-9))


def measure(audio: Audio, cfg: Config, window_sec: float) -> LiveMetrics:
    """Signal metrics over the recent window. Cheap enough for ~4 Hz."""
    from ..analysis.spectral import (
        noise_floor_and_snr,
        sibilance_peaks,
        spectral_tilt,
    )

    m = LiveMetrics()
    x = audio.samples
    if len(x) < audio.sr // 10:            # under 100 ms - nothing to say yet
        return m

    peak = float(np.max(np.abs(x)))
    m.peak_db = round(_db(peak), 1)
    m.rms_db = round(_db(float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))), 1)

    # CLIPPING IS A RUN OF SAMPLES AT FULL SCALE, NOT ONE SAMPLE.
    #
    # This was `peak >= 0.99` over the whole window, which meant a single
    # half-millisecond transient lit the red light - and kept it lit - for the
    # next four seconds. Measured on a real session: 0.03% of samples touched
    # full scale, but 22% of four-second windows contained at least one, so the
    # light was on a fifth of the time for something that was mostly one tick
    # on a plosive.
    #
    # It matters more than cosmetics, because line approval reads this flag.
    # Same run-length definition the offline detector uses, plus a minimum
    # count so an isolated tick is not treated as a ruined take.
    from ..analysis.spectral import clipping_times

    try:
        m.clip_runs = len(clipping_times(audio))
    except Exception:
        m.clip_runs = 0
    m.clipping = m.clip_runs >= int(cfg.get("live.clip_runs_to_flag", 3))
    m.peak_hot = peak >= 0.99

    try:
        nf, snr = noise_floor_and_snr(audio)
        m.noise_floor_db = round(nf, 1) if nf is not None else None
        m.snr_db = round(snr, 1) if snr is not None else None
    except Exception:
        pass

    try:
        pts = spectral_tilt(audio)
        if pts:
            m.tilt_db_per_oct = round(
                float(np.mean([p.value for p in pts])), 2)
    except Exception:
        pass

    try:
        band = (float(cfg.get("sibilance.band_hz_low")),
                float(cfg.get("sibilance.band_hz_high")))
        peaks = sibilance_peaks(audio, band_hz=band)
        m.sibilance_per_min = round(len(peaks) / max(window_sec / 60.0, 1e-6), 1)
    except Exception:
        pass

    return m


@dataclass
class SessionState:
    """Everything the HUD shows, plus the history the coach needs.

    Deliberately holds the CLAUSE history rather than the whole transcript: the
    live coach reasons about the last thing said, and a growing transcript in
    the payload would make every call slower and more expensive as the session
    went on - exactly backwards for a feature whose whole point is latency.
    """

    started_at: float = field(default_factory=time.time)
    metrics: LiveMetrics = field(default_factory=LiveMetrics)
    clauses: list[dict] = field(default_factory=list)
    # One entry per script line: its text, its pause target, and every attempt
    # made at it. Empty when no script was given, in which case the session
    # behaves as a plain continuous recording.
    lines: list[dict] = field(default_factory=list)
    current_line: int = 0
    notes: list[dict] = field(default_factory=list)
    coach_calls: int = 0
    coach_budget: int = 0
    # Clauses that were never transcribed because ASR could not keep up.
    # Surfaced rather than hidden: a transcript with silent holes in it is
    # worse than one that says how much it missed.
    asr_skipped: int = 0
    take_id: str | None = None
    recording_to: str | None = None
    # Written on stop when a script was used: the approved takes only.
    comped: str | None = None
    comp_missing: int = 0
    error: str | None = None

    @property
    def elapsed(self) -> float:
        return time.time() - self.started_at

    def add_clause(self, t_start: float, t_end: float, text: str,
                   wpm: float | None, words: int) -> dict:
        """Returns the dict, which the ASR thread MUTATES IN PLACE once the
        transcript lands. The clause appears on the HUD the instant the
        boundary is detected; the words arrive whenever they arrive."""
        c = {"i": len(self.clauses), "t_start": round(t_start, 2),
             "t_end": round(t_end, 2), "text": text,
             "wpm": round(wpm, 1) if wpm else None, "words": words,
             "pending": False, "skipped": False}
        self.clauses.append(c)
        return c

    def load_script(self, lines: list[dict]) -> None:
        self.lines = [{"i": i, "text": l.get("text", ""),
                       "pause": l.get("pause"), "state": "pending",
                       "attempts": []} for i, l in enumerate(lines)]
        if self.lines:
            self.lines[0]["state"] = "current"

    def record_attempt(self, clause: dict, passed: bool) -> None:
        """Log this clause as an attempt at the current line, and advance only
        if it was good.

        ADVANCING IS THE POINT. A director moves on because the line was right,
        not because you stopped talking. Every attempt is kept with its time
        range so the export can take the LAST GOOD one and leave the rest in the
        raw recording.
        """
        clause["line"] = self.current_line if self.lines else None
        clause["passed"] = passed
        if not self.lines:
            return
        if self.current_line >= len(self.lines):
            return                       # script finished; extra audio is tail

        line = self.lines[self.current_line]
        line["attempts"].append({"t_start": clause["t_start"],
                                 "t_end": clause["t_end"], "passed": passed})
        if passed:
            line["state"] = "done"
            self.current_line += 1
            if self.current_line < len(self.lines):
                self.lines[self.current_line]["state"] = "current"
        else:
            line["state"] = "retake"

    @property
    def script_done(self) -> bool:
        return bool(self.lines) and self.current_line >= len(self.lines)

    def add_note(self, text: str, kind: str = "coach",
                 meta: dict | None = None) -> dict:
        n = {"i": len(self.notes), "t": round(self.elapsed, 1), "kind": kind,
             "text": text, "meta": meta or {}}
        self.notes.append(n)
        return n

    def session_wpm(self) -> float | None:
        """Speaking rate over transcribed clauses only.

        Excludes the silence between clauses on purpose. Live, the useful
        question is "how fast am I talking", not "how much have I said in the
        last minute" - a thinking pause should not read as slowing down.
        """
        words = sum(c["words"] for c in self.clauses)
        speech = sum(c["t_end"] - c["t_start"] for c in self.clauses)
        if words < 8 or speech <= 0:
            return None
        return round(words / (speech / 60.0), 1)

    def to_dict(self) -> dict:
        m = self.metrics.to_dict()
        m["wpm"] = self.session_wpm()
        return {
            "elapsed": round(self.elapsed, 1),
            "metrics": m,
            "clauses": self.clauses[-8:],
            "notes": self.notes[-6:],
            "coach_calls": self.coach_calls,
            "coach_budget": self.coach_budget,
            "asr_skipped": self.asr_skipped,
            # Attempts are not sent - the HUD only needs to know each line's
            # state, and the list grows with every retake.
            "lines": [{k: v for k, v in l.items() if k != "attempts"}
                      for l in self.lines],
            "current_line": self.current_line,
            "script_done": self.script_done,
            "take_id": self.take_id,
            "recording_to": self.recording_to,
            "comped": self.comped,
            "comp_missing": self.comp_missing,
            "error": self.error,
        }
