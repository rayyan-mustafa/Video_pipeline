"""Clause-boundary detection.

WHY THIS IS THE CENTRAL PIECE OF PHASE 5. Live coaching is only allowed to
speak between clauses. A note that lands mid-sentence does not help - it
derails the read, and the speaker has to start the line again. So the whole
live loop is gated on this detector, and everything else waits for it.

HOW. Short-term RMS over 20 ms frames against an adaptive noise floor. A
boundary is: enough continuous speech, followed by enough continuous silence.
Both minimums matter -

  without `min_speech`  a cough or a lip noise between two silences reads as a
                        clause, and the coach interrupts a speaker who has not
                        said anything yet.
  without `min_silence` every inter-word gap is a boundary, and coaching fires
                        several times a second.

ADAPTIVE, NOT FIXED. The threshold tracks a rolling percentile of recent frame
levels rather than sitting at a fixed dBFS. A fixed threshold works in one room
at one gain setting and silently fails in every other - too high and the whole
take reads as silence, too low and nothing is ever a pause. The A6V's noise
floor also changes with mic distance, which is a thing this project actively
coaches the user to vary.

NO ASR HERE. This runs on raw energy, so it costs microseconds and does not
wait for a transcript. Punctuation-accurate clause detection would need the
words, and by the time they arrive the moment has passed.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

FRAME_SEC = 0.02


@dataclass
class Boundary:
    """One detected clause end, in seconds since capture started."""

    t_start: float          # where the speech run began
    t_end: float            # where it stopped, i.e. the clause's last audio
    speech_sec: float


@dataclass
class ClauseDetector:
    """Feed it audio blocks; it yields boundaries.

    Stateful and order-dependent: blocks must arrive in capture order, which
    they do because the PortAudio callback is serialized.
    """

    sr: int
    min_speech_sec: float = 0.7
    min_silence_sec: float = 0.45
    gate_db: float = 10.0          # above the rolling floor to count as speech
    floor_window_sec: float = 8.0

    _frames: deque = field(default_factory=deque, init=False)   # recent frame dB
    _carry: np.ndarray = field(default_factory=lambda: np.zeros(0, "float32"),
                               init=False)
    _t: float = field(default=0.0, init=False)          # time at end of _carry
    _speech_run: float = field(default=0.0, init=False)
    _silence_run: float = field(default=0.0, init=False)
    _speech_start: float | None = field(default=None, init=False)
    _last_speech_end: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self._frame_n = max(1, int(FRAME_SEC * self.sr))
        self._floor_frames = max(8, int(self.floor_window_sec / FRAME_SEC))

    # -----------------------------------------------------------------------
    @property
    def noise_floor_db(self) -> float:
        """20th percentile of recent frame levels.

        Not the minimum: one digitally-silent frame (a dropout, a buffer
        underrun) would drag a minimum to -inf and open the gate on everything.
        The same reasoning as `_DIGITAL_SILENCE` in the offline SNR estimator.
        """
        if len(self._frames) < 8:
            return -60.0
        return float(np.percentile(np.fromiter(self._frames, dtype=np.float64), 20))

    @property
    def threshold_db(self) -> float:
        return self.noise_floor_db + self.gate_db

    @property
    def in_speech(self) -> bool:
        return self._speech_run > 0.0

    def feed(self, block: np.ndarray) -> list[Boundary]:
        """Process one audio block, returning any boundaries it completed."""
        x = np.asarray(block, dtype=np.float32).reshape(-1)
        if len(x):
            x = np.concatenate([self._carry, x]) if len(self._carry) else x
        else:
            x = self._carry

        n_full = len(x) // self._frame_n
        self._carry = x[n_full * self._frame_n:].copy()
        if n_full == 0:
            return []

        frames = x[: n_full * self._frame_n].reshape(n_full, self._frame_n)
        # +1e-12 so a digitally silent frame is a very low number rather than
        # -inf, which would poison the percentile.
        levels = 20.0 * np.log10(np.sqrt((frames.astype(np.float64) ** 2).mean(1))
                                 + 1e-12)

        out: list[Boundary] = []
        for lvl in levels:
            self._t += FRAME_SEC
            thr = self.threshold_db
            self._frames.append(float(lvl))
            if len(self._frames) > self._floor_frames:
                self._frames.popleft()

            if lvl >= thr:
                if self._speech_run == 0.0:
                    self._speech_start = self._t - FRAME_SEC
                self._speech_run += FRAME_SEC
                self._silence_run = 0.0
                self._last_speech_end = self._t
            else:
                self._silence_run += FRAME_SEC
                if (self._speech_run >= self.min_speech_sec
                        and self._silence_run >= self.min_silence_sec):
                    out.append(Boundary(
                        t_start=self._speech_start or 0.0,
                        t_end=self._last_speech_end,
                        speech_sec=self._speech_run))
                    self._speech_run = 0.0
                    self._speech_start = None
                elif self._silence_run >= self.min_silence_sec:
                    # Too short to be a clause - a cough, a lip noise, a false
                    # trigger. Drop it silently rather than coaching on it.
                    self._speech_run = 0.0
                    self._speech_start = None
        return out
