"""Audio loading and shared primitives.

One place that decides how a file becomes a float array, so every analysis
module measures the same signal at the same rate. Silent disagreement about
sample rate between two modules is the sort of bug that shows up as a
plausible-but-wrong number rather than a crash.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass
class Audio:
    """Mono float32 samples plus the rate they are at."""

    samples: np.ndarray
    sr: int
    path: str | None = None

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sr

    def slice(self, t0: float, t1: float) -> np.ndarray:
        return self.samples[int(t0 * self.sr): int(t1 * self.sr)]


def load(path: str | Path, target_sr: int | None = None) -> Audio:
    """Load to mono float. Resamples only when asked and only when needed.

    Note the mono fold is a plain channel mean. For a cardioid condenser like
    the A6V that is always a single capsule, so there is no phase-cancellation
    risk here; a true stereo source would need care.
    """
    x, sr = sf.read(str(path), always_2d=False, dtype="float64")
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)

    if target_sr and sr != target_sr:
        # Rational resampling keeps the phase behaviour predictable, which
        # matters because pause boundaries are measured in samples downstream.
        from scipy.signal import resample_poly

        g = np.gcd(int(target_sr), int(sr))
        x = resample_poly(x, target_sr // g, sr // g)
        sr = target_sr

    return Audio(samples=x.astype(np.float32), sr=int(sr), path=str(path))


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + 1e-20))


def db(amplitude: float) -> float:
    """Amplitude ratio -> dB. Floored so digital silence returns a very low
    number instead of -inf, which would propagate NaN through the rubric."""
    return float(20.0 * np.log10(max(amplitude, 1e-10)))


def frame_signal(x: np.ndarray, sr: int, win_sec: float, hop_sec: float):
    """Yield (t_center, frame) windows. Used by every time-varying metric so
    they all share one framing convention."""
    win = max(1, int(win_sec * sr))
    hop = max(1, int(hop_sec * sr))
    for start in range(0, max(1, len(x) - win + 1), hop):
        yield (start + win / 2) / sr, x[start: start + win]


def is_clipped(x: np.ndarray, threshold: float = 0.99,
               min_run: int = 3) -> list[float]:
    """Sample indices (as seconds needs sr - returns fractional positions) where
    consecutive samples sit at full scale.

    A single sample at 1.0 is not clipping; a run of them is. Returns the start
    index of each run, in samples.
    """
    at_ceiling = np.abs(x) >= threshold
    if not at_ceiling.any():
        return []

    runs: list[float] = []
    count = 0
    for i, hot in enumerate(at_ceiling):
        if hot:
            count += 1
        else:
            if count >= min_run:
                runs.append(i - count)
            count = 0
    if count >= min_run:
        runs.append(len(at_ceiling) - count)
    return runs
