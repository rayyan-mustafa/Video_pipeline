"""Pitch and stress, via Praat (parselmouth).

parselmouth wraps Praat's autocorrelation pitch tracker, which is markedly more
reliable on real voice than librosa's spectral methods - particularly for the
low end of a male range, where octave errors are common.
"""

from __future__ import annotations

import numpy as np

from ..schemas import Pitch
from .audio import Audio

# Wide enough for any adult voice. Narrowing this per-speaker after calibration
# would cut octave errors further, but a wrong narrow range silently truncates
# real pitch movement, so the default stays generous.
F0_MIN = 60.0
F0_MAX = 400.0


def pitch_track(audio: Audio, f0_min: float = F0_MIN,
                f0_max: float = F0_MAX) -> np.ndarray:
    """Voiced-frame F0 values in Hz. Unvoiced frames are dropped, not zeroed -
    averaging zeros in would drag the mean toward nonsense."""
    import parselmouth

    snd = parselmouth.Sound(audio.samples.astype(np.float64), audio.sr)
    p = snd.to_pitch(pitch_floor=f0_min, pitch_ceiling=f0_max)
    values = p.selected_array["frequency"]
    return values[values > 0]


def analyze_pitch(audio: Audio) -> Pitch:
    """Mean / range / variance over voiced frames.

    Range uses the 5th-95th percentile rather than min-max: a single octave
    error at the extreme would otherwise dominate the number and make a normal
    read look wildly expressive.
    """
    v = pitch_track(audio)
    if v.size == 0:
        return Pitch(mean_hz=0.0, range_hz=0.0, variance=0.0)

    p5, p95 = np.percentile(v, [5, 95])
    return Pitch(
        mean_hz=round(float(np.mean(v)), 2),
        range_hz=round(float(p95 - p5), 2),
        variance=round(float(np.var(v)), 2),
    )


def voiced_fraction(audio: Audio) -> float:
    """Share of the take that is voiced. A useful sanity check: a very low
    value usually means the analysis is running on the wrong channel or on
    near-silence, not that the speaker whispered."""
    import parselmouth

    snd = parselmouth.Sound(audio.samples.astype(np.float64), audio.sr)
    p = snd.to_pitch(pitch_floor=F0_MIN, pitch_ceiling=F0_MAX)
    freq = p.selected_array["frequency"]
    return float((freq > 0).mean()) if freq.size else 0.0
