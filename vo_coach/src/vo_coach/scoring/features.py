"""Reduce an AnalysisResult to the comparable feature vector.

The same function runs over the user's take and over a reference voice, so the
two are guaranteed to be measured identically. Any asymmetry here would show up
as a delta that is really just a methodology difference.
"""

from __future__ import annotations

import numpy as np

from ..schemas import AnalysisResult, FeatureVector


def extract(r: AnalysisResult) -> FeatureVector:
    """Normalize everything rate-like to per-minute so a 60 s reference and a
    3 min take are directly comparable. Counting raw events instead would make
    every longer take look worse on sibilance and pauses."""
    minutes = max(r.duration_sec / 60.0, 1e-6)

    wpm_p10 = wpm_p90 = None
    if r.pacing.wpm_windowed:
        vals = [p.wpm for p in r.pacing.wpm_windowed]
        wpm_p10, wpm_p90 = (round(float(v), 2) for v in np.percentile(vals, [10, 90]))

    tilt = r.proximity_effect.spectral_tilt_db_per_octave
    tilt_mean = float(np.mean([p.value for p in tilt])) if tilt else 0.0

    pause_durations = [p.duration for p in r.pauses]
    unaligned = None
    if r.pauses:
        # Only meaningful when a script was supplied. Without one every pause is
        # flagged unaligned by construction, which would read as a real finding.
        if any(p.aligned_to_punctuation for p in r.pauses):
            unaligned = round(
                sum(1 for p in r.pauses if not p.aligned_to_punctuation) / len(r.pauses),
                3,
            )

    return FeatureVector(
        wpm_overall=r.pacing.wpm_overall,
        wpm_p10=wpm_p10,
        wpm_p90=wpm_p90,
        pitch_mean_hz=r.pitch.mean_hz,
        pitch_range_hz=r.pitch.range_hz,
        spectral_tilt_mean=round(tilt_mean, 3),
        sibilance_per_min=round(len(r.sibilance.peaks) / minutes, 2),
        pause_rate_per_min=round(len(r.pauses) / minutes, 2),
        mean_pause_sec=round(float(np.mean(pause_durations)), 3) if pause_durations else 0.0,
        unaligned_pause_ratio=unaligned,
    )
