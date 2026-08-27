"""Generate synthetic test fixtures with known, labeled defects.

Dev-tooling item 1: validate each analysis module against signals whose correct
answer is known before pointing the pipeline at a real recording. Otherwise
every pipeline bug looks like a problem with the user's voice.

WHAT SYNTHETIC FIXTURES CAN AND CANNOT COVER - be honest about this:

  CAN  - spectral tilt (proximity effect), sibilance band energy, noise floor
         and SNR, pause/VAD boundaries, clipping detection. These are pure DSP
         measurements over a signal, so a synthesized signal with a known
         answer is a genuine test.

  CANNOT - WPM, word stress, mispronunciation. Those need real speech and a
         real ASR pass. Fixtures for those must be actual recordings, labeled
         by hand. `make_fixtures.py` writes a manifest slot for them so the
         gap is visible rather than forgotten.

Run:  python scripts/make_fixtures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 48000
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "fixtures"


def _speechlike(dur: float, seed: int = 0) -> np.ndarray:
    """Noise shaped roughly like a voice: energy concentrated 100-4000 Hz with
    a syllable-rate amplitude envelope. Not speech - just a carrier with
    voice-ish spectral and temporal statistics to measure against."""
    rng = np.random.default_rng(seed)
    n = int(dur * SR)
    x = rng.standard_normal(n)

    # crude bandpass via FFT masking
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1 / SR)
    shape = np.exp(-((np.log10(np.maximum(f, 1)) - np.log10(700)) ** 2) / 0.25)
    X *= shape
    x = np.fft.irfft(X, n)

    # ~4 Hz syllable envelope with pauses
    t = np.arange(n) / SR
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * t)
    env *= (np.sin(2 * np.pi * 0.25 * t) > -0.5)  # gaps between "phrases"
    x *= env
    return x / (np.max(np.abs(x)) + 1e-9) * 0.5


def _norm(x: np.ndarray, peak: float = 0.5) -> np.ndarray:
    return x / (np.max(np.abs(x)) + 1e-9) * peak


def clean(dur: float = 6.0) -> np.ndarray:
    """Baseline: the control. Every other fixture is this plus one defect."""
    return _speechlike(dur, seed=1)


def heavy_proximity(dur: float = 6.0) -> np.ndarray:
    """Bass buildup from close-miking: strong low-shelf boost below ~200 Hz.
    Expected: spectral tilt clearly more negative than the clean baseline."""
    x = _speechlike(dur, seed=1)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    boost = 1.0 + 8.0 * np.exp(-(f / 150.0) ** 2)   # +~18 dB at DC, tapering
    return _norm(np.fft.irfft(X * boost, len(x)))


def harsh_sibilance(dur: float = 6.0) -> np.ndarray:
    """Four short 5-9 kHz bursts on top of the baseline.
    Expected: 4 sibilance peaks detected, at ~1.0/2.0/4.0/5.5 s.

    BURST TIMES MUST LAND INSIDE VOICED REGIONS. `_speechlike` gates to silence
    between phrases, and a sibilant burst dropped into one of those gaps is not
    a realistic test signal - real sibilance happens mid-word, surrounded by
    vowel energy. The detector's speech-activity gate correctly rejects an
    isolated hiss burst in silence, so a fixture placing one there would be
    asserting the wrong behaviour. 2.5 s falls in a gap; 2.0 s does not.
    """
    x = _speechlike(dur, seed=1)
    rng = np.random.default_rng(7)
    for t0 in (1.0, 2.0, 4.0, 5.5):
        i0 = int(t0 * SR)
        n = int(0.12 * SR)
        burst = rng.standard_normal(n)
        B = np.fft.rfft(burst)
        f = np.fft.rfftfreq(n, 1 / SR)
        B *= ((f > 5000) & (f < 9000)).astype(float)
        burst = np.fft.irfft(B, n)
        burst = _norm(burst, 0.45) * np.hanning(n)
        x[i0:i0 + n] += burst
    return _norm(x)


def noisy(dur: float = 6.0, snr_db: float = 12.0) -> np.ndarray:
    """Baseline plus white noise at a known SNR.

    SNR IS DEFINED OVER SPEECH-ACTIVE REGIONS, not the whole file. This matters:
    the syllable envelope has near-silent gaps, and whole-file RMS sits ~4 dB
    below speech-active RMS because those gaps drag it down. Defining the
    fixture against whole-file RMS while the estimator measures speech-active
    windows builds a guaranteed ~4 dB disagreement into the ground truth.

    Speech-active is also the definition that means anything for the rubric:
    the question is whether hiss is masking WORDS, not whether it is loud
    relative to the silence between them.
    """
    x = _speechlike(dur, seed=1)
    active = np.abs(x) > 0.05 * np.max(np.abs(x))
    sig_rms = float(np.sqrt(np.mean(x[active] ** 2)))
    noise_rms = sig_rms / (10 ** (snr_db / 20))
    rng = np.random.default_rng(3)
    # _norm scales signal and noise together, so the ratio survives it.
    return _norm(x + rng.standard_normal(len(x)) * noise_rms, 0.6)


def long_pauses(dur: float = 8.0) -> np.ndarray:
    """Baseline with two inserted 1.2 s silences at ~2.0 s and ~5.0 s.
    Expected: VAD finds exactly those two long pauses."""
    x = _speechlike(dur, seed=1)
    for t0 in (2.0, 5.0):
        i0 = int(t0 * SR)
        x[i0:i0 + int(1.2 * SR)] = 0.0
    return x


def clipped(dur: float = 6.0) -> np.ndarray:
    """Driven past full scale and hard-clipped AT full scale.

    Clipping must be constructed at +/-1.0, not at some lower ceiling: real ADC
    clipping happens when the converter runs out of range, which is full scale
    by definition. A waveform flat-topped at 0.7 is a limiter, not clipping,
    and a detector that fired on it would flag every peak-normalized file.
    """
    return np.clip(_speechlike(dur, seed=1) * 2.5, -1.0, 1.0)


FIXTURES = {
    "clean_baseline":    (clean,           {"defect": None}),
    "heavy_proximity":   (heavy_proximity, {"defect": "proximity_effect",
                                            "expect": "tilt more negative than baseline"}),
    "harsh_sibilance":   (harsh_sibilance, {"defect": "sibilance",
                                            "expect_peaks_at": [1.0, 2.0, 4.0, 5.5]}),
    "noisy_snr12":       (noisy,           {"defect": "noise_floor",
                                            "expect_snr_db": 12.0, "tolerance_db": 2.0}),
    "long_pauses":       (long_pauses,     {"defect": "pauses",
                                            "expect_pauses_at": [2.0, 5.0],
                                            "expect_duration_sec": 1.2}),
    "clipped":           (clipped,         {"defect": "clipping"}),
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {}

    for name, (fn, meta) in FIXTURES.items():
        x = fn()
        path = OUT / f"{name}.wav"
        sf.write(path, x.astype(np.float32), SR, subtype="PCM_16")
        manifest[name] = {"file": path.name, "sample_rate": SR,
                          "duration_sec": round(len(x) / SR, 3), **meta}
        print(f"  wrote {path.name:<24} {len(x)/SR:.2f}s  ({meta.get('defect') or 'control'})")

    # Slots for fixtures that CANNOT be synthesized - real recordings, hand
    # labeled. Listed explicitly so the coverage gap stays visible.
    manifest["_needs_real_recordings"] = {
        "note": "WPM, word stress and mispronunciation need real speech + ASR. "
                "Record these on the A6V and label by hand before trusting "
                "pacing/pronunciation scoring.",
        "wanted": [
            {"name": "too_fast", "expect": "sustained >170 wpm for 3+ sentences"},
            {"name": "too_slow", "expect": "sustained <140 wpm"},
            {"name": "mispronounced_word", "expect": "one flagged word, labeled"},
            {"name": "wrong_syllable_stress", "expect": "one flagged word, labeled"},
        ],
    }

    with open(OUT / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nmanifest -> {OUT / 'manifest.json'}")


if __name__ == "__main__":
    main()
