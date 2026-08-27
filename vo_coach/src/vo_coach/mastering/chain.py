"""The mastering chain.

    noise reduction -> high-pass -> de-esser -> EQ -> compressor
                    -> loudness normalize -> true-peak limit

ORDER IS NOT ARBITRARY. Each stage is placed where it is for a reason:

  1. NOISE REDUCTION FIRST. The compressor later in the chain raises quiet
     passages, and anything still in the noise floor at that point gets raised
     with them. Removing hiss after compression means removing hiss that has
     already been made louder and more audible.

  2. HIGH-PASS SECOND. Rumble and proximity buildup below ~80 Hz carry real
     energy, so leaving them in makes the compressor duck the whole voice every
     time a plosive lands. Removing them first means the compressor responds to
     the voice instead of to the desk being bumped.

  3. DE-ESS BEFORE THE COMPRESSOR. A broadband compressor reacts to sibilant
     peaks by pulling down the entire signal, which is audible as a duck on
     every "s". Taming the sibilant band first leaves the compressor working on
     the body of the voice.

  4. LOUDNESS LAST. Every stage above changes gain, so measuring LUFS before
     they run measures something that no longer exists.

  5. TRUE-PEAK LIMIT AFTER NORMALIZATION, then re-measure. Limiting changes
     loudness slightly, so the reported LUFS is taken after the limiter, not
     before - otherwise the report states a number the file does not have.
"""

from __future__ import annotations

import numpy as np

from ..config import Config, load_config
from ..schemas import StageReport
from ..analysis.audio import Audio, db


# Peak level to stage at before compression. -6 dBFS puts a normal voice's body
# around -18 to -20 dB RMS, which straddles the compressor threshold - engaged
# on the loud syllables, transparent on the quiet ones. Hotter than this and
# the compressor works on everything, which lifts the noise floor between words.
STAGING_PEAK_DB = -6.0

# Normalize -> limit passes. Each pass raises loudness and shaves the peaks that
# raise created, so the next needs less gain. Converges in 2-3 on speech; the
# cap stops a pathological take from looping.
LOUDNESS_ITERATIONS = 6
LOUDNESS_TOLERANCE_DB = 0.3


def _peak_db(x: np.ndarray) -> float:
    return db(float(np.max(np.abs(x)))) if x.size else -100.0


def _rms_db(x: np.ndarray) -> float:
    return db(float(np.sqrt(np.mean(np.square(x, dtype=np.float64)) + 1e-20)))


def _report(stage: str, x: np.ndarray, applied: bool = True,
            detail: str | None = None) -> StageReport:
    return StageReport(stage=stage, applied=applied, detail=detail,
                       peak_db=round(_peak_db(x), 2), rms_db=round(_rms_db(x), 2))


# ---------------------------------------------------------------------------
#  1. Noise reduction
# ---------------------------------------------------------------------------
def reduce_noise(x: np.ndarray, sr: int, amount: float = 0.75) -> np.ndarray:
    """Spectral-gate denoise.

    `amount` is deliberately below 1.0. Full-strength reduction on a voice with
    an audible floor (the A6V has one) produces the watery, phasey artefact
    people recognize instantly as "noise-reduced" - which is more distracting
    than the hiss it removed. Leaving some floor in place sounds cleaner than
    removing all of it badly.
    """
    import noisereduce as nr

    return nr.reduce_noise(y=x, sr=sr, stationary=True, prop_decrease=amount)


# ---------------------------------------------------------------------------
#  3. De-esser  (pedalboard has no de-esser - this is built here)
# ---------------------------------------------------------------------------
def deess(x: np.ndarray, sr: int, freq_hz: float = 6500.0,
          threshold_db: float = -28.0, ratio: float = 4.0) -> np.ndarray:
    """Split-band de-esser: compress only the sibilant band, then recombine.

    Pedalboard ships no de-esser, and a plain high-shelf cut is the wrong tool -
    it dulls every consonant in the take, not just the harsh ones, because it
    cannot tell an "s" from a "t". A de-esser has to be DYNAMIC: transparent
    when nothing is hissing, active only when it is.

    Implementation is a Linkwitz-Riley-style split via matched 4th-order
    Butterworth filters. Their sum is not perfectly flat, so a small phase
    colouration remains around the crossover - inaudible on speech at these
    ratios, and the alternative (a linear-phase FIR split) costs latency and
    complexity that a voice chain does not need.
    """
    from pedalboard import Compressor
    from scipy.signal import butter, sosfilt

    nyq = sr / 2.0
    if freq_hz >= nyq * 0.95:                 # nothing meaningful to split
        return x

    low_sos = butter(4, freq_hz / nyq, btype="low", output="sos")
    high_sos = butter(4, freq_hz / nyq, btype="high", output="sos")

    low = sosfilt(low_sos, x).astype(np.float32)
    high = sosfilt(high_sos, x).astype(np.float32)

    # Fast attack so the leading edge of a sibilant is caught, quick release so
    # the band recovers before the following vowel and does not sound dull.
    comp = Compressor(threshold_db=threshold_db, ratio=ratio,
                      attack_ms=1.0, release_ms=60.0)
    high_c = comp(high, sr)

    return (low + high_c).astype(np.float32)


# ---------------------------------------------------------------------------
#  2 / 4 / 5. Filters, EQ, compression
# ---------------------------------------------------------------------------
def build_board(cfg: Config):
    """High-pass, gentle corrective EQ, then compression.

    The EQ is deliberately mild. This is a corrective chain, not a colouring
    one: a big presence boost flatters a single take and then fights every
    other take on the channel. Two small moves only - a shelf trimming the
    low-mid mud that close-miking builds up, and a slight presence lift for
    consonant clarity.
    """
    from pedalboard import Compressor, HighpassFilter, LowShelfFilter, Pedalboard, PeakFilter

    return Pedalboard([
        HighpassFilter(cutoff_frequency_hz=cfg.get("mastering.highpass_hz")),
        # Boxiness from proximity effect concentrates around 200-400 Hz.
        LowShelfFilter(cutoff_frequency_hz=300.0, gain_db=-1.5, q=0.7),
        # Consonant definition, not brightness - kept under 2 dB on purpose.
        PeakFilter(cutoff_frequency_hz=3200.0, gain_db=1.5, q=0.8),
        Compressor(
            threshold_db=cfg.get("mastering.compressor_threshold_db"),
            ratio=cfg.get("mastering.compressor_ratio"),
            attack_ms=8.0,        # slow enough to let consonant transients through
            release_ms=140.0,     # fast enough to recover between phrases
        ),
    ])


# ---------------------------------------------------------------------------
#  6 / 7. Loudness and true peak
# ---------------------------------------------------------------------------
def measure_lufs(x: np.ndarray, sr: int) -> float | None:
    """Integrated loudness, or None when the file is too short to measure.

    EBU R128 needs at least 400 ms for a single gating block. Below that
    pyloudnorm returns -inf, and normalizing to -inf silences the take - so
    this returns None and the caller skips normalization instead.
    """
    import pyloudnorm as pyln

    if len(x) < int(0.4 * sr):
        return None
    loudness = pyln.Meter(sr).integrated_loudness(x.astype(np.float64))
    return None if not np.isfinite(loudness) else float(loudness)


def normalize_loudness(x: np.ndarray, sr: int, target_lufs: float) -> tuple[np.ndarray, float | None]:
    current = measure_lufs(x, sr)
    if current is None:
        return x, None
    gain = 10 ** ((target_lufs - current) / 20.0)
    return (x * gain).astype(np.float32), current


def limit_true_peak(x: np.ndarray, sr: int, ceiling_db: float,
                    lookahead_ms: float = 2.0,
                    release_ms: float = 60.0) -> np.ndarray:
    """Lookahead peak limiter. Guarantees peak <= ceiling, and never boosts.

    NEITHER pedalboard EFFECT WORKS HERE, both measured directly:

      Limiter    applies a fixed makeup gain - it lifts a -26 dBFS tone to
                 -21.3 dBFS despite that being 25 dB BELOW its threshold. In an
                 iterated normalize->limit loop that boost compounds, and six
                 passes landed 4 dB OVER target instead of converging.
      Compressor only ever reduces (correct), but its level detector is too
                 slow for speech transients - peaks still reached +5.9 dBFS
                 against a -1 dB threshold, so the final trim had to remove
                 6.9 dB and destroyed the loudness match.

    So the limiter is written here. Per-sample required gain, a running minimum
    over the lookahead window so gain is ALREADY down before a peak arrives
    (a limiter without lookahead always lets the first transient through), and
    a one-pole release so gain returns smoothly instead of stepping - a hard
    step back to unity is audible as a click.
    """
    from scipy.ndimage import minimum_filter1d

    ceil = 10.0 ** (ceiling_db / 20.0)
    absx = np.abs(x).astype(np.float64)
    if absx.max() <= ceil:
        return x

    required = np.ones_like(absx)
    hot = absx > ceil
    required[hot] = ceil / absx[hot]

    look = max(1, int(lookahead_ms * 1e-3 * sr))
    target_gain = minimum_filter1d(required, size=2 * look + 1, mode="nearest")

    # One-pole release: gain drops instantly (attack), recovers slowly.
    alpha = float(np.exp(-1.0 / max(1.0, release_ms * 1e-3 * sr)))
    smoothed = np.empty_like(target_gain)
    g = 1.0
    for i, t in enumerate(target_gain):
        g = t if t < g else alpha * g + (1.0 - alpha) * t
        smoothed[i] = g

    # Smoothing can only raise gain relative to the target, so re-clamp to keep
    # the guarantee exact rather than approximate.
    out = (x * np.minimum(smoothed, target_gain)).astype(np.float32)

    peak = float(np.max(np.abs(out)))
    if peak > ceil:                      # belt and braces
        out = (out * (ceil / peak)).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
#  The chain
# ---------------------------------------------------------------------------
def master(
    audio: Audio,
    cfg: Config | None = None,
    denoise: bool = True,
    deesser: bool = True,
) -> tuple[np.ndarray, list[StageReport], list[str]]:
    """Run the full chain. Returns (samples, stage_reports, warnings)."""
    cfg = cfg or load_config()
    x = audio.samples.astype(np.float32).copy()
    sr = audio.sr

    stages = [_report("input", x)]
    warnings: list[str] = []

    if denoise:
        try:
            x = reduce_noise(x, sr).astype(np.float32)
            stages.append(_report("noise_reduction", x, detail="spectral gate, 75%"))
        except Exception as e:
            stages.append(_report("noise_reduction", x, applied=False, detail=str(e)))
            warnings.append(f"noise reduction skipped: {e}")
    else:
        stages.append(_report("noise_reduction", x, applied=False, detail="disabled"))

    if deesser:
        x = deess(x, sr, freq_hz=cfg.get("mastering.deesser_freq_hz"))
        stages.append(_report("deesser", x,
                              detail=f"split at {cfg.get('mastering.deesser_freq_hz')} Hz"))
    else:
        stages.append(_report("deesser", x, applied=False, detail="disabled"))

    target = cfg.get("mastering.target_lufs")
    ceiling = cfg.get("mastering.true_peak_ceiling_db")

    # ---- GAIN STAGING: BY PEAK, NOT BY LOUDNESS --------------------------
    # A compressor threshold is an ABSOLUTE level, so the signal has to arrive
    # in its working range or the compressor does nothing. Denoising leaves the
    # take near -29 dB RMS, far under the -18 dBFS threshold.
    #
    # Staging by LUFS was the obvious fix and it was wrong: loudness ignores
    # peaks, so normalizing a high-crest-factor voice to -12 LUFS drove peaks
    # to +11.5 dBFS and slammed the compressor, which then squashed 7 dB of RMS
    # and lifted the noise floor with it (SNR fell 14.4 -> 9.6 dB).
    #
    # Peak staging is predictable: it puts the loudest sample at a known level,
    # so the compressor sees the crest factor the take actually has.
    peak = _peak_db(x)
    if np.isfinite(peak) and peak > -100:
        x = (x * 10 ** ((STAGING_PEAK_DB - peak) / 20.0)).astype(np.float32)
        stages.append(_report("gain_staging", x,
                              detail=f"peak {peak:.1f} -> {STAGING_PEAK_DB:.1f} dBFS, "
                                     f"into the compressor's range"))

    x = build_board(cfg)(x, sr).astype(np.float32)
    stages.append(_report("hpf_eq_compressor", x,
                          detail=f"hpf {cfg.get('mastering.highpass_hz')} Hz, "
                                 f"comp {cfg.get('mastering.compressor_ratio')}:1"))

    # ---- ITERATED NORMALIZE -> LIMIT --------------------------------------
    # Doing each of these ONCE cannot work, and both orderings fail differently:
    #
    #   normalize then limit : the limiter crushes the overshoot and changes
    #                          loudness, landing 2.6 dB off target.
    #   limit then normalize : normalization re-inflates the peaks, the final
    #                          ceiling trim removes ~8.5 dB, and the file ends
    #                          up 8.6 dB UNDER target - the trim undoing the
    #                          normalization entirely.
    #
    # The reason is crest factor. Speech arrives here around 23 dB peak-to-
    # loudness; hitting -16 LUFS under a -1 dBTP ceiling requires roughly 15 dB.
    # No single pass can both raise loudness and reduce crest factor, because
    # each undoes the other.
    #
    # Iterating converges: normalize lifts loudness, the limiter shaves the
    # peaks that lift created (lowering crest), and the next pass needs less
    # gain. This is how broadcast loudness normalizers work.
    initial = measure_lufs(x, sr)
    if initial is None:
        warnings.append(
            "too short to measure integrated loudness (needs >=400 ms) - "
            "loudness normalization skipped"
        )
        stages.append(_report("loudness", x, applied=False, detail="too short"))
    else:
        achieved = initial
        for i in range(LOUDNESS_ITERATIONS):
            x, _ = normalize_loudness(x, sr, target)
            x = limit_true_peak(x, sr, ceiling).astype(np.float32)
            measured = measure_lufs(x, sr)
            if measured is None:
                break
            achieved = measured
            if abs(measured - target) < LOUDNESS_TOLERANCE_DB:
                break

        stages.append(_report(
            "loudness", x,
            detail=f"{initial:.1f} -> {achieved:.1f} LUFS "
                   f"(target {target:.1f}, {i + 1} pass(es))"))

        if abs(achieved - target) >= 1.0:
            warnings.append(
                f"settled at {achieved:.1f} LUFS, {abs(achieved - target):.1f} dB "
                f"from the {target:.1f} target. The take's crest factor is too "
                f"high to reach it under a {ceiling} dBTP ceiling without "
                f"audible pumping - a more consistent read, or a lower "
                f"compressor threshold, gets you closer than more limiting will."
            )

    # ---- Guaranteed ceiling ----------------------------------------------
    # pedalboard's Limiter is not a brickwall - it overshoots. A flat trim is
    # the only way to guarantee the ceiling holds. After the loop above the
    # overshoot is small, so this costs a fraction of a dB rather than undoing
    # the normalization.
    peak = _peak_db(x)
    if peak > ceiling:
        x = (x * 10 ** ((ceiling - peak) / 20.0)).astype(np.float32)
        stages.append(_report("ceiling_trim", x,
                              detail=f"trimmed {ceiling - peak:.2f} dB"))

    return x, stages, warnings
