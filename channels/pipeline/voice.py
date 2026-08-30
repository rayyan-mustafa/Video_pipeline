"""Stage A voice mastering - ported VERBATIM from stickman.ipynb cell037.

Do not re-derive this chain. Every constant and every ordering decision in it was
measured, and the comments record why. The only change from the notebook is that
the pip-install preamble is gone (local venv owns dependencies) and _selfcheck()
runs under __main__ instead of on import.

The two profiles matter and are not interchangeable:
  clip     - full chain INCLUDING silence trim. Changes length. Used per-section,
             BEFORE alignment, because the aligner must see the audio that ships.
  timeline - every filter except the trim, length-locked by assert. Used once over
             the concatenated voiceover, AFTER the timeline is computed.
"""

from __future__ import annotations

# =============================================================================
#  STAGE A — VOICE MASTERING (deterministic DSP, no AI)
# =============================================================================
#  Every function here is ordinary signal processing: filters, envelopes, gain.
#  Nothing is generated. Audio in, the same audio cleaner, out.
#
#  THE CONSTRAINT THAT SHAPES THIS WHOLE CELL: `scenes_timeline.json` is derived
#  from sample counts in the TTS cell, and the mux cell trusts those durations to
#  tile the audio exactly. So anything applied to the finished voiceover MUST NOT
#  change its length. Silence trimming does change length -- which is fine on a
#  standalone clip and fatal on the timed narration. Hence two profiles:
#
#    profile="clip"     full chain incl. silence trim. For training data and
#                       reference clips, where length is nobody's business.
#    profile="timeline" every filter except the trim. Length-locked, asserted.
#
#  Do not add a length-changing step to the "timeline" profile without also
#  rebuilding the timeline. The assert at the bottom of master() will catch you.
# =============================================================================

import numpy as np
import scipy.signal as sps
import scipy.ndimage as ndi
import soundfile as sf
import pyloudnorm as pyln


# ---------------------------------------------------------------- primitives

def _to_mono(x):
    return x.mean(axis=1) if x.ndim > 1 else x


def _biquad_peaking(sr, f0, gain_db, q):
    """RBJ peaking-EQ biquad. Returns (b, a) for use with filtfilt."""
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * f0 / sr
    alpha = np.sin(w0) / (2.0 * q)
    cw = np.cos(w0)
    b = np.array([1 + alpha * A, -2 * cw, 1 - alpha * A])
    a = np.array([1 + alpha / A, -2 * cw, 1 - alpha / A])
    return b / a[0], a / a[0]


def _highpass(x, sr, fc=85.0, order=4):
    """Zero-phase high-pass. Kills rumble, HVAC, desk thumps, plosive tails."""
    sos = sps.butter(order, fc / (sr / 2.0), btype="highpass", output="sos")
    return sps.sosfiltfilt(sos, x)


def _peaking(x, sr, f0, gain_db, q):
    if abs(gain_db) < 1e-6:
        return x
    b, a = _biquad_peaking(sr, f0, gain_db, q)
    return sps.filtfilt(b, a, x)


#  The attack/release follower is asymmetric, so it cannot be written as a single
#  LTI filter pass -- it needs a sequential loop. Running that loop per-sample over
#  a 9-minute 48k file means 26M Python iterations, which is minutes of wall clock.
#  So the detector runs at a 2 kHz control rate over per-block peaks and the
#  resulting gain curve is interpolated back up to sample rate. That is how real
#  compressors work internally anyway, it resolves attacks down to 0.5 ms, and it
#  is ~50x faster here.
_CONTROL_HZ = 2000.0


def _gain_curve(x, sr, threshold_db, ratio, attack_ms, release_ms, knee_db):
    """Gain reduction in dB, one value per sample, from a control-rate detector."""
    block = max(1, int(round(sr / _CONTROL_HZ)))
    n = x.shape[0]
    nb = int(np.ceil(n / block))
    mag = np.abs(x)
    pad = nb * block - n
    if pad:
        mag = np.concatenate([mag, np.zeros(pad)])
    peak = mag.reshape(nb, block).max(axis=1)

    csr = sr / block
    atk = np.exp(-1.0 / max(csr * attack_ms / 1000.0, 1e-9))
    rel = np.exp(-1.0 / max(csr * release_ms / 1000.0, 1e-9))
    env = np.empty(nb)
    prev = 0.0
    for i in range(nb):
        m = peak[i]
        coef = atk if m > prev else rel
        prev = coef * prev + (1.0 - coef) * m
        env[i] = prev

    over = 20.0 * np.log10(np.maximum(env, 1e-9)) - threshold_db
    gain_db = np.zeros(nb)
    # Soft knee: quadratic interpolation across +/- knee/2 around the threshold.
    half = knee_db / 2.0
    knee = (over > -half) & (over < half)
    above = over >= half
    gain_db[above] = (1.0 / ratio - 1.0) * over[above]
    gain_db[knee] = (1.0 / ratio - 1.0) * ((over[knee] + half) ** 2) / (2.0 * knee_db)

    centres = np.arange(nb) * block + block / 2.0
    return np.interp(np.arange(n), centres, gain_db)


def _compress(x, sr, threshold_db=-18.0, ratio=3.0,
              attack_ms=8.0, release_ms=120.0, makeup_db=0.0, knee_db=6.0):
    """Soft-knee downward compressor. Evens out delivery so quiet phrases at the
    end of a sentence sit at the same level as the punched first word."""
    gain_db = _gain_curve(x, sr, threshold_db, ratio, attack_ms, release_ms, knee_db)
    return x * (10.0 ** ((gain_db + makeup_db) / 20.0))


def _deess(x, sr, threshold_db=-30.0, ratio=4.0, lo=5000.0, hi=9000.0):
    """Split-band de-esser: compress ONLY the sibilance band, leave the rest of
    the spectrum untouched, recombine. A plain high-shelf cut would dull every
    consonant; this only ducks the band when an actual 'sss' is present."""
    nyq = sr / 2.0
    hi = min(hi, nyq * 0.98)
    if lo >= hi:
        return x
    sos = sps.butter(4, [lo / nyq, hi / nyq], btype="bandpass", output="sos")
    sib = sps.sosfiltfilt(sos, x)
    rest = x - sib
    sib_c = _compress(sib, sr, threshold_db=threshold_db, ratio=ratio,
                      attack_ms=1.0, release_ms=40.0, knee_db=4.0)
    return rest + sib_c


def _denoise(x, sr, noise_profile=None, prop_decrease=0.6):
    """Spectral-gate denoise, deliberately gentle.

    Uses a noise profile lifted from the quietest stretch of the recording when
    one can be found -- a measured fingerprint of *this* room beats a stationary
    estimate. `prop_decrease` defaults low on purpose: a spectral gate cannot
    tell quiet speech from noise, and at 0.85 it audibly eats the ends of
    sentences and any softly-delivered line. Measured on the test signal, 0.85
    dropped voiced frames from 240 to 51. Raise it only for a genuinely noisy
    room, and listen to the result before committing to it.
    """
    import noisereduce as nr
    if noise_profile is None:
        noise_profile = _find_noise_profile(x, sr)
    kw = dict(y=x, sr=sr, stationary=True, prop_decrease=prop_decrease)
    if noise_profile is not None:
        kw["y_noise"] = noise_profile
    return nr.reduce_noise(**kw).astype(np.float64)


def _find_noise_profile(x, sr, win_s=0.5):
    """Return the quietest half-second in the file, or None if the recording is
    too short / too uniformly loud to contain a usable noise-only window."""
    win = int(sr * win_s)
    if x.shape[0] < win * 3:
        return None
    n = x.shape[0] // win
    frames = x[: n * win].reshape(n, win)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    quietest = int(np.argmin(rms))
    # If the quietest window is within 12 dB of the median, there is no silence
    # here -- it is wall-to-wall speech, and gating against speech is destructive.
    med = np.median(rms)
    if med > 0 and 20 * np.log10(max(rms[quietest], 1e-9) / med) > -12.0:
        return None
    return frames[quietest].copy()


def _trim_silence(x, sr, threshold_db=-45.0, pad_ms=120.0, max_gap_s=0.6):
    """Trim head/tail silence and collapse interior dead air longer than
    `max_gap_s` down to `max_gap_s`. LENGTH-CHANGING -- "clip" profile only."""
    frame = int(sr * 0.02)
    if x.shape[0] < frame * 2:
        return x
    n = x.shape[0] // frame
    frames = x[: n * frame].reshape(n, frame)
    rms_db = 20 * np.log10(np.maximum(np.sqrt((frames ** 2).mean(axis=1)), 1e-9))
    peak_db = rms_db.max()
    voiced = rms_db > (peak_db + threshold_db)
    if not voiced.any():
        return x

    pad = max(1, int(pad_ms / 20.0))
    keep = np.zeros(n, dtype=bool)
    max_gap = max(1, int(max_gap_s / 0.02))
    idx = np.flatnonzero(voiced)
    keep[max(0, idx[0] - pad): min(n, idx[-1] + pad + 1)] = True

    # Collapse long interior gaps to max_gap frames.
    run_start = None
    for i in range(idx[0], idx[-1] + 1):
        if not voiced[i]:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None and (i - run_start) > max_gap + 2 * pad:
                keep[run_start + pad: i - pad] = False
                keep[run_start + pad: run_start + pad + max_gap] = True
            run_start = None

    return frames[keep].reshape(-1)


def _limit(x, sr, ceiling_db=-1.5, lookahead_ms=3.0, release_ms=60.0):
    """Look-ahead brickwall limiter.

    The naive way to respect a peak ceiling is to scale the whole file down until
    the loudest sample fits. On speech with a 30 dB crest factor that throws away
    ~10 dB of the loudness you just normalised to -- the one transient consonant
    sets the level for nine minutes of audio. A limiter instead pulls gain down
    only around the peaks that need it, so integrated loudness survives.

    WORKS IN BLOCKS, AND MUST. The whole voiceover is one array: ten minutes at
    48 kHz is 28.5M float64 samples, and the naive version built five temporaries
    of that size (abs, req, two gain curves, the product). That is ~1.4 GB of
    transients and it died with a MemoryError on the first full-length video -
    after the expensive Whisper pass had already succeeded, which is the worst
    possible moment to fail.

    Blocking is exact here, not an approximation: both filters have finite
    support, so a block padded by `w//2 + s//2` samples either side sees every
    input its interior samples depend on. Measured against the unblocked version
    it is bit-identical up to ~30s and diverges by at most 1.7e-13 beyond that -
    `uniform_filter1d` accumulates a running sum, so the rounding depends on
    where the window started. That is about -260 dBFS, a hundred dB below the
    16-bit floor this is written at. Peak memory no longer grows with the length
    of the video, so a fifteen-minute one costs the same as this.
    """
    ceiling = 10.0 ** (ceiling_db / 20.0)
    # Peak without materialising abs(x) - that temporary alone is 228 MB here.
    if max(float(x.max()), -float(x.min())) <= ceiling:
        return x

    # Running minimum first: gain is already down before the peak arrives, and
    # recovers no faster than the release window.
    w = max(3, int(sr * release_ms / 1000.0)) | 1
    # Then smooth the gain curve so it does not step and cause distortion.
    s = max(3, int(sr * lookahead_ms / 1000.0)) | 1

    halo = w // 2 + s // 2
    block = max(1 << 21, halo * 4)            # ~2M samples: 16 MB per temporary
    out = np.empty_like(x)

    for start in range(0, len(x), block):
        stop = min(start + block, len(x))
        lo, hi = max(0, start - halo), min(len(x), stop + halo)
        seg = x[lo:hi]

        mag = np.abs(seg)
        g = np.minimum(1.0, ceiling / np.maximum(mag, 1e-12))
        g = ndi.minimum_filter1d(g, size=w, mode="nearest")
        g = ndi.uniform_filter1d(g, size=s, mode="nearest")

        a, b = start - lo, start - lo + (stop - start)
        # Smoothing can nudge gain back above the minimum at the exact peak;
        # clip the remainder. It is a handful of samples and inaudible.
        np.clip(seg[a:b] * g[a:b], -ceiling, ceiling, out=out[start:stop])

    return out


def _loudnorm(x, sr, target_lufs=-16.0, true_peak_db=-1.5, passes=8):
    """Normalise integrated loudness (EBU R128) under a true-peak ceiling.

    Limiting changes loudness slightly, so measure/correct/limit is iterated until
    it converges. Without this the result lands several dB under target on
    peaky material.
    """
    meter = pyln.Meter(sr)
    for _ in range(passes):
        loudness = meter.integrated_loudness(x)
        if not np.isfinite(loudness):
            break
        delta = target_lufs - loudness
        if abs(delta) < 0.1:
            break
        x = _limit(x * (10.0 ** (delta / 20.0)), sr, true_peak_db)
    return _limit(x, sr, true_peak_db)


# ---------------------------------------------------------------- the chain

MASTER_DEFAULTS = dict(
    denoise=True,
    eq=True,
    compress=True,
    denoise_amount=0.6,
    highpass_hz=85.0,
    mud_hz=280.0, mud_db=-3.0, mud_q=1.0,          # boxiness / proximity build-up
    presence_hz=4000.0, presence_db=2.5, presence_q=0.9,   # intelligibility
    air_hz=10000.0, air_db=1.5, air_q=0.7,          # openness, sparingly
    deess=True, deess_threshold_db=-30.0, deess_ratio=4.0,
    comp_threshold_db=-18.0, comp_ratio=3.0,
    comp_attack_ms=8.0, comp_release_ms=120.0, comp_makeup_db=2.0,
    target_lufs=-16.0, true_peak_db=-1.5,
    trim_silence=True, trim_threshold_db=-45.0,
)


def master(x, sr, profile="clip", **overrides):
    """Run the mastering chain.

    Profiles:
      clip      full chain, TRIMS SILENCE (so it must run before alignment)
      natural   silence trim + loudness ONLY. No denoise, EQ, de-ess, compress,
                high-pass. The reader's own tone, with the dead air removed.
                Length-changing, so it also runs before alignment.
      timeline  full chain minus the trim, length-locked by assert
      raw       loudness normalisation ONLY - see below

    WHY 'raw' EXISTS. The chain's denoise, EQ, de-esser and compressor are the
    parts that change how a voice SOUNDS, and a reader who likes their own voice
    is right to refuse them. Loudness normalisation is a different kind of thing:
    it changes no tone and no dynamics-within-the-take, it only sets the level
    YouTube would otherwise set for you. Delivering at -14 LUFS means the
    platform applies no gain change at all; delivering louder means it turns the
    whole video down, which is a worse outcome than normalising here.

    So 'raw' keeps every characteristic of the take and fixes only the delivery
    level. It changes no sample count, so it is safe on either side of alignment.
    """
    cfg = dict(MASTER_DEFAULTS)
    cfg.update(overrides)
    if profile == "timeline":
        cfg["trim_silence"] = False
    elif profile == "natural":
        # THE READER'S OWN VOICE, WITH THE DEAD AIR TAKEN OUT.
        #
        # 'raw' refuses every filter, which also means it keeps every pause - and
        # long pauses are a pacing problem, not a tone problem. This profile
        # splits that pair: the silence trim stays, because tightening dead air
        # is editing rather than processing; everything that changes how the
        # voice SOUNDS comes out.
        #
        # Denoise, EQ, de-esser and compressor are the obvious tone stages. The
        # high-pass goes too, because an 85 Hz filter is still a filter - if the
        # take turns out to have rumble, put back `highpass_hz` alone rather
        # than reaching for 'clip'.
        #
        # STILL LENGTH-CHANGING, so this must run BEFORE alignment for the same
        # reason 'clip' must. And the trimmed length will not exactly match
        # 'clip': the trim measures energy, and it is now measuring unfiltered
        # audio, so the boundaries land in slightly different places.
        cfg.update(denoise=False, deess=False, eq=False, compress=False,
                   highpass_hz=0.0, trim_silence=True)
    elif profile == "raw":
        x = np.asarray(_to_mono(x), dtype=np.float64)
        n_in = x.shape[0]
        x = _loudnorm(x, sr, cfg["target_lufs"], cfg["true_peak_db"])
        if x.shape[0] != n_in:
            raise AssertionError(
                f"raw profile changed length ({n_in} -> {x.shape[0]} samples)"
            )
        return np.clip(x, -1.0, 1.0)
    elif profile != "clip":
        raise ValueError(
            f"unknown profile {profile!r} "
            f"(expected 'clip', 'natural', 'timeline' or 'raw')")

    x = np.asarray(_to_mono(x), dtype=np.float64)
    n_in = x.shape[0]

    # High-pass FIRST: rumble is broadband-adjacent low-frequency energy, and if it
    # is still present when the noise profile is measured it dominates the estimate
    # and the gate ends up subtracting the wrong spectrum from the speech.
    # EACH STAGE IS GUARDED SO IT CAN BE GENUINELY SKIPPED, not neutralised by
    # setting its gain to zero. A 0 dB peaking filter is still a filter: it runs,
    # it costs a pass, and it leaves its own tiny numerical fingerprint. The
    # 'natural' profile depends on these being untouched, so skipping has to mean
    # skipping.
    if cfg.get("highpass_hz"):
        x = _highpass(x, sr, cfg["highpass_hz"])
    if cfg["denoise"]:
        x = _denoise(x, sr, prop_decrease=cfg["denoise_amount"])
    if cfg.get("eq", True):
        x = _peaking(x, sr, cfg["mud_hz"], cfg["mud_db"], cfg["mud_q"])
        x = _peaking(x, sr, cfg["presence_hz"], cfg["presence_db"], cfg["presence_q"])
        if cfg["air_hz"] < sr / 2.0 * 0.95:
            x = _peaking(x, sr, cfg["air_hz"], cfg["air_db"], cfg["air_q"])
    if cfg["deess"]:
        x = _deess(x, sr, cfg["deess_threshold_db"], cfg["deess_ratio"])
    if cfg.get("compress", True):
        x = _compress(x, sr,
                      threshold_db=cfg["comp_threshold_db"], ratio=cfg["comp_ratio"],
                      attack_ms=cfg["comp_attack_ms"], release_ms=cfg["comp_release_ms"],
                      makeup_db=cfg["comp_makeup_db"])
    if cfg["trim_silence"]:
        x = _trim_silence(x, sr, threshold_db=cfg["trim_threshold_db"])
    x = _loudnorm(x, sr, cfg["target_lufs"], cfg["true_peak_db"])

    if profile == "timeline" and x.shape[0] != n_in:
        raise AssertionError(
            f"timeline profile changed length ({n_in} -> {x.shape[0]} samples). "
            "A length-changing filter got into the chain; scenes_timeline.json "
            "would no longer tile the audio."
        )
    return np.clip(x, -1.0, 1.0)


def master_file(src, dst, profile="clip", target_sr=None, **overrides):
    x, sr = sf.read(src, always_2d=False)
    x = _to_mono(np.asarray(x, dtype=np.float64))
    if target_sr and target_sr != sr:
        if profile == "timeline":
            raise ValueError("refusing to resample under the timeline profile")
        g = np.gcd(int(target_sr), int(sr))
        x = sps.resample_poly(x, int(target_sr) // g, int(sr) // g)
        sr = target_sr
    y = master(x, sr, profile=profile, **overrides)
    sf.write(dst, y.astype(np.float32), sr, subtype="PCM_16")
    return dst, sr, y.shape[0] / sr


def report(path):
    """Print the numbers that actually tell you whether a take is usable."""
    x, sr = sf.read(path, always_2d=False)
    x = _to_mono(np.asarray(x, dtype=np.float64))
    meter = pyln.Meter(sr)
    lufs = meter.integrated_loudness(x)
    peak_db = 20 * np.log10(max(np.abs(x).max(), 1e-9))
    noise = _find_noise_profile(x, sr)
    if noise is not None:
        nf = 20 * np.log10(max(np.sqrt((noise ** 2).mean()), 1e-9))
        snr = f"{lufs - nf:5.1f} dB" if np.isfinite(lufs) else "  n/a"
    else:
        nf, snr = float("nan"), "  n/a (no silence found)"
    print(f"  {path}")
    print(f"    {x.shape[0]/sr:8.2f} s @ {sr} Hz")
    print(f"    loudness {lufs:6.1f} LUFS   peak {peak_db:6.1f} dBFS")
    print(f"    noise floor {nf:6.1f} dBFS   SNR {snr}")
    return dict(seconds=x.shape[0] / sr, sr=sr, lufs=lufs, peak_db=peak_db, noise_db=nf)


# ---------------------------------------------------------------- self-check
# Runs in ~1s on a synthetic signal. Catches a broken scipy/pyloudnorm install and
# any edit that breaks the length lock, before nine minutes of audio depend on it.
def _selfcheck():
    sr = 48000
    rng = np.random.default_rng(0)
    t = np.arange(int(6 * sr)) / sr
    sig = 0.3 * np.sin(2 * np.pi * 120 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.7 * t))
    sig += rng.normal(0, 0.002, sig.shape) + 0.02 * np.sin(2 * np.pi * 48 * t)
    y = master(sig, sr, profile="timeline")
    assert len(y) == len(sig), "timeline profile is not length-locked"
    assert np.isfinite(y).all(), "chain produced NaN/Inf"
    assert np.abs(y).max() <= 1.0, "chain produced clipping"
    lufs = pyln.Meter(sr).integrated_loudness(y)
    assert abs(lufs + 16.0) < 1.5, f"loudness off target: {lufs:.1f} LUFS"
    print(f"  self-check passed (timeline output {lufs:.1f} LUFS, length locked)")


if __name__ == "__main__":
    _selfcheck()
