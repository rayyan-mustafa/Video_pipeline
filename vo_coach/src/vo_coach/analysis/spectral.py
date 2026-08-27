"""Spectral measurements: proximity effect, sibilance, noise floor, SNR.

These are the metrics the synthetic fixtures can genuinely validate, because
each is a deterministic measurement over a signal whose correct answer is known
by construction. Everything here should be testable without ASR.
"""

from __future__ import annotations

import numpy as np

from ..schemas import SibilancePeak, TiltPoint
from .audio import Audio, db, frame_signal, rms

# Tilt is measured across the band where voice actually lives. Going below
# ~80 Hz just measures rumble and HVAC; above ~8 kHz it measures air and hiss.
TILT_BAND = (80.0, 8000.0)


def _spectrum(frame: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    win = np.hanning(len(frame))
    mag = np.abs(np.fft.rfft(frame * win))
    freqs = np.fft.rfftfreq(len(frame), 1.0 / sr)
    return freqs, mag


def spectral_tilt(audio: Audio, win_sec: float = 0.25,
                  hop_sec: float = 0.10) -> list[TiltPoint]:
    """dB per octave, fitted over the voice band, per window.

    More negative = more low-frequency energy relative to high = the bass
    buildup you get from working a cardioid too close. This is a proxy for
    proximity effect, not a direct measure of mic distance: a naturally deep
    voice also tilts negative. That is why the rubric compares tilt against the
    speaker's own baseline rather than an absolute number.
    """
    out: list[TiltPoint] = []
    lo, hi = TILT_BAND

    for t, frame in frame_signal(audio.samples, audio.sr, win_sec, hop_sec):
        if rms(frame) < 1e-5:          # silence has no meaningful tilt
            continue
        freqs, mag = _spectrum(frame, audio.sr)
        band = (freqs >= lo) & (freqs <= hi)
        if band.sum() < 8:
            continue

        # Least-squares line through (log2 f, dB). Slope is dB/octave directly.
        octaves = np.log2(freqs[band])
        levels = 20.0 * np.log10(mag[band] + 1e-10)
        slope = float(np.polyfit(octaves, levels, 1)[0])
        out.append(TiltPoint(t=round(t, 3), value=round(slope, 3)))

    return out


# Sibilance is judged against the voice band, not against the whole signal.
VOICE_BAND = (300.0, 4000.0)

# How far below median voice-band level a frame may sit and still count as
# speech. Wide enough to keep quiet consonants, tight enough to reject gaps.
ACTIVITY_GATE_DB = 12.0

# How far the sibilant band must rise above its own speech-active median, in
# absolute terms, before a frame counts as a sibilant rather than as hiss.
ABSOLUTE_PROMINENCE_DB = 4.0


def sibilance_peaks(audio: Audio, band_hz: tuple[float, float] = (5000.0, 9000.0),
                    win_sec: float = 0.05, hop_sec: float = 0.02,
                    prominence_db: float = 6.0) -> list[SibilancePeak]:
    """Find moments where the sibilant band spikes RELATIVE TO the voice band.

    The metric is the ratio 5-9 kHz : 300-4000 Hz, not absolute energy in the
    sibilant band. Absolute band energy compared against its own median is
    unusable in practice for two reasons, both observed on the fixtures:

      - broadband noise (mic self-noise, room tone) lives in 5-9 kHz too, so a
        clean take fires on constant low-level fluctuation - 19 false peaks on
        a fixture containing no sibilance at all;
      - adding white noise RAISES the median, flattening prominence, so a
        genuinely noisy take reports FEWER peaks than a clean one.

    A real /s/ is defined by disproportionate high-frequency energy relative to
    the vowel energy around it, which is exactly what the ratio captures - and
    it is level-independent, so it does not punish recording hot.
    """
    lo, hi = band_hz
    vlo, vhi = VOICE_BAND
    times: list[float] = []
    energies: list[float] = []     # sibilant band level, for reporting
    ratios: list[float] = []       # sibilant : voice, for detection
    voice_levels: list[float] = []  # for the speech-activity gate

    for t, frame in frame_signal(audio.samples, audio.sr, win_sec, hop_sec):
        if rms(frame) < 1e-5:      # silence has no sibilance
            continue
        freqs, mag = _spectrum(frame, audio.sr)
        sib = (freqs >= lo) & (freqs <= hi)
        voice = (freqs >= vlo) & (freqs <= vhi)
        if not sib.any() or not voice.any():
            continue

        sib_e = float(np.sqrt(np.mean(mag[sib] ** 2)))
        voice_e = float(np.sqrt(np.mean(mag[voice] ** 2)))
        times.append(t)
        energies.append(db(sib_e))
        voice_levels.append(db(voice_e))
        ratios.append(db(sib_e) - db(voice_e))

    if not ratios:
        return []

    e = np.asarray(energies)
    r = np.asarray(ratios)
    v = np.asarray(voice_levels)

    # SPEECH-ACTIVITY GATE. Between words the voice band drops out while mic
    # hiss - which is broadband, so relatively HF-heavy - keeps going. The
    # sibilant:voice ratio therefore spikes in every gap, and a hissy take
    # reports sibilance everywhere it is silent. Only frames carrying actual
    # voice energy are eligible.
    active = v > (float(np.median(v)) - ACTIVITY_GATE_DB)
    if active.sum() < 3:
        return []

    baseline = float(np.median(r[active]))
    prominence = r - baseline

    # A real sibilant is loud in ABSOLUTE terms too, not merely HF-weighted.
    # Broadband hiss raises the ratio uniformly across a take while its absolute
    # 5-9 kHz level stays flat - on the noisy fixture the entire spread is under
    # 3 dB. Requiring absolute prominence as well as ratio prominence separates
    # "this frame has an /s/ in it" from "this whole recording is hissy", which
    # is a mastering problem, not a delivery note.
    abs_baseline = float(np.median(e[active]))

    peaks: list[SibilancePeak] = []
    for i in range(1, len(r) - 1):
        if not active[i]:
            continue
        if prominence[i] < prominence_db:
            continue
        if (e[i] - abs_baseline) < ABSOLUTE_PROMINENCE_DB:
            continue
        if r[i] < r[i - 1] or r[i] < r[i + 1]:   # local max only
            continue
        # Suppress hits inside the same sibilant (~120 ms apart)
        if peaks and times[i] - peaks[-1].t < 0.12:
            if prominence[i] > (peaks[-1].band_energy_db - baseline):
                peaks[-1] = SibilancePeak(t=round(times[i], 3),
                                          band_energy_db=round(float(e[i]), 2))
            continue
        peaks.append(SibilancePeak(t=round(times[i], 3),
                                   band_energy_db=round(float(e[i]), 2)))

    return peaks


# Below this amplitude a window is treated as digital silence rather than as a
# measurement of the room. ~-80 dBFS: quieter than any real mic preamp.
_DIGITAL_SILENCE = 1e-4


def noise_floor_and_snr(audio: Audio, win_sec: float = 0.05,
                        hop_sec: float = 0.025) -> tuple[float, float | None]:
    """(noise_floor_db, snr_db) - snr is None when it cannot be measured.

    The noise floor is a low percentile of windowed RMS rather than the minimum:
    the minimum lands on a single freak-quiet window and reads far too
    optimistic. Signal level is the median of windows above that floor, which
    ignores both silence and the loudest peaks.

    DIGITAL SILENCE IS EXCLUDED FROM THE FLOOR ESTIMATE. A file containing true
    zero samples - anything gated, edited, or synthesized - otherwise drives the
    percentile to 0, which floors to -200 dB and yields an SNR of ~175 dB. That
    number is not a clean recording, it is a broken measurement, and a rubric
    reading it would conclude the take is flawless. When no genuine room tone
    exists to measure, this returns None instead of a fabricated number.

    For the A6V specifically the real value does real work: its self-noise is
    audible, and an SNR under ~20 dB means hiss is starting to mask consonants.
    """
    levels = np.asarray([
        rms(frame) for _, frame in frame_signal(audio.samples, audio.sr,
                                                win_sec, hop_sec)
    ])
    if levels.size == 0:
        return -100.0, None

    audible = levels[levels > _DIGITAL_SILENCE]
    if audible.size < max(4, levels.size // 20):
        # Almost everything is digital silence: no room tone to characterize.
        return db(float(levels.max())), None

    noise_amp = float(np.percentile(audible, 10))
    if noise_amp <= _DIGITAL_SILENCE:
        return db(noise_amp), None

    speech = audible[audible > noise_amp * 2.0]
    signal_amp = float(np.median(speech)) if speech.size else float(audible.max())

    noise_db = db(noise_amp)
    return round(noise_db, 2), round(float(db(signal_amp) - noise_db), 2)


def clipping_times(audio: Audio, threshold: float = 0.99) -> list[float]:
    """Seconds at which a run of full-scale samples starts."""
    from .audio import is_clipped

    return [round(i / audio.sr, 3)
            for i in is_clipped(audio.samples, threshold=threshold)]
