"""
The QC gate: measure the recordings on the laptop, before any GPU is paid for.

=============================================================================
 WHY THIS EXISTS
=============================================================================
Today the first thing that notices bad audio is Whisper, in Colab Phase 1, on a
paid runtime -- and it does not report it. It just aligns badly. The frames are
then drawn against a bad timeline and you find out at mux time, or on watching
the thing. By then the T4 minutes are spent and the fix is "re-record", which
was free an hour earlier.

Everything here is arithmetic over the samples. No model, no GPU, no network.

=============================================================================
 THE MEASUREMENTS ARE vo_coach's, DELIBERATELY
=============================================================================
Clipping is runs of full-scale samples (>= 0.99, three or more in a row), not
single samples -- one sample at full scale is not clipping, and vo_coach's live
monitor learned that the hard way when a one-sample flag stopped the reader on
line one. The noise floor is the 10th percentile of windowed RMS with digital
silence excluded, because a file containing true zeros otherwise reports a ~175
dB SNR and reads as flawless.

Those definitions are ported rather than imported: `pipeline/` has to stay
installable with nothing but numpy, and vo_coach pulls in soundfile, scipy,
sounddevice and an ASR stack. Ported code drifts, so the thresholds and the
algorithm both live here in one place and the tests pin the numbers.

=============================================================================
 WHY THERE IS A WAV READER IN HERE
=============================================================================
The obvious answer is soundfile, and soundfile means libsndfile. `wave` from the
standard library refuses float WAVs, which is exactly what vo_coach writes when
`audio.bit_depth` is 32. Parsing the RIFF chunks directly is about fifty lines,
handles both, and keeps the local half installable with one dependency.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# -----------------------------------------------------------------------------
#  Thresholds. Every one of these is a decision, so each says where it came from.
# -----------------------------------------------------------------------------
CLIP_THRESHOLD = 0.99        # |sample| at or above this is at the ceiling
CLIP_MIN_RUN = 3             # consecutive samples before it counts as a run
CLIP_RUNS_WARN = 1           # any clipping at all is worth saying out loud
CLIP_RUNS_PER_MIN_FAIL = 1.0  # sustained clipping - the take is not recoverable

PEAK_TARGET_DB = -9.0        # where peaks want to sit (STATUS.md)
PEAK_HOT_WARN_DB = -3.0      # above this there is no headroom left for mastering
PEAK_HOT_FAIL_DB = -0.5      # effectively at the ceiling
PEAK_QUIET_WARN_DB = -30.0   # so far down that mastering has to lift the hiss too

SNR_FAIL_DB = 20.0           # vo_coach thresholds.yaml must_fix_snr_db
SNR_WARN_DB = 30.0           # vo_coach thresholds.yaml silence_snr_db

# Pace, expressed as implied words-per-minute against the read sheet's word count.
# The band is deliberately wide: the personal baseline does not exist yet (the
# three calibration takes were never recorded), so this is checking for a wrong
# or truncated file, NOT coaching the delivery. vo_coach judges pace; this does
# not.
WPM_WARN = (110.0, 200.0)
WPM_FAIL = (70.0, 260.0)

MIN_DURATION_S = 5.0         # shorter than this is a misfire, not a section
TARGET_SR = 48000            # what the notebook resamples everything to

_DIGITAL_SILENCE = 1e-4      # ~-80 dBFS: quieter than any real preamp


# =============================================================================
#  WAV READING
# =============================================================================

class UnreadableAudio(Exception):
    """Not a WAV, or a WAV in a form this reader will not guess at."""


def read_wav(path: str | Path) -> tuple[np.ndarray, int, dict]:
    """(mono float64 in [-1, 1], sample rate, info). No resampling.

    info carries channels, bit depth and the encoding, because a section
    recorded at a different rate or channel count than its neighbours is a real
    finding even though the notebook would resample it away.
    """
    path = Path(path)
    raw = path.read_bytes()
    if len(raw) < 12 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise UnreadableAudio(f"{path.name} is not a RIFF/WAVE file")

    fmt = None
    data = None
    pos = 12
    while pos + 8 <= len(raw):
        cid = raw[pos:pos + 4]
        size = struct.unpack_from("<I", raw, pos + 4)[0]
        body = raw[pos + 8: pos + 8 + size]
        if cid == b"fmt ":
            fmt = body
        elif cid == b"data":
            # A file still being written by Drive can declare more data than it
            # holds. Take what is actually there rather than raising - the
            # duration check is what should report a short file.
            data = body
        pos += 8 + size + (size & 1)      # chunks are word-aligned

    if fmt is None or data is None:
        raise UnreadableAudio(f"{path.name} has no fmt/data chunk")

    audio_format, channels, sr, _byte_rate, _align, bits = struct.unpack_from("<HHIIHH", fmt, 0)
    if audio_format == 0xFFFE and len(fmt) >= 26:      # WAVE_FORMAT_EXTENSIBLE
        audio_format = struct.unpack_from("<H", fmt, 24)[0]

    if audio_format == 1:                              # PCM
        if bits == 8:
            x = (np.frombuffer(data, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
        elif bits == 16:
            x = np.frombuffer(data, dtype="<i2").astype(np.float64) / 32768.0
        elif bits == 24:
            b = np.frombuffer(data[: len(data) - len(data) % 3], dtype=np.uint8)
            b = b.reshape(-1, 3).astype(np.int32)
            v = (b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16))
            v = np.where(v & 0x800000, v - 0x1000000, v)
            x = v.astype(np.float64) / 8388608.0
        elif bits == 32:
            x = np.frombuffer(data, dtype="<i4").astype(np.float64) / 2147483648.0
        else:
            raise UnreadableAudio(f"{path.name}: {bits}-bit PCM is not supported")
    elif audio_format == 3:                            # IEEE float
        dt = "<f4" if bits == 32 else "<f8"
        x = np.frombuffer(data[: len(data) - len(data) % (bits // 8)], dtype=dt).astype(np.float64)
    else:
        raise UnreadableAudio(
            f"{path.name}: WAV format tag {audio_format} (compressed?) - "
            "re-export as PCM or float WAV")

    if channels > 1:
        usable = len(x) - (len(x) % channels)
        x = x[:usable].reshape(-1, channels).mean(axis=1)

    return x, int(sr), {"channels": int(channels), "bits": int(bits),
                        "encoding": "float" if audio_format == 3 else "pcm"}


# =============================================================================
#  MEASUREMENTS
# =============================================================================

def _db(amplitude: float) -> float:
    return float(20.0 * np.log10(max(float(amplitude), 1e-10)))


def clip_runs(x: np.ndarray, threshold: float = CLIP_THRESHOLD,
              min_run: int = CLIP_MIN_RUN) -> int:
    """Number of runs of `min_run`+ consecutive full-scale samples.

    Vectorised because a section is millions of samples and this is supposed to
    run in the seconds between finishing a take and starting the next one.
    """
    hot = np.abs(x) >= threshold
    if not hot.any():
        return 0
    edges = np.diff(np.concatenate(([0], hot.view(np.int8), [0])))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return int(np.count_nonzero((ends - starts) >= min_run))


def _windowed_rms(x: np.ndarray, sr: int, win_sec: float = 0.05,
                  hop_sec: float = 0.025) -> np.ndarray:
    win = max(1, int(win_sec * sr))
    hop = max(1, int(hop_sec * sr))
    if len(x) < win:
        return np.asarray([np.sqrt(np.mean(np.square(x)) + 1e-20)]) if len(x) else np.asarray([])
    n = 1 + (len(x) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    return np.sqrt(np.mean(np.square(x[idx]), axis=1) + 1e-20)


def noise_floor_and_snr(x: np.ndarray, sr: int) -> tuple[float, float | None]:
    """(noise floor dBFS, SNR dB). SNR is None when there is no room tone to measure.

    Ported from vo_coach analysis/spectral.py -- see the module docstring for why
    digital silence is excluded rather than measured.
    """
    levels = _windowed_rms(x, sr)
    if levels.size == 0:
        return -100.0, None

    audible = levels[levels > _DIGITAL_SILENCE]
    if audible.size < max(4, levels.size // 20):
        return _db(levels.max()), None

    noise_amp = float(np.percentile(audible, 10))
    if noise_amp <= _DIGITAL_SILENCE:
        return _db(noise_amp), None

    speech = audible[audible > noise_amp * 2.0]
    signal_amp = float(np.median(speech)) if speech.size else float(audible.max())
    noise_db = _db(noise_amp)
    return round(noise_db, 2), round(_db(signal_amp) - noise_db, 2)


# =============================================================================
#  THE VERDICT
# =============================================================================

@dataclass
class SectionReport:
    section: int
    path: str | None = None
    ok: bool = True                       # False only for a FAIL; warnings still pass
    duration: float = 0.0
    sr: int = 0
    channels: int = 0
    bits: int = 0
    peak_db: float = -100.0
    clip_runs: int = 0
    clip_runs_per_min: float = 0.0
    noise_floor_db: float = -100.0
    snr_db: float | None = None
    expected_words: int | None = None
    implied_wpm: float | None = None
    fails: list[str] = field(default_factory=list)
    warns: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        if self.fails:
            return "FAIL"
        return "WARN" if self.warns else "PASS"


def check_file(path: str | Path, section: int,
               expected_words: int | None = None) -> SectionReport:
    """Measure one section file and decide what it means, in words."""
    r = SectionReport(section=section, path=str(path), expected_words=expected_words)

    if Path(path).suffix.lower() != ".wav":
        r.ok = False
        r.fails.append(
            f"{Path(path).suffix or 'this format'} cannot be measured here - "
            "the notebook will accept it, but nothing checks it first. "
            "Re-export the take as a WAV.")
        return r

    try:
        x, sr, info = read_wav(path)
    except FileNotFoundError:
        r.ok = False
        r.fails.append("the file is not there any more")
        return r
    except (UnreadableAudio, OSError, ValueError, struct.error) as e:
        # A file Drive is still syncing is truncated rather than absent, and a
        # truncated header is a struct.error, not something recognisable.
        r.ok = False
        r.fails.append(f"cannot be read ({e}). If Drive is still syncing it, wait; "
                       f"otherwise re-export it as a WAV.")
        return r

    r.sr, r.channels, r.bits = sr, info["channels"], info["bits"]
    r.duration = round(len(x) / sr, 2) if sr else 0.0

    if len(x) == 0:
        r.ok = False
        r.fails.append("the file has no audio in it at all")
        return r

    # --- level ------------------------------------------------------------
    r.peak_db = round(_db(float(np.max(np.abs(x)))), 2)
    r.clip_runs = clip_runs(x)
    r.clip_runs_per_min = round(r.clip_runs / max(r.duration / 60.0, 1e-6), 2)

    if r.clip_runs_per_min >= CLIP_RUNS_PER_MIN_FAIL:
        r.fails.append(
            f"clipping: {r.clip_runs} runs of full-scale samples "
            f"({r.clip_runs_per_min:.1f}/min). The waveform is gone where it is "
            f"flat and no model gets it back - it becomes a worse transcript, "
            f"then scene boundaries in the wrong place. Turn the Windows input "
            f"volume down until peaks sit near {PEAK_TARGET_DB:.0f} dBFS and "
            f"record this section again.")
    elif r.clip_runs >= CLIP_RUNS_WARN:
        r.warns.append(
            f"clipping: {r.clip_runs} run(s), {r.clip_runs_per_min:.1f}/min. Not "
            f"enough to throw the take away, but the gain is at the edge.")

    if r.peak_db >= PEAK_HOT_FAIL_DB:
        r.fails.append(
            f"peak {r.peak_db:.1f} dBFS - sitting on the ceiling. "
            f"Target is about {PEAK_TARGET_DB:.0f}.")
    elif r.peak_db >= PEAK_HOT_WARN_DB:
        r.warns.append(
            f"peak {r.peak_db:.1f} dBFS leaves mastering no headroom; "
            f"target is about {PEAK_TARGET_DB:.0f}.")
    elif r.peak_db <= PEAK_QUIET_WARN_DB:
        r.warns.append(
            f"peak {r.peak_db:.1f} dBFS is very quiet - mastering has to lift the "
            f"room with the voice. Target is about {PEAK_TARGET_DB:.0f}.")

    # --- noise ------------------------------------------------------------
    r.noise_floor_db, r.snr_db = noise_floor_and_snr(x, sr)
    if r.snr_db is None:
        r.warns.append(
            "no measurable room tone - the gaps are digital silence. Usually "
            "noise reduction or a gate was applied on the way in; record flat.")
    elif r.snr_db < SNR_FAIL_DB:
        r.fails.append(
            f"SNR {r.snr_db:.1f} dB (floor {r.noise_floor_db:.1f} dBFS). Hiss is "
            f"masking consonants, which is where Whisper starts guessing - and it "
            f"spends far longer doing it. Get closer to the mic or quieten the room.")
    elif r.snr_db < SNR_WARN_DB:
        r.warns.append(
            f"SNR {r.snr_db:.1f} dB is usable but noisy "
            f"(floor {r.noise_floor_db:.1f} dBFS).")

    # --- length -----------------------------------------------------------
    if r.duration < MIN_DURATION_S:
        r.fails.append(
            f"only {r.duration:.1f}s long - that is a misfire, not a section.")
    elif expected_words:
        r.implied_wpm = round(expected_words / (r.duration / 60.0), 1)
        lo_f, hi_f = WPM_FAIL
        lo_w, hi_w = WPM_WARN
        if not (lo_f <= r.implied_wpm <= hi_f):
            r.fails.append(
                f"{r.duration:.0f}s for {expected_words} words is "
                f"{r.implied_wpm:.0f} wpm. That is not a pace, that is the wrong "
                f"file or a truncated one - check the section number.")
        elif not (lo_w <= r.implied_wpm <= hi_w):
            r.warns.append(
                f"{r.implied_wpm:.0f} wpm ({r.duration:.0f}s for {expected_words} "
                f"words). Possible, but check you read the whole section.")

    if sr != TARGET_SR:
        r.warns.append(
            f"{sr} Hz - the notebook resamples to {TARGET_SR}, so this works, but "
            f"recording at {TARGET_SR} everywhere avoids the extra pass.")

    r.ok = not r.fails
    return r


def check_run(files: dict[int, Path], expected: dict[int, int] | None = None,
              missing: list[int] | None = None) -> dict:
    """Measure every section, then the things that only exist ACROSS sections."""
    expected = expected or {}
    reports = [check_file(p, sec, expected.get(sec)) for sec, p in sorted(files.items())]

    run_fails: list[str] = []
    run_warns: list[str] = []

    if missing:
        run_fails.append(
            "no recording for section(s) " + ", ".join(str(s) for s in missing) +
            ". The notebook refuses to start Step 17 without every one of them.")

    good = [r for r in reports if r.sr]
    rates = {r.sr for r in good}
    if len(rates) > 1:
        run_warns.append(
            "sections were recorded at different sample rates (" +
            ", ".join(f"{s} Hz" for s in sorted(rates)) +
            "). The notebook resamples, so this is survivable, but it usually "
            "means the mic or its settings changed mid-script.")
    chans = {r.channels for r in good}
    if len(chans) > 1:
        run_warns.append(
            "sections have different channel counts (" +
            ", ".join(str(c) for c in sorted(chans)) + ").")

    # A section recorded at a noticeably different level than its neighbours is
    # audible as a jump: Step 18 masters the concatenation as one piece, so it
    # will not level the sections against each other.
    peaks = [r.peak_db for r in good if r.peak_db > -100]
    if len(peaks) > 1 and (max(peaks) - min(peaks)) > 12.0:
        run_warns.append(
            f"section levels span {max(peaks) - min(peaks):.0f} dB "
            f"({min(peaks):.1f} to {max(peaks):.1f} dBFS peak). Step 18 masters "
            f"the whole thing as one piece and will not even them out, so the "
            f"loud sections will stay loud.")

    return {
        "sections": reports,
        "run_fails": run_fails,
        "run_warns": run_warns,
        "ok": not run_fails and all(r.ok for r in reports),
    }


def to_json(result: dict) -> str:
    return json.dumps({
        "ok": result["ok"],
        "run_fails": result["run_fails"],
        "run_warns": result["run_warns"],
        "sections": [
            {k: v for k, v in vars(r).items()} | {"verdict": r.verdict}
            for r in result["sections"]
        ],
    }, indent=2)
