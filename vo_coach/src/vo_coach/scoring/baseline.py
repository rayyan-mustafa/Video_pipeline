"""Personal baseline: what this speaker sounds like when they are reading well.

WHY THIS EXISTS. The rubric's absolute numbers came from US-English VO norms -
140-170 wpm, a 0.55 ASR confidence floor for "probably mispronounced". The user
of this tool is a Pakistani English speaker. For a speaker those norms were not
written for, "outside the band" and "wrong" are not the same statement, and
without a baseline the coach cannot tell them apart: it would flag a clear,
intelligible read as too slow, and flag Whisper's own uncertainty about an
accent as a mispronunciation. That second one is the rubric penalising how
someone sounds, which the spec forbids outright.

WHAT IT DOES. Aggregate several of the user's own takes into a robust
description of their natural range, then blend that with the absolute rubric at
`baseline.weight`. Blend, not replace: a pure personal baseline would normalise
away real faults ("you always mumble, therefore mumbling is fine"), and a pure
rubric is the problem above. Start at a low weight and raise it as takes
accumulate.

WHAT IT IS NOT. Not a grade, and not a target to hit. Nothing here is compared
against another speaker.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import numpy as np

from ..config import PROJECT_ROOT, Config, load_config
from ..schemas import AnalysisResult, Baseline, BaselineStat

# A calibration take has to be long enough for the medians to mean anything.
# Below this the numbers are describing one sentence, not a speaker.
MIN_DURATION_SEC = 20.0
MIN_WORDS = 40


class CalibrationError(RuntimeError):
    """A take is not usable as calibration input. Deliberately fatal rather than
    a warning: a baseline silently built from a faulty recording bakes that
    fault in as "normal", and every take afterwards is judged against it."""


def _stat(values: list[float]) -> BaselineStat | None:
    """Median and MAD.

    Median rather than mean, MAD rather than standard deviation: with three to
    five takes a single odd read drags a mean anywhere, and the point of a
    baseline is that it describes the usual case, not the average case.
    """
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    med = float(np.median(vals))
    mad = float(np.median([abs(v - med) for v in vals]))
    return BaselineStat(median=round(med, 3), mad=round(mad, 3), n=len(vals))


def check_suitable(r: AnalysisResult, cfg: Config | None = None) -> None:
    """Reject takes that describe a bad recording rather than a natural voice.

    Calibration is supposed to capture how the speaker sounds when the read is
    going well. A clipped or hissy take describes the microphone that day.
    """
    cfg = cfg or load_config()
    if r.duration_sec < MIN_DURATION_SEC:
        raise CalibrationError(
            r.take_id + " is only " + format(r.duration_sec, ".1f") + "s; "
            "calibration needs at least " + str(int(MIN_DURATION_SEC)) + "s of "
            "continuous reading.")
    if len(r.words) < MIN_WORDS:
        raise CalibrationError(
            r.take_id + " has " + str(len(r.words)) + " words; at least "
            + str(MIN_WORDS) + " are needed for the medians to mean anything.")
    if r.mouth_clicks:
        raise CalibrationError(
            r.take_id + " contains clipping. A clipped take describes a gain "
            "mistake, not a voice - re-record it before calibrating.")
    must_fix_snr = cfg.get("noise.must_fix_snr_db")
    if r.snr_db is not None and r.snr_db < must_fix_snr:
        raise CalibrationError(
            r.take_id + " measures " + format(r.snr_db, ".0f") + " dB SNR, below "
            "the must-fix floor of " + format(must_fix_snr, ".0f") + " dB. "
            "Calibrating on it would make a noisy room the norm.")


def build(analyses: list[AnalysisResult], cfg: Config | None = None) -> Baseline:
    """Aggregate calibration takes into a baseline. Every take is checked first."""
    cfg = cfg or load_config()
    if not analyses:
        raise CalibrationError("No takes given.")
    for r in analyses:
        check_suitable(r, cfg)

    versions = {r.analyzer_version for r in analyses}
    if len(versions) > 1:
        raise CalibrationError(
            "These takes were analyzed by different analyzer versions "
            + repr(sorted(v for v in versions if v is not None)) + ". Re-run "
            "`analyze` on all of them so the numbers are comparable.")

    def per_take(fn) -> list[float]:
        out = []
        for r in analyses:
            v = fn(r)
            if v is not None:
                out.append(v)
        return out

    def _median_confidence(r: AnalysisResult) -> float | None:
        # Median within the take first, then across takes: one take with three
        # times as many words must not dominate the speaker's baseline.
        if not r.words:
            return None
        return float(np.median([w.confidence for w in r.words]))

    def _tilt(r: AnalysisResult) -> float | None:
        pts = r.proximity_effect.spectral_tilt_db_per_octave
        return float(np.mean([p.value for p in pts])) if pts else None

    def _pct(r: AnalysisResult, q: float) -> float | None:
        pts = r.pacing.wpm_windowed
        return float(np.percentile([p.wpm for p in pts], q)) if pts else None

    return Baseline(
        created_at=_dt.datetime.now().isoformat(timespec="seconds"),
        take_ids=[r.take_id for r in analyses],
        analyzer_version=next(iter(versions)),
        wpm=_stat(per_take(lambda r: r.pacing.wpm_overall)),
        wpm_p10=_stat(per_take(lambda r: _pct(r, 10))),
        wpm_p90=_stat(per_take(lambda r: _pct(r, 90))),
        word_confidence=_stat(per_take(_median_confidence)),
        pitch_mean_hz=_stat(per_take(lambda r: r.pitch.mean_hz)),
        pitch_range_hz=_stat(per_take(lambda r: r.pitch.range_hz)),
        spectral_tilt_mean=_stat(per_take(_tilt)),
        sibilance_per_min=_stat(per_take(
            lambda r: len(r.sibilance.peaks) / max(r.duration_sec / 60.0, 1e-6))),
    )


# ---------------------------------------------------------------------------
#  Persistence
# ---------------------------------------------------------------------------
def path_for(cfg: Config | None = None) -> Path:
    cfg = cfg or load_config()
    p = Path(cfg.get("baseline.path", "data/baseline.json"))
    return p if p.is_absolute() else PROJECT_ROOT / p


def save(b: Baseline, cfg: Config | None = None) -> Path:
    p = path_for(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(b.model_dump_json(indent=2), encoding="utf-8")
    return p


def load(cfg: Config | None = None, require_enabled: bool = True) -> Baseline | None:
    """None when calibration is off or has never been run. Every caller treats
    that as "use the absolute rubric", so the tool works before calibration and
    improves after it."""
    cfg = cfg or load_config()
    if require_enabled and not cfg.get("baseline.enabled", False):
        return None
    p = path_for(cfg)
    if not p.exists():
        return None
    try:
        return Baseline.model_validate_json(p.read_text(encoding="utf-8"))
    except Exception:
        return None


# ---------------------------------------------------------------------------
#  Applying the baseline
# ---------------------------------------------------------------------------
def _weight(cfg: Config) -> float:
    return max(0.0, min(1.0, float(cfg.get("baseline.weight", 0.3))))


def wpm_band(cfg: Config | None = None,
             band: tuple[float, float] | None = None,
             b: Baseline | None = None) -> tuple[float, float] | None:
    """Pull the pacing band toward the speaker's own comfortable range.

    `band` is whatever the caller would otherwise use - a niche band derived
    from the reference, or None for the global config band. The personal end of
    the blend is the speaker's own p10/p90 windowed rate, i.e. the range they
    actually live in rather than a single average.

    Returns the band unchanged when there is no baseline, so this is safe to
    call unconditionally.
    """
    cfg = cfg or load_config()
    b = b if b is not None else load(cfg)
    if b is None or b.wpm_p10 is None or b.wpm_p90 is None:
        return band

    lo, hi = band or (float(cfg.get("pacing.target_wpm_min")),
                      float(cfg.get("pacing.target_wpm_max")))
    w = _weight(cfg)
    plo, phi = b.wpm_p10.median, b.wpm_p90.median
    blended = (round((1 - w) * lo + w * plo, 1), round((1 - w) * hi + w * phi, 1))
    # A blend can invert if the personal range sits entirely on one side of the
    # rubric band. Order defensively rather than returning a band nothing fits.
    return (min(blended), max(blended))


def min_word_confidence(cfg: Config | None = None,
                        b: Baseline | None = None) -> float:
    """The mispronunciation floor, moved to sit under this speaker's normal.

    THE SHARPEST EDGE IN THE WHOLE RUBRIC. Whisper reports lower word confidence
    on accents further from its training distribution. A fixed 0.55 floor
    therefore fires on the ACCENT, not on the pronunciation - the model's
    uncertainty about how someone sounds, dressed up as a fault in how they
    said it.

    The fix is to define the floor relative to the speaker's own median
    confidence rather than absolutely. `median - 2*MAD` marks words that are
    unusual FOR THIS SPEAKER, which is the thing actually worth flagging. The
    config value stays the ceiling: calibration can only loosen this threshold,
    never tighten it past the absolute rubric.
    """
    cfg = cfg or load_config()
    absolute = float(cfg.get("pronunciation.min_word_confidence"))
    b = b if b is not None else load(cfg)
    if b is None or b.word_confidence is None:
        return absolute

    wc = b.word_confidence
    # 2*MAD is roughly a 3-sigma tail for normal data; with one take MAD is 0,
    # so fall back to a flat 15% margin below the median.
    spread = 2.0 * wc.mad if wc.mad > 0 else 0.15 * wc.median
    personal = wc.median - spread
    w = _weight(cfg)
    blended = (1 - w) * absolute + w * personal
    return round(min(absolute, max(0.0, blended)), 3)


def render(b: Baseline | None) -> str:
    if b is None:
        return ("No baseline. Record 3 natural takes, analyze them, then:\n"
                "  python -m vo_coach calibrate <take_id> <take_id> <take_id>")
    rows = [
        ("speaking rate", b.wpm, "wpm"),
        ("  slow end (p10)", b.wpm_p10, "wpm"),
        ("  fast end (p90)", b.wpm_p90, "wpm"),
        ("ASR word confidence", b.word_confidence, ""),
        ("pitch mean", b.pitch_mean_hz, "Hz"),
        ("pitch range", b.pitch_range_hz, "Hz"),
        ("spectral tilt", b.spectral_tilt_mean, "dB/oct"),
        ("sibilants", b.sibilance_per_min, "/min"),
    ]
    out = ["baseline from " + str(len(b.take_ids)) + " take(s), built "
           + b.created_at, ""]
    for label, stat, unit in rows:
        if stat is None:
            continue
        out.append("  " + label.ljust(22) + format(stat.median, "8.2f") + " "
                   + unit.ljust(7) + " (+/- " + format(stat.mad, ".2f") + ")")
    return "\n".join(out)
