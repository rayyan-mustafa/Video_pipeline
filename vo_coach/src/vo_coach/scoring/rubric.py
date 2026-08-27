"""The scoring rubric: must-fix vs stylistic vs silent.

Two thresholds per metric drive everything:

    below `silence`      -> say nothing at all
    between the two      -> logged, shown on the HUD, never spoken
    above `must_fix`     -> always flagged

The width of that gap is the coach's personality, which is why both live in
config rather than here. This module only decides WHICH BAND a measurement
falls into and writes the sentence that goes with it.

The spec's stylistic list is enforced as a hard rule, not a preference: accent
colouring, mild pitch narrowness and small proximity variation can never be
promoted to must_fix regardless of magnitude.
"""

from __future__ import annotations

from ..config import Config, load_config
from ..schemas import AnalysisResult, Finding, ScoreResult, Severity

# Metrics that are stylistic BY DEFINITION. Nothing in this module may return
# must_fix for them - the coach is not in the business of normalizing voices.
NEVER_MUST_FIX = {"pitch_range", "pitch_mean", "accent"}


def _band(value: float, silence: float, must_fix: float,
          higher_is_worse: bool = True) -> Severity:
    """Place a measurement into ok / stylistic / must_fix."""
    if higher_is_worse:
        if value >= must_fix:
            return "must_fix"
        return "stylistic" if value >= silence else "ok"
    if value <= must_fix:
        return "must_fix"
    return "stylistic" if value <= silence else "ok"


def _pacing_findings(r: AnalysisResult, cfg: Config,
                     wpm_band: tuple[float, float] | None = None) -> list[Finding]:
    """Pacing is judged on SUSTAINED drift, not on any single fast sentence.

    A one-off quick line is human and reads as energy. The spec's rule is
    "outside the band for more than N consecutive sentences", so a take that
    darts in and out of range never trips this - only a genuinely rushed or
    dragging passage does.
    """
    from ..analysis.pacing import sentence_wpm

    lo, hi = wpm_band or (cfg.get("pacing.target_wpm_min"),
                          cfg.get("pacing.target_wpm_max"))
    silence_m = cfg.get("pacing.silence_wpm_margin")
    mustfix_m = cfg.get("pacing.must_fix_wpm_margin")
    need = cfg.get("pacing.sustained_sentences_before_flag")

    findings: list[Finding] = []
    sentences = sentence_wpm(r.words) if r.words else []
    if not sentences:
        return findings

    # Longest run of consecutive sentences outside the band, in each direction.
    for direction, cmp in (("fast", lambda w: w > hi), ("slow", lambda w: w < lo)):
        run = best = 0
        worst = 0.0
        for w in sentences:
            if cmp(w):
                run += 1
                best = max(best, run)
                worst = max(worst, abs(w - (hi if direction == "fast" else lo)))
            else:
                run = 0
        if best < need:
            continue

        sev = _band(worst, silence_m, mustfix_m, higher_is_worse=True)
        if sev == "ok":
            continue
        findings.append(Finding(
            metric="pacing",
            severity=sev,
            message=(
                f"{best} consecutive sentences run {direction} "
                f"({worst:.0f} wpm outside the {lo:.0f}-{hi:.0f} band). "
                + ("Let the ends of sentences land before starting the next."
                   if direction == "fast"
                   else "Tighten the gaps between phrases; the read is dragging.")
            ),
        ))
    return findings


def _noise_findings(r: AnalysisResult, cfg: Config) -> list[Finding]:
    if r.snr_db is None:
        # Honest silence: the SNR estimator refused to guess, so neither do we.
        return []
    sev = _band(r.snr_db,
                cfg.get("noise.silence_snr_db"),
                cfg.get("noise.must_fix_snr_db"),
                higher_is_worse=False)
    if sev == "ok":
        return []
    return [Finding(
        metric="noise",
        severity=sev,
        message=(f"Signal-to-noise is {r.snr_db:.0f} dB. Hiss is starting to sit "
                 f"on top of consonants - move closer, raise gain at the mic "
                 f"rather than in software, or treat the room."),
    )]


def _sibilance_findings(r: AnalysisResult, cfg: Config) -> list[Finding]:
    if not r.sibilance.peaks:
        return []
    minutes = max(r.duration_sec / 60.0, 1e-6)
    rate = len(r.sibilance.peaks) / minutes

    # Rate, not raw count: a longer take must not score worse for being longer.
    # These are deliberately generous - a de-esser fixes mild sibilance in
    # mastering, so only genuinely distracting rates are worth a delivery note.
    sev = _band(rate, 12.0, 30.0, higher_is_worse=True)
    if sev == "ok":
        return []
    worst = max(r.sibilance.peaks, key=lambda p: p.band_energy_db)
    return [Finding(
        metric="sibilance",
        severity=sev,
        message=(f"{rate:.0f} sharp sibilants per minute (worst at "
                 f"{worst.t:.1f}s). Angle the mic slightly off-axis; the "
                 f"de-esser will handle what is left."),
        t=worst.t,
    )]


def _proximity_findings(r: AnalysisResult, cfg: Config) -> list[Finding]:
    tilt = r.proximity_effect.spectral_tilt_db_per_octave
    if not tilt:
        return []
    mean = sum(p.value for p in tilt) / len(tilt)
    sev = _band(mean,
                cfg.get("proximity_effect.silence_tilt"),
                cfg.get("proximity_effect.must_fix_tilt"),
                higher_is_worse=False)
    if sev == "ok":
        return []
    return [Finding(
        metric="proximity_effect",
        severity=sev,
        message=(f"Spectral tilt averages {mean:.1f} dB/octave - heavy low-end "
                 f"buildup from working the mic too close. Back off a few "
                 f"centimetres; a high-pass filter alone will thin the voice."),
    )]


def _pause_findings(r: AnalysisResult, cfg: Config) -> list[Finding]:
    """Only meaningful with a script. Without one, `aligned_to_punctuation` is
    False everywhere by construction, which would fabricate a finding on every
    take."""
    if not r.pauses or not any(p.aligned_to_punctuation for p in r.pauses):
        return []
    limit = cfg.get("pauses.unaligned_pause_sec")
    bad = [p for p in r.pauses if not p.aligned_to_punctuation and p.duration >= limit]
    if not bad:
        return []
    ratio = len(bad) / len(r.pauses)
    sev: Severity = "must_fix" if ratio > 0.35 else "stylistic"
    worst = max(bad, key=lambda p: p.duration)
    return [Finding(
        metric="pauses",
        severity=sev,
        message=(f"{len(bad)} pauses land mid-clause rather than on punctuation "
                 f"(longest {worst.duration:.1f}s at {worst.t:.1f}s). That reads "
                 f"as hesitation. Breathe at the commas, not between them."),
        t=worst.t,
    )]


def _clipping_findings(r: AnalysisResult) -> list[Finding]:
    if not r.mouth_clicks:
        return []
    return [Finding(
        metric="clipping",
        severity="must_fix",
        message=(f"{len(r.mouth_clicks)} clipped run(s) - the signal hit full "
                 f"scale and the waveform is flat-topped. No amount of "
                 f"mastering recovers this; re-record with lower gain."),
        t=r.mouth_clicks[0].t,
    )]


def _pitch_findings(r: AnalysisResult, cfg: Config) -> list[Finding]:
    """Always stylistic. Listed so the HUD can show it, never escalated."""
    floor = cfg.get("pitch.flag_range_hz_below")
    if r.pitch.range_hz >= floor or r.pitch.range_hz <= 0:
        return []
    return [Finding(
        metric="pitch_range",
        severity="stylistic",
        message=(f"Pitch range is {r.pitch.range_hz:.0f} Hz - a fairly flat "
                 f"delivery. Not a fault, but landing key words a little "
                 f"higher adds shape."),
    )]


def score(r: AnalysisResult, cfg: Config | None = None,
          wpm_band: tuple[float, float] | None = None,
          timings_reliable: bool = True) -> ScoreResult:
    """Full rubric pass over one take.

    `timings_reliable=False` suppresses every finding derived from ASR word
    timestamps. Some languages segment into multi-word chunks rather than words
    on the smaller Whisper checkpoints, which makes pacing and pause metrics
    meaningless rather than merely noisy - and a confident "you are speaking too
    slowly" built on that would be actively misleading. The signal metrics never
    touch ASR and still run.
    """
    cfg = cfg or load_config()

    findings: list[Finding] = []
    findings += _clipping_findings(r)
    findings += _noise_findings(r, cfg)
    findings += _proximity_findings(r, cfg)
    findings += _sibilance_findings(r, cfg)
    findings += _pitch_findings(r, cfg)
    if timings_reliable:
        findings += _pacing_findings(r, cfg, wpm_band=wpm_band)
        findings += _pause_findings(r, cfg)

    # Hard guarantee, not a convention: stylistic metrics can never be
    # promoted, no matter what a threshold change upstream might imply.
    for f in findings:
        if f.metric in NEVER_MUST_FIX and f.severity == "must_fix":
            f.severity = "stylistic"

    order = {"must_fix": 0, "stylistic": 1, "ok": 2}
    findings.sort(key=lambda f: order[f.severity])
    return ScoreResult(take_id=r.take_id, findings=findings)
