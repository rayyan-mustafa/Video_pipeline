"""Diff a take against its niche's reference voice.

Per-metric deltas, never a single score. A composite number would let a good
pacing result cancel out a proximity problem, which is exactly the information
the coach needs to keep separate.

Not every metric is comparable between two speakers. The ones that are not are
reported in `excluded` with a reason, so a missing row never reads as "we
checked that and it was fine".
"""

from __future__ import annotations

from ..config import Config, load_config
from ..schemas import (
    AnalysisResult,
    BenchmarkResult,
    FeatureVector,
    MetricDelta,
    ReferenceFeatures,
    Severity,
)
from .features import extract

# metric -> (tolerance, must_fix_at, higher_than_reference_is_worse)
#
# Tolerances are wide on purpose. The goal is "good for a US audience", not
# "indistinguishable from this specific creator" - a take should be able to
# differ substantially in style without being told it is wrong.
COMPARISONS: dict[str, tuple[float, float, str]] = {
    "wpm_overall":        (20.0, 45.0, "either"),
    "pitch_range_hz":     (40.0, 90.0, "lower"),
    "spectral_tilt_mean": (2.0, 4.0, "lower"),
    "sibilance_per_min":  (10.0, 25.0, "higher"),
    "pause_rate_per_min": (8.0, 18.0, "either"),
    "mean_pause_sec":     (0.25, 0.6, "either"),
}

_LABEL = {
    "wpm_overall": "speaking rate",
    "pitch_range_hz": "pitch range",
    "spectral_tilt_mean": "spectral tilt",
    "sibilance_per_min": "sibilants/min",
    "pause_rate_per_min": "pauses/min",
    "mean_pause_sec": "mean pause",
}


def _severity(metric: str, delta: float, tol: float, hard: float,
              worse_when: str) -> Severity:
    """Direction matters. A pitch range WIDER than the reference is not a
    problem; narrower might be. Treating |delta| as the whole story would flag
    a more expressive read as a fault."""
    if worse_when == "either":
        mag = abs(delta)
    elif worse_when == "higher":
        mag = max(delta, 0.0)
    else:                                   # "lower"
        mag = max(-delta, 0.0)

    if mag >= hard:
        return "must_fix"
    return "stylistic" if mag >= tol else "ok"


# Metrics computed entirely from ASR word timestamps. When those timestamps are
# not trustworthy for a language, these are meaningless rather than merely noisy.
TIMING_METRICS = {"wpm_overall", "pause_rate_per_min", "mean_pause_sec"}


def benchmark(take: AnalysisResult | FeatureVector, ref: ReferenceFeatures,
              cfg: Config | None = None) -> BenchmarkResult:
    cfg = cfg or load_config()
    feats = take if isinstance(take, FeatureVector) else extract(take)
    take_id = "unknown" if isinstance(take, FeatureVector) else take.take_id

    excluded = dict(cfg.section("excluded_from_benchmark"))
    deltas: list[MetricDelta] = []

    # Drop every timing metric when this niche's word timestamps are unreliable.
    # Reporting a WPM delta built on 1.0 s "words" would be worse than reporting
    # nothing: it looks like a real measurement and would drive real coaching.
    niches = cfg.section("niches")
    nc = niches.get(ref.niche or "", {})
    if not nc.get("asr_word_timestamps_reliable", True):
        for m in TIMING_METRICS:
            excluded[m] = (
                f"word timestamps are unreliable for {ref.language!r} on model "
                f"{nc.get('asr_model')!r} - the same clip yields 33-94 words and "
                f"34-89 wpm across identical runs (English is bit-identical), so "
                f"every timing metric is meaningless rather than merely noisy"
            )

    for metric, (tol, hard, worse_when) in COMPARISONS.items():
        if metric in excluded:
            continue
        tv = getattr(feats, metric, None)
        rv = getattr(ref.features, metric, None)
        if tv is None or rv is None:
            excluded[metric] = "not measurable on the take or the reference"
            continue

        delta = float(tv) - float(rv)
        sev = _severity(metric, delta, tol, hard, worse_when)

        note = None
        if metric == "spectral_tilt_mean" and ref.provenance == "youtube_rip":
            # Lossy codecs discard high-frequency content, which tilts the
            # reference darker than the source actually was.
            note = ("reference is a lossy rip - its high end is codec-limited, "
                    "so a darker reference tilt is partly compression, not mic "
                    "technique")
        if metric == "wpm_overall" and ref.language != "en":
            note = (f"reference is {ref.language}; words-per-minute is not "
                    f"directly comparable across languages")

        deltas.append(MetricDelta(
            metric=metric,
            take_value=round(float(tv), 3),
            reference_value=round(float(rv), 3),
            delta=round(delta, 3),
            severity=sev,
            note=note,
        ))

    order = {"must_fix": 0, "stylistic": 1, "ok": 2}
    deltas.sort(key=lambda d: order[d.severity])

    return BenchmarkResult(take_id=take_id, reference_name=ref.name,
                           niche=ref.niche, deltas=deltas, excluded=excluded)


def render(b: BenchmarkResult) -> str:
    """Human-readable table for the CLI."""
    mark = {"must_fix": "!!", "stylistic": " ~", "ok": " ."}
    lines = [f"  vs {b.reference_name}  (niche: {b.niche})", ""]
    lines.append(f"  {'':2} {'metric':<18} {'take':>9} {'ref':>9} {'delta':>9}")
    for d in b.deltas:
        lines.append(f"  {mark[d.severity]} {_LABEL.get(d.metric, d.metric):<18} "
                     f"{d.take_value:>9.1f} {d.reference_value:>9.1f} "
                     f"{d.delta:>+9.1f}")
        if d.note:
            lines.append(f"       -> {d.note}")
    if b.excluded:
        lines.append("")
        lines.append("  not compared:")
        for k, why in b.excluded.items():
            lines.append(f"    {k}: {' '.join(why.split())}")
    return "\n".join(lines)
