"""The mentor's system prompt and payload rendering.

Two halves, deliberately separated:

  SYSTEM_PROMPT  is byte-stable across every call, so it can sit behind a cache
                 breakpoint. Nothing take-specific goes in here.
  build_payload  renders the take-specific JSON that goes in the user turn.

That split is what makes prompt caching work: caching is a prefix match, so a
single varying byte in the system prompt would invalidate the cache on every
request. See shared prompt-caching guidance.
"""

from __future__ import annotations

import json

from ..schemas import AnalysisResult, BenchmarkResult, ScoreResult

# ---------------------------------------------------------------------------
#  System prompt - FROZEN. Do not interpolate anything into this string.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a voice coach for a solo voiceover creator. You are \
given the output of an automated analysis of one take and you respond the way a \
good coach in the room would: briefly, specifically, and only when it matters.

WHAT YOU ARE OPTIMIZING FOR
The target is intelligibility and trust with a US listener. It is NOT a \
studio-perfect read, and it is NOT sounding like the reference voice. A take \
that is clear, warm and easy to follow has succeeded even if it is imperfect. \
Say so when that is the case.

THE RUBRIC HAS ALREADY RUN
You are handed findings that are already classified:
  must_fix    - crossed a threshold that hurts intelligibility or trust
  stylistic   - noticed, but a matter of taste; the coach's own judgment
Do not re-derive these classifications and do not promote a stylistic finding \
into a fault. If the analysis flagged nothing, do not go looking for something \
to say.

HARD RULES
1. At most THREE fixes. If there are more findings than that, choose the three \
that most affect the listener and ignore the rest. A list of eight notes is not \
coaching, it is a report.
2. Order by impact on the listener, not by how easy each is to fix.
3. For every fix, say WHY it matters to someone listening. "You are at 14 dB \
SNR" is a measurement. "There is enough hiss that consonants are starting to \
blur, so listeners work harder to follow you" is coaching.
4. Never tell the speaker to change their voice. Pitch, accent and natural \
timbre are identity, not delivery. Range and expression are fair game.
5. Never invent a measurement. If a number is not in the data you were given, \
you do not have it. Some metrics are deliberately excluded or unmeasurable - \
treat those as unknown, not as fine.
6. Benchmark deltas are context, not targets. The reference is one creator who \
works in this niche, not a standard to match. A take can differ substantially \
and still be good.

TONE
Direct and warm. Second person. No preamble, no "great job!", no numbered \
scoring. Do not restate the metrics back as a summary - the user can already \
see them. Two or three sentences per fix is plenty.

OUTPUT SHAPE
Open with one sentence on how the take reads overall - honest, not inflated. \
Then the fixes, each as a short paragraph starting with a bolded name for the \
issue. If there is nothing worth fixing, say that plainly in one or two \
sentences and stop.
"""


def _findings_payload(score: ScoreResult, max_fixes: int) -> list[dict]:
    """must_fix first, then stylistic, capped a little above `max_fixes`.

    A small amount of headroom is deliberate: the model chooses the final three,
    and it needs a couple of alternatives to choose between. Sending everything
    would invite a wall of notes; sending exactly three removes the choice.
    """
    order = {"must_fix": 0, "stylistic": 1, "ok": 2}
    ranked = sorted(score.findings, key=lambda f: order.get(f.severity, 9))
    return [
        {"metric": f.metric, "severity": f.severity, "detail": f.message,
         **({"at_seconds": round(f.t, 1)} if f.t is not None else {})}
        for f in ranked[: max_fixes + 2]
    ]


def _benchmark_payload(b: BenchmarkResult | None) -> dict | None:
    if b is None:
        return None
    return {
        "reference": b.reference_name,
        "niche": b.niche,
        "deltas": [
            {"metric": d.metric, "take": d.take_value, "reference": d.reference_value,
             "delta": d.delta, "severity": d.severity,
             **({"caveat": d.note} if d.note else {})}
            for d in b.deltas if d.severity != "ok"
        ],
        # Passed through so the model can distinguish "measured and fine" from
        # "never measured". Rule 5 depends on this being visible.
        "not_measured": sorted(b.excluded),
    }


def _history_payload(history: list[dict] | None) -> list[dict] | None:
    """Prior takes, oldest first. Lets the mentor notice a trend - something
    fixed, or something recurring across sessions - which is the main thing a
    single-take analysis cannot see."""
    if not history:
        return None
    return history[-5:]


def build_payload(
    analysis: AnalysisResult,
    score: ScoreResult,
    benchmark: BenchmarkResult | None = None,
    history: list[dict] | None = None,
    max_fixes: int = 3,
    timings_reliable: bool = True,
) -> str:
    """The user-turn content: everything take-specific, as JSON."""
    payload: dict = {
        "take": {
            "id": analysis.take_id,
            "duration_sec": round(analysis.duration_sec, 1),
            "words": len(analysis.words),
            "wpm": analysis.pacing.wpm_overall or None,
        },
        "findings": _findings_payload(score, max_fixes),
        "max_fixes": max_fixes,
    }

    if analysis.transcript:
        # Trimmed: the mentor coaches delivery, not writing. Enough to know what
        # was being said, not the whole script.
        t = analysis.transcript
        payload["transcript_excerpt"] = t[:600] + ("..." if len(t) > 600 else "")

    if not timings_reliable:
        payload["timing_metrics_unavailable"] = (
            "Word timestamps are unreliable for this language/model, so pacing "
            "and pause metrics were not computed. Do not comment on speaking "
            "rate or pausing - you have no data on them."
        )

    if (bm := _benchmark_payload(benchmark)) is not None:
        payload["benchmark"] = bm
    if (h := _history_payload(history)) is not None:
        payload["previous_takes"] = h

    return json.dumps(payload, indent=2, ensure_ascii=False)
