"""Mechanical checks that the mentor obeyed its own system prompt.

THE PROBLEM THIS SOLVES. The system prompt states six hard rules. Nothing in a
prompt makes a model follow them, and a cheap model is exactly where obedience
degrades first. Without this module the only way to know whether
Ling-3.0-flash is good enough is to read its coaching and form an impression -
which is not a decision procedure, and is worse still when the reader is being
coached on their own delivery and has no outside reference for what good
coaching sounds like.

WHAT IT DOES. Every rule that can be checked against the reply and the payload
it was given is checked here. The result is a per-call violation list and, via
`log()`, an accumulating failure RATE per model. That rate is the switch-models
signal: a model whose hard-violation rate is climbing is not doing the job,
regardless of how good any single reply reads.

WHAT IT DELIBERATELY DOES NOT DO. It does not judge whether the coaching is
good. "Was this the most useful of the three findings to raise" is a judgment
call and belongs to a golden set or an LLM judge, not to a regex. Everything
here is a rule with a yes/no answer.

hard vs soft:
  hard - a stated rule, checkable without judgment. Counts toward the rate.
  soft - a heuristic on tone or shape. Reported, never counted, because these
         can fire on a reply that is actually fine.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
from pathlib import Path

from ..config import DATA_DIR
from ..schemas import ValidationReport, Violation

AUDIT_LOG = DATA_DIR / "mentor_audit.jsonl"

# ---------------------------------------------------------------------------
#  Rule 5: never invent a measurement
# ---------------------------------------------------------------------------
#  Only unit-bearing numbers are checked. A bare "3" is usually a list index or
#  a count of fixes; a bare number with "dB" after it is a claim about the
#  recording, and if it is not in the payload the model made it up.
_UNIT = r"(?:dBFS|dBTP|dB|LUFS|kHz|Hz|wpm|words per minute|%|ms|seconds|secs?\b|s\b)"
_MEASUREMENT = re.compile(r"(-?\d+(?:\.\d+)?)\s*(" + _UNIT + r")", re.IGNORECASE)
_ANY_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def _numbers_in(payload: str) -> list[float]:
    return [float(m) for m in _ANY_NUMBER.findall(payload)]


def _is_supported(value: float, known: list[float]) -> bool:
    """Rounding is not invention.

    The payload says 14.4 dB; the model writes "14 dB". That is good coaching,
    not a fabricated number, so the tolerance is generous - half a unit, or 2%
    on larger values where the model is more likely to round to a whole number
    (188.1 wpm -> "188", or even "190").
    """
    for k in known:
        if abs(k - value) <= max(0.5, abs(k) * 0.02):
            return True
    return False


def _check_invented_numbers(text: str, payload: str) -> list[Violation]:
    known = _numbers_in(payload)
    out: list[Violation] = []
    seen: set[str] = set()
    for m in _MEASUREMENT.finditer(text):
        raw = m.group(0)
        if raw.lower() in seen:
            continue
        seen.add(raw.lower())
        if not _is_supported(float(m.group(1)), known):
            out.append(Violation(
                rule="invented_number",
                severity="hard",
                detail=repr(raw) + " does not appear in the data the model was "
                       "given (rule 5: never invent a measurement).",
                excerpt=_around(text, m.start()),
            ))
    return out


# ---------------------------------------------------------------------------
#  Rule 4: never tell the speaker to change their voice
# ---------------------------------------------------------------------------
#  This is the rule that matters most to a non-US speaker: it is the difference
#  between a coach and an accent-reduction service. The patterns look for
#  INSTRUCTIONS to change identity, not for the words themselves - a reply may
#  legitimately say "your accent is not the problem here".
_IDENTITY = [
    (r"\b(neutrali[sz]e|soften|reduce|lose|work on|correct)\b[^.]{0,40}\baccent\b",
     "instructs the speaker to change their accent"),
    (r"\bsound more\b[^.]{0,30}\b(american|native|neutral|british)\b",
     "instructs the speaker to sound like a different kind of speaker"),
    (r"\b(raise|lower|deepen|change|adjust)\b[^.]{0,25}\b(your\s+)?(natural\s+)?"
     r"(pitch|voice|timbre)\b",
     "instructs the speaker to change pitch or timbre, which is identity"),
    (r"\baccent\s+(is|was)\s+(a\s+)?(problem|issue|distracting|hard\s+to)",
     "treats the accent itself as the fault"),
]


def _check_identity(text: str) -> list[Violation]:
    out: list[Violation] = []
    for pat, why in _IDENTITY:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            out.append(Violation(
                rule="identity_coaching", severity="hard",
                detail="Rule 4: " + why + ".",
                excerpt=_around(text, m.start())))
    return out


# ---------------------------------------------------------------------------
#  Rule 5 again: metrics that were never measured are unknown, not fine
# ---------------------------------------------------------------------------
_TOPIC_WORDS = {
    "pacing": r"\b(pace|pacing|speaking rate|wpm|words per minute|too fast|"
              r"too slow|rushed|dragging)\b",
    "pauses": r"\b(pause|pauses|pausing|breaths|breathing)\b",
    "snr_db": r"\b(snr|signal[- ]to[- ]noise|hiss|noise floor)\b",
    "noise_floor_db": r"\b(noise floor|room tone|hiss)\b",
    "pitch_mean_hz": r"\b(pitch)\b",
}


def mentions(text: str, topic: str) -> bool:
    """Public: does this reply talk about `topic` at all?

    The eval harness asserts on the same patterns the validators use. Two
    separate word lists would drift, and then a case could pass the eval while
    failing the validator on the same text.
    """
    pat = TOPIC_PATTERNS.get(topic)
    if pat is None:
        raise KeyError(
            repr(topic) + " is not a known topic. Known: "
            + ", ".join(sorted(TOPIC_PATTERNS)))
    return re.search(pat, text, re.IGNORECASE) is not None


def _check_forbidden_topics(text: str, payload: dict) -> list[Violation]:
    """Two separate sources of "you do not have this number":

      timing_metrics_unavailable - ASR word timings were not trustworthy, so
        pacing and pause metrics were never computed at all.
      benchmark.not_measured     - deliberately excluded from BENCHMARKING (a
        lossy reference's noise floor measures the codec, not a target).

    The second is weaker: a metric excluded from the benchmark may still have
    been measured on the take itself, so only the first is a hard violation.
    """
    out: list[Violation] = []
    if "timing_metrics_unavailable" in payload:
        for topic in ("pacing", "pauses"):
            m = re.search(_TOPIC_WORDS[topic], text, re.IGNORECASE)
            if m:
                out.append(Violation(
                    rule="commented_on_unmeasured", severity="hard",
                    detail="Payload declared timing metrics unavailable, but the "
                           "reply comments on " + topic + ".",
                    excerpt=_around(text, m.start())))
    for metric in (payload.get("benchmark") or {}).get("not_measured", []):
        pat = _TOPIC_WORDS.get(metric)
        m = re.search(pat, text, re.IGNORECASE) if pat else None
        if m:
            out.append(Violation(
                rule="compared_excluded_metric", severity="soft",
                detail=metric + " was excluded from benchmarking; check the reply "
                       "is not comparing it against the reference.",
                excerpt=_around(text, m.start())))
    return out


# ---------------------------------------------------------------------------
#  Rule 1: at most three fixes.  And: say nothing if nothing is wrong.
# ---------------------------------------------------------------------------
#  A fix is a paragraph opening with a bolded name, per the prompt's OUTPUT
#  SHAPE. Bullet and numbered prefixes are tolerated because models add them
#  unprompted, and a numbered list of four fixes is still four fixes.
_FIX_HEADING = re.compile(r"^\s*(?:[-*]\s*|\d+[.)]\s*)?\*\*(.+?)\*\*",
                          re.MULTILINE)


def _fix_headings(text: str) -> list[str]:
    return [m.group(1).strip() for m in _FIX_HEADING.finditer(text)]


def _check_shape(text: str, payload: dict, max_fixes: int) -> list[Violation]:
    out: list[Violation] = []
    heads = _fix_headings(text)
    if len(heads) > max_fixes:
        out.append(Violation(
            rule="too_many_fixes", severity="hard",
            detail=str(len(heads)) + " fixes, limit is " + str(max_fixes)
                   + " (rule 1).",
            excerpt="; ".join(heads)))

    if not payload.get("findings") and heads:
        out.append(Violation(
            rule="invented_findings", severity="hard",
            detail="The rubric flagged nothing, but the reply raises fixes. The "
                   "prompt says not to go looking for something to say.",
            excerpt="; ".join(heads)))

    # Rule 3: a fix has to say what the LISTENER experiences, not just restate a
    # measurement. Soft, because there are many ways to say it.
    if heads and not re.search(
            r"\b(listener|listeners|audience|hear|hears|hearing|follow|"
            r"following|ear)\b", text, re.IGNORECASE):
        out.append(Violation(
            rule="no_listener_rationale", severity="soft",
            detail="Rule 3: no fix explains what the listener experiences."))

    if re.match(r"\s*(great|nice|good|excellent|awesome|well done|fantastic)\b",
                text, re.IGNORECASE):
        out.append(Violation(
            rule="praise_preamble", severity="soft",
            detail="Opens with praise; the prompt asks for an honest one-line read.",
            excerpt=text[:60]))
    return out


# ---------------------------------------------------------------------------
#  "Do not promote a stylistic finding into a fault"
# ---------------------------------------------------------------------------
_MUST_LANGUAGE = re.compile(
    r"\b(must|need to|have to|has to|critical|serious|unacceptable|re-?record|"
    r"fix this|before you (publish|upload))\b", re.IGNORECASE)

_METRIC_WORDS = {
    "pitch_range": r"\b(pitch range|flat delivery|monotone|inflection)\b",
    "sibilance": r"\b(sibilan|sharp s)\w*",
    "proximity_effect": r"\b(proximity|low[- ]end|boom|bass buildup)\b",
    "clipping": r"\b(clip|clipped|clipping|distort\w*|full scale|flat-?top\w*)\b",
    "pacing": _TOPIC_WORDS["pacing"],
    "pauses": _TOPIC_WORDS["pauses"],
    "noise": _TOPIC_WORDS["snr_db"],
}

# Everything nameable as a subject the coaching could raise. `mentions()` and
# the eval harness both read this, so the two can never disagree about what
# "the reply talked about pacing" means.
TOPIC_PATTERNS = {**_TOPIC_WORDS, **_METRIC_WORDS,
                  "accent": r"\baccent\b"}


def _check_promotion(text: str, payload: dict) -> list[Violation]:
    """A metric the rubric called stylistic, discussed in must-fix language.

    Soft on purpose: this reads tone, and tone is where a regex is least
    reliable. It is here because promotion is the specific failure mode a weaker
    reasoner is most likely to have, so it is worth surfacing even imperfectly.
    """
    sev_by_metric: dict[str, set[str]] = {}
    for f in payload.get("findings", []):
        sev_by_metric.setdefault(f.get("metric", "?"), set()).add(f.get("severity"))

    out: list[Violation] = []
    for metric, sevs in sev_by_metric.items():
        if "must_fix" in sevs or metric not in _METRIC_WORDS:
            continue
        for para in re.split(r"\n\s*\n", text):
            if not re.search(_METRIC_WORDS[metric], para, re.IGNORECASE):
                continue
            m = _MUST_LANGUAGE.search(para)
            if m:
                out.append(Violation(
                    rule="promoted_stylistic", severity="soft",
                    detail=metric + " was scored stylistic but is discussed as a "
                           "fault (" + repr(m.group(0)) + ").",
                    excerpt=para[:160]))
                break
    return out


def _around(text: str, i: int, width: int = 70) -> str:
    return text[max(0, i - width // 2): i + width].replace("\n", " ").strip()


# ---------------------------------------------------------------------------
#  Entry point
# ---------------------------------------------------------------------------
def validate(text: str, payload: str, max_fixes: int = 3,
             take_id: str = "?", model: str = "?",
             provider: str | None = None) -> ValidationReport:
    """Check one mentor reply against the payload it was given."""
    try:
        parsed = json.loads(payload)
    except Exception:
        parsed = {}

    violations: list[Violation] = []
    violations += _check_shape(text, parsed, max_fixes)
    violations += _check_invented_numbers(text, payload)
    violations += _check_identity(text)
    violations += _check_forbidden_topics(text, parsed)
    violations += _check_promotion(text, parsed)

    return ValidationReport(
        take_id=take_id, model=model, provider=provider,
        violations=violations,
        checked_at=_dt.datetime.now().isoformat(timespec="seconds"),
    )


# ---------------------------------------------------------------------------
#  The audit log - one line per real API call
# ---------------------------------------------------------------------------
def log(report: ValidationReport, path: Path | None = None) -> None:
    """Append-only. Cached replies are NOT logged: counting the same reply twice
    would move the failure rate without any new evidence behind it."""
    p = path or AUDIT_LOG
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "checked_at": report.checked_at,
            "take_id": report.take_id,
            "model": report.model,
            "provider": report.provider,
            "n_hard": len(report.hard),
            "n_soft": len(report.violations) - len(report.hard),
            "rules": [v.rule for v in report.violations],
        }, ensure_ascii=False) + "\n")


def rates(path: Path | None = None) -> dict[str, dict]:
    """Hard-violation rate per model - the switch-models signal.

    Read this, not any single reply. One bad reply is noise; a model breaking a
    stated rule on a third of its calls is a model that cannot do this job.
    """
    p = path or AUDIT_LOG
    if not p.exists():
        return {}
    out: dict[str, dict] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except Exception:
            continue
        m = out.setdefault(rec.get("model", "?"),
                           {"calls": 0, "with_hard": 0, "hard": 0, "soft": 0,
                            "by_rule": {}})
        m["calls"] += 1
        m["hard"] += rec.get("n_hard", 0)
        m["soft"] += rec.get("n_soft", 0)
        if rec.get("n_hard", 0):
            m["with_hard"] += 1
        for r in rec.get("rules", []):
            m["by_rule"][r] = m["by_rule"].get(r, 0) + 1
    for m in out.values():
        m["hard_rate"] = round(m["with_hard"] / m["calls"], 3) if m["calls"] else 0.0
    return out
