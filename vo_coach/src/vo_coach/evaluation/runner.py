"""Run the golden set against a model and score it.

The score is a pass rate over cases, where a case passes only if the reply
breaks no hard rule AND meets that case's explicit expectations. Both halves
matter: a reply can be perfectly rule-compliant and still fail to mention the
one thing that was actually wrong with the take.

Results are stored per model so two models can be compared without re-running
the first one.
"""

from __future__ import annotations

import datetime as _dt
import json
from dataclasses import dataclass, field
from pathlib import Path

from ..config import DATA_DIR, Config, load_config
from ..mentor import validate as _v
from ..mentor.client import MentorError, OfflineCacheMiss, coach
from ..schemas import ValidationReport
from ..scoring.rubric import score as run_rubric
from .cases import CASES, Case

EVAL_DIR = DATA_DIR / "eval"


@dataclass
class CaseResult:
    case_id: str
    passed: bool
    failures: list[str] = field(default_factory=list)
    text: str = ""
    report: ValidationReport | None = None
    cached: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def status(self) -> str:
        return "pass" if self.passed else "FAIL"


@dataclass
class EvalRun:
    model: str
    provider: str | None
    ran_at: str
    results: list[CaseResult] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def rate(self) -> float:
        return self.passed / len(self.results) if self.results else 0.0

    @property
    def tokens(self) -> tuple[int, int, int]:
        return (sum(r.input_tokens for r in self.results),
                sum(r.output_tokens for r in self.results),
                sum(r.reasoning_tokens for r in self.results))


# ---------------------------------------------------------------------------
#  Checking one reply
# ---------------------------------------------------------------------------
def check(case: Case, text: str, report: ValidationReport) -> list[str]:
    """Everything wrong with this reply, as plain sentences."""
    failures = [v.rule + ": " + v.detail for v in report.hard]

    heads = _v._fix_headings(text)
    e = case.expect
    if e.max_fixes is not None and len(heads) > e.max_fixes:
        failures.append("raised " + str(len(heads)) + " fixes, expected at most "
                        + str(e.max_fixes))
    if e.min_fixes is not None and len(heads) < e.min_fixes:
        failures.append("raised " + str(len(heads)) + " fixes, expected at least "
                        + str(e.min_fixes))

    for topic in e.topics_required:
        if not _v.mentions(text, topic):
            failures.append("never mentions " + topic + ", which is the thing "
                            "actually wrong with this take")
    for topic in e.topics_forbidden:
        if _v.mentions(text, topic):
            failures.append("raises " + topic + ", which this case forbids")
    return failures


def check_rubric(case: Case, cfg: Config | None = None) -> list[str]:
    """Verify the case still produces the findings it was written around.

    Runs before any API call. If a threshold change emptied the payload, the
    reply would be judged against a take that no longer resembles the case, and
    the resulting score would be meaningless - so this fails loudly instead.
    """
    scored = run_rubric(case.analysis, cfg=cfg,
                        timings_reliable=case.timings_reliable)
    actual = {(f.metric, f.severity) for f in scored.findings}
    expected = set(case.expect.expect_rubric)
    out = []
    for miss in sorted(expected - actual):
        out.append("rubric no longer produces " + miss[0] + "/" + miss[1])
    for extra in sorted(actual - expected):
        out.append("rubric now also produces " + extra[0] + "/" + extra[1])
    return out


# ---------------------------------------------------------------------------
#  Running
# ---------------------------------------------------------------------------
def run_case(case: Case, cfg: Config | None = None,
             force: bool = False) -> CaseResult:
    cfg = cfg or load_config()

    drift = check_rubric(case, cfg)
    if drift:
        return CaseResult(case_id=case.id, passed=False,
                          failures=["case is stale - " + d for d in drift])

    scored = run_rubric(case.analysis, cfg=cfg,
                        timings_reliable=case.timings_reliable)
    text, meta = coach(case.analysis, scored, benchmark=case.benchmark,
                       cfg=cfg, timings_reliable=case.timings_reliable,
                       force=force)

    report = meta.get("validation")
    if report is None:                    # validation disabled in config
        report = _v.validate(text, "{}", take_id=case.analysis.take_id)

    failures = check(case, text, report)
    return CaseResult(
        case_id=case.id, passed=not failures, failures=failures, text=text,
        report=report, cached=bool(meta.get("cached")),
        input_tokens=meta.get("input_tokens", 0) or 0,
        output_tokens=meta.get("output_tokens", 0) or 0,
        reasoning_tokens=meta.get("reasoning_tokens", 0) or 0,
    )


def run_all(cases: list[Case] | None = None, cfg: Config | None = None,
            force: bool = False, on_case=None) -> EvalRun:
    cfg = cfg or load_config()
    run = EvalRun(model=cfg.get("mentor.model"),
                  provider=cfg.get("mentor.provider", None),
                  ran_at=_dt.datetime.now().isoformat(timespec="seconds"))
    for case in (cases if cases is not None else CASES):
        try:
            r = run_case(case, cfg=cfg, force=force)
        except OfflineCacheMiss:
            r = CaseResult(case_id=case.id, passed=False,
                           failures=["no cached reply and mentor.offline is true"])
        except MentorError as e:
            r = CaseResult(case_id=case.id, passed=False,
                           failures=["mentor failed: " + str(e)])
        run.results.append(r)
        if on_case:
            on_case(r)
    return run


# ---------------------------------------------------------------------------
#  Storage and reporting
# ---------------------------------------------------------------------------
def _slug(model: str) -> str:
    return model.replace("/", "__").replace(":", "_")


def save(run: EvalRun) -> Path:
    """One file per model, overwritten. The point is the latest score for each
    model side by side, not a history of every run."""
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    p = EVAL_DIR / (_slug(run.model) + ".json")
    p.write_text(json.dumps({
        "model": run.model, "provider": run.provider, "ran_at": run.ran_at,
        "passed": run.passed, "total": len(run.results), "rate": run.rate,
        "results": [{"case_id": r.case_id, "passed": r.passed,
                     "failures": r.failures, "text": r.text}
                    for r in run.results],
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def stored() -> list[dict]:
    if not EVAL_DIR.exists():
        return []
    out = []
    for p in sorted(EVAL_DIR.glob("*.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    return sorted(out, key=lambda d: -d.get("rate", 0))


def render(run: EvalRun, verbose: bool = False) -> str:
    lines = [run.model + ("  via " + run.provider if run.provider else ""), ""]
    for r in run.results:
        lines.append("  [" + r.status + "] " + r.case_id
                     + ("  (cached)" if r.cached else ""))
        for f in r.failures:
            lines.append("         - " + f)
        if verbose and r.text:
            for ln in r.text.splitlines():
                lines.append("         | " + ln)
    tin, tout, tthink = run.tokens
    lines += ["", "  " + str(run.passed) + "/" + str(len(run.results))
              + " passed  (" + format(run.rate, ".0%") + ")"]
    if tin or tout:
        lines.append("  " + str(tin) + " in / " + str(tout) + " out"
                     + (" (" + str(tthink) + " thinking)" if tthink else "")
                     + " tokens")
    return "\n".join(lines)


def render_comparison(rows: list[dict]) -> str:
    if not rows:
        return ("No stored eval runs. Run:  python -m vo_coach eval\n"
                "then switch mentor.model in config/thresholds.yaml and run it "
                "again - the two scores are the comparison.")
    out = ["stored eval runs, best first:", ""]
    for d in rows:
        out.append("  " + format(d.get("rate", 0), ">4.0%") + "  "
                   + str(d.get("passed", 0)) + "/" + str(d.get("total", 0))
                   + "  " + str(d.get("model")) + "   " + str(d.get("ran_at")))
        failed = [r["case_id"] for r in d.get("results", []) if not r["passed"]]
        if failed:
            out.append("        failed: " + ", ".join(failed))
    out += ["", "A model that fails `clean_take` produces coaching on takes with "
            "nothing wrong;\na model that fails `accent_bait` coaches identity. "
            "Weigh those two above\nthe headline rate."]
    return "\n".join(out)
