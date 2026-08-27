"""Rank multiple takes of the same line and surface the best one.

The ranking explains itself. An auto-picked winner the user cannot interrogate
is not trustworthy - if the tool says take 3 beat take 1, the reason has to be
inspectable, because the user heard both and may disagree.

ORDERING RULES, most significant first:

  1. FEWER MUST-FIX FINDINGS. A take with clipping or masking hiss loses to one
     without, regardless of how well it was performed. These are the faults
     mastering cannot repair.
  2. FEWER STYLISTIC FINDINGS.
  3. BETTER SNR. Among takes that are otherwise equal, the quieter recording
     gives the mastering chain more room to work.

Delivery quality - whether the read actually lands - is NOT scored here. No
metric in this pipeline measures it, and inventing a proxy would produce a
confident ranking of something never measured. The tool narrows the field; the
final call between close takes stays with the person who heard them.
"""

from __future__ import annotations

from ..config import Config, load_config
from ..schemas import AnalysisResult, RankingResult, TakeRanking
from ..scoring.rubric import score


def _reason(must_fix: int, stylistic: int, snr: float | None, rank: int) -> str:
    if rank == 1:
        if must_fix == 0 and stylistic == 0:
            return "clean - nothing flagged"
        if must_fix == 0:
            return f"no must-fix issues ({stylistic} stylistic note(s))"
        return f"fewest must-fix issues ({must_fix})"
    bits = []
    if must_fix:
        bits.append(f"{must_fix} must-fix")
    if stylistic:
        bits.append(f"{stylistic} stylistic")
    if snr is not None:
        bits.append(f"SNR {snr:.0f} dB")
    return ", ".join(bits) if bits else "no issues flagged"


def rank(
    analyses: list[AnalysisResult],
    cfg: Config | None = None,
    wpm_band: tuple[float, float] | None = None,
    timings_reliable: bool = True,
) -> RankingResult:
    cfg = cfg or load_config()

    rows = []
    for a in analyses:
        s = score(a, cfg=cfg, wpm_band=wpm_band, timings_reliable=timings_reliable)
        must = sum(1 for f in s.findings if f.severity == "must_fix")
        styl = sum(1 for f in s.findings if f.severity == "stylistic")
        rows.append({
            "take_id": a.take_id,
            "must": must,
            "styl": styl,
            # An unmeasurable SNR must not win the tie-break by sorting as
            # "infinitely good"; -inf pushes it to the back of that key only.
            "snr": a.snr_db,
            "wpm": a.pacing.wpm_overall or None,
        })

    rows.sort(key=lambda r: (r["must"], r["styl"],
                             -(r["snr"] if r["snr"] is not None else -1e9)))

    return RankingResult(candidates=[
        TakeRanking(
            take_id=r["take_id"], rank=i + 1,
            must_fix_count=r["must"], stylistic_count=r["styl"],
            snr_db=r["snr"], wpm=r["wpm"],
            reason=_reason(r["must"], r["styl"], r["snr"], i + 1),
        )
        for i, r in enumerate(rows)
    ])


def render(result: RankingResult) -> str:
    lines = [f"  {'':4} {'take':<24} {'must':>5} {'styl':>5} {'SNR':>7}  why"]
    for c in result.candidates:
        mark = "->" if c.rank == 1 else "  "
        snr = f"{c.snr_db:.1f}" if c.snr_db is not None else "n/a"
        lines.append(f"  {mark}{c.rank:<2} {c.take_id:<24} {c.must_fix_count:>5} "
                     f"{c.stylistic_count:>5} {snr:>7}  {c.reason}")
    lines.append("")
    lines.append("  Delivery quality is not scored - nothing here measures whether")
    lines.append("  the read lands. Between close takes, trust your ears.")
    return "\n".join(lines)
