"""Timing repair: correct performed pauses, and mark scripts with a niche's rhythm.

This is stage 3 of the pipeline - repair - which sits between take selection and
mastering, and was the missing link. It runs BEFORE any step that locks audio
length for picture sync, because everything here changes duration by design.

Two entry points:

    fix_pauses(wav, targets)   adjust the gaps in a recording, silently
    mark_script(text, niche)   write a script marked with a niche's pause rhythm

`fix_pauses` reports nothing by design. It fixes what it can and leaves what it
cannot - see `gaps.MIN_REAL_GAP` for the one hard limit, which is a property of
what post-production can honestly do rather than a policy choice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import gaps as _gaps
from . import marking, roomtone


@dataclass
class FixResult:
    """What happened. Written to a log file, never printed unless asked."""

    path: str
    duration_before: float
    duration_after: float
    gaps_found: int = 0
    adjusted: int = 0
    skipped_too_short: int = 0
    tone_found: bool = True
    note: str | None = None
    details: list[dict] = field(default_factory=list)

    @property
    def delta(self) -> float:
        return self.duration_after - self.duration_before


def fix_pauses(wav: str | Path, script_marked: str | None = None,
               niche: str | None = None, out: str | Path | None = None,
               in_place: bool = False) -> FixResult:
    """Set every performed pause to its target length.

    Targets come from a marked script when one is supplied, matched to the gaps
    IN ORDER. When the counts disagree - a sentence run together, an unplanned
    breath - the ordered match would silently pair the wrong pause with the
    wrong target and quietly wreck the phrasing, so it falls back to the niche
    rhythm instead, which needs no alignment.
    """
    import soundfile as sf

    wav = Path(wav)
    x, sr = sf.read(str(wav), always_2d=False, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1).astype(np.float32)
    duration = len(x) / sr

    res = FixResult(path=str(wav), duration_before=duration,
                    duration_after=duration)

    found = _gaps.interior(_gaps.find(x, sr), duration)
    res.gaps_found = len(found)
    if not found:
        res.note = "no interior pauses found"
        return res

    try:
        tone = roomtone.extract(x, sr)
    except roomtone.NoRoomTone as e:
        # Without room tone a gap can only be SHORTENED, never lengthened.
        # Doing half the job silently is right here; inventing silence is not.
        res.tone_found = False
        res.note = str(e)
        tone = np.zeros(0, dtype=np.float32)

    style = marking.style_for(niche)
    wanted = marking.targets(script_marked) if script_marked else []

    fixable = [g for g in found if g.fixable]
    if wanted and len(wanted) == len(fixable):
        pairs = list(zip(fixable, wanted))
    else:
        if wanted:
            res.note = ((res.note + "; ") if res.note else "") + (
                f"script has {len(wanted)} targets but {len(fixable)} pauses "
                f"were performed - using {niche or 'default'} rhythm instead")
        pairs = [(g, _target_for(g, style)) for g in fixable]

    if not res.tone_found:
        pairs = [(g, t) for g, t in pairs if t < g.duration]

    res.details = [{"at": round(g.start, 2), "was": round(g.duration, 2),
                    "now": round(t, 2)} for g, t in pairs]
    res.skipped_too_short = len(found) - len(fixable)

    y, adjusted, skipped = _gaps.apply(x, sr, pairs, tone)
    res.adjusted = adjusted
    res.skipped_too_short += skipped
    res.duration_after = len(y) / sr

    target = Path(out) if out else (wav if in_place
                                    else wav.with_name(wav.stem + "_timed.wav"))
    sf.write(str(target), y, sr, subtype="PCM_16")
    res.path = str(target)
    return res


def _target_for(gap: _gaps.Gap, style: marking.PauseStyle) -> float:
    """Which kind of pause was this, judged only by how long it was held.

    Without a transcript there is no punctuation to consult, so length is the
    only evidence of intent - and it is decent evidence: a speaker holding 0.9 s
    meant a fuller stop than one holding 0.3 s. The classification is therefore
    a nudge toward the nearest kind of pause in this style, never a rewrite of
    what was performed.
    """
    candidates = (style.clause, style.sentence, style.paragraph)
    return min(candidates, key=lambda c: abs(c - gap.duration))


def mark_script(text: str, niche: str | None = None) -> tuple[str, marking.PauseStyle]:
    style = marking.style_for(niche)
    return marking.render(text, style), style
