"""Assemble the approved takes into one clean file.

WHY THIS EXISTS. The moment lines can be retaken, the recording stops being the
deliverable. It contains three attempts at line two, a cough, and the good
version buried in the middle. Something has to pick the keepers out, and doing
it by hand in a DAW is the work this project is supposed to remove.

WHAT IT KEEPS. The LAST passing attempt at each line. Last rather than best:
"best" would need a quality score to rank attempts on, and the honest ranking
signal - whether the coach objected - has already been applied to decide that
the attempt passed at all. Between two passing takes, the later one is the one
the speaker meant, because they stopped there.

THE RAW FILE IS NEVER TOUCHED. This writes a second file. A comp built from a
misfiring rubric would otherwise destroy a good read with no way back, and the
rubric is threshold-driven on a machine whose thresholds are still being tuned.

THE GAPS ARE THE SCRIPT'S. Because the marked script already carries a pause
target for every line, the assembled file gets those pauses exactly - not
whatever silence happened to follow the accepted attempt. That makes this the
same operation as `fix-pauses`, done at assembly time instead of afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class CompResult:
    path: str | None = None
    lines_used: int = 0
    lines_missing: int = 0
    attempts_dropped: int = 0
    duration: float = 0.0
    note: str | None = None
    missing: list[int] = field(default_factory=list)


def build(wav: str | Path, lines: list[dict], out: str | Path | None = None,
          pad: float = 0.12) -> CompResult:
    """Write a file containing only the approved attempt at each line.

    `pad` keeps a little audio either side of each accepted clause: consonants
    at the edges of a phrase are quiet, and cutting exactly on the detected
    boundary clips them. It is small enough not to drag in the neighbouring
    attempt.
    """
    import soundfile as sf

    from ..timing import roomtone

    wav = Path(wav)
    x, sr = sf.read(str(wav), always_2d=False, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1).astype(np.float32)

    res = CompResult()
    if not lines:
        res.note = "no script, so there is nothing to assemble"
        return res

    try:
        tone = roomtone.extract(x, sr)
    except roomtone.NoRoomTone:
        # No room tone means the gaps between lines have to come from somewhere
        # else. Silence is not acceptable, so fall back to the quietest audio
        # available rather than fabricating a hole in the noise floor.
        tone = np.zeros(0, dtype=np.float32)
        res.note = "no room tone found; using minimal gaps"

    pieces: list[np.ndarray] = []
    for line in lines:
        good = [a for a in line.get("attempts", []) if a.get("passed")]
        if not good:
            res.lines_missing += 1
            res.missing.append(line.get("i", -1))
            continue
        keep = good[-1]
        res.attempts_dropped += len(line.get("attempts", [])) - 1

        a = max(0, int((keep["t_start"] - pad) * sr))
        b = min(len(x), int((keep["t_end"] + pad) * sr))
        if b <= a:
            res.lines_missing += 1
            res.missing.append(line.get("i", -1))
            continue
        pieces.append(x[a:b])
        res.lines_used += 1

        gap = float(line.get("pause") or 0.0)
        if gap > 0 and len(tone):
            pieces.append(roomtone.fill(tone, int(gap * sr), sr))

    if not pieces:
        res.note = "no line was ever accepted, so nothing was assembled"
        return res

    y = _join(pieces, sr)
    target = Path(out) if out else wav.with_name(wav.stem + "_comped.wav")
    sf.write(str(target), y, sr, subtype="PCM_16")
    res.path = str(target)
    res.duration = len(y) / sr
    return res


def _join(pieces: list[np.ndarray], sr: int) -> np.ndarray:
    """Concatenate with short equal-power crossfades.

    Butt-joining two pieces recorded seconds apart puts a step in the waveform
    wherever the noise happened to be at a different point in its cycle, and
    that step is an audible tick at every single edit.
    """
    from ..timing.roomtone import _crossfade

    n = max(1, int(0.006 * sr))
    out = pieces[0].astype(np.float32)
    for piece in pieces[1:]:
        out = _crossfade(out, piece.astype(np.float32), n)
    return out.astype(np.float32)
