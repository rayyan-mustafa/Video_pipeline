"""Find the pauses in a take, and change how long they are.

NO ASR HERE. Gaps are found from energy alone, which matters on the hardware
this runs on: transcribing a 2 s clause takes ~4 s on the target machine, so a
tool that needed a transcript to adjust timing would take longer than the
recording. Energy-based detection runs in well under a second on a whole
section.

THE FLOOR IS THE WHOLE DESIGN. A pause shorter than `MIN_REAL_GAP` is not
stretched, ever. When a speaker runs two sentences together, the first one's
intonation never closed - it still carries the rising contour of someone about
to continue. Padding room tone in behind that produces the exact "this was
edited" sound the room tone was supposed to avoid. The gap has to have been
performed before it can be corrected.

Correction, not creation. That is the line.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .roomtone import _crossfade

FRAME_SEC = 0.02

# Below this, a gap was not a pause - it was the space between two words, or a
# breath taken mid-thought. Stretching it fabricates a beat the speaker never
# performed. This number is the honest limit of what post-correction can do.
MIN_REAL_GAP = 0.25


@dataclass
class Gap:
    """One silence, in seconds from the start of the file."""

    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def fixable(self) -> bool:
        return self.duration >= MIN_REAL_GAP


def find(x: np.ndarray, sr: int, gate_db: float = 10.0,
         min_gap: float = 0.12) -> list[Gap]:
    """Silences, using a noise floor measured from this recording.

    The gate is relative to a rolling percentile rather than a fixed dBFS, for
    the same reason the live clause detector is: a fixed threshold is correct
    in exactly one room at one gain setting.
    """
    n = max(1, int(FRAME_SEC * sr))
    usable = (len(x) // n) * n
    if usable == 0:
        return []
    f = x[:usable].astype(np.float64).reshape(-1, n)
    db = 20.0 * np.log10(np.sqrt((f ** 2).mean(axis=1)) + 1e-12)

    # THE THRESHOLD ADAPTS TO THE DYNAMIC RANGE, not just to the floor.
    #
    # A fixed percentile for the floor breaks on the exact material this tool
    # is for: measured narration with generous pauses can be 25-30% silence, so
    # a 20th-percentile "floor" lands inside SPEECH, the gate opens on
    # everything, and the whole file reads as one enormous pause. Taking the
    # floor from the 10th percentile and the speech level from the 90th, then
    # placing the gate a fraction of the way up that span, degrades sensibly
    # whether the take is 5% silence or 40%.
    floor = float(np.percentile(db, 10))
    speech = float(np.percentile(db, 90))
    span = max(speech - floor, 1e-6)
    thr = floor + min(gate_db, max(6.0, span * 0.4))
    quiet = db < thr

    gaps: list[Gap] = []
    i = 0
    while i < len(quiet):
        if not quiet[i]:
            i += 1
            continue
        j = i
        while j < len(quiet) and quiet[j]:
            j += 1
        start, end = i * FRAME_SEC, j * FRAME_SEC
        if end - start >= min_gap:
            gaps.append(Gap(start=start, end=end))
        i = j
    return gaps


def interior(gaps: list[Gap], duration: float, edge: float = 0.35) -> list[Gap]:
    """Drop the silence at the very start and end of the file.

    Leading and trailing silence is head and tail, not a pause between two
    thoughts. Stretching the tail of a section would push the next section
    later and desync everything downstream.
    """
    return [g for g in gaps
            if g.start > edge and g.end < duration - edge]


# ---------------------------------------------------------------------------
#  Rewriting
# ---------------------------------------------------------------------------
def apply(x: np.ndarray, sr: int, targets: list[tuple[Gap, float]],
          tone: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Rebuild the audio with each listed gap set to its target duration.

    Returns (audio, adjusted, skipped).

    Works back to front so that earlier gap positions stay valid as later ones
    change length - editing forwards would shift every subsequent index by the
    amount already inserted, which is the classic way this kind of routine
    produces audio that drifts further out of place the longer the file runs.
    """
    from .roomtone import fill

    out = x.astype(np.float32).copy()
    adjusted = skipped = 0
    xfade = max(1, int(0.008 * sr))

    for gap, target in sorted(targets, key=lambda t: t[0].start, reverse=True):
        if not gap.fixable:
            skipped += 1
            continue
        delta = target - gap.duration
        if abs(delta) < 0.05:            # already close enough to hear as right
            continue

        a = int(gap.start * sr)
        b = int(gap.end * sr)
        if b <= a or b > len(out):
            skipped += 1
            continue

        if delta > 0:
            # Lengthen: splice room tone into the MIDDLE of the existing gap,
            # so the speaker's own decay into silence and their breath back in
            # are both preserved at the edges.
            mid = (a + b) // 2
            pad = fill(tone, int(delta * sr), sr)
            left = _crossfade(out[:mid], pad, xfade)
            out = _crossfade(left, out[mid:], xfade)
        else:
            # Shorten: remove from the middle, for the same reason.
            cut = min(int(-delta * sr), (b - a) - int(MIN_REAL_GAP * sr))
            if cut <= 0:
                skipped += 1
                continue
            mid = (a + b) // 2
            half = cut // 2
            out = _crossfade(out[:mid - half], out[mid - half + cut:], xfade)
        adjusted += 1

    return out.astype(np.float32), adjusted, skipped
