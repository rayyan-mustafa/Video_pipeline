"""Pacing and pauses.

Two things live here that the rubric leans on heavily:

  WPM  - both overall and windowed. Overall WPM alone hides the actual problem:
         a read that sprints through one paragraph and crawls through the next
         averages out to a perfectly normal number.

  PAUSES - and crucially whether each one lands on punctuation. A pause at a
         comma is phrasing; the same pause mid-clause reads as hesitation.
         Distinguishing them is the whole value of this module.
"""

from __future__ import annotations

import re

from ..schemas import Pacing, Pause, Word, WpmPoint

# Speaking-rate convention: a "word" is a whitespace token. This matches how
# published WPM norms (the 140-170 band) are counted, so the numbers stay
# comparable to the target band rather than being internally consistent only.

PUNCT_END = tuple(".!?")
PUNCT_MID = tuple(",;:-")


def wpm_overall(words: list[Word], speaking_time: float | None = None) -> float:
    """Words per minute.

    By default this is measured over SPEAKING time (first word start to last
    word end), not wall-clock file duration - leading and trailing room tone
    would otherwise make every take look slower than it read.
    """
    if len(words) < 2:
        return 0.0
    span = speaking_time if speaking_time is not None else (words[-1].end - words[0].start)
    if span <= 0:
        return 0.0
    return round(len(words) / span * 60.0, 2)


def wpm_windowed(words: list[Word], win_sec: float = 5.0,
                 hop_sec: float = 1.0) -> list[WpmPoint]:
    """Rolling WPM.

    A word counts toward a window when its MIDPOINT falls inside it. Using the
    start time instead biases every window early by roughly half a word, which
    is exactly the kind of quiet systematic error that never surfaces as a
    crash and quietly shifts the whole curve.
    """
    if not words:
        return []

    t_end = words[-1].end
    mids = [(w.start + w.end) / 2.0 for w in words]

    points: list[WpmPoint] = []
    t = words[0].start
    while t + win_sec <= max(t_end, words[0].start + win_sec):
        n = sum(1 for m in mids if t <= m < t + win_sec)
        points.append(WpmPoint(t=round(t + win_sec / 2, 3),
                               wpm=round(n / win_sec * 60.0, 2)))
        t += hop_sec

    return points


def _script_punctuation_positions(script: str) -> list[tuple[str, str]]:
    """(normalized_word, trailing_punctuation) for each script token."""
    out: list[tuple[str, str]] = []
    for token in script.split():
        stripped = token.rstrip("".join(PUNCT_END + PUNCT_MID + ('"', "'", ")")))
        trailing = token[len(stripped):]
        norm = re.sub(r"[^a-z0-9']", "", stripped.lower())
        if norm:
            out.append((norm, trailing))
    return out


def _norm(word: str) -> str:
    return re.sub(r"[^a-z0-9']", "", word.lower())


def find_pauses(words: list[Word], script: str | None = None,
                min_pause: float = 0.25) -> list[Pause]:
    """Gaps between consecutive words, tagged by whether punctuation precedes them.

    When a script is supplied, alignment is done by walking the ASR words
    against script tokens - NOT by index. ASR insertions and deletions are
    common enough that index-based pairing drifts and mislabels punctuation
    from that point on.

    Without a script, `aligned_to_punctuation` is False everywhere, since there
    is nothing to align to. That is honest rather than guessed: ASR punctuation
    is itself model output and would make the check circular.
    """
    if len(words) < 2:
        return []

    script_tokens = _script_punctuation_positions(script) if script else []
    cursor = 0

    pauses: list[Pause] = []
    for prev, nxt in zip(words, words[1:]):
        gap = nxt.start - prev.end
        if gap < min_pause:
            # still advance the alignment cursor for this word
            if script_tokens:
                cursor = _advance(script_tokens, cursor, prev.word)
            continue

        aligned = False
        if script_tokens:
            cursor = _advance(script_tokens, cursor, prev.word)
            if 0 <= cursor < len(script_tokens):
                aligned = bool(script_tokens[cursor][1])

        pauses.append(Pause(t=round(prev.end, 3), duration=round(gap, 3),
                            aligned_to_punctuation=aligned))

    return pauses


def _advance(script_tokens: list[tuple[str, str]], cursor: int,
             asr_word: str, lookahead: int = 4) -> int:
    """Move the script cursor to the token matching `asr_word`.

    Searches a small window forward so a single ASR insertion or deletion does
    not desynchronize the rest of the take. Returns the matched index; if no
    match is found nearby, advances by one and keeps going rather than giving up.
    """
    target = _norm(asr_word)
    if not target:
        return cursor
    for i in range(cursor, min(cursor + lookahead + 1, len(script_tokens))):
        if script_tokens[i][0] == target:
            return i
    return min(cursor + 1, len(script_tokens))


def analyze_pacing(words: list[Word], win_sec: float = 5.0) -> Pacing:
    return Pacing(
        wpm_overall=wpm_overall(words),
        wpm_windowed=wpm_windowed(words, win_sec=win_sec),
    )


def sentence_wpm(words: list[Word], script: str | None = None) -> list[float]:
    """WPM per sentence, used by the rubric's "sustained for N sentences" rule.

    Sentence boundaries come from the script when available; otherwise they are
    inferred from long gaps, which is a weaker signal and is documented as such
    at the call site.
    """
    if not words:
        return []

    if script:
        counts = [0]
        tokens = _script_punctuation_positions(script)
        cursor = 0
        bounds: list[list[Word]] = [[]]
        for w in words:
            cursor = _advance(tokens, cursor, w.word)
            bounds[-1].append(w)
            if cursor < len(tokens) and tokens[cursor][1].endswith(PUNCT_END):
                bounds.append([])
        groups = [g for g in bounds if g]
    else:
        groups, current = [], [words[0]]
        for prev, nxt in zip(words, words[1:]):
            if nxt.start - prev.end > 0.55:      # inferred boundary
                groups.append(current)
                current = []
            current.append(nxt)
        groups.append(current)

    out = []
    for g in groups:
        if len(g) < 2:
            continue
        span = g[-1].end - g[0].start
        if span > 0:
            out.append(round(len(g) / span * 60.0, 2))
    return out
