"""Scene splitting and word-level alignment - ported VERBATIM from stickman.ipynb cell039.

build_chunks() is the ONLY place scene boundaries are decided. align() maps the
words Whisper actually heard onto the words the script expected, via difflib, so
every scene start is anchored to a real timestamp rather than to a wpm estimate.

Alignment is per-section on purpose: an error inside one section cannot propagate
into the next.
"""

from __future__ import annotations

import difflib
import re

import numpy as np


MAX_CHUNK_CHARS = 280        # a scene longer than this gets split at a comma/colon
MIN_CHUNK_CHARS = 60         # a fragment shorter than this merges into the next


def build_chunks(tagged):
    """[{'sentence','section',...}] -> [{'text','section'}], one entry per scene."""
    chunks, buf, buf_sec = [], "", 1
    for it in tagged:
        s = it["sentence"].strip()
        if not s:
            continue
        if len(s) > MAX_CHUNK_CHARS:
            parts, cur = [], ""
            for piece in re.split(r"(?<=[,;:])\s+", s):
                if len(cur) + len(piece) + 1 <= MAX_CHUNK_CHARS:
                    cur = f"{cur} {piece}".strip()
                else:
                    if cur:
                        parts.append(cur)
                    cur = piece
            if cur:
                parts.append(cur)
        else:
            parts = [s]

        for p in parts:
            if buf and (len(buf) + len(p) + 1 > MAX_CHUNK_CHARS or len(buf) >= MIN_CHUNK_CHARS):
                chunks.append({"text": buf, "section": buf_sec})
                buf, buf_sec = p, it["section"]
            elif buf:
                buf = f"{buf} {p}"
            else:
                buf, buf_sec = p, it["section"]
    if buf:
        chunks.append({"text": buf, "section": buf_sec})
    return chunks


def norm_words(t):
    return re.sub(r"[^a-z0-9' ]", " ", t.lower()).split()


def anchor_report(words, chunk_texts):
    """Per-scene: did this scene get a REAL timestamp, or an interpolated guess?

    WHY THIS IS THE NUMBER THAT MATTERS, AND WORD-MATCH % IS NOT.
    align() returns `len(e2h) / len(expected)` - the fraction of ALL script words
    Whisper reproduced. That is a transcription-quality figure, and it can never
    reach 100% on a script containing numerals or proper nouns: the script writes
    "2007" and Whisper hears "two thousand and seven"; the script writes
    "Cramon-Taubadel" and Whisper writes whatever it writes. Those tokens are
    guaranteed misses and they say nothing about picture timing.

    What actually decides whether a picture lands on the right words is narrower:
    each scene needs ONE matched word near its start to anchor on (see the anchor
    loop in align()). A scene whose opening word matched gets that word's exact
    Whisper timestamp - the boundary is then correct to the millisecond, no matter
    how many words later in the scene were misheard.

    Only scenes with NO match in their opening window fall back to interpolation
    between neighbours, and only those can actually drift.

    Returns (anchored_flags, offsets) where offsets[i] is how many words past the
    scene's start the anchor was found - 0 means the very first word matched.
    """
    heard = []
    hw = []
    for w in words:
        nw = norm_words(w.word)
        if nw:
            heard.append(nw[0])
            hw.append(w)

    expected, owner = [], []
    for ci, t in enumerate(chunk_texts):
        for tok in norm_words(t):
            expected.append(tok)
            owner.append(ci)

    e2h = {}
    if expected and heard:
        sm = difflib.SequenceMatcher(None, expected, heard, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                for k in range(i2 - i1):
                    e2h[i1 + k] = j1 + k

    first_of = {}
    for ei, ci in enumerate(owner):
        first_of.setdefault(ci, ei)

    n = len(chunk_texts)
    anchored = [False] * n
    offsets = [None] * n
    for ci in range(n):
        for ei in range(first_of[ci], len(expected)):
            if owner[ei] != ci and not anchored[ci] and ei > first_of[ci] + 8:
                break
            if ei in e2h:
                anchored[ci] = True
                offsets[ci] = ei - first_of[ci]
                break
    return anchored, offsets


def align(words, chunk_texts, total_s):
    """Map each chunk to (start, end) seconds within one section.

    `words` is Whisper's word list, `chunk_texts` the scenes the script says
    this section contains. Returns one (start, end) per chunk, monotonic and
    spanning the whole section.
    """
    heard, hw = [], []
    for w in words:
        nw = norm_words(w.word)
        if nw:
            heard.append(nw[0])
            hw.append(w)

    expected, owner = [], []
    for ci, t in enumerate(chunk_texts):
        for tok in norm_words(t):
            expected.append(tok)
            owner.append(ci)

    e2h = {}
    if expected and heard:
        sm = difflib.SequenceMatcher(None, expected, heard, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                for k in range(i2 - i1):
                    e2h[i1 + k] = j1 + k

    first_of = {}
    for ei, ci in enumerate(owner):
        first_of.setdefault(ci, ei)

    # Anchor each chunk on the first of its words that Whisper actually matched.
    n = len(chunk_texts)
    starts = [None] * n
    for ci in range(n):
        for ei in range(first_of[ci], len(expected)):
            if owner[ei] != ci and starts[ci] is None and ei > first_of[ci] + 8:
                break            # searched well past this chunk; give up on it
            if ei in e2h:
                starts[ci] = hw[e2h[ei]].start
                break
    starts[0] = 0.0

    # Unmatched chunks (a dropped or misread opening word) get interpolated
    # between their nearest anchored neighbours, weighted by word count.
    known = [i for i, s in enumerate(starts) if s is not None]
    weights = np.cumsum([0] + [max(1, len(norm_words(t))) for t in chunk_texts])
    for i in range(n):
        if starts[i] is not None:
            continue
        lo = max([k for k in known if k < i], default=None)
        hi = min([k for k in known if k > i], default=None)
        if lo is None:
            starts[i] = 0.0
        elif hi is None:
            span = total_s - starts[lo]
            frac = (weights[i] - weights[lo]) / max(weights[n] - weights[lo], 1)
            starts[i] = starts[lo] + span * frac
        else:
            frac = (weights[i] - weights[lo]) / max(weights[hi] - weights[lo], 1)
            starts[i] = starts[lo] + (starts[hi] - starts[lo]) * frac

    starts = np.maximum.accumulate(np.array(starts, dtype=float))  # force monotonic
    ends = np.append(starts[1:], total_s)
    ends = np.maximum(ends, starts + 0.05)
    ends[-1] = total_s
    return list(zip(starts, ends)), len(e2h) / max(len(expected), 1)
