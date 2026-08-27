"""How the script is cut into scenes. One decision, used by two commands.

WHY THIS IS A CHOICE AND NOT A CONSTANT
---------------------------------------
build_chunks() (ported verbatim from the notebook) MERGES short sentences and
splits long ones, targeting a readable ~20-word scene. That was right when one
frame was drawn per scene by a model: a three-word line should not become its own
0.6-second frame.

It is wrong when the pictures were commissioned one-per-sentence. Merging then
puts two images' worth of narration into one scene, and the planner has to divide
it back out - reconstructing, badly, information that was thrown away.

So the split is a mode, recorded in script_plan.json, and the timeline builder
reads it back. Both sides MUST use the same one or the scene indices stop lining
up and the fingerprint check fails - which is the point of the fingerprint.

    chunk       merge short, split long, target ~20 words   (the notebook's)
    sentence    one scene per sentence
    dash        one scene per sentence, and also split on em-dash

WHY "dash" EXISTS. An editorial script uses the em-dash as a beat -
"Neanderthals had thicker jaws - and not one of them ever grew a chin." A writer
reads that as two thoughts and an illustrator draws two pictures, so a tool asked
for "one image per sentence" tends to produce one per beat. On the chin script:
107 sentences, 125 beats, 122 images actually delivered.
"""

from __future__ import annotations

import re

from .align import build_chunks

MODES = ("chunk", "sentence", "dash")

# Sentence end, plus (in dash mode) a spaced em/en-dash. The dash must be spaced
# on both sides: an unspaced one is a compound word ("chin-related"), not a beat.
_SENT = r"(?<=[.!?])\s+"
_DASH = r"\s+[—–]\s+"

# Below this, a fragment is glued to its neighbour even in sentence/dash mode.
# "Not one." "Ever." are two sentences and one picture; giving each its own scene
# creates sub-second scenes that no image can sit on.
MIN_WORDS = 3


def _split_line(line: str, mode: str) -> list[str]:
    pat = _SENT if mode == "sentence" else f"{_SENT}|{_DASH}"
    return [s.strip() for s in re.split(pat, line) if s.strip()]


def _glue_runts(parts: list[str]) -> list[str]:
    """Merge fragments under MIN_WORDS into the previous part (or the next)."""
    out: list[str] = []
    for p in parts:
        if out and len(p.split()) < MIN_WORDS:
            out[-1] = f"{out[-1]} {p}"
        else:
            out.append(p)
    # A leading runt has no previous part to join, so it absorbs the next one.
    if len(out) > 1 and len(out[0].split()) < MIN_WORDS:
        out[1] = f"{out[0]} {out[1]}"
        out.pop(0)
    return out


def make_scenes(tagged: list[dict], mode: str = "chunk") -> list[dict]:
    """[{'sentence','section'}] -> [{'text','section'}], one entry per scene.

    Returns the same shape build_chunks() does, so every downstream consumer -
    the aligner, the planner, the fingerprint - is unchanged.
    """
    if mode not in MODES:
        raise ValueError(f"unknown split mode {mode!r} (expected one of {MODES})")
    if mode == "chunk":
        return build_chunks(tagged)

    by_section: dict[int, list[str]] = {}
    for it in tagged:
        s = it["sentence"].strip()
        if s:
            by_section.setdefault(it["section"], []).extend(_split_line(s, mode))

    scenes: list[dict] = []
    for sec in sorted(by_section):
        for text in _glue_runts(by_section[sec]):
            scenes.append({"text": text, "section": sec})
    return scenes


def match_scenes(tagged: list[dict], targets: dict[int, int]) -> list[dict]:
    """Split the script into EXACTLY `targets[section]` scenes per section.

    WHY THIS EXISTS. When the pictures were commissioned one-per-sentence, the
    pictures themselves are the record of how the script was divided - and that
    record is more authoritative than any splitting rule reconstructed after the
    fact. Section 4 of the chin script has 36 sentences and 36 images, so the
    division is known exactly. Section 3 has 36 sentences and 35 images, so the
    generator merged one pair, and guessing which pair from punctuation alone is
    exactly the kind of reconstruction that puts every later picture one sentence
    out.

    So instead: start from raw sentences and merge the CHEAPEST adjacent pair
    (fewest combined words) until the count matches. Cheapest-first means the
    merge lands on the short fragments a human would also have combined -
    "Not one." "Ever." - rather than on two full sentences.

    If a section has fewer sentences than pictures, the longest scene is split at
    its best internal break instead.
    """
    by_section: dict[int, list[str]] = {}
    for it in tagged:
        s = it["sentence"].strip()
        if s:
            by_section.setdefault(it["section"], []).append(s)

    scenes: list[dict] = []
    for sec in sorted(by_section):
        parts = list(by_section[sec])
        want = int(targets.get(sec, len(parts)))

        while len(parts) > want and len(parts) > 1:
            costs = [len(parts[i].split()) + len(parts[i + 1].split())
                     for i in range(len(parts) - 1)]
            i = costs.index(min(costs))
            parts[i : i + 2] = [f"{parts[i]} {parts[i + 1]}"]

        while len(parts) < want:
            # Split the wordiest scene at its best internal break. A scene with
            # no break left cannot be divided, so bail rather than loop forever.
            i = max(range(len(parts)), key=lambda j: len(parts[j].split()))
            bits = [b.strip() for b in re.split(r"\s+[—–]\s+|(?<=[,;:])\s+", parts[i])
                    if b.strip()]
            if len(bits) < 2:
                break
            mid = len(bits) // 2
            parts[i : i + 1] = [" ".join(bits[:mid]), " ".join(bits[mid:])]

        for text in parts:
            scenes.append({"text": text, "section": sec})
    return scenes


def counts(tagged: list[dict]) -> dict[str, int]:
    """Scene count under every mode - for showing the user which one fits."""
    return {m: len(make_scenes(tagged, m)) for m in MODES}
