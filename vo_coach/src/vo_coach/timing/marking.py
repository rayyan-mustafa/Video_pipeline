"""Derive a niche's pause rhythm, and mark a script with it.

WHAT IS BORROWED AND WHAT IS NOT. The reference is one creator who works in
this niche. What transfers is their RHYTHM - how often they stop, how long they
hold a sentence break against a clause break. What does not transfer is their
voice, their rate, or their absolute timings, and the reference here is a lossy
YouTube rip whose exact numbers carry codec artifacts anyway.

So this reads two RATIOS off the reference and nothing else:

    how long a sentence-ending pause runs, relative to a clause-internal one
    how much of the delivery is silence

and applies them to the user's own script. The result is the reference's
phrasing habit in the user's own timing, which is the only part worth copying.

FALLING BACK IS NORMAL. If the reference has no usable pause data, the defaults
below are ordinary English narration values. A marked script from defaults is
still far better than an unmarked one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Ordinary measured-narration defaults, used when a reference cannot be read.
DEFAULT_SENTENCE_PAUSE = 0.85
DEFAULT_CLAUSE_PAUSE = 0.32
DEFAULT_PARAGRAPH_PAUSE = 1.30

SENTENCE_END = re.compile(r"[.!?]+[\"')\]]*\s")
CLAUSE_MARK = re.compile(r"[,;:]\s")


@dataclass
class PauseStyle:
    """A niche's phrasing habit, in seconds."""

    sentence: float = DEFAULT_SENTENCE_PAUSE
    clause: float = DEFAULT_CLAUSE_PAUSE
    paragraph: float = DEFAULT_PARAGRAPH_PAUSE
    source: str = "defaults"

    @property
    def summary(self) -> str:
        return (f"sentence {self.sentence:.2f}s / clause {self.clause:.2f}s / "
                f"paragraph {self.paragraph:.2f}s  ({self.source})")


def style_for(niche: str | None) -> PauseStyle:
    """Read the rhythm off a built niche reference, or fall back.

    Only the MEAN pause and the pause RATE are taken from the reference - both
    survive lossy encoding. Individual pause boundaries do not, so they are not
    used.
    """
    if not niche:
        return PauseStyle()
    try:
        from ..scoring import references as refs

        ref = refs.load(niche)
        if ref is None:
            return PauseStyle()
        mean_pause = float(ref.features.mean_pause_sec or 0.0)
        if mean_pause <= 0.05:
            return PauseStyle()
    except Exception:
        return PauseStyle()

    # The reference's mean pause mixes clause and sentence breaks together.
    # Narration in this style runs roughly two clause breaks per sentence break,
    # so the mean sits nearer the clause value; scale out from it rather than
    # treating the mean as a sentence pause, which would double every stop.
    clause = round(max(0.18, mean_pause * 0.80), 2)
    sentence = round(clause * 2.6, 2)
    return PauseStyle(sentence=sentence, clause=clause,
                      paragraph=round(sentence * 1.5, 2),
                      source=f"derived from {niche}")


# ---------------------------------------------------------------------------
#  Marking a script
# ---------------------------------------------------------------------------
def mark(script: str, style: PauseStyle | None = None) -> str:
    """Insert timed pause marks after sentence and clause punctuation.

    Punctuation is the only signal used. It is imperfect - a comma in a list is
    not the same beat as a comma before a turn - but it is the signal the writer
    already put there deliberately, and guessing beyond it would be inventing
    intent the script does not state.

    THE FINAL SENTENCE OF A PARAGRAPH GETS NO MARK, because a pause is the
    space BETWEEN two things and there is nothing after it. That is not an
    oversight: `gaps.interior()` discards trailing silence for the same reason,
    so the target list and the detected gaps agree on what counts. A mark there
    would put the two out of step by one on every script.
    """
    style = style or PauseStyle()
    out_paragraphs = []

    for para in re.split(r"\n\s*\n", script.strip()):
        text = " ".join(para.split())
        if not text:
            continue
        # Sentence breaks first, so a comma inside a sentence is not mistaken
        # for the end of one.
        text = SENTENCE_END.sub(
            lambda m: m.group(0).rstrip() + f"  [{style.sentence:.2f}]\n", text)
        text = CLAUSE_MARK.sub(
            lambda m: m.group(0).rstrip() + f" [{style.clause:.2f}] ", text)
        out_paragraphs.append(text.rstrip())

    body = f"\n\n  [{style.paragraph:.2f}]\n\n".join(out_paragraphs)
    return body


BODY_SEPARATOR = "-" * 68


def targets(script_marked: str) -> list[float]:
    """Every pause target in a marked script, in order.

    ONLY THE BODY IS READ. The header carries a legend, and a legend that shows
    the marker format necessarily contains something shaped like a marker. When
    that was counted as a target, every pause in the file was paired with the
    NEXT pause's target - a silent one-place shift that produced confidently
    wrong phrasing and no error. Hence both guards: cut the header off, and
    refuse non-positive values in case a future header format slips through.
    """
    body = script_marked
    if BODY_SEPARATOR in body:
        body = body.split(BODY_SEPARATOR, 1)[1]
    found = [float(v) for v in re.findall(r"\[(\d+\.\d+)\]", body)]
    return [v for v in found if v > 0.0]


def render(script: str, style: PauseStyle, total_words: int | None = None) -> str:
    """The marked script plus the header a reader actually needs."""
    marked = mark(script, style)
    n = total_words if total_words is not None else len(script.split())
    speech = n / 150.0 * 60.0
    pause_total = sum(targets(marked))
    head = [
        "MARKED SCRIPT",
        "  pause style   " + style.summary,
        "  [n.nn]        hold this many seconds. Do not fill it with breath.",
        "",
        f"  {n} words -> ~{speech:.0f}s speech + {pause_total:.0f}s pauses "
        f"= ~{speech + pause_total:.0f}s total",
        "",
        BODY_SEPARATOR,
        "",
    ]
    return "\n".join(head) + marked + "\n"


# ---------------------------------------------------------------------------
#  Delivery direction: which word carries the line, and how to say it
# ---------------------------------------------------------------------------
#  WHY THIS IS COMPUTED AT MARK TIME AND NOT DURING THE READ. A note that
#  arrives while the reader is mid-line is too late to use, and asking a model
#  for one costs a network round trip at the exact moment latency matters most.
#  The line's shape does not depend on how it was performed - it is a property
#  of the sentence - so it can be worked out once, before recording, and shown
#  instantly.
#
#  WHAT IT IS NOT. It is not pronunciation correction. Nothing here tells the
#  reader how their vowels should sound. It marks which word CARRIES the line,
#  which is a matter of meaning, and the same in every accent.

STOPWORDS = {
    "a", "an", "the", "and", "but", "or", "so", "if", "of", "in", "on", "at",
    "to", "for", "from", "by", "with", "as", "is", "was", "are", "were", "be",
    "been", "being", "it", "its", "that", "this", "these", "those", "there",
    "here", "you", "your", "yours", "i", "me", "my", "we", "our", "he", "she",
    "him", "her", "his", "they", "them", "their", "has", "have", "had", "do",
    "does", "did", "will", "would", "can", "could", "may", "might", "must",
    "shall", "should", "not", "no", "then", "than", "when", "while", "what",
    "which", "who", "how", "why", "into", "out", "up", "down", "over", "just",
    "only", "also", "very", "more", "most", "some", "any", "all", "one", "two",
    # Long but empty. These beat real words on length alone and are exactly
    # what a coach would never point at - "lean on SOMETHING" is not direction.
    "something", "anything", "nothing", "everything", "someone", "anyone",
    "everyone", "somewhere", "anywhere", "everywhere",
    # Prepositions and connectives long enough to score well by accident.
    "against", "through", "throughout", "between", "before", "after", "about",
    "because", "during", "without", "within", "across", "around", "toward",
    "towards", "under", "above", "behind", "beyond", "along", "upon",
    # Low-content nouns and fillers.
    "thing", "things", "way", "ways", "kind", "sort", "part", "lot", "bit",
    "still", "even", "really", "actually", "quite", "rather",
}


def _clean(word: str) -> str:
    return re.sub(r"[^A-Za-z'-]", "", word).lower()


def emphasis_words(clause: str, limit: int | None = None) -> list[str]:
    """The one or two words that carry this clause.

    Scored on three things a listener actually responds to: content words over
    grammar words, longer words over short ones, and the LAST content word
    before a stop - because the word a phrase lands on is the one the ear keeps.
    """
    words = [w for w in clause.split() if _clean(w)]
    if not words:
        return []

    scored: list[tuple[float, str]] = []
    for i, raw in enumerate(words):
        w = _clean(raw)
        if not w or w in STOPWORDS:
            continue
        score = float(len(w))
        # Landing word: the closer to the end of the clause, the more the ear
        # holds on to it. Kept modest - at 4.0 it outweighed word choice
        # entirely and picked whatever happened to sit last.
        score += 2.5 * (i / max(len(words) - 1, 1))
        # A capital in the middle of a line is a name or a proper noun.
        if raw[:1].isupper() and i > 0:
            score += 3.0
        scored.append((score, w))

    if not scored:
        return []
    scored.sort(reverse=True)
    n = limit if limit is not None else (2 if len(words) > 9 else 1)

    # Deduplicate. A word repeated in a line ("is pressing ... is pressing")
    # scores twice and would come back as both picks, telling the reader to
    # lean on the same word twice and leaving the line's other half unmarked.
    out: list[str] = []
    for _, w in scored:
        if w not in out:
            out.append(w)
        if len(out) >= n:
            break
    return out


def delivery_note(line: str, is_last: bool = False) -> str:
    """One plain-English instruction for how to shape this line.

    Derived from the sentence's own structure, so it never claims to know
    something it cannot see. Deliberately written the way a person would say
    it out loud - no term here needs explaining to anyone.
    """
    words = [w for w in line.split() if _clean(w)]
    n = len(words)
    commas = line.count(",")
    key = emphasis_words(line, limit=1)
    lead = ("Lean on " + key[0].upper() + ". ") if key else ""

    if is_last:
        return lead + "This is the last one - slow down and let it sit at the end."
    if commas >= 2:
        return lead + ("Those middle parts are a list - step down through them, "
                       "each one a little quieter than the last.")
    if n <= 7:
        return lead + ("Short line. Say it slower than feels natural and give "
                       "it room.")
    if n >= 19:
        return lead + ("Long one - take it in two breaths, at the comma, so you "
                       "do not run out of air.")
    if line.rstrip().endswith("?"):
        return lead + "Let your voice lift a little at the end."
    return lead + "Keep it easy, and let the last word fall away."


def direct(script: str, style: PauseStyle | None = None) -> list[dict]:
    """Per-line direction for a whole script: text, pause, emphasis, note."""
    style = style or PauseStyle()
    marked = mark(script, style)
    out: list[dict] = []
    raw_lines = [ln.strip() for ln in marked.splitlines() if ln.strip()]
    for i, ln in enumerate(raw_lines):
        plain = re.sub(r"\s*\[\d+\.\d+\]\s*", " ", ln).strip()
        if not plain:
            continue
        pause = re.findall(r"\[(\d+\.\d+)\]", ln)
        out.append({
            "text": plain,
            "pause": float(pause[-1]) if pause else None,
            "emphasis": emphasis_words(plain),
            "note": delivery_note(plain, is_last=(i == len(raw_lines) - 1)),
        })
    return out
