"""
Step 15b -- the read sheets. One file per section, containing exactly the words
you say into the mic.

=============================================================================
 WHY THIS EXISTS
=============================================================================
`vo_raw.txt` is one continuous blob and `script_tagged.json` is tagged sentences.
Neither is something you can read from, and `vo_coach mark-script` wants one
plain-text file per section. Until this module existed the only copy of that
logic was a copy-paste snippet in a notebook markdown cell -- in Phase 1, which
happens AFTER you have already recorded.

=============================================================================
 WHY build_chunks() IS DUPLICATED HERE
=============================================================================
It is not duplicated for convenience. The notebook's aligner (Step 17) decides
which words it EXPECTS to hear in section N by building chunks and keeping the
ones tagged with that section, then matching them against Whisper's transcript
with difflib. If the read sheet groups sentences even slightly differently, the
words you actually read and the words the aligner expects disagree, the word
match rate drops, and the scene boundaries in that section become guesses.

The difference is real and not hypothetical: build_chunks() merges a short
sentence into the buffer that is already open, and that buffer keeps the section
of the sentence that OPENED it. So a short sentence at the top of section 4 can
belong to a chunk the aligner files under section 3. Grouping by the `section`
field alone would put that sentence in the wrong read sheet.

So the chunker below is a verbatim port of the notebook's, and
`test_notebook.py` executes the real cell and asserts the two agree on a
generated script. If someone edits one, that check fails.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from . import config, paths

# Both constants are the notebook's. Changing either here without changing it
# there is exactly the drift the parity test exists to catch.
MAX_CHUNK_CHARS = 280        # a scene longer than this gets split at a comma/colon
MIN_CHUNK_CHARS = 60         # a fragment shorter than this merges into the next

SECTIONS_DIR = "sections"
MANIFEST = "manifest.json"


def build_chunks(tagged: list[dict]) -> list[dict]:
    """[{'sentence','section',...}] -> [{'text','section'}], one entry per scene.

    Verbatim port of build_chunks() in notebook Step 17. Do not 'improve' it.
    """
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


def section_texts(tagged: list[dict]) -> dict[int, list[str]]:
    """{section number: [chunk text, ...]} -- the aligner's own grouping."""
    out: dict[int, list[str]] = {}
    for c in build_chunks(tagged):
        out.setdefault(c["section"], []).append(c["text"])
    return dict(sorted(out.items()))


# =============================================================================
#  WRITING THE SHEETS
# =============================================================================
# The .txt files contain the script and NOTHING else -- no header, no section
# name, no word count. `vo_coach mark-script` reads the whole file as the script,
# so a header line would be marked up and then read aloud.  Everything a human
# needs is in README.txt beside them.
#
# One sentence per line, no blank lines. mark-script collapses single newlines
# inside a paragraph, but treats a BLANK line as a paragraph break worth an extra
# pause -- so a blank line here would silently add ~1 s of room tone mid-section.
# =============================================================================

def write(tagged: list[dict], beat_sheet: dict | None = None,
          niche: str = "zenn") -> list[Path]:
    """Write sections/section_NN.txt, sections/README.txt, sections/manifest.json."""
    texts = section_texts(tagged)
    if not texts:
        return []

    d = paths.p(SECTIONS_DIR)
    d.mkdir(parents=True, exist_ok=True)

    # Stale sheets from a previous topic in the same working folder are worse
    # than no sheets: you would record section_07 of a script that no longer has
    # one, and the aligner would never look for it.
    for old in d.glob("section_*.txt"):
        old.unlink()

    beat_sheet = beat_sheet or {}
    written: list[Path] = []
    entries = []

    for sec, chunks in texts.items():
        # One sentence per line for reading; the words are identical either way.
        body = "\n".join(chunks) + "\n"
        f = d / f"section_{sec:02d}.txt"
        f.write_text(body, encoding="utf-8")
        written.append(f)

        spec = beat_sheet.get(sec) or beat_sheet.get(str(sec)) or {}
        words = len(" ".join(chunks).split())
        entries.append({
            "section": sec,
            "name": spec.get("name", ""),
            "file": f"{SECTIONS_DIR}/{f.name}",
            "recording": f"voice_recordings/section_{sec:02d}.wav",
            "words": words,
            "chunks": len(chunks),
            "expected_seconds": round(words / config.SPOKEN_WPM * 60.0, 1),
        })

    (d / MANIFEST).write_text(json.dumps({
        "wpm_assumed": config.SPOKEN_WPM,
        "sections": entries,
    }, indent=2), encoding="utf-8")

    (d / "README.txt").write_text(_readme(entries, niche), encoding="utf-8")
    return written


def load_manifest() -> dict | None:
    f = paths.p(SECTIONS_DIR, MANIFEST)
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return None


def _readme(entries: list[dict], niche: str) -> str:
    total_words = sum(e["words"] for e in entries)
    total_min = total_words / config.SPOKEN_WPM

    lines = [
        "READ SHEETS",
        "=" * 68,
        "",
        f"{len(entries)} sections, {total_words} words, ~{total_min:.1f} min "
        f"at {config.SPOKEN_WPM} wpm.",
        "",
        "Read each file EXACTLY as written. The notebook finds where each picture",
        "goes by matching your words against this text, so improvising moves the",
        "pictures off your voice. Same room, same mic, same distance every time,",
        "and no noise reduction on the way in.",
        "",
        "-" * 68,
        f"{'section':<9}{'name':<26}{'words':>7}{'~mm:ss':>9}",
        "-" * 68,
    ]
    for e in entries:
        secs = e["expected_seconds"]
        lines.append(f"{e['section']:<9}{e['name'][:25]:<26}{e['words']:>7}"
                     f"{int(secs // 60):>6}:{int(secs % 60):02d}")
    lines += [
        "-" * 68,
        "",
        "FOR EACH SECTION, IN ORDER:",
        "",
    ]

    for e in entries:
        n = e["section"]
        lines += [
            f"  --- section {n}{(' - ' + e['name']) if e['name'] else ''} "
            f"({e['words']} words) ---",
            f"    python -m vo_coach mark-script {SECTIONS_DIR}/section_{n:02d}.txt "
            f"--niche {niche}",
            f"    python -m vo_coach live --script {SECTIONS_DIR}/section_{n:02d}_marked.txt "
            f"--niche {niche}",
            f"    python collect_takes.py --section {n}",
            "",
        ]

    lines += [
        "THEN, BEFORE YOU OPEN COLAB:",
        "",
        "    python run_qc.py",
        "",
        "It measures clipping, noise and length on the laptop, where re-recording",
        "is free. Whisper will not tell you the audio was bad - it will just align",
        "it badly, on a paid runtime, after the frames are drawn.",
        "",
        "DO NOT run `vo_coach master` on these files. The notebook masters twice by",
        "design (per section before alignment, then once over the whole thing), and",
        "a limiter applied three times sounds squashed. Hand it the comped,",
        "unmastered take.",
        "",
        "Anything that changes DURATION - comping, fix-pauses - happens here, on the",
        "laptop, before the notebook. Step 18 locks the length so the pictures stay",
        "in sync; nothing may change it after that.",
    ]
    return "\n".join(lines) + "\n"
