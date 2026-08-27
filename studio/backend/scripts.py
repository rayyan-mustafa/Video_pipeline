"""The script: read, write, and everything derivable from it before recording.

All of this is computed by importing the pipeline's own functions rather than
reimplementing them, so the UI can never disagree with what `plan_script.py`
would do on the same text. `find_brackets`, `split_sections` and `to_tagged` come
straight from plan_script; the split counts come from scenesplit; the pacing
verdict from pacing.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

from . import paths

paths.ensure_importable()
from pipeline import pacing, scenesplit          # noqa: E402


def _plan_script_module():
    """Import plan_script.py by path.

    It lives at channels/ root, not inside the `pipeline` package, so a normal
    import will not find it. Loading it by spec keeps the bracket rule and the
    section-splitting rule in ONE place - the alternative is retyping the regexes
    here and having them quietly drift from the CLI's.
    """
    spec = importlib.util.spec_from_file_location("_plan_script", paths.PLAN_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)                 # type: ignore[union-attr]
    return mod


def read(proj) -> str:
    return proj.script.read_text(encoding="utf-8") if proj.script.exists() else ""


def write(proj, text: str) -> dict:
    """Save the script. Returns the fresh analysis so the UI updates in one trip.

    Normalises CRLF to LF: the sentence splitter keys on `\\n`, and a file saved
    from a browser textarea on Windows would otherwise carry `\\r` into every
    scene's text - which then changes the fingerprint and silently invalidates
    the image bindings.
    """
    proj.dir.mkdir(parents=True, exist_ok=True)
    proj.script.write_text(text.replace("\r\n", "\n").replace("\r", "\n"),
                           encoding="utf-8")
    return analyse(proj)


def analyse(proj, wpm: int = 150) -> dict:
    """Everything the Script page shows, from the text alone - no audio needed."""
    text = read(proj)
    ps = _plan_script_module()

    brackets = ps.find_brackets(text)
    blocks = ps.split_sections(text)
    tagged = ps.to_tagged(blocks) if blocks else []

    words = len(text.split())
    est_seconds = pacing.seconds_for_words(words, wpm)

    sections = [{
        "n": i,
        "words": len(b.split()),
        "lines": len([ln for ln in b.splitlines() if ln.strip()]),
        "est_seconds": round(pacing.seconds_for_words(len(b.split()), wpm), 1),
        "preview": b.splitlines()[0][:90] if b.splitlines() else "",
    } for i, b in enumerate(blocks, 1)]

    # What each split mode would produce. This is the number that decides how
    # many pictures the video can have, so it belongs next to the pacing target
    # rather than buried in a CLI flag.
    counts = scenesplit.counts(tagged) if tagged else {}

    return {
        "text": text,
        "words": words,
        "est_seconds": round(est_seconds, 1),
        "est_minutes": round(est_seconds / 60, 2),
        "wpm": wpm,
        "sections": sections,
        "brackets": brackets,
        "bracket_ok": not brackets,
        "split_counts": counts,
        "pacing": {
            "target_low": pacing.TARGET_LOW,
            "target_high": pacing.TARGET_HIGH,
            "hard_max": pacing.HARD_MAX,
            "options": pacing.plan_table(est_seconds),
            # Which split mode lands closest to the pacing target, so the UI can
            # recommend one instead of making the user work it out.
            "recommended_split": _recommend(counts, est_seconds, words),
        },
    }


def _recommend(counts: dict[str, int], est_seconds: float,
               total_words: int = 0) -> dict | None:
    """Which split mode to use, and which binding strategy that implies.

    A SCENE IS NOT THE HARD CAP ON HOW FAST THE VIDEO CAN CUT - a word is.
    There are two binding strategies and they have different limits:

      1:1 binding   one picture per scene, every cut on a sentence edge. Needs
                    the scene count to roughly equal the picture count, so the
                    split mode caps it. This is the cleanest result.
      word cuts     more pictures than scenes; each section is divided at exact
                    Whisper word timestamps (imageplan._exact_word_cuts). Cuts
                    land inside sentences, but on real word boundaries, never
                    interpolated.

    So wanting 210 pictures from a 114-scene script is entirely reachable - it
    just means strategy 2, at about six words per picture. Reporting that as
    "unreachable" (as this function first did) would have sent the user off to
    rewrite a script that was fine.
    """
    if not counts:
        return None
    want = pacing.images_for(est_seconds)
    best = min(counts.items(), key=lambda kv: abs(kv[1] - want))
    finest = max(counts.values())
    words_each = (total_words / want) if want else 0

    if finest >= want:
        strategy, note = "1:1", (
            f"{best[0]} gives {best[1]} scenes against ~{want} pictures wanted, "
            f"so each picture can own one scene and every cut lands on a "
            f"sentence edge."
        )
    else:
        # Below roughly 4 words a cut arrives mid-phrase often enough to read as
        # a stutter rather than as pace.
        tight = words_each < 4
        strategy = "word-cuts"
        note = (
            f"~{want} pictures over {finest} scenes means cutting inside "
            f"sentences, at exact word timestamps - about {words_each:.1f} words "
            f"per picture."
            + (" That is very tight; consider a slower pace or shorter sentences."
               if tight else " That is comfortable for a fast-cut edit.")
        )

    return {
        "mode": best[0] if finest >= want else max(counts, key=counts.get),
        "scenes": best[1] if finest >= want else finest,
        "want_images": want,
        "strategy": strategy,
        "words_per_image": round(words_each, 1),
        "reachable": True,          # both strategies are supported; see above
        "note": note,
    }
