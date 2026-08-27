"""
Steps 8 / 11 / 14 -- write the script, and the two machines that edit it.

Three things live here:

  write()           the first draft, from the beat sheet and the brief
  apply_fixes()     line-level repair, used by BOTH the grounding gate and the
                    retention gate with a different instruction string
  rewrite_section() whole-section rebuild, for problems no single line can fix

parse_tagged_script / strip_tags_to_voiceover / save_script are the format layer.
The [S<section>|<sources>] tag is the spine of the entire pipeline: it is what
lets each gate judge a section against its own spec instead of guessing boundaries
from running word counts, and it is what the fact-checker uses to check a claim
against only the source it actually cites.
"""

from __future__ import annotations

import json
import re

from . import config, humanize, paths
from .brief import Brief
from .engine import call_model

TAG_RE = re.compile(r"^(.*?)\s*\[\s*S(\d+)\s*\|\s*([\d,\s]*)\s*\]\s*$", re.IGNORECASE)


# =============================================================================
#  FORMAT LAYER
# =============================================================================

def parse_tagged_script(text: str) -> list[dict]:
    """Turns tagged lines into [{'sentence', 'section', 'sources'}]."""
    parsed = []
    current_section = 1
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = TAG_RE.match(line)
        if m:
            sentence = m.group(1).strip()
            section = int(m.group(2))
            sources = [int(n) for n in re.findall(r"\d+", m.group(3)) if int(n) != 0]
            current_section = section
        else:
            # Model dropped the tag. Keep the sentence, inherit the running section,
            # mark it untagged so it surfaces in the linter rather than vanishing.
            sentence, section, sources = line, current_section, []
        parsed.append({"sentence": sentence, "section": section, "sources": sources})
    return parsed


def strip_tags_to_voiceover(tagged_sentences: list[dict]) -> str:
    return " ".join(item["sentence"] for item in tagged_sentences)


def save_script(tagged_sentences: list[dict], label: str = "") -> None:
    with open(paths.p("script_tagged.json"), "w", encoding="utf-8") as f:
        json.dump(tagged_sentences, f, indent=2)
    with open(paths.p("vo_raw.txt"), "w", encoding="utf-8") as f:
        f.write(strip_tags_to_voiceover(tagged_sentences))
    if label:
        print(f"  saved ({label})")


def load_script() -> list[dict]:
    with open(paths.p("script_tagged.json"), "r", encoding="utf-8") as f:
        return json.load(f)


def section_table(tagged_sentences: list[dict], beat_sheet: dict) -> None:
    by_sec: dict[int, list] = {}
    for it in tagged_sentences:
        by_sec.setdefault(it["section"], []).append(it)
    for n in sorted(by_sec):
        words = sum(len(i["sentence"].split()) for i in by_sec[n])
        name = beat_sheet.get(n, {}).get("name", "?")
        tgt = beat_sheet.get(n, {}).get("words", (0, 0))
        flag = "" if tgt[0] <= words <= tgt[1] else "  <-- OUT OF RANGE"
        print(f"  S{n} {name:<24} {words:>4} words (target {tgt[0]}-{tgt[1]}){flag}")


# =============================================================================
#  THE WRITER
# =============================================================================

def _system_prompt(b: Brief) -> str:
    return f"""You are the writer for 'Weird Human Biology', a YouTube channel that
takes one ordinary thing about the human body and makes a viewer unable to stop watching.
Write the voiceover script for: {b.topic}

Your reader is one person, alone, holding a phone. Write to them, not to an audience.

=============================================================================
 THE HUMAN STAKE - the difference between interesting and unmissable
=============================================================================
A strange fact about the body earns attention. It does not earn a share, a
comment, or a subscribe. What does that is a fact that turns out to be about
SOMEONE - not only about the listener's own body.

So every script needs both, and the second one is the one usually missing:

  THE CURIOSITY GAP   something the viewer assumed, broken by evidence.
  THE HUMAN STAKE     who this is actually happening to, and what it costs
                      them. Whose ordinary day is quietly harder because of
                      it. Who has been living with it unnoticed.

The stake must be a real person or a real group the sources support - people
with a condition, in an occupation, at an age, in a place. Never invented, never
a hypothetical "imagine someone who...", never a statistic standing in for a
person. If the sources will not support a human stake, write the section without
one rather than fabricating it; the fact-checker will cut it either way.

Do not announce the stake. Do not write "and that matters because". Put the
person in the sentence and let the weight arrive on its own - the same rule
Section 5 already follows for feeling.

=============================================================================
 WHAT YOU ARE ALLOWED TO SAY
=============================================================================
Everything factual in this script must come from the brief below. This is not a
style note - a sentence containing a fact that is not in this brief will be caught
by an automated checker and cut.

THE ONLY NUMBERS YOU MAY USE (using any other number will be flagged and removed):
{b.numbers_block}

THE ONLY NAMES, INSTITUTIONS AND YEARS YOU MAY USE:
{b.entities_block}

KEY FINDINGS (reproduce the certainty label - if it says "debated", the script must
say it is debated, not state it as settled):
{b.findings_block}

THE MYTH THIS PUSHES BACK ON: {b.profile.get('the_common_myth')}

YOUR STRONGEST HOOK MATERIAL: {b.profile.get('the_best_counter_evidence')}

MECHANISM AS THE SOURCES DESCRIBE IT: {b.profile.get('mechanism')}

THINGS THE SOURCES DO NOT SUPPORT - you may not write these even if they feel true:
{b.gaps_block}

=============================================================================
 THE BEAT SHEET - follow it exactly, section by section, in order
=============================================================================
Word-count ranges are hard constraints. Each beat must appear, in sequence. Section
boundaries must be completely invisible to a listener - only the pacing and the beats
carry the structure, never a label or a transition announcement.

{b.render_beat_sheet()}

=============================================================================
 VOICE RULES - these apply to every section
=============================================================================
1. BANNED PHRASES. Never use any of these: {b.banned_block}
   They are the tells of machine-written narration and an automated linter will
   reject the script if it finds them.

2. NEVER LABEL A FEELING. You may not tell the viewer that something is unsettling,
   fascinating, remarkable, beautiful, or surprising. Describe the thing accurately
   enough that they feel it themselves. If you find yourself reaching for an emotion
   word, you have not described the fact concretely enough yet - go back and add the
   concrete detail instead.

3. ONE ADJECTIVE PER NOUN. Never stack them.

4. SENTENCE LENGTH MUST VARY. After any sentence over 20 words, the next must be
   under 10. A run of three similar-length sentences reads as narration and loses
   the viewer.

5. NEVER RUN TWO FACT-CARRYING SENTENCES BACK TO BACK. A statistic must always be
   followed by something the viewer can picture. Two numbers in a row is where
   people close the tab.

6. CONCRETE NOUNS. "Skin", "muscle", "hair follicle", "cold water" - not "physiological
   response", "biological system", "evolutionary pressure".

7. SECOND PERSON. It is happening in the viewer's body, right now, as they listen.

8. NO CALLS TO ACTION anywhere. No subscribing, no commenting, no thanking.

{humanize.writer_block()}

=============================================================================
 OUTPUT FORMAT - follow exactly
=============================================================================
Write ONE sentence per line. End every single line with a tag in this exact shape:

    [S<section number>|<source numbers>]

Examples:
    Press two fingers against the back of your arm. [S1|0]
    That reflex shows up in nearly every mammal that has been studied. [S3|1]
    Two teams disagree about why, and neither has settled it. [S3|1,2]
    You are still carrying the wiring for a coat you no longer have. [S5|0]

The section number says which beat-sheet section that line belongs to. The source
numbers say where its factual content comes from - use 0 when the line carries no
factual claim at all.

CRITICAL RULE ABOUT [0]: tag a line 0 ONLY if it makes no checkable assertion about
the world. "Your ancestors needed this to survive the cold" is a factual claim and
must cite a source, even though it sounds like narration. Rhetorical questions,
direct address, and pure imagery are the only things that are genuinely 0. Lines
tagged 0 are checked separately for smuggled claims, so mis-tagging costs you.

Every line ends with exactly one tag. No headers, no section names, no stage
directions, no sound cues, no markdown, no commentary before or after. The tags are
stripped before the script is voiced, so each sentence must read naturally on its
own and the whole thing must flow as one continuous piece of speech."""


def write(b: Brief) -> list[dict]:
    """First draft. Writes script_tagged.json + vo_raw.txt."""
    print("Writing script...")
    raw = call_model(
        config.MODEL_WRITER,
        f"Write the full script for: {b.topic}",
        system=_system_prompt(b),
        max_tokens=8000,       # a 1,400-word tagged script can brush a 3500 ceiling
        temperature=0.6,
    )

    tagged = parse_tagged_script(raw)
    save_script(tagged, "first draft")

    wc = len(strip_tags_to_voiceover(tagged).split())
    print(f"\nDraft written: {wc} words, {len(tagged)} sentences "
          f"(~{wc/config.SPOKEN_WPM:.1f} min at {config.SPOKEN_WPM} wpm)")
    section_table(tagged, b.beat_sheet)
    return tagged


# =============================================================================
#  THE SHARED FIXER
# =============================================================================
# v4 mapped rewrites back onto originals by POSITION, through a dict keyed on the
# rewritten sentence text. Two ways that silently corrupted a script: if the model
# dropped one sentence, every later fix shifted up by one and attached to the wrong
# original; and if two rewrites came back identical, the dict deduplicated them and
# shifted everything after. Neither raised an error - you just got a script with
# sentences swapped into the wrong places.
#
# This version gives every flagged sentence a stable numeric id and requires the model
# to echo it back. Order-independent, drop-safe, duplicate-safe.
# =============================================================================

def apply_fixes(tagged_sentences: list[dict], issues: list[dict], instruction: str,
                b: Brief, temperature: float = 0.4) -> list[dict]:
    if not issues:
        return tagged_sentences

    # ALL occurrences, not just the first.
    #
    # This used to be `setdefault`, which mapped every copy of a repeated
    # sentence to the FIRST index. That is safe -- it never shifts lines, which
    # was the v4 corruption bug -- but it is not sufficient: when a model emits
    # the same line three times, the judge flags all three and the fixer rewrites
    # copy #1 three times over while copies #2 and #3 survive untouched. Observed
    # for real on a cheap-preset run: "Feel the tiny bumps on your forearm."
    # appeared 3x in the finished script after four repair rounds.
    #
    # Each flagged issue now claims the next unclaimed occurrence.
    occurrences: dict[str, list[int]] = {}
    for idx, it in enumerate(tagged_sentences):
        occurrences.setdefault(it["sentence"], []).append(idx)

    targets: dict[int, list[str]] = {}
    for iss in issues:
        idx = None
        free = [i for i in occurrences.get(iss["sentence"], []) if i not in targets]
        if free:
            idx = free[0]
        elif iss["sentence"] in occurrences:
            # every copy already claimed - fold this reason into the first one
            idx = occurrences[iss["sentence"]][0]
        else:
            # fuzzy fallback - the judge occasionally normalises punctuation
            for text, idxs in occurrences.items():
                if iss["sentence"][:45] and iss["sentence"][:45] in text:
                    unclaimed = [i for i in idxs if i not in targets]
                    idx = unclaimed[0] if unclaimed else idxs[0]
                    break
        if idx is None:
            print(f"  [warn] could not locate flagged sentence, skipping: {iss['sentence'][:60]!r}")
            continue
        targets.setdefault(idx, []).append(iss["reason"])

    if not targets:
        return tagged_sentences

    lines = []
    for idx in sorted(targets):
        it = tagged_sentences[idx]
        src_tag = ",".join(map(str, it["sources"])) if it["sources"] else "0"
        lines.append(f'#{idx} [S{it["section"]}|{src_tag}] "{it["sentence"]}"')
        for r in targets[idx]:
            lines.append(f"     PROBLEM: {r}")
    flagged_block = "\n".join(lines)

    context_lines = []
    for idx in sorted(targets):
        before = tagged_sentences[idx - 1]["sentence"] if idx > 0 else "(start of script)"
        after = (tagged_sentences[idx + 1]["sentence"]
                 if idx + 1 < len(tagged_sentences) else "(end of script)")
        context_lines.append(f'#{idx} sits between: "{before}"  ...  "{after}"')
    context_block = "\n".join(context_lines)

    prompt = f"""{instruction}

You are editing individual lines inside a finished voiceover script. Each flagged line has
an id. Rewrite the line so it solves the stated problem while still reading naturally in
the gap it sits in.

ABSOLUTE CONSTRAINT: every factual thing in your rewrite must still be supported by the
source excerpts below. A punchier line that invents a detail is worse than the line you
started with. If nothing in the sources can support a version of the line, delete it.

BEAT SHEET (the line must still perform its section's job):
{b.render_beat_sheet()}

THE ONLY NUMBERS ALLOWED ANYWHERE IN THIS SCRIPT:
{b.numbers_block}

THE ONLY NAMES / INSTITUTIONS / YEARS ALLOWED:
{b.entities_block}

NEVER USE THESE PHRASES: {b.banned_block}

SOURCE EXCERPTS:
{b.sources_block}

WHERE EACH LINE SITS:
{context_block}

FLAGGED LINES:
{flagged_block}

OUTPUT FORMAT - one line per fix, nothing else, no commentary:
    #<id> | <rewritten sentence> [S<section>|<sources>]
or, to delete the line entirely:
    #<id> | DROP

Echo the id exactly. Keep the section number the same. Update the source numbers if your
rewrite now draws on a different source, or use 0 if it is now pure narration."""

    raw = call_model(config.MODEL_WRITER, prompt, max_tokens=4000, temperature=temperature)

    replacements: dict[int, dict | None] = {}
    for line in raw.splitlines():
        line = line.strip()
        m = re.match(r"^#(\d+)\s*\|\s*(.+)$", line)
        if not m:
            continue
        idx, body = int(m.group(1)), m.group(2).strip()
        if body.upper().startswith("DROP"):
            replacements[idx] = None
            continue
        tm = TAG_RE.match(body)
        if tm:
            replacements[idx] = {
                "sentence": tm.group(1).strip(),
                "section": int(tm.group(2)),
                "sources": [int(n) for n in re.findall(r"\d+", tm.group(3)) if int(n) != 0],
            }
        else:
            replacements[idx] = {
                "sentence": body,
                "section": tagged_sentences[idx]["section"] if idx < len(tagged_sentences) else 1,
                "sources": [],
            }

    out = []
    for idx, it in enumerate(tagged_sentences):
        if idx in replacements:
            rep = replacements[idx]
            if rep is not None:
                out.append(rep)
            # None -> deliberately dropped
        else:
            out.append(it)

    n_fixed = sum(1 for v in replacements.values() if v is not None)
    n_dropped = sum(1 for v in replacements.values() if v is None)
    n_missed = len(targets) - len(replacements)
    print(f"  {n_fixed} rewritten, {n_dropped} dropped, {n_missed} not returned by the model")
    return out


# =============================================================================
#  WHOLE-SECTION REBUILD
# =============================================================================
# Some problems can't be fixed one line at a time - a section 90 words over target,
# or missing a beat entirely, needs the whole section rebuilt against its spec. v4 had
# no mechanism for this at all; a blown word count just shipped.
# =============================================================================

def rewrite_section(tagged_sentences: list[dict], section_n: int, reasons: list[str],
                    b: Brief) -> list[dict]:
    spec = b.beat_sheet.get(section_n)
    if not spec:
        return tagged_sentences

    idxs = [i for i, it in enumerate(tagged_sentences) if it["section"] == section_n]
    if not idxs:
        start = next((i for i, it in enumerate(tagged_sentences) if it["section"] > section_n),
                     len(tagged_sentences))
        end = start
        current_block = "(this section is missing from the script entirely)"
    else:
        start, end = idxs[0], idxs[-1] + 1
        current_block = "\n".join(
            f'{it["sentence"]} [S{it["section"]}|{",".join(map(str, it["sources"])) or "0"}]'
            for it in tagged_sentences[start:end]
        )

    before = tagged_sentences[start - 1]["sentence"] if start > 0 else "(start of script)"
    after = tagged_sentences[end]["sentence"] if end < len(tagged_sentences) else "(end of script)"
    problems = "\n".join("  - " + r for r in reasons)

    prompt = f"""Rewrite ONE section of a voiceover script so it meets its spec.

PROBLEMS WITH THE CURRENT VERSION:
{problems}

THE SPEC THIS SECTION MUST MEET:
{b.render_section_spec(section_n)}

It must run straight on from: "{before}"
and hand off cleanly into: "{after}"

Keep every fact that is already grounded. Do not introduce any new factual claim.

THE ONLY NUMBERS ALLOWED:
{b.numbers_block}

THE ONLY NAMES / INSTITUTIONS / YEARS ALLOWED:
{b.entities_block}

NEVER USE THESE PHRASES: {b.banned_block}

VOICE RULES: second person; concrete nouns; one adjective per noun; never label a feeling
for the viewer; vary sentence length; never put two fact-carrying sentences side by side.

SOURCE EXCERPTS:
{b.sources_block}

CURRENT VERSION OF THE SECTION:
{current_block}

Return ONLY the rewritten section: one sentence per line, each ending with
[S{section_n}|<source numbers>] or [S{section_n}|0]. No commentary."""

    raw = call_model(config.MODEL_WRITER, prompt, max_tokens=3000, temperature=0.55)
    new_items = parse_tagged_script(raw)
    for it in new_items:
        it["section"] = section_n
    if not new_items:
        print(f"  [warn] section {section_n} rewrite returned nothing - keeping original")
        return tagged_sentences
    print(f"  section {section_n} rewritten: {len(new_items)} lines, "
          f"{sum(len(i['sentence'].split()) for i in new_items)} words")
    return tagged_sentences[:start] + new_items + tagged_sentences[end:]
