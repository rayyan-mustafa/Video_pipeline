"""
Steps 12 / 13 -- the retention gate, and the hook tournament.

=============================================================================
 WHY THE RETENTION GATE IS SEPARATE FROM THE GROUNDING GATE
=============================================================================
Fact-checking is binary and source-anchored: cite it or cut it. Retention is a
judgment about delivery. Merging them into one prompt lets the model trade
accuracy for punchiness to satisfy the fuzzier objective. Kept separate and run
second, this gate inherits clean sourced sentences and is only allowed to reshape
how they land.

The judge renders its criteria from the SAME beat sheet the writer was given
(Brief.render_section_spec), so it can only hold the script to rules the writer
actually received. That is why the beat sheet lives in a dict rather than being
typed out inside a prompt string.
"""

from __future__ import annotations

import json
import re

from . import config, paths, script as script_mod
from .brief import Brief
from .engine import call_model, call_model_json
from .factcheck import audit_untagged_claims, validate_claims

_BASE_SYSTEM = """You are a retention editor for a YouTube science channel with a
brutal standard: every sentence has to earn the next one. You are not fact-checking -
another pass already did that, and every claim here is sourced. Your only question is
whether a viewer keeps watching.

You are strict. A section that is merely competent gets flagged. You are looking for:

  - BEAT FAILURE: a beat from the spec is missing, out of order, or technically present
    but doing none of the work it was supposed to do.
  - PACING: fact-carrying sentences stacked together; a run of similar-length sentences;
    a paragraph where nothing changes register.
  - TELLING NOT SHOWING: the script naming an emotion instead of producing it; adjective
    stacking; "this is fascinating" instead of a fascinating detail.
  - DEAD WEIGHT: a sentence that restates the previous one, sets up something that never
    arrives, or could be deleted with no loss.
  - ABSTRACTION: a metaphor the viewer cannot physically picture; a mechanism described
    in category words instead of concrete nouns.
  - FLAT OPENINGS: sentences that begin with the same construction repeatedly.
  - NO ONE IN IT: the section is entirely second-person-somatic - "your body", "you" -
    and never lands on anyone else. A viewer will finish a video about their own body;
    they share a video about a person. Flag a section that explains a mechanism
    thoroughly and names nobody it is happening to. This is a judgement about REACH,
    not about warmth: do not flag a section for being unsentimental, and do not ask
    for a person where the sources support none. Section 1 is exempt - the cold hook
    is deliberately somatic and must stay that way.

Do not flag something just to have flagged it. If a line works, leave it. But do not
pass a section that a viewer would scrub past."""


def system_prompt(learnings: dict) -> str:
    """Real audience data, when it exists, is a stronger signal than any prior baked
    into the base prompt -- appended rather than interleaved so it stays visible as
    'what this channel's own viewers have actually done', not mixed into house style."""
    return _BASE_SYSTEM + f"""

LEARNED FROM THIS CHANNEL'S OWN PUBLISHED VIDEOS (real viewer retention data, not a prior):
{config.learnings_block(learnings, ["retention_notes"])}
"""


# =============================================================================
#  THE GATE
# =============================================================================

def judge_section(tagged_sentences: list[dict], section_n: int, b: Brief,
                  system: str) -> list[dict]:
    spec = b.beat_sheet.get(section_n)
    items = [it for it in tagged_sentences if it["section"] == section_n]
    if not spec or not items:
        return []

    idxs = [i for i, it in enumerate(tagged_sentences) if it["section"] == section_n]
    prev_line = (tagged_sentences[idxs[0] - 1]["sentence"] if idxs[0] > 0
                 else "(this is the opening of the video)")
    next_line = (tagged_sentences[idxs[-1] + 1]["sentence"]
                 if idxs[-1] + 1 < len(tagged_sentences) else "(this is the end of the video)")

    body = "\n".join(
        f'{it["sentence"]}   <{"FACT" if it["sources"] else "narration"}>'
        for it in items
    )
    words = sum(len(it["sentence"].split()) for it in items)

    seam_note = ""
    if section_n == 1:
        seam_note = ("This is the first 30-40 seconds of the video. It decides everything. Judge it "
                     "harder than any other section: if the first sentence does not make someone stop "
                     "scrolling, nothing else matters.")
    elif section_n == 4:
        seam_note = ("This section sits around the 4-minute mark, where the largest single block of "
                     "viewers leaves. Its first beat is a Re-Hook and it must genuinely reopen the "
                     "mystery, not just continue explaining.")
    elif section_n == 6:
        seam_note = ("This is the close. The final line has to be a real open question a viewer could "
                     "argue about - not a statement with a question mark, and never a call to action.")

    prompt = f"""Judge ONE section of a voiceover script about "{b.topic}".

THE SPEC THIS SECTION WAS WRITTEN TO:
{b.render_section_spec(section_n)}

{seam_note}

THE SEAM: this section must run on naturally from the line before it and hand off into
the line after it. Judge both joins.
  Line immediately before: "{prev_line}"
  Line immediately after:  "{next_line}"

THE SECTION AS WRITTEN ({words} words, target {spec['words'][0]}-{spec['words'][1]}).
Each line is marked FACT (carries a sourced claim) or narration:
{body}

Respond with ONLY valid JSON, no fences:
{{"verdict": "pass" or "fail",
  "issues": [{{"sentence": "the sentence verbatim, exactly as written above",
               "reason": "what is wrong and what would fix it, in one sentence"}}]}}

Quote each flagged sentence exactly. If the problem is a missing beat rather than a bad
line, quote the sentence where the missing beat should have gone and say so in the reason."""

    result = call_model_json(config.MODEL_JUDGE, prompt, system=system,
                             max_tokens=3000, temperature=0.2)
    issues = result.get("issues", [])
    for i in issues:
        i["kind"] = "retention"
        i["section"] = section_n
    return issues


def judge_whole_script(tagged_sentences: list[dict], b: Brief, system: str) -> list[dict]:
    """One pass over the arc, not the sections. Catches the failure that is invisible
    section by section: the video is fine everywhere and dull overall."""
    body = "\n".join(f'[S{it["section"]}] {it["sentence"]}' for it in tagged_sentences)
    prompt = f"""Read this complete voiceover script for "{b.topic}" end to end and judge it
as one viewing experience.

Look for problems that only show up at full length:
  - the same idea explained twice in different sections
  - a promise made early that the script never pays off
  - tension that peaks in the middle and then sags
  - a metaphor or image reused until it stops working
  - the middle third being where you would stop watching

Respond with ONLY valid JSON, no fences:
{{"issues": [{{"sentence": "verbatim line where the problem is worst", "reason": "..."}}]}}

Flag at most 6 issues, only real ones.

SCRIPT:
{body}"""
    result = call_model_json(config.MODEL_JUDGE, prompt, system=system,
                             max_tokens=2500, temperature=0.2)
    issues = result.get("issues", [])
    for i in issues:
        i["kind"] = "arc"
        i["section"] = 0
    return issues


def run_gate(tagged_sentences: list[dict], b: Brief, system: str) -> list[dict]:
    all_issues = []
    print("RETENTION GATE")
    print("-" * 70)
    for n in sorted(b.beat_sheet):
        issues = judge_section(tagged_sentences, n, b, system)
        name = b.beat_sheet[n]["name"]
        print(f"  S{n} {name:<24} {'PASS' if not issues else f'{len(issues)} issue(s)'}")
        for i in issues:
            print(f"      {i['reason']}")
            print(f"        > {i['sentence'][:75]}")
        all_issues.extend(issues)

    arc = judge_whole_script(tagged_sentences, b, system)
    print(f"  {'whole-script arc':<28} {'PASS' if not arc else f'{len(arc)} issue(s)'}")
    for i in arc:
        print(f"      {i['reason']}")
        print(f"        > {i['sentence'][:75]}")
    all_issues.extend(arc)
    print("-" * 70)
    print(f"  {len(all_issues)} total retention issue(s)\n")
    return all_issues


# =============================================================================
#  HOOK TOURNAMENT
# =============================================================================
# The first 30 seconds decide whether the other eight minutes are watched at all, and
# in the single-pass version they got exactly one attempt, same as every other section.
# That is the worst allocation of effort in the whole pipeline. This generates several
# genuinely different cold opens, has a judge rank them against each other, grounds the
# winner, and splices it in.
#
# Cheap: 1 generation call + 1 judging call + 1 validation call, no GPU.
# =============================================================================

N_HOOKS = 5

HOOK_ANGLES = [
    "Start from a physical sensation the viewer can produce in their own body in the next two seconds.",
    "Start from the single strangest concrete detail in the research, stated flatly with no setup.",
    "Start by stating the thing everyone believes, in the confident voice of someone who believes it, then break it.",
    "Start from an absence - something that should be there and is not, or should happen and does not.",
    "Start mid-scene, as though the viewer walked in on something already happening.",
]

GROUNDING_INSTRUCTION = (
    "These voiceover lines failed a grounding check: they either state something the cited "
    "source does not support, present a debated finding as settled, use a number that appears "
    "in no source, or smuggle a factual claim into a line marked as pure narration. Fix each "
    "one so it is fully supported, correctly tagged, and still reads well aloud."
)


def hook_tournament(tagged_sentences: list[dict], b: Brief, system: str,
                    n_hooks: int = N_HOOKS) -> list[dict]:
    hook_spec = b.render_section_spec(1)
    angles_block = "\n".join(f"{i+1}. {a}" for i, a in enumerate(HOOK_ANGLES[:n_hooks]))

    gen_prompt = f"""Write {n_hooks} completely different cold opens for a YouTube video
about: {b.topic}

Each one must satisfy this spec in full:

{hook_spec}

Use a different opening strategy for each. Strategy per candidate:
{angles_block}

These must be genuinely different openings - not one hook reworded {n_hooks} times. Different
first sentence, different route in, different image.

THE ONLY NUMBERS YOU MAY USE ANYWHERE:
{b.numbers_block}

THE ONLY NAMES / INSTITUTIONS / YEARS YOU MAY USE:
{b.entities_block}

THE MYTH TO BREAK: {b.profile.get('the_common_myth')}
THE STRONGEST CONCRETE DETAIL AVAILABLE: {b.profile.get('the_best_counter_evidence')}

NEVER USE THESE PHRASES: {b.banned_block}

Output format - exactly this, nothing else:

=== HOOK 1 ===
<sentence> [S1|<sources>]
<sentence> [S1|<sources>]
=== HOOK 2 ===
...

One sentence per line, every line tagged, no commentary."""

    print(f"Generating {n_hooks} candidate cold opens...")
    raw_hooks = call_model(config.MODEL_WRITER, gen_prompt, max_tokens=4000, temperature=0.95)

    candidates = []
    for block in re.split(r"===\s*HOOK\s*\d+\s*===", raw_hooks):
        block = block.strip()
        if not block:
            continue
        items = script_mod.parse_tagged_script(block)
        for it in items:
            it["section"] = 1
        if items:
            candidates.append(items)

    # keep the one we already have in the running - it was written to the same spec
    incumbent = [it for it in tagged_sentences if it["section"] == 1]
    if incumbent:
        candidates.append([dict(i) for i in incumbent])

    if not candidates:
        print("  no candidates parsed - keeping the existing hook")
        return tagged_sentences

    print(f"  {len(candidates)} candidates (including the current one)")

    listing = "\n\n".join(
        f"--- CANDIDATE {i+1} ---\n" + "\n".join(it["sentence"] for it in c)
        for i, c in enumerate(candidates)
    )

    judge_prompt = f"""Rank these candidate cold opens for a YouTube video about "{b.topic}".

Judge them the way a viewer does: the first sentence appears, and within about two seconds
they either keep watching or swipe. Nothing else about the video exists yet.

Rank on, in order of weight:
  1. STOP POWER of the very first sentence on its own. Does it create a specific gap the
     viewer needs closed? A generic opener loses no matter how good sentence four is.
  2. SPECIFICITY. Concrete, physical, checkable. Vague openers feel like every other video.
  3. THE SNAP. Is the negation genuinely 2-4 words and does it land hard?
  4. THE PROMISE. Does the last line make the next section feel unmissable?
  5. NOT SOUNDING WRITTEN. Any candidate that reads like narration rather than someone
     talking to one person should drop.

Be decisive. Do not rank them all as similar quality - say which one wins and why the
others lose.

Respond with ONLY valid JSON, no fences:
{{"winner": <candidate number>,
  "ranking": [<candidate numbers, best first>],
  "why_winner": "two sentences",
  "why_others_lost": "one sentence naming the single biggest weakness across the losers",
  "improvement": "one specific change that would make the winner better, or null"}}

{listing}"""

    print("  judging...")
    verdict = call_model_json(config.MODEL_JUDGE, judge_prompt, system=system,
                              max_tokens=1500, temperature=0.2)

    win_idx = int(verdict.get("winner", 1)) - 1
    win_idx = max(0, min(win_idx, len(candidates) - 1))
    winner = candidates[win_idx]

    print("\n" + "=" * 70)
    held = "  (the original hook held up)" if win_idx == len(candidates) - 1 and incumbent else ""
    print(f"WINNER: candidate {win_idx + 1}{held}")
    print("=" * 70)
    for it in winner:
        print(f"  {it['sentence']}")
    print(f"\n  why: {verdict.get('why_winner')}")
    print(f"  losers: {verdict.get('why_others_lost')}")
    if verdict.get("improvement"):
        print(f"  suggested tweak: {verdict['improvement']}")
    print(f"  ranking: {verdict.get('ranking')}")

    with open(paths.p("hook_tournament.json"), "w", encoding="utf-8") as f:
        json.dump({
            "verdict": verdict,
            "candidates": [[i["sentence"] for i in c] for c in candidates],
        }, f, indent=2)

    # --- splice the winner in ---
    rest = [it for it in tagged_sentences if it["section"] != 1]
    tagged_sentences = winner + rest

    # --- the winner is brand new text, so it has to clear the same gates ---
    print("\n  grounding the new hook...")
    hook_issues = validate_claims(winner, b) + audit_untagged_claims(winner, b)
    if hook_issues:
        print(f"  {len(hook_issues)} issue(s) in the winning hook - repairing")
        for i in hook_issues:
            print(f"    - {i['reason']}")
        tagged_sentences = script_mod.apply_fixes(
            tagged_sentences, hook_issues, GROUNDING_INSTRUCTION, b, temperature=0.3)
    else:
        print("  hook is clean.")

    script_mod.save_script(tagged_sentences, "hook spliced")
    s1_words = sum(len(i['sentence'].split()) for i in tagged_sentences if i['section'] == 1)
    print(f"\n  Section 1 is now {s1_words} words.")
    return tagged_sentences
