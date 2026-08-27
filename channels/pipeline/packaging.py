"""
Step 15 -- YouTube packaging.

v4 wrote a source list and stopped there. On a channel like this, packaging is not
a nice-to-have downstream of the script: title and thumbnail decide whether the
script is ever heard at all. Generated after the final script exists, so the title
promises something the video actually delivers.

packaging.json is also read by the notebook's thumbnail step (it pulls
thumbnail_text) and by the upload step, so this is a genuine cross-machine
handoff file, not just a local artifact.
"""

from __future__ import annotations

import json

from . import config, paths
from .brief import Brief
from .engine import call_model, parse_json_response


def _prompt(b: Brief, hook_lines: str, learnings: dict,
            tagged_sentences: list[dict]) -> str:
    title_rules = "\n".join("    - " + r for r in config.TITLE_FORMULA["rules"])
    stake_rules = "\n".join("    - " + r for r in config.TITLE_FORMULA["stake_rules"])
    stake_examples = "; ".join(f'"{e}"' for e in config.TITLE_FORMULA["stake_examples"])
    n_stake = config.TITLE_MIX["stake"]
    n_cur = config.TITLE_MIX["curiosity"]

    two_am = config.TWO_AM_TEST
    two_am_block = "\n".join(
        ["THE 2AM TEST - every title faces this, whatever its shape:",
         f"  {two_am['question']}",
         "",
         "  PASSES:"] +
        [f"    - {p}" for p in two_am["passes"]] +
        ["", "  FAILS:"] +
        [f"    - {f}" for f in two_am["fails"]]
    )

    # Section 2 carries the Human Stake beat. The stake the TITLE may use is the
    # one the SCRIPT actually made - fact-checked and grounded - not the research
    # lead in TOPICS. If the writer skipped the beat for want of a source, there
    # is nothing here, and the block below tells the model to drop to 8 curiosity
    # titles rather than invent a population.
    section2 = " ".join(i["sentence"] for i in tagged_sentences if i["section"] == 2)
    stake_block = f"""
THE HUMAN STAKE, AS THE SCRIPT ITSELF TELLS IT (Section 2 verbatim). {"All" if n_cur == 0 else n_stake} of your
eight titles are built from this, so read it closely. If no group of real people
appears here, do NOT invent one: say so with "stake_titles_possible": false and
give 8 curiosity titles instead.
{section2 if section2 else "(section 2 is empty - produce 8 curiosity titles)"}
"""

    # With TITLE_MIX["curiosity"] == 0 the curiosity formula is not being used
    # this run, so the whole block comes out of the prompt. Left in, it renders
    # as "0 of the 8 titles are curiosity-led, and at least 2 of those must
    # follow this exact shape" - a self-contradicting instruction that costs
    # tokens and invites the model to split the difference.
    if n_cur > 0:
        curiosity_block = f"""
TITLE FORMULA THIS CHANNEL IS TESTING (borrowed from a channel with far larger reach -
imitate the *shape*, never the topic):
  Formula: {config.TITLE_FORMULA['pattern']}
  Reference example that works: {config.TITLE_FORMULA['reference_examples'][0]}
  Rules:
{title_rules}
{n_cur} of the 8 titles are curiosity-led, and at least 2 of those must follow this exact
shape. The rest of the curiosity ones stay free to use the angles further down.
"""
    else:
        curiosity_block = """
NO CURIOSITY-LED TITLES THIS RUN. All 8 are stake-led (config.TITLE_MIX). Do not
produce a mechanism question as a spare - every one of the 8 names real people.
The only exception is the fallback below, when the script supports no group at all.
"""

    return f"""You are packaging a YouTube video for a channel called
"Weird Human Biology". The channel takes one ordinary thing about the human body,
shows the viewer it is stranger than they assumed, and then shows who it is quietly
happening to - the person or group living with it unnoticed. Strangeness earns the
click; the human stake earns the watch-through and the comment.

Video topic: {b.topic}
The video's cold open: {hook_lines}
The single most striking fact in it: {b.profile.get('the_best_counter_evidence')}
The myth it overturns: {b.profile.get('the_common_myth')}

{two_am_block}

Shape decides the FORM of a title. The 2am test decides whether it is worth shipping.
A title that fails at 2am is cut and rewritten, no matter how well it fits its shape.

{curiosity_block}
THE OTHER {n_stake} ARE STAKE-LED, AND THEY ARE THE MAJORITY ON PURPOSE.
The reference channel's biggest videos are not curiosity questions - they are about
people, and that shape is expected to win the click. Every video still ships both shapes
so the published numbers can settle it. Exactly {n_stake}:
  Shape: {config.TITLE_FORMULA['stake_pattern']}
  Examples of the shape: {stake_examples}
  Rules:
{stake_rules}
{stake_block}

LEARNED FROM THIS CHANNEL'S OWN PUBLISHED VIDEOS AND COMPETITOR DATA (real performance,
not a prior - empty unless you have hand-written a learnings.json):
  Titles:
{config.learnings_block(learnings, ["title_notes"], n=8)}
  Thumbnails:
{config.learnings_block(learnings, ["thumbnail_notes"], n=8)}

Produce ONLY valid JSON, no fences:
{{
  "stake_titles_possible": true or false,
  "titles": [
    {{"text": "...",
      "shape": "curiosity" or "stake",
      "angle": "curiosity gap | contradiction | second person | number | question | human stake",
      "two_am": "one line: why a tired person clicks this at 2am instead of sleeping",
      "why": "one line on who clicks this and why"}}
  ],
  "thumbnail_text": [
    {{"text": "3-5 words maximum, readable at phone size", "visual": "what the stickman is doing"}}
  ],
  "description": "the full YouTube description. Open with two sentences that restate the hook and would make a browser click from the search page. Then a plain-language summary of what the video covers. No hashtag spam. No 'in this video we will'.",
  "chapters": [{{"label": "short chapter title", "section": 1}}],
  "tags": ["15-20 search tags, mixed broad and specific"],
  "pinned_comment": "one comment that asks the audience the video's open question in a way that starts an argument, without saying 'comment below'",
  "community_post": "one short post to publish the day before, teasing the question without answering it"
}}

Rules for titles - produce exactly 8:
  - Every title must pass the 2am test above, and "two_am" must say why in one line.
    If you cannot write that line honestly, the title is not good enough - rewrite it.
  - Mark each one "shape": "curiosity" or "stake". Exactly {n_cur} curiosity and {n_stake}
    stake, unless "stake_titles_possible" is false, in which case 8 curiosity and 0 stake.
  - Under 60 characters so nothing truncates on mobile.
  - Each one a genuinely different angle. Do not give me eight rewordings of the same title.
  - No clickbait the video does not pay off. Every title must be true.
  - No ALL CAPS words, no "SHOCKING", no "you won't believe".
  - At least two must work as a question, at least two as a flat statement of the weird fact.

Rules for thumbnail_text - produce exactly 4. Under 5 words. It must NOT repeat the title;
it should complete it or contradict it.

Chapters: one per section of the script, in order, labelled the way a curious viewer would
scan them - never "Introduction" or "Conclusion"."""


_BANNED_TITLE_WORDS = ("shocking", "you won't believe", "you wont believe",
                       "this is why", "mind-blowing", "insane", "gone wrong")


def validate(pack: dict) -> list[str]:
    """-> a list of problems, empty if the packaging is well-formed.

    Same shape as shots.validate(): soft problems come back as strings for the
    caller to print. The ONE unsafe case raises instead -- see below.

    Everything here is checkable in code. 'Is this title any good' is not, and
    is deliberately not attempted: the 2am test lives in the prompt, and the
    only thing enforced here is that the model actually answered it.
    """
    problems = []
    titles = pack.get("titles", [])

    if len(titles) != 8:
        problems.append(f"{len(titles)} titles, expected 8")

    stake = [t for t in titles if t.get("shape") == "stake"]
    cur = [t for t in titles if t.get("shape") == "curiosity"]
    untagged = [t for t in titles if t.get("shape") not in ("stake", "curiosity")]

    # ---- the hard one ---------------------------------------------------
    # stake_titles_possible is false when the script named no real group,
    # because the sources did not support one. A stake title in that case is a
    # population the model invented. On a health channel that is not a style
    # slip, it is a fabricated patient in the most visible line of the video --
    # so it stops the build rather than printing a warning nobody reads.
    if pack.get("stake_titles_possible") is False and stake:
        raise ValueError(
            f"{len(stake)} stake-led title(s) produced, but the script names no real "
            f"group (stake_titles_possible=false). These describe people the sources "
            f"do not support: {[t.get('text') for t in stake]}. "
            f"Add a stake-bearing source to this topic in config.TOPICS, or ship the "
            f"8 curiosity titles."
        )

    if untagged:
        problems.append(f"{len(untagged)} title(s) have no \"shape\" tag - the A/B cannot be read")

    if pack.get("stake_titles_possible") is not False:
        want_s, want_c = config.TITLE_MIX["stake"], config.TITLE_MIX["curiosity"]
        if len(stake) != want_s or len(cur) != want_c:
            problems.append(
                f"split is {len(cur)} curiosity / {len(stake)} stake, "
                f"expected {want_c} / {want_s} (config.TITLE_MIX)")

    for t in titles:
        text = (t.get("text") or "").strip()
        if not text:
            problems.append("a title is empty")
            continue
        if len(text) > 60:
            problems.append(f"{len(text)} chars, truncates on mobile: {text!r}")
        if not (t.get("two_am") or "").strip():
            problems.append(f"no 2am justification, so the gate did not run: {text!r}")
        if t.get("shape") == "stake" and text.endswith("?"):
            problems.append(f"stake titles are flat statements, not questions: {text!r}")
        low = text.lower()
        for w in _BANNED_TITLE_WORDS:
            if w in low:
                problems.append(f"banned phrase {w!r} in: {text!r}")
        if text.isupper():
            problems.append(f"all caps: {text!r}")

    # Eight rewordings of one title is one option, not eight. Compared on word
    # sets so "Why Your Fingers Wrinkle" and "Why Fingers Wrinkle" collide.
    seen = {}
    for t in titles:
        key = frozenset((t.get("text") or "").lower().replace("?", "").split())
        if key and key in seen:
            problems.append(f"near-duplicate of {seen[key]!r}: {t.get('text')!r}")
        seen[key] = t.get("text")

    thumbs = pack.get("thumbnail_text", [])
    if len(thumbs) != 4:
        problems.append(f"{len(thumbs)} thumbnail options, expected 4")
    for th in thumbs:
        txt = (th.get("text") or "").strip()
        if len(txt.split()) > 5:
            problems.append(f"thumbnail text over 5 words: {txt!r}")

    return problems


def run(tagged_sentences: list[dict], b: Brief, learnings: dict) -> dict:
    hook_lines = " ".join(i["sentence"] for i in tagged_sentences if i["section"] == 1)

    print("Generating packaging...")
    pack = parse_json_response(
        call_model(config.MODEL_WRITER, _prompt(b, hook_lines, learnings, tagged_sentences),
                   max_tokens=4000, temperature=0.8)
    )

    # Saved BEFORE validation on purpose: if validate() raises, the model call is
    # already paid for and the output is on disk to inspect. Losing it to an
    # exception would mean re-running the call just to see what went wrong.
    with open(paths.p("packaging.json"), "w", encoding="utf-8") as f:
        json.dump(pack, f, indent=2)

    issues = validate(pack)
    if issues:
        print("\n" + "-" * 70)
        print(f"PACKAGING: {len(issues)} problem(s) - packaging.json is written, review before use")
        for p in issues:
            print(f"  - {p}")
        print("-" * 70)

    # Grouped by shape so the A/B is visible at the moment of choosing. Picking
    # 8 curiosity titles in a row silently kills the test, and nothing else in
    # the pipeline would ever tell you that happened.
    titles = pack.get("titles", [])
    cur = [t for t in titles if t.get("shape") != "stake"]
    stk = [t for t in titles if t.get("shape") == "stake"]

    print("\n" + "=" * 70)
    print("TITLE OPTIONS")
    print("=" * 70)
    print("\n  CURIOSITY-LED (house style)\n")
    for i, t in enumerate(cur, 1):
        print(f"{i:>2}. {t['text']}  ({len(t['text'])} chars)")
        print(f"    [{t.get('angle')}] {t.get('why')}")
        print(f"    2am: {t.get('two_am')}")

    if stk:
        print("\n  STAKE-LED (the challenger - ship one of these sometimes, "
              "or the A/B never runs)\n")
        for i, t in enumerate(stk, len(cur) + 1):
            print(f"{i:>2}. {t['text']}  ({len(t['text'])} chars)")
            print(f"    [{t.get('angle')}] {t.get('why')}")
            print(f"    2am: {t.get('two_am')}")
    elif pack.get("stake_titles_possible") is False:
        print("\n  STAKE-LED: none. The script names no real group, so there is")
        print("  nothing to build one from. That is correct behaviour, not a bug -")
        print("  add a stake-bearing source to this topic in config.TOPICS.\n")

    print("\n" + "=" * 70)
    print("THUMBNAIL TEXT")
    print("=" * 70)
    for t in pack.get("thumbnail_text", []):
        print(f'  "{t["text"]}"  -  stickman: {t.get("visual")}')

    # --- Description with real sources appended ---
    description = pack.get("description", "")
    description += "\n\nSources for this video:\n"
    for i, s in enumerate(b.source_pack):
        description += f"{i+1}. {s['title']}\n   {s['url']}\n"
    description += "\nChapters:\n(timestamps filled in after upload)\n"
    for c in pack.get("chapters", []):
        description += f"  {c.get('label')}\n"

    with open(paths.p("youtube_description.txt"), "w", encoding="utf-8") as f:
        f.write(description)
    with open(paths.p("youtube_tags.txt"), "w", encoding="utf-8") as f:
        f.write(", ".join(pack.get("tags", [])))
    with open(paths.p("pinned_comment.txt"), "w", encoding="utf-8") as f:
        f.write(pack.get("pinned_comment", ""))

    print("\n" + "=" * 70)
    print("DESCRIPTION")
    print("=" * 70)
    print(description)
    print("=" * 70)
    print(f"TAGS: {', '.join(pack.get('tags', []))}")
    print(f"\nPINNED COMMENT: {pack.get('pinned_comment')}")
    print(f"\nCOMMUNITY POST: {pack.get('community_post')}")
    print("\nWritten: packaging.json, youtube_description.txt, youtube_tags.txt, pinned_comment.txt")

    paths.mark_done("packaging", ["packaging.json", "youtube_description.txt",
                                  "youtube_tags.txt", "pinned_comment.txt"])
    return pack
