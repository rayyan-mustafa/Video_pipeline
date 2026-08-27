"""
Step 14b -- the humanizer gate. Turns a passing script into one a person wrote.

=============================================================================
 WHY THIS IS A THIRD GATE AND NOT MORE RULES IN THE OTHER TWO
=============================================================================
The grounding gate asks "is this true". The retention gate asks "will they keep
watching". Both can be fully satisfied by prose that is unmistakably machine
written, and on this pipeline they routinely are -- the retention fixer actively
pushes that way, because "make it land harder" is the instruction that produces
the em-dash reveal, the rule of three, and the parallel-Maybe ending.

Measured on a finished chin script that passed BOTH existing gates (933 words,
50 lines):

    lines containing an em dash ............ 15 / 50   (a human essay: 1-2)
    "not X, but Y" / "nothing more than" .... 7
    "a, b, and c" tricolons ................. 4
    cushion words (just/simply/really/...) .. 19
    parallel "Maybe... / Maybe..." close ..... yes
    banned phrases that shipped anyway ...... 2  ("unsettling", "strangely beautiful")

None of that is a fact problem and none of it is a retention problem. It is a
SHAPE problem, and shape is not something a phrase list can hold.

=============================================================================
 WHY IT RUNS BEFORE THE RE-GROUNDING PASS, NOT AFTER
=============================================================================
Same argument run_script.py already makes about the retention gate: this pass
rewrites sentences for texture, and texture is exactly the pressure that
produces a confident invented specific. Slotting it in front of the existing
mandatory re-grounding means it inherits that safety net for free rather than
needing one of its own.

    grounding -> hook tournament -> retention -> HUMANIZE -> re-grounding

=============================================================================
 WHY THE DETERMINISTIC SWEEP COMES FIRST
=============================================================================
Every tell above is countable. Counting is free and never has an off day, so the
paid judge only ever sees what genuinely needs judgment -- the same split
lint.py already makes against the fact-checker. In practice the audits below
find most of it and the judge exists for the one thing code cannot see: prose
that breaks no quota and still reads like nobody wrote it.
"""

from __future__ import annotations

import re
from collections import Counter

from . import config
from .brief import Brief
from .engine import call_model, parse_json_response

# =============================================================================
#  THE BUDGET
# =============================================================================
#  Tuned to a ~1,000-word / ~50-line video. These are the numbers the humanizer
#  gate enforces AND the numbers the writer is handed in its system prompt --
#  same contract Brief.render_section_spec keeps between the writer and the
#  retention judge. A gate that fails a script on a rule the writer never
#  received just burns fix rounds.
# =============================================================================
VOICE_QUOTAS = {
    "em_dash_max_lines": 3,        # lines containing an em dash, whole script
    "tricolon_max": 1,             # "a, b, and c" parallel lists
    "antithesis_max": 2,           # "not X but Y" / "nothing more than X"
    "imagine_max": 1,              # Picture... / Imagine... / Rewind...
    "cushion_per_1000": 6,         # just / simply / really / actually / quietly
    "repeat_max": 2,               # times one distinctive content word may appear
    "signpost_max_ratio": 0.20,    # share of lines opening on So/Then/But/And/Now
    "same_shape_max_run": 2,       # consecutive lines of identical construction
    "same_shape_max_ratio": 0.35,  # share of lines in any one construction
}

# Counted, not banned. One is voice; nineteen is a fingerprint.
CUSHION_WORDS = {
    "just", "simply", "really", "actually", "genuinely", "truly", "quietly",
    "merely", "entirely", "completely", "utterly", "precisely", "exactly",
    "somehow", "strangely", "oddly", "curiously", "surprisingly",
}

SIGNPOST_OPENERS = {
    "so", "then", "but", "and", "now", "here's", "picture", "imagine",
    "rewind", "meanwhile", "turns", "consider",
}

# Words that carry no fingerprint however often they appear.
_STOP = set(
    "the a an and or but of to in on it is are was were you your this that with for "
    "as at be been by from has have had not no its it's they them their we our us "
    "what when where which who how why if then than so up out one two do does did "
    "about into over under more most some any all can could would should will "
    "there here these those being other others still while where whose itself "
    "than very much many such each both".split()
)


# =============================================================================
#  POSITIVE OBLIGATIONS  --  injected into the writer's system prompt
# =============================================================================
#  Every existing VOICE RULE in script._system_prompt is a prohibition. That is
#  the whole problem: strip out everything a prohibition flags and what is left
#  is the safest available sentence, every time, and the safest available
#  sentence IS the machine register. These are the other half -- things the
#  script must CONTAIN.
# =============================================================================
HUMAN_VOICE_RULES = """1. HAVE AN OPINION. You are allowed to find one of these
   theories weak and say so, allowed to be unimpressed, allowed to think a study
   is thin. A script with no point of view reads as generated, because nothing
   generated has one.

2. LEAVE AT LEAST THREE SENTENCES PLAIN. No image, no reveal, no rhythm - the
   fact, flatly, then move on. Machine writing makes every sentence pull its
   weight. People do not, and the flat sentences are what make the sharp ones
   land.

3. SAY WHAT YOU DO NOT KNOW, IN YOUR OWN VOICE, AT LEAST ONCE. Not "scientists
   debate this" - "nobody has a good answer", "this part is a guess", "I have no
   idea either".

4. ONE CONCRETE DETAIL THAT PAYS OFF NOTHING. Include one specific true thing
   because it is interesting, not because the argument needs it. Every single
   detail serving the thesis is the texture of a summary, not of someone talking.

5. START AT LEAST TWO SENTENCES ON "And", "But" or "Because", AND LET AT LEAST
   TWO RUN SLIGHTLY UNTIDY - a trailing clause, an afterthought, a correction
   mid-sentence. Perfectly balanced sentences back to back is the tell.

6. VARY THE CONSTRUCTION, NOT ONLY THE LENGTH. Varied length with identical
   sentence shape is still monotony. Rotate: plain statement, question, fragment,
   sentence opening on a subordinate clause, direct address. Never three of the
   same shape in a row.

7. NO EM-DASH REVEALS AS A HABIT. "Statement - and here is the surprising part"
   is the most recognisable machine construction in English. Three lines in the
   whole script, maximum. Prefer a full stop and a new sentence.

8. ONE IMAGE PER SECTION, AND NEVER EXPLAIN IT. If you write "picture two
   buildings with a gap between them", do not then write "architects call that a
   spandrel, and your chin is exactly that". Trust the picture or cut it.

9. THE ENDING IS NOT A POEM. Do not close on two parallel "Maybe..." lines, do
   not close on an abstract noun, do not tell the viewer what it all meant. Ask
   the question and stop."""

# =============================================================================
#  VOICE SAMPLES  --  the only part of this file a model must not write
# =============================================================================
#  Style transfer from real human prose beats every rule above put together, and
#  it is the one input here that cannot be generated. A sample written by a
#  language model does not just fail to help: it hands the writer the exact
#  register this whole module exists to remove, with the gate's own authority
#  behind it. The audits would then measure the script against the thing they
#  were built to catch.
#
#  So samples live in a FILE you paste into, not in this source, and
#  check_voice_sample.py scores a candidate before it earns a place there.
#  Two rules for what goes in:
#
#    1. Provenance beats taste. The only guarantee that prose is human is that it
#       was written before language models could write it. Anything published
#       before ~2021 is safe by date alone; anything after needs a reason.
#    2. Not the competitor. Zenn's register is the one being copied here, and
#       vo_coach/data/scripts/mic_test_zenn.txt - written in that voice - runs a
#       tricolon and a cushion word inside 76 words, 13x the human rate for both.
#       Even if it were clean, matching a competitor's cadence is how you sound
#       like the competitor.
#
#  Empty is correct on a new channel, not a bug. The rules above are the backstop
#  and they work alone; samples make them unnecessary.
# =============================================================================
VOICE_SAMPLES_FILE = "voice_samples.txt"   # beside run_script.py, samples split on a --- line

# In-code override. Prefer the file: it keeps prose out of a source tree, and it
# means adding a sample is a paste rather than an edit.
VOICE_SAMPLES: list[str] = []


def voice_samples() -> list[str]:
    """The file if it exists, the constant otherwise. Silent when neither does."""
    from pathlib import Path
    if VOICE_SAMPLES:
        return list(VOICE_SAMPLES)
    f = Path(__file__).resolve().parent.parent / VOICE_SAMPLES_FILE
    if not f.exists():
        return []
    blocks = [b.strip() for b in re.split(r"^\s*-{3,}\s*$", f.read_text(encoding="utf-8"),
                                          flags=re.M)]
    return [b for b in blocks if b and not b.startswith("#")]

HUMANIZE_INSTRUCTION = (
    "These voiceover lines are factually clean and they hold attention, but they read as "
    "machine-written. Rewrite each one so a person would believe a person wrote it. Keep the "
    "same factual content exactly - you are changing sentence SHAPE, not substance. Break the "
    "em-dash reveal into two sentences. Kill the three-part list. Drop the cushion word instead "
    "of finding a better one. Where a line is doing something clever, consider whether the plain "
    "version is stronger; often it is. Do not add a fact, number, name or year that is not "
    "already in the line you are rewriting, and do not make the line longer."
)


def writer_block() -> str:
    """The half of the writer's system prompt that says what to DO. Rendered here
    rather than typed into script.py so the rules the writer receives and the
    rules this gate enforces can never drift apart."""
    q = VOICE_QUOTAS
    quotas = f"""  - em dashes: at most {q['em_dash_max_lines']} lines in the entire script
  - "a, b, and c" three-part lists: at most {q['tricolon_max']} in the entire script
  - "not X, but Y" / "nothing more than X": at most {q['antithesis_max']}
  - "Picture..." / "Imagine..." / "Rewind...": at most {q['imagine_max']}
  - two lines in a row opening on the same word: never
  - just / simply / really / actually / genuinely / quietly: at most {q['cushion_per_1000']} in the whole script
  - any one distinctive word (e.g. "strange", "real"): at most {q['repeat_max']} times total"""

    samples = ""
    loaded = voice_samples()
    if loaded:
        joined = "\n\n".join(f"  ---\n  {s.strip()}" for s in loaded)
        samples = f"""

 THE CADENCE TO MATCH
 -------------------------------------------------------------------------
 Human-written narration in the register this channel wants. Do not copy its
 content or its sentences. Match how it moves: where it stops, where it runs on,
 how often it does nothing clever.

{joined}"""

    return f"""=============================================================================
 HOW A PERSON WRITES THIS - obligations, not prohibitions
=============================================================================
The rules above tell you what to avoid. Avoiding everything lands you on the
safest possible sentence every time, and that register is itself the thing a
viewer recognises as machine-written. These say what the script must CONTAIN.

{HUMAN_VOICE_RULES}

 THE BUDGET - counted mechanically after you write, so stay inside it
 -------------------------------------------------------------------------
{quotas}{samples}"""


# =============================================================================
#  DETERMINISTIC SWEEP  --  free, runs before the judge
# =============================================================================

def _shape(sentence: str) -> str:
    """Classify a sentence by CONSTRUCTION. This is the check the existing linter
    is missing: audit_sentence_rhythm varies length, and a model satisfies it
    perfectly while writing fifty sentences of one identical shape."""
    s = sentence.strip()
    low = s.lower()
    if s.endswith("?"):
        return "question"
    if len(s.split()) <= 3:
        # "Not one." / "That's just wrong." A four-word sentence is a short
        # sentence, not a fragment, and calling it one made three ordinary plain
        # lines in a row look like a rhythm tic.
        return "fragment"
    if "—" in s or " -- " in s:
        return "dash-reveal"
    if re.match(r"^(when|if|as|while|after|before|because|since|though|although)\b", low):
        return "subordinate-first"
    if re.match(r"^(touch|feel|press|picture|imagine|rewind|look|watch|think|go ahead|try)\b", low):
        return "imperative"
    if re.match(r"^[A-Za-z']+ing\b", s) or re.match(r"^(in|on|at|for|from|by|under|over)\b.{0,40},", low):
        return "fronted-phrase"
    if "," in s and re.search(r",\s+(and|but|which|who|where)\b", low):
        return "compound"
    return "plain"


def audit_em_dashes(items: list[dict]) -> list[dict]:
    """The loudest single tell. 15 of 50 lines on the measured chin script."""
    hits = [it for it in items if "—" in it["sentence"] or " -- " in it["sentence"]]
    cap = VOICE_QUOTAS["em_dash_max_lines"]
    if len(hits) <= cap:
        return []
    # Flag the overflow only, worst first: the longest lines lean on it hardest.
    hits.sort(key=lambda it: -len(it["sentence"].split()))
    return [{
        "sentence": it["sentence"], "section": it["section"],
        "reason": (f"em-dash reveal ({len(hits)} lines in this script, budget {cap}) - "
                   f"split it into two sentences, or replace the dash with a full stop"),
    } for it in hits[:len(hits) - cap]]


def audit_tricolons(items: list[dict]) -> list[dict]:
    pat = re.compile(r"\b[\w'-]+,\s+[\w'-]+(\s+[\w'-]+)?,\s+and\s+[\w'-]+", re.I)
    hits = [it for it in items if pat.search(it["sentence"])]
    cap = VOICE_QUOTAS["tricolon_max"]
    return [{
        "sentence": it["sentence"], "section": it["section"],
        "reason": (f"rule-of-three list ({len(hits)} in this script, budget {cap}) - "
                   f"cut it to the one item that actually carries the point"),
    } for it in hits[cap:]]


def audit_antithesis(items: list[dict]) -> list[dict]:
    """'not X, but Y' and its family. One is rhetoric. Seven is a template."""
    pat = re.compile(
        r"\b(not just\b|not merely\b|nothing more than\b|wasn't just\b|isn't just\b|"
        r"it's not\b.{0,40}\bit's\b|not because\b.{0,40}\bbut because\b)", re.I)
    hits = [it for it in items if pat.search(it["sentence"])]
    cap = VOICE_QUOTAS["antithesis_max"]
    return [{
        "sentence": it["sentence"], "section": it["section"],
        "reason": (f"'not X, but Y' construction ({len(hits)} in this script, budget {cap}) - "
                   f"state the positive half on its own"),
    } for it in hits[cap:]]


def audit_anaphora(items: list[dict]) -> list[dict]:
    """Two consecutive lines opening on the same word. The parallel 'Maybe... /
    Maybe...' close is the single most recognisable LLM ending in existence."""
    findings = []
    for a, b in zip(items, items[1:]):
        wa = a["sentence"].split()[:1]
        wb = b["sentence"].split()[:1]
        if wa and wb and wa[0].lower().strip(",.") == wb[0].lower().strip(",."):
            if wa[0].lower().strip(",.") in _STOP and wa[0].lower() not in {"maybe", "every", "no", "not"}:
                continue
            findings.append({
                "sentence": b["sentence"], "section": b["section"],
                "reason": (f"opens on '{wb[0]}' immediately after the previous line did - "
                           f"parallel openings read as written-for-effect; recast this one"),
            })
    return findings


def audit_cushion_density(items: list[dict]) -> list[dict]:
    words = " ".join(it["sentence"] for it in items).lower()
    total = len(words.split()) or 1
    found = [w for w in re.findall(r"[a-z']+", words) if w in CUSHION_WORDS]
    budget = max(1, round(VOICE_QUOTAS["cushion_per_1000"] * total / 1000))
    if len(found) <= budget:
        return []
    common = Counter(found).most_common(3)
    worst = {w for w, _ in common}
    findings = []
    for it in items:
        low = it["sentence"].lower()
        hit = next((w for w in worst if re.search(rf"\b{re.escape(w)}\b", low)), None)
        if hit:
            findings.append({
                "sentence": it["sentence"], "section": it["section"],
                "reason": (f"cushion word '{hit}' ({len(found)} cushion words in this script, "
                           f"budget {budget}) - delete it rather than replacing it"),
            })
    return findings[:6]


def audit_word_repetition(items: list[dict], topic: str = "") -> list[dict]:
    """One distinctive word reused across sections is a fingerprint. The chin
    script ran strange/stranger/strangely five times without repeating a token,
    so stems are compared, not words.

    The topic's own words are exempt. A video called "Why Do We Have a Chin?"
    will say chin and human many times and must -- flagging those would send the
    fixer hunting for synonyms for the subject of the video, which is the one
    thing a human writer would never do."""
    exempt = {re.sub(r"(ly|ness|er|est|ing|ed|s)$", "", w)
              for w in re.findall(r"[a-z]+", topic.lower()) if len(w) > 3}
    stems: dict[str, list[dict]] = {}
    for it in items:
        seen = set()
        for w in re.findall(r"[a-z]+", it["sentence"].lower()):
            if w in _STOP or len(w) < 5:
                continue
            stem = re.sub(r"(ly|ness|er|est|ing|ed|s)$", "", w)
            if len(stem) < 4 or stem in seen or stem in exempt:
                continue
            seen.add(stem)
            stems.setdefault(stem, []).append(it)
    cap = VOICE_QUOTAS["repeat_max"]
    findings = []
    for stem, hits in stems.items():
        if len(hits) > cap:
            for it in hits[cap:]:
                findings.append({
                    "sentence": it["sentence"], "section": it["section"],
                    "reason": (f"'{stem}...' appears {len(hits)} times across this script "
                               f"(budget {cap}) - find a different word or cut the line"),
                })
    return findings[:6]


def audit_signposts(items: list[dict]) -> list[dict]:
    hits = [it for it in items
            if it["sentence"].split() and
            it["sentence"].split()[0].lower().strip(",.'") in SIGNPOST_OPENERS]
    ratio = len(hits) / max(1, len(items))
    if ratio <= VOICE_QUOTAS["signpost_max_ratio"]:
        return []
    return [{
        "sentence": it["sentence"], "section": it["section"],
        "reason": (f"opens on a signpost ('{it['sentence'].split()[0]}') - "
                   f"{ratio:.0%} of lines do, budget {VOICE_QUOTAS['signpost_max_ratio']:.0%}; "
                   f"start on the subject instead"),
    } for it in hits[int(len(items) * VOICE_QUOTAS["signpost_max_ratio"]):]]


def audit_construction_variety(items: list[dict]) -> list[dict]:
    """Length variety with identical shape is still monotony, and the existing
    rhythm check cannot see it. This is the deep version."""
    findings = []
    shapes = [_shape(it["sentence"]) for it in items]

    run = 1
    for i in range(1, len(shapes)):
        run = run + 1 if shapes[i] == shapes[i - 1] else 1
        # "plain" is exempt from the run check and from the ratio check below.
        # Three plain sentences in a row is not a defect, it is obligation 2 --
        # flagging it would have the fixer put the ornament straight back on.
        if shapes[i] != "plain" and run > VOICE_QUOTAS["same_shape_max_run"]:
            findings.append({
                "sentence": items[i]["sentence"], "section": items[i]["section"],
                "reason": (f"{run} '{shapes[i]}' sentences in a row - change the construction, "
                           f"not just the length"),
            })

    counts = Counter(shapes)
    for shape, n in counts.items():
        ratio = n / max(1, len(shapes))
        if shape != "plain" and ratio > VOICE_QUOTAS["same_shape_max_ratio"]:
            worst = next(it for it, s in zip(items, shapes) if s == shape)
            findings.append({
                "sentence": worst["sentence"], "section": worst["section"],
                "reason": (f"{ratio:.0%} of the script is the '{shape}' construction "
                           f"(budget {VOICE_QUOTAS['same_shape_max_ratio']:.0%}) - rebuild "
                           f"these lines on different shapes"),
            })
    return findings[:8]


def audit_poetic_close(items: list[dict]) -> list[dict]:
    """The last four lines, judged harder than anything else. A script that has
    been good for eight minutes still gives itself away in the outro."""
    tail = items[-4:]
    findings = []
    abstract = re.compile(
        r"\b(what was left|becoming human|the rest of you|everything else|"
        r"for any single reason|meant to be|part of you|what it means to be)\b", re.I)
    for it in tail:
        if abstract.search(it["sentence"]):
            findings.append({
                "sentence": it["sentence"], "section": it["section"],
                "reason": ("the close resolves into abstraction - end on the physical thing "
                           "the viewer can touch, not on what it all means"),
            })
    return findings


def run_audits(items: list[dict], topic: str = "", quiet: bool = False) -> list[dict]:
    checks = {
        "em-dash reveals": audit_em_dashes(items),
        "rule-of-three lists": audit_tricolons(items),
        "not-X-but-Y": audit_antithesis(items),
        "parallel openings": audit_anaphora(items),
        "cushion words": audit_cushion_density(items),
        "word repetition": audit_word_repetition(items, topic),
        "signpost openings": audit_signposts(items),
        "construction variety": audit_construction_variety(items),
        "poetic close": audit_poetic_close(items),
    }
    all_issues = []
    if not quiet:
        print("VOICEPRINT AUDIT (deterministic, no API call)")
        print("-" * 70)
    for name, issues in checks.items():
        for i in issues:
            i["kind"] = "voiceprint"
        if not quiet:
            print(f"  {name:<22} {'PASS' if not issues else f'{len(issues)} issue(s)'}")
            for iss in issues:
                print(f"      S{iss['section']}: {iss['reason']}")
                print(f"        > {iss['sentence'][:75]}")
        all_issues.extend(issues)
    if not quiet:
        print("-" * 70)
        print(f"  {len(all_issues)} total voiceprint issue(s)\n")
    return all_issues


# =============================================================================
#  THE JUDGE  --  one whole-script call, for what code cannot count
# =============================================================================
#  Deliberately ONE call, not one per section like the retention gate. The tells
#  this is looking for are global by nature (repetition, sameness, an ending that
#  performs), the audits above already do the line-level work for free, and a
#  seventh judge pass per script is real money for a second opinion on the same
#  fifty lines.
# =============================================================================

_JUDGE_SYSTEM = """You detect machine-written prose. You are shown a voiceover script
that is already accurate and already paced well - do not comment on either. Your only
question is whether a person wrote it.

You know what the tells are, because you produce them:

  - the em-dash reveal: a statement, a dash, then the interesting half
  - the rule of three, especially as a closing rhythm
  - "not X, but Y" and "nothing more than X"
  - every sentence pulling its weight; no plain sentences anywhere
  - a metaphor immediately explained rather than left alone
  - an ending that resolves into abstraction, or two parallel lines
  - cushion words: just, simply, really, actually, genuinely, quietly
  - perfect balance - no untidy sentence, no afterthought, no interruption
  - no opinion anywhere; a narrator with no stake in what they are saying
  - the same distinctive word reused across sections

Be specific and be hard to please. A line that reads fine on its own but is the
fourth of its shape in a row is a finding. Do not flag a line for being simple -
plain sentences are the point."""


def judge_voiceprint(items: list[dict], b: Brief, max_issues: int = 10) -> list[dict]:
    body = "\n".join(f'[S{it["section"]}] {it["sentence"]}' for it in items)
    prompt = f"""Read this finished voiceover script about "{b.topic}" and find the lines
that give away that a language model wrote it.

For each one, name the tell and say what a person would have written instead. Do not
rewrite the line yourself - the reason is an instruction for the editor who will.

Then answer one more question honestly: if you had to bet, is this script machine
written? Score 0-100, where 0 is unmistakably a person and 100 is unmistakably a model.

Respond with ONLY valid JSON, no fences:
{{"machine_score": <0-100>,
  "worst_habit": "the single construction this script leans on hardest",
  "issues": [{{"sentence": "the line verbatim, exactly as written below",
               "reason": "which tell, and what to do instead, in one sentence"}}]}}

Flag at most {max_issues} lines, only real ones. Quote each one exactly.

SCRIPT:
{body}"""

    result = parse_json_response(
        call_model(config.MODEL_JUDGE, prompt, system=_JUDGE_SYSTEM,
                   max_tokens=2500, temperature=0.2)
    )
    issues = result.get("issues", [])
    for i in issues:
        i["kind"] = "voiceprint-judge"
        i.setdefault("section", 0)
    score = result.get("machine_score")
    if score is not None:
        print(f"  machine score: {score}/100   worst habit: {result.get('worst_habit')}")
    return issues


def run_gate(items: list[dict], b: Brief, use_judge: bool = True) -> list[dict]:
    print("HUMANIZER GATE")
    print("=" * 70)
    issues = run_audits(items, b.topic)
    if use_judge:
        print("  asking the judge for what the counters cannot see...")
        judged = judge_voiceprint(items, b)
        print(f"  {len(judged)} judge issue(s)")
        for i in judged:
            print(f"      {i['reason']}")
            print(f"        > {i['sentence'][:75]}")
        issues += judged
    print("-" * 70)
    print(f"  {len(issues)} total humanizer issue(s)\n")
    return issues
