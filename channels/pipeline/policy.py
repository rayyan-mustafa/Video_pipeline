"""
Step 14c -- the platform/content-policy gate.

Every other gate here (lint, factcheck, retention, humanizer) asks whether the
script is TRUE, WELL-GROUNDED and WELL-WRITTEN. None of them ask whether it is
safe to PUBLISH - and this channel's whole subject matter is human biology, so
every topic brushes past health information a viewer could act on (Pierre
Robin sequence and a baby's airway, a pilot blacking out at a light
transition, nerve damage). This gate is the question the others don't ask.

Cheap and mechanical by design, same reasoning as MODEL_CHECKER: this is a
screen against a fixed checklist, not a subjective judgment call about prose,
so it runs on the cheap model rather than the writer/judge model.
"""

from __future__ import annotations

from . import config
from .brief import Brief
from .engine import call_model_json

POLICY_CHECKLIST = """\
1. UNQUALIFIED MEDICAL/HEALTH ADVICE - a line that tells the viewer what to do
   about a real symptom or condition ("if this happens to you, stop doing X" /
   "see a doctor for Y") rather than narrating how a body works in general.
   Explaining a mechanism is fine; instructing the viewer's own health
   decisions is not.
2. DIAGNOSTIC CLAIMS ABOUT THE VIEWER - a line that tells the viewer they
   personally have, or are at risk of, a specific condition.
3. DANGEROUS-ACTIVITY INSTRUCTION - a line that could read as a how-to for
   self-harm or for endangering someone else, even in passing or as a joke.
4. UNQUALIFIED CAUSATION ON A CONTESTED FINDING - a finding the evidence
   profile itself marks [debated] or hedged, stated in the script as settled
   fact instead of as "one theory" / "researchers think."
"""

POLICY_INSTRUCTION = (
    "Each of these lines was flagged by a platform-policy screen for stating "
    "unqualified health/medical advice, a diagnostic claim about the viewer, "
    "dangerous-activity instruction, or unhedged causation on a finding the "
    "sources mark as debated. Rewrite each line to fix the specific problem "
    "named in its reason - reframe as narration about the mechanism or finding "
    "rather than advice, diagnosis or settled fact, adding only the hedge the "
    "source actually supports. Keep the same facts and citation tag; change "
    "only what makes the line read as advice, diagnosis, instruction, or false "
    "certainty."
)


def _prompt(tagged_sentences: list[dict], b: Brief) -> str:
    body = "\n".join(f'[S{it["section"]}] {it["sentence"]}' for it in tagged_sentences)
    return f"""Read this voiceover script about "{b.topic}" for a general-audience
science/biology channel and flag ONLY lines that trip one of these categories -
nothing about writing quality or style, only these:

{POLICY_CHECKLIST}

FINDINGS THE SOURCES MARK AS DEBATED, for check 4 (reason strictly from this list -
do not invent hedging beyond what it says):
{b.findings_block}

Respond with ONLY valid JSON, no fences:
{{"issues": [{{"sentence": "the sentence verbatim", "category": "1-4", "reason": "..."}}]}}

Flag nothing if the script is clean - most scripts should return an empty list. Do not
flag narration that merely mentions a body part, symptom or condition; only flag advice,
diagnosis, instruction, or false certainty as defined above.

SCRIPT:
{body}"""


def run(tagged_sentences: list[dict], b: Brief) -> list[dict]:
    print("Screening for platform/content-policy issues...")
    result = call_model_json(config.MODEL_CHECKER, _prompt(tagged_sentences, b),
                             max_tokens=2000, temperature=0)
    issues = result.get("issues", [])
    for i in issues:
        i["kind"] = "policy"
        i.setdefault("section", 0)
    print(f"  {len(issues)} policy issue(s)")
    for i in issues:
        snippet = i["sentence"][:90] + ("..." if len(i["sentence"]) > 90 else "")
        print(f"    S{i['section']} [{i.get('category', '?')}]: {i['reason']}")
        print(f"      > {snippet}")
    return issues
