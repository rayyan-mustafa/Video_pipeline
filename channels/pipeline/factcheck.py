"""
Step 10 -- the grounding checks that need a model.

Two passes, and the second one is the important one:

  validate_claims()       checks every SOURCED sentence against only the source
                          it actually cites
  audit_untagged_claims() checks every [0]-tagged sentence for a smuggled claim

THE GAP v4 LEFT WIDE OPEN. v4's validator only looked at sentences tagged with a
source. But the beat sheet pushes Sections 5 and 6 to a very low stat ratio, so
roughly 40% of the finished script is tagged [0] and was never checked by
anything. A sentence like "your ancestors would have frozen to death without
this" is a hard factual claim about the world, and tagged [0] it sailed straight
through to the voiceover.
"""

from __future__ import annotations

from . import config
from .brief import Brief
from .engine import call_model, parse_json_response


def validate_claims(tagged_sentences: list[dict], b: Brief) -> list[dict]:
    """Checks every source-carrying sentence against ONLY the source(s) it cites."""
    claim_items = [i for i in tagged_sentences if i["sources"]]
    if not claim_items:
        return []

    claims_block = "\n".join(
        f'{i+1}. "{it["sentence"]}" - cites Source(s) {it["sources"]}'
        for i, it in enumerate(claim_items)
    )
    prompt = f"""You are a fact-checker. Below are numbered SOURCE EXCERPTS and a list of
CLAIMS, each tagged with which source(s) it is supposed to come from.

For each claim, check it ONLY against the source(s) it cites. A claim is unsupported if
that specific source does not directly support it. Paraphrase is fine. New specifics,
numbers, mechanisms, studies, names or years that are not in the cited source are NOT
supported, even if another source in the pack happens to cover them - that is a
citation error and counts as an issue.

Also flag a claim if the source presents it as debated, contested or one team's view but
the sentence states it as settled fact.

Respond with ONLY valid JSON, no fences:
{{"issues": [{{"sentence": "the sentence verbatim", "reason": "..."}}]}}

SOURCE EXCERPTS:
{b.sources_block}

CLAIMS:
{claims_block}"""

    result = parse_json_response(
        call_model(config.MODEL_CHECKER, prompt, max_tokens=3000, temperature=0))
    issues = result.get("issues", [])
    for i in issues:
        i["kind"] = "fact"
    return issues


def audit_untagged_claims(tagged_sentences: list[dict], b: Brief) -> list[dict]:
    """Reads every [0] sentence and asks one question: does this assert something
    checkable? If yes, it has to be either grounded and re-tagged, or softened into
    genuine narration."""
    zero_items = [i for i in tagged_sentences if not i["sources"]]
    if not zero_items:
        return []

    block = "\n".join(f'{i+1}. "{it["sentence"]}"' for i, it in enumerate(zero_items))
    prompt = f"""These sentences from a science voiceover were marked as carrying NO factual
claim - pure narration, direct address, rhetorical question, or imagery.

Your job is to find the ones that are actually smuggling a factual assertion. A sentence
carries a factual claim if a reasonable viewer could ask "is that true?" and expect an
answer from evidence.

Carries a claim (FLAG IT):
  - "Your ancestors relied on this to survive freezing nights."
  - "This reflex disappears completely after the age of forty."
  - "Every mammal on earth does this."

Does not carry a claim (LEAVE IT):
  - "Run your hand over your forearm."      (a directive)
  - "So why does it still happen?"          (a question)
  - "You are sitting in a heated room."     (about the viewer's setting, not a claim about the world)
  - "It feels like nothing at all."         (subjective description)

For each flagged sentence, also check it against the SOURCE EXCERPTS and say whether the
sources actually support it.

Respond with ONLY valid JSON, no fences:
{{"issues": [{{"sentence": "verbatim", "reason": "states X as fact; sources do/do not support this"}}]}}

SOURCE EXCERPTS:
{b.sources_block}

SENTENCES MARKED AS NON-FACTUAL:
{block}"""

    result = parse_json_response(
        call_model(config.MODEL_CHECKER, prompt, max_tokens=3000, temperature=0))
    issues = result.get("issues", [])
    for i in issues:
        i["kind"] = "smuggled-claim"
    return issues


def run(tagged_sentences: list[dict], b: Brief, quiet: bool = False) -> tuple[list, list]:
    """-> (fact_issues, smuggled_issues)."""
    if not quiet:
        print("Fact-checking cited claims...")
    fact_issues = validate_claims(tagged_sentences, b)
    if not quiet:
        print(f"  {len(fact_issues)} unsupported claim(s)")
        for i in fact_issues:
            print(f"    - {i['sentence'][:70]}...\n        {i['reason']}")

        print("\nAuditing [0]-tagged lines for smuggled factual claims...")
    smuggled_issues = audit_untagged_claims(tagged_sentences, b)
    if not quiet:
        print(f"  {len(smuggled_issues)} smuggled claim(s)")
        for i in smuggled_issues:
            print(f"    - {i['sentence'][:70]}...\n        {i['reason']}")

    return fact_issues, smuggled_issues
