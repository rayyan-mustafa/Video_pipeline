"""
Step 7 -- read the sources and report only what is actually in them.

The profiler's output picks the shape of Section 3, which is the fix for the
biggest structural flaw in the v4 notebook: a beat sheet that demanded "the
university, year and volunteer group" for EVERY topic left the model two options
on a morphometrics topic -- fabricate an experiment, or dump methodology numbers
to fill the beat. It picked the second. See SECTION_3_TEMPLATES in config.py.
"""

from __future__ import annotations

import json

from . import config, paths, sources
from .engine import call_model_json


def _prompt(topic: str, sources_block: str) -> str:
    return f"""You are a research analyst preparing a brief for a science
scriptwriter. Read the source excerpts below about: {topic}

Your job is to report ONLY what is actually in these sources. Do not add background
knowledge. If the sources do not contain something, say so - that is a useful finding,
not a failure.

Produce this exact JSON shape, no markdown fences, no commentary:

{{
  "evidence_type": one of
      "controlled_experiment"   (volunteers/subjects, an intervention, measured outcomes),
      "observational_study"     (measurement or survey, numbers, but no intervention),
      "comparative_anatomy"     (fossils, species or population comparison, morphology),
      "clinical_case"           (patients, disorders, lesions, medical observation),
      "competing_hypotheses"    (no decisive study; two or more rival explanations argued),
      "mechanism_only"          (the physiology is explained, but no headline study),
  "evidence_type_reason": "one sentence on why you chose that",
  "named_entities": [
     {{"kind": "researcher|institution|study|journal|year", "value": "...", "source": 1}}
  ],
  "hard_numbers": [
     {{"value": "the number exactly as written in the source", "means": "what it measures", "source": 1}}
  ],
  "key_findings": [
     {{"finding": "one sentence in plain language", "source": 1, "certainty": "established|debated|speculative"}}
  ],
  "competing_explanations": [
     {{"explanation": "...", "source": 1}}
  ],
  "mechanism": "two or three sentences on the physical mechanism as the sources describe it, or null",
  "the_common_myth": "the widely-believed schoolbook explanation these sources push back on, or null",
  "the_best_counter_evidence": "the single most concrete, surprising, specific fact in these sources - the thing that would make someone stop scrolling",
  "gaps": ["things a scriptwriter might reach for that these sources do NOT support"]
}}

Rules:
- "hard_numbers" must contain ONLY numbers that literally appear in the excerpts. This
  list becomes the writer's entire allowance of statistics. If a number is not here, the
  writer may not use it. Be thorough - miss one and the script loses a real fact.
- "certainty" matters. Mark a finding "debated" if the source hedges it, attributes it to
  one team, or presents a rival view. The script must reproduce that hedging.
- If there is no named researcher/year in the sources, return an empty "named_entities"
  list rather than guessing. The script is built to work without one.

SOURCE EXCERPTS:
{sources_block}"""


def run(topic: str, source_pack: list[dict]) -> tuple[dict, dict, str]:
    """-> (profile, beat_sheet, evidence_type). Writes evidence_profile.json."""
    sources_block = sources.render_block(source_pack)

    print("Profiling what evidence the sources actually contain...")
    profile = call_model_json(config.MODEL_EXTRACTOR, _prompt(topic, sources_block),
                              max_tokens=4000, temperature=0)

    with open(paths.p("evidence_profile.json"), "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)

    beat_sheet, ev_type = config.build_beat_sheet(profile.get("evidence_type", "mechanism_only"))

    # The resolved beat sheet is written out because it CANNOT be rebuilt
    # downstream: Section 3 is chosen from the evidence, and SECTION_3_TEMPLATES
    # lives only in this package. The notebook's archive cell needs the section
    # names and word targets, so they cross the boundary as data rather than the
    # notebook keeping a second copy of the templates that can drift.
    with open(paths.p("beat_sheet.json"), "w", encoding="utf-8") as f:
        json.dump({"evidence_type": ev_type,
                   "sections": {str(k): v for k, v in beat_sheet.items()}}, f, indent=2)

    print("\n" + "=" * 70)
    print(f"EVIDENCE TYPE: {ev_type}")
    print(f"  reason: {profile.get('evidence_type_reason')}")
    print(f'  -> Section 3 template: "{beat_sheet[3]["name"]}"')
    print("=" * 70)

    print(f"\nNamed entities available: {len(profile.get('named_entities', []))}")
    for e in profile.get("named_entities", [])[:8]:
        print(f"  - {e.get('kind')}: {e.get('value')}  [S{e.get('source')}]")
    if not profile.get("named_entities"):
        print("  (none - the script will be written without a named pioneer, by design)")

    hard = profile.get("hard_numbers", [])
    print(f"\nHard numbers the writer is allowed to use: {len(hard)}")
    for h in hard:
        print(f"  - {h.get('value')}: {h.get('means')}  [S{h.get('source')}]")

    print(f"\nKey findings: {len(profile.get('key_findings', []))}")
    for k in profile.get("key_findings", []):
        print(f"  - [{k.get('certainty')}] {k.get('finding')}")

    print(f"\nStrongest hook material:\n  {profile.get('the_best_counter_evidence')}")
    if profile.get("gaps"):
        print("\nThings the sources do NOT support (writer is blocked from these):")
        for g in profile["gaps"]:
            print(f"  - {g}")

    paths.mark_done("evidence", ["evidence_profile.json", "beat_sheet.json"])
    return profile, beat_sheet, ev_type
