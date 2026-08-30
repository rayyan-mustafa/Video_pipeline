import sys
import os
import json

# Add channels to path so we can import pipeline
sys.path.insert(0, r"f:\wierd_bio\channels")

from pipeline import script, config, evidence, factcheck, retention, humanize

def main():
    topic_key = "finger_pruning"
    topic_title = config.TOPICS[topic_key]["title"]
    
    # Load cached json
    workdir = r"f:\wierd_bio\channels\_local_workdir"
    with open(os.path.join(workdir, "source_pack.json"), "r", encoding="utf-8") as f:
        source_pack = json.load(f)
        
    with open(os.path.join(workdir, "evidence_profile.json"), "r", encoding="utf-8") as f:
        profile = json.load(f)
        
    with open(os.path.join(workdir, "beat_sheet.json"), "r", encoding="utf-8") as f:
        beat_sheet = json.load(f)
        
    ev_type = beat_sheet.get("evidence_type", "multi-source")
    beat_sheet = {int(k): v for k, v in beat_sheet["sections"].items()}
    
    from pipeline.brief import Brief
    learnings = config.load_learnings()
    
    b = Brief(
        topic=topic_title,
        source_pack=source_pack,
        profile=profile,
        beat_sheet=beat_sheet,
        banned=config.banned_phrases(learnings)
    )
    
    prompts = {}
    
    # 1. Extractor Prompt
    from pipeline import sources
    sources_block = sources.render_block(source_pack)
    prompts["1. EVIDENCE EXTRACTOR"] = evidence._prompt(b.topic, sources_block)
    
    # 2. Writer Prompt
    prompts["2. THE WRITER (First Draft)"] = script._system_prompt(b)
    
    # 3. Fact-Checker
    prompts["3a. FACT-CHECKER (validate_claims)"] = """You are a fact-checker. Below are numbered SOURCE EXCERPTS and a list of
CLAIMS, each tagged with which source(s) it is supposed to come from.

For each claim, check it ONLY against the source(s) it cites. A claim is unsupported if
that specific source does not directly support it. Paraphrase is fine. New specifics,
numbers, mechanisms, studies, names or years that are not in the cited source are NOT
supported, even if another source in the pack happens to cover them - that is a
citation error and counts as an issue.

Also flag a claim if the source presents it as debated, contested or one team's view but
the sentence states it as settled fact.

Respond with ONLY valid JSON, no fences:
{"issues": [{"sentence": "the sentence verbatim", "reason": "..."}]}

SOURCE EXCERPTS:
[SOURCES INJECTED HERE]

CLAIMS:
[SENTENCES INJECTED HERE]"""

    prompts["3b. FACT-CHECKER (audit_untagged_claims)"] = """These sentences from a science voiceover were marked as carrying NO factual
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
{"issues": [{"sentence": "verbatim", "reason": "states X as fact; sources do/do not support this"}]}

SOURCE EXCERPTS:
[SOURCES INJECTED HERE]

SENTENCES MARKED AS NON-FACTUAL:
[SENTENCES INJECTED HERE]"""

    
    # 4. Retention Judge
    prompts["4. RETENTION JUDGE"] = retention.system_prompt(learnings)
    
    # 5. The Fixer (Rewriter)
    prompts["5. THE FIXER (Applying Judge Feedback)"] = script._fix_system()
    
    # 6. The Humanizer
    prompts["6. THE HUMANIZER"] = humanize.HUMANIZE_INSTRUCTION
    
    out_path = r"C:\Users\HP\.gemini\antigravity-ide\brain\7af41bf1-8eb0-49d3-9703-1515d34a02c0\pipeline_prompts.md"
    
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Weird Bio - Full Pipeline Prompts\n\n")
        f.write("Here are the system prompts used by each AI agent in the pipeline. Note that some prompts (like the writer) are dynamically generated per-topic based on the specific constraints and beat sheet rules.\n\n")
        
        for name, p_text in prompts.items():
            f.write(f"## {name}\n\n```text\n{p_text}\n```\n\n")

if __name__ == "__main__":
    main()
