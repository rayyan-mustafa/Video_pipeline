import sys
import os
import json

# Add channels to path so we can import pipeline
sys.path.insert(0, r"f:\wierd_bio\channels")

from pipeline import script, config

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
    # Hack the int keys back for beat_sheet as json loads them as strings
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
    
    prompt = script._system_prompt(b)
    user_prompt = f"Write the full script for: {b.topic}"
    
    with open("prompt_output.md", "w", encoding="utf-8") as f:
        f.write("# System Prompt\n\n```text\n")
        f.write(prompt)
        f.write("\n```\n\n# User Prompt\n\n```text\n")
        f.write(user_prompt)
        f.write("\n```\n")

if __name__ == "__main__":
    main()
