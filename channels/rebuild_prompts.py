"""Rebuild visual_prompt in an existing scenes_planned.json. No model calls.

Step 22 (run_scenes.py) decides the ACTION, the SHOT and the EXPRESSION by
calling a model, and those decisions cost money. The visual_prompt is then a
pure function of those three plus the constants in pipeline/config.py -- so when
only a constant changes (a reworded bible, a new LoRA trigger), re-running Step
22 would spend the API budget again AND re-roll every action, which destroys the
basis for any before/after comparison.

This rebuilds the prompt string in place from the fields already on disk. Same
actions, same shots, same expressions, same seed -- only the prompt text moves,
which is exactly what an A/B of a prompt-level change needs.

    python rebuild_prompts.py                 # rewrite in place, with a backup
    python rebuild_prompts.py --dry-run       # show the diff, touch nothing
"""

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import config, scenes as sc  # noqa: E402


def rebuild(path: str, dry_run: bool = False) -> int:
    with open(path, "r", encoding="utf-8") as f:
        planned = json.load(f)

    changed = 0
    for s in planned:
        before = s.get("visual_prompt", "")
        after = sc.visual_prompt(
            s.get("action_clause", ""),
            s.get("shot"),
            s.get("expression") or None,
        )
        if before != after:
            changed += 1
            if dry_run:
                print(f"scene {s['scene_index']:>3}  [{s.get('shot')}]")
                print(f"   - {before[:110]}")
                print(f"   + {after[:110]}")
        s["visual_prompt"] = after

    print(f"\n{changed} of {len(planned)} prompts change.")
    trigger = getattr(config, "LORA_TRIGGER", "")
    if trigger:
        n = sum(1 for s in planned if trigger in s["visual_prompt"])
        print(f"{n} of {len(planned)} carry the LoRA trigger {trigger!r} "
              f"(character frames only).")

    if dry_run:
        print("dry run - nothing written.")
        return changed

    backup = path + ".bak"
    shutil.copy(path, backup)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(planned, f, indent=2, ensure_ascii=False)
    print(f"wrote {path}  (backup at {backup})")
    return changed


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", default=os.environ.get("VIDEO_PIPELINE_WORKDIR", "_local_workdir"),
                    help="folder holding scenes_planned.json")
    ap.add_argument("--dry-run", action="store_true", help="show the diff, write nothing")
    args = ap.parse_args()

    target = os.path.join(args.workdir, "scenes_planned.json")
    if not os.path.exists(target):
        sys.exit(f"no scenes_planned.json in {args.workdir}")
    rebuild(target, args.dry_run)
