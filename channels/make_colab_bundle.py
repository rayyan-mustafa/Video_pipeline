"""Zip everything the voice stage needs, so it can run on a Colab GPU.

    python make_colab_bundle.py --project wierdbiology/chin

WHY ONLY THE VOICE STAGE GOES TO COLAB. Of the three stages, only this one is
compute-heavy, and only because large-v3 is. Planning is arithmetic and muxing is
libx264 on CPU - Colab bills wall-clock, so encoding there costs compute units
for work the laptop does identically (the notebook's own Step 25 says exactly
this). Send the transcription, keep the rest.

WHAT COMES BACK. scenes_timeline.json, words.json and full_vo.wav. Those three
are the entire contract between this stage and the next: the timeline says when
each sentence is spoken, words.json gives every word's exact timestamp, and
full_vo.wav is the mastered audio the mux reads. Nothing else needs to travel.

THE MASTERING GOES WITH IT, and that is not optional. The "clip" profile trims
silence, which moves every timestamp, so alignment has to see the audio that
actually ships. Mastering locally and transcribing remotely would align against
audio that no longer exists.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pipeline.project import Project

# The pipeline modules build_video.py actually imports at module load time - ALL
# of them, even the ones the voice stage itself never calls, because `import
# build_video` runs every top-level import in the file before build_voice() is
# even reachable. imageplan.py belongs to the plan/mux stages, not voice, but its
# absence here is exactly the bug this comment used to not warn about: the whole
# module failed to import on Colab because one entry was missing.
#
# Kept explicit rather than zipping the whole package: draw.py, props.py and
# compositor.py pull torch and PIL for generated-frame work that has nothing to
# do with this path, and are legitimately fine to leave out.
MODULES = ("__init__.py", "voice.py", "align.py", "scenesplit.py", "project.py",
          "imageplan.py")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--full", action="store_true",
                    help="also pack the images and script_plan.json, so the whole "
                         "build (voice -> plan -> mux) can run remotely and return "
                         "a finished mp4 instead of just the voice artefacts. "
                         "Only the images the plan actually references are packed, "
                         "so regenerated alternates do not inflate the upload.")
    args = ap.parse_args()

    proj = Project.parse(args.project)
    here = Path(__file__).parent
    out = args.out or (proj.dir / f"{proj.topic}_voice_bundle.zip")

    recs = sorted(p for p in proj.recordings.iterdir()
                  if p.suffix.lower() in (".wav", ".flac", ".m4a", ".mp3")) \
        if proj.recordings.is_dir() else []
    sects = sorted(proj.sections.glob("section_[0-9]*.txt"))

    if not recs:
        raise SystemExit(f"no recordings in {proj.recordings}")
    if not sects:
        raise SystemExit(f"no read sheets in {proj.sections} - run plan_script.py")

    # The split mode has to travel too. The timeline built in Colab must use the
    # same one the plan did, or scene N stops meaning the same thing and the
    # fingerprint check rejects the result after the GPU time is already spent.
    split = "chunk"
    if proj.plan.exists():
        import json
        split = json.loads(proj.plan.read_text(encoding="utf-8")).get("split", "chunk")

    # "match" mode has no rule to reapply - its scenes are exactly what
    # script_plan.json recorded, one per commissioned image (see scenesplit.py's
    # match_scenes). So build_voice() cannot regenerate them from the script text
    # on its own; the plan's scene list has to travel too, or the remote side
    # cannot reconstruct the same 105 scenes the local plan already fixed.
    plan_scenes = None
    if split == "match":
        if not proj.plan.exists():
            raise SystemExit(f"split mode is 'match' but {proj.plan} does not "
                             f"exist - re-run plan_script.py first")
        import json
        plan_scenes = json.loads(proj.plan.read_text(encoding="utf-8"))["scenes"]

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for m in MODULES:
            src = here / "pipeline" / m
            if src.exists():
                z.write(src, f"pipeline/{m}")
            elif m == "__init__.py":
                z.writestr("pipeline/__init__.py", "")
        z.write(here / "build_video.py", "build_video.py")
        for f in sects:
            z.write(f, f"sections/{f.name}")
        for f in recs:
            z.write(f, f"voice_recordings/{f.name}")
        z.writestr("SPLIT.txt", split)
        if plan_scenes is not None:
            import json as _json
            z.writestr("plan_scenes.json", _json.dumps(plan_scenes))

        if args.full:
            if not proj.plan.exists():
                raise SystemExit(f"--full needs {proj.plan} - run plan_script.py")
            z.write(proj.plan, "build/script_plan.json")
            import json as _json
            planned = _json.loads(proj.plan.read_text(encoding="utf-8"))["images"]
            for img in planned:
                src = proj.images / img["filename"]
                if not src.exists():
                    raise SystemExit(f"plan references a missing image: {src}")
                z.write(src, f"images/{img['filename']}")
            print(f"  packed {len(planned)} planned images "
                  f"(the {len(list(proj.images.iterdir())) - len(planned)} "
                  f"unused alternates were left out)")

    mb = out.stat().st_size / 1e6
    print(f"  {out}  ({mb:.1f} MB)")
    print(f"  {len(sects)} read sheet(s), {len(recs)} recording(s), split={split!r}")
    print()
    print("  Next: upload this zip to Colab and follow COLAB_VOICE.md.")
    if mb > 200:
        print(f"  NOTE: {mb:.0f} MB is a slow browser upload - consider Drive instead.")


if __name__ == "__main__":
    main()
