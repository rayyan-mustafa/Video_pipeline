"""The voice stage: get takes onto disk, named right, then aligned.

TWO PROBLEMS, AND ONLY THE SECOND ONE IS THE INTERESTING ONE.

The interesting problem is alignment: Whisper reads the take and tells us where
every word lands, which is where all downstream timing comes from. The boring
problem is that the files are usually not called `section_01.wav`, and until they
are, the interesting problem cannot start at all. That boring problem cost a
whole session on the first project, so `pipeline.recordings` now solves it by
measurement and this module exposes it: the UI shows the proposed mapping with
its evidence, and one click applies it.

WHY THE PROPOSAL IS SHOWN AND NOT APPLIED AUTOMATICALLY. The mapping is inferred
from how constant the reader's words-per-minute comes out, and that is strong
evidence but not proof. A wrong mapping wastes a Whisper pass - on `large-v3`
that is real money - so the operator sees the margin and confirms. When the
margin is thin the resolver says `ambiguous` and the UI must not offer one-click
apply at all.

MODEL SIZE IS A DELIBERATE CHOICE, NOT A DEFAULT. `base` on CPU is minutes and
good enough to check the plumbing; `large-v3` is what ships, and on this laptop
it is roughly ten times slower - which is the whole reason the Colab GPU path
exists. The UI must state which one it is about to run and what that costs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi import UploadFile

from . import jobs, paths, settings

paths.ensure_importable()
from pipeline import recordings                    # noqa: E402

# THE AUTHORITATIVE READINESS CHECK. Imported rather than reimplemented: the UI
# must never claim a project is ready when the CLI would refuse it, and the only
# way to guarantee that is to ask the same function.
from build_video import find_recording             # noqa: E402

# What `build_video --whisper` accepts, in the order the UI should offer them.
# `large-v3` is the shipping choice; the rest exist for checking the plumbing
# without paying for it.
MODELS = ("base", "small", "medium", "large-v3")
DEFAULT_MODEL = "large-v3"

# Rough CPU cost per second of audio, measured relative to `base`. Advisory only
# - it exists so the UI can warn before someone starts a two-hour local run by
# accident, not to predict a wall-clock time.
RELATIVE_COST = {"base": 1.0, "small": 2.5, "medium": 6.0, "large-v3": 10.0}


def _section_words(proj) -> dict[int, int]:
    """Section number -> word count, read from the read sheets.

    These are what turn a pile of durations into a testable prediction, so a
    project with no `sections/` cannot have its recordings resolved at all.
    """
    words: dict[int, int] = {}
    if not proj.sections.is_dir():
        return words
    for p in sorted(proj.sections.glob("section_*.txt")):
        m = re.match(r"^section_(\d{1,2})$", p.stem, re.I)
        if not m:
            continue
        try:
            words[int(m.group(1))] = len(p.read_text(encoding="utf-8").split())
        except OSError:
            continue
    return words


def status(proj) -> dict:
    """Everything the Voice page needs to render, for any project.

    Reports facts and a proposal; it does not decide. `ready` is the one
    judgement, and it means exactly "build_video --stage voice would find every
    section", which is checked by asking `find_recording` rather than by
    re-deriving the rule here.
    """
    words = _section_words(proj)
    files = recordings.list_audio(proj.recordings)

    proposal = recordings.propose(proj.recordings, words)

    resolved = {s: [p.name for p in find_recording(proj.recordings, s)]
                for s in sorted(words)}
    missing = [s for s, hits in resolved.items() if not hits]

    return {
        "sections": [{"section": s, "words": w,
                      "files": resolved.get(s, []),
                      "text": _preview(proj, s)}
                     for s, w in sorted(words.items())],
        "n_sections": len(words),
        "files": [{"name": f.name, "path": str(f.resolve()),
                   "canonical": recordings.is_canonical(f)}
                  for f in files],
        "n_files": len(files),
        "missing_sections": missing,
        "ready": bool(words) and not missing,
        "proposal": proposal.as_dict(),
        "can_revert": (proj.recordings / "naming.json").is_file(),
        "models": list(MODELS),
        "default_model": DEFAULT_MODEL,
        "voice_profiles": ["natural", "clip", "raw"],
        "colab": settings.check_colab(),
        "results": results(proj),
    }


def _preview(proj, section: int, n: int = 90) -> str:
    """First few words of a section, so the operator can tell them apart."""
    p = proj.sections / f"section_{section:02d}.txt"
    try:
        t = " ".join(p.read_text(encoding="utf-8").split())
    except OSError:
        return ""
    return t[:n] + ("..." if len(t) > n else "")


def propose(proj) -> dict:
    """The mapping we think is right, with its evidence."""
    return recordings.propose(proj.recordings, _section_words(proj)).as_dict()


def apply(proj, force: bool = False) -> dict:
    """Rename takes onto section_NN. Refuses a low-confidence guess.

    `force` exists because the operator can listen to the takes and know the
    answer the arithmetic could not reach - but it has to be asked for, so a
    thin margin is never silently trusted.
    """
    p = recordings.propose(proj.recordings, _section_words(proj))
    if p.status == "ambiguous" and not force:
        raise ValueError(
            f"the mapping is not clear enough to apply automatically: {p.note}. "
            f"Check the takes, then apply with force if it is right.")
    if p.status == "impossible":
        raise ValueError(p.note)
    if p.status == "ambiguous":
        p.status = "ok"                       # forced: the operator has decided
    return recordings.apply(proj.recordings, p)


def revert(proj) -> dict:
    """Undo the last apply, from the record it wrote."""
    return recordings.revert(proj.recordings)


async def upload(proj, files: list[UploadFile]) -> dict:
    """Save uploaded takes verbatim - naming is resolved as a separate step.

    Deliberately does NOT rename on the way in. The resolver needs to see every
    file together before it can group them, and a per-file guess made during
    upload would be exactly the kind of silent decision this stage exists to
    avoid.
    """
    proj.recordings.mkdir(parents=True, exist_ok=True)
    saved, skipped = [], []
    for f in files:
        name = Path(f.filename or "").name
        if not name or Path(name).suffix.lower().lstrip(".") not in recordings.AUDIO_EXTS:
            skipped.append({"name": name or "(unnamed)",
                            "why": "not an audio file this pipeline reads"})
            continue
        dst = proj.recordings / name
        if dst.exists():
            skipped.append({"name": name, "why": "already here"})
            continue
        dst.write_bytes(await f.read())
        saved.append(name)
    return {"saved": saved, "skipped": skipped, "status": status(proj)}


async def run(proj, whisper: str = DEFAULT_MODEL, raw_voice: bool = False,
              split: str | None = None, voice_profile: str = "natural",
              use_colab: bool = False):
    """Start `build_video.py --stage voice` as a streaming job.

    `use_colab` moves ONLY the transcribe step to a GPU worker - mastering and
    alignment stay local either way, so the result is identical apart from which
    Whisper weights produced the timestamps. See pipeline/transcribe.py.
    """
    if whisper not in MODELS:
        raise ValueError(f"unknown whisper model {whisper!r} - "
                         f"choose one of {', '.join(MODELS)}")
    argv = [
        paths.PYTHON, paths.BUILD_VIDEO,
        "--project", f"{proj.channel}/{proj.topic}",
        "--stage", "voice",
        "--whisper", whisper,
        "--voice", voice_profile,
    ]
    if raw_voice:
        argv += ["--raw-voice"]
    if split:
        argv += ["--split", split]
    if use_colab:
        url = settings.load()["colab_url"].strip()
        if not url:
            raise ValueError(
                "no Colab worker URL saved - start studio/colab_worker.py in a "
                "Colab cell and paste the URL it prints")
        argv += ["--colab-url", url]

    return await jobs.registry.start(
        kind="voice", project=f"{proj.channel}/{proj.topic}",
        argv=argv, cwd=paths.CHANNELS,
    )


def results(proj) -> dict | None:
    """What the last alignment produced - the numbers that decide picture timing.

    ANCHORED IS THE REAL NUMBER, NOT WORD-MATCH %. A scene anchors on its first
    matched word and then carries that word's exact timestamp, so a section can
    score 74% word-match and still place all nine of its scenes perfectly. The
    two are reported separately because conflating them sends the operator off
    to re-record a take that was fine.
    """
    # WHICH FILE HOLDS WHAT. The voice stage writes scenes_timeline.json (one
    # entry per SCENE, from the aligner) and words.json. render_plan.json is
    # written later by --stage plan and holds one entry per PICTURE. They are
    # the same list only when the split is 1:1; under word-cuts there are more
    # pictures than scenes. Pacing judges how long a PICTURE is on screen, so it
    # must come from the render plan and is simply absent until that stage runs.
    timeline = _load(proj.timeline)
    if timeline is None:
        return None
    scenes = timeline if isinstance(timeline, list) else timeline.get("scenes", [])

    plan = _load(proj.render_plan)
    picture_holds = [float(i["duration"]) for i in (plan or [])
                     if isinstance(i, dict) and "duration" in i]

    words_file = proj.build / "words.json"
    n_words = 0
    if words_file.exists():
        try:
            n_words = len(json.loads(words_file.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            pass

    scene_ends = [float(s.get("end", 0.0)) for s in scenes if isinstance(s, dict)]
    runtime = max(scene_ends) if scene_ends else 0.0

    out = {
        "scenes": len(scenes),
        "words": n_words,
        "has_voice": proj.voice.exists(),
        "voice_path": str(proj.voice.resolve()) if proj.voice.exists() else None,
        "runtime": round(runtime, 2),
        "pictures": len(picture_holds),
        "has_render_plan": bool(picture_holds),
    }
    if picture_holds:
        from pipeline import pacing
        out["pacing"] = pacing.verdict(picture_holds)
        # Only meaningful on MEASURED holds: which slots actually sit at the
        # ceiling cannot be known from word-count estimates.
        out["motion_dose"] = pacing.motion_dose(sum(picture_holds))
    return out


def _load(path: Path):
    """Read a JSON file, or None if it is absent or half-written."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
