"""
Where files live, and how the local half of the pipeline talks to the Colab half.

=============================================================================
 THE CONTRACT
=============================================================================
Every stage of this pipeline reads its input from a file and writes its output
to a file. Nothing is passed in memory across a stage boundary. That was already
true inside the notebook -- the Control Panel's STAGES list is literally a
manifest of these files -- and it is what makes the local/Colab split possible
at all.

So the "connection" between the local scripts and the notebook is: one folder,
synced by Google Drive, with an agreed set of filenames in it.

    local  ->  source_pack.json, evidence_profile.json, script_tagged.json,
               vo_raw.txt, packaging.json, scenes_planned.json
    Colab  ->  vo_raw.wav, scenes_timeline.json, full_vo.wav, frames/,
               final_video.mp4, thumbnails/

=============================================================================
 WHY chdir IS NOT USED
=============================================================================
The notebook does os.chdir(WORKDIR) in Step 2 and then opens everything by bare
relative name. That is fine for a notebook, which is one process doing one thing.
A library that chdir's is a library that breaks the caller, so instead everything
here goes through p(). The notebook calls set_workdir(WORKDIR) once and then the
two halves resolve to the same absolute paths.

=============================================================================
 WHY THERE IS A DONE-MARKER AT ALL
=============================================================================
Google Drive sync is not atomic. A file that is half-uploaded still satisfies
os.path.exists(), and the Step 25 preflight in the notebook only checks existence
and PIL-readability -- neither of which catches a WAV that is 40% of its final
length. That failure arrives as a 0.6s "drift" error at mux time, after the frames
are already paid for.

mark_done() records the size of every output of a stage. require() checks those
sizes back. It is a cheap guard against the one failure mode the split introduces
that the single-notebook version did not have.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# The default assumes Google Drive for Desktop with the standard drive letter.
# Override with VIDEO_PIPELINE_WORKDIR, an entry in .env, or --workdir.
_DEFAULT_LOCAL = r"G:\My Drive\videos\video_pipeline"
_COLAB_DEFAULT = "/content/drive/MyDrive/videos/video_pipeline"

_workdir: Path | None = None

STAGES_FILE = "_stages.json"


def set_workdir(path: str | os.PathLike) -> Path:
    """Point the pipeline at a working folder. Call once, early."""
    global _workdir
    _workdir = Path(path).expanduser().resolve()
    _workdir.mkdir(parents=True, exist_ok=True)
    return _workdir


def workdir() -> Path:
    """Resolve the working folder, in priority order, the first time it is asked for."""
    global _workdir
    if _workdir is not None:
        return _workdir

    env = os.environ.get("VIDEO_PIPELINE_WORKDIR")
    if env:
        return set_workdir(env)

    if Path("/content").is_dir():          # running inside Colab
        return set_workdir(_COLAB_DEFAULT)

    return set_workdir(_DEFAULT_LOCAL)


def p(*parts: str) -> Path:
    """Absolute path to something inside the working folder."""
    return workdir().joinpath(*parts)


def force_utf8_stdout() -> None:
    """A Windows console defaults to cp1252 and raises UnicodeEncodeError on the
    first curly quote that comes back from a model or a fetched source. In the
    notebook this never happened, because Colab's stdout is UTF-8 -- which is
    exactly why it is easy to ship a local script that dies at the print()
    AFTER the paid API call has already succeeded. vo_coach hit this same bug and
    fixed it the same way."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


# --------------------------------------------------------------------------
#  Stage markers
# --------------------------------------------------------------------------

def _load_stages() -> dict:
    f = p(STAGES_FILE)
    if not f.exists():
        return {}
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return {}


def mark_done(stage: str, files: list[str]) -> None:
    """Record that `stage` finished, and how big each of its outputs was.

    Missing outputs are recorded as null rather than skipped, so a stage that
    half-completed is visibly half-complete instead of merely absent."""
    import time

    stages = _load_stages()
    stages[stage] = {
        "at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "files": {name: (p(name).stat().st_size if p(name).exists() else None)
                  for name in files},
    }
    p(STAGES_FILE).write_text(json.dumps(stages, indent=2), encoding="utf-8")


def require(stage: str) -> None:
    """Fail loudly if a stage's outputs are missing or have not finished syncing.

    Raises rather than warning: the whole point is to stop before spending GPU
    time on an input that is not all here yet."""
    stages = _load_stages()
    if stage not in stages:
        raise FileNotFoundError(
            f"Stage {stage!r} has not been run in {workdir()}.\n"
            f"  Stages present: {', '.join(sorted(stages)) or '(none)'}"
        )

    problems = []
    for name, expected in stages[stage]["files"].items():
        f = p(name)
        if expected is None:
            problems.append(f"{name}: was never written")
        elif not f.exists():
            problems.append(f"{name}: missing")
        elif f.stat().st_size != expected:
            problems.append(
                f"{name}: {f.stat().st_size} bytes on disk, {expected} expected "
                f"- Drive is probably still syncing this file"
            )

    if problems:
        raise RuntimeError(
            f"Stage {stage!r} is not usable yet:\n" +
            "\n".join(f"  - {x}" for x in problems) +
            "\n  Wait for Drive to finish syncing, or re-run the stage."
        )


def stage_report() -> str:
    """Human-readable 'where am I' summary. The CLI equivalent of the notebook's
    Control Panel checklist."""
    stages = _load_stages()
    lines = [f"Working folder: {workdir()}", ""]
    if not stages:
        lines.append("  (no stages recorded yet)")
        return "\n".join(lines)
    for name, info in stages.items():
        total = sum(v for v in info["files"].values() if v)
        lines.append(f"  {name:<22} {info['at']}   {total:,} bytes")
        for fname, size in info["files"].items():
            ok = "ok" if size and p(fname).exists() and p(fname).stat().st_size == size else "STALE"
            lines.append(f"      {fname:<28} {ok}")
    return "\n".join(lines)
