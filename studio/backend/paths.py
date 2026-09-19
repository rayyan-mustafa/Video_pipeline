"""Where everything lives, and which Python runs the pipeline.

WHY THIS IS ITS OWN MODULE. Two things here are easy to get wrong and expensive
to debug, so they are stated once:

1. THE INTERPRETER. `channels\\.venv` exists but lacks faster-whisper,
   noisereduce and pyloudnorm - the voice stage dies in it. `vo_coach\\.venv`
   (Python 3.12.10) has the full audio stack plus fastapi. Every subprocess this
   backend spawns must use that one, so the choice is not repeated per call site.

2. THE PROJECT ROOT. `build_video.py --project wierdbiology/chin` resolves paths
   relative to its own working directory, so subprocesses must run with cwd set
   to `channels/`. Passing an absolute --project would not help: Project.parse()
   rejects anything that is not exactly "<channel>/<topic>".
"""

from __future__ import annotations

import sys
from pathlib import Path

# studio/backend/paths.py -> studio/backend -> studio -> repo root
REPO = Path(__file__).resolve().parent.parent.parent
CHANNELS = REPO / "channels"
STUDIO = REPO / "studio"

# The interpreter every pipeline subprocess must use. See note 1 above.
PYTHON = REPO / "vo_coach" / ".venv" / "Scripts" / "python.exe"
if not PYTHON.exists():                      # POSIX layout, for a future port
    PYTHON = REPO / "vo_coach" / ".venv" / "bin" / "python"

SETTINGS_FILE = STUDIO / "backend" / "settings.json"

# The CLIs this backend drives. Named rather than string-built at call sites so a
# rename shows up as an ImportError here instead of a FileNotFoundError at runtime.
BUILD_VIDEO = CHANNELS / "build_video.py"
RENDER_REMOTE = CHANNELS / "render_remote.py"
PLAN_SCRIPT = CHANNELS / "plan_script.py"
RUN_SCRIPT = CHANNELS / "run_script.py"
MAKE_PROMPTS = CHANNELS / "make_nb_prompts.py"
YOUTUBE_PACKAGER = CHANNELS / "youtube_packager.py"


def ensure_importable() -> None:
    """Put `channels` on sys.path so `from pipeline import ...` works in-process.

    The read-only parts of the pipeline - imageplan, scenesplit, project - are
    clean importable modules, and calling them directly is far cheaper than
    spawning a subprocess to ask how many scenes a split mode produces.
    """
    p = str(CHANNELS)
    if p not in sys.path:
        sys.path.insert(0, p)


def check() -> list[str]:
    """Problems that would break the backend, as human-readable strings."""
    problems = []
    if not PYTHON.exists():
        problems.append(f"pipeline interpreter not found: {PYTHON}")
    if not CHANNELS.is_dir():
        problems.append(f"channels/ not found: {CHANNELS}")
    for f in (BUILD_VIDEO, PLAN_SCRIPT):
        if not f.exists():
            problems.append(f"missing CLI: {f}")
    return problems
