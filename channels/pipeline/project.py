"""One video = one folder. Every path in the build derives from it.

    channels/
      wierdbiology/                 <- channel
        chin/                       <- topic
          script.txt                the words, as written
          sections/                 read sheets, one file per section
          voice_recordings/         section_01.wav ... what you record
          images/                   the pictures, named <order>section<section>
          motion/                   external Flow clips, one per motion slot
          build/                    everything derived - plan, prompts, timeline
          chin.mp4                  the output

WHY A PROJECT OBJECT RATHER THAN SIX PATH FLAGS. The earlier build took --work,
--sections, --recordings, --images and --out separately, all defaulting to the
current directory. That works for exactly one video and then quietly breaks: the
second topic's `--stage plan` happily reads the first topic's cached timeline,
because nothing in the path says which video it belongs to. Naming the project
once makes every derived path unambiguous and makes two topics impossible to
cross-contaminate.

Anything under build/ is derived and safe to delete - re-running regenerates it.
Anything outside build/ is yours: the script, the recordings, the pictures.
"""

from __future__ import annotations

import re
from pathlib import Path

# Everything the build writes lives here, so "delete build/ and re-run" is always
# a safe recovery. Inputs never go in it.
BUILD_DIRNAME = "build"

_SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class Project:
    """Resolves every path for one video from `<root>/<channel>/<topic>`."""

    def __init__(self, channel: str, topic: str, root: Path | str = "."):
        for part, label in ((channel, "channel"), (topic, "topic")):
            if not _SAFE.match(part):
                raise ValueError(
                    f"{label} name {part!r} is not a safe folder name - use "
                    f"letters, digits, dot, dash or underscore"
                )
        self.channel = channel
        self.topic = topic
        self.dir = Path(root) / channel / topic

    # ---- inputs (yours) ---------------------------------------------------
    @property
    def script(self) -> Path:
        return self.dir / "script.txt"

    @property
    def sections(self) -> Path:
        return self.dir / "sections"

    @property
    def recordings(self) -> Path:
        return self.dir / "voice_recordings"

    @property
    def images(self) -> Path:
        return self.dir / "images"

    @property
    def motion(self) -> Path:
        """External video clips (Google Flow), one per motion slot.

        AN INPUT, NOT A DERIVED FOLDER. Nothing in this pipeline generates a
        clip - it ingests, conforms and trims what you drop here, exactly as it
        does for images/. So it sits outside build/ and `delete build/ and
        re-run` must never touch it.
        """
        return self.dir / "motion"

    # ---- derived (safe to delete) -----------------------------------------
    @property
    def build(self) -> Path:
        return self.dir / BUILD_DIRNAME

    @property
    def plan(self) -> Path:
        return self.build / "script_plan.json"

    @property
    def prompts(self) -> Path:
        return self.build / "nb_prompts.txt"

    @property
    def timeline(self) -> Path:
        return self.build / "scenes_timeline.json"

    @property
    def voice(self) -> Path:
        return self.build / "full_vo.wav"

    @property
    def render_plan(self) -> Path:
        return self.build / "render_plan.json"

    @property
    def overrides(self) -> Path:
        """The timeline editor's only output: roll edits, replaced assets, and
        motion on/off, keyed by render_plan.json slot index. See
        studio/backend/overrides.py and imageplan.plan_from_binding's
        `overrides` param - this file is applied fresh every time the plan is
        (re)computed, never merged into script_plan.json or the binding
        itself, so the fingerprint that protects the binding never sees it."""
        return self.build / "overrides.json"

    @property
    def out(self) -> Path:
        return self.dir / f"{self.topic}.mp4"

    # ---- helpers ----------------------------------------------------------
    def make_dirs(self) -> None:
        for d in (self.sections, self.recordings, self.images,
                  self.motion, self.build):
            d.mkdir(parents=True, exist_ok=True)

    @classmethod
    def parse(cls, spec: str, root: Path | str = ".") -> "Project":
        """'wierdbiology/chin' -> Project. Accepts either slash."""
        parts = [p for p in re.split(r"[\\/]+", spec.strip()) if p]
        if len(parts) != 2:
            raise ValueError(
                f"--project must be '<channel>/<topic>', e.g. 'wierdbiology/chin' "
                f"(got {spec!r})"
            )
        return cls(parts[0], parts[1], root=root)

    def describe(self) -> str:
        rows = [
            ("script", self.script), ("sections", self.sections),
            ("recordings", self.recordings), ("images", self.images),
            ("motion", self.motion),
            ("build", self.build), ("output", self.out),
        ]
        w = max(len(n) for n, _ in rows)
        return "\n".join(f"  {n:<{w}}  {p}" for n, p in rows)

    def __repr__(self) -> str:
        return f"Project({self.channel}/{self.topic})"
