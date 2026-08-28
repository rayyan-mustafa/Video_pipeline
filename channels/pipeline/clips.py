"""Turn one slot into one video clip, then join the clips.

WHY PER-SLOT CLIPS AND NOT ONE BIG FILTERGRAPH. `build_video.mux()` feeds every
picture to ffmpeg's concat demuxer with a single `-vf`, which is exactly right
while nothing moves: one filter describes all 105 images. The moment each slot
needs its OWN move, that model breaks, and the obvious repair - one
`filter_complex` with 105 inputs and 105 zoompans - is the wrong one:

  * Windows caps a command line at 32767 characters. 200 slots of path plus
    filter blows through that, and it fails as a truncated-argument error that
    looks like nothing to do with length.
  * It is all-or-nothing. One corrupt jpeg 190 slots in throws away the whole
    render.
  * Nothing is reusable. Re-render after changing a single picture and you pay
    for all 200 again.

Rendering each slot to its own file and concatenating with `-c copy` fixes all
three, costs one extra container per slot, and does NOT cost an extra encode:
the clips are written at final quality and the join is a stream copy.

WHY THIS IS A PROVIDER AND NOT A FUNCTION - the important part.
A provider answers one question: given a picture and how long it is on screen,
produce that many seconds of video. Ken Burns answers it with a zoompan. A
generative interpolator (ToonCrafter and friends) answers it by sending the
picture to a GPU and getting animation back. Same signature, same cache, same
concat, wildly different cost - which is the point: the expensive one can be
used for a handful of slots while the cheap one covers the rest, and the
timeline never has to know which is which.

CACHING IS NOT AN OPTIMISATION HERE. A zoompan render of 200 slots at
SUPERSAMPLE 3 is minutes; a generative render of the same 200 is hours and real
money. The cache key covers the picture's bytes, the hold, the move and the
output format, so re-running after editing one slot re-renders one slot.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Protocol

from . import motion


@dataclass(frozen=True)
class Slot:
    """One picture and the time it owns. The unit every provider works in."""

    index: int
    image: Path
    hold: float

    def fingerprint(self, extra: str) -> str:
        """Identity of the RENDERED RESULT, not of the slot.

        Includes the source bytes' size and mtime rather than the path, so
        regenerating a picture under the same filename - which the `_2` variant
        workflow does constantly - invalidates the clip instead of silently
        serving the old one.
        """
        try:
            st = self.image.stat()
            stamp = f"{st.st_size}:{int(st.st_mtime)}"
        except OSError:
            stamp = "missing"
        raw = f"{self.image.name}|{stamp}|{self.hold:.4f}|{extra}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


class MotionProvider(Protocol):
    """Given a still and a duration, write a clip of exactly that duration.

    `name` participates in the cache key, so two providers never collide and
    switching provider re-renders rather than reusing the other one's output.
    """

    name: str

    def signature(self, slot: Slot, width: int, height: int, fps: int) -> str:
        """Everything about this provider's output that is not the slot itself."""
        ...

    def render(self, slot: Slot, out: Path,
               width: int, height: int, fps: int) -> None:
        ...


def _run(cmd: list[str], what: str) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{what} failed:\n{proc.stderr[-2000:]}")


def _encode_args(fps: int) -> list[str]:
    """Final-quality encode settings, shared by every provider.

    They must be IDENTICAL across providers or `concat -c copy` refuses to join
    the clips - which is the whole reason the join is free.
    """
    return [
        "-r", str(fps),
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-pix_fmt", "yuv420p",
    ]


class StillProvider:
    """No motion. What the pipeline does today, expressed as a clip.

    Exists so the A/B comparison is honest: the "before" side goes through the
    same encoder, the same scaler and the same concat as the "after" side, so
    any difference you see is the motion and not the plumbing.
    """

    name = "still"

    def signature(self, slot: Slot, width: int, height: int, fps: int) -> str:
        return f"{self.name}|{width}x{height}@{fps}"

    def render(self, slot: Slot, out: Path,
               width: int, height: int, fps: int) -> None:
        n = motion.frames_for(slot.hold, fps)
        vf = (f"scale={width}:{height}:force_original_aspect_ratio=decrease:flags=lanczos,"
              f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:white,format=yuv420p")
        _run(["ffmpeg", "-y", "-loop", "1", "-i", str(slot.image),
              "-vf", vf, "-frames:v", str(n), *_encode_args(fps), str(out)],
             f"still slot {slot.index}")


class KenBurnsProvider:
    """A slow zoom or drift, sized by the hold. See `motion.py` for the rules."""

    name = "kenburns"

    def __init__(self, supersample: int = motion.SUPERSAMPLE):
        self.supersample = supersample

    def signature(self, slot: Slot, width: int, height: int, fps: int) -> str:
        m = motion.move_for(slot.index, slot.hold)
        return (f"{self.name}|{m.name}|{m.z0:.4f}-{m.z1:.4f}|"
                f"{m.px0:.4f}-{m.px1:.4f}|{m.py0:.4f}-{m.py1:.4f}|"
                f"ss{self.supersample}|{width}x{height}@{fps}")

    def render(self, slot: Slot, out: Path,
               width: int, height: int, fps: int) -> None:
        m = motion.move_for(slot.index, slot.hold)
        n = motion.frames_for(slot.hold, fps)
        vf = motion.filter_chain(m, slot.hold, width, height, fps, self.supersample)
        # No -loop: zoompan's d= turns the single input frame into n outputs.
        # Looping as well would queue frames the filter never consumes and the
        # encode would not terminate on some builds.
        _run(["ffmpeg", "-y", "-i", str(slot.image),
              "-vf", vf, "-frames:v", str(n), *_encode_args(fps), str(out)],
             f"kenburns slot {slot.index}")


PROVIDERS: dict[str, Callable[[], MotionProvider]] = {
    "still": StillProvider,
    "kenburns": KenBurnsProvider,
}


def render_sequence(slots: Iterable[Slot], provider: MotionProvider,
                    cache_dir: Path, width: int, height: int, fps: int,
                    on_progress: Callable[[int, int, str], None] | None = None,
                    ) -> list[Path]:
    """Render every slot, reusing any clip whose inputs have not changed."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    slots = list(slots)
    out: list[Path] = []
    hits = 0

    for i, slot in enumerate(slots, 1):
        key = slot.fingerprint(provider.signature(slot, width, height, fps))
        clip = cache_dir / f"{slot.index:04d}_{key}.mp4"
        if clip.exists() and clip.stat().st_size > 0:
            hits += 1
        else:
            provider.render(slot, clip, width, height, fps)
        out.append(clip)
        if on_progress:
            on_progress(i, len(slots), clip.name)

    if on_progress:
        on_progress(len(slots), len(slots), f"{hits} cached, {len(slots) - hits} rendered")
    return out


def concat_clips(clips: list[Path], out: Path, audio: Path | None = None,
                 work: Path | None = None) -> None:
    """Join clips into one file. Stream copy for video - no second encode."""
    if not clips:
        raise ValueError("nothing to concat")
    work = work or out.parent
    work.mkdir(parents=True, exist_ok=True)
    listing = work / f"{out.stem}_clips.txt"
    with listing.open("w", encoding="utf-8") as f:
        for c in clips:
            f.write(f"file '{c.resolve().as_posix()}'\n")

    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listing)]
    if audio:
        cmd += ["-i", str(audio), "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                "-shortest"]
    cmd += ["-c:v", "copy", "-movflags", "+faststart", str(out)]
    _run(cmd, "concat")


def probe_duration(path: Path) -> float:
    """Actual duration of a rendered file, for verifying against the plan."""
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True)
    if proc.returncode != 0:
        return 0.0
    return float(json.loads(proc.stdout)["format"]["duration"])


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
