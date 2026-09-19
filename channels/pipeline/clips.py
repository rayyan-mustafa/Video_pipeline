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

    def command(self, slot: Slot, out: Path,
                width: int, height: int, fps: int) -> list[str]:
        """The exact ffmpeg argv `render` would run. Runs nothing itself.

        SPLIT OUT SO THE RENDER CAN HAPPEN SOMEWHERE ELSE. A remote render
        (see `pipeline/renderjob.py`) ships these argv lists to a worker rather
        than shipping the pipeline, which is what keeps the worker unable to
        drift from this repo: every editorial decision - the move, the frame
        count, the encode settings - is made HERE by the same code the local
        render uses, and the worker only executes.
        """
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

    def command(self, slot: Slot, out: Path,
                width: int, height: int, fps: int) -> list[str]:
        n = motion.frames_for(slot.hold, fps)
        vf = f"{motion.fit_pad_vf(width, height)},format=yuv420p"
        return ["ffmpeg", "-y", "-loop", "1", "-i", str(slot.image),
                "-vf", vf, "-frames:v", str(n), *_encode_args(fps), str(out)]

    def render(self, slot: Slot, out: Path,
               width: int, height: int, fps: int) -> None:
        _run(self.command(slot, out, width, height, fps),
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

    def command(self, slot: Slot, out: Path,
                width: int, height: int, fps: int) -> list[str]:
        m = motion.move_for(slot.index, slot.hold)
        n = motion.frames_for(slot.hold, fps)
        vf = motion.filter_chain(m, slot.hold, width, height, fps, self.supersample)
        # No -loop: zoompan's d= turns the single input frame into n outputs.
        # Looping as well would queue frames the filter never consumes and the
        # encode would not terminate on some builds.
        return ["ffmpeg", "-y", "-i", str(slot.image),
                "-vf", vf, "-frames:v", str(n), *_encode_args(fps), str(out)]

    def render(self, slot: Slot, out: Path,
               width: int, height: int, fps: int) -> None:
        _run(self.command(slot, out, width, height, fps),
             f"kenburns slot {slot.index}")


class VideoAssetProvider:
    """Conform an EXTERNAL video clip (Google Flow) onto one slot.

    THIS DOES NOT GENERATE. Flow makes the clip; this ingests it - scale/pad to
    the build size, conform fps, trim to the slot's hold, and emit something that
    concatenates byte-identically with the Ken Burns clips. All the hard part
    already happened somewhere else, which is why this is small.

    IT CHECKS, IT DOES NOT TRUST. A Flow clip is an artefact nothing in this repo
    produced, so every assumption is verified before encoding. The alternative is
    discovering a bad clip at slot 140 at the end of a five-minute render, which
    is the expensive moment to find it - the same reasoning behind `mux()`'s
    PIL-verify preflight over every image.

    THE RESIDUAL IS TAKEN OFF THE TAIL. A slot's hold lands on a real word
    boundary, so it is rarely exactly 4.000s - `pacing` accepts [3.5, 4.0], which
    leaves up to half a second to lose. That comes off the END of the clip, never
    by slowing it: `motion.py`'s founding rule is that the eye reads speed, and
    these are already a slow ambient drift - stretching 4.0s over 3.6s of screen
    time would make it read as static. The tail is also the least informative
    part of an inbetween, so it is the cheapest half-second to give up.

    NEVER FREEZE, NEVER EXTEND. A clip shorter than its slot is a HARD FAIL, not
    something to pad: covering the gap would mean holding a frame, and a hold
    longer than the clip pushes past HARD_MAX by definition.
    """

    name = "videoasset"

    def __init__(self, clip_for: Callable[[int], Path | None]):
        # Injected rather than reading motion_slots.json here, so this stays a
        # pure "given a slot, make a clip" provider and the selection policy
        # lives entirely in motionplan.py.
        self.clip_for = clip_for

    def source(self, slot: Slot) -> Path:
        src = self.clip_for(slot.index)
        if src is None:
            raise FileNotFoundError(
                f"slot {slot.index} is marked as motion but no clip is in the "
                f"project's motion/ folder yet")
        return src

    def signature(self, slot: Slot, width: int, height: int, fps: int) -> str:
        # The SOURCE CLIP's identity has to be in the cache key, not just the
        # slot's - the slot's own fingerprint hashes the still image, which for a
        # motion slot is only the Flow seed and does not change when the clip is
        # re-exported.
        try:
            src = self.source(slot)
            st = src.stat()
            stamp = f"{src.name}:{st.st_size}:{int(st.st_mtime)}"
        except (FileNotFoundError, OSError):
            stamp = "missing"
        return f"{self.name}|{stamp}|{width}x{height}@{fps}"

    def command(self, slot: Slot, out: Path,
                width: int, height: int, fps: int) -> list[str]:
        src = self.source(slot)
        want = motion.frames_for(slot.hold, fps)
        vf = f"{motion.fit_pad_vf(width, height)},fps={fps},format=yuv420p"
        # -frames:v is what actually trims the tail; -t would land on a
        # timestamp and could leave the count one frame either side, and
        # `concat -c copy` has no way to fix drift afterwards.
        return ["ffmpeg", "-y", "-i", str(src),
                "-vf", vf, "-frames:v", str(want), "-an",
                *_encode_args(fps), str(out)]

    def render(self, slot: Slot, out: Path,
               width: int, height: int, fps: int) -> None:
        src = self.source(slot)
        want = motion.frames_for(slot.hold, fps)

        info = probe_video(src)
        if info is None:
            raise RuntimeError(
                f"slot {slot.index}: ffprobe cannot read {src.name} - it is not "
                f"a video file this build can use")
        if info["duration"] + 1e-3 < slot.hold:
            raise RuntimeError(
                f"slot {slot.index}: {src.name} is {info['duration']:.2f}s but "
                f"the slot holds {slot.hold:.2f}s. A clip shorter than its slot "
                f"cannot be covered without slowing or freezing it, and both are "
                f"banned - re-export it at least {slot.hold:.2f}s.")

        _run(self.command(slot, out, width, height, fps),
             f"motion slot {slot.index} ({src.name})")

        got = probe_frames(out)
        if got is not None and got != want:
            raise RuntimeError(
                f"slot {slot.index}: conformed {src.name} to {got} frames but "
                f"the slot needs exactly {want} at {fps}fps. Concat cannot "
                f"correct drift, so this would desync everything after it.")


class MixedProvider:
    """Per-slot dispatch: motion slots get a real clip, everything else the base.

    THIS IS WHAT A FINISHED VIDEO ACTUALLY USES. The other providers answer
    "how should EVERY slot be treated"; a real build treats 11 of 217 slots as
    ingested Flow clips and the other 206 as Ken Burns stills. Expressing that as
    a provider rather than as a branch inside the mux keeps `render_sequence`,
    the cache and the concat completely unchanged - they still see one provider
    with one contract.

    The sub-provider's name goes into the signature, so a slot that switches
    between still and motion re-renders instead of serving the other one's clip
    out of the cache.
    """

    name = "mixed"

    def __init__(self, base: MotionProvider, video: MotionProvider,
                 motion_indices: set[int]):
        self.base = base
        self.video = video
        self.motion_indices = set(motion_indices)

    def _for(self, slot: Slot) -> MotionProvider:
        return self.video if slot.index in self.motion_indices else self.base

    def signature(self, slot: Slot, width: int, height: int, fps: int) -> str:
        p = self._for(slot)
        return f"{p.name}|{p.signature(slot, width, height, fps)}"

    def command(self, slot: Slot, out: Path,
                width: int, height: int, fps: int) -> list[str]:
        return self._for(slot).command(slot, out, width, height, fps)

    def render(self, slot: Slot, out: Path,
               width: int, height: int, fps: int) -> None:
        self._for(slot).render(slot, out, width, height, fps)


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

    _run(concat_command(listing, out, audio), "concat")


def concat_command(listing: Path, out: Path, audio: Path | None) -> list[str]:
    """The join, as argv. Same split as the providers' `command()` - see there."""
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listing)]
    if audio:
        cmd += ["-i", str(audio), "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                "-shortest"]
    cmd += ["-c:v", "copy", "-movflags", "+faststart", str(out)]
    return cmd


def probe_duration(path: Path) -> float:
    """Actual duration of a rendered file, for verifying against the plan."""
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True)
    if proc.returncode != 0:
        return 0.0
    return float(json.loads(proc.stdout)["format"]["duration"])


def probe_video(path: Path) -> dict | None:
    """Duration, size and fps of an external clip, or None if unreadable.

    Deliberately returns None rather than raising: the caller wants to fail with
    a message naming the slot and the file, which is far more use than an
    ffprobe stack trace.
    """
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate:format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True)
    if proc.returncode != 0:
        return None
    try:
        data = json.loads(proc.stdout)
        stream = (data.get("streams") or [{}])[0]
        num, _, den = (stream.get("r_frame_rate") or "0/1").partition("/")
        return {
            "duration": float(data["format"]["duration"]),
            "width": int(stream.get("width", 0)),
            "height": int(stream.get("height", 0)),
            "fps": (float(num) / float(den)) if float(den or 0) else 0.0,
        }
    except (KeyError, ValueError, IndexError, json.JSONDecodeError):
        return None


def probe_frames(path: Path) -> int | None:
    """Exact frame count of a rendered clip. None if it cannot be counted."""
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "json", str(path)],
        capture_output=True, text=True)
    if proc.returncode != 0:
        return None
    try:
        return int(json.loads(proc.stdout)["streams"][0]["nb_read_frames"])
    except (KeyError, ValueError, IndexError, json.JSONDecodeError):
        return None


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
