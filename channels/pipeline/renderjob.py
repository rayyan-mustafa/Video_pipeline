"""A render, packed into a folder that any machine with ffmpeg can execute.

WHY A BUNDLE OF ARGV AND NOT A COPY OF THE PIPELINE. The obvious way to render
on Colab is to ship `pipeline/` there and import it. That is the way that rots:
the worker notebook then has a copy of the editorial rules, and the day
`motion.py` changes its zoom rate the remote render quietly keeps using last
month's. `studio/colab_worker.py` already refuses that trade for the voice
stage - "this file knows nothing about the pipeline" - and this is the same
seam one step wider.

So the LAPTOP decides everything, using the exact provider objects the local
render uses, and writes the decisions down as ffmpeg argv. The worker unzips,
runs them in order, and joins the result. It owns no editorial opinion at all,
which means it cannot hold a stale one.

WHAT IS IN THE BUNDLE

    manifest.json      the argv, in order, plus the join and a checksum table
    assets/NNN.<ext>   one still per slot, renamed by slot so two sections
                       cannot collide on a filename
    clips/NNN.mp4      the external Flow clips, same renaming
    audio.wav          the mastered voiceover, byte-for-byte

EVERY PATH IN THE MANIFEST IS RELATIVE TO THE BUNDLE ROOT, and the worker runs
with its working directory set there. No path rewriting happens on the far
side - the strings that leave here are the strings ffmpeg sees, which is one
less thing that can differ between a local render and a remote one.

THE WORKER NEVER CHOOSES A PROGRAM. Every argv in the manifest starts with
"ffmpeg" and the worker refuses any that does not, so a manifest is a list of
filter arguments and not a way to run something else. It is generated on the
operator's own machine from their own project; the check is there to keep an
honest file honest, not to make a hostile one safe.
"""

from __future__ import annotations


import json
import shutil
from pathlib import Path, PurePosixPath

from . import bundlerun
from . import clips as clips_mod
from . import motion

# The run half lives in `bundlerun`, which has no imports of its own so it can
# be copied into the bundle and used on the far side. Re-exported here so a
# caller only has to know about one module.
from .bundlerun import MANIFEST, VERSION, check, execute, verify_files  # noqa: F401

_sha256 = bundlerun.sha256

# `build_video.py --motion none` is not a provider - it is the plain concat
# path from before motion existed: every slot is its own still, held for its
# hold, and the external Flow clips are NOT used. StillProvider is the per-slot
# expression of exactly that ("the 'before' side goes through the same encoder,
# the same scaler and the same concat as the 'after' side" - see its docstring),
# so `none` maps onto it with the clips switched off. Same pictures, same
# durations, same encode settings as the local `none`.
NO_MOTION = "none"


def build(plan_entries: list[dict], audio: Path, out_name: str, dest: Path,
          provider_name: str, width: int, height: int, fps: int,
          on_progress=None) -> dict:
    """Write a self-contained bundle for this plan into `dest`. Returns the manifest.

    `plan_entries` is render_plan.json AFTER `motionplan.attach_kinds` has run,
    so a slot that is marked motion and has its clip carries `clip_path`. A slot
    marked motion whose clip has NOT arrived falls back to the still provider,
    exactly as `build_video._mux_with_motion` does - a half-delivered set still
    renders rather than failing the whole build.
    """
    if provider_name != NO_MOTION and provider_name not in clips_mod.PROVIDERS:
        raise ValueError(f"unknown motion provider {provider_name!r} - "
                         f"choose one of {[NO_MOTION, *sorted(clips_mod.PROVIDERS)]}")
    if not plan_entries:
        raise ValueError("nothing to render - the plan is empty")
    if not audio.is_file():
        raise FileNotFoundError(f"no mastered voiceover at {audio}")

    dest = Path(dest)
    (dest / "assets").mkdir(parents=True, exist_ok=True)
    (dest / "clips").mkdir(parents=True, exist_ok=True)

    no_motion = provider_name == NO_MOTION
    base = (clips_mod.StillProvider() if no_motion
            else clips_mod.PROVIDERS[provider_name]())

    # Same rule as the local mux: a slot only becomes a video slot when its clip
    # is actually on disk. See build_video._mux_with_motion. Under `none` there
    # are no video slots at all - the local path writes each slot's IMAGE into
    # the concat list even for a motion slot, so the clips go unused there too.
    motion_slots: set[int] = set()
    if not no_motion:
        for i, e in enumerate(plan_entries):
            if e.get("kind") == "video" and e.get("clip_path"):
                motion_slots.add(i)

    # Copy the assets in under slot-numbered names, and remember where each
    # slot's input landed so the argv can point at it.
    inputs: dict[int, str] = {}
    files: dict[str, str] = {}          # bundle-relative path -> sha256

    def take(src: Path, rel: str) -> str:
        dst = dest / rel
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            shutil.copy2(src, dst)
        files[rel] = _sha256(dst)
        return rel

    for i, e in enumerate(plan_entries):
        if i in motion_slots:
            src = Path(e["clip_path"])
            inputs[i] = take(src, f"clips/{i:04d}{src.suffix.lower()}")
        else:
            src = Path(e["path"])
            if not src.is_file():
                raise FileNotFoundError(
                    f"slot {i}: {src} is not on disk - a bundle cannot carry a "
                    f"picture that does not exist yet")
            inputs[i] = take(src, f"assets/{i:04d}{src.suffix.lower()}")
        if on_progress:
            on_progress(i + 1, len(plan_entries), "packing")

    audio_rel = take(audio, "audio.wav")

    # THE EXECUTOR TRAVELS WITH THE JOB. See bundlerun's header: the worker
    # imports this copy, so the code that runs a manifest is always the code
    # that shipped with it.
    shutil.copy2(Path(bundlerun.__file__), dest / "bundlerun.py")

    # The providers build the argv, against BUNDLE-RELATIVE paths - which is
    # why the Slot objects below carry the copied name rather than the original.
    provider = base
    if motion_slots:
        provider = clips_mod.MixedProvider(
            base,
            clips_mod.VideoAssetProvider(
                lambda idx: PurePosixPath(inputs[idx]) if idx in motion_slots else None),
            motion_slots,
        )

    steps = []
    for i, e in enumerate(plan_entries):
        slot = clips_mod.Slot(i, PurePosixPath(inputs[i]), float(e["duration"]))
        out_rel = f"clips/out_{i:04d}.mp4"
        argv = provider.command(slot, PurePosixPath(out_rel), width, height, fps)
        steps.append({
            "index": i,
            "input": inputs[i],
            "output": out_rel,
            "frames": motion.frames_for(float(e["duration"]), fps),
            "duration": round(float(e["duration"]), 4),
            "kind": "video" if i in motion_slots else "still",
            "argv": [str(a) for a in argv],
        })

    listing = "concat_list.txt"
    join = clips_mod.concat_command(PurePosixPath(listing), PurePosixPath(out_name),
                                    PurePosixPath(audio_rel))

    manifest = {
        "version": VERSION,
        "provider": provider_name,
        "width": width, "height": height, "fps": fps,
        "audio": audio_rel,
        "output": out_name,
        "concat_listing": listing,
        "total_duration": round(sum(float(e["duration"]) for e in plan_entries), 3),
        "motion_slots": sorted(motion_slots),
        "steps": steps,
        "join_argv": [str(a) for a in join],
        "files": files,
    }
    # WINDOWS SEPARATORS MUST NEVER REACH THE MANIFEST. A bundle is built on
    # this laptop and run on Linux, so a "\\" in an argv is a literal
    # character in a filename over there and ffmpeg reports the file as
    # missing - which is a five-minute upload followed by an error about slot
    # 0. Cheap to check here, expensive to discover there.
    for step in steps:
        for a in step["argv"]:
            if "\\" in a:
                raise ValueError(
                    f"slot {step['index']}: a Windows path separator reached the "
                    f"manifest ({a!r}). Every bundle path must be POSIX.")
    for a in manifest["join_argv"]:
        if "\\" in a:
            raise ValueError(f"join: a Windows path separator reached the "
                             f"manifest ({a!r}). Every bundle path must be POSIX.")

    (dest / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
