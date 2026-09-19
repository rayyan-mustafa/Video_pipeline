"""Run a render bundle. Imports nothing, decides nothing.

THIS FILE IS COPIED INTO EVERY BUNDLE `renderjob.build()` writes, and the
worker imports it from there rather than from its own installation. So the
code that executes a manifest is always the code that shipped with that
manifest, and "the worker is running last month's executor against this
month's manifest" stops being a state the system can reach.

It is deliberately dependency-free - no `pipeline` imports, no third-party
packages - because anything it imported would have to exist on the far side
too, and the entire point of the bundle is that the far side needs nothing
but Python and ffmpeg.

WHAT IT REFUSES. Every argv in a manifest must begin with "ffmpeg". A
manifest is a list of filter arguments, not a way to choose a program, and
this is the line that keeps it that way.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

MANIFEST = "manifest.json"
VERSION = 1


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check(manifest: dict) -> list[str]:
    """Everything wrong with a manifest, as human-readable strings.

    Run before anything executes: a truncated upload is the normal failure
    here, and finding it now costs a message rather than a five-minute render
    that ends in a corrupt mp4.
    """
    problems = []
    if manifest.get("version") != VERSION:
        problems.append(f"manifest version {manifest.get('version')!r}, "
                        f"this executor speaks {VERSION}")
    if not manifest.get("steps"):
        problems.append("manifest has no steps")
    for step in manifest.get("steps", []):
        argv = step.get("argv") or []
        if not argv or argv[0] != "ffmpeg":
            problems.append(f"step {step.get('index')}: argv does not start with "
                            f"ffmpeg - refusing to run it")
    join = manifest.get("join_argv") or []
    if not join or join[0] != "ffmpeg":
        problems.append("join_argv does not start with ffmpeg - refusing to run it")
    return problems


def verify_files(root: Path, manifest: dict) -> list[str]:
    """Which bundled files are missing or arrived with the wrong bytes."""
    root = Path(root)
    bad = []
    for rel, want in (manifest.get("files") or {}).items():
        p = root / rel
        if not p.is_file():
            bad.append(f"{rel}: missing from the bundle")
        elif sha256(p) != want:
            bad.append(f"{rel}: checksum does not match the manifest")
    return bad


def execute(root: Path, manifest: dict, run, on_progress=None) -> Path:
    """Run a bundle: every step, then the join. Returns the finished file.

    `run(argv, cwd, label)` is injected so the caller owns how a command is
    spawned and how its failure is reported - the worker wants the ffmpeg
    stderr in its own log, and a test wants to assert on the argv without
    spawning anything at all.
    """
    root = Path(root)
    problems = check(manifest)
    if problems:
        raise ValueError("; ".join(problems))

    steps = manifest["steps"]
    for n, step in enumerate(steps, 1):
        # MAKE THE OUTPUT'S FOLDER FIRST. ffmpeg will not create a missing
        # directory for its output, and a zip does not carry empty ones - so a
        # bundle with no external clips arrives with no `clips/` at all and
        # every slot fails with "No such file or directory" pointing at the
        # OUTPUT, which reads like a missing input. Cheap to guarantee here,
        # and it keeps the bundle format from depending on what the zip
        # happened to preserve.
        out_rel = step.get("output")
        if out_rel:
            (root / out_rel).parent.mkdir(parents=True, exist_ok=True)
        run(step["argv"], root, f"slot {step['index']} ({step['kind']})")
        if on_progress:
            on_progress(n, len(steps), f"slot {step['index']}")

    # Written here rather than packed, because a listing is derived from the
    # steps and a packed copy could disagree with them.
    listing = root / manifest["concat_listing"]
    listing.write_text(
        "".join(f"file '{s['output']}'\n" for s in steps), encoding="utf-8")

    (root / manifest["output"]).parent.mkdir(parents=True, exist_ok=True)
    run(manifest["join_argv"], root, "join")
    return root / manifest["output"]
