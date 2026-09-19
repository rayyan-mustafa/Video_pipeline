"""Round-trip a render bundle the way a remote render actually uses one.

    ..\\vo_coach\\.venv\\Scripts\\python.exe test_renderjob.py

WHY THIS EXISTS, AND WHY IT EXTRACTS TO A DIFFERENT FOLDER. Two bugs shipped
to the Colab worker in a row, and neither could be reproduced by building a
bundle and running it where it was built:

  1. Windows separators reached the manifest. `Path("assets/0000.jpeg")`
     stringifies as "assets\\0000.jpeg" on Windows, which on Linux is a
     filename containing a backslash - ffmpeg said the INPUT was missing.
     Invisible when both ends are the same OS.

  2. `clips/` arrived missing. A zip does not carry empty directories, so a
     bundle with no external clips (motion=none) unpacked without the folder
     every step writes into - ffmpeg said the OUTPUT was missing, which reads
     exactly like a missing input. Invisible when the bundle is executed in
     the folder that built it, because `build` had already made the directory.

Both are only reachable through zip -> unpack ELSEWHERE -> run, so that is
what this does. The `none` case is covered explicitly: it is the one that
packs nothing into clips/.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import renderjob                                  # noqa: E402

PASS = FAIL = 0


def check(label: str, got, want) -> None:
    global PASS, FAIL
    if got == want:
        PASS += 1
        print(f"  ok    {label}")
    else:
        FAIL += 1
        print(f"  FAIL  {label}\n          got  {got!r}\n          want {want!r}")


def _ffmpeg(*args: str) -> None:
    p = subprocess.run(["ffmpeg", "-y", "-v", "error", *args],
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-800:])


def _fixture(root: Path, n_slots: int, with_clip: bool) -> tuple[list[dict], Path]:
    """A tiny project: `n_slots` stills, optionally one real video clip."""
    src = root / "src"
    src.mkdir(parents=True, exist_ok=True)

    entries = []
    for i in range(n_slots):
        img = src / f"{i+1}section1.png"
        _ffmpeg("-f", "lavfi", "-i", f"color=c=0x{(i*40)%256:02x}3050:s=320x240",
                "-frames:v", "1", str(img))
        entries.append({"path": str(img), "duration": 1.0, "kind": "image",
                        "section": 1, "order": i + 1})

    if with_clip:
        clip = src / "flow.mp4"
        _ffmpeg("-f", "lavfi", "-i", "color=c=green:s=320x240:d=2",
                "-r", "30", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(clip))
        entries[-1] = {**entries[-1], "kind": "video", "clip_path": str(clip)}

    audio = src / "vo.wav"
    _ffmpeg("-f", "lavfi", "-i", f"sine=frequency=440:duration={n_slots}",
            "-ar", "48000", str(audio))
    return entries, audio


def roundtrip(tmp: Path, provider: str, with_clip: bool) -> Path:
    """build -> zip -> unpack SOMEWHERE ELSE -> execute. Returns the output."""
    root = tmp / f"{provider}_{'clip' if with_clip else 'noclip'}"
    entries, audio = _fixture(root, 3, with_clip)

    built = root / "built"
    manifest = renderjob.build(entries, audio, "out.mp4", built,
                               provider_name=provider,
                               width=320, height=240, fps=30)

    # No Windows separator may reach the manifest - bug 1.
    bad = [a for s in manifest["steps"] for a in s["argv"] if "\\" in a]
    bad += [a for a in manifest["join_argv"] if "\\" in a]
    check(f"{provider}/{'clip' if with_clip else 'noclip'}: no backslash in argv",
          bad, [])

    zpath = root / "bundle.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_STORED) as z:
        for p in sorted(built.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(built).as_posix())

    # THE POINT: unpack into a folder that build() never touched, so nothing
    # the builder created on the side is available to the run - bug 2.
    far = root / "worker"
    far.mkdir()
    with zipfile.ZipFile(zpath) as z:
        z.extractall(far)

    m = json.loads((far / "manifest.json").read_text(encoding="utf-8"))
    check(f"{provider}: executor shipped in the bundle",
          (far / "bundlerun.py").is_file(), True)
    check(f"{provider}: manifest passes its own check", renderjob.check(m), [])
    check(f"{provider}: bundle arrived intact", renderjob.verify_files(far, m), [])

    def run(argv, cwd, label):
        p = subprocess.run([str(a) for a in argv], cwd=str(cwd),
                           capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(f"{label}: {p.stderr[-800:]}")

    return renderjob.execute(far, m, run)


def main() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="renderjob_test_"))
    try:
        for provider, with_clip in (("kenburns", True),
                                    ("kenburns", False),
                                    ("still", False),
                                    ("none", False),
                                    ("none", True)):
            print(f"\n{provider}, {'with' if with_clip else 'no'} external clip")
            out = roundtrip(tmp, provider, with_clip)
            check(f"{provider}: output exists", out.is_file(), True)
            check(f"{provider}: output is not empty", out.stat().st_size > 0, True)

        # `none` must ignore external clips, exactly as build_video's plain
        # concat path does - it writes each slot's IMAGE even for a motion slot.
        root = tmp / "none_clip" / "built"
        m = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        check("none: external clips are not used", m["motion_slots"], [])
        kb = json.loads((tmp / "kenburns_clip" / "built" / "manifest.json")
                        .read_text(encoding="utf-8"))
        check("kenburns: the external clip IS used", kb["motion_slots"], [2])

        print(f"\n{PASS} passed, {FAIL} failed")
        sys.exit(1 if FAIL else 0)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
