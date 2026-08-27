"""
Move a finished take into place, with a record of which take it was.

    python collect_takes.py --list             # comped takes, newest first
    python collect_takes.py --section 3        # newest unused comped take -> section_03.wav
    python collect_takes.py --section 3 --take live_20260823_101934
    python collect_takes.py --section 3 --from some/other/file.wav
    python collect_takes.py --status           # what is in voice_recordings, and where from

=============================================================================
 WHY THIS EXISTS
=============================================================================
`vo_coach live` writes data/takes/live_TIMESTAMP_comped.wav. The notebook wants
voice_recordings/section_NN.wav. Renaming that by hand, six times, is one typo
away from section 4 holding section 5's audio - and nothing downstream raises.
Whisper transcribes it fine, difflib matches almost nothing, the boundaries in
both sections become guesses, and the first symptom is pictures drifting in a
finished video.

So this refuses the two mistakes that are silent:

  - it will not overwrite a section that already has audio
  - it will not use the same take for two sections

and it records provenance in voice_recordings/_takes.json, so "which read is
this?" has an answer three days later.

=============================================================================
 THE COMPED FILE IS THE KEEPER, AND IT IS NOT MASTERED
=============================================================================
`live` writes two files per session: the raw capture and `_comped`, which is the
raw one with the failed attempts removed and the script's own pauses put back.
The comped file is what the notebook wants.

Do NOT run `vo_coach master` first. The notebook masters per section before
alignment and again over the concatenation; a third pass squashes it. And
anything that changes DURATION - comping, fix-pauses - has to happen before
notebook Step 18 locks the length, which means here, on the laptop.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import paths, qc, readsheet  # noqa: E402

HUMAN_DIR = "voice_recordings"
LEDGER = "_takes.json"

_DEFAULT_TAKES = Path(__file__).resolve().parent.parent / "vo_coach" / "data" / "takes"


def takes_dir(override: str | None) -> Path:
    return Path(override or os.environ.get("VO_COACH_TAKES") or _DEFAULT_TAKES)


def comped_takes(d: Path) -> list[Path]:
    """Comped takes, newest first. Raw captures are deliberately not listed."""
    return sorted(d.glob("*_comped.wav"), key=lambda p: p.stat().st_mtime, reverse=True)


# ---------------------------------------------------------------------------
#  The ledger
# ---------------------------------------------------------------------------

def _ledger_path() -> Path:
    return paths.p(HUMAN_DIR, LEDGER)


def load_ledger() -> dict:
    f = _ledger_path()
    if not f.exists():
        return {}
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_ledger(led: dict) -> None:
    _ledger_path().parent.mkdir(parents=True, exist_ok=True)
    _ledger_path().write_text(json.dumps(led, indent=2), encoding="utf-8")


def one_line_qc(path: Path, expected_words: int | None) -> str:
    """The same measurements run_qc.py makes, on one file, said in one line.

    Cheap enough to do at copy time, and the fastest possible feedback: if the
    gain was wrong you find out now, while the mic is still in front of you.
    """
    r = qc.check_file(path, section=0, expected_words=expected_words)
    if r.fails:
        return f"  {r.verdict}  " + r.fails[0]
    snr = f"{r.snr_db:.0f} dB" if r.snr_db is not None else "n/a"
    line = (f"  {r.verdict}  {r.duration:.0f}s, peak {r.peak_db:.1f} dBFS, "
            f"SNR {snr}, {r.clip_runs} clip run(s)")
    if r.warns:
        line += f"\n  warn  {r.warns[0]}"
    return line


# ---------------------------------------------------------------------------
#  Commands
# ---------------------------------------------------------------------------

def cmd_list(d: Path, led: dict) -> int:
    takes = comped_takes(d)
    if not takes:
        print(f"No comped takes in {d}")
        print("  `vo_coach live --script sections/section_NN_marked.txt` writes them.")
        return 1

    used = {Path(v["source"]).name: k for k, v in led.items()}
    print(f"Comped takes in {d} (newest first):\n")
    for p in takes[:20]:
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime))
        mark = f"-> section {used[p.name]}" if p.name in used else ""
        print(f"  {p.name:<40} {when}  {p.stat().st_size/1e6:>6.1f} MB  {mark}")
    if len(takes) > 20:
        print(f"  ... and {len(takes) - 20} older")
    return 0


def cmd_status(led: dict) -> int:
    sections, words, source = _expected()
    d = paths.p(HUMAN_DIR)
    print(f"{d}\n")
    if not sections:
        print("  No script in this working folder yet.")
        return 1
    for sec in sections:
        f = d / f"section_{sec:02d}.wav"
        if not f.exists():
            print(f"  section {sec:>2}  -")
            continue
        info = led.get(str(sec), {})
        src = Path(info.get("source", "")).name or "(unknown origin)"
        print(f"  section {sec:>2}  {f.name}  <- {src}")
    done = sum(1 for s in sections if (d / f"section_{s:02d}.wav").exists())
    print(f"\n  {done}/{len(sections)} sections recorded"
          + ("" if done < len(sections) else " - run: python run_qc.py"))
    return 0


def _expected() -> tuple[list[int], dict[int, int], str]:
    man = readsheet.load_manifest()
    if man and man.get("sections"):
        return ([e["section"] for e in man["sections"]],
                {e["section"]: e["words"] for e in man["sections"]},
                "sections/manifest.json")
    f = paths.p("script_tagged.json")
    if f.exists():
        texts = readsheet.section_texts(json.loads(f.read_text(encoding="utf-8")))
        return (sorted(texts),
                {s: len(" ".join(t).split()) for s, t in texts.items()},
                "script_tagged.json")
    return [], {}, ""


def cmd_collect(args, d: Path, led: dict) -> int:
    sec = args.section
    sections, words, _ = _expected()
    if sections and sec not in sections:
        print(f"This script has sections {sections[0]}-{sections[-1]}; there is no "
              f"section {sec}.")
        return 1

    # --- pick the source --------------------------------------------------
    if args.from_path:
        src = Path(args.from_path)
        if not src.exists():
            print(f"No file at {src}")
            return 1
    elif args.take:
        stem = args.take[:-4] if args.take.endswith(".wav") else args.take
        stem = stem[:-7] if stem.endswith("_comped") else stem
        src = d / f"{stem}_comped.wav"
        if not src.exists():
            print(f"No comped take {stem!r} in {d}.")
            print("  `live` writes the comped file when at least one line was accepted;")
            print("  if only the raw capture exists, the session never accepted a take.")
            return 1
    else:
        used = {Path(v["source"]).name for v in led.values()}
        # --force means "I know which take I mean"; without it, a take already
        # assigned to another section is never picked automatically.
        candidates = [p for p in comped_takes(d) if args.force or p.name not in used]
        if not candidates:
            print(f"No usable comped take in {d}.")
            if used:
                print("  Every comped take there is already assigned to a section - "
                      "see --list.")
                print("  Record this section, or name one explicitly with --take.")
            else:
                print("  `vo_coach live --script sections/section_NN_marked.txt` writes "
                      "them.")
            return 1
        src = candidates[0]

    if not src.name.endswith("_comped.wav") and not args.from_path:
        print(f"  note: {src.name} is a raw capture, not a comped take.")

    # --- refuse the silent mistakes --------------------------------------
    dest = paths.p(HUMAN_DIR, f"section_{sec:02d}.wav")
    if dest.exists() and not args.force:
        info = led.get(str(sec), {})
        print(f"section_{sec:02d}.wav already exists"
              + (f" (from {Path(info['source']).name})" if info.get("source") else "") + ".")
        print("  Re-run with --force to replace it.")
        return 1

    already = next((s for s, v in led.items()
                    if Path(v.get("source", "")).name == src.name and s != str(sec)), None)
    if already and not args.force:
        print(f"{src.name} is already section {already}.")
        print("  Using one take for two sections means one section is never recorded,")
        print("  and the aligner will not notice. Record this section, or --force.")
        return 1

    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)

    r = qc.check_file(dest, section=sec, expected_words=words.get(sec))
    led[str(sec)] = {
        "source": str(src),
        "copied_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "bytes": dest.stat().st_size,
        "duration": r.duration,
        "peak_db": r.peak_db,
        "clip_runs": r.clip_runs,
        "snr_db": r.snr_db,
    }
    save_ledger(led)

    print(f"{src.name}  ->  {dest}")
    print(one_line_qc(dest, words.get(sec)))

    remaining = [s for s in sections if not paths.p(HUMAN_DIR, f"section_{s:02d}.wav").exists()]
    if remaining:
        print(f"\n  still to record: {', '.join(str(s) for s in remaining)}")
    else:
        print("\n  every section is in place - run: python run_qc.py")
    return 0 if not r.fails else 1


def main() -> int:
    paths.force_utf8_stdout()

    ap = argparse.ArgumentParser(
        description="Put a vo_coach take into voice_recordings/ as section_NN.wav.")
    ap.add_argument("--workdir", help="the shared Drive folder (default: VIDEO_PIPELINE_WORKDIR)")
    ap.add_argument("--takes-dir", help="vo_coach's takes folder (default: ../vo_coach/data/takes)")
    ap.add_argument("--section", type=int, help="which section this take is")
    ap.add_argument("--take", help="take id, e.g. live_20260823_101934 (default: newest unused)")
    ap.add_argument("--from", dest="from_path", metavar="PATH",
                    help="use this file instead of a take from vo_coach")
    ap.add_argument("--force", action="store_true",
                    help="overwrite the section, or reuse a take already assigned")
    ap.add_argument("--list", action="store_true", help="list comped takes and exit")
    ap.add_argument("--status", action="store_true",
                    help="show which sections are recorded, and from which take")
    args = ap.parse_args()

    if args.workdir:
        paths.set_workdir(args.workdir)

    d = takes_dir(args.takes_dir)
    led = load_ledger()

    if args.status:
        return cmd_status(led)
    if args.list:
        if not d.is_dir():
            print(f"No takes folder at {d} - point at it with --takes-dir.")
            return 1
        return cmd_list(d, led)
    if args.section is None:
        ap.error("--section is required (or use --list / --status)")
    if not args.from_path and not d.is_dir():
        print(f"No takes folder at {d} - point at it with --takes-dir.")
        return 1

    return cmd_collect(args, d, led)


if __name__ == "__main__":
    raise SystemExit(main())
