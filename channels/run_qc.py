"""
The QC gate. Run this on the laptop after recording, BEFORE you open Colab.

    python run_qc.py                     # check every section
    python run_qc.py --section 3         # just one, while you are still recording it
    python run_qc.py --json report.json  # machine-readable, for a future hook
    python run_qc.py --strict            # warnings fail too

No model, no GPU, no network - it is arithmetic over the samples and takes
seconds. Exit code 0 means the recordings are safe to hand to the notebook,
1 means something needs re-recording.

=============================================================================
 WHY IT RUNS HERE AND NOT IN THE NOTEBOOK
=============================================================================
The notebook's first look at your audio is Whisper, in Phase 1, on a paid T4.
It does not report bad audio - it aligns it badly, and you find out when the
pictures drift against your voice, after the frames are drawn. This is the last
point in the chain where re-recording costs nothing but your time, so it is the
right place to be told.

It checks the five things that actually go wrong:

  clipping     runs of full-scale samples. Permanent - no model recovers it,
               and it flows straight into a worse transcript and then into
               scene boundaries in the wrong place.
  level        peaks want to sit near -9 dBFS, not on the ceiling.
  noise        SNR, with the digital-silence trap handled.
  length       duration against the read sheet's word count - this is how you
               catch section_04.wav actually containing section 5.
  the set      missing files, and mismatched rates or levels between sections.

Pace is checked only widely enough to catch the wrong file. Judging delivery is
vo_coach's job, and it needs the calibration takes that do not exist yet.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import paths, qc, readsheet  # noqa: E402

HUMAN_DIR = "voice_recordings"


def find_section_file(sec: int) -> Path | None:
    """The notebook's own discovery rule, so the two agree on what exists.

    Anything it would accept, this must find - otherwise QC passes a set of
    files the notebook then reads differently.
    """
    d = paths.p(HUMAN_DIR)
    for stem in (f"section_{sec:02d}", f"section_{sec}"):
        for ext in ("wav", "flac", "m4a", "mp3"):
            hits = sorted(glob.glob(str(d / f"{stem}.{ext}")))
            if hits:
                return Path(hits[0])
    return None


def expected_sections() -> tuple[list[int], dict[int, int], str]:
    """(section numbers, words per section, where that came from).

    The read sheet manifest is the truth when it exists, because it was written
    from the same chunking the aligner uses. script_tagged.json is the fallback
    for a working folder produced before read sheets existed.
    """
    man = readsheet.load_manifest()
    if man and man.get("sections"):
        return ([e["section"] for e in man["sections"]],
                {e["section"]: e["words"] for e in man["sections"]},
                "sections/manifest.json")

    f = paths.p("script_tagged.json")
    if f.exists():
        tagged = json.loads(f.read_text(encoding="utf-8"))
        texts = readsheet.section_texts(tagged)
        return (sorted(texts),
                {s: len(" ".join(t).split()) for s, t in texts.items()},
                "script_tagged.json")

    return [], {}, ""


def print_report(result: dict, source: str) -> None:
    print()
    for r in result["sections"]:
        head = f"  section {r.section:>2}  {r.verdict:<5}"
        if r.path is None:
            print(f"{head} (no file)")
        else:
            snr = f"{r.snr_db:>5.1f}" if r.snr_db is not None else "    -"
            print(f"{head} {r.duration:>6.1f}s  peak {r.peak_db:>6.1f} dBFS  "
                  f"SNR {snr} dB  clip runs {r.clip_runs:>3}"
                  + (f"  {r.implied_wpm:>5.0f} wpm" if r.implied_wpm else ""))
        for m in r.fails:
            print(f"       FAIL  {m}")
        for m in r.warns:
            print(f"       warn  {m}")

    for m in result["run_fails"]:
        print(f"\n  FAIL  {m}")
    for m in result["run_warns"]:
        print(f"\n  warn  {m}")

    n_fail = sum(len(r.fails) for r in result["sections"]) + len(result["run_fails"])
    n_warn = sum(len(r.warns) for r in result["sections"]) + len(result["run_warns"])

    print("\n" + "=" * 70)
    if result["ok"] and not n_warn:
        print("  PASS - every section is clean. Open the notebook and start at Step 16.")
    elif result["ok"]:
        print(f"  PASS with {n_warn} warning(s). Nothing here stops the notebook, but")
        print("  read them - they are the things that show up later as drift.")
    else:
        print(f"  FAIL - {n_fail} problem(s). Fix these before spending a T4 session;")
        print("  re-recording is free right now and stops being free in Colab.")
    print("=" * 70)
    if source:
        print(f"  expected sections and word counts from {source}")


def main() -> int:
    paths.force_utf8_stdout()

    ap = argparse.ArgumentParser(
        description="Check voice_recordings/ before handing it to the notebook.")
    ap.add_argument("--workdir", help="the shared Drive folder (default: VIDEO_PIPELINE_WORKDIR)")
    ap.add_argument("--section", type=int, action="append",
                    help="check only this section (repeatable)")
    ap.add_argument("--strict", action="store_true",
                    help="treat warnings as failures")
    ap.add_argument("--json", dest="json_out", metavar="PATH",
                    help="also write the full report as JSON")
    args = ap.parse_args()

    if args.workdir:
        paths.set_workdir(args.workdir)

    d = paths.p(HUMAN_DIR)
    print(f"Checking {d}")

    sections, words, source = expected_sections()
    if not sections:
        print("\n  No script in this working folder, so there is nothing to check against.")
        print("  Run:  python run_script.py --topic <topic>")
        return 1

    if args.section:
        wanted = [s for s in sections if s in set(args.section)]
        unknown = sorted(set(args.section) - set(sections))
        if unknown:
            print(f"  (no section {', '.join(map(str, unknown))} in this script - ignoring)")
        if not wanted:
            return 1
    else:
        wanted = sections

    files: dict[int, Path] = {}
    missing: list[int] = []
    for sec in wanted:
        f = find_section_file(sec)
        if f is None:
            missing.append(sec)
        else:
            files[sec] = f

    # Files that are not part of this script at all. Usually a leftover from the
    # previous topic in the same folder, which the notebook would happily read.
    if not args.section:
        stray = sorted(
            p.name for p in d.glob("section_*")
            if p.is_file() and p not in set(files.values())
        )
        if stray:
            print(f"\n  note: {len(stray)} file(s) in {HUMAN_DIR}/ that this script does "
                  f"not use: {', '.join(stray[:6])}"
                  + (" ..." if len(stray) > 6 else ""))

    if not files:
        print(f"\n  Nothing recorded yet. Read sheets are in {paths.p('sections')}.")
        return 1

    result = qc.check_run(files, expected=words, missing=missing)
    # A single-section run is a spot check mid-recording; the others are simply
    # not recorded yet, and saying so as a failure would be noise.
    if args.section:
        result["run_fails"] = [m for m in result["run_fails"] if "no recording" not in m]
        result["ok"] = not result["run_fails"] and all(r.ok for r in result["sections"])

    for sec in missing:
        result["sections"].append(qc.SectionReport(
            section=sec, ok=False, path=None,
            fails=[f"not recorded - read sheet is "
                   f"{readsheet.SECTIONS_DIR}/section_{sec:02d}.txt"]))
    result["sections"].sort(key=lambda r: r.section)

    print_report(result, source)

    if args.json_out:
        Path(args.json_out).write_text(qc.to_json(result), encoding="utf-8")
        print(f"  report -> {args.json_out}")

    has_warn = any(r.warns for r in result["sections"]) or bool(result["run_warns"])
    if not result["ok"] or (args.strict and has_warn):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
