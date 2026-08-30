"""Does the recording resolver find the right mapping for any topic?

Runs on generated silence, so it needs no real takes and no Whisper. The chin
case is included as a regression because it is the one that was solved by hand
and is therefore the one whose answer is known independently of this code.

    ..\\vo_coach\\.venv\\Scripts\\python.exe test_recordings.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from pipeline import recordings  # noqa: E402

SR = 8000  # silence, so nothing depends on the rate but the duration


def make(dirpath: Path, names: list[str], seconds: list[float]) -> None:
    dirpath.mkdir(parents=True, exist_ok=True)
    for name, secs in zip(names, seconds):
        sf.write(dirpath / name, np.zeros(int(SR * secs), dtype=np.float32), SR)


PASS = FAIL = 0


def check(label: str, got, want) -> None:
    global PASS, FAIL
    if got == want:
        PASS += 1
        print(f"  ok    {label}")
    else:
        FAIL += 1
        print(f"  FAIL  {label}\n          got  {got}\n          want {want}")


def shape(p: recordings.Proposal) -> list[list[str]]:
    return [[f.name for f in a.files] for a in p.assignments]


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="rec_test_"))
    try:
        # -- 1. THE CHIN REGRESSION ---------------------------------------
        # Six takes, five sections. Known answer: section 3 was read twice, so
        # 3.wav and 4.wav belong together. Solved by hand before this module
        # existed, which is what makes it a real test rather than a tautology.
        d = root / "chin"
        make(d, [f"{i}.wav" for i in range(1, 7)],
             [36.07, 81.40, 165.62, 87.73, 153.34, 78.17])
        p = recordings.propose(d, {1: 88, 2: 178, 3: 508, 4: 352, 5: 170})
        print("chin (6 files, 5 sections)")
        check("groups files correctly",
              shape(p), [["1.wav"], ["2.wav"], ["3.wav", "4.wav"],
                         ["5.wav"], ["6.wav"]])
        check("status ok", p.status, "ok")
        check("confident", p.confidence, "high")
        check("multi-take section gets _a/_b targets",
              [n for _, n in p.assignments[2].targets()],
              ["section_03_a.wav", "section_03_b.wav"])
        check("single-take section gets a plain name",
              [n for _, n in p.assignments[0].targets()], ["section_01.wav"])
        print(f"  note: {p.note}")

        # -- 2. THE ORDINARY CASE: one file per section --------------------
        d = root / "simple"
        make(d, ["a.wav", "b.wav", "c.wav"], [30.0, 60.0, 45.0])
        p = recordings.propose(d, {1: 75, 2: 150, 3: 112})
        print("\none file per section")
        check("1:1 in order", shape(p), [["a.wav"], ["b.wav"], ["c.wav"]])
        check("status ok", p.status, "ok")

        # -- 3. NUMERIC SORT, NOT LEXICOGRAPHIC ----------------------------
        # 10.wav must come after 2.wav. Lexicographic order would put it second
        # and shift every section by one.
        d = root / "tenplus"
        make(d, [f"{i}.wav" for i in range(1, 11)], [30.0] * 10)
        check("\n10 files sort numerically",
              [f.name for f in recordings.list_audio(d)][:3],
              ["1.wav", "2.wav", "3.wav"])

        # -- 4. ALREADY CANONICAL: do not touch ----------------------------
        d = root / "canon"
        make(d, ["section_01.wav", "section_02.wav"], [30.0, 60.0])
        p = recordings.propose(d, {1: 75, 2: 150})
        print("\nalready named section_NN")
        check("status already", p.status, "already")
        check("no rename needed", p.needs_rename, False)

        # -- 5. NOT ENOUGH AUDIO -------------------------------------------
        d = root / "short"
        make(d, ["1.wav", "2.wav"], [30.0, 60.0])
        p = recordings.propose(d, {1: 75, 2: 150, 3: 100})
        print("\nfewer files than sections")
        check("status impossible", p.status, "impossible")

        # -- 6. AMBIGUOUS: EQUAL SECTIONS, NO EVIDENCE ---------------------
        # Four identical files over three equal sections: several groupings
        # score the same, so the honest answer is "I cannot tell".
        d = root / "ambig"
        make(d, ["1.wav", "2.wav", "3.wav", "4.wav"], [30.0] * 4)
        p = recordings.propose(d, {1: 75, 2: 75, 3: 75})
        print("\nno evidence to choose between groupings")
        check("status ambiguous", p.status, "ambiguous")
        check("says so", p.confidence, "low")

        # -- 7. APPLY IS REVERSIBLE ----------------------------------------
        d = root / "apply"
        make(d, [f"{i}.wav" for i in range(1, 7)],
             [36.07, 81.40, 165.62, 87.73, 153.34, 78.17])
        p = recordings.propose(d, {1: 88, 2: 178, 3: 508, 4: 352, 5: 170})
        recordings.apply(d, p)
        print("\napply then revert")
        check("renamed onto the canonical scheme",
              sorted(f.name for f in recordings.list_audio(d)),
              ["section_01.wav", "section_02.wav", "section_03_a.wav",
               "section_03_b.wav", "section_04.wav", "section_05.wav"])
        check("wrote a provenance record", (d / "naming.json").is_file(), True)

        # build_video must now find every section - the whole point.
        from build_video import find_recording
        found = {s: [f.name for f in find_recording(d, s)] for s in range(1, 6)}
        check("find_recording resolves all five sections", found, {
            1: ["section_01.wav"], 2: ["section_02.wav"],
            3: ["section_03_a.wav", "section_03_b.wav"],
            4: ["section_04.wav"], 5: ["section_05.wav"]})

        recordings.revert(d)
        check("revert restores the originals",
              sorted(f.name for f in recordings.list_audio(d)),
              ["1.wav", "2.wav", "3.wav", "4.wav", "5.wav", "6.wav"])
        check("revert removes the record", (d / "naming.json").exists(), False)

        # -- 8. A PERMUTATION MUST NOT LOSE A FILE -------------------------
        # `section_01.wav` is both a source and someone else's destination.
        # Renaming in place would destroy one of them halfway through, so apply
        # stages every move through a temporary name. Durations identify which
        # file ended up where.
        d = root / "permute"
        make(d, ["1.wav", "2.wav", "section_01.wav"], [30.0, 60.0, 99.0])
        p = recordings.propose(d, {1: 75, 2: 150, 3: 247})
        recordings.apply(d, p)
        print("\npermutation (a source is also a destination)")
        check("all three files survive",
              sorted(f.name for f in recordings.list_audio(d)),
              ["section_01.wav", "section_02.wav", "section_03.wav"])
        check("each landed on the right name",
              [round(recordings.duration(d / n))
               for n in ("section_01.wav", "section_02.wav", "section_03.wav")],
              [30, 60, 99])

        # -- 9. APPLY REFUSES TO CLOBBER A BYSTANDER -----------------------
        # A destination that is not itself being moved would be overwritten, so
        # nothing at all is renamed and the operator is told which file to deal
        # with. Built by hand because the resolver never proposes one.
        d = root / "clobber"
        make(d, ["1.wav", "occupied.wav"], [30.0, 60.0])
        (d / "section_01.wav").write_bytes((d / "occupied.wav").read_bytes())
        bad = recordings.Proposal(
            status="ok",
            assignments=[recordings.Assignment(1, [d / "1.wav"], 30.0, 75)])
        print("\nrefuses to overwrite a bystander")
        try:
            recordings.apply(d, bad)
            check("raised", False, True)
        except FileExistsError as e:
            check("raised FileExistsError", True, True)
            check("names the file", "section_01.wav" in str(e), True)
        check("1.wav was not moved", (d / "1.wav").is_file(), True)

    finally:
        shutil.rmtree(root, ignore_errors=True)

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
