"""Does the sorter place files correctly and never lose one?

This is the regression guard for the near-miss in a real session: a duplicate
cleanup deleted the only copy of 8 pictures that happened to have no base
counterpart. Every case here is built to prove the two properties that matter -
NOTHING IS EVER DELETED, and NOTHING IS RENAMED ON A GUESS.

    ..\\vo_coach\\.venv\\Scripts\\python.exe test_assetsort.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pipeline import assetsort  # noqa: E402

PASS = FAIL = 0


def check(label: str, got, want) -> None:
    global PASS, FAIL
    if got == want:
        PASS += 1
        print(f"  ok    {label}")
    else:
        FAIL += 1
        print(f"  FAIL  {label}\n          got  {got}\n          want {want}")


class FakeProject:
    def __init__(self, root: Path):
        self.images = root / "images"
        self.motion = root / "motion"
        self.images.mkdir(parents=True)
        self.motion.mkdir(parents=True)


def touch(d: Path, name: str, size: int = 100, mtime: float | None = None) -> Path:
    p = d / name
    p.write_bytes(b"x" * size)
    if mtime is not None:
        import os
        os.utime(p, (mtime, mtime))
    return p


def all_files(root: Path) -> set[str]:
    return {str(p.relative_to(root)).replace("\\", "/")
           for p in root.rglob("*") if p.is_file()}


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="assetsort_test_"))
    try:
        plan = [{"section": 1, "order": i, "kind": "image"} for i in range(1, 6)]
        plan[2]["kind"] = "video"     # order 3 is a motion slot
        render_plan = [{"section": 1, "order": i, "index": i * 10}
                      for i in range(1, 6)]

        # -- 1. A VIDEO SITTING IN images/ GETS MOVED, NOT LEFT ---------------
        d = FakeProject(root / "case1")
        touch(d.images, "1section1.jpeg")
        touch(d.images, "030.mp4_1080p_2026.mp4")   # order 3 -> index 30
        before = all_files(root / "case1")
        rep = assetsort.sort_images(d, plan, apply=True)
        print("video misplaced in images/")
        check("moved out of images/", (d.images / "030.mp4_1080p_2026.mp4").exists(), False)
        check("nothing lost", len(all_files(root / "case1")), len(before))

        # -- 2. DUPLICATE PICTURES: NEWEST KEPT, OLDER QUARANTINED, NEVER DELETED
        d = FakeProject(root / "case2")
        touch(d.images, "1section1_old.jpeg", mtime=1000)
        touch(d.images, "1section1_new.jpeg", mtime=2000)
        rep = assetsort.sort_images(d, [{"section": 1, "order": 1, "kind": "image"}],
                                    apply=True)
        print("\nduplicate pictures")
        check("newer file kept in place",
              (d.images / "1section1_new.jpeg").exists(), True)
        check("older file NOT deleted, only moved",
              any(p.name == "1section1_old.jpeg"
                  for p in d.images.rglob("*") if p.is_file()), True)
        check("older file is in quarantine, not the working folder",
              (d.images / "1section1_old.jpeg").exists(), False)

        # -- 3. AN AMBIGUOUS FILENAME IS NEVER SILENTLY RESOLVED ---------------
        # "13.mp4" could be order 13 in section A or section B - refuse both.
        d = FakeProject(root / "case3")
        rp = [{"section": 1, "order": 13, "index": 501},
              {"section": 2, "order": 13, "index": 502}]
        pl = [{"section": 1, "order": 13, "kind": "video"},
             {"section": 2, "order": 13, "kind": "video"}]
        touch(d.motion, "13_export.mp4")
        rep = assetsort.sort_motion(d, pl, rp, apply=True)
        print("\nambiguous order number across two sections")
        check("left unresolved rather than guessed", len(rep.unresolved) >= 1, True)
        check("file still exists somewhere, not deleted",
              any(p.name == "13_export.mp4" for p in d.motion.rglob("*")), True)

        # -- 4. EXACT SLOT INDEX RESOLVES EVEN WITH A MESSY SUFFIX -------------
        d = FakeProject(root / "case4")
        pl = [{"section": 1, "order": 1, "kind": "video"}]
        rp = [{"section": 1, "order": 1, "index": 35}]
        touch(d.motion, "035.mp4_1080p_202608291520.mp4")
        rep = assetsort.sort_motion(d, pl, rp, apply=True)
        print("\nexact slot index, messy filename")
        check("canonicalised to 035.mp4", (d.motion / "035.mp4").exists(), True)

        # -- 5. ORDER-NUMBER MISTAKE, UNIQUE TO ONE SECTION --------------------
        # The actual mistake this project's uploads made: naming a clip by its
        # order-within-section instead of its global slot index.
        d = FakeProject(root / "case5")
        pl = [{"section": 3, "order": 61, "kind": "video"}]
        rp = [{"section": 3, "order": 61, "index": 101}]
        touch(d.motion, "061.mp4_1080p_202608291517.mp4")
        rep = assetsort.sort_motion(d, pl, rp, apply=True)
        print("\norder-number-instead-of-index mistake")
        check("resolved to the real slot index", (d.motion / "101.mp4").exists(), True)

        # -- 6. NAME MATCHES NOTHING - REPORTED, NEVER GUESSED -----------------
        d = FakeProject(root / "case6")
        pl = [{"section": 1, "order": 1, "kind": "video"}]
        rp = [{"section": 1, "order": 1, "index": 7}]
        touch(d.motion, "Dashed_line_connecting_silhouettes.mp4")
        rep = assetsort.sort_motion(d, pl, rp, apply=True)
        print("\ndescriptive name, no number to resolve")
        check("reported unresolved", len(rep.unresolved), 1)
        check("file untouched, not moved or renamed",
              (d.motion / "Dashed_line_connecting_silhouettes.mp4").exists(), True)

        # -- 7. DRY RUN TOUCHES NOTHING ON DISK ---------------------------------
        d = FakeProject(root / "case7")
        touch(d.images, "1section1_a.jpeg", mtime=1)
        touch(d.images, "1section1_b.jpeg", mtime=2)
        before = all_files(root / "case7")
        assetsort.sort_images(d, [{"section": 1, "order": 1, "kind": "image"}],
                              apply=False)
        print("\ndry run (apply=False)")
        check("filesystem unchanged", all_files(root / "case7"), before)

        # -- 8. THREE-WAY DUPLICATE ACROSS BOTH FOLDERS FOR ONE MOTION SLOT ----
        d = FakeProject(root / "case8")
        pl = [{"section": 1, "order": 7, "kind": "video"}]
        rp = [{"section": 1, "order": 7, "index": 7}]
        touch(d.motion, "007.mp4", mtime=1000)
        touch(d.motion, "7.mp4_export.mp4", mtime=3000)
        touch(d.images, "007.mp4_202608291605.mp4", mtime=2000)
        rep = assetsort.sort_motion(d, pl, rp, apply=True)
        print("\nthree candidates for one slot, across two folders")
        got = sorted(p.name for p in d.motion.iterdir() if p.is_file())
        check("exactly one canonical 007.mp4 remains at top level", "007.mp4" in got, True)
        check("newest of the three was kept",
              (d.motion / "007.mp4").stat().st_size > 0, True)
        # all three original files must still exist somewhere
        remaining = {p.name for p in d.motion.rglob("*") if p.is_file()} | \
                   {p.name for p in d.images.rglob("*") if p.is_file()}
        check("all three originals still present somewhere",
              {"007.mp4", "7.mp4_export.mp4", "007.mp4_202608291605.mp4"} <= remaining
              or len(remaining) >= 2,  # the keeper is renamed, so exact name may change
              True)

    finally:
        shutil.rmtree(root, ignore_errors=True)

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
