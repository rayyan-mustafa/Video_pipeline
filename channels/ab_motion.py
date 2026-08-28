"""A/B a slice of a video across motion providers, at production quality.

WHY A SLICE AND NOT THE WHOLE VIDEO. The decision this answers - "is motion
worth it, and is generated motion worth paying a GPU for" - is visible in
thirty seconds and costs eight minutes to ask badly. Rendering a window means
the loser is discarded after one coffee rather than one evening, and it means
the expensive provider can be evaluated on fifteen slots instead of two hundred.

WHY IT DOES NOT NEED THE VOICE STAGE. Holds come from `render_plan.json` when
the voice stage has produced one, and otherwise from the word counts already in
`script_plan.json` at the planner's wpm - which is exactly what `--stage check`
does, and exactly the estimate the pacing report is built on. Whisper's job is
to place cuts ON THE RIGHT WORD; it has nothing to say about whether a picture
should drift while it is up. So a visual A/B that waited on transcription would
be waiting for an answer to a different question.

WHAT IT DELIBERATELY DOES NOT DO: add audio. The window is silent. Narration
over a 30-second slice tells you about the writing and the read, and this is
asking about the pictures.

    python ab_motion.py --project wierdbiology/chin --from 120 --seconds 30
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from pipeline import clips, motion, pacing
from pipeline.project import Project


def load_slots(proj: Project) -> tuple[list[clips.Slot], str]:
    """Every slot in the video, with the best hold estimate available.

    Returns the slots and a one-line description of where the holds came from,
    because "measured" and "estimated" are different claims and the report
    should not blur them.
    """
    if proj.render_plan.exists():
        entries = json.loads(proj.render_plan.read_text(encoding="utf-8"))
        rows = entries["entries"] if isinstance(entries, dict) else entries
        slots = [clips.Slot(i, Path(e["path"]), float(e["duration"]))
                 for i, e in enumerate(rows)]
        return slots, "measured from the voice stage (render_plan.json)"

    if not proj.plan.exists():
        raise SystemExit(
            f"no plan at {proj.plan} - run plan_script.py first")

    plan = json.loads(proj.plan.read_text(encoding="utf-8"))
    wpm = int(plan.get("wpm", 150))
    slots = []
    for i, img in enumerate(plan["images"]):
        # Resolve through the Project rather than the recorded path: the plan
        # stores paths relative to whatever directory planning ran in, and this
        # may not be that directory.
        path = proj.images / img["filename"]
        hold = pacing.seconds_for_words(int(img.get("words", 0)), wpm)
        slots.append(clips.Slot(i, path, hold))
    return slots, f"estimated from word counts at {wpm} wpm (no voice stage yet)"


def pick_window(slots: list[clips.Slot], start_s: float,
                seconds: float) -> list[clips.Slot]:
    """The slots covering [start_s, start_s + seconds) on the estimated timeline.

    Whole slots only - a half-rendered picture would compare the cut rhythm
    against a truncation artefact.
    """
    t = 0.0
    picked = []
    for s in slots:
        end = t + s.hold
        if end > start_s and t < start_s + seconds:
            picked.append(s)
        t = end
        if t >= start_s + seconds:
            break
    if not picked:
        raise SystemExit(
            f"no slots in that window - the video is only {t:.0f}s long")
    return picked


def report_window(window: list[clips.Slot], source: str) -> None:
    holds = [s.hold for s in window]
    v = pacing.verdict(holds)
    print(f"\nWINDOW  {len(window)} slots, {sum(holds):.1f}s")
    print(f"  holds are {source}")
    print(f"  body average   {v['body_avg']:.2f}s   ({v['rating']})")
    print(f"  over {pacing.HARD_MAX:.0f}s        {v['over_hard']} of {v['n']}")
    print(f"  accents        {v['accents']} of {v['n']}")

    static = sum(1 for s in window if motion.move_for(s.index, s.hold).is_static)
    rescued = sum(1 for s in window if s.hold > pacing.HARD_MAX)
    print(f"\nMOTION  {len(window) - static} slots move, {static} held static as accents")
    print(f"        {rescued} slot(s) over the ceiling get full-amplitude motion")
    print("\n  idx   hold  move         zoom            why")
    for s in window:
        d = motion.describe(s.index, s.hold)
        print(f"  {s.index:>3}  {d['hold']:>5.2f}  {d['move']:<11}  "
              f"{d['zoom_from']:.3f}->{d['zoom_to']:.3f}   {d['reason']}")


def build_variant(name: str, window: list[clips.Slot], out_dir: Path,
                  width: int, height: int, fps: int) -> tuple[Path, float]:
    provider = clips.PROVIDERS[name]()
    cache = out_dir / "clips" / name
    print(f"\n[{name}] rendering {len(window)} clips...")
    t0 = time.time()

    def progress(i: int, n: int, note: str) -> None:
        if i == n:
            print(f"  {note}")
        elif i % 5 == 0 or i == 1:
            print(f"  {i}/{n}", end="\r")

    paths = clips.render_sequence(window, provider, cache, width, height, fps,
                                  on_progress=progress)
    out = out_dir / f"{name}.mp4"
    clips.concat_clips(paths, out, work=out_dir)
    took = time.time() - t0
    print(f"  {out.name}  {clips.probe_duration(out):.2f}s  "
          f"{out.stat().st_size / 1e6:.1f} MB  in {took:.0f}s")
    return out, took


def side_by_side(left: Path, right: Path, out: Path,
                 width: int, height: int) -> None:
    """One file, both variants, same timeline. For 'does it feel different'.

    Judge shimmer on the full-size single files instead - halving the width to
    fit two panels hides exactly the pixel-level artefact that supersampling
    exists to prevent.
    """
    w, h = width // 2, height // 2
    fc = (f"[0:v]scale={w}:{h}[l];[1:v]scale={w}:{h}[r];[l][r]hstack=inputs=2")
    clips._run(["ffmpeg", "-y", "-i", str(left), "-i", str(right),
                "-filter_complex", fc, "-c:v", "libx264", "-preset", "medium",
                "-crf", "20", "-pix_fmt", "yuv420p", str(out)], "side-by-side")
    print(f"\n  {out.name}  left={left.stem}  right={right.stem}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", required=True, help="<channel>/<topic>")
    ap.add_argument("--variants", default="still,kenburns",
                    help="comma-separated providers to render and compare")
    ap.add_argument("--from", dest="start", type=float, default=0.0,
                    help="window start, seconds into the video")
    ap.add_argument("--seconds", type=float, default=30.0,
                    help="window length. 30 is enough to judge; 8 minutes is not "
                         "more informative, only slower.")
    ap.add_argument("--out", type=Path, default=None,
                    help="output dir, default <project>/build/ab")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--supersample", type=int, default=motion.SUPERSAMPLE,
                    help="anti-shimmer oversampling for kenburns. Lower is "
                         "faster and makes the line art crawl.")
    ap.add_argument("--no-sbs", action="store_true",
                    help="skip the side-by-side file")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the window and the move table, render nothing")
    args = ap.parse_args()

    if not clips.have_ffmpeg():
        raise SystemExit("ffmpeg and ffprobe must be on PATH")

    motion.SUPERSAMPLE = args.supersample
    proj = Project.parse(args.project)
    slots, source = load_slots(proj)
    window = pick_window(slots, args.start, args.seconds)

    print(f"Project {proj.channel}/{proj.topic}")
    print(f"  {len(slots)} slots total, window from {args.start:.0f}s")
    report_window(window, source)

    if args.dry_run:
        print("\n(dry run - nothing rendered)")
        return

    out_dir = args.out or (proj.build / "ab")
    out_dir.mkdir(parents=True, exist_ok=True)

    names = [n.strip() for n in args.variants.split(",") if n.strip()]
    unknown = [n for n in names if n not in clips.PROVIDERS]
    if unknown:
        raise SystemExit(f"unknown variant(s) {unknown} - "
                         f"have {sorted(clips.PROVIDERS)}")

    built: list[Path] = []
    for name in names:
        path, _ = build_variant(name, window, out_dir, args.width, args.height,
                                args.fps)
        built.append(path)

    if len(built) == 2 and not args.no_sbs:
        side_by_side(built[0], built[1], out_dir / "compare_sbs.mp4",
                     args.width, args.height)

    print(f"\nWatch them in {out_dir}")
    print("  judge shimmer and crop on the full-size files, not the side-by-side")


if __name__ == "__main__":
    main()
