"""Local end-to-end build: read sheets + recordings + hand-supplied images -> mp4.

This is the notebook's Steps 17, 18 and 25 running on the laptop, plus the one
piece the notebook never had: a mapping from hand-named image files onto the
aligned timeline (see pipeline/imageplan.py).

    python build_video.py --sections sections --recordings voice_recordings \
                          --images images --out final_video.mp4

STAGE ORDER IS NOT NEGOTIABLE, and the reason is the same one the notebook
records: mastering with the "clip" profile TRIMS SILENCE, which moves every
timestamp. So each section is mastered FIRST and aligned SECOND - the aligner has
to see the audio that actually ships. The concatenated voiceover is then mastered
a second time with the length-locked "timeline" profile, which asserts it did not
change the sample count, because the picture timings are already derived from it.

Do not master the section files before handing them here. Two passes is the
design; three sounds squashed.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy.signal as sps
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from pipeline import imageplan
from pipeline.align import align, anchor_report
from pipeline.scenesplit import make_scenes
from pipeline.project import Project
from pipeline.voice import master, report

TARGET_SR = 48000

# YouTube normalises uploads to about -14 LUFS. Delivering at -14 means it applies
# no gain change; delivering louder means it turns the whole video DOWN, which
# undoes the mastering. The per-section pass stays at the chain's -16 default,
# because that audio is an intermediate and gets re-normalised here anyway.
DELIVERY_LUFS = -14.0

AUDIO_EXTS = ("wav", "flac", "m4a", "mp3")


# ---------------------------------------------------------------------------
#  Inputs
# ---------------------------------------------------------------------------
def load_sections(sections_dir: Path) -> list[dict]:
    """section_01.txt ... -> [{'sentence','section'}], the shape build_chunks wants."""
    files = sorted(sections_dir.glob("section_[0-9]*.txt"))
    files = [f for f in files if not f.stem.endswith("_marked")]
    if not files:
        raise SystemExit(f"no section_NN.txt files in {sections_dir}")

    tagged = []
    for f in files:
        n = int(re.search(r"(\d+)", f.stem).group(1))
        text = f.read_text(encoding="utf-8").strip()
        # Sentence split on terminal punctuation. Deliberately dumb: build_chunks
        # merges the short ones and splits the long ones anyway, so precision here
        # buys nothing.
        for s in re.split(r"(?<=[.!?])\s+", text):
            if s.strip():
                tagged.append({"sentence": s.strip(), "section": n})
    return tagged


def _split_mode(proj, override: str | None) -> str:
    """Which split the timeline must use.

    THE TIMELINE AND THE PLAN MUST AGREE. Scene N in script_plan.json has to be
    scene N in scenes_timeline.json, and the split mode is what decides where
    scene boundaries fall. So the plan's own recorded mode wins by default - an
    explicit --split is only for building a timeline with no plan beside it.
    Getting this wrong does not corrupt anything silently: the fingerprint check
    in plan_from_binding() catches it. But it wastes a Whisper pass.
    """
    if override:
        return override
    if proj.plan.exists():
        try:
            return json.loads(proj.plan.read_text(encoding="utf-8")).get("split", "chunk")
        except Exception:
            pass
    return "chunk"


def find_recording(rec_dir: Path, sec: int) -> list[Path]:
    """Every file belonging to one section, in order. Empty list if none.

    MULTI-PART SECTIONS. A long section often gets recorded in two takes, so
    `section_03_a.wav` + `section_03_b.wav` are concatenated in filename order
    and treated as one section. That is better than asking the reader to splice
    them by hand: splicing changes duration, and every timestamp downstream is
    derived from the duration this stage measures.

    A whole-section file wins over parts if both somehow exist, because an
    explicit `section_03.wav` is the less ambiguous statement of intent.
    """
    for stem in (f"section_{sec:02d}", f"section_{sec}"):
        for ext in AUDIO_EXTS:
            hits = glob.glob(str(rec_dir / f"{stem}.{ext}"))
            if hits:
                return [Path(hits[0])]

    parts: list[Path] = []
    for stem in (f"section_{sec:02d}", f"section_{sec}"):
        for ext in AUDIO_EXTS:
            parts += [Path(p) for p in glob.glob(str(rec_dir / f"{stem}_*.{ext}"))]
    return sorted(set(parts))


def _read_section_audio(paths: list[Path]) -> tuple[np.ndarray, int]:
    """Load one section's audio, concatenating parts and resampling to TARGET_SR."""
    chunks = []
    for p in paths:
        x, sr = sf.read(p, always_2d=False)
        x = np.asarray(x, dtype=np.float64)
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sr != TARGET_SR:
            g = np.gcd(TARGET_SR, int(sr))
            x = sps.resample_poly(x, TARGET_SR // g, int(sr) // g)
        chunks.append(x)
    return np.concatenate(chunks) if len(chunks) > 1 else chunks[0], TARGET_SR


# ---------------------------------------------------------------------------
#  Stage 1 - master, align, concatenate
# ---------------------------------------------------------------------------
def build_voice(sections_dir: Path, rec_dir: Path, work: Path,
                whisper_size: str = "base",
                split: str = "chunk",
                scenes_override: list[dict] | None = None,
                raw_voice: bool = False) -> tuple[Path, Path]:
    # PREFER THE PLAN'S OWN SCENES. script_plan.json already stores the exact
    # scene texts it bound the pictures to, so reusing them removes every chance
    # of the two sides diverging - and it is the only way "match" mode can work
    # here at all, since that split is defined by the image folder rather than by
    # a rule this function could re-apply.
    if scenes_override:
        chunks = [{"text": s["text"], "section": s["section"]} for s in scenes_override]
        print(f"  using the plan's {len(chunks)} scenes verbatim (split={split!r})")
    else:
        tagged = load_sections(sections_dir)
        chunks = make_scenes(tagged, split)
        print(f"  split mode {split!r} -> {len(chunks)} scenes")
    sections = sorted({c["section"] for c in chunks})

    missing = [s for s in sections if not find_recording(rec_dir, s)]
    if missing:
        raise SystemExit(
            f"missing recordings for section(s) {missing} in {rec_dir}/.\n"
            f"Expected section_01.wav ... section_{max(sections):02d}.wav"
        )

    from faster_whisper import WhisperModel

    # BUG FIX (same one the notebook's cell039 records): float16 is invalid on CPU
    # and on pre-Volta cards, and faster-whisper's error for that case is not
    # obvious. Pick by capability rather than assuming.
    device, ctype = "cpu", "int8"
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            ctype = "float16" if torch.cuda.get_device_capability(0)[0] >= 7 else "int8"
    except ImportError:
        pass

    if whisper_size.startswith("large") and device == "cpu":
        print(f"  NOTE: {whisper_size} on CPU is roughly 10x slower than 'base'.")
        print(f"        Expect tens of minutes. Use --whisper base locally, or run")
        print(f"        this stage on a GPU (see COLAB_VOICE.md).")

    print(f"Loading Whisper ({whisper_size}, {device}/{ctype})...")
    wm = WhisperModel(whisper_size, device=device, compute_type=ctype)

    pieces, scenes, cursor, quality = [], [], 0.0, []
    word_track: list[dict] = []
    print()
    for sec in sections:
        paths = find_recording(rec_dir, sec)
        sec_chunks = [c for c in chunks if c["section"] == sec]
        if len(paths) > 1:
            print(f"  section {sec}: joining {len(paths)} parts "
                  f"({', '.join(p.name for p in paths)})")
        x, sr = _read_section_audio(paths)

        # "clip" trims silence, so it MUST run before alignment. "raw" changes
        # no tone and no length - the take is kept exactly as read, and only the
        # delivery level is set.
        y = master(x, sr, profile="raw" if raw_voice else "clip")

        tmp = work / f"_mastered_section_{sec:02d}.wav"
        sf.write(tmp, y.astype(np.float32), sr, subtype="PCM_16")
        total_s = len(y) / sr

        segs, _ = wm.transcribe(str(tmp), word_timestamps=True)
        words = [w for s in segs for w in (s.words or [])]
        spans, matched = align(words, [c["text"] for c in sec_chunks], total_s)

        # The number that decides whether pictures land right. word-match % is a
        # transcription-quality figure that cannot reach 100% on a script full of
        # proper nouns; what matters is whether each SCENE found a real word to
        # anchor its start on. See align.anchor_report().
        anchored, _ = anchor_report(words, [c["text"] for c in sec_chunks])
        n_anchored = sum(anchored)
        drifting = [i for i, a in enumerate(anchored) if not a]
        quality.append((sec, matched, len(sec_chunks), n_anchored, drifting))

        # KEEP THE WORD TIMESTAMPS. They used to be consumed by align() and
        # dropped, which left the picture planner interpolating a word's position
        # linearly inside its scene - an approximation, when the exact answer had
        # already been computed and discarded. These are the words actually
        # spoken, so they are the ground truth for where any cut belongs.
        for w in words:
            word_track.append({
                "section": sec,
                "w": w.word.strip(),
                "s": round(cursor + float(w.start), 3),
                "e": round(cursor + float(w.end), 3),
            })

        for c, (a, b) in zip(sec_chunks, spans):
            scenes.append({
                "scene_index": len(scenes),
                "start": round(cursor + a, 3),
                "end": round(cursor + b, 3),
                "duration": round(b - a, 3),
                "text": c["text"],
                "section": sec,
            })
        pieces.append(y)
        cursor += total_s
        print(f"  section {sec:>2}  {total_s:6.1f}s  {len(sec_chunks):>3} scenes  "
              f"word match {matched * 100:5.1f}%   "
              f"anchored {n_anchored}/{len(sec_chunks)}")
        tmp.unlink()

    del wm

    full = np.concatenate(pieces)
    vo_raw = work / "vo_raw.wav"
    sf.write(vo_raw, full.astype(np.float32), TARGET_SR, subtype="PCM_16")
    audio_seconds = len(full) / TARGET_SR

    drift = audio_seconds - scenes[-1]["end"]
    scenes[-1]["duration"] = round(scenes[-1]["duration"] + drift, 3)
    scenes[-1]["end"] = round(audio_seconds, 3)

    tl = work / "scenes_timeline.json"
    tl.write_text(json.dumps(scenes, indent=2), encoding="utf-8")

    # The last word's end can sit slightly before the audio's, so clamp rather
    # than let a cut be placed past the file.
    for w in word_track:
        w["s"] = min(w["s"], audio_seconds)
        w["e"] = min(w["e"], audio_seconds)
    (work / "words.json").write_text(json.dumps(word_track), encoding="utf-8")
    print(f"  {len(word_track)} word timestamps kept -> words.json")

    tot_scenes = sum(q[2] for q in quality)
    tot_anchored = sum(q[3] for q in quality)
    worst = min(quality, key=lambda q: q[1])
    print(f"\n  {len(scenes)} scenes, {audio_seconds / 60:.2f} min")
    print(f"  scenes anchored to a real timestamp: {tot_anchored}/{tot_scenes} "
          f"({tot_anchored / max(tot_scenes, 1) * 100:.1f}%)")
    print(f"  worst word match: section {worst[0]} at {worst[1] * 100:.1f}% "
          f"- a transcription figure, not a timing one")

    # ONLY unanchored scenes can actually drift. A scene that found its opening
    # word carries that word's exact Whisper timestamp regardless of how many
    # later words were misheard, so listing the interpolated ones by name is the
    # actionable output - not the percentage.
    unanchored = [(q[0], q[4]) for q in quality if q[4]]
    if unanchored:
        print("\n  Interpolated scenes (no matching word found) - the only ones")
        print("  whose picture can drift:")
        for sec, idxs in unanchored:
            sec_texts = [c["text"] for c in chunks if c["section"] == sec]
            for i in idxs:
                print(f"    section {sec} scene {i}: {sec_texts[i][:62]!r}")
    else:
        print("  every scene anchored - no interpolated boundaries anywhere.")

    # ---- Delivery master: length-locked, asserted -------------------------
    print("\nMastering the full voiceover (timeline profile, length locked)...")
    n_before = len(full)
    y = master(full, TARGET_SR, profile="raw" if raw_voice else "timeline",
               target_lufs=DELIVERY_LUFS)
    assert len(y) == n_before, "mastering changed the length"
    full_vo = work / "full_vo.wav"
    sf.write(full_vo, y.astype(np.float32), TARGET_SR, subtype="PCM_16")
    print()
    report(str(full_vo))
    return tl, full_vo


# ---------------------------------------------------------------------------
#  Stage 3 - mux
# ---------------------------------------------------------------------------
def mux(plan_entries: list[dict], audio: Path, out: Path, work: Path,
        width: int, height: int, fps: int) -> None:
    from PIL import Image

    print("\nPreflight:")
    errors = []

    # A file that is 0 bytes or truncated passes exists() and then kills ffmpeg
    # mid-encode, after everything upstream has already been paid for.
    bad = []
    for e in plan_entries:
        try:
            with Image.open(e["path"]) as im:
                im.verify()
        except Exception:
            bad.append(Path(e["path"]).name)
    if bad:
        errors.append(f"{len(bad)} unreadable image(s): {bad[:10]}")
    print(f"  images readable          {'OK' if not bad else 'CORRUPT ' + str(len(bad))}")

    audio_len = sf.info(str(audio)).frames / float(sf.info(str(audio)).samplerate)
    video_len = sum(e["duration"] for e in plan_entries)
    drift = abs(video_len - audio_len)
    print(f"  audio length             {audio_len:.2f}s")
    print(f"  image durations sum      {video_len:.2f}s")
    print(f"  drift                    {drift:.3f}s {'OK' if drift <= 0.5 else 'TOO HIGH'}")
    if drift > 0.5:
        errors.append(f"video/audio mismatch of {drift:.2f}s")

    free_gb = shutil.disk_usage(work).free / 1024 ** 3
    if free_gb < 2.0:
        errors.append(f"only {free_gb:.1f} GB free - ffmpeg will fail mid-encode")

    if errors:
        print("\nPREFLIGHT FAILED:")
        for e in errors:
            print(f"  - {e}")
        raise SystemExit(1)
    print("  all checks passed")

    # The concat demuxer ignores the duration of the FINAL entry, so the last file
    # is repeated with no duration to close the timeline out.
    concat = work / "concat_list.txt"
    with concat.open("w", encoding="utf-8") as f:
        for e in plan_entries:
            f.write(f"file '{Path(e['path']).resolve().as_posix()}'\n")
            f.write(f"duration {e['duration']}\n")
        f.write(f"file '{Path(plan_entries[-1]['path']).resolve().as_posix()}'\n")

    vf = (f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
          f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:white,format=yuv420p")

    # ffmpeg 7 deprecated -vsync for -fps_mode and some builds error on it.
    # Detect rather than guess.
    helptext = subprocess.run(["ffmpeg", "-hide_banner", "-h", "full"],
                              capture_output=True, text=True).stdout
    fps_flag = ["-fps_mode", "cfr"] if "fps_mode" in helptext else ["-vsync", "cfr"]

    print(f"\nMuxing to {width}x{height} @ {fps}fps...")
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", str(concat),
        "-i", str(audio),
        "-vf", vf, "-r", str(fps), *fps_flag,
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-movflags", "+faststart", "-shortest",
        str(out),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stderr[-3000:])
        raise SystemExit("ffmpeg failed - see the log above.")

    print(f"\n  {out}  ({out.stat().st_size / 1e6:.1f} MB)")


# ---------------------------------------------------------------------------
#  Stage 0 - check the pictures before anything expensive happens
# ---------------------------------------------------------------------------
def check_images(proj, images_dir: Path, wpm: int) -> None:
    """Report per-section image counts and the hold time they imply.

    The distribution is the thing worth looking at. A section that gets a tenth
    of the pictures and a third of the words cuts three times slower than the
    rest of the video, and that reads as a pacing mistake rather than a naming
    one - so it is reported as a ratio, not just a count.
    """
    if proj.plan.exists():
        chunks = json.loads(proj.plan.read_text(encoding="utf-8")).get("scenes", [])
    else:
        chunks = make_scenes(load_sections(proj.sections), "chunk")

    words = {}
    scenes = {}
    for c in chunks:
        words[c["section"]] = words.get(c["section"], 0) + len(c["text"].split())
        scenes[c["section"]] = scenes.get(c["section"], 0) + 1
    total_words = sum(words.values())
    est_total = total_words / wpm * 60.0

    images = imageplan.scan_images(images_dir)
    counts = {}
    for i in images:
        counts[i["section"]] = counts.get(i["section"], 0) + 1

    print(f"{len(images)} images in {images_dir}")
    print(f"script is {total_words} words, ~{est_total / 60:.1f} min at {wpm} wpm\n")
    print(f"  {'sec':>3} {'scenes':>7} {'images':>7} {'words':>6} "
          f"{'est span':>9} {'est hold':>9}")
    print("  " + "-" * 47)

    problems = []
    for sec in sorted(set(scenes) | set(counts)):
        n_img = counts.get(sec, 0)
        n_sc = scenes.get(sec, 0)
        w = words.get(sec, 0)
        span = w / wpm * 60.0
        hold = span / n_img if n_img else 0.0
        flag = ""
        if n_img == 0:
            flag = "  <- NO IMAGES"
            problems.append(f"section {sec} has no images at all")
        elif hold < 1.5:
            flag = "  <- very fast"
        elif hold > 15.0:
            flag = "  <- very slow"
        print(f"  {sec:>3} {n_sc:>7} {n_img:>7} {w:>6} "
              f"{span:>8.0f}s {hold:>8.1f}s{flag}")

    holds = [words.get(s, 0) / wpm * 60.0 / counts[s] for s in counts if counts[s]]
    if holds:
        print(f"\n  overall ~{est_total / len(images):.1f}s per image")
        if max(holds) > 2.5 * min(holds):
            problems.append(
                f"pacing is uneven - the slowest section holds {max(holds):.1f}s "
                f"per image and the fastest {min(holds):.1f}s. Move some pictures "
                f"between sections, or accept that the pacing changes."
            )

    orphan = [s for s in counts if s not in scenes]
    if orphan:
        problems.append(f"images name section(s) {orphan}, which the script "
                        f"does not have (it has {sorted(scenes)})")

    if problems:
        print("\n  problems:")
        for p in problems:
            print(f"    - {p}")
    else:
        print("\n  no problems - every section has pictures and the pacing is even")


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", required=True,
                    help="<channel>/<topic>, e.g. wierdbiology/chin")
    ap.add_argument("--images", type=Path, default=None,
                    help="override <project>/images")
    ap.add_argument("--out", type=Path, default=None,
                    help="override <project>/<topic>.mp4")
    ap.add_argument("--split", default=None,
                    choices=("chunk", "sentence", "dash"),
                    help="override the split mode. Defaults to whatever "
                         "script_plan.json recorded, so both sides agree.")
    ap.add_argument("--raw-voice", action="store_true",
                    help="keep the take exactly as read: no denoise, EQ, de-esser "
                         "or compressor, and no silence trim. Only the delivery "
                         "level is normalised to -14 LUFS, which changes no tone "
                         "and no sample count. Use when you like your own voice "
                         "and do not want it processed.")
    ap.add_argument("--ignore-plan", action="store_true",
                    help="ignore script_plan.json and divide each section between "
                         "whatever images are in the folder. Use this for pictures "
                         "generated outside this pipeline, whose names cannot match "
                         "a binding they were never made from.")
    ap.add_argument("--wpm", type=int, default=150,
                    help="reading speed used by --stage check to estimate holds")
    ap.add_argument("--whisper", default="base",
                    help="faster-whisper model: tiny/base/small/medium/large-v3. "
                         "large-v3 is the accurate one but wants a GPU; on CPU "
                         "prefer base. Alignment quality shows up as the word-match "
                         "percentage this stage prints.")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--stage", default="all",
                    choices=("check", "all", "voice", "plan", "mux"),
                    help="run one stage. 'voice' and 'plan' cache to --work, so a "
                         "re-run after renaming images does not re-transcribe.")
    args = ap.parse_args()

    proj = Project.parse(args.project)
    proj.make_dirs()
    args.work = proj.build
    args.sections = proj.sections
    args.recordings = proj.recordings
    args.images = args.images or proj.images
    args.out = args.out or proj.out

    print(f"Project {proj.channel}/{proj.topic}")
    print(proj.describe())
    print()

    tl_path = proj.timeline
    vo_path = proj.voice
    plan_path = proj.render_plan

    if args.stage == "check":
        # Runs BEFORE any recording exists, so a naming mistake across 122 files
        # is found while fixing it is a rename, not a re-render. Needs no audio:
        # hold times are estimated from the script's word counts at --wpm.
        check_images(proj, args.images, args.wpm)
        return

    if args.stage in ("all", "voice"):
        print("=" * 70)
        print("STAGE 1  master -> align -> concatenate")
        print("=" * 70)
        _override = None
        if proj.plan.exists() and not args.split:
            _override = json.loads(proj.plan.read_text(encoding="utf-8")).get("scenes")
        build_voice(args.sections, args.recordings, args.work, args.whisper,
                    split=_split_mode(proj, args.split), scenes_override=_override,
                    raw_voice=args.raw_voice)

    if args.stage in ("all", "plan"):
        print("\n" + "=" * 70)
        print("STAGE 2  images -> timed plan")
        print("=" * 70)
        if not tl_path.exists():
            raise SystemExit(f"{tl_path} not found - run --stage voice first")
        timeline = json.loads(tl_path.read_text(encoding="utf-8"))

        # Preferred path: a binding made from the script, so no division is
        # guessed. Falls back to dividing a section between whatever images are
        # in the folder only when no plan exists, or when told to ignore it.
        #
        # --ignore-plan exists for pictures made OUTSIDE this pipeline (a
        # different prompt set, a different agent, a different count). Their
        # filenames cannot match a binding they were never generated from, so
        # binding them is impossible and dividing the sections is the only
        # honest option. It is a flag rather than an automatic fallback because
        # silently switching strategies when a filename fails to match is how a
        # video gets built against the wrong pictures without anyone noticing.
        sp_path = proj.plan
        if sp_path.exists() and not args.ignore_plan:
            script_plan = json.loads(sp_path.read_text(encoding="utf-8"))
            entries, warnings = imageplan.plan_from_binding(
                script_plan, timeline, images_dir=args.images)
            print(f"  bound from {sp_path.name} "
                  f"(fingerprint {script_plan.get('fingerprint')})")
        else:
            why = ("--ignore-plan" if args.ignore_plan else "no script_plan.json")
            print(f"  {why} - dividing each section between the images present")
            images = imageplan.scan_images(args.images)
            wpath = args.work / "words.json"
            words = json.loads(wpath.read_text(encoding="utf-8")) if wpath.exists() else None
            if words:
                print(f"  using {len(words)} Whisper word timestamps for cut placement")
            else:
                print("  no words.json - cuts interpolated within scenes "
                      "(re-run --stage voice to get exact word timing)")
            entries, warnings = imageplan.plan(images, timeline, words=words)
        plan_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
        print()
        print(imageplan.render(entries, warnings))

        # Prove the plan tiles the audio rather than assuming it. The audio file
        # is measured here, not just at mux time, so a mismatch is caught before
        # an encode is paid for.
        audio_s = None
        if vo_path.exists():
            info = sf.info(str(vo_path))
            audio_s = info.frames / float(info.samplerate)
        checks, fails = imageplan.verify(entries, timeline, audio_s)
        print("\n  sync check:")
        for c in checks:
            print(f"    {c}")
        if fails:
            print("\n  FAILED:")
            for f in fails:
                print(f"    - {f}")
            raise SystemExit(1)
        print("    all sync checks passed")

        lines = ["image                            start      dur   narration", ""]
        for e in entries:
            lines.append(f"{Path(e['path']).name[:30]:<30} {e['start']:>8.2f} "
                         f"{e['duration']:>7.2f}   {imageplan.words_under(e, timeline)}")
        (args.work / "sync_report.txt").write_text("\n".join(lines), encoding="utf-8")
        print(f"\n  {args.work / 'sync_report.txt'}  <- check the pairing here")

    if args.stage in ("all", "mux"):
        print("\n" + "=" * 70)
        print("STAGE 3  mux")
        print("=" * 70)
        if not plan_path.exists() or not vo_path.exists():
            raise SystemExit("run --stage voice and --stage plan first")
        entries = json.loads(plan_path.read_text(encoding="utf-8"))
        mux(entries, vo_path, args.out, args.work, args.width, args.height, args.fps)


if __name__ == "__main__":
    main()
