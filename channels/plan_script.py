"""script.txt -> read sheets + a stitch plan that binds every image to its words.

    python plan_script.py --project wierdbiology/chin --topic "the human chin"

WHY THIS HAS NO TIMESTAMPS IN IT
--------------------------------
An earlier version divided each section's measured duration between its pictures.
That put timing in the plan, which meant the plan could only be made AFTER
recording - so pictures could not be generated in parallel with the voice work,
and re-recording one section silently invalidated every span in the file.

This plan carries no seconds at all. It binds each image to the exact SENTENCES
it illustrates, and nothing else. At build time Whisper timestamps those same
sentences, and each image simply inherits the span of the scenes it owns:

    image start = first owned scene's start
    image end   = last owned scene's end

So timing is derived, never stored, and it stays correct no matter how fast or
slow you happen to read. Re-record a section and the picture bindings still hold.

THE JOIN IS BY scene_index, AND IT IS CHECKED
---------------------------------------------
Both this file and the aligned timeline come from build_chunks() over the same
script, so scene N here is scene N there. That is an invariant, not a hope, so
script_plan.json stores a fingerprint of its scene texts and the build refuses
to stitch against a timeline that does not match it. Editing the script after
planning is the one thing that breaks the binding, and it now fails loudly
instead of shifting every picture by one sentence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pipeline.project import Project
from pipeline import imageplan
from pipeline.scenesplit import counts as split_counts, make_scenes, match_scenes

# 150 wpm is the read sheets' own assumption (see sections/README.txt), so image
# budgets and the "expected minutes" line stay consistent with it.
DEFAULT_WPM = 150


def split_sections(text: str) -> list[str]:
    """Blank line = section break. That is the convention the read sheets use."""
    return [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]


def find_brackets(text: str) -> list[str]:
    """Stage directions the writer left in. They must never reach a read sheet -
    you would read them aloud, and Whisper would then align real audio against
    words the script says are spoken."""
    out = []
    for pat in (r"\[[^\]]*\]", r"\([^)]*\)", r"\{[^}]*\}", r"<[^>]*>"):
        out += re.findall(pat, text)
    return out


def to_tagged(sections: list[str]) -> list[dict]:
    """[section text] -> [{'sentence','section'}], the shape build_chunks wants."""
    tagged = []
    for n, block in enumerate(sections, 1):
        for line in block.splitlines():
            line = line.strip()
            if not line:
                continue
            for s in re.split(r"(?<=[.!?])\s+", line):
                if s.strip():
                    tagged.append({"sentence": s.strip(), "section": n})
    return tagged


def group_into_images(chunks: list[dict], words_per_image: int) -> list[dict]:
    """Consecutive scenes -> image slots, one run of scenes per picture.

    Grouping is by WORD COUNT, not by time, because no time exists yet. At a
    steady read the two are the same thing; where they differ, the words are what
    the picture actually has to illustrate.

    A slot never spans a section boundary - sections are recorded as separate
    files, so a picture straddling one would have to be cut across two takes.
    """
    images: list[dict] = []
    order_in_section: dict[int, int] = {}

    buf: list[int] = []
    buf_words = 0
    buf_section: int | None = None

    def flush():
        nonlocal buf, buf_words, buf_section
        if not buf:
            return
        order_in_section[buf_section] = order_in_section.get(buf_section, 0) + 1
        order = order_in_section[buf_section]
        images.append({
            "section": buf_section,
            "order": order,
            "filename": f"{order}section{buf_section}.png",
            "scene_indices": list(buf),
            "narration": [chunks[i]["text"] for i in buf],
            "words": buf_words,
        })
        buf, buf_words, buf_section = [], 0, None

    for i, c in enumerate(chunks):
        n = len(c["text"].split())
        if buf and (c["section"] != buf_section or buf_words + n > words_per_image):
            flush()
        if buf_section is None:
            buf_section = c["section"]
        buf.append(i)
        buf_words += n
    flush()
    return images


def fingerprint(chunks: list[dict]) -> str:
    """Hash of the scene texts. The build compares this against the timeline it is
    handed, so an edited script cannot silently shift every picture."""
    h = hashlib.sha256()
    for c in chunks:
        h.update(c["text"].strip().encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def render_md(images: list[dict], chunks: list[dict], sections: list[str],
              wpm: int) -> str:
    total_words = sum(len(c["text"].split()) for c in chunks)
    L = [
        "# Stitch plan",
        "",
        f"{len(sections)} sections, {len(chunks)} scenes, **{len(images)} images**, "
        f"{total_words} words (~{total_words / wpm:.1f} min at {wpm} wpm).",
        "",
        "No timestamps here on purpose - Whisper supplies those at build time by "
        "matching the sentences below. Save each picture under the filename in its "
        "heading.",
        "",
        "| section | scenes | images | words |",
        "| --- | --- | --- | --- |",
    ]
    for n in range(1, len(sections) + 1):
        sc = [c for c in chunks if c["section"] == n]
        im = [i for i in images if i["section"] == n]
        L.append(f"| {n} | {len(sc)} | {len(im)} | "
                 f"{sum(len(c['text'].split()) for c in sc)} |")
    L += ["", "---", ""]

    current = None
    for img in images:
        if img["section"] != current:
            current = img["section"]
            L += [f"## Section {current}", ""]
        scenes = ", ".join(str(i) for i in img["scene_indices"])
        L += [
            f"### `{img['filename']}`",
            "",
            f"*scene {scenes} &middot; {img['words']} words*",
            "",
            "> " + " ".join(img["narration"]),
            "",
        ]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", required=True,
                    help="<channel>/<topic>, e.g. wierdbiology/chin")
    ap.add_argument("--script", type=Path, default=None,
                    help="defaults to <project>/script.txt")
    ap.add_argument("--split", default="chunk",
                    choices=("chunk", "sentence", "dash", "match"),
                    help="how the script becomes scenes. chunk merges into "
                         "~20-word scenes (default); sentence is one per "
                         "sentence; dash also splits on a spaced em-dash. Use "
                         "sentence or dash when the pictures were commissioned "
                         "one per sentence.")
    ap.add_argument("--seconds-per-image", type=float, default=8.0)
    ap.add_argument("--wpm", type=int, default=DEFAULT_WPM)
    ap.add_argument("--topic", default=None,
                    help="what the whole video is about, e.g. 'the human chin'. "
                         "Injected into every image prompt so a span that never "
                         "names the subject cannot drift onto another one. "
                         "Defaults to the script's filename")
    ap.add_argument("--allow-brackets", action="store_true",
                    help="plan anyway despite bracketed text in the script")
    args = ap.parse_args()

    proj = Project.parse(args.project)
    proj.make_dirs()
    script_path = args.script or proj.script
    if not script_path.exists():
        raise SystemExit(
            f"no script at {script_path}.\n"
            f"  Put the script there, or pass --script <path>."
        )
    args.script = script_path
    args.work = proj.build
    args.images_dir = proj.images

    text = script_path.read_text(encoding="utf-8")

    brackets = find_brackets(text)
    if brackets and not args.allow_brackets:
        raise SystemExit(
            f"{len(brackets)} bracketed passage(s) in {args.script.name} - these "
            f"would be read aloud or break alignment:\n  "
            + "\n  ".join(brackets[:12])
            + "\n\nRemove them, or pass --allow-brackets."
        )
    print(f"brackets: none" if not brackets else f"brackets: {len(brackets)} ALLOWED")

    sections = split_sections(text)
    tagged = to_tagged(sections)
    # "match" reads the pictures that already exist and divides the script into
    # exactly that many scenes per section, giving a true 1:1 binding. The
    # pictures ARE the record of how the script was divided, and that beats any
    # rule reconstructed after the fact.
    existing = None
    if args.split == "match":
        existing = imageplan.scan_images(proj.images)
        targets: dict[int, int] = {}
        for i in existing:
            targets[i["section"]] = targets.get(i["section"], 0) + 1
        chunks = match_scenes(tagged, targets)
        print(f"  matching {len(existing)} existing images -> {len(chunks)} scenes")
        for n in existing[0].get("_notes", []):
            print(f"  note: {n}")
    else:
        chunks = make_scenes(tagged, args.split)

    print("  scenes by split mode:", split_counts(tagged))
    words_per_image = max(1, round(args.wpm * args.seconds_per_image / 60.0))
    if existing is not None:
        # One scene per picture, in order, using the REAL filenames on disk.
        images = []
        for idx, (img, ch) in enumerate(zip(existing, chunks)):
            images.append({
                "section": ch["section"],
                "order": img["order"],
                "filename": Path(img["path"]).name,
                "scene_indices": [idx],
                "narration": [ch["text"]],
                "words": len(ch["text"].split()),
            })
    else:
        images = group_into_images(chunks, words_per_image)

    work = args.work
    proj.sections.mkdir(parents=True, exist_ok=True)

    # Read sheets: one file per section, exactly as it must be read.
    for n, block in enumerate(sections, 1):
        (proj.sections / f"section_{n:02d}.txt").write_text(block + "\n",
                                                            encoding="utf-8")

    # Default to the project topic ("chin"), not the filename - every
    # project's script is called script.txt, so the stem carries nothing.
    topic = args.topic or proj.topic.replace("_", " ").replace("-", " ")

    plan = {
        "script": str(args.script),
        "topic": topic,
        "fingerprint": fingerprint(chunks),
        "split": args.split,
        "wpm": args.wpm,
        "seconds_per_image": args.seconds_per_image,
        "words_per_image": words_per_image,
        "images_dir": str(args.images_dir),
        "sections": len(sections),
        "scenes": [
            {"scene_index": i, "section": c["section"], "text": c["text"]}
            for i, c in enumerate(chunks)
        ],
        "images": [
            {**img, "path": str(args.images_dir / img["filename"])}
            for img in images
        ],
    }
    (work / "script_plan.json").write_text(json.dumps(plan, indent=2),
                                            encoding="utf-8")
    (work / "script_plan.md").write_text(
        render_md(images, chunks, sections, args.wpm), encoding="utf-8")

    total_words = sum(len(c["text"].split()) for c in chunks)
    print(f"\n  {len(sections)} sections, {len(chunks)} scenes, "
          f"{len(images)} images, {total_words} words "
          f"(~{total_words / args.wpm:.1f} min)")
    print(f"  ~{words_per_image} words per image "
          f"(~{args.seconds_per_image:.0f}s at {args.wpm} wpm)\n")
    for n in range(1, len(sections) + 1):
        sc = [c for c in chunks if c["section"] == n]
        im = [i for i in images if i["section"] == n]
        print(f"  section {n}   {len(sc):>3} scenes   {len(im):>3} images   "
              f"{sum(len(c['text'].split()) for c in sc):>4} words")
    print(f"\n  fingerprint {plan['fingerprint']}")
    print(f"\n  {work / 'script_plan.json'}")
    print(f"  {work / 'script_plan.md'}")
    print(f"  {proj.sections / 'section_01.txt'} ... "
          f"section_{len(sections):02d}.txt   <- record these")


if __name__ == "__main__":
    main()
