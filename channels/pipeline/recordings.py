"""Which audio file belongs to which section - for any topic, not just one.

WHY THIS EXISTS. `build_video.find_recording()` answers "where is section 3's
audio" and it answers it correctly, but only if the files are already called
`section_03.wav`. They usually are not. Takes arrive as `1.wav`..`6.wav` from a
phone, as `live_1743..._comped.wav` from `vo_coach live`, or as one file per
paragraph with no numbering scheme at all - and a long section often gets read in
two takes, so the count does not even match.

The chin project hit exactly this: six files, five sections, and the voice stage
refused to start. That was resolved by hand, by measuring durations and comparing
them against section word counts. **This module is that reasoning written down**,
so the next fourteen topics do not each need a session of it.

WHAT IT WILL AND WILL NOT DO. It PROPOSES a mapping and shows its evidence; it
never renames anything on its own. The proposal is only as good as its margin
over the runner-up, so `confidence` is reported and the operator confirms. A
wrong mapping is not silently corrupt - every timestamp downstream comes from the
audio this stage measures, so the video would simply be wrong in an obvious way -
but it costs a Whisper pass to discover, and on `large-v3` that is real money.

THE METHOD. A section's word count predicts its spoken duration, and a reader's
words-per-minute is roughly constant across one sitting. So of all the ways the
files could line up in order, the right one is the one where every section comes
out at about the same wpm. On chin that was decisive: the correct mapping scored
a 24.9 wpm spread and the runner-up 100.3.

Files are only ever assigned IN ORDER. Recordings are made front to back, and
allowing reordering would multiply the search space by a factor no evidence could
then narrow back down.
"""

from __future__ import annotations

import itertools
import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

# Same set build_video accepts, kept here so both agree on what "a recording" is.
AUDIO_EXTS = ("wav", "flac", "mp3", "m4a", "ogg")

# A mapping has to beat the next-best one by this ratio before it is worth
# calling confident. Below it, the two readings are close enough that the
# operator should look rather than trust the arithmetic.
CONFIDENT_MARGIN = 1.8

# Above this many candidate groupings, stop searching and say so rather than
# hanging. C(F-1, S-1) grows fast and no proposal is worth a stalled UI.
MAX_CANDIDATES = 20000


def _natural_key(p: Path) -> tuple:
    """Sort `2.wav` before `10.wav`, and `a_2` before `a_10`.

    Plain lexicographic order puts `10.wav` first, which would silently shift
    every section by one on any project with ten or more takes.
    """
    return tuple(
        int(t) if t.isdigit() else t.lower()
        for t in re.split(r"(\d+)", p.name)
    )


def list_audio(rec_dir: Path) -> list[Path]:
    """Every recording in the folder, in the order it was most likely read."""
    if not rec_dir.is_dir():
        return []
    hits: list[Path] = []
    for ext in AUDIO_EXTS:
        hits += rec_dir.glob(f"*.{ext}")
        hits += rec_dir.glob(f"*.{ext.upper()}")
    return sorted(set(hits), key=_natural_key)


def canonical_name(section: int, part: int | None = None, ext: str = "wav") -> str:
    """The name `find_recording()` looks for. One place, so both sides agree."""
    if part is None:
        return f"section_{section:02d}.{ext}"
    # Parts are joined in filename order, so the suffix must sort correctly.
    return f"section_{section:02d}_{chr(ord('a') + part)}.{ext}"


def is_canonical(p: Path) -> bool:
    """Does this filename already resolve to a section without inference?"""
    return bool(re.match(r"^section_\d{1,2}(_[a-z0-9]+)?$", p.stem, re.I))


def duration(path: Path) -> float:
    """Seconds of audio. Raises if the file is not readable as audio."""
    import soundfile as sf
    return float(sf.info(str(path)).duration)


# ---------------------------------------------------------------------------
#  The proposal
# ---------------------------------------------------------------------------
@dataclass
class Assignment:
    """One section and the files proposed for it."""

    section: int
    files: list[Path]
    seconds: float
    words: int

    @property
    def wpm(self) -> float:
        return (self.words / self.seconds * 60.0) if self.seconds > 0 else 0.0

    def targets(self) -> list[tuple[Path, str]]:
        """(source, canonical name) pairs. Multi-take sections become _a/_b."""
        multi = len(self.files) > 1
        return [
            (f, canonical_name(self.section,
                               i if multi else None,
                               f.suffix.lstrip(".").lower()))
            for i, f in enumerate(self.files)
        ]

    def as_dict(self) -> dict:
        return {
            "section": self.section,
            "files": [f.name for f in self.files],
            "targets": [name for _, name in self.targets()],
            "seconds": round(self.seconds, 2),
            "words": self.words,
            "wpm": round(self.wpm, 1),
        }


@dataclass
class Proposal:
    """What we think the mapping is, and how much to trust it."""

    status: str                       # ok | ambiguous | impossible | already
    assignments: list[Assignment] = field(default_factory=list)
    spread: float = 0.0               # wpm max-min of the winner
    runner_up: float = 0.0            # wpm spread of the next-best grouping
    confidence: str = "none"          # high | low | none
    note: str = ""
    unmatched: list[Path] = field(default_factory=list)

    @property
    def needs_rename(self) -> bool:
        return any(f.name != name
                   for a in self.assignments for f, name in a.targets())

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "confidence": self.confidence,
            "needs_rename": self.needs_rename,
            "spread": round(self.spread, 1),
            "runner_up": round(self.runner_up, 1),
            "note": self.note,
            "assignments": [a.as_dict() for a in self.assignments],
            "unmatched": [p.name for p in self.unmatched],
        }


def _groupings(n_files: int, n_sections: int):
    """Every way to cut `n_files` ordered files into `n_sections` runs.

    Yields lists of group sizes. Contiguity is the whole constraint: takes are
    recorded front to back, so section 2's audio cannot sit between two of
    section 4's.
    """
    if n_sections > n_files:
        return
    # Choose the n_sections-1 cut positions among the n_files-1 internal gaps.
    for cuts in itertools.combinations(range(1, n_files), n_sections - 1):
        bounds = (0,) + cuts + (n_files,)
        yield [bounds[i + 1] - bounds[i] for i in range(n_sections)]


def propose(rec_dir: Path, section_words: dict[int, int]) -> Proposal:
    """Work out which files belong to which section.

    `section_words` maps section number -> word count, which is what turns a
    pile of durations into a testable prediction.
    """
    sections = sorted(section_words)
    files = list_audio(rec_dir)

    if not files:
        return Proposal(status="impossible",
                        note=f"no audio files in {rec_dir}")
    if not sections:
        return Proposal(status="impossible", unmatched=files,
                        note="no sections - write the script and split it first")

    # ALREADY CORRECT is the common case once a project has been set up, and it
    # must not be disturbed. Group canonical files by the section they name.
    canon: dict[int, list[Path]] = {}
    for f in files:
        m = re.match(r"^section_(\d{1,2})", f.stem, re.I)
        if m:
            canon.setdefault(int(m.group(1)), []).append(f)
    if set(canon) >= set(sections):
        return Proposal(
            status="already",
            assignments=[
                Assignment(s, sorted(canon[s], key=_natural_key), 0.0,
                           section_words[s])
                for s in sections
            ],
            confidence="high",
            note=f"all {len(sections)} sections already named section_NN - "
                 f"nothing to do",
        )

    if len(files) < len(sections):
        return Proposal(
            status="impossible", unmatched=files,
            note=f"{len(files)} recording(s) for {len(sections)} sections - "
                 f"at least one section has not been recorded",
        )

    try:
        durs = [duration(f) for f in files]
    except Exception as e:
        return Proposal(status="impossible", unmatched=files,
                        note=f"could not read a recording: {e}")

    n_cand = 1
    for i in range(len(sections) - 1):
        n_cand = n_cand * (len(files) - 1 - i) // (i + 1)
    if n_cand > MAX_CANDIDATES:
        return Proposal(
            status="ambiguous", unmatched=files,
            note=f"{len(files)} files over {len(sections)} sections is "
                 f"{n_cand} possible groupings - too many to choose between. "
                 f"Rename the obvious ones to section_NN and re-run.",
        )

    # SCORE EVERY GROUPING BY HOW CONSTANT THE READER'S PACE COMES OUT.
    # Spread is normalised by the mean so the metric does not reward a slow
    # reader; a fast one would otherwise always look tidier.
    scored: list[tuple[float, float, list[int]]] = []
    for sizes in _groupings(len(files), len(sections)):
        i = 0
        wpms, raw = [], []
        for s, n in zip(sections, sizes):
            secs = sum(durs[i:i + n])
            i += n
            if secs <= 0:
                break
            wpms.append(section_words[s] / secs * 60.0)
        else:
            spread = max(wpms) - min(wpms)
            scored.append((spread / (sum(wpms) / len(wpms)), spread, sizes))
    if not scored:
        return Proposal(status="impossible", unmatched=files,
                        note="every candidate grouping had an empty section")

    scored.sort(key=lambda r: r[0])
    best_norm, best_spread, sizes = scored[0]
    second = scored[1] if len(scored) > 1 else None

    i = 0
    assignments = []
    for s, n in zip(sections, sizes):
        grp = files[i:i + n]
        assignments.append(
            Assignment(s, grp, sum(durs[i:i + n]), section_words[s]))
        i += n

    if second is None:
        confidence, status = "high", "ok"
        note = "only one possible grouping"
    else:
        margin = (second[0] / best_norm) if best_norm > 0 else float("inf")
        confident = margin >= CONFIDENT_MARGIN
        confidence = "high" if confident else "low"
        status = "ok" if confident else "ambiguous"
        note = (f"reader holds {best_spread:.0f} wpm spread on this mapping "
                f"vs {second[1]:.0f} on the next-best "
                f"({margin:.1f}x margin)")
        if not confident:
            note += " - too close to call, check it by ear before renaming"

    multi = [a for a in assignments if len(a.files) > 1]
    if multi:
        note += (". Section(s) " +
                 ", ".join(str(a.section) for a in multi) +
                 " read as multiple takes, joined in order")

    return Proposal(status=status, assignments=assignments,
                    spread=best_spread,
                    runner_up=second[1] if second else 0.0,
                    confidence=confidence, note=note)


# ---------------------------------------------------------------------------
#  Applying it
# ---------------------------------------------------------------------------
def apply(rec_dir: Path, proposal: Proposal, dry_run: bool = False) -> dict:
    """Rename files onto the canonical scheme, recording what was done.

    REVERSIBLE BY DESIGN. The mapping is written to `naming.json` beside the
    audio before anything moves, so the rename can always be undone - these are
    the operator's own recordings and a bad guess must not cost a re-record.
    Refuses to clobber: if a target name already exists and is not its own
    source, nothing at all is renamed.
    """
    if proposal.status not in ("ok", "already"):
        raise ValueError(f"refusing to apply a {proposal.status} proposal: "
                         f"{proposal.note}")

    moves = [(f, rec_dir / name)
             for a in proposal.assignments for f, name in a.targets()
             if f.name != name]
    if not moves:
        return {"renamed": [], "note": "already canonical"}

    sources = {f.resolve() for f, _ in moves}
    for _, dst in moves:
        if dst.exists() and dst.resolve() not in sources:
            raise FileExistsError(
                f"{dst.name} already exists and is not part of this rename - "
                f"resolve it by hand so nothing is lost")

    rows = [{"was": f.name, "now": dst.name} for f, dst in moves]
    if dry_run:
        return {"renamed": rows, "dry_run": True}

    record = rec_dir / "naming.json"
    record.write_text(json.dumps({
        "resolved": date.today().isoformat(),
        "confidence": proposal.confidence,
        "note": proposal.note,
        "spread_wpm": round(proposal.spread, 1),
        "runner_up_wpm": round(proposal.runner_up, 1),
        "sections": [a.as_dict() for a in proposal.assignments],
        "mapping": rows,
    }, indent=2), encoding="utf-8")

    # Two passes through a temporary name: a mapping may be a permutation
    # (`section_02.wav` -> `section_03.wav` and back), and renaming in place
    # would destroy one of them halfway through.
    staged = []
    for f, dst in moves:
        tmp = f.with_name(f".{f.name}.staged")
        f.rename(tmp)
        staged.append((tmp, dst))
    for tmp, dst in staged:
        tmp.rename(dst)

    return {"renamed": rows, "record": str(record)}


def revert(rec_dir: Path) -> dict:
    """Undo the last `apply()` using the naming.json it wrote."""
    record = rec_dir / "naming.json"
    if not record.is_file():
        raise FileNotFoundError(f"no naming.json in {rec_dir} - nothing to undo")
    data = json.loads(record.read_text(encoding="utf-8"))
    moves = [(rec_dir / r["now"], rec_dir / r["was"]) for r in data["mapping"]]
    missing = [src.name for src, _ in moves if not src.exists()]
    if missing:
        raise FileNotFoundError(f"cannot undo - missing {', '.join(missing)}")

    staged = []
    for src, dst in moves:
        tmp = src.with_name(f".{src.name}.staged")
        src.rename(tmp)
        staged.append((tmp, dst))
    for tmp, dst in staged:
        tmp.rename(dst)
    record.unlink()
    return {"reverted": [{"was": r["now"], "now": r["was"]}
                         for r in data["mapping"]]}
