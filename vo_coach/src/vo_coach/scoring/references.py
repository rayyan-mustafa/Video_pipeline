"""Build and cache competitor reference features, one per niche.

Extraction runs once and is cached in SQLite: re-analyzing a reference on every
session would mean re-running ASR over the same 60 s clip forever.
"""

from __future__ import annotations

import datetime as _dt
import json

from ..config import PROJECT_ROOT, load_config
from ..schemas import ReferenceFeatures
from ..storage import db
from .features import extract


def niche_names() -> list[str]:
    return sorted(load_config().section("niches").keys())


def niche_config(niche: str) -> dict:
    niches = load_config().section("niches")
    if niche not in niches:
        raise KeyError(f"Unknown niche {niche!r}. Known: {', '.join(sorted(niches))}")
    return niches[niche]


def build(niche: str, force: bool = False) -> ReferenceFeatures:
    """Analyze this niche's reference clip and cache its feature vector."""
    nc = niche_config(niche)
    name = nc["reference_name"]

    if not force:
        cached = load(niche)
        if cached is not None:
            return cached

    path = PROJECT_ROOT / nc["reference"]
    if not path.exists():
        raise FileNotFoundError(
            f"No reference audio for niche {niche!r} at {path}. "
            f"Drop a clean solo-VO clip there first."
        )

    from ..analysis.pipeline import analyze

    # Language matters here in a way that fails silently if ignored: the `.en`
    # checkpoints are English-only and will transcribe Bengali as plausible-
    # looking English garbage rather than raising. WPM computed from that would
    # be meaningless but perfectly well-formed.
    result = analyze(
        str(path),
        take_id=f"__ref_{name}",
        script=None,
        model_size=nc["asr_model"],
        with_asr=True,
        language=nc.get("language", "en"),
    )

    ref = ReferenceFeatures(
        name=name,
        niche=niche,
        source_path=str(path),
        language=nc.get("language", "en"),
        duration_sec=result.duration_sec,
        features=extract(result),
        provenance=nc.get("provenance", "unknown"),
    )

    db.init_db()
    with db.connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO references_cache "
            "(name, source_path, created_at, features_json) VALUES (?, ?, ?, ?)",
            (name, str(path), _dt.datetime.now().isoformat(timespec="seconds"),
             ref.model_dump_json()),
        )
    return ref


def load(niche: str) -> ReferenceFeatures | None:
    nc = niche_config(niche)
    db.init_db()
    with db.connect() as conn:
        row = conn.execute(
            "SELECT features_json FROM references_cache WHERE name = ?",
            (nc["reference_name"],),
        ).fetchone()
    if row is None:
        return None
    return ReferenceFeatures.model_validate(json.loads(row["features_json"]))


def wpm_band_for(niche: str) -> tuple[float, float] | None:
    """Pacing target derived from the reference, when the niche asks for it.

    The global 140-170 band is a US-English VO norm. Applying it to Bengali
    would flag a perfectly normal read as too fast or too slow, because
    syllable structure and speaking-rate conventions differ by language. A
    voice already succeeding in the niche is the better target.

    Returns None when the niche uses the global band.
    """
    nc = niche_config(niche)
    if not nc.get("wpm_from_reference"):
        return None
    ref = load(niche)
    if ref is None:
        return None
    centre = ref.features.wpm_overall
    # +/-12% around the reference: wide enough not to demand imitation, tight
    # enough to catch a genuinely rushed or dragging read.
    return (round(centre * 0.88, 1), round(centre * 1.12, 1))
