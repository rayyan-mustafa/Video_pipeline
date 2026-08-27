"""SQLite take history.

Local file, no server. Stores one row per take plus the analysis JSON blob, so
score-over-time queries and take ranking (Phase 4) do not need to re-analyze
anything already on disk.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from ..config import DATA_DIR

DB_PATH = DATA_DIR / "sessions.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS takes (
    take_id       TEXT PRIMARY KEY,
    created_at    TEXT NOT NULL,
    script_name   TEXT,
    wav_path      TEXT NOT NULL,
    duration_sec  REAL,
    notes         TEXT
);

CREATE TABLE IF NOT EXISTS analyses (
    take_id       TEXT PRIMARY KEY REFERENCES takes(take_id) ON DELETE CASCADE,
    created_at    TEXT NOT NULL,
    -- full AnalysisResult, so a take never needs re-analyzing to be re-scored
    result_json   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS scores (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    take_id       TEXT NOT NULL REFERENCES takes(take_id) ON DELETE CASCADE,
    created_at    TEXT NOT NULL,
    must_fix_count INTEGER NOT NULL,
    findings_json TEXT NOT NULL
);

-- Competitor reference voices, feature-extracted once and cached (spec:
-- "avoid re-analyzing competitor files every session").
CREATE TABLE IF NOT EXISTS references_cache (
    name          TEXT PRIMARY KEY,
    source_path   TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    features_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_scores_take ON scores(take_id);
"""


@contextmanager
def connect(path: Path | None = None) -> Iterator[sqlite3.Connection]:
    p = path or DB_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(p)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(path: Path | None = None) -> Path:
    with connect(path) as conn:
        conn.executescript(SCHEMA)
    return path or DB_PATH


def record_take(
    take_id: str,
    wav_path: str,
    created_at: str,
    duration_sec: float,
    script_name: str | None = None,
    notes: str | None = None,
    path: Path | None = None,
) -> None:
    with connect(path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO takes "
            "(take_id, created_at, script_name, wav_path, duration_sec, notes) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (take_id, created_at, script_name, wav_path, duration_sec, notes),
        )


def ensure_take(
    take_id: str,
    wav_path: str,
    created_at: str,
    duration_sec: float,
    notes: str | None = None,
    path: Path | None = None,
) -> None:
    """Create the take row only if it is not already there.

    Distinct from `record_take`, which is INSERT OR REPLACE: re-analyzing an
    existing take must not blank out the script_name and notes captured when it
    was recorded.
    """
    with connect(path) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO takes "
            "(take_id, created_at, script_name, wav_path, duration_sec, notes) "
            "VALUES (?, ?, NULL, ?, ?, ?)",
            (take_id, created_at, wav_path, duration_sec, notes),
        )


def save_analysis(take_id: str, result_json: str, created_at: str,
                  path: Path | None = None) -> None:
    with connect(path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO analyses (take_id, created_at, result_json) "
            "VALUES (?, ?, ?)",
            (take_id, created_at, result_json),
        )


def load_analysis(take_id: str, path: Path | None = None) -> dict | None:
    with connect(path) as conn:
        row = conn.execute(
            "SELECT result_json FROM analyses WHERE take_id = ?", (take_id,)
        ).fetchone()
    return json.loads(row["result_json"]) if row else None


def is_stale(analysis: dict) -> bool:
    """True when this record was produced by an older analyzer.

    Stored analyses are reused rather than recomputed, so a metric fix does not
    reach records already on disk. Without this check they keep feeding wrong
    numbers into scoring, ranking and the mentor - silently, because a stale
    result is structurally identical to a fresh one.
    """
    from ..analysis.pipeline import ANALYZER_VERSION

    return (analysis or {}).get("analyzer_version") != ANALYZER_VERSION


def stale_take_ids(path: Path | None = None) -> list[str]:
    out = []
    with connect(path) as conn:
        for row in conn.execute("SELECT take_id, result_json FROM analyses"):
            try:
                if is_stale(json.loads(row["result_json"])):
                    out.append(row["take_id"])
            except Exception:
                out.append(row["take_id"])
    return out


def list_takes(limit: int = 50, path: Path | None = None) -> list[sqlite3.Row]:
    with connect(path) as conn:
        return conn.execute(
            "SELECT t.*, (a.take_id IS NOT NULL) AS analyzed "
            "FROM takes t LEFT JOIN analyses a ON a.take_id = t.take_id "
            "ORDER BY t.created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
