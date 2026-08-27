"""Config loading.

Every tunable number lives in config/thresholds.yaml. This module is the only
thing that reads it, so there is exactly one place where a default could be
silently wrong.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml

# src/vo_coach/config.py -> project root is three levels up
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "thresholds.yaml"

DATA_DIR = PROJECT_ROOT / "data"
TAKES_DIR = DATA_DIR / "takes"
REFERENCES_DIR = DATA_DIR / "references"
CACHE_DIR = DATA_DIR / "cache"


class Config:
    """Thin dotted-path accessor over the YAML, so callers read
    `cfg.get("pacing.target_wpm_min")` instead of chaining dict lookups."""

    def __init__(self, raw: dict[str, Any], path: Path):
        self._raw = raw
        self._path = path

    def get(self, dotted: str, default: Any = ...) -> Any:
        node: Any = self._raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                if default is ...:
                    raise KeyError(
                        f"{dotted!r} missing from {self._path}. "
                        f"Thresholds belong in the YAML, not in code."
                    )
                return default
            node = node[part]
        return node

    def section(self, name: str) -> dict[str, Any]:
        return dict(self.get(name))

    @property
    def raw(self) -> dict[str, Any]:
        return self._raw


@functools.lru_cache(maxsize=1)
def load_config(path: Path | None = None) -> Config:
    p = path or CONFIG_PATH
    if not p.exists():
        raise FileNotFoundError(f"No config at {p}")
    with open(p, "r", encoding="utf-8") as f:
        return Config(yaml.safe_load(f), p)


def ensure_dirs() -> None:
    for d in (TAKES_DIR, REFERENCES_DIR, CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)
