"""
Step 6 -- fetch and cache the grounding sources.

Verbatim from the notebook apart from paths going through paths.p(). This one
belongs local more than any other stage: publishers hand a datacentre IP a
consent wall far more often than a home connection, and looks_like_junk() exists
precisely because that failure arrives as a successful 200.
"""

from __future__ import annotations

import hashlib
import json
import os
import time

import requests
import trafilatura

from . import paths

MAX_CHARS_PER_SOURCE = 6000
MIN_USABLE_CHARS = 700     # below this it's almost always a cookie wall / paywall stub / nav junk
FETCH_RETRIES = 3

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")

CACHE_DIR = "source_cache"


def fetch_source_text(url: str, timeout: int = 20) -> str | None:
    """Fetch + extract clean article body. Retries with backoff; falls back to plain
    requests if trafilatura's fetcher gets blocked."""
    last_err = None
    for attempt in range(FETCH_RETRIES):
        try:
            downloaded = trafilatura.fetch_url(url)
            if not downloaded:
                resp = requests.get(url, timeout=timeout, headers={"User-Agent": UA})
                resp.raise_for_status()
                downloaded = resp.text
            text = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=True,
                favor_recall=True,
            )
            if text and text.strip():
                return text.strip()
            last_err = "extractor returned nothing"
        except Exception as exc:
            last_err = str(exc)
        if attempt < FETCH_RETRIES - 1:
            time.sleep(2 * (attempt + 1))
    print(f"    fetch failed after {FETCH_RETRIES} tries: {url} ({last_err})")
    return None


def looks_like_junk(text: str) -> str | None:
    """Cheap sanity checks on extracted text. Returns a reason string if it looks
    unusable, else None. Catches the case where the fetch 'succeeds' but you actually
    got a consent banner - which would silently starve the writer of real grounding."""
    if len(text) < MIN_USABLE_CHARS:
        return f"only {len(text)} chars extracted (min {MIN_USABLE_CHARS})"
    low = text.lower()
    junk_markers = [
        "enable javascript", "accept cookies", "cookie policy", "subscribe to continue",
        "you have reached your article limit", "please verify you are a human",
        "access denied", "403 forbidden", "captcha",
    ]
    hits = [m for m in junk_markers if m in low[:1500]]
    if hits and len(text) < 2000:
        return f"looks like a consent/paywall page (matched: {hits[0]})"
    return None


def _cache_path(url: str):
    return paths.p(CACHE_DIR, hashlib.sha1(url.encode()).hexdigest()[:16] + ".txt")


def fetch_cached(url: str, use_cache: bool = True) -> tuple[str | None, bool]:
    """Per-URL disk cache. Re-running a topic to iterate on the script no longer
    refetches, which makes script iteration effectively free and avoids hammering
    publishers."""
    os.makedirs(paths.p(CACHE_DIR), exist_ok=True)
    f = _cache_path(url)
    if use_cache and f.exists():
        return f.read_text(encoding="utf-8"), True
    text = fetch_source_text(url)
    if text:
        f.write_text(text, encoding="utf-8")
    return text, False


def run(topic_title: str, srcs: list[dict], use_cache: bool = True) -> list[dict]:
    """-> source_pack, and writes source_pack.json."""
    source_pack, failures = [], []
    print(f'Fetching {len(srcs)} source(s) for "{topic_title}"...\n')

    for s in srcs:
        print(f"  -> {s['title']}")
        text, from_cache = fetch_cached(s["url"], use_cache)
        if from_cache:
            print("     (from cache)")
        if not text:
            failures.append((s, "no text extracted"))
            print("     FAILED\n")
            continue

        reason = looks_like_junk(text)
        if reason:
            failures.append((s, reason))
            print(f"     REJECTED - {reason}\n")
            continue

        excerpt = text[:MAX_CHARS_PER_SOURCE]
        source_pack.append({"title": s["title"], "url": s["url"], "excerpt": excerpt})
        print(f"     OK - {len(excerpt)} chars usable\n")

    # --- Fail loudly rather than quietly writing a thin script off one weak source ---
    if not source_pack:
        raise RuntimeError(
            "No sources fetched. Check network/URLs for this topic, then re-run.\n"
            "Failures:\n" + "\n".join(f"  - {s['url']}: {r}" for s, r in failures)
        )

    total_chars = sum(len(s["excerpt"]) for s in source_pack)
    if total_chars < 3000:
        print("=" * 70)
        print(f"WARNING: only {total_chars} chars of grounding across {len(source_pack)} source(s).")
        print("A ~1,400-word script off this little source text is where the model starts")
        print("improvising. Strongly consider adding a third URL to this topic in config.py")
        print("and re-running, rather than pushing forward.")
        print("=" * 70)

    with open(paths.p("source_pack.json"), "w", encoding="utf-8") as f:
        json.dump(source_pack, f, indent=2)

    print(f"{len(source_pack)}/{len(srcs)} sources usable, {total_chars} chars total grounding.")
    if failures:
        print(f"{len(failures)} source(s) unusable - the script will be written without them:")
        for s, r in failures:
            print(f"  - {s['title']}: {r}")

    paths.mark_done("sources", ["source_pack.json"])
    return source_pack
