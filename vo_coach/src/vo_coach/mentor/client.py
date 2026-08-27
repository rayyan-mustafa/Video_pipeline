"""The mentor call itself.

Wraps the Anthropic Messages API with the disk cache, so repeated runs over an
unchanged take cost nothing. The system prompt is frozen and cached server-side
as well; take-specific JSON goes in the user turn, after the breakpoint.
"""

from __future__ import annotations

import datetime as _dt

from ..config import Config, load_config
from ..schemas import AnalysisResult, BenchmarkResult, ScoreResult
from . import cache as _cache
from .prompt import SYSTEM_PROMPT, build_payload


class MentorError(RuntimeError):
    pass


class OfflineCacheMiss(MentorError):
    """offline=True and nothing cached for this input. Deliberately fatal: the
    alternative is silently spending money in a loop that was meant to be free."""


def _client():
    """Kept for tests that patch the transport. Real calls go through
    `providers.get_provider()`."""
    from .providers import get_provider

    return get_provider(load_config())


def coach(
    analysis: AnalysisResult,
    score: ScoreResult,
    benchmark: BenchmarkResult | None = None,
    history: list[dict] | None = None,
    cfg: Config | None = None,
    timings_reliable: bool = True,
    force: bool = False,
) -> tuple[str, dict]:
    """Return (coaching_text, meta).

    `meta` carries cache status, model and token usage so the CLI can show what
    a call actually cost rather than leaving it invisible.
    """
    cfg = cfg or load_config()
    model = cfg.get("mentor.model")
    max_fixes = cfg.get("mentor.max_fixes", 3)
    use_cache = cfg.get("mentor.cache_responses", True)
    offline = cfg.get("mentor.offline", False)

    payload = build_payload(analysis, score, benchmark=benchmark, history=history,
                            max_fixes=max_fixes, timings_reliable=timings_reliable)

    key = _cache.key_for(
        model, SYSTEM_PROMPT, payload,
        provider=cfg.get("mentor.provider", "openrouter"),
        effort=cfg.get("mentor.effort", "medium"),
        thinking=cfg.get("mentor.thinking", "adaptive"),
    )

    def _validated(text: str, meta: dict, record: bool) -> tuple[str, dict]:
        """Attach the rule-compliance report to every reply.

        Run on cached replies too - the checks are free and the rules may have
        changed since the reply was generated - but only LOGGED on real calls.
        Logging a cache hit would move the per-model failure rate without any
        new evidence behind it.
        """
        if not cfg.get("mentor.validate", True):
            return text, meta
        from . import validate as _v

        report = _v.validate(text, payload, max_fixes=max_fixes,
                             take_id=analysis.take_id,
                             model=meta.get("model") or model,
                             provider=meta.get("provider"))
        if record:
            _v.log(report)
        return text, {**meta, "validation": report}

    if use_cache and not force:
        if (hit := _cache.load(key)) is not None:
            return _validated(hit["text"],
                              {**hit.get("meta", {}), "cached": True, "key": key},
                              record=False)

    if offline:
        raise OfflineCacheMiss(
            f"mentor.offline is true and no cached response exists for this take "
            f"(key {key}). Set mentor.offline: false in config to allow the call, "
            f"or run with --force after disabling offline."
        )

    from .providers import ProviderError, get_provider

    provider = get_provider(cfg)
    try:
        text, meta = provider.complete(SYSTEM_PROMPT, payload, cfg)
    except ProviderError as e:
        raise MentorError(str(e)) from e

    if use_cache:
        _cache.save(key, {
            "text": text,
            "meta": meta,
            "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "take_id": analysis.take_id,
        })

    return _validated(text, {**meta, "key": key}, record=True)
