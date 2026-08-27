"""Provider-layer tests: OpenRouter (default) and Anthropic.

The mentor is the only phase that calls an LLM, and its backend is a config
switch. These verify each provider's request shape without network calls -
catching the class of mistake that otherwise only appears as a 4xx after a take
has already been recorded.
"""

from __future__ import annotations

import pytest

from vo_coach.mentor import cache
from tests.test_mentor import _f, _score
from tests.test_scoring import mk


def cfg_with(**over):
    """A Config with mentor overrides, without touching the real YAML."""
    from pathlib import Path

    from vo_coach.config import Config, load_config

    raw = dict(load_config().raw)
    raw["mentor"] = {**raw["mentor"], **over}
    return Config(raw, Path("<test>"))


class TestProviderSelection:
    def test_defaults_are_openrouter_and_gpt_oss(self):
        from vo_coach.config import load_config

        cfg = load_config()
        assert cfg.get("mentor.provider") == "openrouter"
        assert cfg.get("mentor.model") == "openai/gpt-oss-120b"

    def test_unknown_provider_fails_loudly(self):
        from vo_coach.mentor.providers import ProviderError, get_provider

        with pytest.raises(ProviderError, match="Unknown"):
            get_provider(cfg_with(provider="nope"))

    def test_provider_is_part_of_the_cache_key(self):
        """Two providers must not share a cached answer - the same take coached
        by a different model is a different answer."""
        base = dict(model="m", system="s", payload="p")
        assert (cache.key_for(**base, provider="openrouter")
                != cache.key_for(**base, provider="anthropic"))


def _openrouter(monkeypatch, **cfg_over):
    from vo_coach.mentor import providers

    captured: dict = {}

    class _Msg:
        content = "Solid read. Watch the sibilance."

    class _Choice:
        message = _Msg()
        finish_reason = "stop"

    class _Usage:
        prompt_tokens = 2300
        completion_tokens = 400

    class _Resp:
        choices = [_Choice()]
        usage = _Usage()
        model = "inclusionai/ling-3.0-flash"
        provider = "novita/fp8"

    class _Completions:
        def create(self, **kw):
            captured.update(kw)
            return _Resp()

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

    monkeypatch.setattr(providers.OpenRouterProvider, "_client",
                        lambda self, cfg: _Client())
    text, meta = providers.OpenRouterProvider().complete(
        "SYS", "PAYLOAD", cfg_with(**cfg_over))
    return captured, text, meta


class TestOpenRouterProvider:
    def test_host_is_pinned_and_fallbacks_disabled(self, monkeypatch):
        """Hosts differ ~8x in SPEED for the same model - cerebras/fp16 runs
        970 tok/s against Parasail's 114, at prices spanning $0.03 to $0.35 per
        1M. Unpinned routing makes both cost and latency unreproducible, and
        Phase 5 coaching depends on the latency."""
        req, _, _ = _openrouter(monkeypatch)
        assert req["extra_body"]["provider"]["order"] == ["cerebras/fp16", "groq"]
        assert req["extra_body"]["provider"]["allow_fallbacks"] is False

    def test_ordered_backup_host_is_still_inside_the_pin(self, monkeypatch):
        """Two hosts in order with fallbacks OFF is deliberate: an outage fails
        over to Groq, but never to a host chosen for us."""
        req, _, _ = _openrouter(monkeypatch)
        assert req["extra_body"]["provider"]["order"][0] == "cerebras/fp16"
        assert len(req["extra_body"]["provider"]["order"]) == 2

    def test_no_pin_when_not_configured(self, monkeypatch):
        req, _, _ = _openrouter(monkeypatch, openrouter_providers=None)
        assert "provider" not in (req.get("extra_body") or {})

    def test_reasoning_effort_is_sent(self, monkeypatch):
        """A reasoning model bills and times its chain-of-thought as output
        tokens. Not pinning effort makes the latency budget fiction."""
        req, _, _ = _openrouter(monkeypatch, openrouter_reasoning_effort="low")
        assert req["extra_body"]["reasoning"] == {"effort": "low"}

    def test_reasoning_omitted_when_unset(self, monkeypatch):
        """Non-reasoning models reject or ignore it; do not send it blindly."""
        req, _, _ = _openrouter(monkeypatch, openrouter_reasoning_effort=None)
        assert "reasoning" not in (req.get("extra_body") or {})

    def test_reasoning_tokens_are_reported(self, monkeypatch):
        """Invisible in the text, but paid for and waited on. A model quietly
        thinking for 800 tokens must show up as a number, not as an
        unexplained slow call."""
        from vo_coach.mentor import providers

        class _Details:
            reasoning_tokens = 800

        class _Usage:
            prompt_tokens = 2300
            completion_tokens = 400
            completion_tokens_details = _Details()

        class _Msg:
            content = "ok"

        class _Choice:
            message = _Msg()
            finish_reason = "stop"

        class _Resp:
            choices = [_Choice()]
            usage = _Usage()
            model = "openai/gpt-oss-120b"
            provider = "cerebras/fp16"

        class _Completions:
            def create(self, **kw):
                return _Resp()

        class _Chat:
            completions = _Completions()

        class _Client:
            chat = _Chat()

        monkeypatch.setattr(providers.OpenRouterProvider, "_client",
                            lambda self, cfg: _Client())
        _, meta = providers.OpenRouterProvider().complete("s", "u", cfg_with())
        assert meta["reasoning_tokens"] == 800

    def test_missing_reasoning_details_is_zero_not_a_crash(self, monkeypatch):
        _, _, meta = _openrouter(monkeypatch)
        assert meta["reasoning_tokens"] == 0

    def test_system_and_user_are_separate_messages(self, monkeypatch):
        req, _, _ = _openrouter(monkeypatch)
        assert [m["role"] for m in req["messages"]] == ["system", "user"]
        assert req["messages"][0]["content"] == "SYS"
        assert req["messages"][1]["content"] == "PAYLOAD"

    def test_reports_the_serving_host_and_tokens(self, monkeypatch):
        _, _, meta = _openrouter(monkeypatch)
        assert meta["provider"] == "novita/fp8"
        assert meta["input_tokens"] == 2300
        assert meta["output_tokens"] == 400

    def test_missing_key_says_where_to_get_one(self, monkeypatch):
        from vo_coach.mentor.providers import OpenRouterProvider, ProviderError

        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(ProviderError, match="openrouter.ai/keys"):
            OpenRouterProvider()._client(cfg_with())

    def test_empty_response_reports_the_finish_reason(self, monkeypatch):
        from vo_coach.mentor import providers

        class _Msg:
            content = ""

        class _Choice:
            message = _Msg()
            finish_reason = "length"

        class _Resp:
            choices = [_Choice()]
            usage = None
            model = "m"
            provider = "h"

        class _Completions:
            def create(self, **kw):
                return _Resp()

        class _Chat:
            completions = _Completions()

        class _Client:
            chat = _Chat()

        monkeypatch.setattr(providers.OpenRouterProvider, "_client",
                            lambda self, cfg: _Client())
        with pytest.raises(providers.ProviderError, match="length"):
            providers.OpenRouterProvider().complete("s", "u", cfg_with())


def _anthropic(monkeypatch, **cfg_over):
    from vo_coach.mentor import providers

    captured: dict = {}

    class _U:
        input_tokens = 100
        output_tokens = 50
        cache_read_input_tokens = 0

    class _B:
        type = "text"
        text = "ok"

    class _R:
        content = [_B()]
        stop_reason = "end_turn"
        model = "claude-sonnet-5"
        usage = _U()

    class _Messages:
        def create(self, **kw):
            captured.update(kw)
            return _R()

    class _Client:
        messages = _Messages()

    monkeypatch.setattr(providers.AnthropicProvider, "_client",
                        lambda self: _Client())
    cfg = cfg_with(provider="anthropic", model="claude-sonnet-5", **cfg_over)
    text, meta = providers.AnthropicProvider().complete("SYS", "PAYLOAD", cfg)
    return captured, text, meta


class TestAnthropicProvider:
    """Still supported, so the Claude-specific request shape stays tested."""

    def test_system_prompt_sits_behind_a_cache_breakpoint(self, monkeypatch):
        req, _, _ = _anthropic(monkeypatch)
        assert req["system"][0]["cache_control"] == {"type": "ephemeral"}
        assert req["system"][0]["text"] == "SYS"
        assert req["messages"][0]["content"] == "PAYLOAD"

    def test_effort_is_nested_in_output_config(self, monkeypatch):
        req, _, _ = _anthropic(monkeypatch, effort="high")
        assert req["output_config"] == {"effort": "high"}
        assert "effort" not in req

    def test_adaptive_thinking_sends_no_budget_tokens(self, monkeypatch):
        """budget_tokens is removed on current models and returns 400."""
        req, _, _ = _anthropic(monkeypatch, thinking="adaptive")
        assert req["thinking"] == {"type": "adaptive"}
        assert "budget_tokens" not in req["thinking"]

    def test_thinking_can_be_switched_off(self, monkeypatch):
        req, _, _ = _anthropic(monkeypatch, thinking="off")
        assert "thinking" not in req

    def test_refusal_raises_instead_of_returning_empty(self, monkeypatch):
        from vo_coach.mentor import providers

        class _R:
            content = []
            stop_reason = "refusal"
            model = "claude-sonnet-5"
            stop_details = None

        class _Messages:
            def create(self, **kw):
                return _R()

        class _Client:
            messages = _Messages()

        monkeypatch.setattr(providers.AnthropicProvider, "_client",
                            lambda self: _Client())
        with pytest.raises(providers.ProviderError, match="declined"):
            providers.AnthropicProvider().complete(
                "s", "u", cfg_with(provider="anthropic", model="claude-sonnet-5"))


class TestCachingAcrossProviders:
    def test_second_identical_call_skips_the_provider(self, tmp_path, monkeypatch):
        from vo_coach.mentor import client as mclient
        from vo_coach.mentor import providers

        monkeypatch.setattr(cache, "MENTOR_CACHE", tmp_path / "m")
        calls = {"n": 0}

        class _Fake:
            name = "fake"

            def complete(self, system, user, cfg):
                calls["n"] += 1
                return "coaching", {"model": "m", "provider": "fake",
                                    "input_tokens": 1, "output_tokens": 1,
                                    "cached": False}

        monkeypatch.setattr(providers, "get_provider", lambda cfg: _Fake())

        _, m1 = mclient.coach(mk(), _score(_f()), cfg=cfg_with())
        _, m2 = mclient.coach(mk(), _score(_f()), cfg=cfg_with())
        assert calls["n"] == 1, "second identical call should not hit the provider"
        assert m1["cached"] is False and m2["cached"] is True
