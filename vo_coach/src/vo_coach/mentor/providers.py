"""Pluggable LLM backends for the mentor.

Two providers ship: OpenRouter (default) and Anthropic. Both expose the same
`complete()` signature, so `client.py` never branches on which one is active and
switching is a config edit rather than a code change.

WHY THE SEAM EXISTS. Cost per coaching call differs by ~120x between the
cheapest OpenRouter model and Claude Sonnet - about $1.40/month at Phase 3
volume, which is noise, but roughly $34 vs $0.28 PER VIDEO once Phase 5 fires a
call at every clause boundary. Which model actually coaches better on real
takes is an open question that only A/B testing on your own audio can answer,
so the design keeps both one config line apart.
"""

from __future__ import annotations

from typing import Protocol

from ..config import Config


class ProviderError(RuntimeError):
    pass


class Provider(Protocol):
    name: str

    def complete(self, system: str, user: str, cfg: Config) -> tuple[str, dict]:
        """Return (text, meta). `meta` carries model, tokens and stop reason."""
        ...


# ---------------------------------------------------------------------------
#  OpenRouter (OpenAI-compatible)
# ---------------------------------------------------------------------------
class OpenRouterProvider:
    name = "openrouter"

    BASE_URL = "https://openrouter.ai/api/v1"

    def _client(self, cfg: Config):
        import os

        try:
            from openai import OpenAI
        except ImportError as e:                       # pragma: no cover
            raise ProviderError("`pip install openai` for the OpenRouter provider.") from e

        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise ProviderError(
                "OPENROUTER_API_KEY is not set. Get one at openrouter.ai/keys, "
                "then set it in your environment."
            )
        return OpenAI(base_url=self.BASE_URL, api_key=key)

    def complete(self, system: str, user: str, cfg: Config) -> tuple[str, dict]:
        client = self._client(cfg)
        model = cfg.get("mentor.model")

        extra: dict = {}
        # Pin which upstream host serves the request. OpenRouter otherwise
        # routes freely, and hosts differ in BOTH price and speed for the same
        # model - Novita is ~half the price of Novita Fast and ~10x slower.
        # Leaving that to chance makes cost and latency unreproducible.
        order = cfg.get("mentor.openrouter_providers", None)
        if order:
            extra["provider"] = {
                "order": list(order),
                "allow_fallbacks": bool(cfg.get("mentor.openrouter_allow_fallbacks", False)),
            }

        # Reasoning models bill and TIME their chain-of-thought as output
        # tokens. On a model like gpt-oss-120b, high effort can emit more
        # thinking than coaching - which turns a sub-second reply into several
        # seconds and multiplies the cost, invisibly, because the tokens never
        # appear in the text. Setting effort explicitly is what keeps the
        # latency budget real. Ignored by non-reasoning models.
        effort = cfg.get("mentor.openrouter_reasoning_effort", None)
        if effort:
            extra["reasoning"] = {"effort": effort}

        try:
            resp = client.chat.completions.create(
                model=model,
                max_tokens=cfg.get("mentor.max_tokens", 8000),
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                extra_body=extra or None,
            )
        except Exception as e:
            raise ProviderError(f"{type(e).__name__}: {e}") from e

        if not resp.choices:
            raise ProviderError("No choices in the response.")
        choice = resp.choices[0]
        text = (choice.message.content or "").strip()
        if not text:
            raise ProviderError(
                f"Empty response (finish_reason={choice.finish_reason!r}). "
                f"If this is 'length', raise mentor.max_tokens."
            )

        usage = resp.usage
        return text, {
            # The model actually served, which can differ from what was asked
            # for when fallbacks are allowed.
            "model": getattr(resp, "model", model),
            "provider": getattr(resp, "provider", None) or "openrouter",
            "stop_reason": choice.finish_reason,
            "input_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
            "output_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
            # Billed and timed, but never visible in the text. Reported so a
            # model quietly thinking for 800 tokens shows up as a number rather
            # than as an unexplained slow, expensive call.
            "reasoning_tokens": _reasoning_tokens(usage),
            "cached": False,
        }


def _reasoning_tokens(usage) -> int:
    """OpenRouter nests this under completion_tokens_details, and only for
    models that reason. Absent everywhere else, so never assume it is there."""
    details = getattr(usage, "completion_tokens_details", None) if usage else None
    return int(getattr(details, "reasoning_tokens", 0) or 0) if details else 0


# ---------------------------------------------------------------------------
#  Anthropic
# ---------------------------------------------------------------------------
class AnthropicProvider:
    name = "anthropic"

    def _client(self):
        try:
            import anthropic
        except ImportError as e:                       # pragma: no cover
            raise ProviderError("`pip install anthropic` for the Anthropic provider.") from e

        # Zero-arg on purpose: resolves ANTHROPIC_API_KEY, then
        # ANTHROPIC_AUTH_TOKEN, then an `ant auth login` profile.
        return anthropic.Anthropic()

    def complete(self, system: str, user: str, cfg: Config) -> tuple[str, dict]:
        client = self._client()

        request: dict = {
            "model": cfg.get("mentor.model"),
            "max_tokens": cfg.get("mentor.max_tokens", 8000),
            # Frozen prefix behind a cache breakpoint; take-specific JSON goes
            # in the user turn so the prefix stays byte-identical and caches.
            "system": [{"type": "text", "text": system,
                        "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": user}],
            "output_config": {"effort": cfg.get("mentor.effort", "medium")},
        }
        thinking = cfg.get("mentor.thinking", "adaptive")
        if thinking and thinking != "off":
            request["thinking"] = {"type": thinking}

        try:
            resp = client.messages.create(**request)
        except Exception as e:
            raise ProviderError(f"{type(e).__name__}: {e}") from e

        if resp.stop_reason == "refusal":
            detail = getattr(resp, "stop_details", None)
            raise ProviderError(
                f"The model declined this request"
                f"{f' ({detail.category})' if detail else ''}."
            )

        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        if not text:
            raise ProviderError(
                f"No text in the response (stop_reason={resp.stop_reason!r}). "
                f"If this is 'max_tokens', raise mentor.max_tokens."
            )

        return text, {
            "model": resp.model,
            "provider": "anthropic",
            "stop_reason": resp.stop_reason,
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
            "cache_read_input_tokens": getattr(resp.usage, "cache_read_input_tokens", 0),
            "cached": False,
        }


_PROVIDERS = {
    "openrouter": OpenRouterProvider,
    "anthropic": AnthropicProvider,
}


def get_provider(cfg: Config) -> Provider:
    name = cfg.get("mentor.provider", "openrouter")
    if name not in _PROVIDERS:
        raise ProviderError(
            f"Unknown mentor.provider {name!r}. Known: {', '.join(sorted(_PROVIDERS))}"
        )
    return _PROVIDERS[name]()
