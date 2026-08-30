"""
The one place a model call is made, retried, pinned and costed.

=============================================================================
 EVERYTHING GOES THROUGH OPENROUTER
=============================================================================
One key (OPENROUTER_API_KEY) for Claude and for everything else. OpenRouter is
OpenAI-compatible, so this is the `openai` SDK pointed at a different base_url --
the same shape vo_coach/src/vo_coach/mentor/providers.py already uses. Keeping
both halves of the project on one provider means one key, one bill, and one set
of habits.

=============================================================================
 HOST PINNING IS MANDATORY, NOT TUNING
=============================================================================
This is vo_coach's STATUS.md rule and it applies here unchanged. OpenRouter
routes freely across upstream hosts, and the same model costs different amounts
depending on which one serves it. Measured today for the models below:

    anthropic/claude-sonnet-5   anthropic tag           $2.00 / $10.00 per Mtok
                                google-vertex/us        $2.20 / $11.00
                                amazon-bedrock/us-east-1 $2.20 / $11.00

A 10% spread is not the 6.7x that DeepSeek shows, but unpinned routing makes the
cost of a run unreproducible for no benefit. `allow_fallbacks: false` so a bad
slug fails loudly instead of silently routing somewhere expensive.

=============================================================================
 VERIFY EVERY SLUG AGAINST THE ENDPOINTS API
=============================================================================
    https://openrouter.ai/api/v1/models/<model>/endpoints

Never read a provider tag off the OpenRouter providers tab. vo_coach lost a
first real call to exactly that: the display name said "fp8", the routing slug
was `novita/fast`, and with allow_fallbacks off it was unroutable.

The same trap caught this pipeline. The notebook's checker model was
`claude-haiku-4-5-20251001` -- the Anthropic API's dated id, which is NOT an
OpenRouter slug. On OpenRouter it is `anthropic/claude-haiku-4.5`. Every slug and
provider tag in config.py was checked against the endpoints API on 2026-08-23.
"""

from __future__ import annotations

import json
import os
import re
import time

BASE_URL = "https://openrouter.ai/api/v1"

# Sent for OpenRouter's attribution/rankings. Harmless, and it makes this
# pipeline's spend identifiable on the activity page.
APP_TITLE = "weird-bio stickman pipeline"

# Never let OpenRouter pick the host. See config.PROVIDER_PINS for the tags and
# why each one was chosen; a model with no pin there is refused, not routed.
ALLOW_FALLBACKS = False

# --- Running API spend, so a run that quietly costs $12 in fix-loops is visible ---
# OpenRouter returns the real billed cost per call when usage accounting is on,
# so this table is only the fallback for when it does not come back.
USAGE: dict[str, dict] = {}
PRICE_PER_MTOK = {   # USD per million tokens (input, output), OpenRouter prices
    "anthropic/claude-sonnet-5": (2.0, 10.0),
    "anthropic/claude-haiku-4.5": (1.0, 5.0),
    "anthropic/claude-opus-5": (5.0, 25.0),
    "inclusionai/ling-3.0-flash": (0.042, 0.126),   # the novita/fast tag
    "deepseek/deepseek-v4-flash": (0.049, 0.098),   # baidu/fp8 or streamlake/fp8
}

_client = None


class ProviderError(RuntimeError):
    pass


def provider_for(model: str) -> list[str]:
    """The pinned host order for one model.

    Refuses an unpinned model rather than letting OpenRouter route freely. That
    is the whole discipline: an unpinned call can be served by a host charging
    several times more, and nothing in the result says so."""
    from . import config

    pins = config.PROVIDER_PINS.get(model)
    if not pins:
        raise ProviderError(
            f"No provider pin for {model!r}.\n"
            f"  Add one to PROVIDER_PINS in pipeline/config.py, using a tag from\n"
            f"      https://openrouter.ai/api/v1/models/{model}/endpoints\n"
            f"  Unpinned routing is refused on purpose - it makes a run's cost\n"
            f"  unreproducible, and a wrong tag then fails silently instead of loudly."
        )
    return list(pins)


def _reasoning_for(model: str) -> dict | None:
    """OpenRouter's `reasoning` field for this model, or None to omit it.
    See config.REASONING for the measurements behind each entry."""
    from . import config

    return config.REASONING.get(model)


def client():
    """Lazy, so importing config to list the topic bank does not require a key."""
    global _client
    if _client is not None:
        return _client

    try:
        from openai import OpenAI
    except ImportError as e:                       # pragma: no cover
        raise ProviderError("pip install openai - the OpenRouter client uses it.") from e

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise ProviderError(
            "OPENROUTER_API_KEY is not set.\n"
            "  Get one at openrouter.ai/keys, then add it to f:\\wierd_bio\\.env:\n"
            "      OPENROUTER_API_KEY=sk-or-v1-..."
        )

    _client = OpenAI(
        base_url=BASE_URL,
        api_key=key,
        default_headers={"X-Title": APP_TITLE},
    )
    return _client


def _record(model: str, usage, reported_cost: float | None) -> None:
    u = USAGE.setdefault(model, {"calls": 0, "in": 0, "out": 0, "reasoning": 0, "cost": 0.0,
                                 "cost_is_real": False})
    u["calls"] += 1
    u["in"] += getattr(usage, "prompt_tokens", 0) or 0
    u["out"] += getattr(usage, "completion_tokens", 0) or 0
    u["reasoning"] += _reasoning_tokens(usage)
    if reported_cost is not None:
        u["cost"] += reported_cost
        u["cost_is_real"] = True


def _reasoning_tokens(usage) -> int:
    """OpenRouter nests this under completion_tokens_details, and only for models
    that reason. Billed and TIMED as output tokens but never visible in the text,
    so a model quietly thinking for 800 tokens shows up as a number here rather
    than as an unexplained slow, expensive call. Absent on Claude; never assume
    it is there."""
    details = getattr(usage, "completion_tokens_details", None) if usage else None
    if not details:
        return 0
    return int(getattr(details, "reasoning_tokens", 0) or 0)


def _reported_cost(usage) -> float | None:
    """OpenRouter returns the actual billed cost when usage accounting is enabled.
    Preferred over the local price table, which drifts the moment pricing changes."""
    if usage is None:
        return None
    cost = getattr(usage, "cost", None)
    if cost is None:
        # some SDK versions surface it only in the raw dict
        raw = getattr(usage, "model_extra", None) or {}
        cost = raw.get("cost")
    try:
        return float(cost) if cost is not None else None
    except (TypeError, ValueError):
        return None


def print_usage() -> float:
    total = 0.0
    any_estimated = False
    print("\n  API usage this run:")
    for model, u in USAGE.items():
        if u["cost_is_real"]:
            cost = u["cost"]
            mark = " "
        else:
            pin, pout = PRICE_PER_MTOK.get(model, (0.0, 0.0))
            cost = u["in"] / 1e6 * pin + u["out"] / 1e6 * pout
            mark = "~"
            any_estimated = True
        total += cost
        think = f"  (+{u['reasoning']} reasoning)" if u["reasoning"] else ""
        print(f"    {model:<34} {u['calls']:>3} calls  "
              f"{u['in']:>7} in / {u['out']:>6} out  {mark}${cost:.4f}{think}")
    print(f"    {'TOTAL':<34} {'':>3}        {'':>7}   {'':>6}   {'~' if any_estimated else ' '}"
          f"${total:.4f}")
    if any_estimated:
        print("    (~ = estimated from the local price table; the rest is OpenRouter's billed cost)")
    return total


def call_model(model, prompt, system=None, max_tokens=4000, temperature=0, retries=4):
    """One retry wrapper for every model call in this pipeline. Overloads and
    transient 5xx are common enough on long runs that a bare call will eventually
    kill a pipeline 40 minutes in."""
    extra: dict = {
        "provider": {"order": provider_for(model), "allow_fallbacks": ALLOW_FALLBACKS},
        # Ask OpenRouter to return the real billed cost on every call.
        "usage": {"include": True},
    }
    reasoning = _reasoning_for(model)
    if reasoning is not None:
        extra["reasoning"] = dict(reasoning)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    last = None
    escalated = False
    for attempt in range(retries):
        try:
            resp = client().chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
                extra_body=extra,
            )

            if not getattr(resp, "choices", None):
                # An OpenRouter-level error arrives as a 200 with an `error` key
                # rather than an exception, which is how a bad provider pin looks.
                err = getattr(resp, "error", None)
                raise ProviderError(f"No choices in the response. error={err!r}")

            choice = resp.choices[0]
            text = (choice.message.content or "").strip()
            if not text:
                if choice.finish_reason == "length" and not escalated:
                    # An empty body AT the length ceiling means the whole budget
                    # was spent on invisible reasoning - tokens that bill and time
                    # like output but never reach `text`. REASONING only lists
                    # models this has been MEASURED for (see the note above); a
                    # model with no entry there defaults to whatever OpenRouter
                    # picks, which can be reasoning-on. That is a different failure
                    # than a genuine overload, so the backoff below would just
                    # fail identically - force reasoning off and widen the budget
                    # once instead of losing a run that got this far.
                    escalated = True
                    extra["reasoning"] = {"enabled": False}
                    max_tokens = min(max_tokens * 2, 16000)
                    print(f"    [retry {attempt + 1}/{retries}] empty response at the length "
                          f"ceiling - forcing reasoning off, raising max_tokens to {max_tokens}")
                    continue
                raise ProviderError(
                    f"Empty response (finish_reason={choice.finish_reason!r}). "
                    f"If this is 'length', raise max_tokens."
                )

            usage = getattr(resp, "usage", None)
            _record(model, usage, _reported_cost(usage))
            return text

        except Exception as e:
            last = e
            # RETRY ONLY WHAT CAN SUCCEED ON A RETRY.
            # The notebook retried everything four times with exponential
            # backoff. That is right for an overload and wrong for anything
            # permanent: a bad key, a bad slug or a malformed request will fail
            # identically every time, so the backoff just burns 45 seconds per
            # call before reporting the same error. Over a run with dozens of
            # calls that turns "your key is wrong" into a twenty-minute hang.
            if not _is_retryable(e):
                raise ProviderError(f"{type(e).__name__}: {e}") from e
            wait = 3 * (2 ** attempt)
            print(f"    [retry {attempt+1}/{retries}] {type(e).__name__}: {e} - waiting {wait}s")
            time.sleep(wait)

    raise RuntimeError(f"Model call failed after {retries} attempts: {last}")


# HTTP statuses that a retry can plausibly fix: rate limits, overloads and
# upstream faults. Everything else is a fault in the request itself.
# 529 is Anthropic's "overloaded", which OpenRouter passes through -- the single
# most common transient failure on a long run, so leaving it out would defeat the
# point of retrying at all.
_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504, 520, 522, 524, 529}


def _is_retryable(e: Exception) -> bool:
    status = getattr(e, "status_code", None) or getattr(e, "code", None)
    if isinstance(status, int):
        return status in _RETRYABLE_STATUS
    resp = getattr(e, "response", None)
    if resp is not None and isinstance(getattr(resp, "status_code", None), int):
        return resp.status_code in _RETRYABLE_STATUS
    # No status at all: a connection reset, DNS blip or read timeout. Worth
    # another go. A ProviderError we raised ourselves is not.
    return not isinstance(e, ProviderError)


def parse_json_response(raw: str):
    """Models sometimes fence JSON or add a sentence before it. Strip both."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1:
        raw = raw[start:end + 1]
    return json.loads(raw)


def call_model_json(model, prompt, system=None, max_tokens=4000, temperature=0,
                    retries=4, repair_retries=2):
    """call_model + parse_json_response, with one extra layer: if the reply does
    not parse, ask the model to fix its own output instead of losing the whole
    run to it.

    A response failing to parse is usually one stray unescaped quote or newline
    inside a string value - not the model getting the content wrong - and by the
    time it happens the API call is already paid for. Losing evidence.run() (or
    any other gate) to that, forty seconds and one billed call into a run, is a
    worse failure than spending one more small call asking for a fix. Every
    gate in this pipeline shares this exact call_model -> parse_json_response
    shape, so the retry lives here once rather than at each of the eight sites
    that would otherwise duplicate it.
    """
    raw = call_model(model, prompt, system=system, max_tokens=max_tokens,
                     temperature=temperature, retries=retries)
    try:
        return parse_json_response(raw)
    except json.JSONDecodeError as e:
        last_err, last_raw = e, raw
        for attempt in range(repair_retries):
            print(f"  malformed JSON from the model ({last_err}) - "
                  f"asking it to fix (attempt {attempt + 1}/{repair_retries})...")
            fix_prompt = (
                "Your previous reply was not valid JSON. It failed to parse with this "
                f"error: {last_err}\n\n"
                "Return ONLY the corrected, valid JSON - no explanation, no code fence, "
                "no leading or trailing text. Preserve every field and value from your "
                "original reply; fix only what breaks parsing (commonly: an unescaped "
                "quote or a raw newline inside a string value).\n\n"
                f"YOUR PREVIOUS REPLY:\n{last_raw}"
            )
            try:
                last_raw = call_model(model, fix_prompt, system=system,
                                      max_tokens=max_tokens, temperature=0, retries=retries)
                return parse_json_response(last_raw)
            except json.JSONDecodeError as e2:
                last_err = e2
        raise last_err


def _unknown_slug_message(m: str) -> str:
    return (
        f"{m!r} is not a routable OpenRouter model.\n"
        "  Check the slug at openrouter.ai/models. The Anthropic API's dated ids\n"
        "  (e.g. claude-haiku-4-5-20251001) are NOT OpenRouter slugs - the\n"
        "  OpenRouter form is anthropic/claude-haiku-4.5."
    )


def preflight() -> None:
    """Check the key and every configured slug BEFORE the pipeline spends anything.

    A bad slug with allow_fallbacks off is a hard failure, and discovering it
    after the source fetch and the first writer call is the expensive way to find
    out. This costs one unauthenticated GET per model."""
    import urllib.request

    from . import config

    if not os.environ.get("OPENROUTER_API_KEY"):
        raise ProviderError(
            "OPENROUTER_API_KEY is not set.\n"
            "  Get one at openrouter.ai/keys, then add it to f:\\wierd_bio\\.env:\n"
            "      OPENROUTER_API_KEY=sk-or-v1-..."
        )

    roles = config.active_models()
    models = sorted(set(roles.values()))
    by_model = {m: "/".join(r for r, mm in sorted(roles.items()) if mm == m) for m in models}

    print("Preflight: checking model slugs and provider pins against OpenRouter...")
    for m in models:
        pins = provider_for(m)          # raises if the model has no pin at all
        url = f"{BASE_URL}/models/{m}/endpoints"
        try:
            with urllib.request.urlopen(url, timeout=15) as r:
                data = json.loads(r.read().decode("utf-8")).get("data", {})
        except urllib.error.HTTPError as e:
            # 404 is an ANSWER, not a failure to reach the API: the slug does not
            # exist. Treating it as "could not verify" is how the one bug this
            # function exists to catch would sail straight through - which is
            # exactly what the first version of it did.
            if e.code == 404:
                raise ProviderError(_unknown_slug_message(m)) from e
            print(f"  {m:<34} could not verify (HTTP {e.code}) - continuing")
            continue
        except Exception as e:
            # Genuinely could not reach OpenRouter (offline, DNS, timeout). Not
            # evidence against the slug, so warn and let the run proceed.
            print(f"  {m:<34} could not verify ({type(e).__name__}) - continuing")
            continue

        tags = [e.get("tag") for e in data.get("endpoints", [])]
        if not tags:
            raise ProviderError(_unknown_slug_message(m))
        missing = [p for p in pins if p not in tags]
        if missing:
            raise ProviderError(
                f"{m!r} is not served by pinned provider(s) {missing}. "
                f"With allow_fallbacks={ALLOW_FALLBACKS} this call would hard-fail.\n"
                f"  Available tags: {', '.join(t for t in tags if t)}"
            )
        price = PRICE_PER_MTOK.get(m)
        cost = f"${price[0]:g}/${price[1]:g} per Mtok" if price else ""
        # ", " not "/" - provider tags contain slashes (baidu/fp8), and joining
        # them with another one is unreadable.
        print(f"  {by_model[m]:<22} {m:<28} {cost:<20} via {', '.join(pins)}")
