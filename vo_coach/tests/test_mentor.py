"""Phase 3 tests: prompt construction and the response cache.

No live API calls. The mentor's value is in WHAT IT IS TOLD, so these assert on
the payload and the frozen prompt rather than on generated prose.
"""

from __future__ import annotations

import json

import pytest

from vo_coach.mentor import cache
from vo_coach.mentor.prompt import SYSTEM_PROMPT, build_payload
from vo_coach.schemas import (
    BenchmarkResult,
    Finding,
    MetricDelta,
    ScoreResult,
)
from tests.test_scoring import mk


def _score(*findings: Finding) -> ScoreResult:
    return ScoreResult(take_id="t", findings=list(findings))


def _f(metric="noise", severity="must_fix", message="msg", t=None) -> Finding:
    return Finding(metric=metric, severity=severity, message=message, t=t)


class TestSystemPrompt:
    def test_is_frozen_no_interpolation(self):
        """The prompt sits behind a cache breakpoint. A single varying byte
        would invalidate the prefix on every request, so it must contain no
        format placeholders."""
        assert "{" not in SYSTEM_PROMPT and "}" not in SYSTEM_PROMPT
        assert "%s" not in SYSTEM_PROMPT

    def test_states_the_hard_rules(self):
        low = SYSTEM_PROMPT.lower()
        assert "three" in low                    # max 3 fixes
        assert "never invent" in low             # no fabricated measurements
        assert "identity" in low                 # don't coach voice identity


class TestPayload:
    def test_caps_findings_with_a_little_headroom(self):
        """More than max_fixes are sent so the model has alternatives to choose
        between - but not the whole list, which would invite a wall of notes."""
        s = _score(*[_f(metric=f"m{i}") for i in range(12)])
        p = json.loads(build_payload(mk(), s, max_fixes=3))
        assert 3 < len(p["findings"]) <= 6
        assert p["max_fixes"] == 3

    def test_must_fix_is_ordered_first(self):
        s = _score(_f(metric="a", severity="stylistic"),
                   _f(metric="b", severity="must_fix"))
        p = json.loads(build_payload(mk(), s))
        assert p["findings"][0]["metric"] == "b"

    def test_excluded_metrics_are_named_not_omitted(self):
        """Rule 5 depends on the model being able to tell 'measured and fine'
        from 'never measured'. Dropping excluded metrics silently would let it
        assume they passed."""
        b = BenchmarkResult(take_id="t", reference_name="r", niche="n",
                            deltas=[], excluded={"snr_db": "lossy reference"})
        p = json.loads(build_payload(mk(), _score(_f()), benchmark=b))
        assert "snr_db" in p["benchmark"]["not_measured"]

    def test_ok_deltas_are_not_sent(self):
        """Only deltas that actually differ are worth tokens."""
        b = BenchmarkResult(
            take_id="t", reference_name="r", niche="n",
            deltas=[
                MetricDelta(metric="wpm_overall", take_value=1, reference_value=1,
                            delta=0, severity="ok"),
                MetricDelta(metric="pause_rate_per_min", take_value=9,
                            reference_value=2, delta=7, severity="must_fix"),
            ],
        )
        p = json.loads(build_payload(mk(), _score(_f()), benchmark=b))
        assert [d["metric"] for d in p["benchmark"]["deltas"]] == ["pause_rate_per_min"]

    def test_unreliable_timings_are_declared_explicitly(self):
        """Without this the model would read a missing wpm as 'not a problem'
        and might still comment on pacing."""
        p = json.loads(build_payload(mk(), _score(_f()), timings_reliable=False))
        assert "timing_metrics_unavailable" in p
        assert "speaking rate" in p["timing_metrics_unavailable"]

    def test_transcript_is_truncated(self):
        a = mk(transcript="word " * 500)
        p = json.loads(build_payload(a, _score(_f())))
        assert len(p["transcript_excerpt"]) < 700

    def test_no_transcript_key_when_empty(self):
        p = json.loads(build_payload(mk(transcript=""), _score(_f())))
        assert "transcript_excerpt" not in p

    def test_history_is_capped_and_optional(self):
        hist = [{"take_id": f"t{i}"} for i in range(20)]
        p = json.loads(build_payload(mk(), _score(_f()), history=hist))
        assert len(p["previous_takes"]) == 5
        assert "previous_takes" not in json.loads(build_payload(mk(), _score(_f())))

    def test_payload_is_valid_json(self):
        p = build_payload(mk(), _score(_f(t=12.5)))
        assert json.loads(p)["findings"][0]["at_seconds"] == 12.5


class TestCache:
    def test_key_changes_with_every_input_that_affects_the_answer(self):
        base = dict(model="m", system="s", payload="p")
        k = cache.key_for(**base)
        assert k != cache.key_for(**{**base, "model": "m2"})
        assert k != cache.key_for(**{**base, "system": "s2"})
        assert k != cache.key_for(**{**base, "payload": "p2"})
        assert k != cache.key_for(**base, effort="high")

    def test_field_boundaries_cannot_collide(self):
        """Without a delimiter, ('ab','c') and ('a','bc') would hash the same -
        two different prompts silently sharing one cached answer."""
        assert cache.key_for("ab", "c", "x") != cache.key_for("a", "bc", "x")

    def test_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cache, "MENTOR_CACHE", tmp_path / "m")
        cache.save("k1", {"text": "hello", "meta": {}})
        assert cache.load("k1")["text"] == "hello"
        assert cache.load("missing") is None

    def test_corrupt_entry_is_a_miss_not_an_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cache, "MENTOR_CACHE", tmp_path / "m")
        (tmp_path / "m").mkdir(parents=True)
        (tmp_path / "m" / "bad.json").write_text("{not json", encoding="utf-8")
        assert cache.load("bad") is None

    def test_clear_reports_count(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cache, "MENTOR_CACHE", tmp_path / "m")
        cache.save("a", {"text": "1"})
        cache.save("b", {"text": "2"})
        assert cache.clear() == 2
        assert cache.stats()[0] == 0


class TestOfflineMode:
    def test_offline_miss_raises_rather_than_calling_the_api(self, tmp_path, monkeypatch):
        """A loop meant to be free must not quietly start spending money."""
        from pathlib import Path

        from vo_coach.config import Config, load_config
        from vo_coach.mentor.client import OfflineCacheMiss, coach

        monkeypatch.setattr(cache, "MENTOR_CACHE", tmp_path / "m")
        raw = dict(load_config().raw)
        raw["mentor"] = {**raw["mentor"], "offline": True}
        cfg = Config(raw, Path("<test>"))

        with pytest.raises(OfflineCacheMiss):
            coach(mk(), _score(_f()), cfg=cfg)


