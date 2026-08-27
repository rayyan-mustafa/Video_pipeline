"""Mentor output validators.

These matter more than most tests here: the validators are the only mechanical
evidence about whether a cheap model is doing the job. A validator that misses
a real breach gives false confidence, and one that fires on a good reply
poisons the failure rate it exists to measure - so both directions are tested.
"""

from __future__ import annotations

import json

import pytest

from vo_coach.mentor import validate as V

GOOD = """Solid, easy-to-follow read - the energy holds all the way through.

**Hiss under the voice** You are sitting at 14 dB signal-to-noise, which is low
enough that consonants start to blur and the listener works harder to follow
you. Move closer and raise gain at the mic.

**Sharp sibilants** There are 18 per minute, and they land hard enough to be
distracting on headphones. Angle the mic slightly off-axis.
"""


def payload(**over) -> str:
    base = {
        "take": {"id": "t1", "duration_sec": 60.0, "words": 188, "wpm": 188.1},
        "findings": [
            {"metric": "noise", "severity": "must_fix", "detail": "14.4 dB SNR"},
            {"metric": "sibilance", "severity": "stylistic",
             "detail": "18.2 per minute"},
        ],
        "max_fixes": 3,
    }
    base.update(over)
    return json.dumps(base, indent=2)


class TestCleanReply:
    def test_a_good_reply_produces_nothing(self):
        r = V.validate(GOOD, payload())
        assert r.violations == [], [v.rule for v in r.violations]
        assert r.clean

    def test_rounding_a_measurement_is_not_invention(self):
        """The payload says 14.4; the reply says 14. Flagging that would make
        the failure rate meaningless, because good coaching rounds."""
        r = V.validate("**Noise** You are at 14 dB, and the listener notices.",
                       payload())
        assert not r.hard

    def test_large_values_tolerate_proportional_rounding(self):
        r = V.validate("**Pace** You run at 190 wpm, which the listener feels.",
                       payload())
        assert not r.hard


class TestRule5InventedNumbers:
    def test_a_number_absent_from_the_payload_is_flagged(self):
        r = V.validate("**Noise** Your take measures 42 dB SNR, so listeners "
                       "struggle.", payload())
        assert [v.rule for v in r.hard] == ["invented_number"]

    def test_bare_numbers_are_not_checked(self):
        """A bare '2' is a list index or a count, not a claim about the audio.
        Checking those would fire constantly on correct replies."""
        r = V.validate("**Noise** Two things, the listener hears both. Step 2.",
                       payload())
        assert not r.hard

    def test_each_bad_number_reported_once(self):
        r = V.validate("**A** 42 dB here and 42 dB there, listeners notice.",
                       payload())
        assert len(r.hard) == 1


class TestRule4Identity:
    """The rule that matters most to a non-US speaker: this is the line between
    a delivery coach and an accent-reduction service."""

    @pytest.mark.parametrize("bad", [
        "**Clarity** Try to soften your accent so listeners follow more easily.",
        "**Delivery** You should sound more American here for the audience.",
        "**Tone** Lower your natural pitch a little so the listener settles.",
        "**Read** Your accent is distracting for a US listener.",
    ])
    def test_instructions_to_change_identity_are_flagged(self, bad):
        r = V.validate(bad, payload())
        assert "identity_coaching" in [v.rule for v in r.hard], bad

    def test_reassurance_about_accent_is_not_flagged(self):
        """A reply may legitimately tell the speaker their accent is fine. The
        check is on instructions to change it, not on the word appearing."""
        r = V.validate("**Pace** Your accent is not the issue; the listener "
                       "follows it fine. The rush at the ends of lines is.",
                       payload())
        assert not r.hard

    def test_pitch_range_coaching_is_allowed(self):
        """Range and expression are fair game per the prompt - only mean pitch
        and timbre are identity."""
        r = V.validate("**Flat delivery** Widening your pitch range on key "
                       "words gives the listener something to hold onto.",
                       payload())
        assert not r.hard


class TestRule1Shape:
    def test_more_than_max_fixes(self):
        text = "Reads well.\n\n**One** a\n\n**Two** b\n\n**Three** c\n\n**Four** d"
        r = V.validate(text, payload(), max_fixes=3)
        assert "too_many_fixes" in [v.rule for v in r.hard]

    def test_numbered_fixes_still_count(self):
        text = ("ok\n\n1. **One** a\n\n2. **Two** b\n\n3. **Three** c\n\n"
                "4. **Four** d")
        assert "too_many_fixes" in [v.rule for v in V.validate(text, payload()).hard]

    def test_exactly_max_fixes_is_fine(self):
        text = ("Reads well for the listener.\n\n**One** a\n\n**Two** b\n\n"
                "**Three** c")
        assert not V.validate(text, payload()).hard

    def test_coaching_when_the_rubric_flagged_nothing(self):
        """The prompt says not to go looking for something to say. A model that
        always finds three fixes is not coaching, it is filling a template."""
        r = V.validate("**Something** the listener may notice", payload(findings=[]))
        assert "invented_findings" in [v.rule for v in r.hard]

    def test_saying_nothing_is_worth_saying_is_clean(self):
        r = V.validate("Nothing worth fixing here - it is clear and easy to "
                       "follow.", payload(findings=[]))
        assert r.violations == []


class TestUnmeasuredMetrics:
    def test_pacing_comment_when_timings_were_unavailable(self):
        p = payload(timing_metrics_unavailable="no word timings")
        r = V.validate("**Pace** You are speaking too fast for the listener.", p)
        assert "commented_on_unmeasured" in [v.rule for v in r.hard]

    def test_excluded_benchmark_metric_is_soft_not_hard(self):
        """Excluded from BENCHMARKING is not the same as never measured - SNR is
        still measured on the take itself, just not compared to a lossy
        reference. Hard-failing that would be wrong."""
        p = payload(benchmark={"deltas": [], "not_measured": ["snr_db"]})
        r = V.validate("**Hiss** the listener hears it", p)
        rules = [v.rule for v in r.violations]
        assert "compared_excluded_metric" in rules
        assert not r.hard


class TestSoftChecks:
    def test_praise_preamble(self):
        r = V.validate("Great job! **Noise** the listener hears hiss", payload())
        assert "praise_preamble" in [v.rule for v in r.violations]
        assert not r.hard, "tone is never a hard failure"

    def test_measurement_without_listener_rationale(self):
        r = V.validate("**Noise** You are at 14 dB SNR.", payload())
        assert "no_listener_rationale" in [v.rule for v in r.violations]

    def test_stylistic_finding_discussed_as_a_fault(self):
        r = V.validate("Reads well for the listener.\n\n**Sibilance** You must "
                       "re-record this, the essing is unacceptable.", payload())
        assert "promoted_stylistic" in [v.rule for v in r.violations]

    def test_must_fix_finding_may_use_must_language(self):
        r = V.validate("Reads ok.\n\n**Hiss** You need to fix the noise floor "
                       "before the listener can follow you.", payload())
        assert "promoted_stylistic" not in [v.rule for v in r.violations]


class TestAuditLog:
    def test_rate_is_per_model_and_counts_calls_not_violations(self, tmp_path):
        """A reply with three violations is still ONE failed call. Counting
        violations would let one bad reply look like three."""
        p = tmp_path / "audit.jsonl"
        V.log(V.validate("**A** 42 dB, 99 dB, soften your accent", payload(),
                         model="cheap"), path=p)
        V.log(V.validate(GOOD, payload(), model="cheap"), path=p)
        r = V.rates(path=p)["cheap"]
        assert r["calls"] == 2
        assert r["with_hard"] == 1
        assert r["hard_rate"] == 0.5

    def test_models_are_tracked_separately(self, tmp_path):
        """The whole point: compare the cheap model's rate against another's on
        the same kind of input."""
        p = tmp_path / "a.jsonl"
        V.log(V.validate("**A** 42 dB", payload(), model="cheap"), path=p)
        V.log(V.validate(GOOD, payload(), model="expensive"), path=p)
        r = V.rates(path=p)
        assert r["cheap"]["hard_rate"] == 1.0
        assert r["expensive"]["hard_rate"] == 0.0

    def test_corrupt_line_is_skipped_not_fatal(self, tmp_path):
        p = tmp_path / "a.jsonl"
        V.log(V.validate(GOOD, payload(), model="m"), path=p)
        with open(p, "a", encoding="utf-8") as f:
            f.write("{not json\n")
        assert V.rates(path=p)["m"]["calls"] == 1

    def test_missing_log_is_empty_not_an_error(self, tmp_path):
        assert V.rates(path=tmp_path / "nope.jsonl") == {}


class TestWiredIntoTheClient:
    def test_coach_attaches_a_report(self, tmp_path, monkeypatch):
        from vo_coach.mentor import cache, client as mclient, providers
        from tests.test_mentor import _f, _score
        from tests.test_providers import cfg_with
        from tests.test_scoring import mk

        monkeypatch.setattr(cache, "MENTOR_CACHE", tmp_path / "m")
        monkeypatch.setattr(V, "AUDIT_LOG", tmp_path / "audit.jsonl")

        class _Fake:
            name = "fake"

            def complete(self, system, user, cfg):
                return ("**A** 42 dB\n\n**B** x\n\n**C** y\n\n**D** z",
                        {"model": "fake-1", "provider": "h",
                         "input_tokens": 1, "output_tokens": 1})

        monkeypatch.setattr(providers, "get_provider", lambda cfg: _Fake())
        _, meta = mclient.coach(mk(), _score(_f()), cfg=cfg_with())
        rules = [v.rule for v in meta["validation"].violations]
        assert "invented_number" in rules and "too_many_fixes" in rules
        assert V.rates(path=tmp_path / "audit.jsonl")["fake-1"]["calls"] == 1

    def test_cache_hits_are_checked_but_not_logged(self, tmp_path, monkeypatch):
        """Logging a replayed answer would move the failure rate with no new
        evidence behind it."""
        from vo_coach.mentor import cache, client as mclient, providers
        from tests.test_mentor import _f, _score
        from tests.test_providers import cfg_with
        from tests.test_scoring import mk

        monkeypatch.setattr(cache, "MENTOR_CACHE", tmp_path / "m")
        monkeypatch.setattr(V, "AUDIT_LOG", tmp_path / "audit.jsonl")

        class _Fake:
            name = "fake"

            def complete(self, system, user, cfg):
                return "**A** 42 dB", {"model": "fake-1", "provider": "h"}

        monkeypatch.setattr(providers, "get_provider", lambda cfg: _Fake())
        mclient.coach(mk(), _score(_f()), cfg=cfg_with())
        _, meta = mclient.coach(mk(), _score(_f()), cfg=cfg_with())
        assert meta["cached"] is True
        assert meta["validation"].hard, "cached replies are still checked"
        assert V.rates(path=tmp_path / "audit.jsonl")["fake-1"]["calls"] == 1
