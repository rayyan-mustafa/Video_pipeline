"""The golden-set harness.

Two things are being tested here, and only one of them is the code. The other
is the CASES themselves: a case whose fixture no longer produces the findings it
was written around still runs, still costs an API call, and still reports a
score - just a meaningless one. `test_no_case_is_stale` is the guard, and it is
the most valuable test in this file.
"""

from __future__ import annotations

import pytest

from vo_coach.evaluation import cases as C
from vo_coach.evaluation import runner as R
from vo_coach.mentor import validate as V
from vo_coach.schemas import ValidationReport


class TestTheCasesThemselves:
    def test_no_case_is_stale(self):
        """Every case must still produce exactly the findings it was built to
        exercise. When a threshold moves, this fails first - before an API call
        is spent judging a reply against a take that no longer resembles the
        case."""
        stale = {c.id: R.check_rubric(c) for c in C.CASES}
        stale = {k: v for k, v in stale.items() if v}
        assert not stale, stale

    def test_every_case_has_a_reason(self):
        """A case without a stated reason gets deleted by the next person who
        sees it fail."""
        for c in C.CASES:
            assert len(c.why) > 60, c.id

    def test_case_ids_are_unique(self):
        ids = [c.id for c in C.CASES]
        assert len(ids) == len(set(ids))

    def test_expected_topics_are_real_patterns(self):
        """A typo in `topics_required` would otherwise make a case
        unsatisfiable, and the model would be blamed for it."""
        for c in C.CASES:
            for t in c.expect.topics_required + c.expect.topics_forbidden:
                assert t in V.TOPIC_PATTERNS, f"{c.id}: unknown topic {t!r}"

    def test_the_clean_case_really_is_clean(self):
        """If the rubric flags anything here, the single most important case in
        the set - can the coach say nothing - is not being tested at all."""
        from vo_coach.scoring.rubric import score

        assert score(C.by_id("clean_take").analysis).findings == []

    def test_the_accent_case_offers_no_accent_data(self):
        """The case works only if nothing in the payload names an accent. Any
        mention in the reply then has to be the model inferring one."""
        import json

        from vo_coach.mentor.prompt import build_payload
        from vo_coach.scoring.rubric import score

        c = C.by_id("accent_bait")
        payload = json.dumps(json.loads(
            build_payload(c.analysis, score(c.analysis)))).lower()
        assert "accent" not in payload
        assert "confidence" not in payload

    def test_slow_fixture_produces_multiple_sentences(self):
        """Without inter-sentence gaps the whole fixture is one sentence, the
        sustained-drift rule never trips, and the case tests nothing while
        still appearing to pass."""
        from vo_coach.analysis.pacing import sentence_wpm

        c = C.by_id("sustained_slow")
        assert len(sentence_wpm(c.analysis.words)) >= 4

    def test_by_id_names_the_alternatives(self):
        with pytest.raises(KeyError, match="clean_take"):
            C.by_id("nope")


def _report(*viols) -> ValidationReport:
    return ValidationReport(take_id="t", model="m", violations=list(viols))


class TestChecking:
    def test_a_hard_violation_fails_the_case(self):
        c = C.by_id("clipping_only")
        rep = V.validate("**Clipping** you clipped, listeners hear it. 42 dB.",
                         '{"findings": [{"metric": "clipping"}]}')
        assert any("invented_number" in f for f in R.check(c, "x", rep))

    def test_missing_the_required_topic_fails(self):
        """A reply can break no rule and still be useless - three paragraphs
        that never mention the one thing that was wrong."""
        c = C.by_id("clipping_only")
        fails = R.check(c, "**Delivery** Nice warm read, the listener enjoys it.",
                        _report())
        assert any("never mentions clipping" in f for f in fails)

    def test_raising_a_forbidden_topic_fails(self):
        c = C.by_id("accent_bait")
        fails = R.check(c, "**Hiss** noise floor is high.\n\nWork on your accent.",
                        _report())
        assert any("raises accent" in f for f in fails)

    def test_too_many_fixes_fails_on_the_case_limit(self):
        """Separate from the rule-1 validator: a case can demand FEWER than
        max_fixes, because one real fault does not need three paragraphs."""
        c = C.by_id("clipping_only")          # max_fixes=1
        fails = R.check(c, "**A** clipping\n\n**B** x\n\n", _report())
        assert any("expected at most 1" in f for f in fails)

    def test_a_correct_reply_has_no_failures(self):
        c = C.by_id("clipping_only")
        good = ("Solid read otherwise.\n\n**Clipped peaks** The signal hit full "
                "scale twice, and the listener hears it as a hard edge on those "
                "words. Re-record with lower gain.")
        assert R.check(c, good, _report()) == []


class TestRunning:
    def _fake(self, monkeypatch, text: str):
        from vo_coach.mentor import cache, providers

        class _Fake:
            name = "fake"

            def complete(self, system, user, cfg):
                return text, {"model": "fake-1", "provider": "h",
                              "input_tokens": 10, "output_tokens": 5}

        monkeypatch.setattr(providers, "get_provider", lambda cfg: _Fake())
        return cache

    def test_a_stale_case_never_reaches_the_provider(self, monkeypatch, tmp_path):
        """The expensive call must not happen for a case that cannot produce a
        meaningful score."""
        from vo_coach.mentor import providers

        called = {"n": 0}

        def _boom(cfg):
            called["n"] += 1
            raise AssertionError("provider must not be called")

        monkeypatch.setattr(providers, "get_provider", _boom)
        bad = C.Case(id="bogus", why="x" * 70,
                     analysis=C.by_id("clean_take").analysis,
                     expect=C.Expect(expect_rubric=[("noise", "must_fix")]))
        res = R.run_case(bad)
        assert not res.passed
        assert "stale" in res.failures[0]
        assert called["n"] == 0

    def test_run_all_reports_a_rate(self, monkeypatch, tmp_path):
        cache = self._fake(monkeypatch, "Clean and easy to follow.")
        monkeypatch.setattr(cache, "MENTOR_CACHE", tmp_path / "m")
        run = R.run_all([C.by_id("clean_take"), C.by_id("clipping_only")])
        # The clean case passes on this reply; the clipping case must not.
        assert run.rate == 0.5
        assert run.results[0].passed and not run.results[1].passed

    def test_a_missing_key_is_a_failure_not_a_crash(self, monkeypatch, tmp_path):
        """A whole run must survive one broken case - otherwise a single
        transport error costs the other eight calls."""
        from vo_coach.mentor import cache, providers
        from vo_coach.mentor.providers import ProviderError

        monkeypatch.setattr(cache, "MENTOR_CACHE", tmp_path / "m")

        class _Fake:
            name = "fake"

            def complete(self, system, user, cfg):
                raise ProviderError("OPENROUTER_API_KEY is not set")

        monkeypatch.setattr(providers, "get_provider", lambda cfg: _Fake())
        run = R.run_all([C.by_id("clean_take")])
        assert run.rate == 0.0
        assert "OPENROUTER_API_KEY" in run.results[0].failures[0]


class TestStorage:
    def test_saved_run_is_readable_back(self, monkeypatch, tmp_path):
        monkeypatch.setattr(R, "EVAL_DIR", tmp_path)
        run = R.EvalRun(model="a/b", provider="openrouter", ran_at="now",
                        results=[R.CaseResult(case_id="c", passed=True)])
        R.save(run)
        rows = R.stored()
        assert rows[0]["model"] == "a/b" and rows[0]["rate"] == 1.0

    def test_model_slug_survives_a_slash(self, monkeypatch, tmp_path):
        """`openai/gpt-oss-120b` would otherwise write into a subdirectory that
        does not exist."""
        monkeypatch.setattr(R, "EVAL_DIR", tmp_path)
        R.save(R.EvalRun(model="openai/gpt-oss-120b", provider="o", ran_at="n"))
        assert (tmp_path / "openai__gpt-oss-120b.json").exists()

    def test_comparison_of_nothing_says_what_to_run(self, monkeypatch, tmp_path):
        monkeypatch.setattr(R, "EVAL_DIR", tmp_path)
        assert "vo_coach eval" in R.render_comparison(R.stored())

    def test_comparison_orders_best_first(self, monkeypatch, tmp_path):
        monkeypatch.setattr(R, "EVAL_DIR", tmp_path)
        R.save(R.EvalRun(model="weak", provider="o", ran_at="n",
                         results=[R.CaseResult("a", False)]))
        R.save(R.EvalRun(model="strong", provider="o", ran_at="n",
                         results=[R.CaseResult("a", True)]))
        assert R.stored()[0]["model"] == "strong"
