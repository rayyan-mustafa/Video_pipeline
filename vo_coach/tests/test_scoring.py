"""Phase 2 tests: the rubric and the competitor benchmark.

The rubric's job is deciding WHICH BAND a measurement falls into, so these
tests assert on severity transitions rather than on message wording. Wording
will change; the banding logic is the contract.
"""

from __future__ import annotations

import pytest

from vo_coach.config import load_config
from vo_coach.schemas import (
    AnalysisResult,
    Click,
    FeatureVector,
    Pacing,
    Pause,
    Pitch,
    ProximityEffect,
    ReferenceFeatures,
    Sibilance,
    SibilancePeak,
    TiltPoint,
    Word,
)
from vo_coach.scoring.benchmark import COMPARISONS, benchmark
from vo_coach.scoring.features import extract
from vo_coach.scoring.rubric import NEVER_MUST_FIX, score


def mk(**over) -> AnalysisResult:
    """A clean take. Each test perturbs exactly one dimension."""
    base = dict(
        take_id="t",
        duration_sec=60.0,
        transcript="",
        words=[],
        pacing=Pacing(wpm_overall=155.0),
        pauses=[],
        pitch=Pitch(mean_hz=120.0, range_hz=80.0, variance=200.0),
        proximity_effect=ProximityEffect(
            spectral_tilt_db_per_octave=[TiltPoint(t=1.0, value=-2.0)]
        ),
        sibilance=Sibilance(),
        noise_floor_db=-55.0,
        snr_db=35.0,
        mouth_clicks=[],
    )
    base.update(over)
    return AnalysisResult(**base)


class TestCleanTake:
    def test_clean_take_produces_no_findings(self):
        """The coach must be capable of saying nothing. A rubric that always
        finds something trains the user to ignore it."""
        assert score(mk()).findings == []


class TestSeverityBanding:
    def test_snr_crosses_silence_then_must_fix(self):
        cfg = load_config()
        sil = cfg.get("noise.silence_snr_db")      # 30
        must = cfg.get("noise.must_fix_snr_db")    # 20

        assert score(mk(snr_db=sil + 5)).findings == []
        mid = score(mk(snr_db=(sil + must) / 2)).findings
        assert mid and mid[0].severity == "stylistic"
        bad = score(mk(snr_db=must - 5)).findings
        assert bad and bad[0].severity == "must_fix"

    def test_unmeasurable_snr_produces_no_finding(self):
        """`None` means the estimator refused to guess. The rubric must not
        invent a finding from an absent measurement."""
        assert score(mk(snr_db=None)).findings == []

    def test_proximity_tilt_bands_on_an_inverted_scale(self):
        """Tilt is 'more negative is worse' - the banding must not assume
        higher-is-worse or it inverts the whole metric."""
        def tilt(v):
            return mk(proximity_effect=ProximityEffect(
                spectral_tilt_db_per_octave=[TiltPoint(t=1.0, value=v)]))

        assert score(tilt(-1.0)).findings == []
        assert score(tilt(-4.5)).findings[0].severity == "stylistic"
        assert score(tilt(-8.0)).findings[0].severity == "must_fix"

    def test_clipping_is_always_must_fix(self):
        f = score(mk(mouth_clicks=[Click(t=1.0)])).findings
        assert f and f[0].severity == "must_fix"


class TestStylisticNeverEscalates:
    def test_narrow_pitch_range_stays_stylistic(self):
        """Spec rule: minor pitch-range narrowness is stylistic, never a fault.
        Even an extreme value must not become must_fix."""
        f = score(mk(pitch=Pitch(mean_hz=120.0, range_hz=1.0, variance=1.0))).findings
        assert f, "expected a stylistic note"
        assert all(x.severity != "must_fix" for x in f if x.metric in NEVER_MUST_FIX)

    def test_never_must_fix_list_is_enforced_not_advisory(self):
        assert "pitch_range" in NEVER_MUST_FIX


class TestPacingSustainedRule:
    @staticmethod
    def _words(wpms: list[float], words_per_sentence: int = 6) -> list[Word]:
        """Build word timings that produce the given per-sentence WPMs."""
        out, t = [], 0.0
        for wpm in wpms:
            dur = words_per_sentence / wpm * 60.0
            step = dur / words_per_sentence
            for i in range(words_per_sentence):
                out.append(Word(word=f"w{i}", start=round(t, 3),
                                end=round(t + step * 0.8, 3), confidence=0.9))
                t += step
            t += 0.7          # inferred sentence boundary (>0.55s gap)
        return out

    def test_single_fast_sentence_is_not_flagged(self):
        """One quick line reads as energy, not as a pacing fault. Only
        SUSTAINED drift trips the rule."""
        w = self._words([155, 240, 155, 150])
        assert not [f for f in score(mk(words=w)).findings if f.metric == "pacing"]

    def test_sustained_fast_run_is_flagged(self):
        w = self._words([240, 240, 240, 240])
        f = [x for x in score(mk(words=w)).findings if x.metric == "pacing"]
        assert f, "sustained fast run should be flagged"

    def test_derived_band_overrides_the_global_band(self):
        """A Bengali take must be judged against its niche's derived band, not
        the US-English 140-170 norm."""
        w = self._words([230, 230, 230, 230])
        assert [x for x in score(mk(words=w)).findings if x.metric == "pacing"]
        # same take, band centred on 230 -> no longer a finding
        wide = score(mk(words=w), wpm_band=(205.0, 255.0)).findings
        assert not [x for x in wide if x.metric == "pacing"]


class TestPauseAlignment:
    def test_no_findings_without_a_script(self):
        """Without a script every pause is unaligned by construction. Reporting
        that would fabricate a finding on every scriptless take."""
        pauses = [Pause(t=float(i), duration=1.0, aligned_to_punctuation=False)
                  for i in range(5)]
        assert not [f for f in score(mk(pauses=pauses)).findings if f.metric == "pauses"]

    def test_flags_hesitation_when_a_script_was_used(self):
        pauses = [Pause(t=0.0, duration=1.0, aligned_to_punctuation=True)]
        pauses += [Pause(t=float(i), duration=1.0, aligned_to_punctuation=False)
                   for i in range(1, 6)]
        f = [x for x in score(mk(pauses=pauses)).findings if x.metric == "pauses"]
        assert f and f[0].severity == "must_fix"


class TestFeatureExtraction:
    def test_rates_are_per_minute_not_raw_counts(self):
        """A 3-minute take must not score worse than a 1-minute take purely for
        being longer."""
        peaks = [SibilancePeak(t=float(i), band_energy_db=-5.0) for i in range(30)]
        short = extract(mk(duration_sec=60.0, sibilance=Sibilance(peaks=peaks)))
        long_ = extract(mk(duration_sec=180.0,
                           sibilance=Sibilance(peaks=peaks * 3)))
        assert short.sibilance_per_min == pytest.approx(long_.sibilance_per_min, rel=0.01)

    def test_unaligned_ratio_is_none_without_a_script(self):
        pauses = [Pause(t=1.0, duration=1.0, aligned_to_punctuation=False)]
        assert extract(mk(pauses=pauses)).unaligned_pause_ratio is None


class TestBenchmark:
    @staticmethod
    def _ref(**over) -> ReferenceFeatures:
        f = dict(wpm_overall=160.0, pitch_mean_hz=130.0, pitch_range_hz=120.0,
                 spectral_tilt_mean=-5.0, sibilance_per_min=20.0,
                 pause_rate_per_min=14.0, mean_pause_sec=0.5)
        f.update(over)
        return ReferenceFeatures(
            name="ref", niche="n", source_path="x", language="en",
            duration_sec=60.0, features=FeatureVector(**f),
            provenance="youtube_rip",
        )

    def test_excluded_metrics_are_reported_with_reasons(self):
        """A metric that was never compared must not silently look like a pass."""
        b = benchmark(mk(), self._ref())
        assert "snr_db" in b.excluded and "pitch_mean_hz" in b.excluded
        assert all(m not in {d.metric for d in b.deltas} for m in b.excluded)

    def test_wider_pitch_range_than_reference_is_not_a_fault(self):
        """Direction matters: 'lower is worse' metrics must not flag a take for
        being MORE expressive than the reference."""
        take = extract(mk(pitch=Pitch(mean_hz=120.0, range_hz=300.0, variance=1.0)))
        d = next(x for x in benchmark(take, self._ref()).deltas
                 if x.metric == "pitch_range_hz")
        assert d.delta > 0 and d.severity == "ok"

    def test_narrower_pitch_range_does_register(self):
        take = extract(mk(pitch=Pitch(mean_hz=120.0, range_hz=10.0, variance=1.0)))
        d = next(x for x in benchmark(take, self._ref()).deltas
                 if x.metric == "pitch_range_hz")
        assert d.severity != "ok"

    def test_lossy_reference_gets_a_tilt_caveat(self):
        d = next(x for x in benchmark(mk(), self._ref()).deltas
                 if x.metric == "spectral_tilt_mean")
        assert d.note and "lossy" in d.note

    def test_cross_language_wpm_is_caveated(self):
        ref = self._ref()
        ref.language = "bn"
        d = next(x for x in benchmark(mk(), ref).deltas if x.metric == "wpm_overall")
        assert d.note and "languages" in d.note

    def test_no_composite_score_is_produced(self):
        """Deliberately per-metric. A single number lets good pacing cancel out
        a proximity problem - exactly the information the coach must keep apart."""
        b = benchmark(mk(), self._ref())
        assert not hasattr(b, "score") and not hasattr(b, "total")
        assert len(b.deltas) == len(COMPARISONS)


class TestNonAsciiOutput:
    """A Bengali transcript printed on a Windows console (cp1252) raised
    UnicodeEncodeError *after* the expensive ASR pass had finished - losing a
    real result to a display problem. The CLI now forces UTF-8 on stdout."""

    def test_force_utf8_is_safe_on_any_stream(self):
        from vo_coach.cli import _force_utf8_stdout

        _force_utf8_stdout()          # must not raise under pytest capture

    def test_bengali_text_survives_utf8_roundtrip(self):
        sample = "\u09ac\u09be\u0982\u09b2\u09be \u0995\u09a5\u09be"
        assert not sample.isascii()
        assert sample.encode("utf-8").decode("utf-8") == sample
        with pytest.raises(UnicodeEncodeError):
            sample.encode("cp1252")   # the exact failure the fix addresses


class TestUnreliableTimings:
    """Word timestamps are not stable in every language on the smaller Whisper
    checkpoints. The Bengali reference returns 33-94 words and 34-89 wpm across
    identical runs of the same clip, while English is bit-identical. Timing
    metrics built on that are meaningless, not merely noisy."""

    def test_pacing_and_pause_findings_are_suppressed(self):
        w = TestPacingSustainedRule._words([240, 240, 240, 240])
        pauses = [Pause(t=0.0, duration=1.0, aligned_to_punctuation=True)]
        pauses += [Pause(t=float(i), duration=1.0, aligned_to_punctuation=False)
                   for i in range(1, 6)]
        take = mk(words=w, pauses=pauses)

        assert [f for f in score(take).findings
                if f.metric in {"pacing", "pauses"}]
        suppressed = score(take, timings_reliable=False).findings
        assert not [f for f in suppressed if f.metric in {"pacing", "pauses"}]

    def test_signal_findings_still_run(self):
        """Tilt/sibilance/noise never touch ASR - suppressing timings must not
        disable them, or an unreliable-ASR niche would get no coaching at all."""
        take = mk(snr_db=10.0, mouth_clicks=[Click(t=1.0)])
        f = score(take, timings_reliable=False).findings
        assert {"noise", "clipping"} <= {x.metric for x in f}

    @staticmethod
    def _cfg_with(reliable: bool):
        """Build a Config declaring one unreliable niche.

        Deliberately not pointing at a real configured niche: this behaviour
        must hold for ANY language that turns out to have unstable timestamps,
        and the test should not break when the shipped niche list changes (it
        already did once, when the Bengali reference was replaced).
        """
        from pathlib import Path

        from vo_coach.config import Config, load_config

        raw = dict(load_config().raw)
        raw["niches"] = {"flaky": {"language": "xx", "asr_model": "small",
                                   "asr_word_timestamps_reliable": reliable}}
        return Config(raw, Path("<test>"))

    def test_benchmark_excludes_timing_metrics_for_that_niche(self):
        from vo_coach.scoring.benchmark import TIMING_METRICS

        ref = TestBenchmark._ref()
        ref.niche, ref.language = "flaky", "xx"
        b = benchmark(mk(), ref, cfg=self._cfg_with(reliable=False))

        assert not (TIMING_METRICS & {d.metric for d in b.deltas}), \
            "timing deltas reported despite unreliable word timestamps"
        for m in TIMING_METRICS:
            assert m in b.excluded and "unreliable" in b.excluded[m]

    def test_reliable_niche_still_reports_timing(self):
        ref = TestBenchmark._ref()
        ref.niche, ref.language = "flaky", "xx"
        b = benchmark(mk(), ref, cfg=self._cfg_with(reliable=True))
        assert "wpm_overall" in {d.metric for d in b.deltas}

    def test_defaults_to_reliable_for_an_unknown_niche(self):
        """An unconfigured niche must not silently lose its timing metrics."""
        ref = TestBenchmark._ref()
        ref.niche = "not_in_config"
        assert "wpm_overall" in {d.metric for d in benchmark(mk(), ref).deltas}
