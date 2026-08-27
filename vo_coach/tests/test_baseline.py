"""Personal baseline (calibration).

The point of these is that the baseline can only ever LOOSEN the rubric toward
the speaker's own natural range, never tighten it into a new set of absolutes -
and that it refuses to be built from a bad recording, which would make the
fault the norm.
"""

from __future__ import annotations

import pytest

from vo_coach.schemas import Click, Pitch, Word, WpmPoint
from vo_coach.scoring import baseline as bl
from tests.test_scoring import mk
from tests.test_providers import cfg_with as _mentor_cfg  # noqa: F401


def cfg_with(**over):
    """A Config with baseline overrides, without touching the real YAML."""
    from pathlib import Path

    from vo_coach.config import Config, load_config

    raw = dict(load_config().raw)
    raw["baseline"] = {**raw["baseline"], **over}
    return Config(raw, Path("<test>"))


def take(tid="t", wpm=150.0, conf=0.85, n_words=80, dur=60.0, **over):
    """An analyzed take suitable for calibration unless a test breaks it."""
    words = [Word(word="w", start=float(i) * 0.3, end=float(i) * 0.3 + 0.25,
                  confidence=conf) for i in range(n_words)]
    base = dict(take_id=tid, duration_sec=dur, words=words,
                analyzer_version=4)
    base.update(over)
    r = mk(**base)
    r.pacing.wpm_overall = wpm
    r.pacing.wpm_windowed = [WpmPoint(t=float(i), wpm=wpm + d)
                             for i, d in enumerate((-20, -10, 0, 10, 20))]
    return r


class TestSuitability:
    def test_a_clipped_take_is_refused(self):
        """A clipped take describes a gain mistake, not a voice. Calibrating on
        it would make clipping part of 'normal'."""
        t = take(mouth_clicks=[Click(t=1.0)])
        with pytest.raises(bl.CalibrationError, match="clipping"):
            bl.build([t])

    def test_a_hissy_take_is_refused(self):
        with pytest.raises(bl.CalibrationError, match="SNR"):
            bl.build([take(snr_db=12.0)])

    def test_a_short_take_is_refused(self):
        with pytest.raises(bl.CalibrationError, match="at least"):
            bl.build([take(dur=8.0)])

    def test_a_sparse_take_is_refused(self):
        with pytest.raises(bl.CalibrationError, match="words"):
            bl.build([take(n_words=10)])

    def test_mixed_analyzer_versions_are_refused(self):
        """Two takes measured by different analyzers are not comparable, and a
        median across them is a number with no meaning."""
        with pytest.raises(bl.CalibrationError, match="analyzer version"):
            bl.build([take("a"), take("b", analyzer_version=3)])


class TestBuild:
    def test_median_not_mean(self):
        """One odd take must not drag the baseline. 150/155/400 has a median of
        155; a mean would say 235."""
        b = bl.build([take("a", wpm=150), take("b", wpm=155), take("c", wpm=400)])
        assert b.wpm.median == 155.0

    def test_mad_records_the_spread(self):
        b = bl.build([take("a", wpm=140), take("b", wpm=150), take("c", wpm=160)])
        assert b.wpm.mad == 10.0

    def test_word_confidence_is_medianed_within_a_take_first(self):
        """Otherwise a take with three times the words would dominate the
        speaker's baseline just for being longer."""
        b = bl.build([take("a", conf=0.9, n_words=300),
                      take("b", conf=0.6, n_words=80),
                      take("c", conf=0.6, n_words=80)])
        assert b.word_confidence.median == pytest.approx(0.6)

    def test_take_ids_are_recorded(self):
        b = bl.build([take("a"), take("b")])
        assert b.take_ids == ["a", "b"]


class TestWpmBand:
    def test_no_baseline_leaves_the_band_untouched(self):
        """Everything must work before calibration exists."""
        assert bl.wpm_band(cfg=cfg_with(enabled=False), band=(140.0, 170.0),
                           b=None) == (140.0, 170.0)

    def test_band_moves_toward_the_speaker(self):
        """A speaker who naturally lives at 110-135 should not be permanently
        told they are dragging against a US-VO band."""
        b = bl.build([take("a", wpm=120), take("b", wpm=122), take("c", wpm=124)])
        lo, hi = bl.wpm_band(cfg=cfg_with(enabled=True, weight=0.5),
                             band=(140.0, 170.0), b=b)
        assert lo < 140.0 and hi < 170.0

    def test_weight_zero_is_the_pure_rubric(self):
        b = bl.build([take("a", wpm=120)])
        assert bl.wpm_band(cfg=cfg_with(enabled=True, weight=0.0),
                           band=(140.0, 170.0), b=b) == (140.0, 170.0)

    def test_weight_is_clamped(self):
        """A typo of 5 instead of 0.5 must not extrapolate the band into
        nonsense."""
        b = bl.build([take("a", wpm=120)])
        lo, hi = bl.wpm_band(cfg=cfg_with(enabled=True, weight=5.0),
                             band=(140.0, 170.0), b=b)
        assert lo <= hi
        assert 80.0 <= lo <= 170.0

    def test_band_is_never_inverted(self):
        b = bl.build([take("a", wpm=300)])
        lo, hi = bl.wpm_band(cfg=cfg_with(enabled=True, weight=0.9),
                             band=(140.0, 170.0), b=b)
        assert lo <= hi


class TestMinWordConfidence:
    def test_no_baseline_returns_the_absolute_threshold(self):
        from vo_coach.config import load_config

        expected = load_config().get("pronunciation.min_word_confidence")
        assert bl.min_word_confidence(cfg=cfg_with(enabled=False), b=None) == expected

    def test_calibration_can_only_loosen_never_tighten(self):
        """THE SHARPEST EDGE. A speaker with high ASR confidence must not end up
        with a STRICTER mispronunciation threshold than the rubric's - that
        would punish them for being easy to transcribe."""
        from vo_coach.config import load_config

        absolute = load_config().get("pronunciation.min_word_confidence")
        b = bl.build([take("a", conf=0.98), take("b", conf=0.97)])
        assert bl.min_word_confidence(cfg=cfg_with(enabled=True), b=b) <= absolute

    def test_a_lower_confidence_speaker_gets_a_lower_floor(self):
        """Whisper is less certain on accents further from its training data.
        Without this the flag fires on the accent, not the pronunciation."""
        from vo_coach.config import load_config

        absolute = load_config().get("pronunciation.min_word_confidence")
        b = bl.build([take("a", conf=0.55), take("b", conf=0.55)])
        assert bl.min_word_confidence(cfg=cfg_with(enabled=True, weight=0.5),
                                      b=b) < absolute

    def test_never_goes_negative(self):
        b = bl.build([take("a", conf=0.02), take("b", conf=0.02)])
        assert bl.min_word_confidence(cfg=cfg_with(enabled=True, weight=1.0),
                                      b=b) >= 0.0


class TestPersistence:
    def test_roundtrip(self, tmp_path):
        cfg = cfg_with(enabled=True, path=str(tmp_path / "b.json"))
        b = bl.build([take("a", wpm=133)])
        bl.save(b, cfg)
        assert bl.load(cfg).wpm.median == 133.0

    def test_disabled_config_loads_nothing(self, tmp_path):
        """The file existing is not consent to use it."""
        cfg_on = cfg_with(enabled=True, path=str(tmp_path / "b.json"))
        bl.save(bl.build([take("a")]), cfg_on)
        cfg_off = cfg_with(enabled=False, path=str(tmp_path / "b.json"))
        assert bl.load(cfg_off) is None
        assert bl.load(cfg_off, require_enabled=False) is not None

    def test_corrupt_baseline_is_a_miss_not_a_crash(self, tmp_path):
        p = tmp_path / "b.json"
        p.write_text("{not json", encoding="utf-8")
        assert bl.load(cfg_with(enabled=True, path=str(p))) is None

    def test_missing_file_is_a_miss(self, tmp_path):
        assert bl.load(cfg_with(enabled=True, path=str(tmp_path / "no.json"))) is None


class TestRender:
    def test_no_baseline_says_how_to_make_one(self):
        assert "calibrate" in bl.render(None)

    def test_render_includes_the_take_count(self):
        assert "2 take(s)" in bl.render(bl.build([take("a"), take("b")]))
