"""Phase 1 analysis tests, validated against the labeled fixtures.

Each test asserts against the defect the fixture was BUILT with, so a failure
means the analysis module is wrong - not that someone's voice was unusual.

Tests needing the heavy native deps are skipped (not failed) when those aren't
installed yet, so the suite stays runnable mid-setup.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vo_coach.analysis.audio import Audio, db, is_clipped, load
from vo_coach.analysis import spectral
from vo_coach.analysis.pacing import (
    find_pauses,
    sentence_wpm,
    wpm_overall,
    wpm_windowed,
)
from vo_coach.schemas import Word

FIXTURES = Path(__file__).parent / "fixtures"
MANIFEST = json.loads((FIXTURES / "manifest.json").read_text())


def fx(name: str) -> Audio:
    return load(FIXTURES / MANIFEST[name]["file"])


# ---------------------------------------------------------------------------
#  Pure-signal metrics - no ASR, fully covered by synthetic fixtures
# ---------------------------------------------------------------------------
class TestNoiseAndSNR:
    def test_measured_snr_matches_constructed_snr(self):
        """noisy_snr12.wav was built at a known 12 dB SNR."""
        meta = MANIFEST["noisy_snr12"]
        _floor, snr = spectral.noise_floor_and_snr(fx("noisy_snr12"))
        assert abs(snr - meta["expect_snr_db"]) <= meta["tolerance_db"], \
            f"measured {snr:.1f} dB, fixture built at {meta['expect_snr_db']} dB"

    def test_clean_baseline_has_better_snr_than_noisy(self):
        _, clean_snr = spectral.noise_floor_and_snr(fx("clean_baseline"))
        _, noisy_snr = spectral.noise_floor_and_snr(fx("noisy_snr12"))
        assert clean_snr is not None and noisy_snr is not None
        assert clean_snr > noisy_snr

    def test_digital_silence_returns_none_not_a_fake_snr(self):
        """A gated/edited file has true zero samples. Taking a percentile over
        those drives the floor to -200 dB and yields ~175 dB SNR - a broken
        measurement that a rubric would read as a flawless take. None is the
        honest answer when there is no room tone to characterize."""
        sr = 48000
        x = np.zeros(sr, dtype=np.float32)
        x[: sr // 2] = 0.3 * np.sin(2 * np.pi * 200 * np.arange(sr // 2) / sr)
        _floor, snr = spectral.noise_floor_and_snr(Audio(samples=x, sr=sr))
        assert snr is None or snr < 100.0, \
            f"fabricated an SNR of {snr} dB from digital silence"


class TestProximityEffect:
    def test_heavy_proximity_tilts_more_negative_than_baseline(self):
        """The fixture adds a low-shelf boost; tilt must move downward."""
        def mean_tilt(name: str) -> float:
            pts = spectral.spectral_tilt(fx(name))
            assert pts, f"no tilt points for {name}"
            return float(np.mean([p.value for p in pts]))

        assert mean_tilt("heavy_proximity") < mean_tilt("clean_baseline"), \
            "low-shelf boost did not register as a more negative tilt"

    def test_tilt_points_are_time_ordered(self):
        pts = spectral.spectral_tilt(fx("clean_baseline"))
        assert all(b.t > a.t for a, b in zip(pts, pts[1:]))


class TestSibilance:
    def test_finds_the_four_constructed_bursts(self):
        """harsh_sibilance.wav has bursts at 1.0/2.5/4.0/5.5s."""
        expected = MANIFEST["harsh_sibilance"]["expect_peaks_at"]
        peaks = spectral.sibilance_peaks(fx("harsh_sibilance"))
        found = [p.t for p in peaks]

        for want in expected:
            assert any(abs(f - want) < 0.20 for f in found), \
                f"no sibilance peak near {want}s (found {[round(f,2) for f in found]})"

    def test_finds_no_extra_peaks(self):
        """Count must match the construction, not merely exceed the baseline.

        The earlier version of this test asserted only `clean < harsh`. It
        passed at 19 vs 20 - the detector was firing on essentially every frame
        of broadband noise and the test called that a success. Asserting the
        exact expected count is what caught it.
        """
        expected = MANIFEST["harsh_sibilance"]["expect_peaks_at"]
        peaks = spectral.sibilance_peaks(fx("harsh_sibilance"))
        assert len(peaks) == len(expected), \
            f"expected {len(expected)} peaks, got {len(peaks)}: " \
            f"{[round(p.t, 2) for p in peaks]}"

    def test_clean_baseline_has_no_peaks(self):
        """A fixture built with no sibilance must report none - not 'fewer'."""
        assert spectral.sibilance_peaks(fx("clean_baseline")) == []

    def test_hiss_is_not_reported_as_sibilance(self):
        """Broadband noise is HF-weighted, so a ratio-only detector flags a
        hissy take as sibilant everywhere. Hiss is a mastering problem, not a
        delivery note, and must not surface as sibilance."""
        assert spectral.sibilance_peaks(fx("noisy_snr12")) == []


class TestClipping:
    def test_detects_constructed_clipping(self):
        assert spectral.clipping_times(fx("clipped")), "clipping not detected"

    def test_clean_baseline_is_not_clipped(self):
        assert not spectral.clipping_times(fx("clean_baseline"))

    def test_single_full_scale_sample_is_not_clipping(self):
        """One sample at ceiling is not clipping; a run is. Guards against a
        detector that fires on any peak-normalized file."""
        x = np.zeros(1000, dtype=np.float32)
        x[500] = 1.0
        assert is_clipped(x) == []


# ---------------------------------------------------------------------------
#  Pacing - synthetic Word lists, no audio needed
# ---------------------------------------------------------------------------
def mkwords(spec: list[tuple[str, float, float]]) -> list[Word]:
    return [Word(word=w, start=s, end=e, confidence=0.9) for w, s, e in spec]


class TestPacing:
    def test_wpm_uses_speaking_time_not_file_duration(self):
        """Six words over exactly 2s of speech = 180 wpm, regardless of how
        much room tone pads the file."""
        words = mkwords([(f"w{i}", 10.0 + i * 0.4, 10.0 + i * 0.4 + 0.3)
                         for i in range(6)])
        # span = last.end - first.start = 2.3 - 0.0 -> 6/2.3*60
        assert wpm_overall(words) == pytest.approx(6 / 2.3 * 60, rel=0.01)

    def test_windowed_wpm_uses_word_midpoints(self):
        """A word straddling a window edge counts toward the window holding its
        midpoint. Using start times biases the whole curve early."""
        words = mkwords([("a", 0.0, 1.0), ("b", 4.9, 5.2), ("c", 9.0, 9.2)])
        pts = wpm_windowed(words, win_sec=5.0, hop_sec=5.0)
        # 'b' midpoint is 5.05 -> belongs to window [5,10), not [0,5)
        first = next(p for p in pts if abs(p.t - 2.5) < 1e-6)
        assert first.wpm == pytest.approx(12.0)   # only 'a' -> 1/5*60

    def test_empty_input_is_zero_not_crash(self):
        assert wpm_overall([]) == 0.0
        assert wpm_windowed([]) == []


class TestPauses:
    def test_pause_on_comma_is_marked_aligned(self):
        script = "hello there, friend"
        words = mkwords([("hello", 0.0, 0.4), ("there", 0.4, 0.9),
                         ("friend", 1.8, 2.3)])
        pauses = find_pauses(words, script=script, min_pause=0.25)
        assert len(pauses) == 1
        assert pauses[0].aligned_to_punctuation is True

    def test_pause_mid_clause_is_marked_unaligned(self):
        """Same gap, no punctuation -> hesitation, not phrasing."""
        script = "hello there friend"
        words = mkwords([("hello", 0.0, 0.4), ("there", 0.4, 0.9),
                         ("friend", 1.8, 2.3)])
        pauses = find_pauses(words, script=script, min_pause=0.25)
        assert len(pauses) == 1
        assert pauses[0].aligned_to_punctuation is False

    def test_asr_insertion_does_not_desync_alignment(self):
        """An extra ASR word must not shift punctuation labels for the rest of
        the take - index-based pairing would break here."""
        script = "one two, three four, five"
        words = mkwords([
            ("one", 0.0, 0.3),
            ("uh", 0.3, 0.5),          # ASR insertion not in the script
            ("two", 0.5, 0.8),
            ("three", 1.6, 1.9),       # pause after "two," -> aligned
            ("four", 1.9, 2.2),
            ("five", 3.0, 3.3),        # pause after "four," -> aligned
        ])
        pauses = find_pauses(words, script=script, min_pause=0.25)
        assert len(pauses) == 2
        assert all(p.aligned_to_punctuation for p in pauses), \
            "alignment desynced after the inserted word"

    def test_without_script_alignment_is_false_not_guessed(self):
        words = mkwords([("a", 0.0, 0.3), ("b", 1.5, 1.8)])
        pauses = find_pauses(words, script=None, min_pause=0.25)
        assert pauses and pauses[0].aligned_to_punctuation is False

    def test_sentence_split_on_script_punctuation(self):
        script = "one two. three four five."
        words = mkwords([("one", 0, .3), ("two", .3, .6),
                         ("three", 1.0, 1.3), ("four", 1.3, 1.6),
                         ("five", 1.6, 1.9)])
        assert len(sentence_wpm(words, script=script)) == 2


# ---------------------------------------------------------------------------
#  Modules needing native deps
# ---------------------------------------------------------------------------
class TestProsody:
    def test_pitch_of_a_pure_tone(self):
        """A 150 Hz tone must read as ~150 Hz. Catches octave errors, which are
        the classic pitch-tracker failure."""
        pytest.importorskip("parselmouth")
        from vo_coach.analysis.prosody import analyze_pitch

        sr = 48000
        t = np.arange(sr * 2) / sr
        tone = (0.5 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
        p = analyze_pitch(Audio(samples=tone, sr=sr))
        assert p.mean_hz == pytest.approx(150.0, rel=0.05), \
            f"got {p.mean_hz} Hz - likely an octave error"

    def test_silence_returns_zeros_not_nan(self):
        pytest.importorskip("parselmouth")
        from vo_coach.analysis.prosody import analyze_pitch

        p = analyze_pitch(Audio(samples=np.zeros(48000, np.float32), sr=48000))
        assert p.mean_hz == 0.0 and not np.isnan(p.variance)


class TestHelpers:
    def test_db_floors_instead_of_returning_inf(self):
        """-inf would propagate NaN through every downstream comparison."""
        assert db(0.0) <= -200.0 or db(0.0) == pytest.approx(-200.0, abs=1e6)
        assert np.isfinite(db(0.0))
