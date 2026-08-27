"""Phase 4 tests: the mastering chain and take ranking.

The chain's contract is three guarantees, and every one of them was broken by a
plausible-looking implementation during development:

    loudness lands on target      broken by limiting once, then normalizing once
    peak never exceeds ceiling    broken by pedalboard's Limiter overshooting
    SNR does not get worse        broken by staging so hot the compressor
                                  squashed the signal and lifted the floor

So these assert on measured output, not on whether the code ran.
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from vo_coach.analysis.audio import Audio, load
from vo_coach.analysis.spectral import noise_floor_and_snr
from vo_coach.config import load_config
from vo_coach.mastering.chain import (
    deess,
    limit_true_peak,
    master,
    measure_lufs,
    normalize_loudness,
)
from vo_coach.mastering.ranking import rank
from vo_coach.schemas import Pacing, Pitch, ProximityEffect, Sibilance
from tests.test_scoring import mk

SR = 48000
FIXTURES = __import__("pathlib").Path(__file__).parent / "fixtures"


def speechlike(seconds: float = 3.0, seed: int = 0, peak: float = 0.5) -> Audio:
    """Voice-ish carrier: syllable envelope, energy around 100-4000 Hz."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    x = rng.standard_normal(n)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1 / SR)
    X *= np.exp(-((np.log10(np.maximum(f, 1)) - np.log10(700)) ** 2) / 0.25)
    x = np.fft.irfft(X, n)
    t = np.arange(n) / SR
    x *= 0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * t)
    x = x / (np.max(np.abs(x)) + 1e-9) * peak
    return Audio(samples=x.astype(np.float32), sr=SR)


class TestLimiter:
    """The limiter is written by hand because neither pedalboard effect works."""

    def test_never_boosts_a_quiet_signal(self):
        """pedalboard's Limiter lifts a -26 dBFS tone to -21.3 dBFS - 25 dB
        BELOW its own threshold. Inside the normalize->limit loop that boost
        compounds and the loop diverges upward instead of converging."""
        x = (0.05 * np.sin(2 * np.pi * 200 * np.arange(SR) / SR)).astype(np.float32)
        before = float(np.max(np.abs(x)))
        after = float(np.max(np.abs(limit_true_peak(x, SR, -1.0))))
        assert after == pytest.approx(before, rel=1e-6), "limiter applied makeup gain"

    @pytest.mark.parametrize("ceiling", [-1.0, -3.0, -0.1])
    def test_guarantees_the_ceiling(self, ceiling):
        x = (speechlike(peak=0.9).samples * 4.0).astype(np.float32)   # way over
        out = limit_true_peak(x, SR, ceiling)
        peak_db = 20 * np.log10(float(np.max(np.abs(out))))
        assert peak_db <= ceiling + 1e-4, f"peak {peak_db:.3f} exceeded {ceiling}"

    def test_catches_an_isolated_transient(self):
        """Without lookahead the first transient always escapes - the gain
        reduction arrives one sample too late."""
        x = np.full(SR, 0.1, dtype=np.float32)
        x[SR // 2] = 0.99
        out = limit_true_peak(x, SR, -6.0)
        assert 20 * np.log10(float(np.max(np.abs(out)))) <= -6.0 + 1e-4

    def test_leaves_an_under_ceiling_signal_untouched(self):
        x = (speechlike(peak=0.1).samples).astype(np.float32)
        assert np.allclose(limit_true_peak(x, SR, -1.0), x)


class TestLoudness:
    def test_reaches_target_within_tolerance(self):
        out, stages, warnings = master(speechlike(seconds=4.0), denoise=False)
        lufs = measure_lufs(out, SR)
        target = load_config().get("mastering.target_lufs")
        assert lufs is not None
        assert abs(lufs - target) < 1.0, f"landed {lufs:.1f}, target {target}"

    def test_never_exceeds_the_ceiling(self):
        out, _, _ = master(speechlike(seconds=4.0), denoise=False)
        ceiling = load_config().get("mastering.true_peak_ceiling_db")
        peak_db = 20 * np.log10(float(np.max(np.abs(out))))
        assert peak_db <= ceiling + 0.01

    def test_too_short_to_measure_is_skipped_not_silenced(self):
        """EBU R128 needs >=400 ms. pyloudnorm returns -inf below that, and
        normalizing to -inf would silence the take."""
        short = speechlike(seconds=0.2)
        assert measure_lufs(short.samples, SR) is None
        out, _, warnings = master(short, denoise=False)
        assert float(np.max(np.abs(out))) > 0.0, "short take was silenced"
        assert any("too short" in w for w in warnings)

    def test_normalize_reports_none_for_unmeasurable(self):
        x = np.zeros(int(0.2 * SR), dtype=np.float32)
        _, before = normalize_loudness(x, SR, -16.0)
        assert before is None


class TestSNRNotDegraded:
    """The chain must not make the recording noisier. Staging by LUFS drove
    peaks to +11.5 dBFS, the compressor squashed 7 dB of RMS, and SNR fell from
    14.4 to 9.6 dB - a 'successful' run that made the take worse."""

    @pytest.mark.parametrize("name", ["clean_baseline", "noisy_snr12"])
    def test_snr_does_not_get_worse(self, name):
        audio = load(FIXTURES / f"{name}.wav", target_sr=SR)
        _, snr_before = noise_floor_and_snr(audio)
        out, _, _ = master(audio, denoise=False)
        _, snr_after = noise_floor_and_snr(Audio(samples=out, sr=SR))
        assert snr_before is not None and snr_after is not None
        assert snr_after > snr_before - 1.5, \
            f"SNR fell {snr_before:.1f} -> {snr_after:.1f} dB"


class TestDeesser:
    def test_reduces_sibilant_band_only(self):
        """A high-shelf cut would dull every consonant. A de-esser must be
        dynamic - active on the sibilant band, transparent below it."""
        n = SR
        t = np.arange(n) / SR
        low = 0.3 * np.sin(2 * np.pi * 300 * t)
        sib = 0.6 * np.sin(2 * np.pi * 7000 * t)
        x = (low + sib).astype(np.float32)

        out = deess(x, SR, freq_hz=6500.0, threshold_db=-30.0, ratio=6.0)

        def band(sig, lo, hi):
            X = np.abs(np.fft.rfft(sig))
            f = np.fft.rfftfreq(len(sig), 1 / SR)
            m = (f >= lo) & (f <= hi)
            return float(np.sqrt(np.mean(X[m] ** 2)))

        assert band(out, 6000, 8000) < band(x, 6000, 8000) * 0.9, "sibilant band untouched"
        assert band(out, 200, 400) > band(x, 200, 400) * 0.7, "low band was dulled"

    def test_passthrough_above_nyquist(self):
        x = speechlike(seconds=0.5).samples
        assert np.allclose(deess(x, SR, freq_hz=SR / 2), x)


class TestStageReports:
    def test_every_stage_is_reported_including_skipped(self):
        """A stage that silently did nothing is indistinguishable from one that
        ran, unless it says so."""
        _, stages, _ = master(speechlike(seconds=3.0), denoise=False, deesser=False)
        names = {s.stage for s in stages}
        assert {"input", "noise_reduction", "deesser", "loudness"} <= names
        skipped = {s.stage for s in stages if not s.applied}
        assert {"noise_reduction", "deesser"} <= skipped


class TestRanking:
    @staticmethod
    def _take(take_id, snr=35.0, clipped=False):
        from vo_coach.schemas import Click
        return mk(take_id=take_id, snr_db=snr,
                  mouth_clicks=[Click(t=1.0)] if clipped else [])

    def test_clipped_take_loses_to_clean_one(self):
        """Clipping is unrepairable, so it outranks every other consideration -
        even a much better SNR."""
        r = rank([self._take("clipped", snr=60.0, clipped=True),
                  self._take("clean", snr=25.0)])
        assert r.winner.take_id == "clean"

    def test_snr_breaks_ties(self):
        r = rank([self._take("quiet", snr=22.0), self._take("clean", snr=40.0)])
        assert r.winner.take_id == "clean"

    def test_unmeasurable_snr_does_not_win_the_tiebreak(self):
        """None must not sort as 'infinitely good'."""
        r = rank([self._take("unknown", snr=None), self._take("measured", snr=33.0)])
        assert r.winner.take_id == "measured"

    def test_every_candidate_carries_a_reason(self):
        r = rank([self._take("a"), self._take("b", snr=20.0)])
        assert all(c.reason for c in r.candidates)
        assert [c.rank for c in r.candidates] == [1, 2]

    def test_empty_input_has_no_winner(self):
        assert rank([]).winner is None
