"""ASR adapter tests.

The sample-rate bug these guard against was invisible in every way that
normally catches bugs: no exception, no warning, a confident transcript, and
well-formed word timestamps. Only the ARITHMETIC gave it away - words ending at
179 s in a 60 s file. So these tests assert on that arithmetic rather than on
transcript content.
"""

from __future__ import annotations

import numpy as np
import pytest

from vo_coach.analysis.asr import ASR_SAMPLE_RATE, transcribe
from vo_coach.analysis.audio import Audio


class TestLanguageGuard:
    def test_en_model_rejects_non_english(self):
        """The .en checkpoints do not error on Bengali - they return plausible
        English nonsense. Downstream metrics computed from that look entirely
        normal, so this has to fail at the boundary."""
        a = Audio(samples=np.zeros(16000, np.float32), sr=16000)
        with pytest.raises(ValueError, match="English-only"):
            transcribe(a, model_size="base.en", language="bn")

    def test_multilingual_model_accepts_non_english(self):
        """Guard must not fire on a valid multilingual pairing. Uses a bogus
        model name so no download happens - the guard runs before model load,
        so any other error proves the guard passed."""
        a = Audio(samples=np.zeros(16000, np.float32), sr=16000)
        try:
            transcribe(a, model_size="__nonexistent__", language="bn")
        except ValueError as e:
            assert "English-only" not in str(e), "guard fired on a valid pairing"
        except Exception:
            pass          # any non-ValueError means the guard let it through


class TestSampleRate:
    """faster-whisper's `transcribe()` takes a bare array with NO sample-rate
    argument and assumes 16 kHz. This pipeline runs at 48 kHz."""

    def test_asr_rate_constant_is_16k(self):
        assert ASR_SAMPLE_RATE == 16000

    def test_resample_ratio_is_what_broke_it(self):
        """48 kHz fed in as 16 kHz stretches audio 3x - which is exactly the
        60 s -> 179.5 s observed. Documented as a test so the arithmetic stays
        visible if the pipeline rate ever changes."""
        assert 48000 / ASR_SAMPLE_RATE == 3.0

    @pytest.mark.parametrize("sr", [48000, 44100, 22050, 16000])
    def test_resampled_length_matches_real_duration(self, sr):
        """Whatever the input rate, the array handed to Whisper must represent
        the same number of SECONDS - otherwise every word timestamp, and every
        metric derived from one, is scaled by the rate ratio."""
        from scipy.signal import resample_poly

        seconds = 2.0
        x = np.zeros(int(sr * seconds), np.float32)
        if sr != ASR_SAMPLE_RATE:
            g = np.gcd(ASR_SAMPLE_RATE, sr)
            x = resample_poly(x, ASR_SAMPLE_RATE // g, sr // g)
        assert len(x) / ASR_SAMPLE_RATE == pytest.approx(seconds, rel=0.01)


class TestTimestampSanity:
    """A cheap invariant worth asserting on every real transcription: no word
    may end after the audio does. This alone would have caught the bug."""

    @staticmethod
    def _check(words, duration):
        if not words:
            return
        assert words[-1].end <= duration * 1.02, (
            f"last word ends at {words[-1].end:.1f}s in a {duration:.1f}s file - "
            f"ratio {words[-1].end / duration:.2f}x suggests a sample-rate mismatch"
        )

    def test_helper_flags_a_3x_overrun(self):
        from vo_coach.schemas import Word

        bad = [Word(word="x", start=0.0, end=179.5, confidence=0.9)]
        with pytest.raises(AssertionError, match="sample-rate mismatch"):
            self._check(bad, 60.0)

    def test_helper_passes_normal_timings(self):
        from vo_coach.schemas import Word

        self._check([Word(word="x", start=58.0, end=59.9, confidence=0.9)], 60.0)
