"""Scaffold tests - the contract, the config, and the fixtures.

These are deliberately cheap and dependency-light so they keep running even
while the heavy analysis deps are still being installed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import soundfile as sf

from vo_coach.config import load_config
from vo_coach.schemas import (
    AnalysisResult,
    Finding,
    Pacing,
    Pitch,
    ProximityEffect,
    ScoreResult,
    Sibilance,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _minimal_result(take_id: str = "t1") -> AnalysisResult:
    return AnalysisResult(
        take_id=take_id,
        duration_sec=6.0,
        transcript="hello there",
        pacing=Pacing(wpm_overall=155.0),
        pitch=Pitch(mean_hz=120.0, range_hz=60.0, variance=12.0),
        proximity_effect=ProximityEffect(),
        sibilance=Sibilance(),
        noise_floor_db=-52.0,
    )


class TestContract:
    def test_roundtrip_is_lossless(self):
        r = _minimal_result()
        again = AnalysisResult.model_validate_json(r.model_dump_json())
        assert again == r

    def test_rejects_bad_confidence(self):
        # confidence is bounded 0..1; a bad ASR adapter should fail loudly here
        # rather than silently poisoning downstream scoring.
        with pytest.raises(Exception):
            AnalysisResult.model_validate(
                {**json.loads(_minimal_result().model_dump_json()),
                 "words": [{"word": "x", "start": 0, "end": 1, "confidence": 4.2}]}
            )

    def test_must_fix_filter(self):
        s = ScoreResult(
            take_id="t1",
            findings=[
                Finding(metric="pacing", severity="must_fix", message="too fast"),
                Finding(metric="pitch", severity="stylistic", message="narrow"),
            ],
        )
        assert [f.metric for f in s.must_fix] == ["pacing"]


class TestConfig:
    def test_dotted_lookup(self):
        cfg = load_config()
        assert cfg.get("pacing.target_wpm_min") == 140
        assert cfg.get("audio.sample_rate") == 48000

    def test_missing_key_is_loud(self):
        # A silently-defaulted threshold is how a coach ends up mysteriously
        # never firing. Missing keys must raise.
        with pytest.raises(KeyError):
            load_config().get("pacing.nonexistent")

    def test_silence_threshold_is_below_must_fix(self):
        """The rubric only makes sense if silence < must_fix on every metric.
        Inverting them would make the coach flag everything or nothing."""
        cfg = load_config()
        assert cfg.get("pacing.silence_wpm_margin") < cfg.get("pacing.must_fix_wpm_margin")
        assert cfg.get("sibilance.silence_peak_db") < cfg.get("sibilance.must_fix_peak_db")
        # tilt and SNR are inverted scales: more negative / lower is worse
        assert cfg.get("proximity_effect.silence_tilt") > cfg.get("proximity_effect.must_fix_tilt")
        assert cfg.get("noise.silence_snr_db") > cfg.get("noise.must_fix_snr_db")


class TestFixtures:
    def test_manifest_exists(self):
        assert (FIXTURES / "manifest.json").exists(), \
            "run: python scripts/make_fixtures.py"

    def test_every_fixture_readable_and_labeled(self):
        manifest = json.loads((FIXTURES / "manifest.json").read_text())
        audio = {k: v for k, v in manifest.items() if not k.startswith("_")}
        assert audio, "no fixtures generated"
        for name, meta in audio.items():
            p = FIXTURES / meta["file"]
            assert p.exists(), f"{name} missing"
            info = sf.info(str(p))
            assert info.samplerate == meta["sample_rate"]
            assert info.frames > 0
            assert "defect" in meta, f"{name} has no defect label"

    def test_unsynthesizable_gap_is_recorded(self):
        """WPM/stress/pronunciation can't be synthesized. The manifest must keep
        that gap visible so it isn't mistaken for covered ground."""
        manifest = json.loads((FIXTURES / "manifest.json").read_text())
        assert "_needs_real_recordings" in manifest
        assert manifest["_needs_real_recordings"]["wanted"]
