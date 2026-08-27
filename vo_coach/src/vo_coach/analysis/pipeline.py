"""Phase 1 orchestrator: WAV + script -> AnalysisResult.

This is the only place that knows the order things run in. Each analysis module
stays independent and testable on its own; this file composes them and produces
the one JSON object the rest of the system consumes.
"""

from __future__ import annotations

import datetime as _dt

from ..config import load_config
from ..schemas import AnalysisResult, Click, ProximityEffect, Sibilance
from ..storage import db
from . import pacing as pacing_mod
from . import prosody, spectral
from .audio import Audio, load

# Bump whenever a change alters the NUMBERS an analysis produces.
#
# Stored analyses are reused indefinitely - scoring, ranking and the mentor all
# read them back rather than re-analyzing. Without a version, a fixed metric
# leaves wrong values sitting in the database forever, and nothing says so. That
# already happened: takes analyzed before the SNR digital-silence fix carried
# 175 dB SNR, and `rank` cheerfully sorted on it months of work later.
#
#   1  initial
#   2  SNR returns None on digital silence instead of a fabricated ~175 dB
#   3  sibilance measured as a band ratio with a speech-activity gate
#   4  ASR resamples to 16 kHz (word timestamps were 3x too long)
ANALYZER_VERSION = 4


def analyze(
    wav_path: str,
    take_id: str,
    script: str | None = None,
    model_size: str = "base.en",
    with_asr: bool = True,
    language: str = "en",
) -> AnalysisResult:
    """Run the full Phase 1 chain.

    `with_asr=False` runs only the signal-level metrics (tilt, sibilance, noise,
    clipping). That path exists because those metrics are exactly what the
    synthetic fixtures can validate - the fixtures contain no real speech, so
    forcing ASR on them would produce noise, not a test.
    """
    cfg = load_config()
    sr = cfg.get("audio.sample_rate")
    audio = load(wav_path, target_sr=sr)

    transcript, words = "", []
    if with_asr:
        from .asr import transcribe

        transcript, words = transcribe(audio, model_size=model_size,
                                       language=language)

    pacing = pacing_mod.analyze_pacing(words) if words else pacing_mod.Pacing(
        wpm_overall=0.0
    )
    pauses = pacing_mod.find_pauses(
        words, script=script,
        min_pause=cfg.get("pauses.unaligned_pause_sec", 0.25) / 2,
    ) if words else []

    noise_db, snr = spectral.noise_floor_and_snr(audio)
    band = (cfg.get("sibilance.band_hz_low"), cfg.get("sibilance.band_hz_high"))

    return AnalysisResult(
        take_id=take_id,
        duration_sec=round(audio.duration, 3),
        transcript=transcript,
        words=words,
        pacing=pacing,
        pauses=pauses,
        pitch=prosody.analyze_pitch(audio),
        proximity_effect=ProximityEffect(
            spectral_tilt_db_per_octave=spectral.spectral_tilt(audio)
        ),
        sibilance=Sibilance(
            peaks=spectral.sibilance_peaks(
                audio, band_hz=band,
                prominence_db=cfg.get("sibilance.silence_peak_db"),
            )
        ),
        noise_floor_db=noise_db,
        snr_db=snr,
        mouth_clicks=[Click(t=t) for t in spectral.clipping_times(audio)],
        source_path=str(wav_path),
        analyzer_version=ANALYZER_VERSION,
    )


def analyze_and_store(wav_path: str, take_id: str, script: str | None = None,
                      model_size: str = "base.en",
                      with_asr: bool = True) -> AnalysisResult:
    result = analyze(wav_path, take_id, script=script, model_size=model_size,
                     with_asr=with_asr)
    db.init_db()
    now = _dt.datetime.now().isoformat(timespec="seconds")

    # `analyses` has a FK onto `takes`, so a file analyzed straight from a path
    # (a fixture, or a wav dropped in by hand) needs its take row created first.
    # Registering it here rather than relaxing the constraint keeps every
    # analysis traceable back to an actual take.
    db.ensure_take(
        take_id=take_id,
        wav_path=str(wav_path),
        created_at=now,
        duration_sec=result.duration_sec,
        notes="auto-registered by analyze",
    )
    db.save_analysis(
        take_id=take_id,
        result_json=result.model_dump_json(),
        created_at=now,
    )
    return result


def summarize(r: AnalysisResult) -> str:
    """Human-readable one-screen summary for the CLI."""
    tilt = r.proximity_effect.spectral_tilt_db_per_octave
    mean_tilt = sum(p.value for p in tilt) / len(tilt) if tilt else 0.0
    unaligned = [p for p in r.pauses if not p.aligned_to_punctuation]

    lines = [
        f"take            {r.take_id}",
        f"duration        {r.duration_sec:.2f}s",
        f"words           {len(r.words)}",
        f"wpm (overall)   {r.pacing.wpm_overall:.1f}",
        f"pitch mean      {r.pitch.mean_hz:.1f} Hz   range {r.pitch.range_hz:.1f} Hz",
        f"spectral tilt   {mean_tilt:.2f} dB/oct (mean)",
        f"sibilant peaks  {len(r.sibilance.peaks)}",
        f"noise floor     {r.noise_floor_db:.1f} dB",
        f"SNR             {r.snr_db:.1f} dB" if r.snr_db is not None else "SNR  n/a",
        f"pauses          {len(r.pauses)} ({len(unaligned)} not on punctuation)",
    ]
    if r.mouth_clicks:
        lines.append(f"clipping        {len(r.mouth_clicks)} run(s)")
    if r.transcript:
        t = r.transcript[:110] + ("..." if len(r.transcript) > 110 else "")
        lines.append(f"transcript      {t}")
    return "\n".join("  " + ln for ln in lines)
