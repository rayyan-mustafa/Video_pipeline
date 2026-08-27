"""Static debug plots - dev-tooling item 3.

Build this BEFORE the live HUD. A pause detector that is 200 ms late, or a WPM
curve shifted by half a window, is invisible in raw JSON and obvious the moment
it is drawn against the waveform. Every take should be plottable before any
mentor logic is layered on top.

    python -m vo_coach plot <take_id>
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import load_config
from ..schemas import AnalysisResult
from ..analysis.audio import load


def plot_take(result: AnalysisResult, out_path: str | Path | None = None,
              show_band: bool = True) -> Path:
    """Waveform + pauses + WPM curve + pitch contour + tilt, on a shared x axis.

    Sharing the time axis across all four panels is the point: it makes
    misalignment between a detected event and the audio it came from visible at
    a glance.
    """
    import matplotlib
    matplotlib.use("Agg")            # no display needed; write straight to PNG
    import matplotlib.pyplot as plt

    cfg = load_config()
    if not result.source_path:
        raise ValueError("AnalysisResult has no source_path to plot against.")

    audio = load(result.source_path, target_sr=cfg.get("audio.sample_rate"))
    t = np.arange(len(audio.samples)) / audio.sr

    fig, axes = plt.subplots(4, 1, figsize=(14, 9), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1.4, 1.4, 1.2]})
    fig.suptitle(f"{result.take_id}   {result.duration_sec:.2f}s   "
                 f"{result.pacing.wpm_overall:.0f} wpm", fontsize=12)

    # ---- 1. waveform, pauses, sibilance, clipping -------------------------
    ax = axes[0]
    # Decimate for drawing; full resolution is pointless at this width and slow.
    step = max(1, len(audio.samples) // 20000)
    ax.plot(t[::step], audio.samples[::step], lw=0.4, color="#4a6fa5")
    ax.set_ylabel("waveform")
    ax.set_ylim(-1.05, 1.05)

    for p in result.pauses:
        ax.axvspan(p.t, p.t + p.duration,
                   color=("#8fd694" if p.aligned_to_punctuation else "#e8a33d"),
                   alpha=0.35, lw=0)
    for pk in result.sibilance.peaks:
        ax.axvline(pk.t, color="#c0392b", lw=0.8, alpha=0.7)
    for c in result.mouth_clicks:
        ax.axvline(c.t, color="#000000", lw=1.2, ls=":")

    ax.legend(handles=[
        plt.Line2D([], [], color="#8fd694", lw=6, alpha=.5, label="pause on punctuation"),
        plt.Line2D([], [], color="#e8a33d", lw=6, alpha=.5, label="pause NOT on punctuation"),
        plt.Line2D([], [], color="#c0392b", lw=1, label="sibilance"),
        plt.Line2D([], [], color="#000", lw=1, ls=":", label="clipping"),
    ], loc="upper right", fontsize=7, ncol=4)

    # ---- 2. windowed WPM against the target band --------------------------
    ax = axes[1]
    if result.pacing.wpm_windowed:
        ax.plot([p.t for p in result.pacing.wpm_windowed],
                [p.wpm for p in result.pacing.wpm_windowed],
                color="#2c3e50", lw=1.4)
    if show_band:
        lo = cfg.get("pacing.target_wpm_min")
        hi = cfg.get("pacing.target_wpm_max")
        ax.axhspan(lo, hi, color="#8fd694", alpha=0.25, lw=0)
        ax.axhline(lo, color="#5aa05f", lw=0.6)
        ax.axhline(hi, color="#5aa05f", lw=0.6)
    ax.set_ylabel("wpm")

    # ---- 3. pitch contour -------------------------------------------------
    ax = axes[2]
    try:
        from ..analysis.prosody import F0_MAX, F0_MIN
        import parselmouth

        snd = parselmouth.Sound(audio.samples.astype(np.float64), audio.sr)
        p = snd.to_pitch(pitch_floor=F0_MIN, pitch_ceiling=F0_MAX)
        freq = p.selected_array["frequency"]
        times = p.xs()
        freq = np.where(freq > 0, freq, np.nan)   # gaps, not zeros
        ax.plot(times, freq, lw=1.0, color="#7d3c98")
    except Exception as e:
        ax.text(0.5, 0.5, f"pitch unavailable: {e}", ha="center",
                transform=ax.transAxes, fontsize=8)
    ax.set_ylabel("F0 (Hz)")

    # ---- 4. spectral tilt (proximity effect) ------------------------------
    ax = axes[3]
    tilt = result.proximity_effect.spectral_tilt_db_per_octave
    if tilt:
        ax.plot([p.t for p in tilt], [p.value for p in tilt],
                lw=1.2, color="#b9770e")
        ax.axhline(cfg.get("proximity_effect.silence_tilt"), color="#888",
                   lw=0.7, ls="--", label="silence threshold")
        ax.axhline(cfg.get("proximity_effect.must_fix_tilt"), color="#c0392b",
                   lw=0.7, ls="--", label="must-fix threshold")
        ax.legend(loc="lower right", fontsize=7)
    ax.set_ylabel("tilt dB/oct")
    ax.set_xlabel("seconds")

    fig.tight_layout()
    out = Path(out_path) if out_path else Path(f"{result.take_id}_debug.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out
