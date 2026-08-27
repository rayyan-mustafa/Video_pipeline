"""Phase 4 orchestrator: take -> mastered file + before/after report."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from ..analysis.audio import Audio, db, load
from ..analysis.spectral import noise_floor_and_snr
from ..config import Config, TAKES_DIR, load_config
from ..schemas import MasteringReport
from .chain import master, measure_lufs


def _peak_db(x: np.ndarray) -> float:
    return db(float(np.max(np.abs(x)))) if x.size else -100.0


def master_take(
    wav_path: str | Path,
    take_id: str,
    out_path: str | Path | None = None,
    cfg: Config | None = None,
    denoise: bool = True,
    deesser: bool = True,
) -> MasteringReport:
    cfg = cfg or load_config()
    sr_target = cfg.get("audio.sample_rate")
    audio = load(wav_path, target_sr=sr_target)

    lufs_before = measure_lufs(audio.samples, audio.sr)
    peak_before = _peak_db(audio.samples)
    # SNR, not the absolute noise floor. The chain changes overall level by
    # ~12 dB, which moves the absolute floor by the same amount and makes a
    # successful denoise read as though it got worse. SNR is level-independent,
    # so it actually answers "is there less hiss relative to the voice".
    floor_before, snr_before = noise_floor_and_snr(audio)

    processed, stages, warnings = master(audio, cfg=cfg,
                                         denoise=denoise, deesser=deesser)

    out = Path(out_path) if out_path else TAKES_DIR / f"{take_id}_mastered.wav"
    out.parent.mkdir(parents=True, exist_ok=True)

    # 24-bit rather than 16: this is a delivery master that may still be edited
    # into a video, and the extra headroom costs nothing at these file sizes.
    sf.write(out, processed, audio.sr, subtype="PCM_24")

    lufs_after = measure_lufs(processed, audio.sr)
    peak_after = _peak_db(processed)
    floor_after, snr_after = noise_floor_and_snr(Audio(samples=processed, sr=audio.sr))

    target = cfg.get("mastering.target_lufs")
    ceiling = cfg.get("mastering.true_peak_ceiling_db")

    # Verify the file actually hits its targets rather than assuming the chain
    # worked. A limiter can leave loudness short, and a report that states the
    # target instead of the measurement is worse than no report.
    if lufs_after is not None and abs(lufs_after - target) > 1.0:
        warnings.append(
            f"output is {lufs_after:.1f} LUFS, {abs(lufs_after - target):.1f} dB "
            f"off the {target:.1f} target - the true-peak limiter likely pulled "
            f"it down. Lower the compressor threshold or accept the offset."
        )
    if peak_after > ceiling + 0.1:
        warnings.append(
            f"peak is {peak_after:.2f} dBFS, above the {ceiling} dB ceiling."
        )

    return MasteringReport(
        take_id=take_id,
        source_path=str(wav_path),
        output_path=str(out),
        lufs_before=round(lufs_before, 2) if lufs_before is not None else float("nan"),
        lufs_after=round(lufs_after, 2) if lufs_after is not None else float("nan"),
        target_lufs=target,
        peak_db_before=round(peak_before, 2),
        peak_db_after=round(peak_after, 2),
        true_peak_ceiling_db=ceiling,
        noise_floor_before=round(snr_before, 2) if snr_before is not None else None,
        noise_floor_after=round(snr_after, 2) if snr_after is not None else None,
        stages=stages,
        warnings=warnings,
    )


def render(r: MasteringReport) -> str:
    lines = [
        f"  {r.take_id}  ->  {r.output_path}",
        "",
        f"  {'':22} {'before':>10} {'after':>10}",
        f"  {'loudness (LUFS)':22} {r.lufs_before:>10.1f} {r.lufs_after:>10.1f}"
        f"   (target {r.target_lufs})",
        f"  {'peak (dBFS)':22} {r.peak_db_before:>10.2f} {r.peak_db_after:>10.2f}"
        f"   (ceiling {r.true_peak_ceiling_db})",
    ]
    if r.noise_floor_before is not None and r.noise_floor_after is not None:
        delta = r.noise_floor_after - r.noise_floor_before
        lines.append(f"  {'SNR (dB)':22} {r.noise_floor_before:>10.1f} "
                     f"{r.noise_floor_after:>10.1f}   ({delta:+.1f}, higher is better)")

    lines += ["", "  chain:"]
    for s in r.stages:
        mark = " " if s.applied else "-"
        detail = f"  {s.detail}" if s.detail else ""
        lines.append(f"   {mark} {s.stage:<20} peak {s.peak_db:>7.2f}  "
                     f"rms {s.rms_db:>7.2f}{detail}")

    if r.warnings:
        lines += ["", "  warnings:"]
        lines += [f"    - {w}" for w in r.warnings]
    return "\n".join(lines)
