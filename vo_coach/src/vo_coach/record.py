"""Phase 0 - capture a take.

Records mono 48 kHz / 16-bit WAV (the format every later stage assumes) and
registers it in the take history.

Device selection matters here: the Fifine A6V shows up as a generic USB audio
device, and Windows will happily hand you the laptop's built-in mic instead if
you let it pick. `list_devices()` exists so the right one can be pinned in
advance rather than discovered after a ruined take.
"""

from __future__ import annotations

import datetime as _dt
import queue
import sys
from pathlib import Path

from .config import TAKES_DIR, ensure_dirs, load_config
from .storage import db


def list_devices() -> str:
    import sounddevice as sd

    return str(sd.query_devices())


def default_input() -> int | None:
    import sounddevice as sd

    try:
        return sd.default.device[0]
    except Exception:
        return None


def new_take_id() -> str:
    return _dt.datetime.now().strftime("take_%Y%m%d_%H%M%S")


def record(
    seconds: float | None = None,
    device: int | str | None = None,
    take_id: str | None = None,
    script_name: str | None = None,
) -> Path:
    """Record until `seconds` elapses, or until Ctrl+C if `seconds` is None.

    Writes incrementally to disk rather than buffering the whole take in RAM,
    so an interrupted long read still leaves a valid, playable WAV.
    """
    import sounddevice as sd
    import soundfile as sf

    cfg = load_config()
    sr = cfg.get("audio.sample_rate")
    channels = cfg.get("audio.channels")
    subtype = {16: "PCM_16", 24: "PCM_24", 32: "FLOAT"}[cfg.get("audio.bit_depth")]

    ensure_dirs()
    take_id = take_id or new_take_id()
    out = TAKES_DIR / f"{take_id}.wav"

    q: queue.Queue = queue.Queue()

    def _cb(indata, frames, time_info, status):
        if status:
            print(f"  [audio] {status}", file=sys.stderr)
        q.put(indata.copy())

    print(f"Recording -> {out}")
    print(f"  {sr} Hz, {channels}ch, {subtype}")
    print("  Ctrl+C to stop." if seconds is None else f"  {seconds:.1f}s")

    frames_written = 0
    with sf.SoundFile(out, mode="w", samplerate=sr, channels=channels,
                      subtype=subtype) as f:
        with sd.InputStream(samplerate=sr, channels=channels, device=device,
                            callback=_cb):
            try:
                target = int(seconds * sr) if seconds else None
                while target is None or frames_written < target:
                    block = q.get()
                    f.write(block)
                    frames_written += len(block)
            except KeyboardInterrupt:
                print("\n  stopped.")

    duration = frames_written / sr
    db.init_db()
    db.record_take(
        take_id=take_id,
        wav_path=str(out),
        created_at=_dt.datetime.now().isoformat(timespec="seconds"),
        duration_sec=duration,
        script_name=script_name,
    )
    print(f"Saved {duration:.2f}s as {take_id}")
    return out


def ingest(wav_path: str | Path, take_id: str | None = None,
           script_name: str | None = None) -> Path:
    """Register an already-recorded file as a take, without re-encoding it.

    The file is left where it is; only its path is stored. Useful for takes
    recorded in a DAW.
    """
    import soundfile as sf

    src = Path(wav_path).resolve()
    if not src.exists():
        raise FileNotFoundError(src)

    info = sf.info(str(src))
    take_id = take_id or new_take_id()

    db.init_db()
    db.record_take(
        take_id=take_id,
        wav_path=str(src),
        created_at=_dt.datetime.now().isoformat(timespec="seconds"),
        duration_sec=info.frames / float(info.samplerate),
        script_name=script_name,
        notes=f"ingested; {info.samplerate}Hz {info.channels}ch {info.subtype}",
    )
    print(f"Ingested {src.name} as {take_id} "
          f"({info.frames / info.samplerate:.2f}s, {info.samplerate}Hz, "
          f"{info.channels}ch)")
    if info.samplerate != load_config().get("audio.sample_rate"):
        print(f"  note: {info.samplerate}Hz differs from the configured "
              f"{load_config().get('audio.sample_rate')}Hz - analysis will resample.")
    return src
