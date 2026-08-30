"""Does a remote worker produce the same words as a local model?

Starts a worker in-process that speaks exactly the contract
`studio/colab_worker.py` implements, points `RemoteWhisper` at it, and compares
the result against `LocalWhisper` on the same audio. If these two ever disagree,
the seam is lying and every timestamp downstream is suspect.

Needs no GPU and no Colab - it runs `base` on both sides.

    ..\\vo_coach\\.venv\\Scripts\\python.exe test_transcribe_seam.py
"""

# NO `from __future__ import annotations` HERE, deliberately. It turns every
# annotation into a string, and FastAPI then cannot resolve `UploadFile` when it
# builds the route - the worker below would fail to start with an error that
# points at pydantic rather than at the real cause. The Colab cell does not have
# that import either, for the same reason.

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

PASS = FAIL = 0


def check(label, got, want):
    global PASS, FAIL
    if got == want:
        PASS += 1
        print(f"  ok    {label}")
    else:
        FAIL += 1
        print(f"  FAIL  {label}\n          got  {got}\n          want {want}")


def start_worker(model_name: str, port: int):
    """The worker, byte-for-byte the same contract as the Colab cell."""
    from fastapi import FastAPI, File, Form, UploadFile
    from faster_whisper import WhisperModel
    import uvicorn

    m = WhisperModel(model_name, device="cpu", compute_type="int8")
    app = FastAPI()

    @app.get("/health")
    def health():
        return {"ok": True, "gpu": "cpu(test)", "model": model_name}

    @app.post("/transcribe")
    async def transcribe(audio: UploadFile = File(...), model: str = Form(model_name)):
        tmp = Path(f"_worker_{int(time.time()*1000)}.wav")
        tmp.write_bytes(await audio.read())
        segs, _ = m.transcribe(str(tmp), word_timestamps=True)
        words = [{"word": w.word, "start": float(w.start), "end": float(w.end)}
                 for s in segs for w in (s.words or [])]
        tmp.unlink(missing_ok=True)
        return {"words": words, "model": model_name}

    threading.Thread(
        target=lambda: uvicorn.run(app, host="127.0.0.1", port=port,
                                   log_level="error"),
        daemon=True).start()
    return f"http://127.0.0.1:{port}"


def main() -> int:
    from pipeline.transcribe import LocalWhisper, RemoteWhisper, Word, make

    # A short slice of the real voiceover - synthetic tones transcribe to
    # nothing, which would make the comparison vacuous.
    src = Path("wierdbiology/chin/voice_recordings/section_01.wav")
    if not src.exists():
        print(f"  SKIP - needs {src}")
        return 0

    import numpy as np
    import soundfile as sf
    x, sr = sf.read(src, always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    clip = Path("_seam_clip.wav")
    sf.write(clip, np.asarray(x[: sr * 12], dtype=np.float32), sr)

    try:
        print("starting a worker that speaks the Colab contract...")
        url = start_worker("base", 8901)
        for _ in range(60):
            try:
                RemoteWhisper(url).health()
                break
            except Exception:
                time.sleep(0.5)

        # -- the contract -------------------------------------------------
        r = RemoteWhisper(url, model="base")
        h = r.health()
        print("\nworker contract")
        check("health reports ok", h["ok"], True)
        check("health names its model", h["model"], "base")

        # -- the actual comparison ----------------------------------------
        print("\nremote vs local on the same audio")
        remote = r.words(clip)
        local = LocalWhisper("base").words(clip)

        check("both return Word objects",
              isinstance(remote[0], Word) and isinstance(local[0], Word), True)
        check("same number of words", len(remote), len(local))
        check("same text", [w.word for w in remote], [w.word for w in local])
        worst = max(abs(a.start - b.start) for a, b in zip(remote, local)) \
            if remote and local else 0.0
        check("timestamps identical", worst < 1e-6, True)
        print(f"    {len(remote)} words, worst start delta {worst:.2e}s")

        # -- the factory ---------------------------------------------------
        print("\nfactory picks the right side")
        check("a url means remote", isinstance(make("base", url), RemoteWhisper), True)
        check("no url means local", isinstance(make("base", None), LocalWhisper), True)

        # -- failure is loud, not silent -----------------------------------
        print("\na dead worker fails loudly")
        dead = RemoteWhisper("http://127.0.0.1:9", model="base")
        try:
            dead.words(clip)
            check("raised", False, True)
        except Exception as e:
            check("raises rather than returning nothing",
                  isinstance(e, Exception), True)
            print(f"    {type(e).__name__}")

    finally:
        clip.unlink(missing_ok=True)

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
