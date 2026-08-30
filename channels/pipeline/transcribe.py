"""Where the word timestamps come from - this laptop, or a GPU somewhere else.

THE SEAM IS DELIBERATELY NARROW: audio in, words out. Nothing else crosses it.

Mastering stays local and alignment stays local. Only the transcribe call - the
one step that actually wants a GPU - is allowed to run somewhere else. That is
what keeps this honest:

  * `pipeline.voice` remains the single definition of how audio is mastered.
    Send raw takes to a remote worker and the worker needs its own copy of that
    chain, which then drifts, and the timings it returns stop describing the
    audio the render will use. Sending MASTERED audio makes that impossible.
  * `pipeline.align` remains the single definition of how words become scenes.
    It is pure Python and takes milliseconds; shipping it to a GPU buys nothing
    and doubles the number of places the logic lives.

So a worker is a dumb box with one job, and it needs no pipeline code at all.
That is why the Colab notebook for it is thirty lines and has no version to keep
in step with this repo.

    local:   master(natural) -> [seam] -> align -> scenes_timeline.json
    worker:                     wav in, words out
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class Word:
    """One spoken word and when it was said.

    Mirrors faster-whisper's own word object field for field, so a local model's
    output and a remote worker's JSON are the same thing to every caller.
    """

    word: str
    start: float
    end: float


class Transcriber(Protocol):
    """Given a wav, return the words in it with timestamps."""

    name: str

    def words(self, wav: Path) -> list[Word]:
        ...


# ---------------------------------------------------------------------------
#  Local
# ---------------------------------------------------------------------------
class LocalWhisper:
    """faster-whisper in this process. CUDA if there is a card, else CPU."""

    def __init__(self, model: str = "base"):
        from faster_whisper import WhisperModel

        # BUG FIX the notebook records too: float16 is invalid on CPU and on
        # pre-Volta cards, and faster-whisper's error for it is not obvious.
        # Pick by capability rather than assuming.
        device, ctype = "cpu", "int8"
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
                ctype = ("float16"
                         if torch.cuda.get_device_capability(0)[0] >= 7
                         else "int8")
        except ImportError:
            pass

        self.name = f"local:{model}:{device}/{ctype}"
        self.model = model
        self.device = device
        self._m = WhisperModel(model, device=device, compute_type=ctype)

    def words(self, wav: Path) -> list[Word]:
        segs, _ = self._m.transcribe(str(wav), word_timestamps=True)
        return [Word(w.word, float(w.start), float(w.end))
                for s in segs for w in (s.words or [])]


# ---------------------------------------------------------------------------
#  Remote
# ---------------------------------------------------------------------------
class RemoteWhisper:
    """A worker over HTTP - in practice Colab behind a cloudflared tunnel.

    NO RETRIES, AND THAT IS DELIBERATE. A Colab tunnel dies for exactly one
    reason worth handling: the runtime went away. Retrying a dead runtime just
    delays the message the operator needs, and a half-finished retry against a
    RESTARTED runtime would silently mix words from two different model loads.
    Fail, say which section, let the operator restart the worker.
    """

    def __init__(self, url: str, model: str = "large-v3", timeout: float = 900.0):
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.name = f"colab:{model}:{self.url}"

    def health(self) -> dict:
        """Is the worker up, and which model does it already have loaded?"""
        import httpx
        r = httpx.get(f"{self.url}/health", timeout=20.0)
        r.raise_for_status()
        return r.json()

    def words(self, wav: Path) -> list[Word]:
        import httpx
        with wav.open("rb") as fh:
            r = httpx.post(
                f"{self.url}/transcribe",
                files={"audio": (wav.name, fh, "audio/wav")},
                data={"model": self.model},
                timeout=self.timeout,
            )
        if r.status_code != 200:
            raise RuntimeError(
                f"worker returned {r.status_code} for {wav.name}: "
                f"{r.text[:300]}")
        data = r.json()
        return [Word(w["word"], float(w["start"]), float(w["end"]))
                for w in data["words"]]


def make(model: str = "base", colab_url: str | None = None) -> Transcriber:
    """Pick a transcriber. A URL means remote; no URL means this machine."""
    if colab_url:
        return RemoteWhisper(colab_url, model=model)
    return LocalWhisper(model)
