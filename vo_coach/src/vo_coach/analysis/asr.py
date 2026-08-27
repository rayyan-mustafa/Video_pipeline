"""Offline ASR via faster-whisper.

Word-level timestamps are the backbone of every pacing and pause metric, so
this returns them or fails - it never silently degrades to segment-level times.

Model loading is cached per (size, device, compute_type): re-analyzing ten takes
should load the model once, not ten times.
"""

from __future__ import annotations

import functools

import numpy as np

from ..schemas import Word
from .audio import Audio

DEFAULT_MODEL = "base.en"

# Whisper's fixed input rate. Not configurable - the model was trained at this
# rate and `transcribe()` has no sample-rate parameter to tell it otherwise.
ASR_SAMPLE_RATE = 16000


@functools.lru_cache(maxsize=2)
def _model(size: str, device: str, compute_type: str):
    from faster_whisper import WhisperModel

    return WhisperModel(size, device=device, compute_type=compute_type)


def _pick_device() -> tuple[str, str]:
    """float16 is invalid on CPU and on pre-Volta cards; picking by capability
    avoids an error message that does not name the real cause."""
    try:
        import torch

        if torch.cuda.is_available():
            major = torch.cuda.get_device_capability(0)[0]
            return "cuda", ("float16" if major >= 7 else "int8")
    except Exception:
        pass
    return "cpu", "int8"


def transcribe(audio: Audio, model_size: str = DEFAULT_MODEL,
               language: str = "en") -> tuple[str, list[Word]]:
    """Return (transcript, words). Words always carry start/end/confidence."""
    # The `.en` checkpoints are English-only. Handed Bengali they do not error -
    # they return confident, well-formed English nonsense, and every WPM and
    # pause number computed from it looks entirely plausible. Fail loudly
    # instead, because this is not detectable downstream.
    if model_size.endswith(".en") and language != "en":
        raise ValueError(
            f"Model {model_size!r} is English-only but language is {language!r}. "
            f"Use a multilingual checkpoint (e.g. 'small' or 'base') - the .en "
            f"models silently return garbage rather than failing."
        )

    device, compute_type = _pick_device()
    model = _model(model_size, device, compute_type)

    # WHISPER ASSUMES 16 kHz AND CANNOT BE TOLD OTHERWISE.
    #
    # `model.transcribe()` takes a bare float array with no sample-rate
    # argument, so it interprets whatever it is handed as 16 kHz. The rest of
    # this pipeline runs at the configured rate (48 kHz), and feeding that in
    # directly stretches the audio by 48000/16000 = 3x: a 60 s clip is heard as
    # 180 s of slowed-down speech.
    #
    # Nothing about that failure looks like a failure. Whisper returns a
    # confident transcript (hallucinated, because slowed speech is unintelligible
    # to it) with word timestamps running to 179 s, and every downstream metric
    # is computed from those - producing an impossible 40 wpm and a 6.7 s mean
    # pause on a 60 s file. Resample here, at the boundary, so no caller has to
    # remember this.
    samples = audio.samples.astype("float32")
    if audio.sr != ASR_SAMPLE_RATE:
        from scipy.signal import resample_poly

        g = np.gcd(int(ASR_SAMPLE_RATE), int(audio.sr))
        samples = resample_poly(samples, ASR_SAMPLE_RATE // g,
                                int(audio.sr) // g).astype("float32")

    segments, _info = model.transcribe(
        samples,
        language=language,
        word_timestamps=True,
        vad_filter=False,          # pauses are measured here, not removed
    )

    words: list[Word] = []
    parts: list[str] = []
    for seg in segments:
        parts.append(seg.text)
        for w in (seg.words or []):
            token = w.word.strip()
            if not token:
                continue
            words.append(Word(
                word=token,
                start=round(float(w.start), 3),
                end=round(float(w.end), 3),
                # faster-whisper gives probability, already 0..1
                confidence=round(float(getattr(w, "probability", 1.0)), 4),
            ))

    if not words:
        raise RuntimeError(
            "ASR returned no word timestamps. Every pacing and pause metric "
            "depends on them, so this is fatal rather than degradable. Check "
            "the take is not silent and that the model supports word_timestamps."
        )

    # No word can end after the audio does. This invariant is cheap and it is
    # the ONLY thing that exposed the 48k-fed-as-16k bug: there was no
    # exception, no warning, a fluent (hallucinated) transcript and well-formed
    # timestamps - just words running to 179 s in a 60 s file. A silent 3x
    # scaling of every downstream metric is worse than a crash, so crash.
    overrun = words[-1].end / audio.duration if audio.duration > 0 else 0.0
    if overrun > 1.05:
        raise RuntimeError(
            f"ASR word timestamps run to {words[-1].end:.1f}s in a "
            f"{audio.duration:.1f}s file ({overrun:.2f}x). The audio was almost "
            f"certainly handed to Whisper at the wrong sample rate - it assumes "
            f"{ASR_SAMPLE_RATE} Hz and has no parameter to say otherwise. Every "
            f"pacing and pause metric downstream would be scaled by this factor."
        )

    return "".join(parts).strip(), words
