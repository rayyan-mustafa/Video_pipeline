"""The live audio ring buffer.

ONE BUFFER, ONE RATE. Capture runs at the configured 48 kHz - the same rate
`record()` writes and every offline metric is measured at - and ASR resamples
to 16 kHz at its own boundary, where `asr.transcribe()` already handles it.

That is a deliberate refusal to add a second rate. If live capture ran at some
other rate, live and offline numbers would disagree on the same voice and there
would be no way to tell which was right. This project has already lost a day to
exactly that failure (48 kHz fed to a 16 kHz ASR API), and the lesson was:
resample at one boundary, never carry two rates around.

THREADING. The PortAudio callback writes from its own thread; the asyncio
server reads from the event loop. A lock guards the write index. The callback
must never block, so it does nothing but copy into the array and advance.
"""

from __future__ import annotations

import threading

import numpy as np


class Ring:
    """Fixed-capacity mono float32 ring buffer.

    Sized in seconds. Old audio is overwritten, which is correct here: the HUD
    measures the recent past, and a live coach that falls behind should drop
    history rather than grow without bound over a two-hour session.
    """

    def __init__(self, seconds: float, sr: int):
        self.sr = int(sr)
        self.capacity = int(seconds * sr)
        self._buf = np.zeros(self.capacity, dtype=np.float32)
        self._write = 0            # total samples ever written, not an index
        self._lock = threading.Lock()

    # -- writing (audio thread) ---------------------------------------------
    def write(self, block: np.ndarray) -> None:
        """Append a block. Called from the PortAudio callback - keep it cheap."""
        x = np.asarray(block, dtype=np.float32).reshape(-1)
        n = len(x)
        if n == 0:
            return

        dropped = 0
        if n > self.capacity:
            # A single block larger than the whole buffer: keep only the tail.
            # THE DROPPED SAMPLES STILL COUNT AS TIME. `_write` is the clock
            # every absolute timestamp is measured against - clause boundaries,
            # the spans handed to ASR - so failing to advance it here would
            # shift everything after this block earlier by the dropped amount,
            # and the transcript would be cut from the wrong audio with nothing
            # to indicate it.
            dropped = n - self.capacity
            x = x[-self.capacity:]
            n = self.capacity

        with self._lock:
            self._write += dropped
            start = self._write % self.capacity
            end = start + n
            if end <= self.capacity:
                self._buf[start:end] = x
            else:
                split = self.capacity - start
                self._buf[start:] = x[:split]
                self._buf[:n - split] = x[split:]
            self._write += n

    # -- reading (any thread) -----------------------------------------------
    @property
    def total_samples(self) -> int:
        with self._lock:
            return self._write

    @property
    def elapsed(self) -> float:
        return self.total_samples / self.sr

    def tail(self, seconds: float) -> np.ndarray:
        """The most recent `seconds` of audio, oldest first."""
        return self.between(max(0.0, self.elapsed - seconds), self.elapsed)

    def between(self, t0: float, t1: float) -> np.ndarray:
        """Audio between two absolute times, in seconds since capture started.

        Returns whatever part of that span is still in the buffer. A caller
        asking for audio that has already been overwritten gets a SHORTER array
        rather than silence or an error - silence would be measured as a pause
        that never happened.
        """
        with self._lock:
            written = self._write
            oldest = max(0, written - self.capacity)
            a = max(int(t0 * self.sr), oldest)
            b = min(int(t1 * self.sr), written)
            if b <= a:
                return np.zeros(0, dtype=np.float32)
            n = b - a
            start = a % self.capacity
            end = start + n
            if end <= self.capacity:
                return self._buf[start:end].copy()
            split = self.capacity - start
            return np.concatenate([self._buf[start:], self._buf[:n - split]])

    def audio(self, seconds: float):
        """The tail as an `Audio`, so offline analysis functions run unchanged."""
        from ..analysis.audio import Audio

        return Audio(samples=self.tail(seconds), sr=self.sr)
