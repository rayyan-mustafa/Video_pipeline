"""Room tone: the sound of your room with nobody talking.

WHY THIS MODULE EXISTS AT ALL. Lengthening a pause means putting something in
the gap, and the obvious something - digital silence - is exactly wrong. Your
noise floor sits around -40 to -70 dBFS; drop it to true zero for a second and
the ear hears the FLOOR disappear, not a pause. That discontinuity is the
single clearest tell of amateur editing, and it is audible even to listeners
who could not name what they just heard.

So a stretched gap is filled with the room's own tone, taken from this same
recording, at this same gain, in this same room, on this same day. Nothing else
matches - not a library sample, not tone from last week's take.

IF THERE IS NO ROOM TONE, THERE IS NO FIX. A recording with no quiet passage
long enough to sample cannot be extended honestly, and this module says so
rather than inventing something.
"""

from __future__ import annotations

import numpy as np

MIN_TONE_SEC = 0.30
FRAME_SEC = 0.02


class NoRoomTone(RuntimeError):
    """No passage quiet enough, or long enough, to sample."""


def _frame_db(x: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    n = max(1, int(FRAME_SEC * sr))
    usable = (len(x) // n) * n
    if usable == 0:
        return np.zeros(0), n
    f = x[:usable].astype(np.float64).reshape(-1, n)
    # +1e-12 keeps a digitally silent frame finite instead of -inf, which would
    # otherwise win every "quietest window" search outright.
    return 20.0 * np.log10(np.sqrt((f ** 2).mean(axis=1)) + 1e-12), n


def extract(x: np.ndarray, sr: int, min_sec: float = MIN_TONE_SEC) -> np.ndarray:
    """The quietest continuous stretch of at least `min_sec`.

    Quietest rather than first: the head of a take often contains a chair
    creak or the click of the record key, and using it would splice that noise
    into every lengthened pause in the file.
    """
    db, n = _frame_db(x, sr)
    need = max(1, int(np.ceil(min_sec / FRAME_SEC)))
    if len(db) < need:
        raise NoRoomTone(
            "Recording is too short to sample room tone from - need at least "
            + format(min_sec, ".2f") + "s of continuous quiet.")

    # Mean dB over a sliding window; the minimum is the calmest passage.
    kernel = np.ones(need) / need
    windowed = np.convolve(db, kernel, mode="valid")
    start_frame = int(np.argmin(windowed))

    quiet = float(windowed[start_frame])
    speech = float(np.percentile(db, 90))
    if speech - quiet < 6.0:
        # Nothing here is meaningfully quieter than everything else, so the
        # "quiet" sample would contain speech. Splicing that into a gap would
        # repeat a fragment of a word under the silence.
        raise NoRoomTone(
            "No passage is quiet enough to be room tone (quietest window is "
            + format(speech - quiet, ".1f") + " dB below speech level). "
            "Leave a second of silence at the top of the take.")

    a = start_frame * n
    b = a + need * n
    return x[a:b].astype(np.float32).copy()


def fill(tone: np.ndarray, samples: int, sr: int,
         rng: np.random.Generator | None = None) -> np.ndarray:
    """`samples` worth of room tone, built by tiling the sample.

    Tiled with ALTERNATING DIRECTION and crossfades at the joins. A plain loop
    of a 300 ms sample repeats any faint periodic component - a fan, a hum, a
    hard-drive whine - at exactly 300 ms, and a repeating artifact is far more
    audible than the noise it was cut from. Reversing every other copy breaks
    that periodicity without changing the spectrum.
    """
    if samples <= 0:
        return np.zeros(0, dtype=np.float32)
    if len(tone) == 0:
        raise NoRoomTone("Empty room-tone sample.")

    xfade = min(int(0.010 * sr), len(tone) // 4)
    out = np.zeros(0, dtype=np.float32)
    flip = False
    while len(out) < samples + xfade:
        piece = tone[::-1] if flip else tone
        flip = not flip
        if len(out) == 0:
            out = piece.astype(np.float32).copy()
            continue
        out = _crossfade(out, piece.astype(np.float32), xfade)
    return out[:samples]


def _crossfade(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """Equal-power crossfade, not linear.

    Two uncorrelated noise signals summed with linear gains dip ~3 dB in the
    middle of the fade, which reads as a small hole in the room tone every time
    two pieces are joined. sqrt weighting keeps the power constant.
    """
    if n <= 0 or len(a) < n or len(b) < n:
        return np.concatenate([a, b]).astype(np.float32)
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    fade_out, fade_in = np.sqrt(1.0 - t), np.sqrt(t)
    head, tail = a[:-n], a[-n:]
    joined = tail * fade_out + b[:n] * fade_in
    return np.concatenate([head, joined, b[n:]]).astype(np.float32)
