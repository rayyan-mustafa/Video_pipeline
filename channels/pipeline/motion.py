"""How a still picture moves while it is on screen, and why it moves that much.

WHY THIS IS A MODULE AND NOT A FLAG ON THE MUXER. `pacing.py` already states the
thing this file implements: "motion (a slow zoom) makes a long hold watchable in
a way a static frame is not." That is an editorial rule about the channel, not a
render option, so it belongs next to the pacing constants where both the planner
and the renderer can read it -- exactly like TARGET_LOW and HARD_MAX.

THE ONE IDEA WORTH KEEPING: MOTION IS A RATE, NOT A TOTAL.
A fixed "zoom 12% over the hold" looks right at 3 seconds and looks like a still
at 15, because the same distance spread over five times the time moves at a
fifth of the speed, and the eye reads speed rather than displacement. So the
amount of movement is derived from `<per second> * duration` and then clamped at
both ends: under MIN_TOTAL nothing is visible, over MAX_TOTAL the crop starts
eating the figure. A 15s hold therefore drifts for its whole 15s at a readable
speed instead of creeping imperceptibly.

WHY ACCENTS DO NOT MOVE. `pacing.py` defines a hold under 2s as a deliberate
accent - a point the narration is leaning on. A zoom that starts and stops
inside one second is not motion, it is a twitch, and it damages the emphasis the
short cut exists to create. Accents render static. That is a rule, not a knob.

WHY THE MOVE ROTATES DETERMINISTICALLY. Two neighbouring push-ins read as one
long push interrupted by a cut, which makes the cut vanish - the opposite of
fast-cut editorial. The move is picked from the slot index, so no two adjacent
slots share one and a re-render of the same plan is identical rather than
reshuffling every time.

WHAT THIS DELIBERATELY DOES NOT DO: transitions between slots. A crossfade
softens a cut, and this channel's entire rhythm is that the cut is hard. Motion
inside the frame, hard cut between frames.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import pacing

# ---------------------------------------------------------------------------
#  Editorial constants. Change here, never at a call site.
# ---------------------------------------------------------------------------

# Movement per second on screen. 3% of the frame per second is a drift you
# notice without being able to point at it; by 6% it reads as a camera move and
# competes with the narration for attention.
ZOOM_RATE_PER_S = 0.030
PAN_RATE_PER_S = 0.035

# Clamps on the total, so neither a 2s body cut nor a 15s over-ceiling hold ends
# up outside what the format can carry.
#   MIN_TOTAL  below this the movement is invisible and costs render time for nothing
#   MAX_TOTAL  above this the crop eats the figure, which is drawn to fill most
#              of the frame height by design (see config.py FIGURE_* notes)
MIN_TOTAL = 0.040
MAX_TOTAL = 0.220

# A pan needs somewhere to travel: you cannot slide across an image that exactly
# fills the frame. So a panning move holds a constant zoom, and the pan distance
# is a fraction of the margin that zoom creates.
PAN_BASE_ZOOM = 1.14

# Body holds (2-4s) get a gentler version of the same move than over-ceiling
# holds do. An over-long hold is the one that actually needs rescuing, and it is
# also the one with the most time to move in.
BODY_SCALE = 0.70
RESCUE_SCALE = 1.00

# Supersample factor before zoompan. WITHOUT THIS THE LINES SHIMMER: zoompan
# quantises its zoom to integer source pixels, so on flat line art the strokes
# visibly step and crawl frame to frame. Rendering the move at 3x and scaling
# down turns each step into a third of a pixel, which reads as smooth. This is
# the most expensive constant in the file - measure before lowering it to "save
# time", because shimmering line art is the one artefact that makes the whole
# video look cheap.
SUPERSAMPLE = 3


@dataclass(frozen=True)
class Move:
    """One camera move, as start/end of zoom and of framing.

    `zoom` is a scale factor and is never below 1.0 - zoompan cannot show pixels
    that are not there, and a value under 1.0 produces black edges rather than a
    pull-back. A "pull out" is therefore 1.22 -> 1.00, not 1.00 -> 0.82.

    `px` / `py` are the framing centre in [0, 1] of whatever margin the current
    zoom leaves: 0.5 centred, 0.0 hard left/top, 1.0 hard right/bottom. They are
    normalised against the margin rather than the image, so the same numbers
    mean the same visual thing at any zoom.
    """

    name: str
    z0: float
    z1: float
    px0: float = 0.5
    px1: float = 0.5
    py0: float = 0.5
    py1: float = 0.5

    @property
    def is_static(self) -> bool:
        return (self.z0 == self.z1 and self.px0 == self.px1
                and self.py0 == self.py1)


# The unit vocabulary, written at amplitude 1.0. `scaled()` is what turns these
# into the move actually rendered for a given duration - these numbers are
# directions, not distances.
STATIC = Move("static", 1.0, 1.0)

PUSH_IN = Move("push_in", 1.0, 1.0 + MAX_TOTAL)
PULL_OUT = Move("pull_out", 1.0 + MAX_TOTAL, 1.0)
PAN_RIGHT = Move("pan_right", PAN_BASE_ZOOM, PAN_BASE_ZOOM,
                 px0=0.5 - MAX_TOTAL, px1=0.5 + MAX_TOTAL)
PAN_LEFT = Move("pan_left", PAN_BASE_ZOOM, PAN_BASE_ZOOM,
                px0=0.5 + MAX_TOTAL, px1=0.5 - MAX_TOTAL)
PAN_UP = Move("pan_up", PAN_BASE_ZOOM, PAN_BASE_ZOOM,
              py0=0.5 + MAX_TOTAL, py1=0.5 - MAX_TOTAL)
PAN_DOWN = Move("pan_down", PAN_BASE_ZOOM, PAN_BASE_ZOOM,
                py0=0.5 - MAX_TOTAL, py1=0.5 + MAX_TOTAL)
PUSH_RIGHT = Move("push_right", 1.0, 1.0 + MAX_TOTAL,
                  px0=0.5 - MAX_TOTAL / 2, px1=0.5 + MAX_TOTAL / 2)
PULL_LEFT = Move("pull_left", 1.0 + MAX_TOTAL, 1.0,
                 px0=0.5 + MAX_TOTAL / 2, px1=0.5 - MAX_TOTAL / 2)

# Order matters: consecutive entries are visually unlike each other, so walking
# the list by slot index never repeats a move back to back, and pushes alternate
# with pulls so the sequence does not accumulate in one direction either.
ROTATION: tuple[Move, ...] = (
    PUSH_IN, PAN_RIGHT, PULL_OUT, PAN_UP,
    PUSH_RIGHT, PAN_LEFT, PULL_LEFT, PAN_DOWN,
)


def scaled(base: Move, hold: float, intensity: float = 1.0) -> Move:
    """Resolve a unit move into the distance actually travelled over `hold`.

    This is where "motion is a rate" is applied. `base` supplies the direction
    and a MAX_TOTAL-sized amplitude; the returned Move carries the amplitude
    that `hold` seconds at ZOOM_RATE_PER_S earns, clamped to [MIN, MAX].
    """
    if base.is_static or hold <= 0:
        return base

    def amount(rate: float) -> float:
        return max(MIN_TOTAL, min(MAX_TOTAL, rate * hold * intensity))

    zf = amount(ZOOM_RATE_PER_S) / MAX_TOTAL
    pf = amount(PAN_RATE_PER_S) / MAX_TOTAL

    def shrink_mid(a: float, b: float, f: float) -> tuple[float, float]:
        """Pull a span in towards its own midpoint by factor f.

        Right for framing: a pan that travels less should travel less on each
        side of centre, not drift off to one side of the picture.
        """
        mid = (a + b) / 2.0
        return mid + (a - mid) * f, mid + (b - mid) * f

    def shrink_low(a: float, b: float, f: float) -> tuple[float, float]:
        """Pull a span in towards its LOWER end by factor f.

        Right for zoom, and midpoint anchoring is actively wrong here: shrinking
        a 1.00 -> 1.22 push towards its midpoint gave 1.09 -> 1.13, so a short
        body cut opened already cropped by 9% and never showed the whole
        picture at all. Anchoring low means a push-in always starts on the full
        frame and a pull-out always lands on it.
        """
        lo = min(a, b)
        span = abs(b - a) * f
        return (lo, lo + span) if b >= a else (lo + span, lo)

    z0, z1 = shrink_low(base.z0, base.z1, zf)
    px0, px1 = shrink_mid(base.px0, base.px1, pf)
    py0, py1 = shrink_mid(base.py0, base.py1, pf)

    # A pan's constant zoom IS its travel room, so it must not shrink along with
    # the pan distance - only the distance travelled inside it scales.
    if base.z0 == base.z1:
        z0 = z1 = base.z0

    return Move(base.name, max(1.0, z0), max(1.0, z1), px0, px1, py0, py1)


def move_for(index: int, hold: float) -> Move:
    """The move a slot gets, from its place in the sequence and its length.

    Encodes the two editorial rules at the top of this file: accents do not
    move, and the longer the hold the harder it works.
    """
    if hold < pacing.ACCENT_HIGH:
        return STATIC
    base = ROTATION[index % len(ROTATION)]
    scale = RESCUE_SCALE if hold > pacing.HARD_MAX else BODY_SCALE
    return scaled(base, hold, scale)


def frames_for(hold: float, fps: int) -> int:
    """Frame count for a hold. At least one, or ffmpeg emits nothing at all."""
    return max(1, int(round(hold * fps)))


def filter_chain(move: Move, hold: float, width: int, height: int,
                 fps: int, supersample: int = SUPERSAMPLE) -> str:
    """The complete -vf that turns one still into one moving clip.

    NO COMMAS APPEAR INSIDE ANY EXPRESSION, deliberately. ffmpeg's filtergraph
    parser splits on commas, and the usual fix is nested quoting that behaves
    differently through a shell than through subprocess's argv. Every endpoint
    is computed here in Python instead, so each expression is pure arithmetic
    and the string survives being passed anywhere.
    """
    n = frames_for(hold, fps)
    sw, sh = width * supersample, height * supersample

    # Fit-and-pad to the supersampled frame first, so the move happens against
    # the final composition rather than against whatever the source aspect was.
    prefix = (f"scale={sw}:{sh}:force_original_aspect_ratio=decrease:flags=lanczos,"
              f"pad={sw}:{sh}:(ow-iw)/2:(oh-ih)/2:white")

    if move.is_static:
        # Still a zoompan rather than a plain scale: it is the thing that turns
        # one input frame into n output frames, which keeps every slot the same
        # kind of object whether or not it moves.
        z, x, y = "1.0", "(iw-iw/zoom)/2", "(ih-ih/zoom)/2"
    else:
        den = max(1, n - 1)          # n == 1 would divide by zero
        prog = f"on/{den}"

        def ramp(a: float, b: float) -> str:
            if a == b:
                return f"{a:.6f}"
            return f"({a:.6f}+{b - a:.6f}*{prog})"

        z = ramp(move.z0, move.z1)
        x = f"(iw-iw/zoom)*{ramp(move.px0, move.px1)}"
        y = f"(ih-ih/zoom)*{ramp(move.py0, move.py1)}"

    return (f"{prefix},"
            f"zoompan=z='{z}':x='{x}':y='{y}':d={n}:s={width}x{height}:fps={fps},"
            f"format=yuv420p")


def describe(index: int, hold: float) -> dict:
    """What a slot will do, for the checker and the UI. Renders nothing."""
    m = move_for(index, hold)
    if m.is_static:
        why = (f"accent - under {pacing.ACCENT_HIGH:.0f}s, motion would read "
               f"as a twitch")
    elif hold > pacing.HARD_MAX:
        why = f"over the {pacing.HARD_MAX:.0f}s ceiling - full motion"
    else:
        why = "body cut - gentle motion"
    return {
        "index": index,
        "hold": round(hold, 2),
        "move": m.name,
        "zoom_from": round(m.z0, 4),
        "zoom_to": round(m.z1, 4),
        "static": m.is_static,
        "reason": why,
    }
