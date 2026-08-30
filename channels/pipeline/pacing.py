"""How long a picture may stay on screen, and how many pictures that implies.

WHY THIS IS A MODULE AND NOT A DEFAULT ARGUMENT. `--seconds-per-image` defaulted
to 8.0, which was never a decision - it was the first number that produced a
sane-looking count. The channel's actual target is fast-cut: 2-3 seconds average,
4 seconds absolute ceiling. Those are editorial constants, they apply to every
video, and they belong somewhere both the planner and the checker can read them
rather than being retyped as magic numbers at each call site.

WHAT THE NUMBERS MEAN
    TARGET_LOW / TARGET_HIGH   the average hold should land inside this band
    HARD_MAX                   no single picture may exceed this
    MIN_HOLD                   below this a cut reads as a flash, not a change

The band matters more than any single frame. A video whose average sits at 2.5s
still feels fast even with a couple of 4s holds for the beats that need room; one
averaging 5.7s does not, however many quick cuts are sprinkled through it.

WHY LONG HOLDS ARE A WARNING AND NOT AN ERROR. Sometimes the script genuinely
wants a beat to land - and motion (a slow zoom) makes a long hold watchable in a
way a static frame is not. So this reports, and the operator decides.
"""

from __future__ import annotations

# Editorial targets for the channel. Change here, not at call sites.
#
# THE BODY BAND: the normal cut rhythm. A new scene every 2-4 seconds.
TARGET_LOW = 2.0
TARGET_HIGH = 4.0
HARD_MAX = 4.0

# THE ACCENT ZONE: 0.5-2.0s. A one-second cut is not a mistake - it is how a
# specific point in the narration gets emphasised, and the channel uses them
# deliberately. So a hold below the body band is reported as an accent, not a
# defect. Only holds under MIN_HOLD are wrong, because those read as a flash
# rather than as a beat.
ACCENT_LOW = MIN_HOLD = 0.5
ACCENT_HIGH = TARGET_LOW

# Roughly what share of a video should be accents. Above this the fast cuts stop
# reading as emphasis and start reading as the baseline rhythm - at which point
# nothing is emphasised. Advisory only.
ACCENT_SHARE_MAX = 0.25

# ---------------------------------------------------------------------------
#  MOTION SLOTS - the size of a slot that may carry an external video clip
# ---------------------------------------------------------------------------
# These live here rather than in motion.py because they are PACING facts: they
# describe how long a picture is on screen, which is this module's whole subject.
# motion.py owns how a frame MOVES; this owns how long a slot LASTS.
#
# THE COINCIDENCE THAT WRITES THE RULE. Google Flow's minimum clip is 4.0s and
# HARD_MAX is 4.0s. A motion clip is therefore exactly the longest hold the spec
# allows - so motion belongs on the slots that would otherwise sit at the
# ceiling, which are the ones most at risk of reading as slow.
MOTION_CLIP_S = 4.0

# Shortest hold still allowed to carry a clip. The residual (up to 0.5s) is
# absorbed by trimming the clip's TAIL - never by slowing it (the eye reads
# speed, and these are already a slow drift) and never by freezing it.
MOTION_SLOT_MIN = 3.5

# Cuts land on real Whisper word timestamps, so "exactly 4.000s" is not
# reachable. At ~150wpm a word is ~0.4s, which quantises the achievable holds:
#   9 words = 3.60s accept | 10 = 4.00s ideal | 11 = 4.40s over HARD_MAX, reject
# Worst-case miss from the target is therefore about half a word gap.
MOTION_SLOT_WINDOW = (MOTION_SLOT_MIN, MOTION_CLIP_S)

# Editorial dose, as a share of runtime. Motion scales with length; it is not a
# fixed count. Over-animating costs comprehension for the same reason too many
# accents do - if everything moves, nothing is emphasised.
MOTION_SHARE_LOW = 0.10
MOTION_SHARE_HIGH = 0.15


def motion_slots_for(audio_seconds: float,
                     share: float = (MOTION_SHARE_LOW + MOTION_SHARE_HIGH) / 2.0,
                     clip_s: float = MOTION_CLIP_S) -> int:
    """How many external clips a runtime should carry at a given dose.

    A motion clip REPLACES a still slot - it never appends runtime - so this is
    a count of slots to convert, not of seconds to add.
    """
    if clip_s <= 0:
        raise ValueError("clip_s must be positive")
    return max(0, int(audio_seconds * share / clip_s))


def motion_dose(audio_seconds: float) -> dict:
    """The 10-15% band expressed as a clip count, for the UI and the planner."""
    lo = motion_slots_for(audio_seconds, MOTION_SHARE_LOW)
    hi = motion_slots_for(audio_seconds, MOTION_SHARE_HIGH)
    return {
        "runtime": round(audio_seconds, 2),
        "clip_s": MOTION_CLIP_S,
        "share_low": MOTION_SHARE_LOW,
        "share_high": MOTION_SHARE_HIGH,
        "clips_low": lo,
        "clips_high": hi,
        "seconds_low": round(lo * MOTION_CLIP_S, 1),
        "seconds_high": round(hi * MOTION_CLIP_S, 1),
        "window": list(MOTION_SLOT_WINDOW),
    }


def fits_motion(hold: float) -> bool:
    """Is this hold inside the [3.5, 4.0] window a Flow clip may occupy?"""
    return MOTION_SLOT_MIN <= hold <= MOTION_CLIP_S


# The planner's default: the middle of the body band. Accents are placed
# deliberately in the editor, not produced by the planner's even division.
DEFAULT_SECONDS_PER_IMAGE = (TARGET_LOW + TARGET_HIGH) / 2.0      # 3.0


def images_for(audio_seconds: float, target: float = DEFAULT_SECONDS_PER_IMAGE) -> int:
    """How many pictures a given runtime needs to average `target` seconds each."""
    if target <= 0:
        raise ValueError("target must be positive")
    return max(1, round(audio_seconds / target))


def seconds_for_words(words: int, wpm: int = 150) -> float:
    """Estimated spoken duration of a word count. Used before any audio exists."""
    return words / max(wpm, 1) * 60.0


def verdict(holds: list[float]) -> dict:
    """Score a set of on-screen durations against the channel's pacing targets.

    Returns plain numbers plus a `rating` the UI can colour by, and a `note`
    written for a human rather than a log.
    """
    if not holds:
        return {"n": 0, "rating": "empty", "note": "no images"}

    s = sorted(holds)
    n = len(s)
    avg = sum(s) / n
    median = s[n // 2]

    over_hard = [h for h in s if h > HARD_MAX]
    accents = [h for h in s if ACCENT_LOW <= h < ACCENT_HIGH]
    flashes = [h for h in s if h < MIN_HOLD]
    # THE BODY IS EVERYTHING THAT IS NOT AN ACCENT - with no upper bound.
    # Capping it at HARD_MAX (as this first did) excluded the over-long holds
    # from their own average, so the chin video scored "on target" on a body mean
    # of 2.9s while 67 of its 105 pictures sat above four seconds. An average
    # that ignores the frames you are complaining about measures nothing.
    body = [h for h in s if h >= ACCENT_HIGH]

    # Judge on the body rather than the raw mean: deliberate one-second accents
    # pull the mean down, and a video with good 3s pacing plus a healthy scatter
    # of accents would otherwise be scored "too fast" for doing its job.
    body_avg = (sum(body) / len(body)) if body else avg
    accent_share = len(accents) / n

    if body_avg > HARD_MAX:
        rating = "too slow"
        note = (f"its normal cuts average {body_avg:.1f}s - the band is "
                f"{TARGET_LOW:.0f}-{TARGET_HIGH:.0f}s. About "
                f"{images_for(sum(s))} images would fix it, not {n}.")
    elif body_avg < TARGET_LOW:
        rating = "too fast"
        note = (f"even its normal cuts average {body_avg:.1f}s, under the "
                f"{TARGET_LOW:.0f}s floor - everything reads as an accent, so "
                f"nothing stands out.")
    else:
        rating = "on target"
        note = (f"normal cuts average {body_avg:.1f}s, inside the "
                f"{TARGET_LOW:.0f}-{TARGET_HIGH:.0f}s band.")

    if accents:
        note += (f" {len(accents)} accent cut(s) under {ACCENT_HIGH:.0f}s "
                 f"({accent_share * 100:.0f}%)")
        note += ("." if accent_share <= ACCENT_SHARE_MAX else
                 f" - above {ACCENT_SHARE_MAX * 100:.0f}%, so they stop reading "
                 f"as emphasis.")
    if over_hard:
        note += (f" {len(over_hard)} image(s) hold longer than "
                 f"{HARD_MAX:.0f}s (worst {max(over_hard):.1f}s).")
    if flashes:
        note += f" {len(flashes)} under {MIN_HOLD:.1f}s will flash."

    return {
        "n": n,
        "total": round(sum(s), 2),
        "avg": round(avg, 2),
        "body_avg": round(body_avg, 2),
        "median": round(median, 2),
        "min": round(s[0], 2),
        "max": round(s[-1], 2),
        "accents": len(accents),
        "accent_share": round(accent_share, 3),
        "over_hard": len(over_hard),
        "flashes": len(flashes),
        "target_low": TARGET_LOW,
        "target_high": TARGET_HIGH,
        "hard_max": HARD_MAX,
        "accent_high": ACCENT_HIGH,
        "min_hold": MIN_HOLD,
        "want_images": images_for(sum(s)),
        "rating": rating,
        "note": note,
    }


def plan_table(audio_seconds: float, have: int | None = None) -> list[dict]:
    """Images needed at each candidate pace - the picker the UI shows."""
    rows = []
    for t in (2.0, 2.5, 3.0, 4.0):
        n = images_for(audio_seconds, t)
        rows.append({
            "seconds_per_image": t,
            "images": n,
            "in_band": TARGET_LOW <= t <= TARGET_HIGH,
            "delta": (n - have) if have is not None else None,
        })
    return rows
