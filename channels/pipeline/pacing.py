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
TARGET_LOW = 2.0
TARGET_HIGH = 3.0
HARD_MAX = 4.0

# Mirrors imageplan.MIN_HOLD. Duplicated as a named constant rather than imported
# to keep this module dependency-free, and asserted equal by the tests.
MIN_HOLD = 0.5

# The planner's default, derived from the band rather than picked.
DEFAULT_SECONDS_PER_IMAGE = (TARGET_LOW + TARGET_HIGH) / 2.0      # 2.5


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
    under_min = [h for h in s if h < MIN_HOLD]

    if avg > HARD_MAX:
        rating = "too slow"
        note = (f"averages {avg:.1f}s - the target band is "
                f"{TARGET_LOW:.0f}-{TARGET_HIGH:.0f}s. Needs about "
                f"{images_for(sum(s))} images, not {n}.")
    elif avg > TARGET_HIGH:
        rating = "slow"
        note = (f"averages {avg:.1f}s, just outside the "
                f"{TARGET_LOW:.0f}-{TARGET_HIGH:.0f}s band. "
                f"{images_for(sum(s))} images would land inside it.")
    elif avg < TARGET_LOW:
        rating = "fast"
        note = (f"averages {avg:.1f}s, quicker than the "
                f"{TARGET_LOW:.0f}-{TARGET_HIGH:.0f}s band - fine if deliberate.")
    else:
        rating = "on target"
        note = f"averages {avg:.1f}s, inside the target band."

    if over_hard:
        note += (f" {len(over_hard)} image(s) hold longer than "
                 f"{HARD_MAX:.0f}s (worst {max(over_hard):.1f}s).")
    if under_min:
        note += f" {len(under_min)} hold under {MIN_HOLD:.1f}s and will flash."

    return {
        "n": n,
        "total": round(sum(s), 2),
        "avg": round(avg, 2),
        "median": round(median, 2),
        "min": round(s[0], 2),
        "max": round(s[-1], 2),
        "over_hard": len(over_hard),
        "under_min": len(under_min),
        "target_low": TARGET_LOW,
        "target_high": TARGET_HIGH,
        "hard_max": HARD_MAX,
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
