"""Do motion slots land in the window, whatever the audio does?

This is the regression guard for the bug that shipped: motion slots are sized in
WORDS at plan time, and only measured timestamps can say how long they actually
last. Every re-align - a new Whisper model, a different voice profile, a
re-recorded section - moves those timestamps, and the slots have to follow.

Runs on synthetic word tracks, so it needs no audio and no Whisper.

    ..\\vo_coach\\.venv\\Scripts\\python.exe test_timeplan.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pipeline import pacing, timeplan  # noqa: E402

PASS = FAIL = 0


def check(label: str, got, want) -> None:
    global PASS, FAIL
    if got == want:
        PASS += 1
        print(f"  ok    {label}")
    else:
        FAIL += 1
        print(f"  FAIL  {label}\n          got  {got}\n          want {want}")


def words(n: int, section: int = 1, pace: float = 0.4,
          jitter: list[float] | None = None) -> list[dict]:
    """A word track. `jitter` scales individual gaps, to model uneven delivery."""
    out, t = [], 0.0
    for i in range(n):
        gap = pace * (jitter[i % len(jitter)] if jitter else 1.0)
        out.append({"section": section, "w": f"w{i}", "s": round(t, 3),
                    "e": round(t + gap * 0.8, 3)})
        t += gap
    return out


def plan(n_words: int, n_motion: int, section: int = 1) -> dict:
    """A plan whose slots are all the wrong size, so refine has work to do."""
    imgs, order, i = [], 0, 0
    while i < n_words:
        take = min(7, n_words - i)
        order += 1
        imgs.append({
            "section": section, "order": order,
            "filename": f"{order}section{section}.png",
            "scene_indices": [0],
            "narration": [" ".join(f"w{j}" for j in range(i, i + take))],
            "words": take,
            "kind": "video" if order <= n_motion else "image",
        })
        i += take
    return {"images": imgs, "seconds_per_image": 3.0}


# THE PLAN NEVER STORES A TIMESTAMP - that is the design (see plan_script's
# header), so refine() writes word counts only and the holds live in the layout
# it computed. The window property therefore has to be asserted on
# layout_section, which is the unit that actually decides it.
def layout(word_track: list[dict], n_motion: int, target: float = 3.0):
    return timeplan.layout_section(word_track, n_motion, still_target=target)


def holds(slots: list[dict], kind: str) -> list[float]:
    return [s["hold"] for s in slots if s["kind"] == kind]


def main() -> int:
    W = pacing.MOTION_SLOT_MIN, pacing.MOTION_CLIP_S

    # -- 1. EVEN DELIVERY ---------------------------------------------------
    p = plan(400, 6)
    rep = timeplan.refine(p, words(400), still_target=3.0)
    m = holds(layout(words(400), 6), "video")
    print("even delivery")
    check("every motion slot placed", rep["motion_after"], 6)
    check("all inside the window", all(W[0] <= h <= W[1] for h in m), True)

    # -- 2. UNEVEN DELIVERY - THE ACTUAL BUG --------------------------------
    # Word gaps swing 0.5x to 1.8x, which is what a real read does. Sizing by
    # word count alone produced holds from 2.8s to 5.4s here.
    p = plan(400, 6)
    jit = [0.5, 1.8, 0.7, 1.4, 0.9, 1.6, 0.6, 1.2]
    rep = timeplan.refine(p, words(400, jitter=jit), still_target=3.0)
    m = holds(layout(words(400, jitter=jit), 6), "video")
    print("\nuneven delivery (0.5x - 1.8x word gaps)")
    check("every motion slot placed", rep["motion_after"], 6)
    check("all inside the window", all(W[0] <= h <= W[1] for h in m), True)
    print(f"    holds: {[round(h, 2) for h in m]}")

    # -- 3. A RE-ALIGN MUST NOT LEAVE SLOTS STALE ---------------------------
    # The natural-voice switch lengthened everything by ~1%, and 5 of 11 slots
    # fell out of the window because nothing re-cut them. Refining against the
    # NEW track must put them all back.
    p = plan(400, 6)
    timeplan.refine(p, words(400), still_target=3.0)
    slower = words(400, pace=0.44)                    # same read, 10% slower
    timeplan.refine(p, slower, still_target=3.0)
    m = holds(layout(slower, 6), "video")
    print("\nre-align onto a 10% slower read")
    check("all still inside the window", all(W[0] <= h <= W[1] for h in m), True)

    # -- 4. FEWER WORDS HEARD THAN SCRIPTED ---------------------------------
    # Whisper drops words on any long read. This used to abandon the whole
    # section and keep the stale layout - the one section that most needed
    # re-cutting was the one left alone.
    p = plan(400, 6)
    rep = timeplan.refine(p, words(388), still_target=3.0)
    m = holds(layout(words(388), 6), "video")
    print("\nWhisper heard 388 of 400 words")
    check("section still refined", rep["sections"][0]["refined"], True)
    check("says it resampled",
          "resampled" in rep["sections"][0].get("why", ""), True)
    check("motion slots still placed", rep["motion_after"], 6)
    check("all inside the window", all(W[0] <= h <= W[1] for h in m), True)

    # -- 5. NOTHING HEARD ---------------------------------------------------
    p = plan(400, 6)
    rep = timeplan.refine(p, [], still_target=3.0)
    print("\nno words heard at all")
    check("refuses rather than inventing timings",
          rep["sections"][0]["refined"], False)

    # -- 6. STILLS STAY IN THE PACING BAND ----------------------------------
    # Re-cutting for motion must not wreck the stills around it.
    p = plan(400, 6)
    timeplan.refine(p, words(400, jitter=jit), still_target=3.0)
    s = holds(layout(words(400, jitter=jit), 6), "image")
    inband = sum(1 for h in s if pacing.ACCENT_LOW <= h <= pacing.HARD_MAX)
    print("\nstills stay in the band")
    check("90%+ of stills within 0.5-4.0s", inband / len(s) >= 0.9, True)
    print(f"    {inband}/{len(s)} stills in band")

    # -- 7. IDEMPOTENT ------------------------------------------------------
    p = plan(400, 6)
    w = words(400, jitter=jit)
    timeplan.refine(p, w, still_target=3.0)
    first = [i["words"] for i in p["images"]]
    timeplan.refine(p, w, still_target=3.0)
    print("\nrefining twice changes nothing")
    check("same layout", [i["words"] for i in p["images"]], first)

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
