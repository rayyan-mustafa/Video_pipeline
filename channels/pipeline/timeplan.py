"""Re-size the plan's slots against MEASURED word timestamps.

WHY A SECOND PASS EXISTS AT ALL. `plan_script.py` sizes every slot in words,
because when it runs there is no audio and words are the only ruler it has. That
is right for stills - a slot two words off target is a tenth of a second out and
nobody sees it. It is not good enough for motion: a Flow clip is a fixed 4.0s and
the slot has to land in [3.5, 4.0] or the clip cannot cover it without slowing or
freezing, and both are banned.

Word counts cannot deliver that. Measured on chin, slots built at 9 words - 4.0s
at the section's average pace - came out anywhere from 2.82s to 5.44s, because
delivery speed varies far more within a section than between sections. Only
3 of 11 landed in the window.

So: plan in words, record, align, then come back and re-cut with the real
timestamps. This is that second pass. It changes only WHERE the cuts fall, never
the script, never the scene texts, and never the fingerprint - so the binding the
build checks stays valid.

THE CUT RULE, unchanged from the spec: from a slot's first word, take the cut
word whose timestamp is nearest to start + target WITHOUT EXCEEDING IT. Cuts stay
on real word boundaries, which is what the timeline editor's roll edits depend on.
"""

from __future__ import annotations

from . import pacing


def _cut_index(words: list[dict], i: int, target: float) -> int:
    """Largest j > i whose start is within `target` of word i's start.

    Never returns i itself, so a slot always advances by at least one word and
    the layout cannot loop forever on a long pause.
    """
    n = len(words)
    start = float(words[i]["s"])
    j = best = min(i + 1, n - 1)
    while j < n and float(words[j]["s"]) - start <= target:
        best = j
        j += 1
    return max(best, min(i + 1, n - 1))


def layout_section(words: list[dict], n_motion: int,
                   still_target: float = pacing.DEFAULT_SECONDS_PER_IMAGE,
                   motion_target: float = pacing.MOTION_CLIP_S,
                   ) -> list[dict]:
    """Cut one section's words into slots that hit their targets in real time.

    Returns [{words, kind, start, end}] covering the section with no gaps.

    Motion slots are aimed at points spread evenly through the section rather
    than taken greedily at the front - two clips near each other read as one long
    shot, and clustering them at the start would leave the back half static.
    """
    n = len(words)
    if n == 0:
        return []

    span_start = float(words[0]["s"])
    span_end = float(words[-1]["e"])
    want_at = [span_start + (k + 0.5) * (span_end - span_start) / n_motion
               for k in range(n_motion)] if n_motion else []

    slots: list[dict] = []
    i, mi = 0, 0
    while i < n:
        start = float(words[i]["s"])

        # Is it time for a clip, and is there room left for a whole one? A slot
        # that would run past the end of the section cannot hold 4 seconds.
        want_motion = (mi < len(want_at) and start >= want_at[mi]
                       and (span_end - start) >= motion_target)
        target = motion_target if want_motion else still_target
        cut = _cut_index(words, i, target)
        end = float(words[cut]["s"]) if cut < n else span_end

        # The last slot runs to the end of the section rather than to a word
        # start, so the section tiles exactly.
        remaining = n - cut
        if remaining <= 1:
            end = span_end
            cut = n

        hold = end - start
        kind = "image"
        if want_motion:
            # ACCEPT ONLY INSIDE THE WINDOW. If this stretch of speech cannot
            # produce a 3.5-4.0s hold, it does not get a clip - placement is
            # opportunistic and the script is never bent to make one fit.
            if pacing.fits_motion(hold):
                kind = "video"
                mi += 1
            else:
                # Re-cut as a still and try again at the next opportunity.
                cut = _cut_index(words, i, still_target)
                end = float(words[cut]["s"]) if cut < n else span_end
                if n - cut <= 1:
                    end, cut = span_end, n
                hold = end - start

        slots.append({
            "words": cut - i,
            "kind": kind,
            "start": round(start, 3),
            "end": round(end, 3),
            "hold": round(hold, 3),
        })
        i = cut

    return slots


def refine(script_plan: dict, words: list[dict],
           still_target: float = pacing.DEFAULT_SECONDS_PER_IMAGE) -> dict:
    """Rewrite a plan's slots using measured timestamps. Returns a report.

    The plan is edited IN PLACE. Narration is re-sliced from the plan's own
    script words - never from the Whisper transcript, which is only 75-93%
    word-accurate on a script full of proper nouns and would quietly rewrite the
    prompts. Whisper supplies the clock; the script supplies the words.
    """
    by_section: dict[int, list[dict]] = {}
    for img in script_plan.get("images", []):
        by_section.setdefault(int(img["section"]), []).append(img)

    new_images: list[dict] = []
    report = {"sections": [], "motion_before": 0, "motion_after": 0}

    for sec in sorted(by_section):
        imgs = sorted(by_section[sec], key=lambda i: int(i["order"]))
        report["motion_before"] += sum(1 for i in imgs if i.get("kind") == "video")

        # The section's script words, in order, with the scene each came from.
        stream: list[tuple[int, str]] = []
        for img in imgs:
            scene = img["scene_indices"][0] if img["scene_indices"] else 0
            for w in " ".join(img["narration"]).split():
                stream.append((scene, w))

        sec_words = [w for w in words if int(w["section"]) == sec]
        n_motion = sum(1 for i in imgs if i.get("kind") == "video")

        # Positional correspondence between script words and heard words is the
        # same assumption `_exact_word_cuts` makes, and it is exact only when
        # Whisper heard at least as many words as the script has.
        #
        # WHEN IT HEARD FEWER, RESAMPLE - DO NOT GIVE UP. Keeping the old layout
        # was the safe-looking choice and it is the wrong one: the old layout is
        # exactly what is stale, so a section that most needs re-cutting would be
        # the one section left untouched. Whisper drops a word on almost any long
        # read, and a whole section's motion slots should not hinge on that.
        #
        # Stretching the heard words evenly over the scripted positions keeps the
        # clock honest to within one word gap (~0.4s at this pace), which is well
        # inside what the [3.5, 4.0] window tolerates - and the plan stage
        # measures the real holds afterwards regardless.
        if not sec_words:
            new_images.extend(imgs)
            report["sections"].append(
                {"section": sec, "refined": False, "why": "no words heard"})
            report["motion_after"] += n_motion
            continue

        if len(sec_words) < len(stream):
            n_heard, n_script = len(sec_words), len(stream)
            clock = [sec_words[min(n_heard - 1, round(i * n_heard / n_script))]
                     for i in range(n_script)]
            note = (f"heard {n_heard} words for {n_script} scripted - "
                    f"timestamps resampled")
        else:
            clock = sec_words[:len(stream)]
            note = None

        slots = layout_section(clock, n_motion, still_target=still_target)

        cursor, order = 0, 0
        for s in slots:
            piece = stream[cursor:cursor + s["words"]]
            if not piece:
                continue
            cursor += len(piece)
            order += 1
            new_images.append({
                "section": sec,
                "order": order,
                "filename": f"{order}section{sec}.png",
                "scene_indices": sorted({sc for sc, _ in piece}),
                "narration": [" ".join(w for _, w in piece)],
                "words": len(piece),
                "kind": s["kind"],
            })
        # Any words left by rounding join the last slot rather than becoming a
        # sliver that would flash.
        if cursor < len(stream) and new_images:
            tail = stream[cursor:]
            last = new_images[-1]
            last["narration"] = [last["narration"][0] + " "
                                 + " ".join(w for _, w in tail)]
            last["words"] += len(tail)
            last["scene_indices"] = sorted(set(last["scene_indices"])
                                           | {sc for sc, _ in tail})

        got = sum(1 for s in slots if s["kind"] == "video")
        report["motion_after"] += got
        row = {"section": sec, "refined": True,
               "slots": len(slots), "motion_wanted": n_motion,
               "motion_placed": got}
        if note:
            row["why"] = note          # resampled: worth saying, not a failure
        report["sections"].append(row)

    script_plan["images"] = new_images
    script_plan["refined_from_words"] = True
    report["slots"] = len(new_images)
    return report
