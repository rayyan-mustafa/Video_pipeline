"""
Score a candidate voice sample before it earns a place in voice_samples.txt.

    python check_voice_sample.py mysample.txt
    python check_voice_sample.py voice_samples.txt --all
    python check_voice_sample.py mysample.txt --judge     # costs one model call

=============================================================================
 WHY THIS EXISTS
=============================================================================
voice_samples.txt is the one input to the humanizer that a model must not
write. A machine-written sample does not merely fail to help: it hands the
writer the exact register humanize.py exists to remove, carrying the gate's own
authority. Everything downstream then measures the script against the thing it
was built to catch.

You cannot tell by reading. Prose that reads well is exactly what a model
produces. So count instead.

=============================================================================
 WHY THIS DOES NOT REUSE humanize.run_audits DIRECTLY
=============================================================================
Those quotas are ABSOLUTE budgets tuned to a ~1,000-word script: "at most one
tricolon in the whole thing". Run them on a 76-word paragraph and everything
passes trivially, because one tricolon is inside the budget no matter how short
the passage is.

Measured, for real: vo_coach/data/scripts/mic_test_zenn.txt clears every single
audit. It is 76 words long and contains "Time slips, stretches, and settles" and
"has simply been consequence" - a tricolon rate of 13 per 1,000 words and a
cushion rate of 13 per 1,000, both roughly 13x the human band. The gate was
right; the scale was wrong.

So this scores RATES per 1,000 words, and refuses to judge anything under
MIN_WORDS at all.

=============================================================================
 THE BANDS
=============================================================================
The upper bounds below are the humanizer's own per-script quotas re-expressed as
rates against a 1,000-word script, which is what those numbers were tuned to. A
sample is not required to be exceptional. It is required not to be machine-shaped.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from pipeline import humanize  # noqa: E402

MIN_WORDS = 250   # below this the rates are noise - one tricolon in 80 words is
                  # a 12/1000 rate off a single occurrence. Refuse, do not guess.

# name -> (counter, per-1000 ceiling, what a breach means)
BANDS = {
    "em dashes":        (lambda t, s: sum(1 for x in s if "—" in x or " -- " in x), 3.0,
                         "the statement-then-reveal construction"),
    "tricolons":        (lambda t, s: len(re.findall(
                            r"\b[\w'-]+,\s+[\w'-]+(?:\s+[\w'-]+)?,\s+and\s+[\w'-]+", t, re.I)), 1.0,
                         "the rule of three as a rhythm"),
    "not-X-but-Y":      (lambda t, s: len(re.findall(
                            r"\b(?:not just|not merely|nothing more than|wasn't just|isn't just)\b",
                            t, re.I)), 2.0,
                         "antithesis as a template"),
    "cushion words":    (lambda t, s: sum(1 for w in re.findall(r"[a-z']+", t.lower())
                                          if w in humanize.CUSHION_WORDS), 6.0,
                         "just / simply / really / actually / quietly"),
    "signpost opens":   (lambda t, s: sum(1 for x in s if x.split()
                                          and x.split()[0].lower().strip(",.'")
                                          in humanize.SIGNPOST_OPENERS), 60.0,
                         "every sentence announcing its own turn"),
}


def sentences(text: str) -> list[str]:
    text = " ".join(text.split())
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def score(text: str, label: str = "") -> bool:
    sents = sentences(text)
    words = len(text.split())
    print("=" * 70)
    print(f"{label or 'sample'}: {words} words, {len(sents)} sentences")
    print("=" * 70)

    if words < MIN_WORDS:
        print(f"  TOO SHORT to judge. Need {MIN_WORDS}+ words; this is {words}.")
        print("  Rates off two or three occurrences say nothing. Get more of the")
        print("  same writer's prose - not a different writer's - and re-run.")
        return False

    failed = []
    for name, (count, ceiling, meaning) in BANDS.items():
        n = count(text, sents)
        rate = n * 1000 / words
        bad = rate > ceiling
        if bad:
            failed.append((name, meaning))
        print(f"  {name:<16} {n:>3}  = {rate:>5.1f} /1000   "
              f"(ceiling {ceiling:>4.1f})  {'OVER' if bad else 'ok'}")

    # Construction variety: the deep one. A human writer rotates sentence shapes;
    # a model varies length inside one shape.
    shapes = [humanize._shape(s) for s in sents]
    from collections import Counter
    top, n = Counter(shapes).most_common(1)[0]
    ratio = n / len(sents)
    shape_bad = top != "plain" and ratio > 0.40
    print(f"  {'top shape':<16} {top:>3.3}  = {ratio:>4.0%} of sentences        "
          f"(ceiling  40%)  {'OVER' if shape_bad else 'ok'}")
    if shape_bad:
        failed.append(("construction variety", f"{ratio:.0%} of it is the '{top}' shape"))

    print()
    if failed:
        print("  REJECT. This sample is machine-shaped:")
        for name, meaning in failed:
            print(f"    - {name}: {meaning}")
        print("\n  Seeding the humanizer with this teaches the writer to produce it.")
        return False

    print("  ACCEPT on the counters.")
    print("  The counters cannot check provenance. Confirm one of these yourself:")
    print("    - it was published before ~2021, or")
    print("    - you wrote it, or")
    print("    - you watched a person write or say it.")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("path", help="a text file holding one candidate sample")
    ap.add_argument("--all", action="store_true",
                    help="treat the file as voice_samples.txt and score each --- block separately")
    ap.add_argument("--judge", action="store_true",
                    help="also ask the model for a 0-100 machine score (one API call)")
    args = ap.parse_args()

    text = Path(args.path).read_text(encoding="utf-8")
    blocks = ([b.strip() for b in re.split(r"^\s*-{3,}\s*$", text, flags=re.M) if b.strip()]
              if args.all else [text])

    results = [score(b, f"{args.path} [{i+1}/{len(blocks)}]" if len(blocks) > 1 else args.path)
               for i, b in enumerate(blocks)]

    if args.judge:
        # Deliberately opt-in. The counters are free, deterministic and catch most
        # of it; the judge is a paid second opinion, and asking a model whether
        # prose is machine written is exactly the question models are worst at
        # when the answer is "no". Treat a low score as weak evidence, a high one
        # as strong.
        import os
        if not os.environ.get("OPENROUTER_API_KEY"):
            print("\n  --judge needs OPENROUTER_API_KEY. Skipping.")
        else:
            from pipeline import config
            from pipeline.engine import call_model, parse_json_response
            for i, b in enumerate(blocks):
                verdict = parse_json_response(call_model(
                    config.MODEL_JUDGE,
                    'Score this prose 0-100, where 0 is unmistakably written by a person and '
                    '100 is unmistakably written by a language model. Name the construction it '
                    'leans on hardest. Respond with ONLY valid JSON, no fences: '
                    '{"machine_score": <0-100>, "worst_habit": "...", "why": "one sentence"}'
                    f"\n\nPROSE:\n{b}",
                    max_tokens=500, temperature=0.2))
                print(f"\n  judge [{i+1}]: {verdict.get('machine_score')}/100 - "
                      f"{verdict.get('worst_habit')}")
                print(f"    {verdict.get('why')}")

    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
