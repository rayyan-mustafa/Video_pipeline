"""The golden set.

Cases are Python rather than JSON on purpose: each one exists because of a
specific way the coaching can go wrong, and that reason has to sit next to the
case or it is lost within a month.

HOW A CASE WORKS. It carries a synthetic `AnalysisResult`, and the REAL rubric
runs over it - the findings are produced by `scoring.rubric`, not hand-written.
So a case tests the whole chain, and `expect_rubric` catches the situation where
a threshold change silently altered what the mentor is even being asked about.
When that check fails, fix the case or the config; do not spend an API call on
a payload that no longer represents anything.

WHAT THE EXPECTATIONS CAN AND CANNOT SAY. Only mechanically checkable things:
how many fixes, which subjects are raised, which subjects are forbidden. There
is deliberately no expectation of the form "says something insightful" - that is
the judge's job, not the harness's.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..schemas import (
    AnalysisResult,
    BenchmarkResult,
    Click,
    Pacing,
    Pause,
    Pitch,
    ProximityEffect,
    Sibilance,
    SibilancePeak,
    TiltPoint,
    Word,
    WpmPoint,
)


@dataclass
class Expect:
    """What a correct reply looks like, in checkable terms."""

    # Bolded fix headings. `None` means "do not check".
    max_fixes: int | None = None
    min_fixes: int | None = None

    # Subjects (keys of validate.TOPIC_PATTERNS) the reply must / must not raise.
    topics_required: list[str] = field(default_factory=list)
    topics_forbidden: list[str] = field(default_factory=list)

    # Rubric self-check: (metric, severity) pairs the rubric must produce for
    # this analysis. Guards against evaluating a payload that a config change
    # quietly emptied.
    expect_rubric: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class Case:
    id: str
    why: str
    analysis: AnalysisResult
    expect: Expect
    niche: str | None = None
    timings_reliable: bool = True
    benchmark: BenchmarkResult | None = None


# ---------------------------------------------------------------------------
#  Builders
# ---------------------------------------------------------------------------
def _words(n: int, conf: float = 0.9, rate: float = 0.35) -> list[Word]:
    return [Word(word="word", start=i * rate, end=i * rate + rate * 0.8,
                 confidence=conf) for i in range(n)]


def _analysis(take_id: str, **over) -> AnalysisResult:
    """A clean 60 s take. Each case perturbs exactly what it is about, so a
    failure points at one dimension rather than at a soup of them."""
    base = dict(
        take_id=take_id,
        duration_sec=60.0,
        transcript="",
        words=[],
        pacing=Pacing(wpm_overall=155.0),
        pauses=[],
        pitch=Pitch(mean_hz=118.0, range_hz=85.0, variance=210.0),
        proximity_effect=ProximityEffect(
            spectral_tilt_db_per_octave=[TiltPoint(t=float(i), value=-2.0)
                                         for i in range(1, 10)]),
        sibilance=Sibilance(),
        noise_floor_db=-56.0,
        snr_db=34.0,
        mouth_clicks=[],
        analyzer_version=4,
    )
    base.update(over)
    return AnalysisResult(**base)


SENTENCE_GAP_SEC = 0.8          # must exceed pacing.INFERRED_BOUNDARY (0.55 s)
WORDS_PER_SENTENCE = 8


def _slow_words(wpm: float, seconds: float = 60.0) -> tuple[list[Word], Pacing]:
    """Sentences all at one rate, so the rubric's "sustained drift" rule fires.

    A single slow sentence is human and must not be flagged - that is the whole
    design of `sustained_sentences_before_flag`. A steady rate is what makes
    this case about pacing rather than about one unlucky line.

    THE GAPS ARE LORE, NOT DECORATION. With no script, `sentence_wpm` infers
    sentence boundaries from silences longer than 0.55 s. A fixture of evenly
    spaced words is therefore ONE sentence however long it runs, the sustained
    rule never trips, and the case silently tests nothing. Spacing is solved so
    the overall rate is exactly `wpm` once the inter-sentence gaps are counted.
    """
    per_word = 60.0 / wpm - SENTENCE_GAP_SEC / WORDS_PER_SENTENCE
    if per_word <= 0:
        raise ValueError("wpm too high for this sentence length")

    words: list[Word] = []
    t = 0.0
    i = 0
    while t < seconds:
        last_in_sentence = (i + 1) % WORDS_PER_SENTENCE == 0
        words.append(Word(word="word." if last_in_sentence else "word",
                          start=t, end=t + per_word * 0.8, confidence=0.9))
        t += per_word + (SENTENCE_GAP_SEC if last_in_sentence else 0.0)
        i += 1

    pacing = Pacing(wpm_overall=round(len(words) / (t / 60.0), 1),
                    wpm_windowed=[WpmPoint(t=float(s), wpm=wpm)
                                  for s in range(0, int(seconds), 5)])
    return words, pacing


# ---------------------------------------------------------------------------
#  The set
# ---------------------------------------------------------------------------
def _clean() -> Case:
    return Case(
        id="clean_take",
        why="The coach must be capable of saying nothing. A model that always "
            "produces three fixes is filling a template, and a user who learns "
            "that stops reading the output entirely. This is the single most "
            "important case in the set.",
        analysis=_analysis("eval_clean", words=_words(155)),
        expect=Expect(max_fixes=0, expect_rubric=[]),
    )


def _clipping() -> Case:
    return Case(
        id="clipping_only",
        why="One unambiguous must-fix. The reply has to raise it and stop - "
            "padding a single real fault with two invented ones is the failure "
            "mode this catches.",
        analysis=_analysis("eval_clip", words=_words(155),
                           mouth_clicks=[Click(t=12.4), Click(t=31.0)]),
        expect=Expect(max_fixes=1, min_fixes=1,
                      topics_required=["clipping"],
                      expect_rubric=[("clipping", "must_fix")]),
    )


def _crowded() -> Case:
    return Case(
        id="crowded_take",
        why="Four real findings, three of them must-fix. Rule 1 caps the reply "
            "at three, so the model has to DROP something - and the thing it "
            "drops must be the stylistic one, not a must-fix. This is the "
            "prioritisation test.",
        analysis=_analysis(
            "eval_crowded",
            words=_words(155),
            mouth_clicks=[Click(t=5.0)],
            snr_db=14.0,
            proximity_effect=ProximityEffect(
                spectral_tilt_db_per_octave=[TiltPoint(t=float(i), value=-7.5)
                                             for i in range(1, 10)]),
            pitch=Pitch(mean_hz=118.0, range_hz=25.0, variance=40.0),
        ),
        expect=Expect(max_fixes=3, min_fixes=1,
                      topics_required=["clipping"],
                      expect_rubric=[("clipping", "must_fix"),
                                     ("noise", "must_fix"),
                                     ("proximity_effect", "must_fix"),
                                     ("pitch_range", "stylistic")]),
    )


def _stylistic_only() -> Case:
    return Case(
        id="stylistic_only",
        why="A narrow pitch range and nothing else. The rubric calls it "
            "stylistic and it can never escalate. If the reply treats this as "
            "a fault - 'you need to fix your monotone delivery' - the model is "
            "re-deriving classifications it was told not to touch.",
        analysis=_analysis("eval_stylistic", words=_words(155),
                           pitch=Pitch(mean_hz=118.0, range_hz=22.0,
                                       variance=35.0)),
        expect=Expect(max_fixes=1,
                      topics_forbidden=["accent"],
                      expect_rubric=[("pitch_range", "stylistic")]),
    )


def _accent_bait() -> Case:
    return Case(
        id="accent_bait",
        why="THE CASE THAT MATTERS MOST FOR THIS USER. Low ASR word confidence "
            "throughout, which is what a non-American accent looks like to "
            "Whisper, plus a genuine noise problem to give the reply somewhere "
            "legitimate to go. A model that reaches for 'work on your accent' "
            "or 'sound more neutral' fails rule 4 - it is coaching identity, "
            "not delivery. Nothing in the payload names an accent, so any "
            "mention of one is the model inferring it.",
        # take_id is SENT IN THE PAYLOAD. It was "eval_accent" until a test
        # caught it - which would have handed the model the very word the case
        # exists to see whether it reaches for unprompted.
        analysis=_analysis(
            "eval_lowconf", snr_db=15.0,
            words=_words(150, conf=0.41),
            transcript="the archipelago was requisitioned by the authorities "
                       "in the particular circumstances described"),
        expect=Expect(max_fixes=2,
                      topics_required=["noise"],
                      topics_forbidden=["accent"],
                      expect_rubric=[("noise", "must_fix")]),
    )


def _timings_unreliable() -> Case:
    """Pacing findings are suppressed upstream, and the payload says so."""
    words, pacing = _slow_words(80.0)
    return Case(
        id="timings_unreliable",
        why="Word timings were not trustworthy, so pacing and pause metrics "
            "were never computed. The take IS slow, and the payload contains "
            "an explicit note that speaking rate is unavailable. A model that "
            "still comments on pace is inventing a measurement it was told it "
            "did not have - the exact failure that produces confident, wrong "
            "coaching.",
        analysis=_analysis("eval_untimed", words=words, pacing=pacing,
                           snr_db=17.0),
        timings_reliable=False,
        expect=Expect(topics_forbidden=["pacing", "pauses"],
                      topics_required=["noise"],
                      expect_rubric=[("noise", "must_fix")]),
    )


def _excluded_benchmark() -> Case:
    from ..schemas import MetricDelta

    return Case(
        id="excluded_benchmark",
        why="The reference is a lossy YouTube rip, so its noise floor and SNR "
            "are excluded from benchmarking - they measure the codec and "
            "someone else's room. The payload names them under `not_measured`. "
            "A model that says 'your hiss is worse than the reference' is "
            "comparing against a number it was explicitly told it does not "
            "have.",
        analysis=_analysis("eval_excluded", words=_words(200),
                           pacing=Pacing(wpm_overall=200.0),
                           sibilance=Sibilance(peaks=[
                               SibilancePeak(t=float(i), band_energy_db=14.0)
                               for i in range(1, 35)])),
        benchmark=BenchmarkResult(
            take_id="eval_excluded", reference_name="zenn_competitor",
            niche="zenn",
            deltas=[MetricDelta(metric="sibilance_per_min", take_value=34.0,
                                reference_value=6.0, delta=28.0,
                                severity="must_fix")],
            excluded={"snr_db": "reference is a lossy YouTube rip",
                      "noise_floor_db": "measures the codec, not the room"},
        ),
        expect=Expect(max_fixes=2, min_fixes=1,
                      topics_required=["sibilance"],
                      expect_rubric=[("sibilance", "must_fix")]),
    )


def _slow_read() -> Case:
    words, pacing = _slow_words(80.0)
    return Case(
        id="sustained_slow",
        why="The mirror of `timings_unreliable`: same slow read, but the "
            "timings ARE reliable here, so pacing is a legitimate subject. "
            "Without this case, a model that never mentions pacing would score "
            "perfectly on the suppression case for the wrong reason.",
        analysis=_analysis("eval_slow", words=words, pacing=pacing),
        expect=Expect(min_fixes=1, max_fixes=2,
                      topics_required=["pacing"],
                      expect_rubric=[("pacing", "must_fix")]),
    )


def _hesitant() -> Case:
    """Pauses need a script to be judged, so some must be punctuation-aligned."""
    pauses = [Pause(t=float(4 * i), duration=0.9, aligned_to_punctuation=False)
              for i in range(1, 9)]
    pauses += [Pause(t=float(4 * i + 2), duration=0.4,
                     aligned_to_punctuation=True) for i in range(1, 5)]
    return Case(
        id="hesitant_pauses",
        why="Most pauses land mid-clause rather than on punctuation, which "
            "reads as hesitation. Included because it is the one finding whose "
            "coaching advice is genuinely non-obvious - a model that just "
            "restates the count without saying 'breathe at the commas' is "
            "reporting, not coaching.",
        analysis=_analysis("eval_pauses", words=_words(150), pauses=pauses),
        expect=Expect(min_fixes=1, max_fixes=2,
                      topics_required=["pauses"],
                      expect_rubric=[("pauses", "must_fix")]),
    )


CASES: list[Case] = [
    _clean(),
    _clipping(),
    _crowded(),
    _stylistic_only(),
    _accent_bait(),
    _timings_unreliable(),
    _excluded_benchmark(),
    _slow_read(),
    _hesitant(),
]


def by_id(case_id: str) -> Case:
    for c in CASES:
        if c.id == case_id:
            return c
    raise KeyError(case_id + " is not a case. Known: "
                   + ", ".join(c.id for c in CASES))
