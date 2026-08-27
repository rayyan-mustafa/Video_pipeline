"""The data contract.

`AnalysisResult` is the single JSON object every phase after Phase 1 consumes:
scoring reads it, the benchmark diffs two of them, the mentor prompt is built
from it, and the debug plots draw it. Nothing downstream touches raw audio.

Keeping that contract in one file - with validation - is what stops the later
phases from quietly disagreeing about field names.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------
#  Leaf records
# --------------------------------------------------------------------------
class Word(BaseModel):
    word: str
    start: float = Field(description="seconds from take start")
    end: float
    confidence: float = Field(ge=0.0, le=1.0)


class WpmPoint(BaseModel):
    t: float
    wpm: float


class Pause(BaseModel):
    t: float
    duration: float
    aligned_to_punctuation: bool = Field(
        description="True when this pause lands on a comma/period in the script. "
        "Unaligned pauses are the ones that read as hesitation."
    )


class TiltPoint(BaseModel):
    t: float
    value: float = Field(description="spectral tilt, dB per octave")


class SibilancePeak(BaseModel):
    t: float
    band_energy_db: float


class Click(BaseModel):
    t: float


class EmphasisFlag(BaseModel):
    word: str
    t: float
    expected_stress: bool
    actual_stress: bool


# --------------------------------------------------------------------------
#  Metric groups
# --------------------------------------------------------------------------
class Pacing(BaseModel):
    wpm_overall: float
    wpm_windowed: list[WpmPoint] = []


class Pitch(BaseModel):
    mean_hz: float
    range_hz: float
    variance: float


class ProximityEffect(BaseModel):
    spectral_tilt_db_per_octave: list[TiltPoint] = []


class Sibilance(BaseModel):
    peaks: list[SibilancePeak] = []


# --------------------------------------------------------------------------
#  The contract
# --------------------------------------------------------------------------
class AnalysisResult(BaseModel):
    """Everything Phase 1 extracts from one take. Serialize with
    `.model_dump_json(indent=2)`, reload with `AnalysisResult.model_validate_json(...)`."""

    take_id: str
    duration_sec: float
    transcript: str
    words: list[Word] = []

    pacing: Pacing
    pauses: list[Pause] = []
    pitch: Pitch
    proximity_effect: ProximityEffect
    sibilance: Sibilance

    noise_floor_db: float
    mouth_clicks: list[Click] = []
    emphasis_flags: list[EmphasisFlag] = []

    # Not in the original spec draft, but the mastering chain in Phase 4 needs
    # to know what it is working with, and the rubric needs SNR to judge whether
    # the noise floor is actually masking words.
    snr_db: float | None = None
    source_path: str | None = None

    # Which analyzer produced these numbers. Stored analyses are reused rather
    # than recomputed, so without this a fixed metric leaves wrong values in the
    # database with nothing to flag them.
    analyzer_version: int | None = None


# --------------------------------------------------------------------------
#  Phase 2 outputs
# --------------------------------------------------------------------------
Severity = Literal["ok", "stylistic", "must_fix"]


class FeatureVector(BaseModel):
    """The comparable subset of an AnalysisResult.

    Deliberately narrower than AnalysisResult: only metrics that mean something
    when compared BETWEEN TWO DIFFERENT SPEAKERS belong here. Absolute level,
    noise floor and take duration do not.
    """

    wpm_overall: float
    wpm_p10: float | None = None
    wpm_p90: float | None = None

    # Identity, not a coaching target - stored for context, never diffed as a
    # thing to fix. Telling someone to raise their pitch to match a reference
    # is asking them to change their voice, not their delivery.
    pitch_mean_hz: float
    pitch_range_hz: float

    spectral_tilt_mean: float
    sibilance_per_min: float
    pause_rate_per_min: float
    mean_pause_sec: float

    # Only computable with a script, so references (which have none) leave it
    # unset rather than reporting a fabricated zero.
    unaligned_pause_ratio: float | None = None


class ReferenceFeatures(BaseModel):
    """A competitor voice, feature-extracted once and cached."""

    name: str
    niche: str
    source_path: str
    language: str
    duration_sec: float
    features: FeatureVector
    # Recorded because it changes how the numbers should be read - a lossy
    # YouTube rip is not a clean-room measurement.
    provenance: str = "unknown"


class MetricDelta(BaseModel):
    """One metric, this take vs one reference voice."""

    metric: str
    take_value: float
    reference_value: float
    delta: float
    severity: Severity
    note: str | None = None


class BenchmarkResult(BaseModel):
    """Per-metric deltas against each competitor - deliberately NOT a single
    score. A single number hides which dimension is actually off."""

    take_id: str
    reference_name: str
    niche: str | None = None
    deltas: list[MetricDelta] = []
    # Metrics excluded from this comparison and why, so a missing row never
    # reads as "we checked and it was fine".
    excluded: dict[str, str] = {}


class Finding(BaseModel):
    """One rubric hit on a take. The mentor prompt is assembled from these."""

    metric: str
    severity: Severity
    message: str
    t: float | None = Field(
        default=None, description="where in the take, when the finding is localized"
    )


class ScoreResult(BaseModel):
    take_id: str
    findings: list[Finding] = []

    @property
    def must_fix(self) -> list[Finding]:
        return [f for f in self.findings if f.severity == "must_fix"]


# --------------------------------------------------------------------------
#  Phase 4 outputs
# --------------------------------------------------------------------------
class StageReport(BaseModel):
    """One processing stage, with what it actually changed.

    Recorded per stage so a chain that made things worse is visible rather than
    hidden inside a single before/after pair.
    """

    stage: str
    applied: bool
    detail: str | None = None
    peak_db: float | None = None
    rms_db: float | None = None


class MasteringReport(BaseModel):
    take_id: str
    source_path: str
    output_path: str

    lufs_before: float
    lufs_after: float
    target_lufs: float

    peak_db_before: float
    peak_db_after: float
    true_peak_ceiling_db: float

    noise_floor_before: float | None = None
    noise_floor_after: float | None = None

    stages: list[StageReport] = []
    warnings: list[str] = []


class TakeRanking(BaseModel):
    """One candidate take in a ranking, with the reason for its position."""

    take_id: str
    rank: int
    must_fix_count: int
    stylistic_count: int
    snr_db: float | None = None
    wpm: float | None = None
    reason: str


class RankingResult(BaseModel):
    """Ranked takes of the same line. Deliberately explains itself: an
    auto-picked winner the user cannot interrogate is not trustworthy."""

    candidates: list[TakeRanking] = []

    @property
    def winner(self) -> TakeRanking | None:
        return self.candidates[0] if self.candidates else None


# --------------------------------------------------------------------------
#  Mentor output validation
# --------------------------------------------------------------------------
class Violation(BaseModel):
    """One rule the mentor's reply broke.

    The system prompt states hard rules; nothing in the prompt makes a model
    obey them. These are the subset that can be checked mechanically against
    the reply and the payload it was given, so "is this model good enough"
    becomes a measured failure rate instead of an impression.
    """

    rule: str = Field(description="short id, e.g. invented_number")
    severity: Literal["hard", "soft"] = Field(
        description="hard = objectively checkable breach of a stated rule; "
        "soft = a heuristic on tone or shape that can produce false positives"
    )
    detail: str
    excerpt: str | None = Field(
        default=None, description="the offending span, for eyeballing"
    )


class ValidationReport(BaseModel):
    take_id: str
    model: str
    provider: str | None = None
    violations: list[Violation] = []
    checked_at: str | None = None

    @property
    def hard(self) -> list[Violation]:
        return [v for v in self.violations if v.severity == "hard"]

    @property
    def clean(self) -> bool:
        return not self.hard


# --------------------------------------------------------------------------
#  Personal baseline (calibration)
# --------------------------------------------------------------------------
class BaselineStat(BaseModel):
    """A metric summarized over the user's own calibration takes.

    Median rather than mean, and MAD rather than standard deviation: with three
    to five takes a single odd read would drag a mean anywhere, and the whole
    point of a baseline is that it describes the usual case.
    """

    median: float
    mad: float = Field(description="median absolute deviation; 0.0 with one take")
    n: int


class Baseline(BaseModel):
    """What this speaker sounds like when they are reading well.

    The rubric's absolute numbers came from US-English VO norms. For a speaker
    those norms were not written for, "outside the band" and "wrong" are not the
    same statement, and this is what lets the two be told apart.
    """

    created_at: str
    take_ids: list[str] = []
    analyzer_version: int | None = None

    wpm: BaselineStat | None = None
    wpm_p10: BaselineStat | None = None
    wpm_p90: BaselineStat | None = None
    word_confidence: BaselineStat | None = None
    pitch_mean_hz: BaselineStat | None = None
    pitch_range_hz: BaselineStat | None = None
    spectral_tilt_mean: BaselineStat | None = None
    sibilance_per_min: BaselineStat | None = None
