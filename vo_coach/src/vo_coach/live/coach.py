"""The live mentor: one line, at a clause boundary, only when it earns it.

A SEPARATE PROMPT, NOT A SHORTER REQUEST. The Phase 3 prompt asks for an
opening read plus up to three paragraphs. Asking that prompt for one sentence
produces a compressed version of a report. Live coaching is a different job -
one correction, said the way someone in the room would say it, while the
speaker still remembers the line. It gets its own frozen prompt, and the same
no-interpolation discipline so caching still works.

THROTTLING IS THE FEATURE, NOT AN OPTIMIZATION. Three separate limits, because
they fail for three different reasons:

  cooldown     stops the coach talking over itself. A note every four seconds
               is not coaching, it is heckling.
  budget       stops an 18-minute script becoming hundreds of calls. Bounded
               spend per session, decided before the session starts.
  novelty      stops the same note being repeated. A speaker who has been told
               about sibilance does not need telling again ninety seconds
               later; repetition is how a coach gets ignored.

Novelty is the one that actually matters. Without it the other two just slow
down the repetition.

WHAT IT IS ALLOWED TO SAY. Only findings the rubric already produced, at
must_fix severity. Live is the wrong moment for a stylistic note - the speaker
cannot act on "consider more pitch variation" mid-take, and it costs them the
line. Stylistic findings are saved for the post-take `coach` pass.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import ClassVar

from ..config import Config, load_config
from ..schemas import Finding

# ---------------------------------------------------------------------------
#  FROZEN - no interpolation, so the cached prefix stays byte-stable.
# ---------------------------------------------------------------------------
LIVE_SYSTEM_PROMPT = """You are a warm, experienced voice coach sitting in the room with someone recording a voiceover right now. You speak between sentences, never during one, and you get one or two short lines.

WHO YOU ARE TALKING TO
Someone who speaks English as a second language and wants listeners in the US to enjoy their voice and stay with them. THEY ARE NOT TRYING TO SOUND AMERICAN, and you are not trying to make them. Plenty of the most listened-to voices in the world are not American. Your job is to help them be EASY AND PLEASANT TO LISTEN TO, in their own voice.

HOW YOU TALK
Plain, everyday words. A nine-year-old should understand every sentence you say. Never use a technical term. Say "the room noise is louder than your voice", not "your signal-to-noise is low". Say "you are too close to the mic and it is making you sound boomy", not "proximity effect". If you catch yourself about to say a piece of jargon, say what it SOUNDS LIKE instead.

WHAT GOOD COACHING SOUNDS LIKE
Show them how to say it, do not just name the problem. A real coach says:
  "Lean on the word BRIGHT, then let the rest of the line fall away."
  "That word wants two beats, not one. Try: pa-TIENT."
  "You are pushing. Say it like you are telling one friend, not a room."
  "Smile a little on that line and it will lift on its own."
Give them the SHAPE of the line - which word carries it, where to slow down, where to let go.

RULES
1. One or two short sentences. They are mid-take, not in a classroom.
2. Say what to DO on the next line, in words they can act on immediately.
3. Never state a measurement. "About four steps back from the mic" is fine; "14 dB" is not.
4. NEVER coach their accent, their pitch, or how their voice naturally sounds. That is who they are. Rhythm, emphasis, energy and speed are fair game; the sound of the voice is not.
5. Be kind and be specific. You are on their side. If something was good, you may say so in a handful of words before the fix - but never invent praise.
6. If they must stop and fix something before continuing - the sound is distorting, the mic has fallen over - say that plainly so they know to stop.

NEVER SAY
"Neutralise", "reduce your accent", "sound more American", "pronounce it properly", or anything that treats how they speak as a fault to be corrected.

TONE
Calm, warm, encouraging. Like a good teacher who has all the time in the world for them. Second person, present tense.
"""


def build_live_payload(finding: Finding, recent_text: str = "",
                       already_said: list[str] | None = None) -> str:
    """One finding, the clause it happened in, and what has already been said.

    `already_said` exists so the model does not rephrase its own previous note
    when the same problem persists across two boundaries. The novelty gate below
    should stop that from happening at all - this is the second line of defence.
    """
    payload: dict = {
        "problem": {"metric": finding.metric, "detail": finding.message},
    }
    if recent_text:
        payload["they_just_said"] = recent_text[:200]
    if already_said:
        payload["you_already_told_them"] = already_said[-3:]
    return json.dumps(payload, indent=2, ensure_ascii=False)


@dataclass
class LiveCoach:
    """Decides whether to speak, then speaks. Stateful across a session."""

    cfg: Config = field(default_factory=load_config)
    cooldown_sec: float = 20.0
    budget: int = 40
    repeat_after_sec: float = 90.0

    calls: int = 0
    _last_call_at: float = field(default=0.0, init=False)
    _last_metric_at: dict[str, float] = field(default_factory=dict, init=False)
    _said: list[str] = field(default_factory=list, init=False)
    _local_counts: dict[str, int] = field(default_factory=dict, init=False)

    @classmethod
    def from_config(cls, cfg: Config | None = None) -> "LiveCoach":
        cfg = cfg or load_config()
        return cls(
            cfg=cfg,
            cooldown_sec=float(cfg.get("live.coach_cooldown_sec", 20.0)),
            budget=int(cfg.get("live.coach_budget", 40)),
            repeat_after_sec=float(cfg.get("live.repeat_after_sec", 90.0)),
        )

    # -----------------------------------------------------------------------
    def pick(self, findings: list[Finding], now: float) -> Finding | None:
        """The one finding worth interrupting for, or None.

        Returns None far more often than it returns a finding, and that is the
        intended behaviour. Silence is the default state of a good coach.
        """
        if self.calls >= self.budget:
            return None
        if now - self._last_call_at < self.cooldown_sec:
            return None

        for f in findings:
            if f.severity != "must_fix":
                continue                     # live is the wrong moment for taste
            last = self._last_metric_at.get(f.metric)
            if last is not None and now - last < self.repeat_after_sec:
                continue                     # already said, recently
            return f
        return None

    def speak(self, finding: Finding, recent_text: str = "",
              now: float | None = None) -> tuple[str, dict]:
        """Call the mentor for one line. Raises MentorError like Phase 3 does."""
        from ..mentor.providers import ProviderError, get_provider

        now = now if now is not None else time.time()
        payload = build_live_payload(finding, recent_text, self._said)

        provider = get_provider(self.cfg)
        try:
            text, meta = provider.complete(LIVE_SYSTEM_PROMPT, payload, self.cfg)
        except ProviderError as e:
            from ..mentor.client import MentorError

            raise MentorError(str(e)) from e

        self.calls += 1
        self._last_call_at = now
        self._last_metric_at[finding.metric] = now
        self._said.append(text)
        return text, meta

    # -----------------------------------------------------------------------
    # -----------------------------------------------------------------------
    #  Written to the SAME rules as the live system prompt: one line, an
    #  instruction rather than a measurement, no numbers, nothing about the
    #  voice itself. Truncating the rubric's own message was the obvious
    #  implementation and it was wrong - the rubric writes for READING, so the
    #  fallback came out as "Signal-to-noise is 17 dB", which is a readout, not
    #  something a person can act on mid-take.
    #
    #  Two variants each, alternated, so a repeated problem does not repeat
    #  word for word - identical text twice reads as a stuck program rather
    #  than a coach.
    # -----------------------------------------------------------------------
    #  ClassVar, not a field: this is shared constant text, and a dataclass
    #  would otherwise reject a mutable default.
    LOCAL_LINES: ClassVar[dict[str, tuple[str, ...]]] = {
        "noise": (
            "The room is a bit louder than your voice - get closer to the mic, "
            "about a hand's width away",
            "I can hear the room more than I want to. Move in closer and speak "
            "a little more into it",
        ),
        "proximity_effect": (
            "You are very close and it is making you sound boomy, like talking "
            "into a box. Back off a little",
            "Ease back from the mic and talk just past it, not straight into "
            "it - that thickness will clear up",
        ),
        "sibilance": (
            "Your S sounds are getting sharp. Turn the mic a little to one "
            "side so they slide past it instead of straight in",
            "The esses are hissing a bit. Angle your mouth slightly off the "
            "mic and they will soften",
        ),
        "clipping": (
            "Stop for a second - the sound is breaking up, like a speaker "
            "turned up too far. Turn the input volume down and take the line "
            "again",
            "That came out crackly because it is too loud going in. Lower the "
            "input volume, then pick it up from the same line",
        ),
        "pacing": (
            "Let each sentence finish before you start the next one. Give the "
            "last word a moment to land",
            "You are a touch quick. Take the corners a little slower and let "
            "the ends of lines breathe",
        ),
        "pauses": (
            "Take your breath where the commas are, not in the middle of a "
            "phrase - it will sound much more settled",
            "Try to breathe at the punctuation. Breaking mid-thought makes it "
            "sound like you are unsure",
        ),
    }

    def note_local(self, finding: Finding, now: float,
                   amount: str | None = None) -> str:
        """The zero-cost fallback, used with `--no-mentor`, a spent budget, or
        an unreachable API.

        A live coach that goes silent because of a billing limit is worse than
        one that says something plain - but "plain" has to mean plain SPEECH,
        not a truncated report.
        """
        self._last_metric_at[finding.metric] = now
        self._last_call_at = now

        lines = self.LOCAL_LINES.get(finding.metric)
        if lines:
            lines = tuple(l if l.endswith(".") or amount else l + "."
                          for l in lines)
        if not lines:
            # An unmapped metric falls back to the rubric's wording rather than
            # saying nothing. Better a readout than silence, but every metric
            # that actually fires live should be mapped above.
            first = finding.message.split(". ")[0].strip()
            return first if first.endswith(".") else first + "."

        n = self._local_counts.get(finding.metric, 0)
        self._local_counts[finding.metric] = n + 1
        line = lines[n % len(lines)]

        # AN ACTION QUANTITY IS NOT A MEASUREMENT.
        #
        # Rule 3 forbids stating a number because the speaker cannot act on
        # "14 dB SNR". But "turn it down about 6 dB" is not a readout - it is
        # the size of the correction, which is the single most useful thing a
        # coach can add. Without it the note is "you are too loud", and the
        # reader has to guess whether that means a nudge or a rethink.
        if amount:
            line = line.rstrip(".") + " - " + amount + "."
        return line
