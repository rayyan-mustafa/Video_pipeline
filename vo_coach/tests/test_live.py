"""Phase 5: ring buffer, clause detection, throttling.

None of this can be tested by recording a voice, so the tests drive synthetic
audio with known structure - a tone burst IS a clause, a silence IS a gap - and
assert on counts and timings rather than on anything perceptual.

The throttle tests matter most. Cooldown, budget and novelty are the only
things standing between "a coach" and "a call at every clause boundary for
eighteen minutes", and all three are pure logic with no audio involved.
"""

from __future__ import annotations

import numpy as np
import pytest

from vo_coach.live.clause import FRAME_SEC, ClauseDetector
from vo_coach.live.coach import LiveCoach, build_live_payload
from vo_coach.live.monitor import SessionState
from vo_coach.live.ring import Ring
from vo_coach.schemas import Finding

SR = 48000


def tone(seconds: float, amp: float = 0.2, sr: int = SR) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * 180.0 * t)).astype(np.float32)


def hush(seconds: float, amp: float = 0.0005, sr: int = SR) -> np.ndarray:
    rng = np.random.default_rng(0)
    return (rng.normal(0, amp, int(seconds * sr))).astype(np.float32)


class TestRing:
    def test_tail_returns_the_most_recent_audio(self):
        r = Ring(seconds=2.0, sr=1000)
        r.write(np.arange(500, dtype=np.float32))
        r.write(np.arange(500, 900, dtype=np.float32))
        assert r.tail(0.2)[-1] == 899.0
        assert len(r.tail(0.2)) == 200

    def test_wraps_without_losing_order(self):
        """A wrapped read that returned the two halves the wrong way round
        would put the future before the past, and every timing would be
        nonsense."""
        r = Ring(seconds=1.0, sr=1000)
        r.write(np.arange(1500, dtype=np.float32))
        tail = r.tail(1.0)
        assert len(tail) == 1000
        assert np.array_equal(tail, np.arange(500, 1500, dtype=np.float32))

    def test_between_clamps_to_what_is_still_buffered(self):
        """Audio already overwritten must come back SHORT, not as silence -
        silence would be measured as a pause that never happened."""
        r = Ring(seconds=1.0, sr=1000)
        r.write(np.arange(3000, dtype=np.float32))
        got = r.between(0.0, 3.0)
        assert len(got) == 1000
        assert got[0] == 2000.0

    def test_a_block_larger_than_the_buffer_keeps_the_tail(self):
        r = Ring(seconds=0.5, sr=1000)
        r.write(np.arange(2000, dtype=np.float32))
        assert r.tail(0.5)[-1] == 1999.0

    def test_elapsed_counts_everything_ever_written(self):
        r = Ring(seconds=1.0, sr=1000)
        r.write(np.zeros(2500, dtype=np.float32))
        assert r.elapsed == pytest.approx(2.5)

    def test_empty_read_is_empty_not_an_error(self):
        assert len(Ring(seconds=1.0, sr=1000).tail(0.5)) == 0


class TestClauseDetection:
    def _run(self, det: ClauseDetector, *chunks) -> list:
        out = []
        for c in chunks:
            out += det.feed(c)
        return out

    def test_speech_then_silence_is_one_boundary(self):
        det = ClauseDetector(sr=SR)
        got = self._run(det, hush(1.0), tone(1.5), hush(1.0))
        assert len(got) == 1
        assert got[0].speech_sec >= 1.0

    def test_two_clauses_produce_two_boundaries(self):
        det = ClauseDetector(sr=SR)
        got = self._run(det, hush(1.0), tone(1.2), hush(0.8),
                        tone(1.2), hush(0.8))
        assert len(got) == 2

    def test_a_short_noise_is_not_a_clause(self):
        """Without min_speech, a cough between two silences reads as a clause
        and the coach interrupts someone who has not spoken yet."""
        det = ClauseDetector(sr=SR, min_speech_sec=0.7)
        got = self._run(det, hush(1.0), tone(0.2), hush(1.0))
        assert got == []

    def test_a_short_gap_does_not_split_a_clause(self):
        """Without min_silence, every inter-word gap is a boundary and coaching
        fires several times a second."""
        det = ClauseDetector(sr=SR, min_silence_sec=0.45)
        got = self._run(det, hush(1.0), tone(1.0), hush(0.15), tone(1.0),
                        hush(1.0))
        assert len(got) == 1

    def test_boundary_times_bracket_the_speech(self):
        det = ClauseDetector(sr=SR)
        got = self._run(det, hush(1.0), tone(1.5), hush(1.0))
        b = got[0]
        assert 0.8 <= b.t_start <= 1.2
        assert 2.3 <= b.t_end <= 2.8
        assert b.t_end > b.t_start

    def test_block_size_does_not_change_the_result(self):
        """The callback delivers whatever PortAudio decides, so a detector that
        only worked on frame-aligned blocks would behave differently on a
        different machine."""
        signal = np.concatenate([hush(1.0), tone(1.2), hush(1.0)])
        one = ClauseDetector(sr=SR).feed(signal)
        det = ClauseDetector(sr=SR)
        many = []
        for i in range(0, len(signal), 1237):        # deliberately unaligned
            many += det.feed(signal[i:i + 1237])
        assert len(one) == len(many) == 1
        assert one[0].t_end == pytest.approx(many[0].t_end, abs=FRAME_SEC * 2)

    def test_threshold_tracks_a_loud_room(self):
        """A fixed dBFS gate works in one room at one gain and silently fails
        everywhere else. Loud hiss must raise the floor, not be heard as
        speech."""
        quiet = ClauseDetector(sr=SR)
        quiet.feed(hush(3.0, amp=0.0005))
        loud = ClauseDetector(sr=SR)
        loud.feed(hush(3.0, amp=0.02))
        assert loud.noise_floor_db > quiet.noise_floor_db + 10

    def test_digital_silence_does_not_poison_the_floor(self):
        """One dropout frame would drag a minimum to -inf and open the gate on
        everything; the floor is a percentile for exactly this reason."""
        det = ClauseDetector(sr=SR)
        det.feed(np.concatenate([hush(2.0, amp=0.01),
                                 np.zeros(int(0.1 * SR), dtype=np.float32)]))
        assert det.noise_floor_db > -80.0


def _f(metric="noise", severity="must_fix") -> Finding:
    return Finding(metric=metric, severity=severity, message="Hiss is high. "
                   "Move closer to the mic.")


class TestThrottling:
    def test_speaks_for_a_must_fix(self):
        c = LiveCoach(cooldown_sec=10, budget=5, repeat_after_sec=60)
        assert c.pick([_f()], now=100.0) is not None

    def test_never_speaks_for_a_stylistic_finding(self):
        """Live is the wrong moment for taste: the speaker cannot act on
        'consider more pitch variation' mid-take, and it costs them the line."""
        c = LiveCoach(cooldown_sec=10, budget=5)
        assert c.pick([_f(metric="pitch_range", severity="stylistic")],
                      now=100.0) is None

    def test_cooldown_blocks_a_second_note(self):
        c = LiveCoach(cooldown_sec=20, budget=5, repeat_after_sec=0)
        c._last_call_at = 100.0
        assert c.pick([_f()], now=110.0) is None
        assert c.pick([_f()], now=121.0) is not None

    def test_the_same_metric_is_not_repeated(self):
        """THE ONE THAT MATTERS. Without it the other two limits just slow down
        the repetition, and a repeated note is how a coach gets ignored."""
        c = LiveCoach(cooldown_sec=0, budget=99, repeat_after_sec=90)
        c._last_metric_at["noise"] = 100.0
        assert c.pick([_f(metric="noise")], now=150.0) is None
        assert c.pick([_f(metric="noise")], now=200.0) is not None

    def test_a_different_metric_is_allowed_through_the_repeat_gate(self):
        c = LiveCoach(cooldown_sec=0, budget=99, repeat_after_sec=90)
        c._last_metric_at["noise"] = 100.0
        assert c.pick([_f(metric="clipping")], now=150.0) is not None

    def test_budget_stops_the_session_spending(self):
        c = LiveCoach(cooldown_sec=0, budget=2, repeat_after_sec=0)
        c.calls = 2
        assert c.pick([_f()], now=999.0) is None

    def test_local_fallback_still_marks_the_metric_as_said(self):
        """A budget-exhausted coach must not then repeat its free notes
        forever."""
        c = LiveCoach(cooldown_sec=0, budget=0, repeat_after_sec=90)
        line = c.note_local(_f(), now=100.0)
        assert line and line.endswith(".")
        assert c._last_metric_at["noise"] == 100.0


class TestLivePayload:
    def test_prompt_is_frozen(self):
        from vo_coach.live.coach import LIVE_SYSTEM_PROMPT

        assert "{" not in LIVE_SYSTEM_PROMPT and "}" not in LIVE_SYSTEM_PROMPT

    def test_prompt_forbids_numbers_and_identity(self):
        from vo_coach.live.coach import LIVE_SYSTEM_PROMPT

        low = LIVE_SYSTEM_PROMPT.lower()
        assert "never state a measurement" in low
        assert "never coach their accent" in low
        assert "nine-year-old" in low, "plain-English rule must survive edits"
        assert "sound more american" in low, "the forbidden phrasing is listed"

    def test_payload_carries_one_problem_only(self):
        import json

        p = json.loads(build_live_payload(_f(), recent_text="hello there"))
        assert p["problem"]["metric"] == "noise"
        assert p["they_just_said"] == "hello there"

    def test_previous_notes_are_included_so_it_does_not_rephrase_itself(self):
        import json

        p = json.loads(build_live_payload(_f(), already_said=["move closer"]))
        assert p["you_already_told_them"] == ["move closer"]


class TestSessionState:
    def test_wpm_uses_speech_time_not_wall_clock(self):
        """A thinking pause between clauses must not read as slowing down."""
        s = SessionState()
        s.add_clause(0.0, 4.0, "ten words here", wpm=150.0, words=10)
        s.add_clause(30.0, 34.0, "ten more words", wpm=150.0, words=10)
        assert s.session_wpm() == pytest.approx(150.0, abs=1.0)

    def test_wpm_is_none_until_there_is_enough_speech(self):
        s = SessionState()
        s.add_clause(0.0, 1.0, "hi", wpm=120.0, words=2)
        assert s.session_wpm() is None

    def test_state_dict_is_json_serialisable(self):
        import json

        s = SessionState(coach_budget=40)
        s.add_clause(0.0, 2.0, "text", 150.0, 5)
        s.add_note("move back a little")
        json.dumps(s.to_dict())

    def test_only_recent_history_is_sent(self):
        """The HUD payload must not grow over a two-hour session."""
        s = SessionState()
        for i in range(50):
            s.add_clause(float(i), float(i) + 1, "x", 150.0, 5)
            s.add_note("note")
        d = s.to_dict()
        assert len(d["clauses"]) == 8 and len(d["notes"]) == 6


class TestSessionLoop:
    """The whole worker path with synthetic audio and no microphone."""

    def _session(self, **kw):
        from vo_coach.config import load_config
        from vo_coach.live.session import LiveSession

        return LiveSession(cfg=load_config(), save=False, use_asr=False,
                           use_mentor=False, **kw)

    def _drive(self, s, chunks):
        import threading
        import time

        s._worker = threading.Thread(target=s._work, daemon=True)
        s._worker.start()
        block = int(0.05 * s.sr)
        for c in chunks:
            for i in range(0, len(c), block):
                s._on_audio(c[i:i + block], 0, None, None)
        time.sleep(1.5)
        s._stop.set()
        return s.state.to_dict()

    def test_clauses_become_state(self):
        s = self._session()
        d = self._drive(s, [hush(1.0), tone(1.3), hush(0.9), tone(1.3), hush(0.9)])
        assert len(d["clauses"]) == 2

    def test_a_real_finding_produces_a_note(self):
        """A 180 Hz tone is all low end, so the rubric sees heavy proximity
        buildup - a genuine must_fix, reached through the real rubric rather
        than a stubbed one."""
        s = self._session()
        d = self._drive(s, [hush(1.0), tone(1.3), hush(0.9)])
        assert len(d["notes"]) == 1
        text = d["notes"][0]["text"].lower()
        # An INSTRUCTION about what to do, not a readout of the tilt figure,
        # and phrased so it needs no explaining.
        assert any(w in text for w in ("mic", "back off", "close")), text
        assert "db" not in text, "live notes must never state a measurement"
        assert "tilt" not in text and "octave" not in text, "no jargon"
        assert "proximity" not in text, "no jargon"

    def test_the_cooldown_holds_across_clauses(self):
        """Two clauses with the same problem must not produce two notes."""
        s = self._session()
        d = self._drive(s, [hush(1.0), tone(1.3), hush(0.9), tone(1.3),
                            hush(0.9), tone(1.3), hush(0.9)])
        assert len(d["clauses"]) == 3
        assert len(d["notes"]) == 1

    def test_scoring_does_not_depend_on_the_meter_thread(self):
        """The finding must describe the clause that just ended. Reading the
        HUD's last sample would use whatever the meter happened to catch, or
        nothing at all at startup."""
        s = self._session()
        assert s.state.metrics.rms_db is None      # meter never started
        d = self._drive(s, [hush(1.0), tone(1.3), hush(0.9)])
        assert len(d["notes"]) == 1

    def test_state_stays_json_serialisable_through_a_real_run(self):
        import json

        s = self._session()
        json.dumps(self._drive(s, [hush(1.0), tone(1.3), hush(0.9)]))


class TestServer:
    def test_app_exposes_the_hud_and_the_socket(self):
        from vo_coach.live.server import STATIC, create_app

        app = create_app(save=False, use_asr=False, use_mentor=False)
        paths = {r.path for r in app.routes}
        assert {"/", "/ws", "/api/state", "/api/stop"} <= paths
        assert (STATIC / "index.html").exists()

    def test_the_hud_is_self_contained(self):
        """No CDN: this page has to come up on a machine that is offline
        mid-session, and a blocked request would leave a blank HUD over a live
        take."""
        from vo_coach.live.server import STATIC

        html = (STATIC / "index.html").read_text(encoding="utf-8")
        assert "http://" not in html.replace("http://127.0.0.1", "")
        assert "https://" not in html

    def test_the_websocket_actually_accepts_a_connection(self):
        """REGRESSION. This file uses `from __future__ import annotations`, so
        FastAPI resolves the handler's parameter types from strings against the
        MODULE globals. When `WebSocket` was imported inside `create_app` it
        was a local, invisible to that lookup - FastAPI read the parameter as a
        query param, failed validation, and closed before accepting.

        The symptom was a bare 403 on the upgrade with no traceback anywhere,
        and a HUD frozen at 0:00 that looked exactly like a silent microphone.
        `/api/state` kept working throughout, which made it look like a mic
        problem rather than a routing one. Asserting on route registration
        would NOT have caught it - the route existed and was correct.
        """
        from fastapi.testclient import TestClient

        from vo_coach.live.server import create_app

        app = create_app(save=False, use_asr=False, use_mentor=False)
        with TestClient(app).websocket_connect("/ws") as ws:
            frame = ws.receive_json()
        assert "metrics" in frame and "elapsed" in frame

    def test_websocket_types_are_resolvable_from_module_globals(self):
        """The above only fails on a real connection, which needs a microphone
        present. This checks the underlying condition directly, so the bug is
        caught on a machine with no audio device at all."""
        import vo_coach.live.server as S

        assert hasattr(S, "WebSocket"), (
            "WebSocket must be imported at module level: annotations are "
            "strings here and FastAPI resolves them against module globals")


class TestLocalCoachingLines:
    """The `--no-mentor` fallback has to obey the SAME rules as the live prompt.

    The first implementation truncated the rubric's own message, which produced
    "Signal-to-noise is 17 dB." on the HUD - a readout, not something a person
    can act on mid-take, and a direct violation of the live prompt's rule 3.
    """

    def _coach(self):
        return LiveCoach(cooldown_sec=0, budget=0, repeat_after_sec=0)

    @pytest.mark.parametrize("metric", ["noise", "proximity_effect", "sibilance",
                                        "clipping", "pacing", "pauses"])
    def test_every_live_metric_has_a_spoken_line(self, metric):
        line = self._coach().note_local(_f(metric=metric), now=1.0)
        assert line and line[0].isupper() and line.endswith(".")

    @pytest.mark.parametrize("metric", ["noise", "proximity_effect", "sibilance",
                                        "clipping", "pacing", "pauses"])
    def test_no_line_states_a_measurement(self, metric):
        """Rule 3 of the live prompt: they cannot act on 14 dB."""
        low = self._coach().note_local(_f(metric=metric), now=1.0).lower()
        for unit in (" db", "dbfs", "wpm", "hz", "octave", "signal-to-noise"):
            assert unit not in low, f"{metric}: {unit!r} leaked into a live note"
        assert not any(ch.isdigit() for ch in low)

    def test_repeats_are_reworded(self):
        """Identical text twice reads as a stuck program rather than a coach."""
        c = self._coach()
        first = c.note_local(_f(metric="noise"), now=1.0)
        second = c.note_local(_f(metric="noise"), now=2.0)
        assert first != second

    def test_an_unmapped_metric_still_says_something(self):
        """Better a readout than silence - but every metric that can fire live
        should be mapped, which the parametrized tests above enforce."""
        line = self._coach().note_local(_f(metric="mystery"), now=1.0)
        assert line


class TestTeleprompter:
    """The script panel. Served once, not in every state frame."""

    def _app(self, script=None):
        from vo_coach.live.server import create_app

        return create_app(save=False, use_asr=False, use_mentor=False,
                          script=script)

    def _get(self, script=None):
        from fastapi.testclient import TestClient

        return TestClient(self._app(script)).get("/api/script").json()

    def test_no_script_is_an_empty_list_not_an_error(self):
        """Live mode must work with no script at all - that is the default."""
        assert self._get() == {"lines": []}

    def test_lines_carry_their_pause_target(self):
        from vo_coach.timing import marking as M

        marked = M.render("One thing, then another. A second line here. ",
                          M.PauseStyle(sentence=0.9, clause=0.3))
        lines = self._get(marked)["lines"]
        assert lines
        assert any(l["pause"] == 0.9 for l in lines)

    def test_the_header_is_not_shown_as_script(self):
        """The reader must see their words, not the legend explaining the
        markup - the same header that once got parsed as a pause target."""
        from vo_coach.timing import marking as M

        marked = M.render("Read this line. ", M.PauseStyle())
        text = " ".join(l["text"] for l in self._get(marked)["lines"])
        assert "MARKED SCRIPT" not in text
        assert "pause style" not in text

    def test_markers_are_replaced_not_left_as_numbers(self):
        """A prompter showing '[0.81]' mid-sentence makes the reader read the
        number. The timing belongs on the badge, not in the line."""
        from vo_coach.timing import marking as M

        marked = M.render("One thing, then another. ", M.PauseStyle())
        for line in self._get(marked)["lines"]:
            assert "[0." not in line["text"]

    def test_script_is_not_in_the_state_frame(self):
        """Sent once on load. Repeating it four times a second would grow the
        HUD payload with the length of the read."""
        from vo_coach.timing import marking as M

        marked = M.render("A line here. ", M.PauseStyle())
        from fastapi.testclient import TestClient

        state = TestClient(self._app(marked)).get("/api/state").json()
        assert "lines" not in state and "script" not in state


def _script_lines(n=3, pause=0.8):
    return [{"text": f"line {i}", "pause": pause} for i in range(n)]


class TestLineApproval:
    """The prompter advances because a line was GOOD, not because it ended."""

    def _state(self, n=3):
        s = SessionState()
        s.load_script(_script_lines(n))
        return s

    def test_first_line_starts_current(self):
        s = self._state()
        assert s.lines[0]["state"] == "current"
        assert s.lines[1]["state"] == "pending"
        assert s.current_line == 0

    def test_a_good_take_advances(self):
        s = self._state()
        s.record_attempt(s.add_clause(0.0, 2.0, "", None, 0), passed=True)
        assert s.lines[0]["state"] == "done"
        assert s.current_line == 1
        assert s.lines[1]["state"] == "current"

    def test_a_bad_take_does_not_advance(self):
        """THE POINT OF THE FEATURE. A line the coach objected to must stay lit,
        or the reader is being asked to redo something that has scrolled away."""
        s = self._state()
        s.record_attempt(s.add_clause(0.0, 2.0, "", None, 0), passed=False)
        assert s.lines[0]["state"] == "retake"
        assert s.current_line == 0

    def test_retry_after_a_failure_advances(self):
        s = self._state()
        s.record_attempt(s.add_clause(0.0, 2.0, "", None, 0), passed=False)
        s.record_attempt(s.add_clause(3.0, 5.0, "", None, 0), passed=True)
        assert s.lines[0]["state"] == "done"
        assert s.current_line == 1
        assert len(s.lines[0]["attempts"]) == 2

    def test_every_attempt_is_kept_with_its_time_range(self):
        """The export needs the ranges to cut the keeper out of the raw file."""
        s = self._state()
        s.record_attempt(s.add_clause(0.0, 2.0, "", None, 0), passed=False)
        s.record_attempt(s.add_clause(3.0, 5.0, "", None, 0), passed=True)
        a = s.lines[0]["attempts"]
        assert [x["passed"] for x in a] == [False, True]
        assert a[1]["t_start"] == 3.0 and a[1]["t_end"] == 5.0

    def test_script_done_after_the_last_line(self):
        s = self._state(n=2)
        for t in (0.0, 3.0):
            s.record_attempt(s.add_clause(t, t + 2, "", None, 0), passed=True)
        assert s.script_done is True

    def test_audio_after_the_script_is_not_an_attempt(self):
        """Tail audio must not be recorded against a line that is already
        finished, or the export would replace a good take with a cough."""
        s = self._state(n=1)
        s.record_attempt(s.add_clause(0.0, 2.0, "", None, 0), passed=True)
        s.record_attempt(s.add_clause(5.0, 6.0, "", None, 0), passed=True)
        assert len(s.lines[0]["attempts"]) == 1

    def test_no_script_means_no_line_tracking(self):
        """Without a script the raw recording IS the deliverable."""
        s = SessionState()
        s.record_attempt(s.add_clause(0.0, 2.0, "", None, 0), passed=False)
        assert s.lines == [] and s.script_done is False

    def test_state_dict_omits_attempts(self):
        """Attempts grow with every retake; the HUD only needs the state."""
        s = self._state()
        s.record_attempt(s.add_clause(0.0, 2.0, "", None, 0), passed=False)
        d = s.to_dict()
        assert "attempts" not in d["lines"][0]
        assert d["lines"][0]["state"] == "retake"


class TestComp:
    """Assembling the approved takes."""

    def _wav(self, tmp_path):
        import soundfile as sf

        # Three 1 s "lines" separated by 1 s of room tone.
        parts = []
        for i in range(6):
            parts.append(tone(1.0) if i % 2 == 0 else hush(1.0))
        x = np.concatenate(parts).astype(np.float32)
        p = tmp_path / "raw.wav"
        sf.write(str(p), x, SR, subtype="PCM_16")
        return p

    def test_keeps_only_the_last_passing_attempt(self):
        """Between two good takes the LATER one is what the speaker meant -
        they stopped there."""
        from vo_coach.live.comp import build

        import tempfile
        with tempfile.TemporaryDirectory() as d:
            from pathlib import Path
            wav = self._wav(Path(d))
            lines = [{"i": 0, "pause": 0.0, "attempts": [
                {"t_start": 0.0, "t_end": 1.0, "passed": False},
                {"t_start": 2.0, "t_end": 3.0, "passed": True},
                {"t_start": 4.0, "t_end": 5.0, "passed": True},
            ]}]
            res = build(wav, lines)
            assert res.lines_used == 1
            assert res.attempts_dropped == 2

    def test_a_line_never_accepted_is_reported_not_faked(self):
        """Silently emitting a failed take as if it were approved would be the
        worst possible failure here - the whole point was that it was rejected."""
        from pathlib import Path
        import tempfile

        from vo_coach.live.comp import build

        with tempfile.TemporaryDirectory() as d:
            wav = self._wav(Path(d))
            lines = [{"i": 0, "pause": 0.0, "attempts": [
                {"t_start": 0.0, "t_end": 1.0, "passed": False}]}]
            res = build(wav, lines)
            assert res.lines_missing == 1 and res.missing == [0]
            assert res.path is None

    def test_the_raw_file_is_never_overwritten(self):
        """A comp built from a misfiring rubric must always be recoverable."""
        from pathlib import Path
        import tempfile

        from vo_coach.live.comp import build

        with tempfile.TemporaryDirectory() as d:
            wav = self._wav(Path(d))
            before = wav.read_bytes()
            lines = [{"i": 0, "pause": 0.0, "attempts": [
                {"t_start": 0.0, "t_end": 1.0, "passed": True}]}]
            res = build(wav, lines)
            assert wav.read_bytes() == before
            assert res.path != str(wav)

    def test_gaps_come_from_the_script_targets(self):
        from pathlib import Path
        import tempfile

        import soundfile as sf

        from vo_coach.live.comp import build

        with tempfile.TemporaryDirectory() as d:
            wav = self._wav(Path(d))
            lines = [
                {"i": 0, "pause": 1.5, "attempts": [
                    {"t_start": 0.0, "t_end": 1.0, "passed": True}]},
                {"i": 1, "pause": 0.0, "attempts": [
                    {"t_start": 2.0, "t_end": 3.0, "passed": True}]},
            ]
            res = build(wav, lines)
            y, sr = sf.read(res.path, dtype="float32")
            # two ~1.24s pieces (1s + 2*0.12 pad) plus the 1.5s scripted gap
            assert len(y) / sr == pytest.approx(1.24 * 2 + 1.5, abs=0.15)

    def test_no_script_assembles_nothing(self):
        from pathlib import Path
        import tempfile

        from vo_coach.live.comp import build

        with tempfile.TemporaryDirectory() as d:
            res = build(self._wav(Path(d)), [])
            assert res.path is None and "nothing to assemble" in res.note


class TestLineStatesOverTheWire:
    def _app(self, script):
        from vo_coach.live.server import create_app

        return create_app(save=False, use_asr=False, use_mentor=False,
                          script=script)

    def test_line_states_reach_the_hud(self):
        """The prompter is driven by the SERVER's line states, so they have to
        be in the state frame - the HUD must not re-derive which line you are on
        from a clause count, which is what advanced past a failed line."""
        from fastapi.testclient import TestClient

        from vo_coach.timing import marking as M

        marked = M.render("First line here. Second line here. ", M.PauseStyle())
        app = self._app(marked)
        with TestClient(app).websocket_connect("/ws") as ws:
            d = ws.receive_json()
        assert [l["state"] for l in d["lines"]][0] == "current"
        assert d["current_line"] == 0 and d["script_done"] is False

    def test_stop_reports_where_the_audio_went(self):
        """After the last line the reader needs to be told which file is the
        deliverable, not left to guess between the raw take and the comp."""
        from fastapi.testclient import TestClient

        from vo_coach.timing import marking as M

        app = self._app(M.render("A line. ", M.PauseStyle()))
        client = TestClient(app)
        with client.websocket_connect("/ws"):
            pass
        d = client.post("/api/stop").json()
        assert d["stopped"] is True
        assert "comped" in d and "raw" in d


class TestBlockingVsCoaching:
    """Which faults stop a take, and which are only mentioned.

    This is the difference between a coach and an obstacle. When every must_fix
    blocked, a real session measured 51 sibilants/min against a 30 threshold and
    the reader could never leave line one - for something the de-esser removes
    two steps later.
    """

    def _session(self, blocking=("clipping",)):
        from vo_coach.config import load_config
        from vo_coach.live.session import LiveSession

        s = LiveSession(cfg=load_config(), save=False, use_asr=False,
                        use_mentor=False)
        s.blocking_metrics = set(blocking)
        s.state.load_script(_script_lines(3))
        return s

    def test_clipping_blocks_the_line(self):
        """No mastering recovers a flat-topped waveform."""
        s = self._session()
        s.state.record_attempt(s.state.add_clause(0, 2, "", None, 0),
                               passed=not ("clipping" in s.blocking_metrics))
        assert s.state.lines[0]["state"] == "retake"

    def test_sibilance_does_not_block_the_line(self):
        s = self._session()
        blocking = "sibilance" in s.blocking_metrics
        assert blocking is False
        s.state.record_attempt(s.state.add_clause(0, 2, "", None, 0),
                               passed=not blocking)
        assert s.state.lines[0]["state"] == "done"
        assert s.state.current_line == 1

    def test_default_config_blocks_only_clipping(self):
        from vo_coach.config import load_config

        assert load_config().get("live.blocking_metrics") == ["clipping"]


class TestClippingDetection:
    """A run of samples at full scale, not one sample."""

    def _measure(self, x):
        from vo_coach.analysis.audio import Audio
        from vo_coach.config import load_config
        from vo_coach.live.monitor import measure

        return measure(Audio(samples=x.astype(np.float32), sr=SR),
                       load_config(), 4.0)

    def test_a_single_full_scale_sample_is_not_clipping(self):
        """REGRESSION. One half-millisecond tick on a plosive lit the red light
        for four seconds AND failed the line with it. Measured on a real
        session: 0.03% of samples touched full scale, but 22% of windows
        contained one."""
        x = tone(4.0, amp=0.4)
        x[SR] = 1.0
        m = self._measure(x)
        assert m.peak_hot is True, "the peak still touched full scale"
        assert m.clipping is False, "but one sample is not a ruined take"

    def test_sustained_full_scale_is_clipping(self):
        x = np.clip(tone(4.0, amp=2.0), -1.0, 1.0)
        m = self._measure(x)
        assert m.clipping is True
        assert m.clip_runs >= 3

    def test_clean_audio_is_neither(self):
        m = self._measure(tone(4.0, amp=0.3))
        assert m.clipping is False and m.peak_hot is False


class TestActionAmounts:
    """An action quantity is not a measurement.

    Rule 3 forbids "you are at 14 dB SNR" because nobody can act on it. But the
    SIZE of a correction is the most useful thing a coach adds - without it the
    note is "you are too loud" and the reader guesses whether that is a nudge or
    a rethink.
    """

    def _session(self, **metrics):
        from vo_coach.config import load_config
        from vo_coach.live.monitor import LiveMetrics
        from vo_coach.live.session import LiveSession

        s = LiveSession(cfg=load_config(), save=False, use_asr=False,
                        use_mentor=False)
        s.state.metrics = LiveMetrics(**metrics)
        return s

    def test_clipping_amount_scales_with_how_hot_it_is(self):
        """Bigger overshoot, bigger correction - and said in words, because a
        decibel is not a thing anyone can picture."""
        hot = self._session(peak_db=0.0)._amount_for(_f(metric="clipping"))
        mild = self._session(peak_db=-6.0)._amount_for(_f(metric="clipping"))
        assert "half" in hot
        assert mild and "nudge" in mild
        assert "dB" not in hot and "dB" not in mild

    def test_no_amount_ever_states_a_unit(self):
        """The whole point of the rewrite: these reach a reader mid-take."""
        for peak in (0.0, -3.0, -5.0, -6.5):
            a = self._session(peak_db=peak)._amount_for(_f(metric="clipping"))
            if a:
                assert "dB" not in a and not any(c.isdigit() for c in a)

    def test_no_amount_when_it_cannot_be_inferred(self):
        """A vague instruction beats a wrong number."""
        s = self._session()
        assert s._amount_for(_f(metric="sibilance")) is None
        assert s._amount_for(_f(metric="clipping")) is None

    def test_the_amount_is_appended_to_the_line(self):
        c = LiveCoach(cooldown_sec=0, budget=0, repeat_after_sec=0)
        line = c.note_local(_f(metric="clipping"), now=1.0,
                            amount="about 9 dB should do it")
        assert line.endswith("about 9 dB should do it.")

    def test_lines_still_end_cleanly_without_an_amount(self):
        c = LiveCoach(cooldown_sec=0, budget=0, repeat_after_sec=0)
        assert c.note_local(_f(metric="sibilance"), now=1.0).endswith(".")


class TestCoachSpeaksPlainly:
    """The coach has one job beyond being right: being understood, mid-take, by
    someone who is not a sound engineer and is not a native speaker.

    These guard the two commitments that are easiest to erode by accident - no
    jargon, and never treating how someone speaks as a fault.
    """

    JARGON = ("db", "dbfs", "lufs", "hz", "khz", "octave", "sibilan",
              "proximity", "snr", "signal-to-noise", "spectral", "tilt",
              "amplitude", "waveform", "gain staging", "headroom", "wpm",
              "compressor", "de-esser", "normalis", "transient", "clip ")

    FORBIDDEN = ("neutralise", "neutralize", "reduce your accent",
                 "sound more american", "pronounce it properly",
                 "correct your", "improper", "wrong accent")

    def _lines(self):
        c = LiveCoach(cooldown_sec=0, budget=0, repeat_after_sec=0)
        out = []
        for metric in c.LOCAL_LINES:
            for _ in range(2):        # both phrasings
                out.append(c.note_local(_f(metric=metric), now=1.0))
        return out

    def test_no_local_line_uses_jargon(self):
        for line in self._lines():
            low = line.lower()
            for j in self.JARGON:
                assert j not in low, f"{j!r} in: {line}"

    def test_no_local_line_states_a_number(self):
        for line in self._lines():
            assert not any(ch.isdigit() for ch in line), line

    def test_no_local_line_touches_identity(self):
        for line in self._lines():
            low = line.lower()
            for bad in self.FORBIDDEN:
                assert bad not in low, f"{bad!r} in: {line}"
            assert "accent" not in low

    def test_lines_explain_by_comparison_not_by_term(self):
        """The clipping and proximity notes are the two most technical ideas
        here, so both must arrive as something the reader can picture."""
        c = LiveCoach(cooldown_sec=0, budget=0, repeat_after_sec=0)
        clip = c.note_local(_f(metric="clipping"), now=1.0).lower()
        prox = c.note_local(_f(metric="proximity_effect"), now=1.0).lower()
        assert "speaker turned up too far" in clip or "crackly" in clip
        assert "boomy" in prox or "box" in prox

    def test_the_prompt_forbids_making_them_sound_american(self):
        """The stated goal is being easy to listen to, not sounding like
        someone else. This is the line the whole project is built on."""
        from vo_coach.live.coach import LIVE_SYSTEM_PROMPT

        low = LIVE_SYSTEM_PROMPT.lower()
        assert "not trying to sound" in low
        assert "easy and pleasant to listen to" in low
        assert "never use a technical term" in low
