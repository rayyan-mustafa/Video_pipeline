"""The live session: capture -> clause -> ASR -> rubric -> coach.

THE AUDIO IS CAPTURED BY THIS PROCESS, NOT BY THE BROWSER. The HUD is a viewer;
it receives state over a WebSocket and sends nothing back but commands. That is
a deliberate departure from "stream the mic through the browser", for one
reason that outweighs the rest: the browser would capture through WebAudio at
whatever rate it likes and resample in JavaScript, so live and offline metrics
would be measured on two different signals. When they then disagreed - and they
would - there would be no way to tell which was right. Capturing here means the
live path and `record()` are the same path.

It also means no microphone permission prompt, no binary WebSocket framing, and
the HUD is a plain page that can be opened from any browser on the machine.

THREE THREADS, ONE DIRECTION OF TRAVEL:

  PortAudio callback   writes to the ring, writes to the wav, feeds the clause
                       detector. Never blocks, never calls the network.
  worker thread        transcribes finished clauses and runs the rubric. ASR
                       takes hundreds of milliseconds, which would drop audio
                       frames if it ran in the callback.
  asyncio server       reads state, pushes JSON to the HUD.

State is mutated only by the worker and read by the server, so the HUD can lag
a frame behind without any locking beyond the ring's own.
"""

from __future__ import annotations

import datetime as _dt
import queue
import threading
import time
from pathlib import Path

import numpy as np

from ..config import TAKES_DIR, Config, ensure_dirs, load_config
from .clause import Boundary, ClauseDetector
from .coach import LiveCoach
from .monitor import LiveMetrics, SessionState, measure
from .ring import Ring


class LiveSession:
    """One recording session. Start it, read `state`, stop it."""

    def __init__(self, cfg: Config | None = None, device=None,
                 niche: str | None = None, save: bool = True,
                 use_asr: bool = True, use_mentor: bool = True,
                 script_lines: list[dict] | None = None):
        self.cfg = cfg or load_config()
        self.sr = int(self.cfg.get("audio.sample_rate"))
        self.device = device
        self.niche = niche
        self.save = save
        self.use_asr = use_asr
        self.use_mentor = use_mentor

        self.window_sec = float(self.cfg.get("live.metric_window_sec", 4.0))
        self.ring = Ring(seconds=float(self.cfg.get("live.buffer_sec", 30.0)),
                         sr=self.sr)
        self.detector = ClauseDetector(
            sr=self.sr,
            min_speech_sec=float(self.cfg.get("live.min_speech_sec", 0.7)),
            min_silence_sec=float(self.cfg.get("live.min_silence_sec", 0.45)),
            gate_db=float(self.cfg.get("live.speech_gate_db", 10.0)))
        self.coach = LiveCoach.from_config(self.cfg)
        self.blocking_metrics = set(
            self.cfg.get("live.blocking_metrics", ["clipping"]))

        self.state = SessionState(coach_budget=self.coach.budget)
        if script_lines:
            self.state.load_script(script_lines)
        self.comped_path: str | None = None
        self._boundaries: queue.Queue[Boundary] = queue.Queue()
        # Transcription is a SEPARATE thread from coaching, with its own
        # bounded queue. Coaching must never wait on it - see _handle_boundary.
        self._asr_q: queue.Queue = queue.Queue()
        self._asr_max_pending = int(self.cfg.get("live.asr_max_pending", 2))
        self._asr_thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._stream = None
        self._wav = None
        self._worker: threading.Thread | None = None
        self._metrics_thread: threading.Thread | None = None

    # -----------------------------------------------------------------------
    #  Lifecycle
    # -----------------------------------------------------------------------
    def start(self) -> None:
        import sounddevice as sd
        import soundfile as sf

        if self.save:
            ensure_dirs()
            take_id = _dt.datetime.now().strftime("live_%Y%m%d_%H%M%S")
            out = TAKES_DIR / (take_id + ".wav")
            subtype = {16: "PCM_16", 24: "PCM_24",
                       32: "FLOAT"}[self.cfg.get("audio.bit_depth")]
            # Written incrementally: a live session that crashes or is killed
            # still leaves a playable take. The whole point of recording while
            # coaching is that the good take is not lost.
            self._wav = sf.SoundFile(out, mode="w", samplerate=self.sr,
                                     channels=1, subtype=subtype)
            self.state.take_id = take_id
            self.state.recording_to = str(out)

        self._stream = sd.InputStream(
            samplerate=self.sr, channels=1, device=self.device,
            blocksize=int(0.05 * self.sr), callback=self._on_audio)
        self._stream.start()

        self._worker = threading.Thread(target=self._work, daemon=True)
        self._worker.start()
        self._metrics_thread = threading.Thread(target=self._meter, daemon=True)
        self._metrics_thread.start()
        if self.use_asr:
            self._asr_thread = threading.Thread(target=self._asr_work, daemon=True)
            self._asr_thread.start()

    def stop(self) -> str | None:
        self._stop.set()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._worker is not None:
            self._worker.join(timeout=5.0)
        if self._wav is not None:
            self._wav.close()
            self._wav = None
            self._register_take()
            self._build_comp()
        return self.state.take_id

    def _build_comp(self) -> None:
        """Assemble the approved takes, alongside - never over - the raw file.

        Only when a script was used: without one there are no line boundaries
        and no notion of an attempt being accepted, so the raw recording IS the
        deliverable.
        """
        if not self.state.lines or not self.state.recording_to:
            return
        try:
            from .comp import build

            res = build(self.state.recording_to, self.state.lines)
            self.comped_path = res.path
            self.state.comped = res.path
            self.state.comp_missing = res.lines_missing
        except Exception as e:                       # pragma: no cover
            self.state.error = "comp failed: " + str(e)

    def _register_take(self) -> None:
        """Make the session a normal take, so `analyze` / `master` / `rank` all
        work on it afterwards. A live session that produced a good read and no
        take record would be a wasted read."""
        from ..storage import db

        if not (self.state.take_id and self.state.recording_to):
            return
        try:
            db.init_db()
            db.record_take(
                take_id=self.state.take_id,
                wav_path=self.state.recording_to,
                created_at=_dt.datetime.now().isoformat(timespec="seconds"),
                duration_sec=self.ring.elapsed,
                script_name=None,
                notes="live session" + (f"; niche={self.niche}" if self.niche else ""),
            )
        except Exception as e:                       # pragma: no cover
            self.state.error = "could not register take: " + str(e)

    # -----------------------------------------------------------------------
    #  Audio thread - must stay cheap
    # -----------------------------------------------------------------------
    def _on_audio(self, indata, frames, time_info, status) -> None:
        if status:
            self.state.error = str(status)
        block = np.asarray(indata, dtype=np.float32).reshape(-1)
        self.ring.write(block)
        if self._wav is not None:
            try:
                self._wav.write(block)
            except Exception:
                pass
        for b in self.detector.feed(block):
            self._boundaries.put(b)

    # -----------------------------------------------------------------------
    #  Metering thread - HUD numbers
    # -----------------------------------------------------------------------
    def _meter(self) -> None:
        hz = float(self.cfg.get("live.hud_hz", 4.0))
        while not self._stop.is_set():
            try:
                audio = self.ring.audio(self.window_sec)
                m = measure(audio, self.cfg, self.window_sec)
                m.t = round(self.ring.elapsed, 2)
                m.speaking = self.detector.in_speech
                m.noise_floor_db = round(self.detector.noise_floor_db, 1)
                self.state.metrics = m
            except Exception as e:                   # pragma: no cover
                self.state.error = "meter: " + str(e)
            time.sleep(1.0 / max(hz, 0.5))

    # -----------------------------------------------------------------------
    #  Worker thread - ASR, rubric, coach
    # -----------------------------------------------------------------------
    def _work(self) -> None:
        while not self._stop.is_set():
            try:
                b = self._boundaries.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                self._handle_boundary(b)
            except Exception as e:                   # pragma: no cover
                self.state.error = "worker: " + str(e)

    def _handle_boundary(self, b: Boundary) -> None:
        """Everything that does not need a transcript happens FIRST.

        ASR was originally called at the top of this method, so coaching waited
        on it. That was backwards: `base.en` transcribes a 2 s clause in ~3.9 s
        on CPU, which meant every note arrived four seconds after the moment it
        was about - and the whole justification for clause-boundary coaching is
        that it lands while the speaker still remembers the line.

        Nothing here needs the words. The rubric scores the SIGNAL, and the
        transcript is only context the live prompt can do without. So the clause
        appears and the coach speaks immediately, and ASR fills the text in
        later if it can keep up.
        """
        clause = self.state.add_clause(b.t_start, b.t_end, "", None, 0)
        if self.use_asr:
            self._queue_asr(b, clause)

        finding = self._score_recent()

        # ---------------------------------------------------------------
        #  LINE APPROVAL
        # ---------------------------------------------------------------
        #  A director does not move to line four because you finished saying
        #  line three - they move on because line three was GOOD. Advancing on
        #  clause count alone meant a botched line scrolled away while the coach
        #  was still asking for it again, and the reader had nothing to read.
        #
        #  So the take of a line is accepted only when nothing must-fix fired
        #  during it. Otherwise the line stays put and is marked for a retake.
        #  Every attempt is kept with its time range, because the last GOOD one
        #  is what gets exported - see `comp.py`.
        # ONLY UNRECOVERABLE FAULTS FAIL A LINE.
        #
        # Everything must_fix used to block, and the result was a reader stuck
        # on line one forever. Measured on a real session: sibilance ran
        # 51 per minute against a 30 must-fix threshold, so it fired on nearly
        # every line - and sibilance is precisely what the DE-ESSER in the
        # mastering chain exists to fix. Failing a good read for something the
        # pipeline repairs automatically is worse than not noticing it.
        #
        # A director stops the take for distortion, not for slightly sharp
        # esses. So: clipping blocks, because no mastering recovers a
        # flat-topped waveform. Everything else is still coached, and still
        # shown, but the line is accepted.
        blocking = finding is not None and finding.metric in self.blocking_metrics
        self.state.record_attempt(clause, passed=not blocking)

        if finding is None:
            return

        now = time.time()
        chosen = self.coach.pick([finding], now)
        if chosen is None:
            # Throttled, so nothing is said aloud - but the line still failed
            # and still shows as needing another take. Silence from the coach
            # is not approval.
            return

        if not self.use_mentor:
            self.state.add_note(
                self.coach.note_local(chosen, now, self._amount_for(chosen)),
                kind="rubric", meta={"metric": chosen.metric,
                                     "blocking": blocking})
            return

        try:
            line, meta = self.coach.speak(chosen, recent_text=clause.get("text", ""),
                                          now=now)
            self.state.coach_calls = self.coach.calls
            self.state.add_note(line, kind="coach", meta={
                "metric": chosen.metric,
                "model": meta.get("model"),
                "provider": meta.get("provider"),
                "output_tokens": meta.get("output_tokens"),
            })
        except Exception as e:
            # A dead API must not take the coach down with it. The rubric's own
            # sentence is specific and actionable; falling back to it keeps the
            # session useful instead of silently stopping.
            self.state.add_note(
                self.coach.note_local(chosen, now, self._amount_for(chosen)),
                kind="rubric", meta={"metric": chosen.metric,
                                     "blocking": blocking})
            self.state.error = "mentor unavailable, using local notes: " + str(e)

    # -----------------------------------------------------------------------
    #  ASR thread - always behind, and allowed to be
    # -----------------------------------------------------------------------
    def _queue_asr(self, b: Boundary, clause: dict) -> None:
        """Hand a clause to the transcriber, dropping the oldest work if it is
        falling behind.

        THE QUEUE MUST BE BOUNDED. Transcription runs slower than speech on
        CPU, so an unbounded queue does not just lag - it diverges. Ten minutes
        in, the HUD would be captioning what was said three minutes ago while
        the ring buffer had long since overwritten the audio, and the clause
        list would be confidently wrong rather than merely late.

        Dropping is the honest failure: the clause still appears, still counts,
        and is simply marked as never transcribed. The saved take gets a proper
        transcript from `analyze` afterwards, where there is no clock.
        """
        while self._asr_q.qsize() >= self._asr_max_pending:
            try:
                _, stale = self._asr_q.get_nowait()
                stale["text"] = ""
                stale["skipped"] = True
                self.state.asr_skipped += 1
            except queue.Empty:
                break
        clause["pending"] = True
        self._asr_q.put((b, clause))

    def _asr_work(self) -> None:
        while not self._stop.is_set():
            try:
                b, clause = self._asr_q.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                text, words, wpm = self._transcribe(b)
                clause.update(text=text, words=words,
                              wpm=round(wpm, 1) if wpm else None)
            except Exception as e:                   # pragma: no cover
                self.state.error = "asr: " + str(e)
                clause["text"] = ""
            finally:
                clause["pending"] = False

    def _transcribe(self, b: Boundary) -> tuple[str, int, float | None]:
        """Transcribe ONE CLAUSE, not a sliding window.

        A sliding window has to solve the problem of a word split across two
        windows - transcribed twice, or lost. Clauses are bounded by silence by
        construction, so the boundaries are exactly where a word is not. This
        is the main reason clause detection runs on energy rather than waiting
        for the transcript.
        """
        from ..analysis.asr import transcribe
        from ..analysis.audio import Audio

        pad = 0.15                     # a little either side; consonants are quiet
        samples = self.ring.between(max(0.0, b.t_start - pad), b.t_end + pad)
        if len(samples) < self.sr // 4:
            return "", 0, None

        model = self.cfg.get("live.asr_model", "base.en")
        text, words = transcribe(Audio(samples=samples, sr=self.sr),
                                 model_size=model)
        if not words:
            return text.strip(), 0, None
        speech = max(words[-1].end - words[0].start, 1e-6)
        return text.strip(), len(words), len(words) / (speech / 60.0)

    def _amount_for(self, finding) -> str | None:
        """How big the correction is, in words rather than units.

        The SIZE of a fix is the most useful thing a coach adds - without it
        "you are too loud" leaves the reader guessing between a nudge and a
        rethink. But the size has to arrive in language they can act on: "about
        9 dB" is a number off a meter, and a decibel is not a thing anyone can
        picture. "Try about half" is the same information, usable immediately.

        Returns None when the size cannot be inferred. A vague instruction
        beats a confident wrong number.
        """
        m = self.state.metrics

        if finding.metric == "clipping" and m.peak_db is not None:
            # Aim for peaks near -9 dBFS: headroom for a plosive without
            # burying the voice in the room.
            over = m.peak_db - (-9.0)
            if over >= 8.0:
                return "try about half of what it is now"
            if over >= 4.0:
                return "a good few notches, not a nudge"
            if over >= 2.0:
                return "just a small nudge down"

        if finding.metric == "noise" and m.snr_db is not None:
            short = 30.0 - m.snr_db
            if short >= 10.0:
                return "get properly close, about a hand's width"
            if short >= 3.0:
                return "a few centimetres closer should do it"

        if finding.metric == "proximity_effect" and m.tilt_db_per_oct is not None:
            if m.tilt_db_per_oct <= -8.0:
                return "a good hand's width back, not a couple of centimetres"
            return "a few centimetres is enough"

        return None

    def _score_recent(self):
        """Run the real rubric over the recent window and return its worst
        must-fix finding.

        Only the SIGNAL metrics are meaningful over a few seconds - pacing over
        one clause is a single sentence, and the rubric's own design says a
        single fast sentence is human. `timings_reliable=False` is what tells
        the rubric to leave pacing and pauses alone here.
        """
        from ..schemas import (
            AnalysisResult,
            Pacing,
            Pitch,
            ProximityEffect,
            Sibilance,
            SibilancePeak,
            TiltPoint,
        )
        from ..scoring.rubric import score

        # Measured fresh, not read from the HUD's last sample. The finding has
        # to describe the clause that just ended; the meter thread runs on its
        # own schedule and may have sampled several seconds ago, or - at
        # startup - not yet at all.
        m: LiveMetrics = measure(self.ring.audio(self.window_sec), self.cfg,
                                 self.window_sec)
        m.t = round(self.ring.elapsed, 2)
        if m.rms_db is None:
            return None

        from ..schemas import Click

        r = AnalysisResult(
            take_id=self.state.take_id or "live",
            duration_sec=self.window_sec,
            transcript="",
            words=[],
            pacing=Pacing(wpm_overall=0.0),
            pauses=[],
            pitch=Pitch(mean_hz=0.0, range_hz=0.0, variance=0.0),
            proximity_effect=ProximityEffect(
                spectral_tilt_db_per_octave=(
                    [TiltPoint(t=0.0, value=m.tilt_db_per_oct)]
                    if m.tilt_db_per_oct is not None else [])),
            sibilance=Sibilance(peaks=[
                SibilancePeak(t=0.0, band_energy_db=0.0)
                for _ in range(int((m.sibilance_per_min or 0)
                                   * self.window_sec / 60.0))]),
            noise_floor_db=m.noise_floor_db or -60.0,
            snr_db=m.snr_db,
            mouth_clicks=[Click(t=m.t)] if m.clipping else [],
        )
        scored = score(r, cfg=self.cfg, timings_reliable=False)
        for f in scored.findings:
            if f.severity == "must_fix":
                return f
        return None
