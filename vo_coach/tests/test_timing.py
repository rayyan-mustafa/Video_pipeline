"""Timing repair: room tone, gap detection, pause correction, script marking.

The tests that matter most here are the ones about what the tool REFUSES to do.
A pause fixer that stretches everything it is given will happily fabricate a
beat the speaker never performed, and because it runs silently there is nothing
to tell the user it happened - the only evidence would be a finished video that
sounds subtly edited.
"""

from __future__ import annotations

import numpy as np
import pytest

from vo_coach.timing import gaps as G
from vo_coach.timing import marking as M
from vo_coach.timing import roomtone as RT

SR = 24000


def hush(sec, amp=0.002, seed=0):
    return np.random.default_rng(seed).normal(0, amp, int(sec * SR)).astype(np.float32)


def tone(sec, amp=0.25, f=180.0):
    t = np.arange(int(sec * SR)) / SR
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


def speech_with_gaps(*parts):
    """parts alternate: speech seconds, gap seconds, speech seconds ..."""
    out = []
    for i, sec in enumerate(parts):
        out.append(tone(sec) if i % 2 == 0 else hush(sec, seed=i))
    return np.concatenate(out).astype(np.float32)


class TestRoomTone:
    def test_picks_the_quietest_passage_not_the_first(self):
        """The head of a take often holds the click of the record key. Using
        the first quiet stretch would splice that into every widened pause."""
        x = np.concatenate([hush(0.5, amp=0.02, seed=1),   # noisy "head"
                            tone(1.0),
                            hush(0.6, amp=0.001, seed=2),  # the real room tone
                            tone(1.0)]).astype(np.float32)
        t = RT.extract(x, SR)
        assert float(np.sqrt(np.mean(t.astype(np.float64) ** 2))) < 0.005

    def test_refuses_when_nothing_is_quiet(self):
        """A take with no quiet passage cannot be extended honestly - the
        'room tone' would contain a fragment of a word."""
        with pytest.raises(RT.NoRoomTone):
            RT.extract(tone(3.0), SR)

    def test_refuses_when_too_short_to_sample(self):
        with pytest.raises(RT.NoRoomTone):
            RT.extract(hush(0.1), SR)

    def test_fill_returns_exactly_what_was_asked_for(self):
        t = RT.extract(np.concatenate([tone(1.0), hush(0.8), tone(1.0)]), SR)
        assert len(RT.fill(t, 5000, SR)) == 5000

    def test_fill_does_not_loop_audibly(self):
        """A plain loop repeats any faint hum at exactly the sample length, and
        a repeating artifact is more audible than the noise it came from. The
        fill alternates direction, so the second copy must not equal the
        first."""
        t = RT.extract(np.concatenate([tone(1.0), hush(1.0), tone(1.0)]), SR)
        n = len(t)
        filled = RT.fill(t, n * 3, SR)
        first, second = filled[:n // 2], filled[n: n + n // 2]
        assert not np.allclose(first, second, atol=1e-6)

    def test_fill_is_continuous(self):
        """Equal-power crossfades at the joins: no click, and no 3 dB hole."""
        t = RT.extract(np.concatenate([tone(1.0), hush(1.0), tone(1.0)]), SR)
        filled = RT.fill(t, len(t) * 4, SR)
        step = np.abs(np.diff(filled.astype(np.float64)))
        assert step.max() < 10 * np.percentile(step, 99)


class TestGapDetection:
    def test_finds_the_gaps(self):
        x = speech_with_gaps(1.0, 0.8, 1.0, 0.6, 1.0)
        found = G.find(x, SR)
        assert len(found) >= 2

    def test_gap_timing_is_about_right(self):
        x = speech_with_gaps(1.0, 0.8, 1.0)
        g = G.interior(G.find(x, SR), len(x) / SR)[0]
        assert 0.9 <= g.start <= 1.1
        assert 0.6 <= g.duration <= 1.0

    def test_head_and_tail_silence_are_not_pauses(self):
        """Trailing silence is tail, not a beat between thoughts. Stretching it
        would push every following section later and desync the pictures."""
        x = np.concatenate([hush(1.0), tone(1.0), hush(0.8),
                            tone(1.0), hush(1.0)]).astype(np.float32)
        assert len(G.interior(G.find(x, SR), len(x) / SR)) == 1

    def test_short_gaps_are_marked_unfixable(self):
        assert not G.Gap(start=1.0, end=1.1).fixable
        assert G.Gap(start=1.0, end=1.5).fixable


class TestApply:
    def _setup(self, gap_sec):
        x = speech_with_gaps(1.0, gap_sec, 1.0)
        t = RT.extract(x, SR)
        g = G.interior(G.find(x, SR), len(x) / SR)[0]
        return x, t, g

    def test_lengthening_adds_time(self):
        x, t, g = self._setup(0.5)
        y, adjusted, _ = G.apply(x, SR, [(g, 1.2)], t)
        assert adjusted == 1
        assert len(y) / SR == pytest.approx(len(x) / SR + 0.7, abs=0.08)

    def test_shortening_removes_time(self):
        x, t, g = self._setup(1.2)
        y, adjusted, _ = G.apply(x, SR, [(g, 0.5)], t)
        assert adjusted == 1
        assert len(y) / SR == pytest.approx(len(x) / SR - 0.7, abs=0.08)

    def test_a_gap_that_was_never_performed_is_left_alone(self):
        """THE CENTRAL RULE. Below MIN_REAL_GAP the speaker ran two sentences
        together - their intonation never closed - so padding room tone in
        produces the exact 'this was edited' sound the tool exists to avoid."""
        x, t, _ = self._setup(0.5)
        tiny = G.Gap(start=1.0, end=1.10)
        y, adjusted, skipped = G.apply(x, SR, [(tiny, 1.2)], t)
        assert adjusted == 0 and skipped == 1
        assert len(y) == len(x)

    def test_near_enough_is_left_alone(self):
        """Rewriting a gap to move it 20 ms is churn, not correction."""
        x, t, g = self._setup(0.80)
        _, adjusted, _ = G.apply(x, SR, [(g, 0.82)], t)
        assert adjusted == 0

    def test_multiple_gaps_stay_in_position(self):
        """Edits run back to front. Editing forwards shifts every later gap by
        the amount already inserted, so the file drifts further out of place
        the longer it runs."""
        x = speech_with_gaps(1.0, 0.5, 1.0, 0.5, 1.0)
        t = RT.extract(x, SR)
        found = G.interior(G.find(x, SR), len(x) / SR)
        assert len(found) == 2
        y, adjusted, _ = G.apply(x, SR, [(found[0], 1.5), (found[1], 1.5)], t)
        assert adjusted == 2
        # Both gaps grew by ~1.0s, so the file grew by ~2.0s.
        assert len(y) / SR == pytest.approx(len(x) / SR + 2.0, abs=0.15)

    def test_no_clicks_are_introduced(self):
        """A splice with no crossfade shows up as a sample step far larger than
        anything in the source."""
        x, t, g = self._setup(0.5)
        y, _, _ = G.apply(x, SR, [(g, 1.4)], t)
        assert np.abs(np.diff(y.astype(np.float64))).max() <= \
            np.abs(np.diff(x.astype(np.float64))).max() * 1.05


class TestMarking:
    SCRIPT = ("One thing happens, and then another. A second sentence follows; "
              "it has a clause too.")

    def test_sentence_and_clause_pauses_differ(self):
        s = M.PauseStyle(sentence=0.9, clause=0.3)
        out = M.mark(self.SCRIPT, s)
        assert "[0.90]" in out and "[0.30]" in out

    def test_targets_are_read_in_order(self):
        s = M.PauseStyle(sentence=0.9, clause=0.3)
        assert M.targets(M.mark(self.SCRIPT, s))[0] == 0.3

    def test_the_legend_is_not_read_as_a_target(self):
        """REGRESSION. The header shows the marker format, so it necessarily
        contains something marker-shaped. Counting it shifted every pause onto
        the NEXT pause's target - silently wrong phrasing, no error, and the
        first gap in the real file was set to 0.00 seconds."""
        rendered = M.render(self.SCRIPT, M.PauseStyle(sentence=0.9, clause=0.3))
        assert "[n.nn]" in rendered
        got = M.targets(rendered)
        assert 0.0 not in got
        assert got == M.targets(M.mark(self.SCRIPT, M.PauseStyle(0.9, 0.3)))

    def test_no_target_is_ever_zero(self):
        for v in M.targets(M.render(self.SCRIPT, M.PauseStyle())):
            assert v > 0.0

    def test_unknown_niche_falls_back_to_defaults(self):
        assert M.style_for("nope").source == "defaults"
        assert M.style_for(None).sentence == M.DEFAULT_SENTENCE_PAUSE

    def test_sentence_pause_is_longer_than_clause_pause(self):
        for niche in (None, "zenn", "nope"):
            s = M.style_for(niche)
            assert s.clause < s.sentence < s.paragraph


class TestFixPauses:
    def _wav(self, tmp_path, gap=0.5):
        import soundfile as sf

        x = speech_with_gaps(1.0, gap, 1.0, gap, 1.0)
        p = tmp_path / "t.wav"
        sf.write(str(p), x, SR, subtype="PCM_16")
        return p

    def test_writes_a_new_file_by_default(self, tmp_path):
        from vo_coach.timing import fix_pauses

        res = fix_pauses(self._wav(tmp_path), niche=None)
        assert res.path.endswith("_timed.wav")
        assert res.gaps_found >= 2

    def test_in_place_overwrites(self, tmp_path):
        """section_NN.wav has to keep its name or the notebook will not see the
        corrected file."""
        from vo_coach.timing import fix_pauses

        p = self._wav(tmp_path)
        res = fix_pauses(p, in_place=True)
        assert res.path == str(p)

    def test_mismatched_target_count_falls_back_rather_than_misaligning(self, tmp_path):
        """Pairing gaps to targets in order is only valid when the counts
        agree. One unplanned breath would otherwise shift every pause onto the
        wrong target, and the result would sound wrong with nothing to say
        why."""
        from vo_coach.timing import fix_pauses

        # Three targets against a file with two performed pauses.
        script = M.render("One, two, three, done.", M.PauseStyle())
        assert len(M.targets(script)) == 3
        res = fix_pauses(self._wav(tmp_path), script_marked=script)
        assert res.note and "rhythm instead" in res.note

    def test_matching_counts_use_the_script_targets(self, tmp_path):
        """When the counts DO agree, the script wins - that is the whole point
        of marking one."""
        from vo_coach.timing import fix_pauses

        script = M.render("One thing, then another, done.",
                          M.PauseStyle(sentence=1.4, clause=1.4))
        assert len(M.targets(script)) == 2
        res = fix_pauses(self._wav(tmp_path, gap=0.5), script_marked=script)
        assert res.note is None
        assert all(d["now"] == 1.4 for d in res.details)

    def test_no_room_tone_means_shorten_only(self, tmp_path, monkeypatch):
        """Without room tone a gap can only be TRIMMED, never widened.

        Doing half the job silently is right here; the alternative is splicing
        digital silence in, which drops the noise floor to true zero mid-pause -
        the single clearest tell of amateur editing.
        """
        from vo_coach import timing
        from vo_coach.timing import fix_pauses, roomtone

        def _refuse(*a, **k):
            raise roomtone.NoRoomTone("no quiet passage")

        monkeypatch.setattr(timing.roomtone, "extract", _refuse)
        res = fix_pauses(self._wav(tmp_path, gap=0.30))
        assert res.tone_found is False
        assert res.note == "no quiet passage"
        # Every remaining target must be a shortening, so the file cannot grow.
        assert res.duration_after <= res.duration_before + 0.01


class TestMarkAndGapsAgree:
    def test_no_pause_is_marked_after_the_final_sentence(self):
        """A pause is the space BETWEEN two things. `gaps.interior()` discards
        trailing silence for the same reason, so marking one there would put
        the target list and the detected gaps out of step by one on every
        single script - the misalignment bug, reintroduced structurally."""
        out = M.mark("First one. Second one.", M.PauseStyle(sentence=0.9))
        assert out.count("[0.90]") == 1


class TestDelivery:
    """Direction: which word carries a line, and how to say it.

    Computed at MARK time, never during the read - a note that arrives while
    the reader is mid-line is too late to use, and the shape of a sentence does
    not depend on how it was performed.
    """

    LINE = "Somewhere past the last bright star you can see, space stops here."

    def test_picks_a_content_word(self):
        got = M.emphasis_words(self.LINE)
        assert got
        assert all(w not in M.STOPWORDS for w in got)

    def test_never_picks_an_empty_long_word(self):
        """'Lean on SOMETHING' is not direction, and those words beat real ones
        on length alone."""
        for line in ("It was something patient and quiet.",
                     "Everything since has simply been consequence.",
                     "The silence pressing against those distant systems."):
            for w in M.emphasis_words(line):
                assert w not in {"something", "everything", "against"}, line

    def test_does_not_repeat_a_word(self):
        """A word appearing twice scores twice and came back as both picks,
        telling the reader to lean on the same word twice."""
        got = M.emphasis_words("is pressing, right now, is pressing on you all")
        assert len(got) == len(set(got))

    def test_empty_line_is_handled(self):
        assert M.emphasis_words("") == []
        assert M.emphasis_words("the and of") == []

    def test_notes_are_plain_english(self):
        """A nine-year-old has to be able to act on these. No term in a note
        may need explaining."""
        jargon = ("db", "dbfs", "lufs", "hz", "octave", "sibilan", "proximity",
                  "snr", "signal-to-noise", "spectral", "tilt", "wpm",
                  "amplitude", "frequency", "compressor", "normalis")
        for line in (self.LINE, "Short one.", "A, b, c, and then d.",
                     "word " * 25):
            note = M.delivery_note(line).lower()
            for j in jargon:
                assert j not in note, f"{j!r} leaked into: {note}"

    def test_note_shape_follows_the_sentence(self):
        assert "short" in M.delivery_note("Stand still.").lower()
        assert "two breaths" in M.delivery_note("word " * 25).lower()
        assert "list" in M.delivery_note("A, b, c and d.").lower()
        assert "last one" in M.delivery_note("The end.", is_last=True).lower()

    def test_direct_returns_one_entry_per_line(self):
        out = M.direct("First line here. Second line here. Third one now. ",
                       M.PauseStyle())
        assert len(out) >= 2
        for d in out:
            assert set(d) == {"text", "pause", "emphasis", "note"}
            assert "[" not in d["text"], "pause markers must not reach the reader"

    def test_direction_never_mentions_accent_or_pronunciation(self):
        """The single hard line: this marks which word CARRIES a sentence, which
        is meaning and identical in every accent. It never tells anyone how
        their vowels should sound."""
        for line in ("The physicists say the shape of it was set.",
                     "Stand still for a moment and consider that."):
            note = M.delivery_note(line).lower()
            for bad in ("accent", "pronounc", "properly", "correct", "native",
                        "american"):
                assert bad not in note
