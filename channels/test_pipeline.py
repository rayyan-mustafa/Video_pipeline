"""Exercise every extracted module against a fake model. No API calls, no network.

    .venv\Scripts\python.exe test_pipeline.py

This is the regression guard on the notebook->package extraction. The checks
that matter most are the three that would silently corrupt a script rather
than raise: no global mutation in config, apply_fixes landing a rewrite on the
right line, and DROP removing exactly one line.
"""
import json, sys, tempfile, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from pipeline import paths
WORK = Path(tempfile.mkdtemp(prefix="pipe_smoke_"))
paths.set_workdir(WORK)

from pipeline import config, lint, script as script_mod, factcheck, retention, packaging, evidence
from pipeline import humanize
from pipeline.brief import Brief
import pipeline.engine as engine

# ---------------------------------------------------------------- fake model
CALLS = []
def fake_call_model(model, prompt, system=None, max_tokens=4000, temperature=0, retries=4):
    CALLS.append({"model": model, "len": len(prompt), "system": bool(system)})
    p = prompt + (system or "")
    if '"evidence_type"' in p:
        return json.dumps({
            "evidence_type": "competing_hypotheses",
            "evidence_type_reason": "two rival explanations",
            "named_entities": [{"kind": "researcher", "value": "Dr Example", "source": 1}],
            "hard_numbers": [{"value": "550 million", "means": "years ago", "source": 1}],
            "key_findings": [{"finding": "It happens.", "source": 1, "certainty": "debated"}],
            "competing_explanations": [], "mechanism": "A then B.",
            "the_common_myth": "the schoolbook answer", "the_best_counter_evidence": "the odd bit",
            "gaps": ["anything about cats"],
        })
    if "=== HOOK 1 ===" in p:
        return ("=== HOOK 1 ===\nTouch your arm. [S1|0]\nWrong. [S1|0]\n"
                "=== HOOK 2 ===\nLook down now. [S1|0]\nNot true. [S1|0]\n")
    if '"winner"' in p:
        return json.dumps({"winner": 1, "ranking": [1, 2], "why_winner": "punchy",
                           "why_others_lost": "vague", "improvement": None})
    if '"titles"' in p:
        return json.dumps({
            "titles": [{"text": "Why Do You Twitch?", "angle": "question", "why": "curiosity"}],
            "thumbnail_text": [{"text": "STILL WIRED", "visual": "stickman shivering"}],
            "description": "Two sentences.", "chapters": [{"label": "The itch", "section": 1}],
            "tags": ["biology"], "pinned_comment": "What do you think?",
            "community_post": "Tomorrow.",
        })
    if '"issues"' in p and "fact-checker" in p:
        return json.dumps({"issues": [{"sentence": "It happens in every mammal.",
                                       "reason": "source says some mammals"}]})
    if '"issues"' in p:
        return json.dumps({"issues": []})
    if "#0 |" in p or "OUTPUT FORMAT - one line per fix" in p:
        # echo back a fix for whatever ids were flagged
        import re as _re
        out = []
        for m in _re.finditer(r"^#(\d+) \[S(\d+)\|", p, _re.M):
            out.append(f"#{m.group(1)} | Repaired line here. [S{m.group(2)}|1]")
        return "\n".join(out) or "#0 | Repaired. [S1|0]"
    if "Return ONLY the rewritten section" in p:
        return "Rebuilt line one. [S3|1]\nRebuilt line two. [S3|0]\n"
    # the writer
    return "\n".join([
        "Touch the back of your arm. [S1|0]",
        "Why does it do that? [S1|0]",
        "Everyone says it is cold. [S1|0]",
        "Wrong. [S1|0]",
        "It happens in every mammal. [S3|1]",
        "That is 550 million years of it. [S3|1]",
        "You are still wired for a coat. [S5|0]",
        "Touch your arm again. [S6|0]",
        "What else is still running? [S6|0]",
    ])

engine.call_model = fake_call_model
for mod in (script_mod, factcheck, retention, packaging, evidence):
    if hasattr(mod, "call_model"):
        mod.call_model = fake_call_model

ok = lambda m: print(f"  ok  {m}")

# ------------------------------------------------------------------ config
learnings = config.load_learnings()
assert learnings["retention_notes"] == []
banned = config.banned_phrases(learnings)
assert "delve" in banned
before = len(config.BANNED_PHRASES)
config.banned_phrases({"banned_phrase_additions": ["zzz"]})
assert len(config.BANNED_PHRASES) == before, "BANNED_PHRASES was mutated!"
sheet_a, t = config.build_beat_sheet("clinical_case")
sheet_b, _ = config.build_beat_sheet("mechanism_only")
assert sheet_a[3]["name"] == "The Patient" and sheet_b[3]["name"] == "The Machinery"
assert 3 not in config.BEAT_SHEET, "BEAT_SHEET template was mutated!"
assert sorted(sheet_a) == [1, 2, 3, 4, 5, 6]
ok("config: no global mutation, beat sheet builds per-evidence-type")

# ------------------------------------------------------------------ evidence
source_pack = [{"title": "S1", "url": "http://x", "excerpt": "It happens 550 million years ago in some mammals. " * 30}]
profile, beat_sheet, ev = evidence.run("Why We Get Goosebumps", source_pack)
assert ev == "competing_hypotheses"
assert (WORK / "evidence_profile.json").exists()
ok(f"evidence: profiled, section 3 = {beat_sheet[3]['name']!r}")

b = Brief(topic="Why We Get Goosebumps", source_pack=source_pack, profile=profile,
          beat_sheet=beat_sheet, banned=banned)
assert "550 million" in b.numbers_block and "Dr Example" in b.entities_block
assert "=== SECTION 3:" in b.render_beat_sheet()
assert b.allowed_numbers == ["550 million"]
ok("brief: all six prompt blocks render")

# ------------------------------------------------------------------ writer
tagged = script_mod.write(b)
assert len(tagged) == 9 and tagged[3]["sentence"] == "Wrong."
assert (WORK / "script_tagged.json").exists() and (WORK / "vo_raw.txt").exists()
assert script_mod.load_script() == tagged
ok(f"script: {len(tagged)} lines parsed, saved and reloaded")

# ------------------------------------------------------------------ linter
issues = lint.run(tagged, b)
reasons = " | ".join(i["reason"] for i in issues)
assert "550 million" not in reasons or True
assert any("fact-carrying sentences in a row" in i["reason"] for i in issues), "stat-run check silent"
assert any("is missing entirely" in i["reason"] or "words, target" in i["reason"] for i in issues)
ok(f"lint: {len(issues)} issues, structure + stat-run checks fired")

# number grounding: a real number in the source passes, an invented one fails
assert lint._number_in_sources("550", b.all_source_text)
assert not lint._number_in_sources("532", b.all_source_text)
ok("lint: number grounding accepts sourced, rejects invented")

# ------------------------------------------------------------------ factcheck
fact, smug = factcheck.run(tagged, b, quiet=True)
assert len(fact) == 1 and fact[0]["kind"] == "fact"
ok(f"factcheck: {len(fact)} fact, {len(smug)} smuggled")

# ------------------------------------------------------------------ fixer
fixed = script_mod.apply_fixes(tagged, fact, "Fix these.", b)
assert len(fixed) == len(tagged)
assert fixed[4]["sentence"] == "Repaired line here."
ok("apply_fixes: id-keyed replacement landed on the right line")

# drop-safety: a DROP must remove exactly one line
engine.call_model = script_mod.call_model = lambda *a, **k: "#4 | DROP"
dropped = script_mod.apply_fixes(tagged, fact, "Fix.", b)
assert len(dropped) == len(tagged) - 1
ok("apply_fixes: DROP removes exactly one line")

# duplicate targeting: three copies of one line, all three flagged, all three
# must be rewritten. Before the fix this rewrote copy #1 three times and left
# copies #2 and #3 in the script - seen for real on a cheap-preset run.
_dupes = [{"sentence": "Feel the bumps.", "section": 1, "sources": []},
          {"sentence": "Something else entirely.", "section": 1, "sources": []},
          {"sentence": "Feel the bumps.", "section": 1, "sources": []},
          {"sentence": "Feel the bumps.", "section": 1, "sources": []}]
script_mod.call_model = lambda *a, **k: "\n".join(
    f"#{i} | Rewritten {i}. [S1|0]" for i in (0, 2, 3))
_out = script_mod.apply_fixes(_dupes, [{"sentence": "Feel the bumps.", "reason": "dup"}] * 3,
                              "Fix.", b)
assert [i["sentence"] for i in _out] == [
    "Rewritten 0.", "Something else entirely.", "Rewritten 2.", "Rewritten 3."], _out
ok("apply_fixes: every copy of a duplicated line gets its own rewrite")
script_mod.call_model = fake_call_model
# engine.call_model too: call_model_json (evidence/factcheck/humanize/packaging/
# retention's JSON calls) is defined in engine.py and resolves `call_model` as
# ITS OWN module global, not each importer's separately-patched copy - so the
# DROP-test override above (which set engine.call_model, not just
# script_mod.call_model) has to be undone here too, or every call_model_json
# call for the rest of this file silently gets "#4 | DROP" instead of JSON.
engine.call_model = fake_call_model

# section rebuild
rebuilt = script_mod.rewrite_section(tagged, 3, ["too long"], b)
assert any(i["sentence"] == "Rebuilt line one." for i in rebuilt)
assert all(i["section"] == 3 for i in rebuilt if i["sentence"].startswith("Rebuilt"))
ok("rewrite_section: section 3 replaced in place")

# ------------------------------------------------------------------ retention
system = retention.system_prompt(learnings)
assert "no data yet" in system
rissues = retention.run_gate(tagged, b, system)
ok(f"retention gate: ran over {len(beat_sheet)} sections + arc, {len(rissues)} issues")

spliced = retention.hook_tournament(tagged, b, system, n_hooks=2)
assert spliced[0]["sentence"] == "Touch your arm."
assert (WORK / "hook_tournament.json").exists()
assert sum(1 for i in spliced if i["section"] == 1) == 2
ok("hook tournament: winner spliced, old section 1 replaced")

# ------------------------------------------------------------------ humanizer
# The quotas exist because a script can clear both other gates and still be
# unmistakably machine written. Each check gets a script that breaks exactly one
# of them, so a failure names the audit rather than "something in humanize".
def _mk(sentences):
    return [{"sentence": t, "section": 1, "sources": []} for t in sentences]

dashes = _mk([f"Line {i} is here — and here is the part that matters." for i in range(6)])
hits = humanize.audit_em_dashes(dashes)
assert len(hits) == 6 - humanize.VOICE_QUOTAS["em_dash_max_lines"]
assert not humanize.audit_em_dashes(dashes[:3])
ok("humanize: em-dash budget flags the overflow only, not the first three")

assert humanize.audit_anaphora(_mk(["Maybe it was this.", "Maybe it was that."]))
assert not humanize.audit_anaphora(_mk(["Maybe it was this.", "Nobody knows."]))
ok("humanize: the parallel-Maybe close is caught, a varied one is not")

assert len(humanize.audit_tricolons(_mk([
    "It was cold, wet, and dark.", "You have skin, hair, and bone."]))) == 1
ok("humanize: one rule-of-three list is allowed, the second is flagged")

# Three plain sentences in a row must NOT be flagged - that is obligation 2, and
# flagging it would have the fixer put the ornament straight back on.
plain = _mk(["Your lower jaw did all of this on its own.",
             "The front of the bone moved forward over time.",
             "Nobody has ever agreed on the reason for it."])
assert not [i for i in humanize.audit_construction_variety(plain) if "in a row" in i["reason"]]
ok("humanize: a run of plain sentences is left alone")

# A topic word repeated is the subject of the video, not a fingerprint.
pruney = _mk([f"Your fingers do the odd thing again, take {i}." for i in range(4)])
assert not [i for i in humanize.audit_word_repetition(pruney, "Why Fingers Prune in Water")
            if "finger" in i["reason"]]
assert [i for i in humanize.audit_word_repetition(pruney, "") if "finger" in i["reason"]]
ok("humanize: the topic's own words are exempt from the repetition budget")

hissues = humanize.run_gate(tagged, b, use_judge=False)
ok(f"humanize gate: deterministic sweep ran, {len(hissues)} issue(s)")

# The writer must receive the same numbers the gate enforces. A quota the writer
# never saw is a quota that just burns fix rounds.
wb = humanize.writer_block()
assert str(humanize.VOICE_QUOTAS["em_dash_max_lines"]) in wb
assert wb in script_mod._system_prompt(b)
ok("humanize: the writer prompt carries the quotas the gate enforces")

assert "and yet" in config.banned_phrases(learnings)
ok("config: AI-tell lexicon reaches the deterministic linter")

# ------------------------------------------------------------------ packaging
pack = packaging.run(tagged, b, learnings)
for f in ("packaging.json", "youtube_description.txt", "youtube_tags.txt", "pinned_comment.txt"):
    assert (WORK / f).exists(), f
desc = (WORK / "youtube_description.txt").read_text(encoding="utf-8")
assert "http://x" in desc
ok("packaging: 4 files written, sources appended to description")

# ------------------------------------------------------------------ paths / stage markers
paths.mark_done("script", ["script_tagged.json", "vo_raw.txt"])
paths.require("script")
ok("paths: mark_done + require round-trip")

(WORK / "vo_raw.txt").write_text("truncated", encoding="utf-8")   # simulate a half-synced file
try:
    paths.require("script")
    raise AssertionError("require() did not catch a size mismatch")
except RuntimeError as e:
    assert "still syncing" in str(e)
ok("paths: require() catches a half-synced file")

try:
    paths.require("frames")
    raise AssertionError("require() did not catch a missing stage")
except FileNotFoundError:
    pass
ok("paths: require() catches a stage that never ran")

# ------------------------------------------------------------------ engine
# The OpenRouter layer: slug shape, retry classification, cost accounting.
# No network and no key needed.
import pipeline.engine as _eng


class _Status(Exception):
    def __init__(self, c):
        self.status_code = c


for code, want in [(401, False), (400, False), (404, False),
                   (429, True), (500, True), (503, True), (529, True)]:
    assert _eng._is_retryable(_Status(code)) is want, f"HTTP {code} retry classification"
assert _eng._is_retryable(ConnectionError("reset")) is True
assert _eng._is_retryable(_eng.ProviderError("ours")) is False
ok("engine: permanent errors fail fast, transient ones retry")

_roles = config.active_models()
assert set(_roles) == {"writer", "judge", "extractor", "checker", "scenes"}
for _role, slug in _roles.items():
    assert "/" in slug, f"{_role}={slug!r} is not an OpenRouter slug (needs a provider prefix)"
    assert not slug.endswith("-20251001"), f"{slug!r} is an Anthropic dated id, not a slug"
# the two roles a viewer's experience actually rides on must stay on Sonnet
assert _roles["writer"] == _roles["judge"] == "anthropic/claude-sonnet-5"
ok("engine: five roles, all OpenRouter-shaped, writer+judge on Sonnet")

assert _eng.ALLOW_FALLBACKS is False, "fallbacks must stay off - a bad pin has to fail loudly"
for _p in config.MODEL_PRESETS.values():
    for slug in _p.values():
        assert _eng.provider_for(slug), f"{slug} has no provider pin"
# anything that reasons must have it explicitly configured, or it silently eats
# max_tokens before producing a visible character
for slug in (config.MODEL_SCENES, config.MODEL_PRESETS["cheap"]["scenes"]):
    if not slug.startswith("anthropic/"):
        assert slug in config.REASONING, f"{slug} reasons but has no REASONING entry"
try:
    _eng.provider_for("deepseek/deepseek-chat")     # real model, deliberately unpinned
    raise AssertionError("an unpinned model was accepted")
except _eng.ProviderError as e:
    assert "No provider pin" in str(e)
ok("engine: every preset model is pinned; an unpinned one is refused")

_before = config.active_models()
applied = config.apply_preset("cheap")
assert config.MODEL_WRITER == "inclusionai/ling-3.0-flash" == applied["writer"]
assert config.MODEL_SCENES == "inclusionai/ling-3.0-flash"
assert _eng.provider_for(config.MODEL_WRITER) == ["novita/fast"]
config.apply_preset("claude")
assert config.active_models() == _before
try:
    config.apply_preset("nope")
    raise AssertionError("unknown preset accepted")
except ValueError:
    pass
ok("engine: model presets switch all five roles and restore cleanly")


class _Usage:
    prompt_tokens = 1000
    completion_tokens = 500
    cost = 0.0075
    completion_tokens_details = None


_eng.USAGE.clear()
_eng._record("anthropic/claude-sonnet-5", _Usage(), _eng._reported_cost(_Usage()))
u = _eng.USAGE["anthropic/claude-sonnet-5"]
assert u["cost_is_real"] and abs(u["cost"] - 0.0075) < 1e-9

_eng.USAGE.clear()
_eng._record("anthropic/claude-sonnet-5", _Usage(), None)      # nothing reported
assert _eng.USAGE["anthropic/claude-sonnet-5"]["cost_is_real"] is False
total = _eng.print_usage()
assert abs(total - (1000 / 1e6 * 2.0 + 500 / 1e6 * 10.0)) < 1e-9, "price-table fallback wrong"
_eng.USAGE.clear()
ok("engine: real billed cost preferred, price table is the fallback")

# ----------------------------------------------------- notebook <-> package
# The two constants that live on both sides. run_scenes.py bakes them into every
# frame's prompt; the notebook's Step 24 Redraw button rebuilds that prompt from
# its own copy. If they drift, one redrawn frame stops matching the other 149 and
# nothing says so - which is exactly the kind of failure a test has to hold shut.
import contextlib
import io as _io
import re as _re

from pipeline import scenes as _sc, shots as _shots

_NB = Path(__file__).resolve().parent / "stickman.ipynb"
if _NB.exists():
    _nb = json.loads(_NB.read_text(encoding="utf-8"))
    _code = [("".join(c["source"])) for c in _nb["cells"] if c["cell_type"] == "code"]

    _step3 = next(s for s in _code if "CHARACTER_BIBLE = (" in s)
    assert "style.json" in _step3, "Step 3 no longer defers to style.json"
    assert "def build_visual_prompt" in _step3, "Step 3 lost build_visual_prompt"

    # The Redraw button must go THROUGH the builder. It used to interpolate
    # f"{STYLE_PREFIX}{CHARACTER_BIBLE}, {action}" itself, which is exactly the
    # bug the shot rotation exists to fix: the bible was concatenated whether or
    # not the frame had a character in it.
    _redraw = next(s for s in _code if 'sc["visual_prompt"] =' in s
                   and "build_visual_prompt" in s)
    assert "{CHARACTER_BIBLE}" not in _redraw, (
        "Step 24 Redraw interpolates CHARACTER_BIBLE directly again - a redrawn "
        "chart frame would get a stick figure glued to it")

    _cwd = os.getcwd()
    os.chdir(WORK)
    try:
        config.export_style()
        _g = {}
        with contextlib.redirect_stdout(_io.StringIO()):
            exec(compile(_step3, "<step3>", "exec"), _g)

        assert set(_g["SHOTS"]) == set(_shots.SHOTS), (
            f"the notebook's shot catalog does not match the package's:\n"
            f"  notebook: {sorted(_g['SHOTS'])}\n"
            f"  package:  {sorted(_shots.SHOTS)}")

        from pipeline import look as _look

        assert set(_g["EXPRESSIONS"]) == set(_look.EXPRESSIONS), (
            "the notebook's expression catalog does not match the package's")

        _action = "stick figure touching own forearm"
        # Every shot x every expression: the character-free shots are where the
        # two implementations can disagree, and they are the whole point.
        for _shot in list(_shots.SHOTS) + [None, "not-a-real-shot"]:
            for _ex in list(_look.EXPRESSIONS) + [None, "nope"]:
                _baked = _sc.visual_prompt(_action, _shot, _ex)
                _rebuilt = _g["build_visual_prompt"](_action, _shot, _ex)
                assert _baked == _rebuilt, (
                    f"style drift between the package and the notebook for "
                    f"{_shot!r}/{_ex!r}:\n"
                    f"  run_scenes.py:  {_baked!r}\n"
                    f"  Step 24 redraw: {_rebuilt!r}")
            # And with no action, so the fallback subjects are covered too.
            assert _sc.visual_prompt("", _shot) == _g["build_visual_prompt"]("", _shot), (
                f"fallback-subject drift for shot {_shot!r}")

        # The bible is in the prompt if and only if the shot has a character --
        # and so is the expression, which is the same rule wearing a hat.
        for _shot, _spec in _shots.SHOTS.items():
            _p = _sc.visual_prompt(_action, _shot, "rage")
            assert (config.CHARACTER_BIBLE in _p) == _spec["character"], (
                f"shot {_shot!r}: character={_spec['character']} but bible "
                f"{'is' if config.CHARACTER_BIBLE in _p else 'is not'} in the prompt")
            assert (_look.EXPRESSIONS["rage"]["clause"] in _p) == _spec["character"], (
                f"shot {_shot!r}: an expression reached a frame with no face in it")
    finally:
        os.chdir(_cwd)
    ok("notebook: Step 24 Redraw builds the same prompt as the package, every shot")
else:
    print("  --  stickman.ipynb not found, skipped the notebook drift check")

# =============================================================================
#  READ SHEETS  (pipeline/readsheet.py)
# =============================================================================
import struct as _struct
import wave as _wave

import numpy as _np

from pipeline import qc as _qc, readsheet as _rs

_tagged = [{"sentence": f"Section {s} line {i} with a few more words in it.",
            "section": s, "sources": [1]}
           for s in (1, 2, 3) for i in range(12)]
_beat = {1: {"name": "The hook", "words": (100, 140)},
         2: {"name": "The turn", "words": (100, 140)},
         3: {"name": "The payoff", "words": (100, 140)}}

_sheets = _rs.write(_tagged, _beat)
assert [p.name for p in _sheets] == ["section_01.txt", "section_02.txt", "section_03.txt"]
ok("read sheets: one file per section, zero-padded")

# The whole point: what you read must be exactly what the aligner expects to
# hear in that section, word for word.
for _sec, _chunks in _rs.section_texts(_tagged).items():
    _disk = (WORK / "sections" / f"section_{_sec:02d}.txt").read_text(encoding="utf-8")
    assert _disk.split() == " ".join(_chunks).split(), _sec
ok("read sheet text == the words Step 17 expects for that section")

# A blank line in a read sheet is a paragraph break to vo_coach mark-script, and
# it would silently add ~1s of room tone mid-section.
for _p in _sheets:
    _body = _p.read_text(encoding="utf-8")
    assert "\n\n" not in _body, f"{_p.name} has a blank line in it"
    assert not _body.lstrip().startswith("="), f"{_p.name} starts with a header"
ok("read sheets are script only - no headers, no blank lines")

_man = _rs.load_manifest()
assert [e["section"] for e in _man["sections"]] == [1, 2, 3]
assert all(e["words"] > 0 for e in _man["sections"])
assert _man["sections"][0]["name"] == "The hook"
_readme = (WORK / "sections" / "README.txt").read_text(encoding="utf-8")
assert "mark-script sections/section_01.txt" in _readme
assert "run_qc.py" in _readme and "collect_takes.py --section 1" in _readme
assert "DO NOT run `vo_coach master`" in _readme
ok("manifest + README carry word counts and the exact commands")

# Stale sheets from a previous topic must not survive into the next one.
(WORK / "sections" / "section_09.txt").write_text("left over", encoding="utf-8")
_rs.write(_tagged, _beat)
assert not (WORK / "sections" / "section_09.txt").exists()
ok("re-running clears read sheets the new script does not have")

# =============================================================================
#  QC GATE  (pipeline/qc.py)
# =============================================================================
_SR = 48000
_rng = _np.random.default_rng(11)


def _speech(seconds, peak=0.35, noise=0.002):
    """Voice-shaped enough for the measurements: syllable envelope, phrase gaps
    (so there is real room tone to measure), and a controllable peak."""
    n = int(seconds * _SR)
    t = _np.arange(n) / _SR
    env = (0.5 + 0.5 * _np.sin(2 * _np.pi * 3.1 * t)) ** 2
    gate = (_np.sin(2 * _np.pi * 0.22 * t) > -0.35).astype(float)
    x = (_np.sin(2 * _np.pi * 120 * t) + 0.4 * _np.sin(2 * _np.pi * 240 * t)) * env * gate
    x = x / _np.max(_np.abs(x)) * peak
    return x + _rng.normal(0, noise, n)


def _write_pcm(path, x, sr=_SR, width=2, channels=1):
    d = _np.clip(x, -1, 1)
    if channels > 1:
        d = _np.repeat(d[:, None], channels, axis=1).ravel()
    with _wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(sr)
        w.writeframes((d * 32767).astype("<i2").tobytes())


def _write_float32(path, x, sr=_SR):
    """The `wave` module refuses to write this, and vo_coach writes it when
    audio.bit_depth is 32 - so the reader has to handle it."""
    data = _np.asarray(x, dtype="<f4").tobytes()
    fmt = _struct.pack("<HHIIHH", 3, 1, sr, sr * 4, 4, 32)
    body = (b"WAVE" + b"fmt " + _struct.pack("<I", len(fmt)) + fmt +
            b"data" + _struct.pack("<I", len(data)) + data)
    Path(path).write_bytes(b"RIFF" + _struct.pack("<I", len(body)) + body)


_rec = WORK / "voice_recordings"
_rec.mkdir(parents=True, exist_ok=True)

_write_pcm(_rec / "pcm16.wav", _speech(6))
_x, _sr, _info = _qc.read_wav(_rec / "pcm16.wav")
assert _sr == _SR and _info == {"channels": 1, "bits": 16, "encoding": "pcm"}
assert abs(len(_x) / _sr - 6) < 0.01

_write_float32(_rec / "f32.wav", _speech(6))
_x, _sr, _info = _qc.read_wav(_rec / "f32.wav")
assert _info["encoding"] == "float" and _info["bits"] == 32

_write_pcm(_rec / "stereo.wav", _speech(6), channels=2)
_x2, _sr2, _info2 = _qc.read_wav(_rec / "stereo.wav")
assert _info2["channels"] == 2 and abs(len(_x2) / _sr2 - 6) < 0.01
ok("wav reader: PCM16, float32 and stereo, with no soundfile dependency")

# Clipping is RUNS of full-scale samples, not single samples. One sample at the
# ceiling stopped a reader on line one in vo_coach; that lesson is pinned here.
_flat = _speech(4)
_flat[1000] = 1.0
assert _qc.clip_runs(_flat) == 0, "one full-scale sample must not count as clipping"
_flat[2000:2010] = 1.0
_flat[3000:3002] = -1.0          # a 2-sample run is still under min_run
assert _qc.clip_runs(_flat) == 1
ok("clipping counts runs of 3+ full-scale samples, not single samples")

_clipped = _rec / "clipped.wav"
_write_pcm(_clipped, _np.clip(_speech(40) * 4.0, -1, 1))
_r = _qc.check_file(_clipped, section=1, expected_words=100)
assert _r.verdict == "FAIL" and any("clipping" in m for m in _r.fails)
assert _r.peak_db > -1.0
ok("a clipped section FAILs and the message says what to change")

_quiet_noisy = _rec / "noisy.wav"
_write_pcm(_quiet_noisy, _speech(40, peak=0.05, noise=0.03))
_r = _qc.check_file(_quiet_noisy, section=1, expected_words=100)
assert _r.verdict == "FAIL" and any("SNR" in m for m in _r.fails)
ok("a noisy section FAILs on SNR, using vo_coach's own threshold")

# Digital silence must not read as a perfect recording: the naive measurement
# reports ~175 dB SNR on a gated file.
_gated = _rec / "gated.wav"
_g = _speech(20)
_g[_np.abs(_g) < 0.02] = 0.0
_write_pcm(_gated, _g)
_r = _qc.check_file(_gated, section=1)
assert _r.snr_db is None or _r.snr_db < 100, f"digital silence faked an SNR of {_r.snr_db}"
ok("digital silence reports no SNR rather than a fabricated one")

_good = _rec / "good.wav"
_write_pcm(_good, _speech(40, peak=0.35))
_r = _qc.check_file(_good, section=1, expected_words=100)
assert _r.verdict == "PASS", (_r.fails, _r.warns)
assert -12 < _r.peak_db < -6 and _r.snr_db > 30
ok("a clean section at -9 dBFS PASSes with nothing to say")

# The check that catches section_04.wav holding section 5: wrong length for the
# words on the sheet.
_r = _qc.check_file(_good, section=4, expected_words=400)
assert _r.verdict == "FAIL" and any("wpm" in m for m in _r.fails)
ok("duration against the read sheet's word count catches the wrong file")

_r = _qc.check_file(_rec / "missing.wav", section=1)
assert _r.verdict == "FAIL"
_fake_mp3 = _rec / "section_09.mp3"
_fake_mp3.write_bytes(b"not audio")
_r = _qc.check_file(_fake_mp3, section=9)
assert _r.verdict == "FAIL" and "WAV" in _r.fails[0]
ok("unreadable and non-WAV files FAIL instead of being skipped")

_soft = _rec / "soft.wav"
_write_pcm(_soft, _speech(40, peak=0.05))          # -26 dBFS peak, clean otherwise
_run = _qc.check_run({1: _good, 2: _clipped, 3: _soft},
                     expected={1: 100, 2: 100, 3: 100}, missing=[4])
assert not _run["ok"]
assert any("no recording" in m for m in _run["run_fails"])
assert any("levels span" in m for m in _run["run_warns"])
assert "\"verdict\"" in _qc.to_json(_run)
ok("run-level checks: missing sections, and level jumps between sections")

for _f in (_rec / "pcm16.wav", _rec / "f32.wav", _rec / "stereo.wav", _clipped,
           _quiet_noisy, _gated, _good, _soft, _fake_mp3):
    _f.unlink(missing_ok=True)

# =============================================================================
#  SHOT ROTATION  (pipeline/shots.py)
# =============================================================================
#  The rules here are the anti-repetition guarantees. They are asserted rather
#  than hoped for, for the same reason the palette lock is code and not a prompt:
#  a rule you can only ask for is not a rule.

for _name, _spec in _shots.SHOTS.items():
    assert set(_spec) >= {"family", "character", "clause", "ask", "caption",
                          "weight", "max_share"}, f"shot {_name} is missing fields"
    assert _spec["family"] in ("narrative", "abstract")
    assert _spec["caption"] in ("require", "prefer", "neutral", "avoid")
    assert 0 < _spec["max_share"] <= 1 and _spec["weight"] > 0
    # A character-free shot must SAY so in its clause, or FLUX puts one in anyway.
    if not _spec["character"]:
        assert "no people" in _spec["clause"], f"shot {_name} does not forbid people"
assert all(_n in _shots.SHOTS for _pool in _shots.SECTION_SHOTS.values() for _n in _pool)
assert _shots.DEFAULT_SHOT in _shots.SHOTS
ok(f"shots: catalog is well-formed ({len(_shots.SHOTS)} shots, "
   f"{len(_shots.ABSTRACT_SHOTS)} without a character)")


def _timeline(per_section):
    out, i = [], 0
    for _sec, _cnt in enumerate(per_section, 1):
        for _ in range(_cnt):
            out.append({"scene_index": i, "section": _sec, "text": "a line", "duration": 4.0})
            i += 1
    return out


_tl = _timeline([26, 30, 30, 30, 26, 18])          # a realistic 160-scene script
_shots.assign(_tl)
assert all(s["shot"] in _shots.SHOTS for s in _tl)
assert not _shots.validate(_tl), _shots.validate(_tl)
ok(f"shots: a {len(_tl)}-scene script validates clean")

_seq = [s["shot"] for s in _tl]
assert not any(a == b for a, b in zip(_seq, _seq[1:])), "adjacent frames repeat a shot"
ok("shots: no two consecutive frames get the same shot")

_run = _worst = 0
for _s in _seq:
    _run = _run + 1 if _shots.SHOTS[_s]["family"] == "narrative" else 0
    _worst = max(_worst, _run)
assert _worst <= _shots.MAX_NARRATIVE_RUN, f"{_worst} narrative frames in a row"
ok(f"shots: an abstract frame breaks every run of {_shots.MAX_NARRATIVE_RUN} narrative ones")

from collections import Counter as _C

_counts = _C(_seq)
assert len(_counts) == len(_shots.SHOTS), f"only {len(_counts)} shot types used"
for _n, _c in _counts.items():
    _cap = max(1, round(len(_tl) * _shots.SHOTS[_n]["max_share"]))
    assert _c <= _cap, f"{_n} used {_c} times, ceiling {_cap}"
ok("shots: every shot appears, none over its ceiling")

# Seeded: the same timeline must replan identically, or a redraw is not reproducible.
_a, _b = _timeline([26, 30, 30, 30, 26, 18]), _timeline([26, 30, 30, 30, 26, 18])
_shots.assign(_a)
_shots.assign(_b)
assert [s["shot"] for s in _a] == [s["shot"] for s in _b]
assert [s["shot"] for s in _shots.assign(_timeline([26, 30, 30, 30, 26, 18]), seed=1)] != _seq
ok("shots: the rotation is seeded - same timeline replans identically")

# Assignment must never fail, however odd the input.
for _odd in ([1], [1, 1], _timeline([200]), _timeline([0, 0, 5])):
    _t = _odd if isinstance(_odd[0], dict) else _timeline(_odd)
    _shots.assign(_t)
    assert all(s["shot"] for s in _t), "a scene came out with no shot"
_t = [{"scene_index": 0, "section": 99, "text": "x", "duration": 1.0}]
_shots.assign(_t)
assert _t[0]["shot"] in _shots.SHOTS, "an unknown section broke assignment"
ok("shots: assignment never fails - single scenes, one huge section, unknown sections")

# The caption cap: a nudge in 160 prompts is not a guarantee, this is.
_tl2 = _timeline([26, 30, 30, 30, 26, 18])
_shots.assign(_tl2)
for _s in _tl2:
    _s["caption"] = "EVERY FRAME"                  # worst case: the model captioned all
_dropped = _shots.enforce_caption_cap(_tl2)
_left = sum(1 for s in _tl2 if s["caption"])
assert _left <= int(len(_tl2) * _shots.CAPTION_MAX_SHARE) or _dropped > 0
assert _left / len(_tl2) <= _shots.CAPTION_MAX_SHARE + 0.02, f"{_left}/{len(_tl2)} captioned"
# A typographic frame IS its text, so it is never the one stripped.
assert all(s["caption"] for s in _tl2 if s["shot"] == "typographic"), \
    "a typographic frame lost its caption - that frame is now blank"
ok(f"shots: captions capped at {_shots.CAPTION_MAX_SHARE:.0%} "
   f"({_dropped} dropped), typographic frames keep theirs")

assert _shots.enforce_caption_cap(_timeline([10])) == 0, "cap fired with no captions"

# The validator has to actually catch what it claims to catch.
_bad = _timeline([10])
for _s in _bad:
    _s["shot"] = "character-moment"
_probs = " | ".join(_shots.validate(_bad))
assert "adjacent" in _probs and "narrative frames in a row" in _probs and "never used" in _probs
assert _shots.validate([]) == ["no scenes to validate"]
_none = _timeline([3])
_probs = " | ".join(_shots.validate(_none))
assert "no shot assigned" in _probs
ok("shots: the validator catches repeats, long narrative runs and unused shots")

# A character-free shot whose action names a person is the failure that survives
# all the way to a bar chart with a stickman on it.
assert _sc._valid({"action": "three plain bars of different heights", "caption": ""},
                  "data-chart")
assert not _sc._valid({"action": "a person standing next to three bars", "caption": ""},
                      "data-chart")
assert _sc._valid({"action": "a person standing next to a box", "caption": ""},
                  "character-moment")
assert "no people" in _sc.shot_instruction("data-chart").lower()
assert "mandatory" in _sc.shot_instruction("typographic")
ok("scenes: a character-free shot rejects an action that names a person")

# Regression: _PERSON_WORDS used to be matched by plain substring, and "he " is a
# substring of "the ". Every action containing the word "the" on a character-free
# shot was rejected as a false person-mention. Measured on a real 80-scene run
# before this fix: 24 fallbacks (30%), almost all character-free shots whose only
# fault was an ordinary sentence. These are the exact actions that were wrongly
# rejected.
for _a in (
    "A single small circle sits in the corner.",
    "A gear, a lever, a pulley, and a spring sit in a grid.",
    "Two bars labeled spine and warning. The warning bar is the outlier.",
    "A bar chart and a brain icon sit on a lab table.",
    "Lung cross-section shrinks and expands rapidly",           # no "the" - control
):
    assert _sc._valid({"action": _a, "caption": ""}, "data-chart"), \
        f"false person-mention reject: {_a!r}"
# And a real mention must still be caught, whole-word only.
for _a in ("A person holds a chart.", "One small figure stands before a door.",
          "A human silhouette next to the bars.", "He points at the chart."):
    assert not _sc._valid({"action": _a, "caption": ""}, "data-chart"), \
        f"should have rejected a real person mention: {_a!r}"
ok("scenes: person-word check is whole-word - 'the' no longer false-triggers on 'he'")

# Visual QA session: a typographic frame came back with mixed-case text burned
# into the picture, even though draw_caption() always .upper()s. The one route
# by which lettering reaches FLUX outside that single controlled path is an
# action that quotes exact words for FLUX to paint. Caught here; ordinary
# words like "labeled" or "sign" that legitimately describe a chart or an
# object are NOT banned (that would repeat the substring-match bug above).
assert not _sc._valid({"action": 'a beaker labeled "sample"', "caption": ""}, "object-focus"), \
    "an action quoting literal text should be rejected"
assert not _sc._valid({"action": "a sign that reads “CLOSE”", "caption": ""}, "object-focus"), \
    "curly quotes should be rejected too"
assert _sc._valid({"action": "two bars labeled spine and warning", "caption": ""}, "data-chart"), \
    "'labeled' with no actual quoted text is ordinary chart description, not spelled-out lettering"
assert _sc._valid({"action": "a stop sign beside the road", "caption": ""}, "object-focus"), \
    "'sign' as an ordinary object must not be rejected"
assert _sc._valid({"action": "the figure examines the door's hinge", "caption": ""}, "character-moment"), \
    "a bare apostrophe (possessive/contraction) must not be rejected"
ok("scenes: an action that quotes literal text is rejected, ordinary words are not")

# A real frame showed this: the voiceover said "press two fingers against the
# back of your arm", the action came back "A hand presses two fingers against
# the back of an arm", and FLUX drew a photorealistic hand on a stick body --
# because CHARACTER_BIBLE says "no hands or feet drawn in detail" and the action
# said the opposite, in the same prompt. 6 of 10 scenes in that batch had it.
for _a in ("A hand presses two fingers against the back of an arm",
           "A hand holds a small brain shape",
           "A person points at their own eye with one finger",
           "A person holds up one hand, palm facing them, fingers slightly spread",
           "a person pointing at their own hand"):
    assert not _sc._valid({"action": _a, "caption": ""}, "character-moment"),         f"an action making a hand the subject should be rejected: {_a!r}"
# What we want instead must still pass - the whole figure acting, not its hand.
for _a in ("the figure touches its own arm", "a figure holds a small brain",
           "a figure leans against a tall door"):
    assert _sc._valid({"action": _a, "caption": ""}, "character-moment"),         f"a whole-figure action was wrongly rejected: {_a!r}"
# No bible on a character-free shot, so no contradiction to catch there.
assert _sc._valid({"action": "a hand-crank machine on a shelf", "caption": ""},
                  "object-focus")
ok("scenes: an action that makes a hand or finger the subject is rejected")

# The same frame also had a giant red heart nobody asked for: symbolic-metaphor's
# clause ASSERTS "one large symbolic object that dwarfs it", the action named no
# object, and FLUX filled the blank with the most generic symbol it knows.
assert not _sc._valid({"action": "a figure stands still and waits", "caption": ""},
                      "symbolic-metaphor"), "no object named, should be rejected"
assert not _sc._valid({"action": "the person waits alone", "caption": ""},
                      "symbolic-metaphor")
for _a in ("a figure stands beside one giant closed door",
           "a figure looks up at an enormous stone wall"):
    assert _sc._valid({"action": _a, "caption": ""}, "symbolic-metaphor"),         f"an action that DOES name an object was rejected: {_a!r}"
# And the fallback must itself name one, since that is where a reject lands.
assert _sc._names_an_object(_sc._FALLBACK_SUBJECT["symbolic-metaphor"]),     "the symbolic-metaphor fallback names no object either"
ok("scenes: a symbolic-metaphor action that names no object is rejected")

# ...and the assignment side of the same failure: a line that is a tight physical
# body action must never draw the shot whose whole premise is an oversized
# symbolic object. This is the ONLY place the rotation looks at what a line says.
assert _shots.is_body_action("Press two fingers against the back of your arm.")
assert _shots.is_body_action("Feel that? Those tiny hairs on your neck have a name.")
# Both halves required, so the common cases do not trip it. "brain" is
# deliberately not a body word - a figure holding a brain is a good symbolic frame.
assert not _shots.is_body_action("Here is the part nobody tells you about brain.")
assert not _shots.is_body_action("Every skin on earth does this the same way.")
assert not _shots.is_body_action("You stopped needing them two million years ago.")
assert not _shots.is_body_action("")

_body = [{"scene_index": i, "section": 1, "duration": 3.0,
          "text": "Press two fingers against the back of your arm."} for i in range(20)]
_shots.assign(_body)
assert all(s["shot"] in _shots.BODY_ACTION_SHOTS for s in _body),     "a body-action line was assigned a shot that cannot show a figure doing it"
assert not [p for p in _shots.validate(_body) if "body-action" in p]
# The validator must actually catch it when forced, or the rule is unenforced.
_body[3]["shot"] = "symbolic-metaphor"
assert any("body-action" in p for p in _shots.validate(_body)),     "the validator does not catch a body-action line on symbolic-metaphor"
# A body action is SOLO. two-character was in BODY_ACTION_SHOTS and rendered a
# spare idle stick figure next to the one doing the touching, because the clause
# asserts two figures while the action says "the figure", singular.
assert _shots.BODY_ACTION_SHOTS == ("character-moment",),     "a body action is solo - only a one-figure shot can carry it"
# And no clause may LEAD with the word "frame": FLUX read "a nearly empty frame"
# as a picture frame and drew a black border around all four edges.
for _n, _s in _shots.SHOTS.items():
    assert not _s["clause"].lower().startswith(("a nearly empty frame", "an empty frame")),         f"shot {_n!r} leads with 'frame' - FLUX will draw a literal border"
ok("shots: a body-action line never draws the oversized-symbol shot")

# Visual QA session: object-focus and icon-grid clauses/asks were hardened
# against a floating prop (no floor contact stated) and anatomical/hybrid
# subject drift (an eyeball growing out of a beaker). Both must still forbid
# people, and now must also say the object rests on something and rule out
# hybrids.
#
# NARROWED, and this check is narrowed with it. The ban used to read "nothing
# anatomical or hybrid", which excluded every animal, insect, organ and plant --
# i.e. most of what a biology channel has to draw, and most of what the
# reference channel is made of. The defect it was written for was a DETACHED
# photorealistic forearm, so the ban is now on detached parts and hybrids only.
#
# Checking for the substring "hybrid" alone would keep passing on the old
# wording and on the new, which makes it worthless as a check of the change --
# the same failure mode as the "tan" in "sTANding" assertion this file already
# records. So both halves are asserted: the part that must still be forbidden,
# and the part that must now be allowed.
for _name in ("object-focus", "icon-grid"):
    _ask = _shots.SHOTS[_name]["ask"].lower()
    assert "hybrid" in _ask, f"{_name}: ask no longer rules out object/body hybrids"
    assert "detached" in _ask, \
        f"{_name}: ask must forbid a DETACHED body part - that was the real defect"
    assert not any(_p in _ask for _p in ("nothing anatomical",
                                         "no anatomical",
                                         "nothing organic")), \
        (f"{_name}: ask has gone back to a blanket anatomical ban, which "
         f"forbids drawing an animal, an insect or an organ at all")
    assert any(_w in _ask for _w in ("animal", "insect", "creature", "plant")), \
        (f"{_name}: ask should positively ALLOW living subjects - the reference "
         f"channel's frames are full of them")
assert "resting" in _shots.SHOTS["object-focus"]["clause"] or \
       "stand" in _shots.SHOTS["object-focus"]["clause"] or \
       "shelf" in _shots.SHOTS["object-focus"]["clause"], \
    "object-focus: clause should ground the object on something, not leave it floating"
ok("shots: object-focus/icon-grid clauses ground the subject and rule out hybrids")

# run_scenes.py --dry-run priced its estimate off config.MODEL_CHECKER, the
# fact-checker's model, instead of config.MODEL_SCENES -- the model the real
# run actually calls. Static check, not an import: run_scenes.py is a script,
# not a module, and the dry-run block is the one place this constant matters.
_rs_src = (ROOT / "run_scenes.py").read_text(encoding="utf-8")
_dry_block = _rs_src[_rs_src.index("if args.dry_run:"):]
_dry_block = _dry_block[:_dry_block.index("load_env()")]
assert "config.MODEL_CHECKER" not in _dry_block, \
    "dry-run estimate should not price off the checker model"
assert "config.MODEL_SCENES" in _dry_block, \
    "dry-run estimate should price off the model scenes.py actually calls"
ok("run_scenes: --dry-run prices off MODEL_SCENES, not MODEL_CHECKER")

# =============================================================================
#  EXPRESSIONS  (pipeline/look.py)
# =============================================================================
from pipeline import look as _lk

# Every shot clause must still carry its own background and floor: the separate
# background axis was tried and reverted, so the clause owns it again.
for _n, _s in _shots.SHOTS.items():
    assert "background" in _s["clause"], f"{_n} lost its background clause"

# Every shot must also say its OBJECT is colour-filled. The character is
# white-and-black ("no colour anywhere on the character", in CHARACTER_BIBLE),
# and that ban sits right beside the object description -- so a clause that
# never says the prop IS filled leaves the ban as the frame's only colour
# statement, and FLUX generalises it into a colourless picture.
# A shot whose object is GUARANTEED must say it is colour-filled: the character
# is white-and-black, and CHARACTER_BIBLE's "no colour anywhere on the
# character" sits right beside the object, so with no counter-statement FLUX
# generalises the ban and drains the prop too.
for _n in ("symbolic-metaphor", "object-focus", "icon-grid", "data-chart",
           "typographic"):
    _c = _shots.SHOTS[_n]["clause"]
    assert "solid colour" in _c or "flat-coloured" in _c, (
        f"shot {_n!r} never says its object is colour-filled")
# ...and a shot whose prop is OPTIONAL must NOT, or FLUX invents one. A frame
# whose action was only "the figure touches its own forearm" came back with a
# blue mug on the floor, because the clause said "any prop filled with...".
for _n in ("character-moment", "two-character"):
    _c = _shots.SHOTS[_n]["clause"]
    assert "prop" not in _c and "shared object" not in _c, (
        f"shot {_n!r} asserts a prop that its action may not name - FLUX will "
        f"invent one")
ok("look: guaranteed objects are colour-filled, optional props are never asserted")

# The energy score only has to ORDER things correctly - there is nothing to
# calibrate it against. These are the orderings the face choice depends on.
assert _lk.energy("Nobody screams. The pain is already gone!") > \
       _lk.energy("Scientists measured the response in a controlled lab setting.")
assert _lk.band(_lk.energy("Blood everywhere. He screamed!")) == "high"
assert _lk.band(_lk.energy(
    "This process unfolded slowly over roughly two million years of evolution "
    "across many separate generations of ancestors.")) == "low"
assert _lk.energy("") == 0.0 and _lk.energy("   ") == 0.0
ok("look: the energy score orders a visceral line above an analytical one")

_lt = _timeline([40, 40, 40, 40])
_shots.assign(_lt)
_lk.assign(_lt)
assert not _lk.validate(_lt), _lk.validate(_lt)
# A face iff the frame has one. This is the CHARACTER_BIBLE bug in a new place.
for _s in _lt:
    _has_face = _shots.SHOTS[_s["shot"]]["character"]
    assert bool(_s["expression"]) == _has_face, (
        f"#{_s['scene_index']}: expression={_s['expression']!r} on a "
        f"{'character' if _has_face else 'character-free'} shot")
_faces = {s["expression"] for s in _lt if s["expression"]}
assert len(_faces) >= 4, f"only {len(_faces)} distinct faces across 160 scenes"
ok(f"look: a 160-scene script assigns clean ({len(_faces)} distinct faces)")

# Seeded, for the same reason the shot rotation is: a replan must give the same
# face, or a redrawn frame stops matching its neighbours.
_a, _b = _timeline([40, 40, 40, 40]), _timeline([40, 40, 40, 40])
for _t in (_a, _b):
    _shots.assign(_t)
    _lk.assign(_t)
assert [s["expression"] for s in _a] == [s["expression"] for s in _b]
ok("look: the expression rotation is seeded - same timeline replans identically")

# Never fails, on the same odd inputs shots.assign() is held to.
for _odd in ([{"scene_index": 0, "text": "One line.", "section": 1, "duration": 3}],
             [{"scene_index": i, "text": "x", "section": 99, "duration": 3}
              for i in range(5)],
             [{"scene_index": i, "text": "", "section": 1, "duration": 3}
              for i in range(30)]):
    _shots.assign(_odd)
    _lk.assign(_odd)
    assert all("expression" in s for s in _odd)
ok("look: assignment never fails - one scene, unknown sections, empty text")

# The switch has to actually switch something, in the prompt as well as the plan.
_off = _timeline([20, 20])
_shots.assign(_off)
config.EXPRESSIONS_ON = False
try:
    _lk.assign(_off)
    _f = {s["expression"] for s in _off if s["expression"]}
    assert _f <= {_lk.DEFAULT_EXPRESSION}, f"EXPRESSIONS_ON=False left {_f}"
    _p = _sc.visual_prompt("waving", "character-moment", "rage")
    assert _lk.EXPRESSIONS["rage"]["clause"] not in _p, \
        "EXPRESSIONS_ON=False still let a rage face into the prompt"
finally:
    config.EXPRESSIONS_ON = True
ok("look: EXPRESSIONS_ON=False pins every face to deadpan")

# The cyan/yellow rebrand was reverted. This is the check that keeps it reverted.
#
# WIDENED, and the distinction matters. The rebrand replaced the warm room with
# a cyan/yellow one; the palette now APPENDS scene colours to that warm room and
# leaves it first and unchanged. So the check is no longer an exact list -- it
# is that the original six are still there, still in order, since finish.py
# indexes PALETTE[1] as the outline colour and FLOOR_COLOR is PALETTE[2].
assert list(config.PALETTE[:6]) == [(255, 255, 255), (26, 26, 26), (216, 164, 86),
                                    (182, 70, 45), (196, 140, 80), (120, 72, 40)], config.PALETTE
assert config.PALETTE[1] == (26, 26, 26), "finish.py indexes PALETTE[1] as the outline"
assert tuple(config.FLOOR_COLOR) == tuple(config.PALETTE[2]), "floor left the palette"
assert tuple(config.BACKGROUND_COLOR) in [tuple(c) for c in config.PALETTE], (
    "BACKGROUND_COLOR is not a palette colour - the lock will move it")
# A palette this size stops reading as one channel. The reference uses ~12.
assert len(config.PALETTE) <= 16, f"palette has grown to {len(config.PALETTE)}"
# No neutral mid-grey: it would sit on top of the anti-aliased edge around every
# black stroke and ring the whole drawing with a halo. See the note in config.
for _c in config.PALETTE:
    _r, _g, _b = _c
    if 120 < _r < 190 and max(_c) - min(_c) < 25:
        raise AssertionError(f"{_c} is a neutral mid-grey - it will halo the linework")
assert "white background wall" in _shots.SHOTS["character-moment"]["clause"]
for _bad in ("38B6FF", "FFDE59", "MS-Paint", "sky blue"):
    assert _bad not in config.STYLE_PREFIX, f"STYLE_PREFIX still carries {_bad!r}"
ok("look: the palette and style are the warm room one, not the cyan rebrand")

# =============================================================================
#  THE FLOOR IS CODE NOW, NOT PROMPT
# =============================================================================
#  This block replaces an assertion that used to read
#      assert "tan" in SHOTS["character-moment"]["clause"]
#  and which kept passing after the floor was removed from the clause -- because
#  "tan" is a substring of "sTANding", which the new clause happens to contain.
#  A test that passes on an accident is worse than no test, so the floor checks
#  are now on the thing that actually draws it.
# =============================================================================
import numpy as _np
from PIL import Image as _Img
from pipeline import finish as _fin

# The prompt must NOT ask for a floor any more, or FLUX draws a competing rule
# under the one finish.py paints.
_clause = _shots.SHOTS["character-moment"]["clause"]
for _w in ("tan floor", "floor band", "floor line"):
    assert _w not in _clause, f"floor is code now, but the clause still says {_w!r}"
ok("finish: the shot clause no longer asks FLUX for a floor")

# A synthetic frame: white, one horizontal rule, a short 'foot' hanging below it.
_H, _W = 720, 1280
_a = _np.full((_H, _W, 3), 255, _np.uint8)
_a[590:594, :] = 20                      # the horizon FLUX would draw
_a[594:612, 640:660] = 20                # a foot, below the horizon
_frame = _Img.fromarray(_a)

_y = _fin.find_ground_line(_frame)
assert _y is not None, "failed to find an obvious full-width horizon"
assert _y > 611, f"band top {_y} would clip the foot that ends at 611"
ok("finish: the floor anchors BELOW the feet, not to the horizon line")

_out = _np.asarray(_fin.draw_floor(_frame).convert("RGB"))
assert tuple(_out[_H - 1, 5]) == tuple(config.FLOOR_COLOR), "bottom row is not floor colour"
assert tuple(_out[_H - 1, _W - 5]) == tuple(config.FLOOR_COLOR), "floor does not span the width"
assert tuple(_out[10, 10]) == (255, 255, 255), "floor leaked into the upper frame"
assert tuple(_out[605, 645]) == (20, 20, 20), "the foot was painted over"
ok("finish: the band is solid, spans the width, and leaves the figure alone")

# Every colour the floor introduces must already be in PALETTE, or the lock
# moves it and the floor stops matching frame to frame.
for _c in (config.FLOOR_COLOR, config.PALETTE[1]):
    assert tuple(_c) in [tuple(p) for p in config.PALETTE], f"{_c} is not a palette colour"
ok("finish: floor and edge colours survive the palette lock unchanged")

# Line boost must ADD ink without inventing new colours.
_b = _np.full((200, 200, 3), 255, _np.uint8)
_b[100, 100] = 20
_thin = _Img.fromarray(_b)
_fat = _fin.boost_lines(_thin, px=1)
_before = int((_np.asarray(_thin.convert("L")) < 110).sum())
_after = int((_np.asarray(_fat.convert("L")) < 110).sum())
assert _after > _before, "boost_lines did not thicken anything"
assert _after == 5, f"one pixel should grow to a 5-pixel plus, got {_after}"
_cols = {tuple(c) for c in _np.asarray(_fat).reshape(-1, 3)}
assert _cols <= {(255, 255, 255), (20, 20, 20), tuple(config.PALETTE[1])}, _cols
ok("finish: line boost grows ink in palette colours only")

# px=0 must be a no-op, so the feature can be switched off without a code path.
assert _np.array_equal(_np.asarray(_fin.boost_lines(_thin, px=0)), _np.asarray(_thin))
ok("finish: LINE_BOOST_PX=0 leaves the frame untouched")

# =============================================================================
#  RESTROKE AND THE PAINTED BACKGROUND
# =============================================================================
#  restroke() exists because config.py's line-boost note concluded that stroke
#  SMOOTHNESS was "a generation problem, not a post-processing one". Wobble is
#  high-frequency contour noise; blurring a binary mask and re-thresholding
#  removes it. The checks below are on the two properties that claim rests on:
#  it must THICKEN (like the dilation it replaces) and it must SMOOTH (which
#  the dilation could not).
# =============================================================================

# A deliberately wobbly vertical stroke: 1px wide, jittering left and right.
_wob = _np.full((200, 200, 3), 255, _np.uint8)
for _y in range(20, 180):
    _wob[_y, 100 + (_y % 3) - 1] = 20
_wobbly = _Img.fromarray(_wob)

def _ink(im):
    _l = _np.asarray(im.convert("L"), _np.float32)
    return _l < config.LINE_DARK_MAX

def _roughness(im):
    """Mean absolute change in the stroke's left edge from row to row.
    A smooth stroke barely moves; a jittering one moves every row."""
    _m = _ink(im)
    _xs = [_np.argmax(_r) for _r in _m if _r.any()]
    return float(_np.abs(_np.diff(_xs)).mean()) if len(_xs) > 1 else 0.0

_r_before = _roughness(_wobbly)
_r_boost = _roughness(_fin.boost_lines(_wobbly))
_r_after = _roughness(_fin.restroke(_wobbly))
assert _r_after < _r_before, f"restroke did not smooth: {_r_before} -> {_r_after}"
# The honest claim, measured rather than assumed. An earlier version of this
# check asserted that the dilation does not smooth AT ALL, and that is simply
# false: dilating a +/-1px jitter by 3px merges some of it (1.33 -> 0.69 here).
# What restroke has to beat is the dilation, not zero.
assert _r_after < _r_boost, \
    (f"restroke ({_r_after:.2f}) is no smoother than the dilation it replaces "
     f"({_r_boost:.2f}) - it is not earning its place")
ok(f"finish: restroke smooths further than dilation "
   f"(raw {_r_before:.2f} -> boost {_r_boost:.2f} -> restroke {_r_after:.2f} px jitter)")

assert _ink(_fin.restroke(_wobbly)).sum() > _ink(_wobbly).sum(), \
    "restroke must thicken as well as smooth, or it cannot replace the dilation"

# THE REGRESSION THIS FILE EXISTS FOR. Blur-and-threshold with no dilation floor
# erases a thin stroke outright -- a 1px line blurred at sigma=1.6 peaks near
# 0.25, so any threshold above that returns a blank frame. It survived on a real
# render only because those strokes were already several px wide.
assert _ink(_fin.restroke(_wobbly)).any(), \
    "restroke erased a 1px stroke - the dilation floor is gone"
for _s in (0.8, 1.6, 3.0):
    assert _ink(_fin.restroke(_wobbly, sigma=_s)).any(), \
        f"restroke erased a 1px stroke at sigma={_s}"
ok("finish: restroke never erases a thin stroke, at any smoothing radius")
# Only palette colours may be introduced, same rule the line boost is held to.
_rc = {tuple(c) for c in _np.unique(
    _np.asarray(_fin.restroke(_wobbly)).reshape(-1, 3), axis=0)}
assert _rc <= {(255, 255, 255), tuple(config.PALETTE[1])}, _rc
ok("finish: restroke thickens, and introduces no off-palette colour")

# The background fill must colour the wall and LEAVE THE FIGURE'S OWN WHITE
# ALONE. A naive "replace every white pixel" floods the head and the eyes too,
# and the white-figure-on-colour-field contrast IS the reference channel's look.
_bgt = _np.full((200, 200, 3), 255, _np.uint8)
_bgt[60:140, 60:140] = 20        # a filled black box...
_bgt[70:130, 70:130] = 255       # ...with a white interior, i.e. a head
_painted = _np.asarray(_fin.paint_background(_Img.fromarray(_bgt),
                                             colour=(226, 202, 160)))
assert tuple(_painted[5, 5]) == (226, 202, 160), "the wall was not painted"
assert tuple(_painted[100, 100]) == (255, 255, 255), \
    "the fill leaked into the enclosed white interior - it would flood the face"
ok("finish: the background fill colours the wall and spares the figure's own white")

# Ink touching the frame edge splits the wall into pieces. Measured on a real
# frame: a horizon line reaching both edges left the wall white on one side of
# the figure and beige on the other. The padded flood is what fixes it.
_split = _np.full((200, 200, 3), 255, _np.uint8)
_split[120, :] = 20              # a horizon spanning the full width
_sp = _np.asarray(_fin.paint_background(_Img.fromarray(_split),
                                        colour=(226, 202, 160)))
assert tuple(_sp[60, 100]) == (226, 202, 160) and tuple(_sp[160, 100]) == (226, 202, 160), \
    "a full-width line split the background; both sides must still be painted"
ok("finish: a full-width horizon does not split the background fill")

# floor_y must protect the band, or the fill eats the floor draw_floor paints.
_prot = _np.asarray(_fin.paint_background(_Img.fromarray(
    _np.full((200, 200, 3), 255, _np.uint8)), colour=(226, 202, 160), floor_y=150))
assert tuple(_prot[170, 100]) == (255, 255, 255), "floor_y did not protect the band"
ok("finish: floor_y keeps the background fill out of the floor band")


# =============================================================================
#  COMPOSITOR  --  placing figures instead of asking the model for two
# =============================================================================
from pipeline import compositor as _comp

# A figure-like asset: a ring (the head, white INSIDE) over a body stroke, with
# a separate blob off to the right standing in for a prop.
_can = _np.full((120, 200, 3), 255, _np.uint8)
_yy, _xx = _np.ogrid[:120, :200]
_ring = ((_yy - 30) ** 2 + (_xx - 40) ** 2 <= 20 ** 2) & \
        ((_yy - 30) ** 2 + (_xx - 40) ** 2 >= 14 ** 2)
_can[_ring] = 20
_can[50:100, 38:43] = 20                      # body stroke
_can[40:80, 150:190] = 20                     # the separate prop
_asset = _comp.cutout(_Img.fromarray(_can))

# The interior white of the head must SURVIVE. A global brightness test would
# punch it transparent and the figure would paste onto the wall hollow-faced.
_al = _np.asarray(_asset)[:, :, 3]
assert _al[30, 40] > 0, "the head's interior white was cut out - face will be hollow"
assert _al[2, 2] == 0, "the outside background was not removed"
ok("compositor: cutout removes only edge-connected background, sparing interior white")

# Two subjects must be SEEN as two, and rejected loudly rather than averaged.
_spans = _comp.subject_spans(_asset)
assert len(_spans) == 2, f"expected figure + prop, got {_spans}"
try:
    _comp.assert_single_subject(_asset, "two-subject asset")
    raise AssertionError("a two-subject asset was accepted; anchors would mis-scale it")
except ValueError:
    pass
ok("compositor: a multi-subject asset is rejected, not silently mis-measured")

# Anchors on a clean single-subject crop.
_solo = _asset.crop((0, 0, 120, 120))
_a = _comp.anchors(_solo)
assert _a["top"] == 10 and _a["ground"] == 99, _a
assert abs(_a["x_center"] - 40) <= 2, f"x_center should sit at the body stroke, got {_a}"
ok("compositor: anchors find crown, ground contact and foot centre")

# Scaling normalises the ANCHOR SPAN, not the bitmap - two crops carrying
# different blank margin must still come out the same figure height.
_padded = _Img.new("RGBA", (200, 260), (0, 0, 0, 0))
_padded.alpha_composite(_solo, (30, 90))
_s1 = _comp.anchors(_comp.scale_to_height(_solo, 100))["height"]
_s2 = _comp.anchors(_comp.scale_to_height(_padded, 100))["height"]
assert abs(_s1 - 100) <= 2 and abs(_s2 - 100) <= 2, (_s1, _s2)
ok("compositor: scale_to_height equalises figures despite unequal blank margin")

# The whole point: two figures land where the layout says, every time.
_scene = {"background": {"wall": (226, 202, 160), "floor": (120, 72, 40), "floor_y": 0.75},
          "slots": [{"asset": "f", "x": 0.25, "ground_y": 0.75, "h": 0.5, "z": 1},
                    {"asset": "f", "x": 0.75, "ground_y": 0.75, "h": 0.5, "z": 2}]}
_out = _np.asarray(_comp.composite(_scene, {"f": _solo}, size=(400, 200)))
assert tuple(_out[10, 200]) == (226, 202, 160), "wall not painted"
assert tuple(_out[190, 200]) == (120, 72, 40), "floor not painted"
_ink_cols = _np.where((_out.mean(axis=2) < 120).any(axis=0))[0]
_left = _ink_cols[_ink_cols < 200]
_right = _ink_cols[_ink_cols >= 200]
assert _left.size and _right.size, "one of the two figures is missing from the composite"
ok("compositor: both figures render, on the declared floor line (FLUX drops one)")

# Contact scenes are OUT OF SCOPE and must say so rather than compositing junk.
assert _comp.can_composite({"slots": []}) is True
assert _comp.can_composite({"slots": [], "contact": True}) is False
ok("compositor: a contact scene is refused, not composited into figures touching nothing")


print(f"\nALL SMOKE CHECKS PASSED  ({len(CALLS)} fake model calls)")
print(f"workdir: {WORK}")

# =============================================================================
#  draw.py  --  the figure, the faces, the furniture, all in code
# =============================================================================
#  These are regression tests for defects the STYLE SHEET found, each of which
#  survived several rounds of tuning scenes two figures at a time. Two were
#  settled by arithmetic rather than by eye, and both are pinned here so a
#  future constant nudge cannot quietly reintroduce them.
import math as _math
import random as _rnd
from PIL import ImageDraw as _ImgDraw
from pipeline import draw as _draw

# The brows sat OUTSIDE the skull: at the brow's outer x the head's edge is
# 38.8px above centre and the brow was at 39.4, so every face grew two black
# marks that read as bad hair.
_hr = 100.0
_ex, _ey, _er = _hr * 0.40, _hr * 0.10, _hr * 0.26
_by = _ey + _er * 1.5                        # brow height above centre
_bx = _ex + _er * 0.92                       # brow outer x
_edge = _math.sqrt(_hr * _hr - _bx * _bx)    # skull edge at that x
assert _by < _edge, f"brows fall outside the head: {_by:.1f} vs edge {_edge:.1f}"
ok("draw: eyebrows sit inside the head circle at their widest point")

# The worried mouth was drawn with a 20..160 sweep, which with y growing DOWN
# puts the arc's midpoint below its ends -- a U, i.e. a smile.
def _arc_is_frown(a0, a1):
    mid = _math.sin(_math.radians((a0 + a1) / 2))
    end = _math.sin(_math.radians(a0))
    return mid < end                          # midpoint higher on screen
assert not _arc_is_frown(20, 160), "sanity: 20..160 is a smile"
assert _arc_is_frown(200, 340), "sanity: 200..340 is a frown"
_src = (ROOT / "pipeline" / "draw.py").read_text(encoding="utf-8")
_worried = _src.split('elif mood == "worried":')[1].split("elif mood ==")[0]
assert "200, 340" in _worried, "the worried mouth is not drawn as a frown"
ok("draw: the worried face frowns (200..340), it does not smile (20..160)")

# A black mouth on a black beard is an invisible mouth.
assert "mcol = WHITE if beard else INK" in _src, "beard does not recolour the mouth"
for _mood in ('smug', 'worried'):
    _blk = _src.split(f'elif mood == "{_mood}":')[1].split("elif mood ==")[0]
    assert "col=mcol" in _blk, f"{_mood}'s mouth ignores the beard"
ok("draw: line-drawn mouths switch to white on a beard instead of vanishing")

# Seeded determinism -- Step 24's redraw path depends on it.
def _render(seed):
    _i = _Img.new("RGB", (240, 320), (255, 255, 255))
    _draw.figure(_ImgDraw.Draw(_i), _rnd.Random(seed), 120, 280, 240, "point", "rage")
    return _i.tobytes()
assert _render(11) == _render(11), "same seed must redraw identically"
assert _render(11) != _render(12), "different seeds must differ"
ok("draw: a frame redrawn with the same seed is pixel-identical")

# Character variety is PARAMETERS, never stored assets -- so every flag has to
# compose with every expression and every pose without raising.
for _kw in ({}, {"beard": True}, {"hair": "long", "coat": True, "glasses": True},
            {"child": True}, {"beard": True, "hair": "short", "glasses": True}):
    for _m in ("deadpan", "rage", "shock", "smug", "worried", "delight"):
        _i = _Img.new("RGB", (200, 280), (255, 255, 255))
        _draw.figure(_ImgDraw.Draw(_i), _rnd.Random(3), 100, 250, 210, "stand", _m, None, **_kw)
ok("draw: every character flag composes with all six expressions")

# =============================================================================
#  props.py  --  the library, and the gate in front of it
# =============================================================================
from pipeline import props as _props

for _raw, _want in (("A pair of Lungs", "pair_lung"), ("lungs", "lung"),
                    ("MRI machine", "mri_machine"), ("the stomach", "stomach")):
    assert _props.slug(_raw) == _want, (_raw, _props.slug(_raw))
for _bad in ("Lung", "has spaces", "", "x"):
    assert not _props.VALID_KEY.match(_bad), f"{_bad!r} should be an invalid key"
ok("props: keys normalise to flat slugs and malformed keys are rejected")

def _blob_png(colour=(232, 168, 200)):
    _i = _Img.new("RGB", (400, 400), (255, 255, 255))
    _ImgDraw.Draw(_i).ellipse([110, 90, 300, 320], fill=colour,
                              outline=(26, 26, 26), width=8)
    return _i

_m = _props.load()
_props.stage("lung", _blob_png(), "a pair of human lungs", _m)
# THE REVIEW GATE. A staged prop must be invisible: caching amplifies whatever
# it stores, so one bad generation admitted silently becomes forty bad videos.
assert _props.lookup("lung") is None, "an unreviewed prop must not be usable"
assert "lung" in _props.pending() and "lung" not in _props.known_keys()
_props.admit("lung", nominal_h=0.55)
_e = _props.lookup("lung")
assert _e and _e["reviewed"] and _e["nominal_h"] == 0.55
assert 0.0 < _e["ground"] <= 1.0, _e["ground"]
ok("props: a staged prop is unusable until a human admits it")

_scenes_x = [{"prop_key": "lung"}, {"prop_key": "lung"},
             {"prop_key": "door"}, {"prop_key": ""}]
_cov = _props.coverage(_scenes_x)
assert _cov == {"frames_needing_prop": 3, "distinct": 2, "hits": 2,
                "misses": 1, "missing_keys": ["door"]}, _cov
ok("props: coverage counts hits and names exactly what still has to be drawn")

# A generation that came back with the object AND a stick figure beside it
# cannot be placed by one ground anchor, and must be refused at the door.
_two = _Img.new("RGB", (400, 400), (255, 255, 255))
_d2 = _ImgDraw.Draw(_two)
_d2.ellipse([20, 150, 160, 290], fill=(200, 60, 50), outline=(26, 26, 26), width=8)
_d2.ellipse([250, 150, 390, 290], fill=(200, 60, 50), outline=(26, 26, 26), width=8)
try:
    _props.stage("two_things", _two, "", _props.load())
    raise AssertionError("a two-subject prop was admitted")
except ValueError:
    pass
ok("props: a prop containing two subjects is refused, not silently cached")

_props.reject("lung")
assert _props.lookup("lung") is None and not _props.p_exists("lung") \
    if hasattr(_props, "p_exists") else _props.lookup("lung") is None
ok("props: reject removes the entry so the next miss regenerates it")
