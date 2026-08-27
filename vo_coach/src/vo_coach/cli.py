"""CLI entry point.

Dev-tooling item 8: build CLI-first, UI-last. Every phase gets a subcommand
here before it gets a web view, so "is the pipeline correct" stays a separate
debugging problem from "is the UI correct".

    python -m vo_coach devices
    python -m vo_coach record --seconds 30
    python -m vo_coach ingest path/to/take.wav
    python -m vo_coach takes
"""

from __future__ import annotations

import argparse
import sys


def _force_utf8_stdout() -> None:
    """Windows consoles default to a legacy codepage (cp1252 here), which cannot
    encode Bengali - or Devanagari, CJK, or anything else outside Latin-1.

    Printing a transcript from a non-English niche therefore raises
    UnicodeEncodeError *after* the expensive ASR pass has already completed,
    losing the whole result to a display problem. Reconfiguring the stream costs
    nothing and makes every niche printable.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass          # a non-reconfigurable stream (a pipe, a test capture)


# Published per-1M-token rates, for showing what a call actually cost rather
# than leaving spend invisible. Approximate and will drift - it is a display
# aid, not billing. Sonnet 5 is on intro pricing through 2026-08-31.
_PRICING = {
    # OpenRouter prices are PER HOST, not per model, so these only hold while
    # the pin in config/thresholds.yaml holds. gpt-oss-120b alone spans
    # $0.03-$0.35 input across its 18 hosts - reading a cost off the model name
    # would be a guess dressed as a number.
    "openai/gpt-oss-120b":         (0.35, 0.75),     # cerebras/fp16
    "deepseek/deepseek-v4-flash":  (0.0686, 0.137),  # baidu host
    "inclusionai/ling-3.0-flash":  (0.042, 0.126),   # novita/fast
    # Anthropic
    "claude-opus-5":    (5.00, 25.00),
    "claude-sonnet-5":  (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
}


def _estimate_cost(model: str, in_tokens: int, out_tokens: int) -> float | None:
    for prefix, (pin, pout) in _PRICING.items():
        if model.startswith(prefix):
            return in_tokens / 1e6 * pin + out_tokens / 1e6 * pout
    return None


def main(argv: list[str] | None = None) -> int:
    _force_utf8_stdout()
    p = argparse.ArgumentParser(prog="vo_coach", description="VO coach & mastering agent")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("devices", help="list audio input devices")

    p_rec = sub.add_parser("record", help="record a take")
    p_rec.add_argument("--seconds", type=float, default=None,
                       help="stop after N seconds (default: until Ctrl+C)")
    p_rec.add_argument("--device", default=None,
                       help="input device index or name substring")
    p_rec.add_argument("--script", default=None, help="script name for this take")

    p_ing = sub.add_parser("ingest", help="register an existing wav as a take")
    p_ing.add_argument("path")
    p_ing.add_argument("--script", default=None)

    p_takes = sub.add_parser("takes", help="list recorded takes")
    p_takes.add_argument("--limit", type=int, default=20)

    p_an = sub.add_parser("analyze", help="run Phase 1 analysis on a take")
    p_an.add_argument("target", help="take_id, or a path to a wav")
    p_an.add_argument("--script", default=None,
                      help="path to the script text; enables punctuation-aware "
                           "pause labelling")
    p_an.add_argument("--model", default="base.en", help="faster-whisper model size")
    p_an.add_argument("--no-asr", action="store_true",
                      help="signal metrics only - no transcript, pacing or pauses. "
                           "This is the correct mode for the synthetic fixtures.")
    p_an.add_argument("--json", dest="as_json", action="store_true",
                      help="print the full AnalysisResult instead of a summary")
    p_an.add_argument("--plot", action="store_true", help="also write a debug PNG")

    p_pl = sub.add_parser("plot", help="write the debug plot for an analyzed take")
    p_pl.add_argument("take_id")
    p_pl.add_argument("-o", "--out", default=None)

    p_refs = sub.add_parser("refs", help="build / inspect niche reference voices")
    p_refs.add_argument("action", choices=["build", "list"])
    p_refs.add_argument("--niche", default=None, help="default: all niches")
    p_refs.add_argument("--force", action="store_true",
                        help="re-extract even if cached")

    p_sc = sub.add_parser("score", help="run the rubric over an analyzed take")
    p_sc.add_argument("take_id")
    p_sc.add_argument("--niche", default=None,
                      help="use this niche's derived pacing band")

    p_bm = sub.add_parser("benchmark", help="diff a take against its niche reference")
    p_bm.add_argument("take_id")
    p_bm.add_argument("--niche", required=True)

    p_co = sub.add_parser("coach", help="conversational coaching on an analyzed take")
    p_co.add_argument("take_id")
    p_co.add_argument("--niche", default=None,
                      help="also benchmark against this niche's reference")
    p_co.add_argument("--force", action="store_true",
                      help="ignore the response cache and call the API again")
    p_co.add_argument("--no-history", action="store_true",
                      help="do not send previous takes as context")

    p_mc = sub.add_parser("mentor-cache", help="inspect / clear the mentor cache")
    p_mc.add_argument("action", choices=["stats", "clear"])

    sub.add_parser("mentor-audit",
                   help="per-model rule-violation rate: the switch-models signal")

    p_cal = sub.add_parser(
        "calibrate", help="build your personal baseline from your own takes")
    p_cal.add_argument("take_ids", nargs="+",
                       help="3+ analyzed takes of natural, well-recorded reading")
    p_cal.add_argument("--show", action="store_true",
                       help="print the baseline without writing it")

    sub.add_parser("baseline", help="show the current personal baseline")

    p_ev = sub.add_parser(
        "eval", help="run the golden set against the configured model")
    p_ev.add_argument("--case", default=None, help="run one case by id")
    p_ev.add_argument("--force", action="store_true",
                      help="ignore the response cache and call the API again")
    p_ev.add_argument("--verbose", action="store_true",
                      help="print each reply in full")
    p_ev.add_argument("--compare", action="store_true",
                      help="show stored runs for every model instead of running")
    p_ev.add_argument("--list", dest="list_cases", action="store_true",
                      help="list the cases and why each exists")

    p_ma = sub.add_parser("master", help="run the mastering chain on a take")
    p_ma.add_argument("target", help="take_id, or a path to a wav")
    p_ma.add_argument("-o", "--out", default=None)
    p_ma.add_argument("--no-denoise", action="store_true")
    p_ma.add_argument("--no-deesser", action="store_true")

    p_rk = sub.add_parser("rank", help="rank several takes of the same line")
    p_rk.add_argument("take_ids", nargs="+")
    p_rk.add_argument("--niche", default=None)

    p_mk = sub.add_parser(
        "mark-script", help="mark a script with a niche's pause rhythm")
    p_mk.add_argument("path", help="plain text script")
    p_mk.add_argument("--niche", default=None,
                      help="derive the rhythm from this niche's reference")
    p_mk.add_argument("-o", "--out", default=None,
                      help="default: <name>_marked.txt next to the input")

    p_fx = sub.add_parser(
        "fix-pauses", help="set performed pauses to their target length")
    p_fx.add_argument("target", help="take_id, or a path to a wav")
    p_fx.add_argument("--script", default=None,
                      help="marked script; its [0.00] targets are matched to "
                           "the pauses in order")
    p_fx.add_argument("--niche", default=None,
                      help="rhythm to normalise toward when no script is given")
    p_fx.add_argument("-o", "--out", default=None)
    p_fx.add_argument("--in-place", action="store_true",
                      help="overwrite the input - use for section_NN.wav so the "
                           "notebook picks the corrected file up")
    p_fx.add_argument("--report", action="store_true",
                      help="print what changed (off by default, by design)")

    p_live = sub.add_parser("live", help="Phase 5: live HUD + real-time coaching")
    p_live.add_argument("--port", type=int, default=8765)
    p_live.add_argument("--host", default="127.0.0.1",
                        help="local only by default: this process opens a mic "
                             "and writes files, and there is no auth")
    p_live.add_argument("--device", default=None,
                        help="input device index or name substring")
    p_live.add_argument("--niche", default=None)
    p_live.add_argument("--script", default=None,
                        help="marked script (from mark-script) to show as a "
                             "teleprompter with live pause timing")
    p_live.add_argument("--no-save", action="store_true",
                        help="do not record the session to a take")
    p_live.add_argument("--no-asr", action="store_true",
                        help="signal metrics only - no transcript or wpm")
    p_live.add_argument("--no-mentor", action="store_true",
                        help="use the rubric's own wording instead of calling "
                             "the API. Free, and the right mode for checking "
                             "clause detection.")

    sub.add_parser("config", help="print the resolved config")

    args = p.parse_args(argv)

    if args.cmd == "devices":
        from .record import list_devices, default_input
        print(list_devices())
        print(f"\ndefault input index: {default_input()}")
        return 0

    if args.cmd == "record":
        from .record import record
        device = args.device
        if device is not None and device.isdigit():
            device = int(device)
        record(seconds=args.seconds, device=device, script_name=args.script)
        return 0

    if args.cmd == "ingest":
        from .record import ingest
        ingest(args.path, script_name=args.script)
        return 0

    if args.cmd == "takes":
        from .storage import db
        db.init_db()
        rows = db.list_takes(limit=args.limit)
        if not rows:
            print("No takes yet. Run:  python -m vo_coach record --seconds 30")
            return 0
        print(f"{'take_id':<24} {'dur':>7}  {'analyzed':<8} script")
        for r in rows:
            print(f"{r['take_id']:<24} {r['duration_sec'] or 0:>6.1f}s  "
                  f"{'yes' if r['analyzed'] else 'no':<8} {r['script_name'] or '-'}")
        return 0

    if args.cmd == "analyze":
        from pathlib import Path

        from .analysis.pipeline import analyze_and_store, summarize
        from .storage import db

        target = args.target
        if Path(target).exists():
            wav, take_id = target, Path(target).stem
        else:
            db.init_db()
            row = next((r for r in db.list_takes(limit=500)
                        if r["take_id"] == target), None)
            if row is None:
                print(f"No take {target!r}, and no file at that path.")
                return 1
            wav, take_id = row["wav_path"], row["take_id"]

        script = None
        if args.script:
            script = Path(args.script).read_text(encoding="utf-8")

        result = analyze_and_store(wav, take_id, script=script,
                                   model_size=args.model,
                                   with_asr=not args.no_asr)
        print(result.model_dump_json(indent=2) if args.as_json
              else summarize(result))

        if args.plot:
            from .viz.debug_plots import plot_take
            print(f"\n  plot -> {plot_take(result)}")
        return 0

    if args.cmd == "plot":
        from .schemas import AnalysisResult
        from .storage import db
        from .viz.debug_plots import plot_take

        db.init_db()
        raw = db.load_analysis(args.take_id)
        if raw is None:
            print(f"No analysis for {args.take_id!r}. Run: "
                  f"python -m vo_coach analyze {args.take_id}")
            return 1
        print(f"plot -> {plot_take(AnalysisResult.model_validate(raw), args.out)}")
        return 0

    if args.cmd == "refs":
        from .scoring import references as refs

        names = [args.niche] if args.niche else refs.niche_names()
        if args.action == "build":
            for n in names:
                print(f"building {n} ...")
                r = refs.build(n, force=args.force)
                f = r.features
                print(f"  {r.name}  {r.duration_sec:.0f}s  lang={r.language}  "
                      f"({r.provenance})")
                print(f"    wpm {f.wpm_overall:.0f}   pitch {f.pitch_mean_hz:.0f}Hz "
                      f"range {f.pitch_range_hz:.0f}Hz   tilt "
                      f"{f.spectral_tilt_mean:.2f}dB/oct")
                print(f"    sibilants/min {f.sibilance_per_min:.0f}   "
                      f"pauses/min {f.pause_rate_per_min:.0f}   "
                      f"mean pause {f.mean_pause_sec:.2f}s")
                band = refs.wpm_band_for(n)
                if band:
                    print(f"    derived pacing band: {band[0]:.0f}-{band[1]:.0f} wpm")
            return 0

        for n in names:
            r = refs.load(n)
            print(f"{n:20} {'cached' if r else 'NOT BUILT':<10} "
                  f"{r.name if r else ''}")
        return 0

    if args.cmd in ("score", "benchmark"):
        from .schemas import AnalysisResult
        from .storage import db

        db.init_db()
        raw = db.load_analysis(args.take_id)
        if raw is None:
            print(f"No analysis for {args.take_id!r}. Run: "
                  f"python -m vo_coach analyze {args.take_id}")
            return 1
        if db.is_stale(raw):
            print(f"  warning: {args.take_id} was analyzed by an older analyzer; "
                  f"its numbers may be wrong.\n           re-run: "
                  f"python -m vo_coach analyze {args.take_id}\n")
        result = AnalysisResult.model_validate(raw)

        if args.cmd == "score":
            from .scoring import references as refs
            from .scoring.rubric import score

            from .scoring import baseline as bl

            band = refs.wpm_band_for(args.niche) if args.niche else None
            personal = bl.load()
            if personal is not None:
                widened = bl.wpm_band(band=band, b=personal)
                if widened != band:
                    print(f"  pacing band adjusted to your baseline: "
                          f"{widened[0]:.0f}-{widened[1]:.0f} wpm "
                          f"(from {len(personal.take_ids)} calibration take(s))")
                band = widened
            reliable = True
            if args.niche:
                reliable = refs.niche_config(args.niche).get(
                    "asr_word_timestamps_reliable", True)
                if not reliable:
                    print(f"  note: pacing and pause findings are suppressed for "
                          f"{args.niche} - word timestamps are unreliable for "
                          f"this language/model. Signal metrics still apply.\n")
            s = score(result, wpm_band=band, timings_reliable=reliable)
            if not s.findings:
                print("  Nothing to flag - this take is clean.")
                return 0
            mark = {"must_fix": "MUST FIX", "stylistic": "stylistic", "ok": "ok"}
            for f in s.findings:
                where = f" @{f.t:.1f}s" if f.t is not None else ""
                print(f"  [{mark[f.severity]:>9}] {f.metric}{where}")
                print(f"              {f.message}")
            return 0

        from .scoring import references as refs
        from .scoring.benchmark import benchmark, render

        ref = refs.load(args.niche)
        if ref is None:
            print(f"Reference for {args.niche!r} not built. Run: "
                  f"python -m vo_coach refs build --niche {args.niche}")
            return 1
        print(render(benchmark(result, ref)))
        return 0

    if args.cmd == "mentor-cache":
        from .mentor import cache as mcache

        if args.action == "clear":
            print(f"cleared {mcache.clear()} cached response(s)")
            return 0
        n, size = mcache.stats()
        print(f"{n} cached response(s), {size/1024:.1f} KB")
        print(f"  {mcache.MENTOR_CACHE}")
        return 0

    if args.cmd == "mark-script":
        from pathlib import Path

        from .timing import mark_script

        src = Path(args.path)
        if not src.exists():
            print(f"No script at {src}")
            return 1
        marked, style = mark_script(src.read_text(encoding="utf-8"), args.niche)
        out = Path(args.out) if args.out else src.with_name(src.stem + "_marked.txt")
        out.write_text(marked, encoding="utf-8")
        print(marked)
        print("\nwritten to " + str(out))
        return 0

    if args.cmd == "fix-pauses":
        from pathlib import Path

        from .timing import fix_pauses
        from .storage import db

        target = args.target
        if Path(target).exists():
            wav = target
        else:
            db.init_db()
            row = next((r for r in db.list_takes(limit=500)
                        if r["take_id"] == target), None)
            if row is None:
                print(f"No take {target!r}, and no file at that path.")
                return 1
            wav = row["wav_path"]

        script = None
        if args.script:
            sp = Path(args.script)
            if not sp.exists():
                print(f"No script at {sp}")
                return 1
            script = sp.read_text(encoding="utf-8")

        res = fix_pauses(wav, script_marked=script, niche=args.niche,
                         out=args.out, in_place=args.in_place)
        print(res.path)
        if args.report or res.note:
            print(f"  {res.gaps_found} pause(s) found, {res.adjusted} adjusted, "
                  f"{res.skipped_too_short} left alone (under "
                  f"{__import__('vo_coach.timing.gaps', fromlist=['x']).MIN_REAL_GAP}s)")
            print(f"  {res.duration_before:.2f}s -> {res.duration_after:.2f}s "
                  f"({res.delta:+.2f}s)")
            if res.note:
                print(f"  note: {res.note}")
        if args.report:
            for d in res.details:
                print(f"    {d['at']:>6.2f}s  {d['was']:.2f} -> {d['now']:.2f}")
        return 0

    if args.cmd == "live":
        from .live.server import serve

        device = args.device
        if device is not None and device.isdigit():
            device = int(device)
        script = None
        if args.script:
            from pathlib import Path

            sp = Path(args.script)
            if not sp.exists():
                print(f"No script at {sp}")
                return 1
            script = sp.read_text(encoding="utf-8")

        serve(host=args.host, port=args.port, device=device, niche=args.niche,
              save=not args.no_save, use_asr=not args.no_asr,
              use_mentor=not args.no_mentor, script=script)
        return 0

    if args.cmd == "eval":
        from .config import load_config
        from .evaluation import CASES, render, run_all
        from .evaluation.cases import by_id
        from .evaluation.runner import render_comparison, save, stored

        if args.list_cases:
            for c in CASES:
                print(c.id)
                print("    " + c.why + "\n")
            return 0

        if args.compare:
            print(render_comparison(stored()))
            return 0

        cases = [by_id(args.case)] if args.case else None
        cfg = load_config()
        print(f"running {len(cases or CASES)} case(s) against "
              f"{cfg.get('mentor.model')}\n")
        run = run_all(cases, cfg=cfg, force=args.force)
        print(render(run, verbose=args.verbose))
        if args.case is None:
            print("\nstored: " + str(save(run)))
            print("Switch mentor.model in config/thresholds.yaml and run again "
                  "to compare;\n`--compare` prints every stored model side by "
                  "side.")
        return 0 if run.rate == 1.0 else 1

    if args.cmd == "mentor-audit":
        from .mentor.validate import AUDIT_LOG, rates

        r = rates()
        if not r:
            print(f"No mentor calls logged yet ({AUDIT_LOG}).")
            print("The log fills as you run `coach`; cached replies are not "
                  "counted, so the rate reflects real calls only.")
            return 0
        for model, m in sorted(r.items()):
            print(f"{model}")
            print(f"  {m['calls']} call(s), {m['with_hard']} with a hard "
                  f"violation  ->  hard-violation rate {m['hard_rate']:.0%}")
            print(f"  {m['hard']} hard, {m['soft']} soft violation(s) total")
            for rule, n in sorted(m["by_rule"].items(), key=lambda kv: -kv[1]):
                print(f"    {rule:<26} {n}")
            print()
        print("Reading this: below ~10% is normal model noise. A rule failing "
              "repeatedly\nis a capability problem - switch to the DeepSeek "
              "block in config/thresholds.yaml\nand compare the rate on the "
              "same takes.")
        return 0

    if args.cmd in ("calibrate", "baseline"):
        from .schemas import AnalysisResult
        from .scoring import baseline as bl
        from .storage import db

        if args.cmd == "baseline":
            print(bl.render(bl.load(require_enabled=False)))
            from .config import load_config as _lc
            if not _lc().get("baseline.enabled", False):
                print("\n  baseline.enabled is false in config/thresholds.yaml - "
                      "the numbers above are stored but not being applied.")
            return 0

        db.init_db()
        analyses = []
        for tid in args.take_ids:
            raw = db.load_analysis(tid)
            if raw is None:
                print(f"No analysis for {tid!r}. Run: "
                      f"python -m vo_coach analyze {tid}")
                return 1
            analyses.append(AnalysisResult.model_validate(raw))

        if len(analyses) < 3:
            print(f"  warning: calibrating on {len(analyses)} take(s). Three or "
                  f"more makes the medians stable; fewer describes one session.\n")

        try:
            b = bl.build(analyses)
        except bl.CalibrationError as e:
            print(f"cannot calibrate: {e}")
            return 1

        print(bl.render(b))
        if args.show:
            print("\n  (--show: nothing written)")
            return 0
        p = bl.save(b)
        print(f"\nwritten to {p}")
        print("Set  baseline.enabled: true  in config/thresholds.yaml to apply it.")
        return 0

    if args.cmd == "coach":
        from .config import load_config
        from .mentor.client import MentorError, OfflineCacheMiss, coach
        from .schemas import AnalysisResult
        from .scoring import references as refs
        from .scoring.benchmark import benchmark as run_benchmark
        from .scoring.rubric import score as run_score
        from .storage import db

        db.init_db()
        raw = db.load_analysis(args.take_id)
        if raw is None:
            print(f"No analysis for {args.take_id!r}. Run: "
                  f"python -m vo_coach analyze {args.take_id}")
            return 1
        if db.is_stale(raw):
            print(f"  warning: {args.take_id} was analyzed by an older analyzer; "
                  f"coaching may be based on wrong numbers.\n           re-run: "
                  f"python -m vo_coach analyze {args.take_id}\n")
        result = AnalysisResult.model_validate(raw)

        from .scoring import baseline as bl

        reliable = True
        band = None
        bench = None
        if args.niche:
            nc = refs.niche_config(args.niche)
            reliable = nc.get("asr_word_timestamps_reliable", True)
            band = refs.wpm_band_for(args.niche)
            ref = refs.load(args.niche)
            if ref is None:
                print(f"Reference for {args.niche!r} not built. Run: "
                      f"python -m vo_coach refs build --niche {args.niche}")
                return 1
            bench = run_benchmark(result, ref)

        band = bl.wpm_band(band=band)
        scored = run_score(result, wpm_band=band, timings_reliable=reliable)

        history = None
        if not args.no_history:
            history = [
                {"take_id": r["take_id"], "created_at": r["created_at"],
                 "duration_sec": round(r["duration_sec"] or 0, 1)}
                for r in db.list_takes(limit=6)
                if r["take_id"] != args.take_id and r["analyzed"]
            ]

        try:
            text, meta = coach(result, scored, benchmark=bench, history=history,
                               timings_reliable=reliable, force=args.force)
        except OfflineCacheMiss as e:
            print(f"offline: {e}")
            return 1
        except MentorError as e:
            print(f"mentor failed: {e}")
            return 1

        print()
        print(text)
        print()

        # Rule compliance. Printed after the coaching, not instead of it: a
        # violation means "do not take this part on trust", not "discard".
        report = meta.get("validation")
        if report is not None and report.violations:
            print("  rule check:")
            for v in report.violations:
                mark = "FAIL" if v.severity == "hard" else "note"
                print(f"    [{mark}] {v.rule}: {v.detail}")
                if v.excerpt:
                    print(f"           ...{v.excerpt}...")
            if report.hard:
                print("    (run `python -m vo_coach mentor-audit` for the rate "
                      "across all calls - one bad reply is noise, a pattern is "
                      "a reason to switch models)")
            print()

        if meta.get("cached"):
            print("  (cached response - re-run with --force to call the API)")
        else:
            cost = _estimate_cost(load_config().get("mentor.model"),
                                  meta.get("input_tokens", 0),
                                  meta.get("output_tokens", 0))
            host = meta.get("provider")
            # Reasoning tokens are billed and waited on but never shown, so
            # print them: a call that felt slow for no visible reason usually
            # spent its time here.
            think = meta.get("reasoning_tokens") or 0
            print(f"  {meta.get('model')}"
                  f"{f' via {host}' if host else ''}  "
                  f"in {meta.get('input_tokens')} / out {meta.get('output_tokens')} tokens"
                  f"{f' ({think} thinking)' if think else ''}"
                  f"{f'  ~${cost:.5f}' if cost is not None else ''}")
        return 0

    if args.cmd == "master":
        from pathlib import Path

        from .mastering.pipeline import master_take, render
        from .storage import db

        target = args.target
        if Path(target).exists():
            wav, take_id = target, Path(target).stem
        else:
            db.init_db()
            row = next((r for r in db.list_takes(limit=500)
                        if r["take_id"] == target), None)
            if row is None:
                print(f"No take {target!r}, and no file at that path.")
                return 1
            wav, take_id = row["wav_path"], row["take_id"]

        print(render(master_take(wav, take_id, out_path=args.out,
                                 denoise=not args.no_denoise,
                                 deesser=not args.no_deesser)))
        return 0

    if args.cmd == "rank":
        from .mastering.ranking import rank, render
        from .schemas import AnalysisResult
        from .scoring import references as refs
        from .storage import db

        db.init_db()
        analyses, missing, stale = [], [], []
        for tid in args.take_ids:
            raw = db.load_analysis(tid)
            if raw is None:
                missing.append(tid)
            else:
                if db.is_stale(raw):
                    stale.append(tid)
                analyses.append(AnalysisResult.model_validate(raw))
        if stale:
            print(f"  warning: analyzed by an older analyzer, numbers may be "
                  f"wrong: {', '.join(stale)}")
            print(f"           re-run analyze on those before trusting this "
                  f"ranking.\n")
        if missing:
            print(f"Not analyzed yet: {', '.join(missing)}")
            print("Run: python -m vo_coach analyze <take_id>")
            return 1

        band, reliable = None, True
        if args.niche:
            band = refs.wpm_band_for(args.niche)
            reliable = refs.niche_config(args.niche).get(
                "asr_word_timestamps_reliable", True)

        print(render(rank(analyses, wpm_band=band, timings_reliable=reliable)))
        return 0

    if args.cmd == "config":
        import json
        from .config import load_config, CONFIG_PATH
        print(f"# {CONFIG_PATH}")
        print(json.dumps(load_config().raw, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
