"""
Steps 6-15, locally: fetch sources -> write the script -> both gates -> packaging.

    python run_script.py --preflight            # check key + model slugs, free
    python run_script.py --list
    python run_script.py --topic goosebumps
    python run_script.py --status
    python run_script.py --topic goosebumps --no-hook-tournament --no-cache

Every model call goes through OpenRouter on one OPENROUTER_API_KEY -- Claude and
anything else you swap in. Costs a few cents and no GPU time at all. When it finishes, the
working folder has everything the notebook needs to start at Step 16:

    source_pack.json  evidence_profile.json  script_tagged.json  vo_raw.txt
    packaging.json    youtube_description.txt  youtube_tags.txt  pinned_comment.txt
    sections/         one read sheet per section, plus the commands to record them

=============================================================================
 THE GATES, AND WHY THEY RUN IN THIS ORDER
=============================================================================
  grounding  ->  hook tournament  ->  retention  ->  humanizer  ->  re-grounding

Grounding first, because retention is a judgment about delivery and a model
asked to satisfy both at once will trade accuracy for punchiness.

Humanizer after retention, because the retention fixer is what CREATES most of
the machine-writing tells. "Make this land harder" is the exact instruction that
produces the em-dash reveal, the rule of three and the parallel-Maybe ending, so
a humanizer running before it would be undone. See pipeline/humanize.py.

Re-grounding last, and non-negotiably: the retention and humanizer fixers just
rewrote sentences for punch and for texture, and both are exactly the pressure
that produces a confident invented specific. Without this pass those gates
quietly reopen the hole the grounding gate closed -- and they do it on the lines
you care most about, because those are the ones they touched.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import config, engine, evidence, factcheck, humanize, lint, packaging  # noqa: E402
from pipeline import paths, readsheet, retention, script as script_mod, sources      # noqa: E402
from pipeline.brief import Brief                                                     # noqa: E402

MAX_GROUNDING_ROUNDS = 3
MAX_RETENTION_ROUNDS = 2   # deliberately lower than the grounding cap. Retention is a
                           # subjective target, and a model chasing "pass" for three rounds
                           # will over-punch one section until it no longer matches the
                           # tone of the ones around it.
MAX_HUMANIZE_ROUNDS = 2    # round 1 clears the quota breaches, round 2 catches what the
                           # rewrites themselves introduced. A third round starts trading
                           # sense for texture: the fixer runs out of legitimate ways to
                           # unbuild a sentence and begins mangling it.

RETENTION_INSTRUCTION = (
    "These voiceover lines are factually clean but are not holding the viewer. Rewrite each "
    "one so it lands harder while keeping exactly the same factual content. You are changing "
    "delivery, not substance: sharpen the image, cut the dead weight, vary the rhythm, replace "
    "a named emotion with the concrete detail that produces it. Do not add a single fact, "
    "number, name or year that is not already in the line you are rewriting."
)


def load_env() -> None:
    """Read f:\\wierd_bio\\.env the same way the notebook's Step 2b does, then fail
    early and clearly if the key is missing. Failing here costs nothing; failing
    after the source fetch wastes the fetch.

    One key for everything: Claude and any other model both route through
    OpenRouter. See pipeline/engine.py."""
    try:
        from dotenv import load_dotenv
        for candidate in (Path(__file__).resolve().parent.parent / ".env",
                          Path(__file__).resolve().parent / ".env"):
            if candidate.exists():
                load_dotenv(candidate, override=False)
    except ImportError:
        pass

    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit(
            "No OPENROUTER_API_KEY.\n"
            "  Get one at openrouter.ai/keys, then add it to f:\\wierd_bio\\.env:\n"
            "      OPENROUTER_API_KEY=sk-or-v1-...\n"
            "  or set it in this shell:\n"
            "      $env:OPENROUTER_API_KEY = 'sk-or-v1-...'"
        )


def dump(name: str, obj) -> None:
    with open(paths.p(name), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


# =============================================================================
#  GATE 1 - GROUNDING
# =============================================================================

def grounding_gate(tagged, b, lint_issues, fact_issues, smuggled_issues):
    round_num = 0
    while round_num < MAX_GROUNDING_ROUNDS:
        sentence_issues = fact_issues + smuggled_issues + [i for i in lint_issues if i.get("sentence")]
        section_issues: dict[int, list[str]] = {}
        for i in lint_issues:
            if not i.get("sentence"):
                section_issues.setdefault(i["section"], []).append(i["reason"])

        if not sentence_issues and not section_issues:
            break

        round_num += 1
        print(f"\n--- Grounding fix round {round_num}/{MAX_GROUNDING_ROUNDS} ---")

        if section_issues:
            for sec_n in sorted(section_issues, reverse=True):   # back to front, indices stay valid
                print(f"  rebuilding section {sec_n}: {section_issues[sec_n]}")
                tagged = script_mod.rewrite_section(tagged, sec_n, section_issues[sec_n], b)

        if sentence_issues:
            print(f"  repairing {len(sentence_issues)} flagged line(s)")
            tagged = script_mod.apply_fixes(
                tagged, sentence_issues, retention.GROUNDING_INSTRUCTION, b, temperature=0.35)

        print("\n  re-checking...")
        lint_issues = lint.run(tagged, b)
        fact_issues, smuggled_issues = factcheck.run(tagged, b, quiet=True)
        print(f"  after round {round_num}: {len(fact_issues)} fact, "
              f"{len(smuggled_issues)} smuggled, {len(lint_issues)} lint")

        dump(f"validation_round_{round_num+1}.json",
             {"fact": fact_issues, "smuggled": smuggled_issues, "lint": lint_issues})

    script_mod.save_script(tagged, "post-grounding")

    remaining = len(fact_issues) + len(smuggled_issues) + len(lint_issues)
    if remaining == 0:
        print("\nGROUNDING GATE PASSED - every claim traces to a source, every number is real.")
    else:
        print(f"\nGROUNDING GATE: {remaining} issue(s) remain after {MAX_GROUNDING_ROUNDS} rounds.")
        print("The script continues to the retention gate, but read validation_round_*.json")
        print("before you publish - these are the lines a viewer could fact-check you on.")

    return tagged, lint_issues, fact_issues, smuggled_issues


# =============================================================================
#  GATE 3 - HUMANIZER
# =============================================================================
#  Runs on a script that is already true and already paced. Everything it touches
#  is sentence SHAPE. It is folded into retention_gate() rather than called from
#  main() so that the re-grounding pass below covers its rewrites too -- a
#  humanizer with no fact check behind it is a machine for inventing details.
# =============================================================================

def humanize_gate(tagged, b, use_judge=True):
    issues = humanize.run_gate(tagged, b, use_judge=use_judge)
    dump("humanize_round_1.json", issues)

    for r in range(1, MAX_HUMANIZE_ROUNDS + 1):
        if not issues:
            break
        print(f"--- Humanize fix round {r}/{MAX_HUMANIZE_ROUNDS} ---")
        # Higher temperature than any other fixer on purpose. Every other pass
        # wants the model's most probable sentence; this one is trying to get off
        # it, and a low temperature returns the same machine shape reworded.
        tagged = script_mod.apply_fixes(
            tagged, issues, humanize.HUMANIZE_INSTRUCTION, b, temperature=0.8)

        print("\n  re-checking the voiceprint...")
        # Deterministic sweep only on the re-check. The judge has already named
        # the habits; paying it again to re-read fifty lines it just graded buys
        # very little, and the counters are what actually verify the fix landed.
        issues = humanize.run_audits(tagged)
        dump(f"humanize_round_{r+1}.json", issues)

    if issues:
        print(f"HUMANIZER GATE: {len(issues)} issue(s) remain after {MAX_HUMANIZE_ROUNDS} rounds.\n")
    else:
        print("HUMANIZER GATE PASSED - no quota breached, no tell left standing.\n")
    return tagged, issues


# =============================================================================
#  GATE 2 - RETENTION, THEN HUMANIZER, THEN MANDATORY RE-GROUNDING
# =============================================================================

def retention_gate(tagged, b, system, humanizer=True):
    retention_issues = retention.run_gate(tagged, b, system)
    dump("retention_round_1.json", retention_issues)

    for r in range(1, MAX_RETENTION_ROUNDS + 1):
        if not retention_issues:
            break
        print(f"\n--- Retention fix round {r}/{MAX_RETENTION_ROUNDS} ---")
        tagged = script_mod.apply_fixes(
            tagged, retention_issues, RETENTION_INSTRUCTION, b, temperature=0.55)

        print("\n  re-judging...")
        retention_issues = retention.run_gate(tagged, b, system)
        dump(f"retention_round_{r+1}.json", retention_issues)

    if humanizer:
        print("\n" + "=" * 70)
        tagged, humanize_issues = humanize_gate(tagged, b)
        script_mod.save_script(tagged, "post-humanizer")
    else:
        print("\nHumanizer gate skipped (--no-humanizer).")
        humanize_issues = []

    print("\n" + "=" * 70)
    print("RE-GROUNDING after retention and humanizer edits")
    print("=" * 70)

    lint_issues = lint.run(tagged, b)
    fact_issues, smuggled_issues = factcheck.run(tagged, b, quiet=True)
    print(f"  {len(fact_issues)} fact, {len(smuggled_issues)} smuggled, {len(lint_issues)} lint")

    regrounding = fact_issues + smuggled_issues + [i for i in lint_issues if i.get("sentence")]
    if regrounding:
        print(f"\n  the retention and humanizer passes introduced {len(regrounding)} "
              f"grounding issue(s) - repairing")
        tagged = script_mod.apply_fixes(
            tagged, regrounding,
            retention.GROUNDING_INSTRUCTION +
            " Preserve the improved rhythm and imagery where you can, and do not reach back "
            "for an em dash or a three-part list to make the corrected line sound finished. "
            "Grounding wins over punchiness every time.",
            b, temperature=0.3,
        )
        lint_issues = lint.run(tagged, b)
        fact_issues, smuggled_issues = factcheck.run(tagged, b, quiet=True)
        # The repair above just rewrote lines with a low-temperature, high-probability
        # instruction, which is precisely how a machine shape gets back in. Re-count,
        # so the summary tells the truth about what is actually shipping.
        if humanizer:
            humanize_issues = humanize.run_audits(tagged, quiet=True)

    return tagged, retention_issues, humanize_issues, lint_issues, fact_issues, smuggled_issues


# =============================================================================
#  MAIN
# =============================================================================

def main() -> int:
    paths.force_utf8_stdout()

    ap = argparse.ArgumentParser(description="Steps 6-15 of the stickman pipeline, locally.")
    ap.add_argument("--topic", help="key from the topic bank in pipeline/config.py")
    ap.add_argument("--workdir", help="the shared Drive folder (default: VIDEO_PIPELINE_WORKDIR)")
    ap.add_argument("--list", action="store_true", help="list the topic bank and exit")
    ap.add_argument("--status", action="store_true", help="show which stages have run, and exit")
    ap.add_argument("--models", default="claude", choices=sorted(config.MODEL_PRESETS),
                    help="which models to use. 'cheap' runs everything on one small "
                         "model to prove the wiring for about a cent - never judge a "
                         "script written under it")
    ap.add_argument("--preflight", action="store_true",
                    help="verify the key and every model slug against OpenRouter, then exit")
    ap.add_argument("--no-cache", action="store_true", help="refetch sources instead of using the disk cache")
    ap.add_argument("--no-hook-tournament", action="store_true")
    ap.add_argument("--no-humanizer", action="store_true",
                    help="skip the voiceprint gate. The script will be true and it will "
                         "be paced, and it will read like a language model wrote it")
    ap.add_argument("--hooks", type=int, default=retention.N_HOOKS)
    args = ap.parse_args()

    if args.list:
        # Stake-bearing topics print first and are marked. They are the ones that
        # can actually fill BEAT_SHEET[2]'s Human Stake beat -- the others publish
        # as pure curiosity pieces. See the comment above TOPICS in config.py.
        staked = [(k, v) for k, v in config.TOPICS.items() if v.get("stake")]
        plain = [(k, v) for k, v in config.TOPICS.items() if not v.get("stake")]

        if staked:
            print("\n  HUMAN STAKE - a real population the sources cover.\n")
            for k, v in staked:
                aud = v.get("audience", "?")
                print(f"  * {k:<18} {v['title']}  ({len(v['sources'])} sources, {aud})")
                print(f"    {'':<18} stake: {v['stake']}")

        if plain:
            print("\n  CURIOSITY ONLY - no stake source yet. Adding one is the cheapest")
            print("  upgrade available to any of these.\n")
            for k, v in plain:
                print(f"    {k:<18} {v['title']}  ({len(v['sources'])} sources)")

        # Skew balance, so a run of same-leaning topics is visible before it is
        # scheduled rather than after. See the audience note in config.py.
        from collections import Counter
        c = Counter(v.get("audience", "?") for v in config.TOPICS.values())
        print(f"\n  {len(staked)}/{len(config.TOPICS)} staked  |  skew: "
              f"{c.get('mixed-f', 0)} female-leaning, {c.get('mixed-m', 0)} male-leaning, "
              f"{c.get('mixed', 0)} neutral")
        print("  Target 25-44, both genders. Do not schedule a run of one skew.\n")
        return 0

    if args.workdir:
        paths.set_workdir(args.workdir)

    if args.status:
        print(paths.stage_report())
        return 0

    if args.preflight:
        load_env()
        config.apply_preset(args.models)
        engine.preflight()
        return 0

    if not args.topic:
        ap.error("--topic is required (see --list)")
    if args.topic not in config.TOPICS:
        ap.error(f"unknown topic {args.topic!r} - see --list")

    load_env()
    _preset = config.apply_preset(args.models)
    if args.models != "claude":
        print(f"Models: {args.models} preset -> "
              + ", ".join(f"{r}={m}" for r, m in _preset.items()))
        print("  This preset is for checking the pipeline runs, not the writing.\n")
    engine.preflight()
    print()

    topic_title = config.TOPICS[args.topic]["title"]
    learnings = config.load_learnings()

    print(f"Working folder: {paths.workdir()}")
    print(f'Producing: {topic_title}\n')

    # --- Step 6 ------------------------------------------------------------
    # Ship the prompt constants alongside the outputs, so the notebook's
    # Redraw button builds the same prompt this run baked into the frames.
    config.export_style()

    source_pack = sources.run(topic_title, config.TOPICS[args.topic]["sources"],
                              use_cache=not args.no_cache)

    # --- Step 7 ------------------------------------------------------------
    profile, beat_sheet, ev_type = evidence.run(topic_title, source_pack)

    b = Brief(topic=topic_title, source_pack=source_pack, profile=profile,
              beat_sheet=beat_sheet, banned=config.banned_phrases(learnings))
    print(f"\n  beat sheet: {config.estimate_runtime(beat_sheet)}")

    # --- Step 8 ------------------------------------------------------------
    tagged = script_mod.write(b)

    # --- Steps 9-11: grounding gate ----------------------------------------
    lint_issues = lint.run(tagged, b)
    dump("lint_round_1.json", lint_issues)
    fact_issues, smuggled_issues = factcheck.run(tagged, b)
    dump("validation_round_1.json",
         {"fact": fact_issues, "smuggled": smuggled_issues, "lint": lint_issues})

    tagged, lint_issues, fact_issues, smuggled_issues = grounding_gate(
        tagged, b, lint_issues, fact_issues, smuggled_issues)

    # --- Step 13: hook tournament ------------------------------------------
    system = retention.system_prompt(learnings)
    if args.no_hook_tournament:
        print("\nHook tournament skipped (--no-hook-tournament).")
    else:
        print()
        tagged = retention.hook_tournament(tagged, b, system, n_hooks=args.hooks)

    # --- Steps 12/14/14b: retention gate + humanizer + re-grounding ---------
    (tagged, retention_issues, humanize_issues,
     lint_issues, fact_issues, smuggled_issues) = retention_gate(
        tagged, b, system, humanizer=not args.no_humanizer)

    # --- the one thing that is never allowed to ship -----------------------
    # "unsettling" and "strangely beautiful" are both in BANNED_PHRASES and both
    # reached a finished chin script anyway: every gate above reports its
    # leftovers and continues, which is right for a judgment call and wrong for a
    # string match. A banned phrase needs no judgment, so it gets its own loop.
    for attempt in range(3):
        stragglers = lint.audit_banned_phrases(tagged, b)
        if not stragglers:
            break
        print(f"\n  backstop: {len(stragglers)} banned phrase(s) still present - forcing a rewrite")
        tagged = script_mod.apply_fixes(
            tagged, stragglers,
            "Each of these lines contains a phrase this channel never uses. Rewrite the line "
            "without it. Do not find a synonym for the banned phrase - the phrase is banned "
            "because the job it is doing is the problem. Same facts, same tag, plainer sentence.",
            b, temperature=0.4)
    else:
        print("  backstop: banned phrases survived three forced rewrites - fix these by hand.")

    script_mod.save_script(tagged, "final")

    # --- Step 15b: the read sheets -----------------------------------------
    # Built from the aligner's own chunking rather than from the section tags,
    # so the words you read and the words Step 17 expects to hear in that
    # section are the same words. See pipeline/readsheet.py.
    sheets = readsheet.write(tagged, b.beat_sheet)

    paths.mark_done("script", ["source_pack.json", "evidence_profile.json",
                               "script_tagged.json", "vo_raw.txt"])

    master_script = script_mod.strip_tags_to_voiceover(tagged)
    wc = len(master_script.split())
    print("\n" + "=" * 70)
    print("FINAL SCRIPT")
    print("=" * 70)
    print(f"  {wc} words, {len(tagged)} sentences, ~{wc/config.SPOKEN_WPM:.1f} min "
          f"at {config.SPOKEN_WPM} wpm")
    print(f"  grounding : {len(fact_issues)} fact, {len(smuggled_issues)} smuggled, "
          f"{len(lint_issues)} lint")
    print(f"  retention : {len(retention_issues)} open")
    print(f"  voiceprint: {len(humanize_issues)} open")
    script_mod.section_table(tagged, b.beat_sheet)

    total_open = (len(fact_issues) + len(smuggled_issues) + len(lint_issues)
                  + len(retention_issues) + len(humanize_issues))
    if total_open == 0:
        print("\n  ALL THREE GATES CLEAN - safe to voice.")
    else:
        print(f"\n  {total_open} issue(s) still open. vo_raw.txt is written either way, but read")
        print("  validation_round_*.json, retention_round_*.json and humanize_round_*.json")
        print("  before you record.")

    print("\n" + "-" * 70)
    print(master_script)
    print("-" * 70)

    # --- Step 15 -----------------------------------------------------------
    print()
    packaging.run(tagged, b, learnings)

    engine.print_usage()

    print("\n" + "=" * 70)
    print("NEXT: record your voiceover.")
    print("=" * 70)
    print(f"  read sheets: {paths.p(readsheet.SECTIONS_DIR)}")
    print(f"  {len(sheets)} section(s), and README.txt beside them carries the exact")
    print("  mark-script / live / collect_takes commands for each one.")
    print()
    print("  1. python -m vo_coach mark-script sections/section_01.txt --niche zenn")
    print("  2. python -m vo_coach live --script sections/section_01_marked.txt --niche zenn")
    print("  3. python collect_takes.py --section 1        (repeat 1-3 per section)")
    print("  4. python run_qc.py                           (before you open Colab)")
    print("  5. open the notebook and start at Step 16")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
