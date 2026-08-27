"""
Step 9 -- the deterministic linter. No API call, runs before any judge is paid.

Everything here is a rule that can be checked with code. Doing these mechanically
means the paid judges spend their attention on the things that genuinely need
judgment, and it means the cheap-to-catch failures (banned phrase, blown word
count, a number that appears nowhere in the sources) are caught every single run
rather than whenever a judge happens to notice.

Every function takes the Brief explicitly. In the notebook they closed over
ALL_SOURCE_TEXT, BANNED_PHRASES and BEAT_SHEET as globals, which is what made
them impossible to lift out of the notebook or test.
"""

from __future__ import annotations

import re

from .brief import Brief

NUM_WORDS = {
    "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000,
    "million": 1000000, "billion": 1000000000,
}


def _number_in_sources(token: str, all_source_text: str) -> bool:
    """A number counts as grounded if it appears in the source text as digits or as a
    word, in either direction. Deliberately generous - the goal is to catch inventions
    like '532 skulls' when no source ever said 532, not to police rounding."""
    t = token.lower().strip().rstrip("%").replace(",", "")
    if t in all_source_text or token.lower() in all_source_text:
        return True
    if t.isdigit():
        n = int(t)
        for w, v in NUM_WORDS.items():
            if v == n and w in all_source_text:
                return True
        # digits written with separators in the source, e.g. 550 million / 550,000
        if f"{n:,}" in all_source_text:
            return True
    if t in NUM_WORDS and str(NUM_WORDS[t]) in all_source_text:
        return True
    return False


def audit_numbers(tagged_sentences: list[dict], b: Brief) -> list[dict]:
    """Every number in the script must trace to the source text. This is the check
    that would have caught the chin script's invented measurement counts."""
    all_source_text = b.all_source_text
    findings = []
    for item in tagged_sentences:
        s = item["sentence"]
        tokens = re.findall(r"\b\d[\d,\.]*%?\b", s)
        tokens += [w for w in re.findall(r"\b[a-z]+\b", s.lower()) if w in NUM_WORDS]
        for tok in set(tokens):
            if not _number_in_sources(tok, all_source_text):
                findings.append({
                    "sentence": s, "section": item["section"],
                    "reason": f"the number '{tok}' does not appear in any fetched source",
                })
    return findings


def audit_banned_phrases(tagged_sentences: list[dict], b: Brief) -> list[dict]:
    findings = []
    for item in tagged_sentences:
        low = item["sentence"].lower()
        for phrase in b.banned:
            if phrase in low:
                findings.append({
                    "sentence": item["sentence"], "section": item["section"],
                    "reason": (f"contains banned phrase '{phrase}' - rewrite without "
                               f"labelling the feeling or using filler"),
                })
    return findings


def audit_stacked_adjectives(tagged_sentences: list[dict]) -> list[dict]:
    """Catches 'a quiet, ancient, forgotten reflex' - the Section 5 failure mode."""
    findings = []
    pat = re.compile(r"\b(\w+ly\s+)?(\w+),\s+(\w+),?\s+(and\s+)?(\w+)\s+(\w+)\b")
    for item in tagged_sentences:
        for m in pat.finditer(item["sentence"]):
            chunk = m.group(0)
            if len(chunk.split()) <= 6 and chunk.count(",") >= 2:
                findings.append({
                    "sentence": item["sentence"], "section": item["section"],
                    "reason": f"stacked modifiers ('{chunk}') - one adjective per noun",
                })
                break
    return findings


def audit_stat_runs(tagged_sentences: list[dict]) -> list[dict]:
    """Two or more source-carrying sentences back to back. This is the mechanical
    version of the pacing problem: 'X skulls... Y landmarks... Z distances'."""
    findings = []
    run: list[dict] = []
    for item in tagged_sentences + [{"sentence": "", "section": 0, "sources": []}]:
        if item["sources"]:
            run.append(item)
        else:
            if len(run) >= 2:
                findings.append({
                    "sentence": run[1]["sentence"], "section": run[1]["section"],
                    "reason": (f"{len(run)} fact-carrying sentences in a row - break the run with "
                               f"an image or a consequence the viewer can picture"),
                })
            run = []
    return findings


def audit_sentence_rhythm(tagged_sentences: list[dict]) -> list[dict]:
    """Three consecutive sentences within 3 words of each other reads as flat narration."""
    findings = []
    lens = [len(i["sentence"].split()) for i in tagged_sentences]
    for i in range(len(lens) - 2):
        w = lens[i:i + 3]
        if max(w) - min(w) <= 3 and min(w) >= 12:
            findings.append({
                "sentence": tagged_sentences[i + 1]["sentence"],
                "section": tagged_sentences[i + 1]["section"],
                "reason": f"three consecutive sentences of similar length ({w}) - vary the rhythm",
            })
    return findings


def audit_structure(tagged_sentences: list[dict], b: Brief) -> list[dict]:
    """Beat-level structural checks that don't need a model."""
    findings = []
    by_sec: dict[int, list] = {}
    for it in tagged_sentences:
        by_sec.setdefault(it["section"], []).append(it)

    # --- word counts per section ---
    for n, spec in b.beat_sheet.items():
        sents = by_sec.get(n, [])
        words = sum(len(i["sentence"].split()) for i in sents)
        lo, hi = spec["words"]
        if not sents:
            findings.append({"sentence": "", "section": n,
                             "reason": f"Section {n} ({spec['name']}) is missing entirely"})
        elif words < lo * 0.85 or words > hi * 1.15:
            findings.append({"sentence": "", "section": n,
                             "reason": f"Section {n} is {words} words, target {lo}-{hi}"})

    # --- S1 must contain a 2-4 word Snap ---
    s1 = by_sec.get(1, [])
    if s1 and not any(2 <= len(i["sentence"].rstrip(".!?").split()) <= 4 for i in s1):
        findings.append({"sentence": "", "section": 1,
                         "reason": ("Section 1 has no 2-4 word Snap line - the Absolute "
                                    "Negation beat is missing or too long")})

    # --- S1 sentence-length ceiling ---
    for i in s1:
        if len(i["sentence"].split()) > 14:
            findings.append({"sentence": i["sentence"], "section": 1,
                             "reason": (f"Section 1 sentence is {len(i['sentence'].split())} "
                                        f"words (max 14) - the hook must stay staccato")})

    # --- S6 must end on a real 3-8 word question, and carry no claims ---
    s6 = by_sec.get(6, [])
    if s6:
        last = s6[-1]["sentence"].strip()
        n_words = len(last.rstrip("?").split())
        if not last.endswith("?"):
            findings.append({"sentence": last, "section": 6,
                             "reason": "the script does not end on a question"})
        elif not (3 <= n_words <= 8):
            findings.append({"sentence": last, "section": 6,
                             "reason": f"closing question is {n_words} words (target 3-8)"})
        if re.match(r"^(isn't|aren't|is it not|who knew|how amazing|don't you)", last.lower()):
            findings.append({"sentence": last, "section": 6,
                             "reason": ("closing line is a rhetorical statement wearing a "
                                        "question mark, not an open question")})
        for i in s6:
            if i["sources"]:
                findings.append({"sentence": i["sentence"], "section": 6,
                                 "reason": "Section 6 must carry no new factual claims"})

    # --- somatic loop: outro should reuse hook vocabulary ---
    if s1 and s6:
        stop = set("the a an and or but of to in on it is are was were you your this "
                   "that with for as at be".split())
        hook_words = set(w for w in re.findall(r"[a-z]+", " ".join(i["sentence"].lower() for i in s1))
                         if w not in stop and len(w) > 3)
        outro_words = set(w for w in re.findall(r"[a-z]+", " ".join(i["sentence"].lower() for i in s6))
                          if w not in stop and len(w) > 3)
        if len(hook_words & outro_words) < 2:
            findings.append({"sentence": "", "section": 6,
                             "reason": ("the outro does not audibly loop back to the hook - reuse "
                                        "the specific body part and action words from Section 1")})

    return findings


def run(tagged_sentences: list[dict], b: Brief) -> list[dict]:
    checks = {
        "ungrounded numbers": audit_numbers(tagged_sentences, b),
        "banned phrases": audit_banned_phrases(tagged_sentences, b),
        "stacked adjectives": audit_stacked_adjectives(tagged_sentences),
        "fact-sentence runs": audit_stat_runs(tagged_sentences),
        "flat rhythm": audit_sentence_rhythm(tagged_sentences),
        "structure": audit_structure(tagged_sentences, b),
    }
    all_issues = []
    print("LINTER RESULTS")
    print("-" * 70)
    for name, issues in checks.items():
        status = "PASS" if not issues else f"{len(issues)} issue(s)"
        print(f"  {name:<22} {status}")
        for iss in issues:
            snippet = (iss["sentence"][:70] + "...") if len(iss["sentence"]) > 70 else iss["sentence"]
            print(f"      S{iss['section']}: {iss['reason']}")
            if snippet:
                print(f"        > {snippet}")
        all_issues.extend(issues)
    print("-" * 70)
    print(f"  {len(all_issues)} total issue(s)\n")
    return all_issues
