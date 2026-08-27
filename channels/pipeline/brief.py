"""
The Brief: one object carrying everything the writer, the fixer, the fact-checker,
the hook tournament and the packaging step all need.

=============================================================================
 WHY THIS EXISTS
=============================================================================
In the notebook, six prompt fragments lived as bare globals set in one cell and
read in four others:

    sources_block    set in Step 7,  read by Steps 7, 10, 11, 13
    numbers_block    set in Step 8,  read by Steps 8, 11, 13
    entities_block   set in Step 8,  read by Steps 8, 11, 13
    banned_block     set in Step 8,  read by Steps 8, 11, 13
    profile          set in Step 7,  read by Steps 8, 13, 15
    BEAT_SHEET       mutated in Step 7, read everywhere

That is the single biggest obstacle to splitting the notebook: any function you
lift out silently depends on a cell above it having run. Bundling them means a
function's inputs are its arguments, which is also what makes them testable.

The rendering itself is verbatim from the notebook -- these strings go into paid
prompts, and a "tidier" rewording is an unmeasured change to model behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Brief:
    topic: str
    source_pack: list[dict]
    profile: dict
    beat_sheet: dict
    banned: list[str] = field(default_factory=list)

    # ---------------------------------------------------------------- blocks
    @property
    def sources_block(self) -> str:
        return "\n\n".join(
            f"[Source {i+1}: {s['title']}]\n{s['excerpt']}"
            for i, s in enumerate(self.source_pack)
        )

    @property
    def numbers_block(self) -> str:
        return "\n".join(
            f'  - "{h["value"]}" = {h["means"]}  (Source {h["source"]})'
            for h in self.profile.get("hard_numbers", [])
        ) or "  (none - this topic has no hard numbers in the sources. Do not use ANY statistics.)"

    @property
    def entities_block(self) -> str:
        return "\n".join(
            f'  - {e["kind"]}: {e["value"]}  (Source {e["source"]})'
            for e in self.profile.get("named_entities", [])
        ) or "  (none - do not name any researcher, institution, university or year anywhere in the script.)"

    @property
    def findings_block(self) -> str:
        return "\n".join(
            f'  - [{k["certainty"]}] {k["finding"]}  (Source {k["source"]})'
            for k in self.profile.get("key_findings", [])
        )

    @property
    def gaps_block(self) -> str:
        return "\n".join(f"  - {g}" for g in self.profile.get("gaps", [])) or "  (none listed)"

    @property
    def banned_block(self) -> str:
        return ", ".join(f'"{p}"' for p in self.banned)

    @property
    def allowed_numbers(self) -> list[str]:
        """The writer's entire statistics allowance."""
        return [h["value"] for h in self.profile.get("hard_numbers", [])]

    @property
    def all_source_text(self) -> str:
        """Lowercased concatenation, for the linter's number check."""
        return " ".join(s["excerpt"] for s in self.source_pack).lower()

    # ----------------------------------------------------------- beat sheet
    def render_section_spec(self, n: int) -> str:
        """Renders one section of the beat sheet into prompt text. Used by the WRITER,
        by the SECTION REWRITER and by the RETENTION JUDGE -- same function, same text,
        so the judge can only ever hold the writer to rules the writer actually got."""
        spec = self.beat_sheet[n]
        lo, hi = spec["words"]
        lines = [f'=== SECTION {n}: {spec["name"].upper()} ({lo}-{hi} words) ===',
                 f'Rhythm: {spec["rhythm"]}',
                 "Beats, in this order:"]
        for i, b in enumerate(spec["beats"], 1):
            lines.append(f"  {i}. {b}")
        if spec.get("hard_rules"):
            lines.append("Hard rules for this section:")
            for r in spec["hard_rules"]:
                lines.append(f"  - {r}")
        return "\n".join(lines)

    def render_beat_sheet(self) -> str:
        return "\n\n".join(self.render_section_spec(n) for n in sorted(self.beat_sheet))
