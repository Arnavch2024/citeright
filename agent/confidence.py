"""Code-side confidence. The LLM may explain, never raise, the floor.

final = min(code_level, llm_level). A fluent write-up cannot promote thin evidence.
"""

from __future__ import annotations

from agent.schemas import ConfidenceLevel, ConfidenceSignals, ScoredDocument


_RANK = {
    ConfidenceLevel.LOW: 0,
    ConfidenceLevel.MEDIUM: 1,
    ConfidenceLevel.HIGH: 2,
}


def min_level(a: ConfidenceLevel, b: ConfidenceLevel) -> ConfidenceLevel:
    return a if _RANK[a] <= _RANK[b] else b


def independent_domains(docs: list[ScoredDocument]) -> int:
    return len({d.domain for d in docs if not d.thin})


def code_level(
    *,
    independent_source_count: int,
    subquestions_with_evidence: int,
    subquestions_total: int,
    tool_failures: int,
    replan_used: bool,
    injection_flagged: bool = False,
) -> ConfidenceLevel:
    total = max(1, subquestions_total)
    coverage = subquestions_with_evidence / total
    # Floor conditions: missing independent corroboration, failed tools, or injection.
    if independent_source_count < 2 or coverage < 0.5 or tool_failures >= 2 or injection_flagged:
        return ConfidenceLevel.LOW
    if replan_used or coverage < 1.0 or tool_failures > 0 or independent_source_count < 3:
        return ConfidenceLevel.MEDIUM
    return ConfidenceLevel.HIGH


def assess_confidence(
    *,
    docs: list[ScoredDocument],
    subquestions_with_evidence: int,
    subquestions_total: int,
    tool_failures: int,
    replan_used: bool,
    llm_level: ConfidenceLevel,
    injection_flagged: bool = False,
) -> ConfidenceSignals:
    n_indep = independent_domains(docs)
    flagged = injection_flagged or any(d.injection_flagged for d in docs)
    code = code_level(
        independent_source_count=n_indep,
        subquestions_with_evidence=subquestions_with_evidence,
        subquestions_total=subquestions_total,
        tool_failures=tool_failures,
        replan_used=replan_used,
        injection_flagged=flagged,
    )
    return ConfidenceSignals(
        independent_source_count=n_indep,
        subquestions_with_evidence=subquestions_with_evidence,
        subquestions_total=subquestions_total,
        tool_failures=tool_failures,
        replan_used=replan_used,
        injection_flagged=flagged,
        code_level=code,
        llm_level=llm_level,
        final_level=min_level(code, llm_level),
    )


def rationale(signals: ConfidenceSignals) -> str:
    bits = [
        f"{signals.independent_source_count} independent domains",
        f"covered {signals.subquestions_with_evidence}/{signals.subquestions_total} sub-questions",
        f"{signals.tool_failures} tool failures",
    ]
    if signals.replan_used:
        bits.append("one replan used")
    if signals.injection_flagged:
        bits.append("injection patterns flagged in fetched pages")
    bits.append(f"code={signals.code_level} llm={signals.llm_level} final={signals.final_level}")
    return "; ".join(bits)
