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


def step_down(level: ConfidenceLevel) -> ConfidenceLevel:
    """Lower a confidence level by one step. HIGH->MEDIUM->LOW->LOW."""
    if level is ConfidenceLevel.HIGH:
        return ConfidenceLevel.MEDIUM
    return ConfidenceLevel.LOW


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
    verified_ratio: float = 1.0,
    verifier_ran: bool = False,
    min_verified_ratio: float = 0.7,
    min_coverage: float = 0.5,
    conflicts_found: int = 0,
) -> ConfidenceLevel:
    total = max(1, subquestions_total)
    coverage = subquestions_with_evidence / total
    # Floor conditions: missing independent corroboration, thin coverage, failed
    # tools, injection, or a verified ratio below the abstention threshold.
    if (
        independent_source_count < 2
        or coverage < min_coverage
        or tool_failures >= 2
        or injection_flagged
    ):
        return ConfidenceLevel.LOW
    if verifier_ran and verified_ratio < min_verified_ratio:
        return ConfidenceLevel.LOW
    if replan_used or coverage < 1.0 or tool_failures > 0 or independent_source_count < 3:
        base = ConfidenceLevel.MEDIUM
    else:
        base = ConfidenceLevel.HIGH
    # Conflicts lower the base by one step; do not hide them.
    if conflicts_found > 0:
        base = step_down(base)
    return base


def assess_confidence(
    *,
    docs: list[ScoredDocument],
    subquestions_with_evidence: int,
    subquestions_total: int,
    tool_failures: int,
    replan_used: bool,
    llm_level: ConfidenceLevel,
    injection_flagged: bool = False,
    verified_ratio: float = 1.0,
    claims_dropped: int = 0,
    conflicts_found: int = 0,
    verifier_ran: bool = False,
    min_verified_ratio: float = 0.7,
    min_coverage: float = 0.5,
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
        verified_ratio=verified_ratio,
        verifier_ran=verifier_ran,
        min_verified_ratio=min_verified_ratio,
        min_coverage=min_coverage,
        conflicts_found=conflicts_found,
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
        verified_ratio=verified_ratio,
        claims_dropped=claims_dropped,
        conflicts_found=conflicts_found,
        verifier_ran=verifier_ran,
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
    if signals.verifier_ran:
        bits.append(f"verified_ratio={signals.verified_ratio:.2f} ({signals.claims_dropped} dropped)")
    if signals.conflicts_found:
        bits.append(f"{signals.conflicts_found} conflicts")
    bits.append(f"code={signals.code_level} llm={signals.llm_level} final={signals.final_level}")
    return "; ".join(bits)
