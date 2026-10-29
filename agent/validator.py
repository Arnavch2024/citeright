"""VALIDATE: schema already passed if we have a ResearchAnswer; this is the citation/code pass."""

from __future__ import annotations

from agent.citations import citation_errors
from agent.llm import LLMClient
from agent.schemas import Finding, ResearchAnswer, SynthesisDraft

REPAIR_SYSTEM = """Fix the structured answer. You may only cite source_ids from the allowed list.
Do not invent URLs or ids. Prefer a 'not found' finding over a guess.
Strip any claim you cannot ground in allowed ids.
"""


def validate_answer(
    answer: ResearchAnswer,
    retrieved_ids: set[str],
    retrieved_urls: set[str],
) -> list[str]:
    errors = citation_errors(answer, retrieved_ids, retrieved_urls)
    if not answer.key_findings:
        errors.append("key_findings is empty")
    if not answer.short_answer.strip():
        errors.append("short_answer is empty")
    return errors


def sanitize_answer(
    answer: ResearchAnswer,
    retrieved_ids: set[str],
    retrieved_urls: set[str],
) -> ResearchAnswer:
    """Last resort: drop invented citations. Never add new sources."""
    sources = [s for s in answer.sources if s.id in retrieved_ids and s.url in retrieved_urls]
    allowed = {s.id for s in sources}
    findings: list[Finding] = []
    for finding in answer.key_findings:
        ids = [i for i in finding.source_ids if i in allowed]
        if ids or not finding.source_ids:
            findings.append(Finding(claim=finding.claim, source_ids=ids))
    if not findings:
        findings = [Finding(claim="Not found in retrieved sources.", source_ids=[])]
    return answer.model_copy(update={"sources": sources, "key_findings": findings})


async def repair_draft(
    llm: LLMClient,
    draft: SynthesisDraft,
    errors: list[str],
    allowed_ids: list[str],
) -> SynthesisDraft:
    user = (
        f"Allowed source_ids: {', '.join(allowed_ids) or '(none)'}\n"
        f"Validation errors:\n- "
        + "\n- ".join(errors)
        + f"\n\nDraft JSON:\n{draft.model_dump_json()}"
    )
    return await llm.generate_structured(REPAIR_SYSTEM, user, SynthesisDraft, "strong")
