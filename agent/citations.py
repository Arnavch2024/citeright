"""Code-only citation checks. Used by VALIDATE in phase 3; tested here so the contract is frozen."""

from __future__ import annotations

from agent.schemas import ResearchAnswer


def citation_errors(
    answer: ResearchAnswer,
    retrieved_ids: set[str],
    retrieved_urls: set[str],
) -> list[str]:
    """Every cited id and listed URL must have been retrieved. Never invent a source."""
    errors: list[str] = []
    listed = {s.id: s for s in answer.sources}
    for finding in answer.key_findings:
        for sid in finding.source_ids:
            if sid not in retrieved_ids:
                errors.append(f"cited unknown source_id {sid!r}")
            elif sid not in listed:
                errors.append(f"cited {sid!r} missing from sources[]")
    for src in answer.sources:
        if src.id not in retrieved_ids:
            errors.append(f"sources[] id {src.id!r} was not retrieved")
        if src.url not in retrieved_urls:
            errors.append(f"sources[] url {src.url!r} was not retrieved")
    return errors
