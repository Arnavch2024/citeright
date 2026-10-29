"""CHECK is code, not an LLM. Independent-domain corroboration per sub-question."""

from __future__ import annotations

from agent.schemas import ScoredDocument


def independent_domain_count(docs: list[ScoredDocument]) -> int:
    return len({d.domain for d in docs if not d.thin and d.score > 0})


def insufficient_subquestions(
    by_sub: dict[str, list[ScoredDocument]],
    *,
    min_domains: int = 2,
) -> list[str]:
    """Return sub-question ids that do not yet have enough independent domains."""
    weak: list[str] = []
    for sub_id, docs in by_sub.items():
        if independent_domain_count(docs) < min_domains:
            weak.append(sub_id)
    return weak
