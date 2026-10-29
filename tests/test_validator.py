from datetime import datetime, timezone

from agent.schemas import ConfidenceLevel, Finding, ResearchAnswer, RunMetadata, Source
from agent.validator import sanitize_answer, validate_answer


def _answer(source_ids: list[str]) -> ResearchAnswer:
    return ResearchAnswer(
        question="q",
        short_answer="a",
        key_findings=[Finding(claim="claim", source_ids=source_ids)],
        sources=[
            Source(
                id="s1",
                url="https://docs.example.com/x",
                title="d",
                retrieved_at=datetime.now(timezone.utc),
            )
        ],
        confidence=ConfidenceLevel.HIGH,
        confidence_rationale="x",
        assumptions_and_limitations=[],
        next_steps=[],
        metadata=RunMetadata(
            run_id="r",
            latency_ms=1,
            provider="fake",
            cheap_model="c",
            strong_model="s",
        ),
    )


def test_validate_flags_invented_id() -> None:
    errs = validate_answer(_answer(["s9"]), {"s1"}, {"https://docs.example.com/x"})
    assert any("s9" in e for e in errs)


def test_sanitize_drops_invented_and_keeps_partial() -> None:
    cleaned = sanitize_answer(_answer(["s9"]), {"s1"}, {"https://docs.example.com/x"})
    assert cleaned.key_findings[0].source_ids == [] or "Not found" in cleaned.key_findings[0].claim
    assert all(s.id == "s1" for s in cleaned.sources)
