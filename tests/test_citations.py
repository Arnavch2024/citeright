from datetime import datetime, timezone

from agent.citations import citation_errors
from agent.schemas import ConfidenceLevel, Finding, ResearchAnswer, RunMetadata, Source


def _answer(**kwargs) -> ResearchAnswer:
    defaults = dict(
        question="q",
        short_answer="a",
        key_findings=[Finding(claim="c", source_ids=["s1"])],
        sources=[
            Source(
                id="s1",
                url="https://docs.example.com/x",
                title="Docs",
                retrieved_at=datetime.now(timezone.utc),
            )
        ],
        confidence=ConfidenceLevel.MEDIUM,
        confidence_rationale="ok",
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
    defaults.update(kwargs)
    return ResearchAnswer(**defaults)


def test_citation_ok_when_retrieved() -> None:
    answer = _answer()
    assert citation_errors(answer, {"s1"}, {"https://docs.example.com/x"}) == []


def test_citation_rejects_invented_id() -> None:
    answer = _answer(key_findings=[Finding(claim="c", source_ids=["s9"])])
    errs = citation_errors(answer, {"s1"}, {"https://docs.example.com/x"})
    assert any("s9" in e for e in errs)


def test_citation_rejects_unretrieved_url() -> None:
    errs = citation_errors(_answer(), {"s1"}, {"https://other.example/"})
    assert any("url" in e for e in errs)
