from agent.confidence import assess_confidence, code_level, min_level
from agent.schemas import ConfidenceLevel, DomainType, ScoredDocument, utcnow


def _doc(sid: str, host: str, *, inject: bool = False) -> ScoredDocument:
    return ScoredDocument(
        source_id=sid,
        url=f"https://{host}/p",
        title=sid,
        domain=host,
        domain_type=DomainType.OFFICIAL_DOCS,
        score=1.0,
        retrieved_at=utcnow(),
        text="x" * 400,
        injection_flagged=inject,
    )


def test_code_floor_two_domains() -> None:
    assert (
        code_level(
            independent_source_count=1,
            subquestions_with_evidence=1,
            subquestions_total=1,
            tool_failures=0,
            replan_used=False,
        )
        is ConfidenceLevel.LOW
    )


def test_llm_cannot_raise_code_floor() -> None:
    signals = assess_confidence(
        docs=[_doc("a", "docs.qdrant.tech")],
        subquestions_with_evidence=1,
        subquestions_total=1,
        tool_failures=0,
        replan_used=False,
        llm_level=ConfidenceLevel.HIGH,
    )
    assert signals.code_level is ConfidenceLevel.LOW
    assert signals.final_level is ConfidenceLevel.LOW


def test_high_requires_coverage_and_three_domains() -> None:
    docs = [_doc("a", "docs.qdrant.tech"), _doc("b", "milvus.io"), _doc("c", "weaviate.io")]
    signals = assess_confidence(
        docs=docs,
        subquestions_with_evidence=3,
        subquestions_total=3,
        tool_failures=0,
        replan_used=False,
        llm_level=ConfidenceLevel.HIGH,
    )
    assert signals.final_level is ConfidenceLevel.HIGH


def test_replan_caps_at_medium() -> None:
    docs = [_doc("a", "docs.qdrant.tech"), _doc("b", "milvus.io"), _doc("c", "weaviate.io")]
    signals = assess_confidence(
        docs=docs,
        subquestions_with_evidence=3,
        subquestions_total=3,
        tool_failures=0,
        replan_used=True,
        llm_level=ConfidenceLevel.HIGH,
    )
    assert signals.code_level is ConfidenceLevel.MEDIUM
    assert signals.final_level is ConfidenceLevel.MEDIUM


def test_min_level() -> None:
    assert min_level(ConfidenceLevel.HIGH, ConfidenceLevel.LOW) is ConfidenceLevel.LOW
