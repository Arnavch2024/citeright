from agent.check import independent_domain_count, insufficient_subquestions
from agent.schemas import DomainType, ScoredDocument, utcnow


def _doc(sid: str, host: str, thin: bool = False) -> ScoredDocument:
    return ScoredDocument(
        source_id=sid,
        url=f"https://{host}/p",
        title=sid,
        domain=host,
        domain_type=DomainType.OTHER,
        score=0.0 if thin else 0.5,
        retrieved_at=utcnow(),
        text="x",
        thin=thin,
    )


def test_insufficient_when_one_domain() -> None:
    by = {"q1": [_doc("s1", "a.com"), _doc("s2", "a.com")]}
    assert insufficient_subquestions(by) == ["q1"]


def test_ok_with_two_domains() -> None:
    by = {"q1": [_doc("s1", "a.com"), _doc("s2", "b.com")]}
    assert insufficient_subquestions(by) == []


def test_thin_does_not_count() -> None:
    docs = [_doc("s1", "a.com"), _doc("s2", "b.com", thin=True)]
    assert independent_domain_count(docs) == 1
